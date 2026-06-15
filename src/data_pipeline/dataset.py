import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class OrbitWarsDataset(Dataset):
    """
    High-throughput PyTorch Dataset for Orbit Wars Parquet Database.
    
    Modes:
    - grandmaster_only: Filters the 1-Billion state dataset down to only the episodes won by the Top 1% of players.
    - subsample_ticks: Temporal subsampling. If 5, only loads every 5th tick to save compute.
    - is_iql_mode: If True, aggregates actions from the next `subsample_ticks` and returns them as ground-truth targets.
    """
    def __init__(self, db_path="parquet_db_real", max_action_history=20, grandmaster_only=True, subsample_ticks=5, is_iql_mode=True):
        super().__init__()
        self.db_path = db_path
        self.max_action_history = max_action_history
        self.subsample_ticks = subsample_ticks
        self.is_iql_mode = is_iql_mode
        
        print("Loading Parquet Database into RAM...")
        self.df_episodes = pd.read_parquet(os.path.join(db_path, "episodes.parquet"))
        self.df_player_eps = pd.read_parquet(os.path.join(db_path, "player_episodes.parquet"))
        self.df_episode_planets = pd.read_parquet(os.path.join(db_path, "episode_planets.parquet"))
        self.df_planet_state = pd.read_parquet(os.path.join(db_path, "planet_state.parquet"))
        self.df_actions = pd.read_parquet(os.path.join(db_path, "actions.parquet"))
        
        winners = self.df_player_eps[self.df_player_eps['is_winner'] == 1]
        self.episode_to_winner = dict(zip(winners['episode_id'], winners['slot']))
        
        # Filter by Grandmaster Episodes if enabled
        if grandmaster_only:
            gm_file = os.path.join(db_path, "grandmaster_episodes.csv")
            if os.path.exists(gm_file):
                print("Filtering dataset to Grandmaster episodes only...")
                gm_df = pd.read_csv(gm_file)
                gm_ids = set(gm_df['episode_id'].values)
                mask = self.df_planet_state['episode_id'].isin(gm_ids)
                self.df_planet_state = self.df_planet_state[mask]
                print(f"Grandmaster Filter reduced planet_state from millions to {len(self.df_planet_state)} rows.")
            else:
                print("WARNING: grandmaster_episodes.csv not found. Run filter_grandmasters.py first!")
        
        print("Building index with Temporal Sub-sampling...")
        all_states = self.df_planet_state[['episode_id', 'tick']].drop_duplicates()
        
        # Temporal Sub-sampling (only keep every Nth tick)
        if self.subsample_ticks > 1:
            all_states = all_states[all_states['tick'] % self.subsample_ticks == 0]
            
        self.state_index = all_states.values
        
        print("Indexing DataFrames...")
        self.df_planet_state.set_index(['episode_id', 'tick', 'planet_id'], inplace=True)
        self.df_episode_planets.set_index(['episode_id', 'planet_id'], inplace=True)
        self.df_actions.set_index(['episode_id', 'tick'], inplace=True)
        
        print(f"Dataset ready. Total states to train on: {len(self.state_index)}")

    def __len__(self):
        return len(self.state_index)

    def __getitem__(self, idx):
        episode_id, tick = self.state_index[idx]
        winner_slot = self.episode_to_winner.get(episode_id, -1)
        
        try:
            static_planets = self.df_episode_planets.loc[episode_id]
            dynamic_planets = self.df_planet_state.loc[(episode_id, tick)]
        except KeyError:
            return self.__getitem__((idx + 1) % len(self))
            
        planets = static_planets.join(dynamic_planets, how='left').sort_index()
        n_planets = len(planets)
        
        # 1. State Features
        planet_tensors = np.zeros((50, 10), dtype=np.float32)
        planet_tensors[:, 0] = -1.0
        planet_tensors[:n_planets, 0] = 0.0

        planet_x = planets['initial_x'].values.copy()
        planet_y = planets['initial_y'].values.copy()
        if n_planets > 0:
            center_x = np.mean(planet_x[:n_planets])
            center_y = np.mean(planet_y[:n_planets])
            planet_x[:n_planets] = (planet_x[:n_planets] - center_x) / 50.0
            planet_y[:n_planets] = (planet_y[:n_planets] - center_y) / 50.0

        planet_tensors[:n_planets, 1] = planet_x
        planet_tensors[:n_planets, 2] = planet_y
        planet_tensors[:n_planets, 3] = planets['radius'].values / 10.0
        planet_tensors[:n_planets, 4] = planets['production'].values / 5.0
        planet_tensors[:n_planets, 5] = planets['owner'].values
        planet_tensors[:n_planets, 6] = np.log1p(planets['ships'].values) / 7.0
        planet_tensors[:n_planets, 7] = planets['is_static'].astype(np.float32).values
        planet_tensors[:n_planets, 8] = planets['is_comet'].astype(np.float32).values
        planet_tensors[:n_planets, 9] = tick / 1000.0
        
        # 2. Action History Tokens (Fleets in Flight)
        action_tensors = np.zeros((self.max_action_history, 10), dtype=np.float32)
        action_tensors[:, 0] = 1.0
        
        if episode_id in self.df_actions.index.levels[0]:
            try:
                ep_actions = self.df_actions.loc[episode_id]
                mask = (ep_actions.index >= tick - self.max_action_history) & (ep_actions.index < tick)
                recent_actions = ep_actions[mask]
                
                if len(recent_actions) > 0:
                    n_act = min(len(recent_actions), self.max_action_history)
                    recent_actions = recent_actions.iloc[-n_act:]
                    
                    ticks_since_launch = tick - recent_actions.index.values
                    action_tensors[:n_act, 0] = 1.0
                    action_tensors[:n_act, 1] = recent_actions['src_planet_id'].values
                    action_tensors[:n_act, 2] = recent_actions['angle'].values
                    action_tensors[:n_act, 3] = recent_actions['n_ships'].values
                    action_tensors[:n_act, 4] = recent_actions['slot'].values
                    action_tensors[:n_act, 5] = ticks_since_launch
                    action_tensors[:n_act, 9] = tick / 1000.0
            except KeyError:
                pass
                
        tokens = np.vstack([planet_tensors, action_tensors])
        
        output = {
            "state_tokens": torch.FloatTensor(tokens),
            "winner": torch.LongTensor([winner_slot]).squeeze(),
            "tick": torch.LongTensor([tick]).squeeze()
        }
        
        # 3. IQL Targets (What did the Grandmaster do next?)
        if self.is_iql_mode:
            target_launch = np.zeros(50, dtype=np.float32)
            target_angle = np.full(50, -1, dtype=np.int64) # -1 is ignore_index for CrossEntropy
            target_ships = np.full(50, -1, dtype=np.int64)
            is_owned_by_winner = (planet_tensors[:50, 5] == winner_slot).astype(np.float32)
            
            if episode_id in self.df_actions.index.levels[0]:
                try:
                    # Look ahead `subsample_ticks` to find actions taken
                    ep_actions = self.df_actions.loc[episode_id]
                    future_mask = (ep_actions.index >= tick) & (ep_actions.index < tick + self.subsample_ticks)
                    future_actions = ep_actions[future_mask]
                    
                    # Only clone the WINNER's actions!
                    winner_actions = future_actions[future_actions['slot'] == winner_slot]
                    
                    for _, row in winner_actions.iterrows():
                        pid = int(row['src_planet_id'])
                        if pid < 50:
                            # 1. Did they launch?
                            target_launch[pid] = 1.0
                            
                            # 2. Angle (Discretized into 72 bins)
                            # angle is -pi to pi. (angle + pi) / (2pi) * 72 -> [0, 71]
                            angle_bin = int((row['angle'] + np.pi) / (2 * np.pi) * 72)
                            target_angle[pid] = np.clip(angle_bin, 0, 71)
                            
                            # 3. Ships fraction (Discretized into 10 bins)
                            # We don't have the exact garrison at the exact tick they launched,
                            # but we can use the garrison at 'tick' as an estimate.
                            garrison = planets.loc[pid, 'ships'] if pid in planets.index else row['n_ships']
                            fraction = row['n_ships'] / max(1, garrison)
                            ships_bin = int(fraction * 10)
                            target_ships[pid] = np.clip(ships_bin, 0, 9)
                except KeyError:
                    pass
            
            output["target_launch"] = torch.FloatTensor(target_launch)
            output["target_angle"] = torch.LongTensor(target_angle)
            output["target_ships"] = torch.LongTensor(target_ships)
            output["is_owned_by_winner"] = torch.FloatTensor(is_owned_by_winner)
            
        return output

if __name__ == "__main__":
    ds = OrbitWarsDataset(max_action_history=20, grandmaster_only=True, subsample_ticks=5, is_iql_mode=True)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4)
    for batch in dl:
        print("State:", batch["state_tokens"].shape)
        print("Target Launch:", batch["target_launch"].shape)
        print("Target Angle:", batch["target_angle"].shape)
        print("Target Ships:", batch["target_ships"].shape)
        break
