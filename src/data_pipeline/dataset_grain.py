import os
import pandas as pd
import numpy as np
import grain.python as grain

class OrbitWarsDataSource(grain.RandomAccessDataSource):
    """
    High-throughput Google Grain DataSource for Orbit Wars Parquet Database.
    Replaces the PyTorch Dataset for the Pure JAX/Flax Stack.
    """
    def __init__(self, db_path="parquet_db_real", max_action_history=20, grandmaster_only=True, subsample_ticks=5, is_iql_mode=True):
        self.db_path = db_path
        self.max_action_history = max_action_history
        self.subsample_ticks = subsample_ticks
        self.is_iql_mode = is_iql_mode
        
        print("Loading Parquet Database into RAM for Indexing...")
        self.df_episodes = pd.read_parquet(os.path.join(db_path, "episodes.parquet"))
        self.df_player_eps = pd.read_parquet(os.path.join(db_path, "player_episodes.parquet"))
        
        # We only need the index from planet_state
        planet_state_path = os.path.join(db_path, "planet_state_coords.parquet")
        if not os.path.exists(planet_state_path):
            planet_state_path = os.path.join(db_path, "planet_state.parquet")
        df_planet_state = pd.read_parquet(planet_state_path)
        
        winners = self.df_player_eps[self.df_player_eps['is_winner'] == 1]
        self.episode_to_winner = dict(zip(winners['episode_id'], winners['slot']))
        
        # Load leaderboard to map winner -> win_rate (0.0 to 1.0 continuous gradient)
        lb_path = os.path.join(db_path, "leaderboard.csv")
        player_to_win_rate = {}
        if os.path.exists(lb_path):
            lb = pd.read_csv(lb_path)
            player_to_win_rate = dict(zip(lb['name'], lb['win_rate']))
        else:
            print("WARNING: leaderboard.csv not found. Defaulting to 0.5 win rate.")
        
        self.episode_to_win_rate = {}
        for _, row in winners.iterrows():
            name = row['name']
            self.episode_to_win_rate[row['episode_id']] = player_to_win_rate.get(name, 0.5)
            
        # Load episodes to get angular_velocity for physics calculation
        eps = pd.read_parquet(os.path.join(db_path, "episodes.parquet"))
        self.episode_to_angular_vel = dict(zip(eps['episode_id'], eps['angular_velocity']))
        
        # Filter by Grandmaster Episodes if enabled
        self.gm_ids = None
        if grandmaster_only:
            gm_file = os.path.join(db_path, "grandmaster_episodes.csv")
            if os.path.exists(gm_file):
                print("Filtering dataset to Grandmaster episodes only...")
                gm_df = pd.read_csv(gm_file)
                self.gm_ids = set(gm_df['episode_id'].values)
                mask = df_planet_state['episode_id'].isin(self.gm_ids)
                df_planet_state = df_planet_state[mask]
                print(f"Grandmaster Filter reduced planet_state from millions to {len(df_planet_state)} rows.")
            else:
                print("WARNING: grandmaster_episodes.csv not found. Did you run filter_grandmasters.py?")
        
        # Temporal Sub-sampling (only keep every Nth tick)
        self.target_episodes = df_planet_state['episode_id'].unique()
        print("Building index with Temporal Sub-sampling...")
        all_states = df_planet_state[['episode_id', 'tick']].drop_duplicates()
        
        # Temporal Sub-sampling (only keep every Nth tick)
        if self.subsample_ticks > 1:
            all_states = all_states[all_states['tick'] % self.subsample_ticks == 0]
            
        # Build an efficient Numpy Array + Dict index for planet_state to eliminate Pandas MultiIndex OOM
        print("Building Memory-Efficient NumPy Index for planet_state...")
        self.planet_state_arr = df_planet_state[['planet_id', 'owner', 'ships']].values.astype(np.float32)
        
        # Build dict mapping (episode_id, tick) -> (start_idx, end_idx)
        ep_ticks = df_planet_state[['episode_id', 'tick']].values
        
        self.state_lookup = {}
        curr_key = None
        start_idx = 0
        for i in range(len(ep_ticks)):
            key = (ep_ticks[i, 0], ep_ticks[i, 1])
            if key != curr_key:
                if curr_key is not None:
                    self.state_lookup[curr_key] = (start_idx, i)
                curr_key = key
                start_idx = i
        if curr_key is not None:
            self.state_lookup[curr_key] = (start_idx, len(ep_ticks))
            
        print("Memory-Efficient NumPy Index built.")
        
        self.state_index = list(self.state_lookup.keys())
        # Subsample again if needed
        if self.subsample_ticks > 1:
            self.state_index = [k for k in self.state_index if k[1] % self.subsample_ticks == 0]
        
        # Crucial Memory Fix: Delete the dataframes from the main process!
        del df_planet_state
        import gc
        gc.collect()
        
        # Load episode_planets into a dict of dataframes to eliminate Pandas MultiIndex overhead
        print("Loading episode_planets into memory-efficient dict...")
        df_ep_planets = pd.read_parquet(os.path.join(self.db_path, "episode_planets.parquet"))
        if hasattr(self, 'target_episodes'):
            df_ep_planets = df_ep_planets[df_ep_planets['episode_id'].isin(self.target_episodes)]
        self.episode_planets_dict_np = {}
        for ep_id, group in df_ep_planets.groupby('episode_id'):
            group_sorted = group.sort_values('planet_id')
            self.episode_planets_dict_np[ep_id] = {
                'initial_x': group_sorted['initial_x'].values.astype(np.float32),
                'initial_y': group_sorted['initial_y'].values.astype(np.float32),
                'radius': group_sorted['radius'].values.astype(np.float32),
                'production': group_sorted['production'].values.astype(np.float32),
                'is_static': group_sorted['is_static'].values.astype(np.float32),
                'is_comet': group_sorted['is_comet'].values.astype(np.float32),
                'planet_ids': group_sorted['planet_id'].values.astype(np.int32)
            }
            
        # Load actions into a dict of dataframes to eliminate Pandas MultiIndex overhead
        print("Loading actions into memory-efficient dict...")
        df_actions = pd.read_parquet(os.path.join(self.db_path, "actions.parquet"))
        if hasattr(self, 'target_episodes'):
            df_actions = df_actions[df_actions['episode_id'].isin(self.target_episodes)]
        self.actions_dict_np = {}
        for ep_id, group in df_actions.groupby('episode_id'):
            group_sorted = group.sort_values('tick')
            
            if 'target_planet_id' in group_sorted.columns:
                target_ids = group_sorted['target_planet_id'].fillna(-1).values.astype(np.int32)
            else:
                target_ids = np.full(len(group_sorted), -1, dtype=np.int32)
                
            self.actions_dict_np[ep_id] = {
                'ticks': group_sorted['tick'].values.astype(np.int32),
                'slot': group_sorted['slot'].values.astype(np.float32),
                'src_planet_id': group_sorted['src_planet_id'].values.astype(np.int32),
                'n_ships': group_sorted['n_ships'].values.astype(np.float32),
                'angle': group_sorted['angle'].values.astype(np.float32),
                'target_planet_id': target_ids
            }
            
        print("All memory-efficient indexes built.")
        
        # Clean up
        del df_ep_planets
        del df_actions
        import gc
        gc.collect()
        
        print(f"Grain DataSource ready. Total states to train on: {len(self.state_index)}")



    def __len__(self):
        return len(self.state_index)

    def __getitem__(self, idx):
        episode_id, tick_val = self.state_index[idx]
        
        # STRICT CASTING: remove any Pandas metadata wrappers and keep shapes stable
        episode_id = int(episode_id)
        tick = int(tick_val)
        winner_slot = int(self.episode_to_winner.get(episode_id, -1))
        win_rate = float(self.episode_to_win_rate.get(episode_id, 0.5))
        angular_vel = float(self.episode_to_angular_vel.get(episode_id, 0.0))
        
        try:
            static_planets = self.episode_planets_dict_np[episode_id]
            if (episode_id, tick) not in self.state_lookup:
                raise KeyError(f"State not found: {(episode_id, tick)}")
                
            start_idx, end_idx = self.state_lookup[(episode_id, tick)]
            dynamic_arr = self.planet_state_arr[start_idx:end_idx]
            
            # PURE NUMPY (O(1) indexing, 100x faster than Pandas join)
            raw_owner = np.zeros(50, dtype=np.float32)
            raw_ships = np.zeros(50, dtype=np.float32)
            p_ids = dynamic_arr[:, 0].astype(int)
            raw_owner[p_ids] = dynamic_arr[:, 1]
            raw_ships[p_ids] = dynamic_arr[:, 2]
            
        except KeyError as e:
            # Grain expects deterministic reads, so we fallback to index 0 safely
            return self.__getitem__(0)
            
        n_planets = len(static_planets['planet_ids'])
        
        # Compute exact orbital physics for current_x and current_y
        init_x = static_planets['initial_x']
        init_y = static_planets['initial_y']
        is_static = static_planets['is_static']
        is_comet = static_planets['is_comet']
        
        current_x = np.copy(init_x)
        current_y = np.copy(init_y)
        
        orbiting_mask = (is_static == 0) & (is_comet == 0)
        if np.any(orbiting_mask):
            dx = init_x[orbiting_mask] - 50.0
            dy = init_y[orbiting_mask] - 50.0
            orb_r = np.sqrt(dx**2 + dy**2)
            init_angle = np.arctan2(dy, dx)
            cur_angle = init_angle + angular_vel * tick
            current_x[orbiting_mask] = 50.0 + orb_r * np.cos(cur_angle)
            current_y[orbiting_mask] = 50.0 + orb_r * np.sin(cur_angle)
        
        # Center coordinates to a local frame for translational invariance
        if n_planets > 0:
            center_x = np.mean(current_x[:n_planets])
            center_y = np.mean(current_y[:n_planets])
            current_x[:n_planets] -= center_x
            current_y[:n_planets] -= center_y
        
        # 1. State Features (Egocentric Owners)
        raw_owner_sliced = raw_owner[:n_planets]
        ego_owner = np.zeros_like(raw_owner_sliced, dtype=np.float32)
        ego_owner[raw_owner_sliced == winner_slot] = 1.0 # "Me"
        ego_owner[(raw_owner_sliced > 0) & (raw_owner_sliced != winner_slot)] = 2.0 # "Enemy"

        planet_tensors = np.zeros((50, 14), dtype=np.float32)
        planet_tensors[:, 0] = -1.0
        planet_tensors[:n_planets, 0] = 0.0
        planet_tensors[:n_planets, 1] = current_x / 50.0
        planet_tensors[:n_planets, 2] = current_y / 50.0
        planet_tensors[:n_planets, 3] = static_planets['radius'] / 10.0
        planet_tensors[:n_planets, 4] = static_planets['production'] / 5.0
        planet_tensors[:n_planets, 5] = ego_owner # Replaced raw owners
        planet_tensors[:n_planets, 6] = np.log1p(raw_ships[:n_planets]) / 7.0
        planet_tensors[:n_planets, 7] = is_static.astype(np.float32)
        planet_tensors[:n_planets, 8] = is_comet.astype(np.float32)
        planet_tensors[:n_planets, 9] = tick / 1000.0
        planet_tensors[:n_planets, 10] = win_rate
        planet_tensors[:n_planets, 11] = angular_vel
        
        # 2. Action History Tokens (Spatial Awareness & Egocentric Fleets)
        action_tensors = np.zeros((self.max_action_history, 14), dtype=np.float32)
        action_tensors[:, 0] = -2.0 
        
        if episode_id in self.actions_dict_np:
            try:
                ep_actions = self.actions_dict_np[episode_id]
                ticks = ep_actions['ticks']
                
                # --- ACTION HISTORY ---
                recent_start = np.searchsorted(ticks, tick - 150, side='left')
                recent_end = np.searchsorted(ticks, tick, side='left')
                
                if recent_end > recent_start:
                    # Slice arrays and take the last max_action_history elements
                    n_act = min(recent_end - recent_start, self.max_action_history)
                    slice_start = recent_end - n_act
                    
                    r_slots = ep_actions['slot'][slice_start:recent_end]
                    r_src = ep_actions['src_planet_id'][slice_start:recent_end]
                    r_ships = ep_actions['n_ships'][slice_start:recent_end]
                    r_angles = ep_actions['angle'][slice_start:recent_end]
                    r_ticks = ticks[slice_start:recent_end]
                    
                    ego_fleet_owner = np.zeros_like(r_slots, dtype=np.float32)
                    ego_fleet_owner[r_slots == winner_slot] = 1.0
                    ego_fleet_owner[(r_slots > 0) & (r_slots != winner_slot)] = 2.0
                    
                    src_x = np.zeros(n_act, dtype=np.float32)
                    src_y = np.zeros(n_act, dtype=np.float32)
                    for i, sid in enumerate(r_src):
                        if sid < len(static_planets['planet_ids']):
                            src_x[i] = current_x[sid] / 50.0
                            src_y[i] = current_y[sid] / 50.0
                        else:
                            src_x[i] = 0.0
                            src_y[i] = 0.0
                    
                    ticks_since_launch = tick - r_ticks
                    
                    action_tensors[:n_act, 0] = 1.0
                    action_tensors[:n_act, 1] = src_x  # Actual X instead of pointer
                    action_tensors[:n_act, 2] = src_y  # Actual Y instead of pointer
                    action_tensors[:n_act, 3] = np.log1p(r_ships) / 7.0
                    action_tensors[:n_act, 4] = ego_fleet_owner
                    action_tensors[:n_act, 5] = ticks_since_launch / 150.0
                    action_tensors[:n_act, 6] = r_angles / np.pi
                    action_tensors[:n_act, 9] = tick / 1000.0
                    action_tensors[:n_act, 10] = win_rate
                    action_tensors[:n_act, 11] = angular_vel
            except Exception:
                pass
                
        tokens = np.vstack([planet_tensors, action_tensors])
        
        output = {
            "state_tokens": tokens.astype(np.float32).reshape(70, 14),
            "winner": np.array(winner_slot, dtype=np.int32),
            "tick": np.array(tick, dtype=np.int32)
        }
        
        # 3. IQL Targets
        if self.is_iql_mode:
            target_launch = np.zeros(50, dtype=np.float32).reshape(50)
            target_angle = np.full(50, -1, dtype=np.int32).reshape(50)
            target_ships = np.full(50, -1, dtype=np.int32).reshape(50)
            
            # THE FIX: Check against 1.0 (the egocentric 'Me' token), NOT winner_slot
            is_owned_by_winner = (planet_tensors[:50, 5] == 1.0).astype(np.float32)
            
            if episode_id in self.actions_dict_np:
                try:
                    ep_actions = self.actions_dict_np[episode_id]
                    ticks = ep_actions['ticks']
                    
                    # --- TARGET LABELS ---
                    future_start = np.searchsorted(ticks, tick, side='left')
                    future_end = np.searchsorted(ticks, tick + self.subsample_ticks, side='left')
                    
                    if future_end > future_start:
                        f_slots = ep_actions['slot'][future_start:future_end]
                        f_src = ep_actions['src_planet_id'][future_start:future_end]
                        f_ships = ep_actions['n_ships'][future_start:future_end]
                        f_angles = ep_actions['angle'][future_start:future_end]
                        f_targets = ep_actions['target_planet_id'][future_start:future_end]
                        
                        my_launches = (f_slots == winner_slot)
                        
                        if np.any(my_launches):
                            my_src = f_src[my_launches]
                            my_ships = f_ships[my_launches]
                            my_angles = f_angles[my_launches]
                            my_targets = f_targets[my_launches]
                            
                            for sid, ships, angle, target in zip(my_src, my_ships, my_angles, my_targets):
                                pid = int(sid)
                                if pid < 50:
                                    target_launch[pid] = 1.0
                                    
                                    best_target = None
                                    if target != -1:
                                        try:
                                            best_target = int(np.where(static_planets['planet_ids'] == target)[0][0])
                                            # CRITICAL FIX: Ignore self-launches which crash the Pointer Network
                                            if best_target == pid:
                                                best_target = None
                                        except IndexError:
                                            pass
                                            
                                    if best_target is None:
                                        # Fast orbit_env style calculation
                                        dx = current_x[:n_planets] - current_x[pid]
                                        dy = current_y[:n_planets] - current_y[pid]
                                        planet_angles = np.arctan2(dy, dx)
                                        diff = np.abs(planet_angles - angle)
                                        diff = np.minimum(diff, 2 * np.pi - diff)
                                        diff[pid] = 1e9 # Mask self
                                        inferred = int(np.argmin(diff))
                                        if diff[inferred] < 0.1: # Strict 5.7 degree tolerance
                                            best_target = inferred
                                        
                                    if best_target is None:
                                        target_angle[pid] = -1
                                    else:
                                        target_angle[pid] = best_target
                                        
                                    garrison = raw_ships[pid] if pid < n_planets else ships
                                    fraction = ships / max(1, garrison)
                                    ships_bin = int(fraction * 10)
                                    target_ships[pid] = np.clip(ships_bin, 0, 9)
                except Exception:
                    pass
            
            output["target_launch"] = target_launch.astype(np.float32)
            output["target_angle"] = target_angle.astype(np.int32)
            output["target_ships"] = target_ships.astype(np.int32)
            output["is_owned_by_winner"] = is_owned_by_winner.astype(np.float32)
            
        return output

def build_grain_dataloader(db_path="parquet_db_real", batch_size=256, worker_count=4, grandmaster_only=True, subsample_ticks=5, is_iql_mode=True, num_epochs=None):
    if not os.path.isabs(db_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates = [
            os.path.join(repo_root, db_path),
            os.path.join(repo_root, "parquet_db_real"),
            os.path.join(repo_root, "data", "parquet_db_real"),
        ]
        resolved = None
        for candidate in candidates:
            if os.path.exists(candidate):
                resolved = candidate
                break
        db_path = resolved if resolved is not None else os.path.abspath(os.path.join(repo_root, db_path))
        print(f"Resolved dataset path to {db_path}")

    source = OrbitWarsDataSource(db_path=db_path, grandmaster_only=grandmaster_only, subsample_ticks=subsample_ticks, is_iql_mode=is_iql_mode)
    
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=num_epochs,
        shard_options=grain.NoSharding(),
        shuffle=True,
        seed=42
    )
    
    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)],
        worker_count=worker_count
    )
    
    return dataloader

if __name__ == "__main__":
    print("Testing Grain DataLoader...")
    dl = build_grain_dataloader(batch_size=4, worker_count=1)
    
    for batch in dl:
        print("Batch received via Grain:")
        print("State shape:", batch["state_tokens"].shape)
        print("Winner shape:", batch["winner"].shape)
        if "target_launch" in batch:
            print("Target Launch shape:", batch["target_launch"].shape)
        break
    print("Grain DataLoader works perfectly!")
