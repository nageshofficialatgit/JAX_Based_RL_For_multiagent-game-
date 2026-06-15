import os
import pandas as pd
import numpy as np

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    db_path = os.path.join(base_dir, "parquet_db_real")
    
    print("Loading Leaderboard and Player Mappings...")
    df_leaderboard = pd.read_csv(os.path.join(db_path, "leaderboard.csv"))
    df_players = pd.read_parquet(os.path.join(db_path, "player_episodes.parquet"), engine='pyarrow')
    
    # 1. Process Actions
    print("Loading Actions...")
    df_act = pd.read_parquet(os.path.join(db_path, "actions.parquet"), engine='pyarrow')
    df_act = df_act.merge(df_players[['episode_id', 'slot', 'name']], on=['episode_id', 'slot'], how='inner')
    
    print("Calculating Action Metrics...")
    act_metrics = df_act.groupby('name').agg(
        total_attacks=('n_ships', 'count'),
        avg_payload_vol=('n_ships', 'mean')
    ).reset_index()
    
    ticks_grouped = df_act.groupby(['name', 'episode_id', 'tick']).size().reset_index(name='launches_this_tick')
    tick_metrics = ticks_grouped.groupby('name').agg(
        total_launch_ticks=('tick', 'count'),
        coordinated_strike_ticks=('launches_this_tick', lambda x: (x > 1).sum())
    ).reset_index()
    
    del df_act
    import gc
    gc.collect()
    
    # 2. Process Planet States
    print("Loading Planet States (Memory Intensive)...")
    df_planets = pd.read_parquet(os.path.join(db_path, "planet_state.parquet"), engine='pyarrow', columns=['episode_id', 'tick', 'planet_id', 'owner', 'ships'])
    df_planets_renamed = df_planets.rename(columns={'owner': 'slot'})
    df_owned = df_planets_renamed.merge(df_players[['episode_id', 'slot', 'name']], on=['episode_id', 'slot'], how='inner')
    
    print("Calculating Territory Metrics...")
    planet_counts = df_owned.groupby(['name', 'episode_id', 'tick']).agg(
        planets_held=('slot', 'count'),
        max_doomstack=('ships', 'max')
    ).reset_index()
    
    territory_metrics = planet_counts.groupby('name').agg(
        peak_planets_held=('planets_held', 'max'),
        max_doomstack=('max_doomstack', 'max')
    ).reset_index()
    
    # Tactical Transitions
    print("Calculating Tactical Ownership Transitions (Snipes & Target Preferences)...")
    df_planets_sorted = df_planets_renamed.sort_values(by=['episode_id', 'planet_id', 'tick'])
    df_planets_sorted['prev_slot'] = df_planets_sorted.groupby(['episode_id', 'planet_id'])['slot'].shift(1)
    
    transitions = df_planets_sorted[(df_planets_sorted['slot'] != df_planets_sorted['prev_slot']) & (df_planets_sorted['prev_slot'].notna())].copy()
    
    transitions['next_slot'] = transitions.groupby(['episode_id', 'planet_id'])['slot'].shift(-1)
    transitions['next_tick'] = transitions.groupby(['episode_id', 'planet_id'])['tick'].shift(-1)
    transitions['ticks_held'] = transitions['next_tick'] - transitions['tick']
    
    neutral_captures = transitions[transitions['prev_slot'] == -1].copy()
    neutral_captures['is_sniped'] = neutral_captures['ticks_held'] <= 25
    
    neutral_captures = neutral_captures.merge(df_players[['episode_id', 'slot', 'name']], on=['episode_id', 'slot'], how='inner')
    
    snipe_metrics = neutral_captures.groupby('name').agg(
        total_neutral_captures=('planet_id', 'count'),
        total_snipes_suffered=('is_sniped', 'sum')
    ).reset_index()
    
    # Load Episode Planets for ROI and Static
    df_ep_planets = pd.read_parquet(os.path.join(db_path, "episode_planets.parquet"), engine='pyarrow')
    df_ep_planets['roi'] = df_ep_planets['production'] / df_ep_planets['initial_ships'].clip(lower=1)
    
    neutral_captures = neutral_captures.merge(df_ep_planets[['episode_id', 'planet_id', 'is_static', 'roi']], on=['episode_id', 'planet_id'], how='inner')
    
    tactical_metrics = neutral_captures.groupby('name').agg(
        static_captures=('is_static', 'sum'),
        avg_capture_roi=('roi', 'mean')
    ).reset_index()
    
    # Merge Tactical
    tactical_df = snipe_metrics.merge(tactical_metrics, on='name', how='left')
    tactical_df['snipe_vulnerability_rate'] = tactical_df['total_snipes_suffered'] / tactical_df['total_neutral_captures'].clip(lower=1)
    tactical_df['static_target_rate'] = tactical_df['static_captures'] / tactical_df['total_neutral_captures'].clip(lower=1)
    
    # Total Ticks Played
    ep_lengths = df_planets.groupby('episode_id')['tick'].max().reset_index(name='ep_length')
    player_eps = df_players[['name', 'episode_id']].merge(ep_lengths, on='episode_id')
    total_ticks_metrics = player_eps.groupby('name')['ep_length'].sum().reset_index(name='total_ticks_played')
    
    del df_planets
    del df_owned
    del df_planets_sorted
    gc.collect()
    
    # 3. Aggregate Final Report
    print("Aggregating Final Profile Dataset...")
    final_df = df_leaderboard.copy()
    final_df = final_df.merge(act_metrics, on='name', how='left').fillna({'total_attacks': 0, 'avg_payload_vol': 0})
    final_df = final_df.merge(tick_metrics, on='name', how='left').fillna({'total_launch_ticks': 0, 'coordinated_strike_ticks': 0})
    final_df = final_df.merge(territory_metrics, on='name', how='left').fillna({'peak_planets_held': 0, 'max_doomstack': 0})
    final_df = final_df.merge(tactical_df, on='name', how='left').fillna({
        'snipe_vulnerability_rate': 0, 'static_target_rate': 0, 'avg_capture_roi': 0
    })
    final_df = final_df.merge(total_ticks_metrics, on='name', how='left').fillna({'total_ticks_played': 1})
    
    # Derived Rates
    final_df['fleet_launch_rate'] = final_df['total_launch_ticks'] / final_df['total_ticks_played']
    final_df['attacks_per_turn'] = final_df['total_attacks'] / final_df['total_ticks_played']
    final_df['coordinated_strike_rate'] = final_df['coordinated_strike_ticks'] / final_df['total_launch_ticks'].clip(lower=1)
    
    # Sort
    final_df = final_df.sort_values(by=['win_rate', 'total_games'], ascending=[False, False])
    
    # Order columns nicely
    cols = [
        'name', 'win_rate', 'total_games', 'fleet_launch_rate', 'attacks_per_turn',
        'coordinated_strike_rate', 'max_doomstack', 'snipe_vulnerability_rate', 
        'static_target_rate', 'avg_capture_roi', 'avg_payload_vol', 'peak_planets_held'
    ]
    final_df = final_df[cols]
    
    out_path = os.path.join(db_path, "agent_behavior_profiles.csv")
    final_df.to_csv(out_path, index=False)
    
    print(f"\nSUCCESS: Advanced Tactical Profiles Exported to {out_path}!")

if __name__ == "__main__":
    main()
