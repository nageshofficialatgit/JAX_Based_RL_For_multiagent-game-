import pandas as pd
import numpy as np
import os
import math
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from training.fast_sim import generate_comet_paths, COMET_SPAWN_STEPS

print("Loading DataFrames...")
db_path = "data/parquet_db_real"
df_planet_state = pd.read_parquet(os.path.join(db_path, "planet_state.parquet"))
df_episode_planets = pd.read_parquet(os.path.join(db_path, "episode_planets.parquet"))
df_episodes = pd.read_parquet(os.path.join(db_path, "episodes.parquet"))

print("Merging initial coordinates...")
# Merge initial_x, initial_y, is_static, is_comet
df_merged = df_planet_state.merge(
    df_episode_planets[['episode_id', 'planet_id', 'initial_x', 'initial_y', 'is_static', 'is_comet']],
    on=['episode_id', 'planet_id'],
    how='left'
)

# Merge angular_velocity and seed
df_merged = df_merged.merge(
    df_episodes[['episode_id', 'angular_velocity', 'seed']],
    on='episode_id',
    how='left'
)

print("Calculating Orbiting Planets Coordinates...")
orbit_mask = (df_merged['is_static'] == 0) & (df_merged['is_comet'] == 0)
orbit_df = df_merged[orbit_mask]

dx = orbit_df['initial_x'] - 50.0
dy = orbit_df['initial_y'] - 50.0
orb_r = np.sqrt(dx**2 + dy**2)
init_angle = np.arctan2(dy, dx)
cur_angle = init_angle + orbit_df['angular_velocity'] * orbit_df['tick']

df_merged.loc[orbit_mask, 'current_x'] = 50.0 + orb_r * np.cos(cur_angle)
df_merged.loc[orbit_mask, 'current_y'] = 50.0 + orb_r * np.sin(cur_angle)

print("Setting Static Planets Coordinates...")
static_mask = (df_merged['is_static'] == 1) & (df_merged['is_comet'] == 0)
df_merged.loc[static_mask, 'current_x'] = df_merged.loc[static_mask, 'initial_x']
df_merged.loc[static_mask, 'current_y'] = df_merged.loc[static_mask, 'initial_y']

print("Calculating Comet Paths... (This might take a few minutes)")
import random
comet_mask = df_merged['is_comet'] == 1
if np.any(comet_mask):
    # Group by episode to avoid regenerating the whole map per row
    comet_df = df_merged[comet_mask]
    
    # We need the full initial_planets list per episode for generate_comet_paths
    # Let's group df_episode_planets
    grouped_planets = df_episode_planets.groupby('episode_id')
    
    # Dictionary to cache generated paths per episode
    episode_comet_cache = {}
    
    # We need to iterate over the comet rows and fill the current_x and current_y
    x_vals = np.zeros(len(comet_df), dtype=np.float32)
    y_vals = np.zeros(len(comet_df), dtype=np.float32)
    
    for i, (idx, row) in tqdm(enumerate(comet_df.iterrows()), total=len(comet_df)):
        ep_id = row['episode_id']
        tick = row['tick']
        pid = row['planet_id']
        
        if ep_id not in episode_comet_cache:
            # Generate all comets for this episode
            seed = row['seed']
            angular_vel = row['angular_velocity']
            
            # Format initial_planets
            ep_planets = grouped_planets.get_group(ep_id)
            initial_planets = []
            comet_pids = []
            for _, p_row in ep_planets.iterrows():
                # [id, owner, x, y, radius, ships, production]
                planet_tuple = [
                    p_row['planet_id'], p_row['initial_owner'], p_row['initial_x'], p_row['initial_y'],
                    p_row['radius'], p_row['initial_ships'], p_row['production']
                ]
                initial_planets.append(planet_tuple)
                if p_row['is_comet'] == 1:
                    comet_pids.append(p_row['planet_id'])
            
            # Now simulate comet generation for all spawn steps
            paths_dict = {} # pid -> path list
            for spawn_step in COMET_SPAWN_STEPS:
                rng = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
                paths = generate_comet_paths(initial_planets, angular_vel, spawn_step, comet_pids, rng)
                if paths:
                    # Which pids are these? They are sequential starting from next_id
                    # We can map them by assuming the kaggle environment gave them sequential IDs
                    # But wait, in dataset they already have IDs. We can just match them by spawn_step
                    # Actually, the kaggle environment assigns them IDs strictly sequentially.
                    pass
            episode_comet_cache[ep_id] = paths_dict
        
        # This is too complex to reconstruct perfectly because the pid assignment is tricky.
        # Instead, we will do a simpler hack: Comets don't move in the transformer because they are deleted when they hit planets.
        # Wait, they do move!

# Just save it for now to see if it works
out_path = os.path.join(db_path, "planet_state_coords.parquet")
print(f"Saving to {out_path}...")
df_merged.to_parquet(out_path)
print("Done!")
