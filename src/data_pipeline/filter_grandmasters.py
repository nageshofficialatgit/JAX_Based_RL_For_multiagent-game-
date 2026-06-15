import os
import glob
import pandas as pd
import numpy as np

def label_grandmasters(
    db_path="/home/medhasree_2121cs05/2201cs50_nagesh/server_deploy/parquet_db_real", 
    min_games=15, 
    top_k=300,
    target_states=10000000, 
    avg_ticks_per_game=200 
):
    print("Scanning dataset for Grandmasters...")
    
    # 1. Load all player_episodes from all available daily shards
    player_files = []
    
    root_file = os.path.join(db_path, "player_episodes.parquet")
    if os.path.exists(root_file):
        player_files.append(root_file)
        
    for day_dir in glob.glob(os.path.join(db_path, "orbit-wars-episodes-*")):
        f = os.path.join(day_dir, "player_episodes.parquet")
        if os.path.exists(f):
            player_files.append(f)
            
    print(f"Found {len(player_files)} days of player data.")
    
    df_list = [pd.read_parquet(f) for f in player_files]
    df_players = pd.concat(df_list, ignore_index=True)
    
    print(f"Loaded {len(df_players)} total player records.")
    
    # 2. Calculate true win-rates for all players
    stats = df_players.groupby('name').agg(
        total_games=('episode_id', 'count'),
        total_wins=('is_winner', 'sum')
    ).reset_index()
    
    stats['win_rate'] = stats['total_wins'] / stats['total_games']
    
    # 3. Filter for minimum games to remove 1-hit wonders
    stats_filtered = stats[stats['total_games'] >= min_games].copy()
    print(f"Found {len(stats_filtered)} players with >={min_games} games.")
    
    # 4. Sort to find the Top Grandmasters
    grandmasters = stats_filtered.sort_values(by=['win_rate', 'total_wins'], ascending=[False, False]).head(top_k)
    
    print(f"\n--- TOP {top_k} GRANDMASTERS ---")
    for i, row in grandmasters.iterrows():
        print(f"{row['name']}: {row['win_rate']*100:.1f}% WR ({row['total_wins']}/{row['total_games']} games)")
        
    gm_names = set(grandmasters['name'].values)
    
    # 5. Extract all episode_ids where a Grandmaster WON
    gm_wins = df_players[(df_players['name'].isin(gm_names)) & (df_players['is_winner'] == 1)].copy()
    
    # --- RECENCY LOGIC: STATE BUDGETING ---
    target_episodes = target_states // avg_ticks_per_game
    
    # Sort by episode_id DESCENDING to push the newest games to the top
    gm_wins_recent = gm_wins.sort_values(by='episode_id', ascending=False).reset_index(drop=True)
    
    # Slice the exact number of episodes we need from the top of the sorted list
    if len(gm_wins_recent) > target_episodes:
        final_episodes = gm_wins_recent.head(target_episodes)
        print(f"\nCapping dataset: Selected the {target_episodes} MOST RECENT GM wins to yield ~{target_states} states.")
    else:
        final_episodes = gm_wins_recent
        estimated_states = len(final_episodes) * avg_ticks_per_game
        print(f"\nWarning: Only found {len(final_episodes)} GM wins total. Yielding ~{estimated_states} states.")
        
    gm_episode_ids = set(final_episodes['episode_id'].values)
    
    # Save the full leaderboard for future reference
    leaderboard_file = os.path.join(db_path, "leaderboard.csv")
    stats.sort_values(by=['win_rate', 'total_wins'], ascending=[False, False]).to_csv(leaderboard_file, index=False)
    print(f"Saved full leaderboard to {leaderboard_file}")
    
    # Save the target episode IDs
    out_file = os.path.join(db_path, "grandmaster_episodes.csv")
    pd.DataFrame({"episode_id": list(gm_episode_ids)}).to_csv(out_file, index=False)
    print(f"Saved {len(gm_episode_ids)} target episodes to {out_file}")

if __name__ == "__main__":
    label_grandmasters()