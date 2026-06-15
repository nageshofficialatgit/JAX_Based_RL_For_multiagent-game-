import os
import pandas as pd
import numpy as np

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    db_path = os.path.join(base_dir, "parquet_db_real")
    
    print("Loading Planet States (This may take a minute due to 32M rows)...")
    # Only load what we need to prevent OOM
    df_state = pd.read_parquet(os.path.join(db_path, "planet_state.parquet"), engine='pyarrow', columns=['episode_id', 'tick', 'planet_id', 'owner'])
    
    print("Sorting timelines...")
    df_state = df_state.sort_values(by=['episode_id', 'planet_id', 'tick'])
    
    print("Calculating Ownership Transitions...")
    # Find when a planet changes owner
    df_state['prev_owner'] = df_state.groupby(['episode_id', 'planet_id'])['owner'].shift(1)
    
    # Filter to only the exact ticks where ownership changed
    transitions = df_state[(df_state['owner'] != df_state['prev_owner']) & (df_state['prev_owner'].notna())].copy()
    
    # Free up memory immediately
    del df_state
    import gc
    gc.collect()
    
    print("Analyzing Snipes (Capture -> Lost within 25 ticks)...")
    # For each transition, how long until the NEXT transition?
    transitions['next_owner'] = transitions.groupby(['episode_id', 'planet_id'])['owner'].shift(-1)
    transitions['next_tick'] = transitions.groupby(['episode_id', 'planet_id'])['tick'].shift(-1)
    transitions['ticks_held'] = transitions['next_tick'] - transitions['tick']
    
    # A "Neutral Capture" is when prev_owner == -1
    neutral_captures = transitions[transitions['prev_owner'] == -1]
    total_neutral_captures = len(neutral_captures)
    
    # A "Snipe" is when a player captures a neutral, but loses it in < 25 ticks
    snipes = neutral_captures[neutral_captures['ticks_held'] <= 25]
    total_snipes = len(snipes)
    snipe_rate = total_snipes / total_neutral_captures if total_neutral_captures > 0 else 0
    
    # ---------------------------------------------------------
    print("Loading Episode Planets...")
    df_planets = pd.read_parquet(os.path.join(db_path, "episode_planets.parquet"), engine='pyarrow')
    
    print("Analyzing Orbit vs Static Preferences...")
    # Merge neutral captures with planet attributes
    captured_planets_attr = neutral_captures.merge(df_planets, on=['episode_id', 'planet_id'], how='inner')
    
    static_captures = len(captured_planets_attr[captured_planets_attr['is_static'] == True])
    orbit_captures = len(captured_planets_attr[captured_planets_attr['is_static'] == False])
    
    # ---------------------------------------------------------
    print("Analyzing Neutral Planet ROI (Production / Initial Garrison)...")
    # Get all planets that started as Neutral
    initial_neutrals = df_planets[df_planets['initial_owner'] == -1].copy()
    
    # Calculate ROI (Production per ship cost)
    # Using max(1) to avoid division by zero
    initial_neutrals['roi'] = initial_neutrals['production'] / initial_neutrals['initial_ships'].clip(lower=1)
    
    # Did they get captured? (Check if they exist in neutral_captures)
    captured_keys = set(zip(neutral_captures['episode_id'], neutral_captures['planet_id']))
    initial_neutrals['was_captured'] = initial_neutrals.apply(lambda r: (r['episode_id'], r['planet_id']) in captured_keys, axis=1)
    
    captured_roi = initial_neutrals[initial_neutrals['was_captured'] == True]['roi'].mean()
    ignored_roi = initial_neutrals[initial_neutrals['was_captured'] == False]['roi'].mean()
    
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("               TACTICAL META-GAME ANALYSIS REPORT")
    print("="*80)
    
    print("\n1. THE SNIPE META (25-Tick Window)")
    print(f"Total Neutral Planets Captured: {total_neutral_captures:,}")
    print(f"Total Times the Conqueror Got Sniped: {total_snipes:,}")
    print(f"Snipe Incident Rate: {snipe_rate:.2%} -> If you take a neutral, there is a {snipe_rate:.2%} chance someone steals it while you're weak!")
    
    print("\n2. TARGET PREFERENCE (Orbit vs Static)")
    print(f"Static Planets Captured: {static_captures:,}")
    print(f"Orbiting Planets Captured: {orbit_captures:,}")
    total_caps = static_captures + orbit_captures
    if total_caps > 0:
        print(f"Ratio: {static_captures/total_caps:.1%} Static / {orbit_captures/total_caps:.1%} Orbiting")
    
    print("\n3. THE SMART BOT THEORY (Production / Garrison ROI)")
    print(f"Average ROI of Neutrals that get ATTACKED: {captured_roi:.4f}")
    print(f"Average ROI of Neutrals that get IGNORED:  {ignored_roi:.4f}")
    
    if ignored_roi < captured_roi:
        print(f"-> THEORY CONFIRMED: Players actively ignore trash planets with huge garrisons and low production! (Ignored ROI is {((captured_roi-ignored_roi)/captured_roi)*100:.1f}% worse)")
    else:
        print("-> Theory busted? Check the raw numbers.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
