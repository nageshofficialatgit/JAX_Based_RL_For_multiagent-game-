import pandas as pd
import numpy as np

def verify_claim():
    # Load a few rows from planet_state and episodes
    print("Loading samples from dataset...")
    df_planets = pd.read_parquet("parquet_db_real/planet_state.parquet")
    df_episodes = pd.read_parquet("parquet_db_real/episodes.parquet")
    
    # Let's find an episode where Player 0 won
    # Player 0 won if agents[0] matches the winner (which might not be explicitly stored, but we can look at the winner index)
    # The dataset_grain_v2.py uses `winner_slot` = 0 or 1.
    
    # For demonstration, we'll just synthesize a small tensor representing the owners to prove the math
    # Parquet uses -1 for Neutral, and 0, 1 for players.
    print("\n--- SYNTHETIC VERIFICATION ---")
    raw_owner = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
    print(f"Raw Parquet Owners: {raw_owner}  (Neutral is -1. Players are 0-3)")
    
    # Case 1: Player 0 won
    winner_slot = 0.0
    print(f"\nScenario: Player 0 won the game (winner_slot = {winner_slot})")
    
    # Old Buggy Logic
    old_ego_owner = np.where(
        raw_owner == 0, 0.0,
        np.where(
            raw_owner == winner_slot, 1.0,
            np.where(raw_owner < winner_slot, raw_owner + 1.0, raw_owner.astype(np.float32))
        )
    )
    print(f"OLD BUGGY MAPPING: {old_ego_owner}")
    print("Notice that:")
    print("1) Neutral (-1) became 0.0 (Correct, but accidentally)")
    print("2) Player 0 (0) became 0.0 (FATAL! Player 0 was marked as Neutral!)")
    print("3) Player 1 (1) became 1.0 (FATAL! Player 1 was marked as Ego, even though Player 0 won!)")

    # New Fixed Logic
    new_ego_owner = np.where(
        raw_owner == -1.0, 0.0,
        np.where(
            raw_owner == winner_slot, 1.0,
            np.where(raw_owner < winner_slot, raw_owner + 2.0, raw_owner + 1.0)
        )
    )
    print(f"\nNEW FIXED MAPPING: {new_ego_owner}")
    print("Notice that:")
    print("1) Neutral (-1) is explicitly 0.0")
    print("2) Player 0 (winner) is correctly mapped to 1.0 (Ego)")
    print("3) Player 1 is correctly mapped to 2.0 (Enemy 1)")

    # Case 2: Player 1 won
    winner_slot = 1.0
    print(f"\n-------------------------------------------------")
    print(f"\nScenario: Player 1 won the game (winner_slot = {winner_slot})")
    
    # Old Buggy Logic
    old_ego_owner = np.where(
        raw_owner == 0, 0.0,
        np.where(
            raw_owner == winner_slot, 1.0,
            np.where(raw_owner < winner_slot, raw_owner + 1.0, raw_owner.astype(np.float32))
        )
    )
    print(f"OLD BUGGY MAPPING: {old_ego_owner}")
    print("Notice that:")
    print("1) Neutral (-1) became 0.0 (Correct)")
    print("2) Player 0 (0) became 0.0 (FATAL! Player 0 was marked as Neutral, instead of Enemy!)")
    print("3) Player 1 (1) became 1.0 (Correct, because it matched winner_slot)")
    print("4) Player 2 (2) became 2.0 (Correct)")
    
    # New Fixed Logic
    new_ego_owner = np.where(
        raw_owner == -1.0, 0.0,
        np.where(
            raw_owner == winner_slot, 1.0,
            np.where(raw_owner < winner_slot, raw_owner + 2.0, raw_owner + 1.0)
        )
    )
    print(f"\nNEW FIXED MAPPING: {new_ego_owner}")
    print("Notice that:")
    print("1) Neutral (-1) is 0.0")
    print("2) Player 0 is mapped to 2.0 (Enemy 1)")
    print("3) Player 1 (winner) is mapped to 1.0 (Ego)")
    print("4) Player 2 is mapped to 3.0 (Enemy 2)")


if __name__ == "__main__":
    verify_claim()
