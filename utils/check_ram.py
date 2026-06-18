import pandas as pd
import os
import sys

db_path = "data/parquet_db_real"
df_planet_state = pd.read_parquet(os.path.join(db_path, "planet_state.parquet"))
print(f"Planet State RAM: {df_planet_state.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
df_actions = pd.read_parquet(os.path.join(db_path, "actions.parquet"))
print(f"Actions RAM: {df_actions.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
