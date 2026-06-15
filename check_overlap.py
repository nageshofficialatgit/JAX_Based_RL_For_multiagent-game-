import pandas as pd
import os

parent = pd.read_parquet("parquet_db_real/episodes.parquet")
working = pd.read_parquet("parquet_db_real/working/episodes.parquet")

print("Parent episodes:", len(parent))
print("Working episodes:", len(working))
overlap = set(parent['episode_id']).intersection(set(working['episode_id']))
print("Overlap count:", len(overlap))
