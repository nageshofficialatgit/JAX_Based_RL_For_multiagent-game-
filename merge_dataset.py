import os
import pyarrow.parquet as pq
import pyarrow as pa
import shutil
import pandas as pd

parent_dir = "parquet_db_real"
working_dir = "data_backup/parquet_db_real_20260615_045446"
cache_file = os.path.join(parent_dir, "bc_dataset_v2.npz")

files = [
    "actions.parquet",
    "episode_planets.parquet",
    "episodes.parquet",
    "planet_state.parquet",
    "player_episodes.parquet",
    "tick_summary.parquet"
]

print("Starting dataset merge...")

for f in files:
    parent_path = os.path.join(parent_dir, f)
    working_path = os.path.join(working_dir, f)
    
    if os.path.exists(working_path):
        print(f"Reading {working_path}...")
        # Read as pandas to safely deduplicate if needed, though we know there's 0 overlap.
        # But for huge files like planet_state, maybe just pyarrow concat is safer for RAM.
        working_table = pq.read_table(working_path)
        
        # Schema matching: Drop x and y if present, because parent doesn't have them
        if 'x' in working_table.column_names:
            working_table = working_table.drop(['x', 'y'])
            
        if os.path.exists(parent_path):
            print(f"Merging with {parent_path}...")
            parent_table = pq.read_table(parent_path)
            
            if 'x' in parent_table.column_names:
                parent_table = parent_table.drop(['x', 'y'])
                
            merged = pa.concat_tables([parent_table, working_table])
        else:
            merged = working_table
            
        print(f"Saving merged {f} (Total rows: {merged.num_rows})...")
        pq.write_table(merged, parent_path, compression='snappy')
        
# Remove the cache file to force rebuilding
if os.path.exists(cache_file):
    print(f"Removing old cache: {cache_file}")
    os.remove(cache_file)

disk_cache_dir = os.path.join(parent_dir, "bc_v2_disk_cache")
if os.path.exists(disk_cache_dir):
    print(f"Removing old disk cache directory: {disk_cache_dir}")
    shutil.rmtree(disk_cache_dir)
print("Dataset compiled successfully!")
