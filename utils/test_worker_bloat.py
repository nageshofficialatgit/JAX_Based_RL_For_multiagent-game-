import os
import psutil
import time
import multiprocessing as mp

# FORCE SPAWN like the real script
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import grain.python as grain
from src.data_pipeline.dataset_grain_v2 import OrbitWarsDataSource

def get_total_memory():
    # Sum memory of this process and all its children (the Grain workers)
    current_process = psutil.Process(os.getpid())
    total_rss = current_process.memory_info().rss
    for child in current_process.children(recursive=True):
        total_rss += child.memory_info().rss
    return total_rss / 1024 / 1024

def print_mem(msg):
    print(f"{msg}: {get_total_memory():.2f} MB")

if __name__ == '__main__':
    print_mem("Start (Main Process Only)")
    source = OrbitWarsDataSource(db_path="parquet_db_real", grandmaster_only=False, subsample_ticks=5, is_iql_mode=True)
    
    # Tiny chunk: just 1000 items
    source.state_index = source.state_index[:1000]
    print_mem("After source init (Main Process)")

    sampler = grain.SequentialSampler(num_records=len(source.state_index), shard_options=grain.NoSharding())

    dl = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=2,
        worker_buffer_size=2,
    )

    print("Spawning 2 workers...")
    for i, batch in enumerate(dl):
        if i == 0:
            print_mem("After first batch (2 Workers fully spawned & loaded data)")
        if i >= 5:
            break

    print_mem("End of test")
