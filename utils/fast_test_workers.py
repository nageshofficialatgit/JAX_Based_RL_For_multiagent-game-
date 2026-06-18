import os
import psutil
import time
import pickle
import multiprocessing as mp

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import grain.python as grain
from src.data_pipeline.dataset_grain_v2 import OrbitWarsDataSource

def print_mem(msg):
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss
    print(f"[{p.pid}] {msg}: {rss / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    print_mem("Start (Main Process)")
    
    # Fast bypass: just load the pre-built index instead of parsing 50GB Parquet!
    source = OrbitWarsDataSource(db_path="parquet_db_real", grandmaster_only=False, subsample_ticks=5, is_iql_mode=True)
    
    # We force the parent to drop its arrays (simulating the end of __init__)
    source.state_lookup = None
    source.planet_state_arr = None
    source.episode_planets_dict_np = None
    source.actions_dict_np = None
    
    # Subset
    source.state_index = source.state_index[:50]
    
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
            print_mem("After first batch (Workers fully loaded)")
            os.system("ps -o pid,ppid,user,%mem,rss,cmd -p $(pgrep -f fast_test_workers.py)")
        if i > 5:
            break

    print("Done")
