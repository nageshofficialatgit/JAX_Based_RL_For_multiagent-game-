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
    
    # Fast bypass
    class MockSource(OrbitWarsDataSource):
        def __init__(self):
            self.db_path = "parquet_db_real"
            self.subsample_ticks = 5
            self.is_iql_mode = True
            
            # Load metadata
            cache_dict_path = os.path.join(self.db_path, "metadata_dicts_cache.pkl")
            with open(cache_dict_path, 'rb') as f:
                d = pickle.load(f)
                self.state_index = list(d['state_lookup'].keys())
                self.episode_to_winner = d['episode_to_winner']
                self.episode_to_win_rate = d['episode_to_win_rate']
                self.episode_to_angular_vel = d['episode_to_angular_vel']
                
            # Filter to 100 for fast test
            self.state_index = self.state_index[:100]
            
            # Clear worker variables
            self.state_lookup = None
            self.planet_state_arr = None
            self.episode_planets_dict_np = None
            self.actions_dict_np = None

    source = MockSource()
    print_mem("After fast init")
    
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
            os.system("ps -o pid,ppid,user,%mem,rss,cmd -p $(pgrep -f fast_test_workers2.py)")
        if i > 5:
            break

    print("Done")
