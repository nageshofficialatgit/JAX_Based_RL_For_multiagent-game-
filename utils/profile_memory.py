import os
import psutil
import polars as pl
import gc

def print_mem(msg):
    process = psutil.Process(os.getpid())
    print(f"{msg}: {process.memory_info().rss / 1024 / 1024:.2f} MB")

db_path = "parquet_db_real"
print_mem("Start")

# Take just 10MB chunk (e.g. 500,000 rows)
print("Scanning planet_state.parquet...")
df_pl = pl.scan_parquet(os.path.join(db_path, "planet_state.parquet")).head(500000).collect()
print_mem("After Polars collect")

df_pd = df_pl.to_pandas()
print_mem("After Pandas conversion")

temp_arr = df_pd[['planet_id', 'owner', 'ships']].values
print_mem("After numpy values")

ep_ticks = df_pd[['episode_id', 'tick']].values
print_mem("After ep_ticks values")

del df_pd
del df_pl
gc.collect()
print_mem("After GC")
