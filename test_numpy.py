import pandas as pd
import numpy as np

df = pd.read_parquet("data/parquet_db_real/planet_state.parquet")
print("Total rows:", len(df))
print("Memory usage:\n", df.memory_usage(deep=True))
print("Index memory usage:", df.index.memory_usage(deep=True))
