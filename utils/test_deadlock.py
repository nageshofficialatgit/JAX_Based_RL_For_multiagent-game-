import pandas as pd
import multiprocessing as mp

def worker():
    print("Worker started")
    df = pd.read_parquet("data/parquet_db_real/episode_planets.parquet")
    print("Worker finished")

if __name__ == '__main__':
    print("Main process loading...")
    df = pd.read_parquet("data/parquet_db_real/episodes.parquet")
    print("Forking...")
    p = mp.Process(target=worker)
    p.start()
    p.join()
