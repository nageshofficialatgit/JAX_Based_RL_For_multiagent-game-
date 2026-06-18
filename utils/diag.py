from src.data_pipeline.dataset_grain import build_grain_dataloader

base_dir = os.path.abspath(os.path.dirname(__file__))
dataset_path = os.path.join(base_dir, "data", "parquet_db_real")
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(base_dir, "parquet_db_real")
print(f"Using dataset path: {dataset_path}")
dl = build_grain_dataloader(db_path=dataset_path, batch_size=256, worker_count=4)
print(next(iter(dl)))
