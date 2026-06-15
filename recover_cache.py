import os
import numpy as np
import glob
from tqdm import tqdm

dataset_cache_dir = "/home/medhasree_2121cs05/2201cs50_nagesh/server_deploy/parquet_db_real/bc_v2_disk_cache"
tmp_dir = os.path.join(dataset_cache_dir, "tmp_chunks")
flag_file = os.path.join(dataset_cache_dir, "CACHE_READY.flag")

chunk_files = sorted(glob.glob(os.path.join(tmp_dir, "chunk_*.npz")), key=lambda x: int(x.split('_')[-1].split('.')[0]))
print(f"Found {len(chunk_files)} chunks. Starting recovery...")

if len(chunk_files) == 0:
    print("No chunks found!")
    exit(1)

first_batch_metadata = None
total_rows = 0

for f in tqdm(chunk_files, desc="Calculating shapes"):
    # Just load the first one for metadata, but we must load all to get total rows reliably
    # Actually, we can assume all but the last are batch_size.
    # But since np.load lazily loads shapes via .zip, it's fast enough.
    with np.load(f) as data:
        if first_batch_metadata is None:
            first_batch_metadata = {k: (data[k].shape, data[k].dtype) for k in data.keys()}
        total_rows += data["state_tokens"].shape[0]

print(f"Pre-allocating contiguous disk space for {total_rows} rows...")
memmaps = {}
for k, (shape, dtype) in first_batch_metadata.items():
    full_shape = (total_rows,) + shape[1:]
    mmap_path = os.path.join(dataset_cache_dir, f"{k}.npy")
    memmaps[k] = np.lib.format.open_memmap(mmap_path, mode='w+', shape=full_shape, dtype=dtype)

print("Consolidating chunks into final disk arrays...")
current_idx = 0
for b_path in tqdm(chunk_files, desc="Consolidating"):
    with np.load(b_path) as data:
        b_size = data["state_tokens"].shape[0]
        for k in memmaps.keys():
            memmaps[k][current_idx:current_idx+b_size] = data[k]
    current_idx += b_size

for k, m in memmaps.items():
    m.flush()
    del m

with open(flag_file, 'w') as f:
    f.write("READY")

print("Consolidation complete! You can now run train_bc_v2.py instantly.")
