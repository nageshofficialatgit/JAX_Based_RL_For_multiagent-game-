# Fixing the Multiprocessing IPC Bottleneck

Currently, `workers=0` processes 1 batch every ~31 seconds, which would take 3.4 hours to spool the entire dataset. This is far too slow. 

The previous 8-worker attempt failed because of the **IPC (Inter-Process Communication) Pipe**. Even though we solved the 50GB `planet_state_arr` with `mmap`, the class `OrbitWarsDataSource` still contains multiple metadata dictionaries:
1. `state_lookup` (3.2 million keys)
2. `episode_planets_dict_np` 
3. `actions_dict_np` 

When the `spawn` multiprocessing method creates 8 worker Python processes, the parent process tries to serialize these large dictionaries and push them through a small IPC pipe. This causes a single-threaded deadlock that pegs the CPU at 100% and hangs indefinitely.

## Proposed Changes

We will completely bypass the IPC pipe by moving the dictionary serialization to the hard drive:

### [MODIFY] `src/data_pipeline/dataset_grain_v2.py`
1. In `__init__`, after building all the metadata dictionaries, we will save them to a `metadata_dicts_cache.pkl` file using Python's `pickle` module with `HIGHEST_PROTOCOL`.
2. Crucially, we will **delete** these dictionaries from the `self` object (`self.state_lookup = None`, etc.) in the parent process.
3. This guarantees the `OrbitWarsDataSource` object is practically empty and can be passed through the IPC pipe to the 8 workers in microseconds.
4. In `__getitem__`, we will add a check: if `self.state_lookup is None`, the worker will load the `metadata_dicts_cache.pkl` file directly from its own local disk. 

### [MODIFY] `src/training/train_bc_v2.py`
1. Revert `workers = 0` back to `workers = 8`. 

This will result in the 8 workers spawning instantly, each spending ~2 seconds loading the metadata directly from disk into their own RAM, and then seamlessly processing the batches in parallel, slashing the 3.4 hour ETA down to minutes.
