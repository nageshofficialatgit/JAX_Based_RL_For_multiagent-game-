# Fast & Distributed RL Architecture: Engineering & Design

Building a superhuman Reinforcement Learning agent requires an infrastructure capable of streaming billions of state transitions per hour. In the *Orbit Wars* environment, the raw dataset consists of over **508 million physics states**, totaling more than **50GB of raw data**. 

Standard Python data loaders (like PyTorch `DataLoader` or raw Pandas) instantly collapse under this weight, either triggering the Linux Out-Of-Memory (OOM) Killer or deadlocking the CPU via the Global Interpreter Lock (GIL).

This document outlines the advanced engineering and design patterns implemented in our pipeline to achieve **zero-RAM-spike distributed streaming** using JAX, Flax, and Google Grain.

---

## 1. Zero-Copy Shared Memory (The `mmap` Fix)

### The Problem
When using standard multiprocessing (`worker_count > 0`), Linux duplicates the memory space of the parent process for every worker. If the parent holds a 6GB NumPy array of planetary states, spawning 8 workers instantly demands 48GB of RAM, triggering an immediate OOM crash.

### The Engineering Solution
We utilize **Zero-Copy Shared Memory** via `numpy.memmap`.
Instead of storing the 508M-row array in RAM, we write it to a raw `.npy` binary file on disk exactly once. We then connect to this file using `mmap_mode='r'`. 

**Why it works:**
The Linux kernel takes over memory management. When Worker 1, Worker 2, and Worker 8 all request row `#5000`, the kernel loads that exact page from the SSD into the **Page Cache** and hands all 8 workers the exact same physical memory pointer. 
- **Result:** RAM usage stays completely flat, allowing infinite multi-core scaling.

---

## 2. Bypassing the IPC Pickling Deadlock

### The Problem
Even with `mmap` solving the raw array memory, Python's `multiprocessing.spawn` method has a fatal flaw: **The IPC Pipe**. 
When a worker is spawned, the parent process attempts to serialize (via `pickle`) the entire `OrbitWarsDataSource` object and push it through the Inter-Process Communication pipe. Because our object contained massive metadata dictionaries (`episode_planets_dict` and a 3.2-million key `state_lookup` index), the single-threaded `pickle` module would peg the CPU at 100% and completely deadlock the pipeline.

### The Engineering Solution
**Serialization Evasion via Disk-Caching.**
1. In the `__init__` of the main process, we dump the massive dictionaries to a local `.pkl` file.
2. We explicitly **delete** the dictionaries from the `self` object (`self.state_lookup = None`).
3. The parent process now passes an empty, lightweight shell to the 8 workers (taking 0.001 seconds).
4. Inside `__getitem__`, we implemented a **Worker Lazy-Load**. The first time a worker tries to fetch a batch, it reads the `.pkl` file from the disk directly into its own local memory.
- **Result:** Complete elimination of the IPC bottleneck. Workers spawn instantly and independently.

---

## 3. Polars: Rust-Backed High Throughput I/O

### The Problem
Loading hundreds of megabytes of Parquet files using `pandas` is slow and consumes massive amounts of intermediate RAM due to object wrappers and `MultiIndex` overhead.

### The Engineering Solution
We migrated all core I/O operations to **Polars**, a dataframe library built entirely in Rust. 
Polars executes operations like `.filter()`, `.sort()`, and `.partition_by()` using aggressive multi-threading and zero-copy memory representations. 
- **Result:** Dataset loading times dropped from several minutes to under 15 seconds.

---

## 4. Bare-Metal Physics via Numba JIT

### The Problem
Inside the dataloader `__getitem__`, the pipeline has to calculate complex physics:
1. Interpolating fleet speeds using logarithmic curves.
2. Raycasting dynamic intercept paths against the Sun's radius to determine `path_blocked`.

Doing this in pure Python for 8,192 items per batch causes severe CPU bottlenecking, crippling the GPU which is forced to wait for data.

### The Engineering Solution
We wrapped the mathematical bottlenecks in `@njit(cache=True)`. 
Numba's Just-In-Time compiler translates the Python physics loops directly into optimized LLVM machine code. We specifically engineered the arrays to use contiguous memory blocks to maximize CPU L1 cache hits.
- **Result:** Physics calculations execute at C/C++ speeds, fully saturating the Dataloader throughput so the JAX GPU runs at 100% utilization.

---

## 5. Google Grain Integration

To orchestrate this architecture, we utilize **Google Grain**, a deterministic, distributed data-loading framework designed specifically for JAX.
By leveraging Grain alongside our zero-copy memory maps and Numba physics, the pipeline can:
- Guarantee deterministic shuffling and replayability (critical for RL debugging).
- Survive mid-training pre-emptions without losing position in the dataset.
- Stream data directly from NVMe SSDs to the GPU with sub-millisecond overhead.
