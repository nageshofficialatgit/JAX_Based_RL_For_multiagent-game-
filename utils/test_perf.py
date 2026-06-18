import numpy as np
import time
from braniac_v2 import fast_score_action_via_delta

p_owners = np.full(50, -1, dtype=np.int32)
p_ships = np.zeros(50, dtype=np.float64)
p_prods = np.zeros(50, dtype=np.float64)
p_x = np.zeros(50, dtype=np.float64)
p_y = np.zeros(50, dtype=np.float64)

arr_pid = np.array([1, 2, 3], dtype=np.int32)
arr_eta = np.array([10, 20, 30], dtype=np.int32)
arr_owner = np.array([0, 1, 2], dtype=np.int32)
arr_ships = np.array([10.0, 20.0, 30.0], dtype=np.float64)

eval_weights = np.ones(12, dtype=np.float64)
arrivals_table = np.zeros((101, 50, 6), dtype=np.float64)

# warmup
fast_score_action_via_delta(p_owners, p_ships, p_prods, p_x, p_y, arr_pid, arr_eta, arr_owner, arr_ships, 100, 0, True, eval_weights, arrivals_table)

t0 = time.time()
n = 10000
for _ in range(n):
    fast_score_action_via_delta(p_owners, p_ships, p_prods, p_x, p_y, arr_pid, arr_eta, arr_owner, arr_ships, 100, 0, True, eval_weights, arrivals_table)
t1 = time.time()

print(f"Time for {n} calls: {(t1-t0):.4f}s")
print(f"Time per call: {((t1-t0)/n * 1e6):.2f} us")
