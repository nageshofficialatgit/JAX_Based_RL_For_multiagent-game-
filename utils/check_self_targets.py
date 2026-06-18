import numpy as np
from src.data_pipeline.dataset_grain import build_grain_dataloader

loader = build_grain_dataloader(batch_size=2000, worker_count=1, num_epochs=1)
it = iter(loader)
print("Fetching batch...")
batch = next(it)

batch_launch = batch['target_launch']
batch_angle = batch['target_angle']

self_targets = 0
padded_targets = 0
state = batch['state_tokens']
valid_planets = (state[:, 1:51, 0] == 0.0)

for b in range(batch_launch.shape[0]):
    for p in range(50):
        if batch_launch[b, p] == 1.0 and batch_angle[b, p] >= 0:
            target = batch_angle[b, p]
            if target == p:
                self_targets += 1
            if not valid_planets[b, target]:
                padded_targets += 1

print(f"Self targets found: {self_targets}")
print(f"Padded targets found: {padded_targets}")
