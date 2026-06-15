import jax
jax.config.update('jax_platform_name', 'cpu')
import numpy as np
from src.data_pipeline.dataset_grain import build_grain_dataloader
import time

loader = build_grain_dataloader(batch_size=2000, worker_count=1, num_epochs=1)
it = iter(loader)
print("Fetching first batch...")
batch = next(it)
state = batch['state_tokens'] # [B, 70, 14]
launch = batch['target_launch'] # [B, 50]
angle = batch['target_angle'] # [B, 50]

valid_planets = (state[:, 1:51, 0] == 0.0)

for b in range(launch.shape[0]):
    for p in range(50):
        if launch[b, p] == 1.0 and angle[b, p] >= 0:
            target = angle[b, p]
            if not valid_planets[b, target]:
                print(f"BINGO! Launch at {p} targets padded planet {target}!")
                print("state[:, 0]:", state[b, 1:51, 0])
                exit(0)

print("No padded planets targeted. Checked", launch.shape[0], "samples.")
