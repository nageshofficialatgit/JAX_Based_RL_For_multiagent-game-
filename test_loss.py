import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import optax
import numpy as np
from src.data_pipeline.dataset_grain import build_grain_dataloader

loader = build_grain_dataloader(batch_size=2000, worker_count=1, num_epochs=1)
it = iter(loader)
print("Fetching batch...")
batch = next(it)

batch_state = batch['state_tokens']
batch_launch = batch['target_launch']
batch_angle = batch['target_angle']
batch_ships = batch['target_ships']

# Emulate valid_planets mask and diagonal
valid_planets = (batch_state[:, 1:51, 0] == 0.0) # [B, 50]
mask = np.eye(50, dtype=bool)[None, :, :]
mask = mask | (~valid_planets[:, None, :])

# Emulate logits (perfectly uniform except for mask)
angle_logits = np.zeros((batch_launch.shape[0], 50, 50))
angle_logits = np.where(mask, -1e9, angle_logits)

# Safe angle
safe_angle = np.maximum(batch_angle, 0)

# Compute CE manually to find the exact B and P
for b in range(batch_launch.shape[0]):
    for p in range(50):
        if batch_launch[b, p] == 1.0 and batch_angle[b, p] >= 0:
            target = safe_angle[b, p]
            logit = angle_logits[b, p, target]
            if logit < -1000:
                print(f"EXPLOSION at B={b}, Source={p}, Target={target}")
                print(f"valid_planets[Target]: {valid_planets[b, target]}")
                print(f"is_diagonal: {p == target}")
                exit(0)

print("No explosion found in dummy logits.")
