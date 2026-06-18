import jax
import jax.numpy as jnp
import numpy as np
from src.data_pipeline.dataset_grain import build_grain_dataloader

dl = build_grain_dataloader(worker_count=1, num_epochs=1, batch_size=256)
batch = next(iter(dl))

state = batch['state_tokens']
print("State max:", np.max(state))
print("State min:", np.min(state))
print("State mean:", np.mean(state))

angles = batch['target_angle']
print("Angles max:", np.max(angles))
print("Angles min:", np.min(angles))

