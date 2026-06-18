import jax
import jax.numpy as jnp
from src.data_pipeline.dataset_grain import build_grain_dataloader
import numpy as np

dl = build_grain_dataloader(worker_count=1, num_epochs=1, batch_size=2)
batch = next(iter(dl))

print("Batch Angle:")
print(batch['target_angle'])

print("Is Owned:")
print(batch['is_owned_by_winner'])
