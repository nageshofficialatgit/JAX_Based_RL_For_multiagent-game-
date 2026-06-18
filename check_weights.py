import os
import sys
import jax
import jax.numpy as jnp
from flax import nnx
from flax import serialization

sys.path.insert(0, os.path.abspath("src"))
from models.entity_transformer_flax_v2 import EntityTransformer

model = EntityTransformer(num_features=37, num_classes=5, rngs=nnx.Rngs(0))
graphdef, state = nnx.split(model)

with open("checkpoints/ppo_v2_3550.bin", "rb") as f:
    raw_bytes = f.read()

state_flat = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, state)
state_restored = serialization.from_bytes(state_flat, raw_bytes)

has_nan = False
has_inf = False
max_val = 0.0

for path, leaf in jax.tree_util.tree_leaves_with_path(state_restored):
    val = jnp.asarray(leaf)
    
    # Safely format path
    path_parts = []
    for p in path:
        if hasattr(p, 'key'):
            path_parts.append(str(p.key))
        elif hasattr(p, 'idx'):
            path_parts.append(str(p.idx))
        else:
            path_parts.append(str(p))
    path_str = ".".join(path_parts)
    
    if jnp.isnan(val).any():
        print(f"NaN found in: {path_str}")
        has_nan = True
    if jnp.isinf(val).any():
        print(f"Inf found in: {path_str}")
        has_inf = True
        
    m_val = float(jnp.max(jnp.abs(val)))
    if m_val > max_val:
        max_val = m_val
        
    if m_val > 50.0:
         print(f"Large weight in {path_str}: max_abs={m_val:.2f}")

print(f"\nOverall Max Abs Weight: {max_val:.2f}")
if has_nan:
    print("Model CONTAINS NaNs!")
elif has_inf:
    print("Model CONTAINS Infs!")
else:
    print("Model is clean of NaNs and Infs.")
