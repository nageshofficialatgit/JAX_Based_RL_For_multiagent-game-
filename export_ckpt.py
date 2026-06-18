import os
import sys
import jax.numpy as jnp
from flax import nnx
from flax import serialization
import orbax.checkpoint as ocp

sys.path.insert(0, os.path.abspath("src"))
from models.entity_transformer_flax_v2 import EntityTransformer

ckpt_dir = "checkpoints/ppo_v2"
mngr = ocp.CheckpointManager(os.path.abspath(ckpt_dir))

# Determine step: either from CLI arg or latest
if len(sys.argv) > 1:
    step = int(sys.argv[1])
else:
    step = mngr.latest_step()
    if step is None:
        print("No checkpoints found!")
        sys.exit(1)

print(f"Exporting step {step}...")
raw = mngr.restore(step)

populated_state = raw['model']

def force_to_pure_dict(node):
    if hasattr(node, 'items'):
        return {k: force_to_pure_dict(v) for k, v in node.items()}
    elif hasattr(node, 'value'):
        return force_to_pure_dict(node.value)
    elif isinstance(node, (list, tuple)):
        return type(node)(force_to_pure_dict(v) for v in node)
    return node

pure_dict = force_to_pure_dict(populated_state)

output_bin = f"checkpoints/ppo_v2_{step}.bin"
with open(output_bin, "wb") as f:
    f.write(serialization.to_bytes(pure_dict))
print(f"Exported step {step} to {output_bin} successfully!")

