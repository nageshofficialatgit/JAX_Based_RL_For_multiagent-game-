import jax
import jax.numpy as jnp
from flax import nnx
from flax import serialization
import numpy as np

class SimpleModel(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jnp.array(1.0))
        self.b = nnx.Param(jnp.array(2.0))

m1 = SimpleModel()
m2 = SimpleModel()
m2.w.value = jnp.array(99.0)
m2.b.value = jnp.array(88.0)

_, state2 = nnx.split(m2)
def force_to_pure_dict(pt):
    if hasattr(pt, 'items'):
        return {k: force_to_pure_dict(v) for k, v in pt.items()}
    if hasattr(pt, 'value'):
        return np.array(pt.value)
    return np.array(pt) if isinstance(pt, jax.Array) else pt

state2_dict = force_to_pure_dict(state2)
bytes_data = serialization.to_bytes(state2_dict)

_, state1 = nnx.split(m1)
template = force_to_pure_dict(state1)
restored = serialization.from_bytes(template, bytes_data)

restored_state = nnx.State(restored)
nnx.update(m1, restored_state)

print(f"m1.w: {m1.w.value}")
