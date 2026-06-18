import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np

from src.models.entity_transformer_flax_small import EntityTransformer

rngs = nnx.Rngs(42)
m = EntityTransformer(rngs=rngs)

B = 2
batch_state = jnp.zeros((B, 70, 14), dtype=jnp.float32)
batch_launch = jnp.zeros((B, 50), dtype=jnp.float32)
batch_angle = jnp.zeros((B, 50), dtype=jnp.int32)
batch_ships = jnp.zeros((B, 50), dtype=jnp.int32)

# Pass through model
v_logits, launch_logits, angle_logits, ships_logits, _ = m(
    batch_state, 
    return_policy=True,
    target_launch=batch_launch,
    target_angle=batch_angle
)

angle_ce = optax.softmax_cross_entropy_with_integer_labels(angle_logits, batch_angle)

print("angle_logits max:", jnp.max(angle_logits))
print("angle_logits min:", jnp.min(angle_logits))
print("angle_logits mean:", jnp.mean(angle_logits))
print("angle_logits std:", jnp.std(angle_logits))

print("angle_ce max:", jnp.max(angle_ce))
print("angle_ce min:", jnp.min(angle_ce))
print("angle_ce mean:", jnp.mean(angle_ce))
