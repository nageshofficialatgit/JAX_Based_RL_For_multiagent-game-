import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np

# Mocking the model to just check shapes and outputs
from src.models.entity_transformer_flax_small import EntityTransformer

rngs = nnx.Rngs(42)
m = EntityTransformer(rngs=rngs)

B = 16
batch_state = jnp.zeros((B, 70, 14), dtype=jnp.float32)
batch_launch = jnp.zeros((B, 50), dtype=jnp.float32)
batch_angle = jnp.full((B, 50), -1, dtype=jnp.int32)
batch_ships = jnp.full((B, 50), -1, dtype=jnp.int32)

v_logits, launch_logits, angle_logits, ships_logits, _ = m(
    batch_state, 
    return_policy=True,
    target_launch=batch_launch,
    target_angle=batch_angle
)

batch_owned = jnp.ones((B, 50), dtype=jnp.float32)
owner_mask = (batch_owned == 1.0).astype(jnp.float32)
valid_target_mask = owner_mask * (batch_angle != -1)

angle_ce = optax.softmax_cross_entropy_with_integer_labels(angle_logits, jnp.maximum(batch_angle, 0))
angle_loss = jnp.sum(angle_ce * valid_target_mask) / jnp.maximum(jnp.sum(valid_target_mask), 1.0)

print(f"angle_logits shape: {angle_logits.shape}")
print(f"angle_ce shape: {angle_ce.shape}")
print(f"angle_loss: {angle_loss}")
print(f"jnp.sum(valid_target_mask): {jnp.sum(valid_target_mask)}")
