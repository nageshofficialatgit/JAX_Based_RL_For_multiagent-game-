import jax
import jax.numpy as jnp
import optax
from flax import nnx
from src.models.entity_transformer_flax_small import EntityTransformer

rngs = nnx.Rngs(42)
m = EntityTransformer(rngs=rngs)

B = 2
batch_state = jax.random.normal(jax.random.PRNGKey(0), (B, 70, 14)) * 100.0 # Force huge state
batch_launch = jnp.ones((B, 50), dtype=jnp.float32)
batch_angle = jnp.zeros((B, 50), dtype=jnp.int32)
batch_ships = jnp.zeros((B, 50), dtype=jnp.int32)

v_logits, launch_logits, angle_logits, ships_logits, _ = m(
    batch_state, 
    return_policy=True,
    target_launch=batch_launch,
    target_angle=batch_angle
)

angle_ce = optax.softmax_cross_entropy_with_integer_labels(angle_logits, batch_angle)
pos_mask = jnp.ones((B, 50), dtype=jnp.float32)
target_loss = jnp.sum(angle_ce * pos_mask) / jnp.maximum(jnp.sum(pos_mask), 1.0)

print("Target Loss with HUGE state:", target_loss)
print("Angle Logits Max:", jnp.max(angle_logits))
print("Angle CE Max:", jnp.max(angle_ce))

