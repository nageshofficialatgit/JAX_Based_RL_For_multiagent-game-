import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import jax
import jax.numpy as jnp
from src.env_jax.orbit_env_v2 import reset_env, build_observation
from src.models.entity_transformer_flax_v2 import EntityTransformer
from flax import nnx

# Initialize environment
rng = jax.random.PRNGKey(42)
state = reset_env(rng)

# Build observation
obs = build_observation(state, player_id=1, win_rate=1.0)
print("Obs shape:", obs.shape)
print("Obs contains NaN?", jnp.isnan(obs).any())
print("Obs contains Inf?", jnp.isinf(obs).any())

# Pass through model
model = EntityTransformer(num_features=37, rngs=nnx.Rngs(0))
v, launch, ships, angle, ppo_v = model(obs[None, :, :], return_policy=True)

print("ppo_v:", ppo_v)
print("v_logits contains NaN?", jnp.isnan(v).any())
print("angle_logits contains NaN?", jnp.isnan(angle).any())
