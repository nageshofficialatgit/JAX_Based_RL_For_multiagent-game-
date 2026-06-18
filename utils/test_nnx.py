import jax
import jax.numpy as jnp
from src.models.entity_transformer_flax_v2 import EntityTransformer
from flax import nnx

rngs = nnx.Rngs(0)
model = EntityTransformer(num_features=37, rngs=rngs)
dummy_x = jnp.ones((2, 50, 37))
# make planet 1 active
dummy_x = dummy_x.at[0, 1, 0].set(0.0) 

v, launch, ships, angle, ppo_v = model(dummy_x, return_policy=True)
print("v_logits sum:", jnp.sum(v))
print("ppo_v sum:", jnp.sum(ppo_v))
print("is NaN?", jnp.isnan(jnp.sum(v)))
