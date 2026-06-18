import jax
import jax.numpy as jnp
import numpy as np
import os
import sys

from agent_bc import BCAgent
import json

# Setup agent
agent = BCAgent()

# Load a test observation (we can use raw data or generate dummy)
obs = {
    "step": 10,
    "player": 0,
    "planets": [
        [0, 0, 10, 10, 5, 100, 2],
        [1, 1, 90, 90, 5, 100, 2],
        [2, 0, 50, 50, 5, 10, 1],
    ],
    "angular_velocity": 0.05
}

p_tensor, id_to_sorted_idx, sorted_idx_to_p, player_id = agent.extract_features(obs)
tensor_jax = jnp.array(p_tensor)[None, ...]

valid_mask = jnp.zeros((1, 50), dtype=jnp.bool_)
valid_mask = valid_mask.at[0, 0].set(True) # Just fake it

_, l_logits, a_logits, s_logits, _ = agent.merged_model(
    tensor_jax, return_policy=True, valid_launch_mask=valid_mask
)

probs = jax.nn.sigmoid(l_logits[0])
print("Launch Probs:", probs[:3])
print("Max prob:", jnp.max(probs))
print("Mean prob:", jnp.mean(probs))

