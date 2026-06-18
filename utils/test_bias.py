import jax
import jax.numpy as jnp
from flax import serialization
import numpy as np

with open('optimized_model.bin', 'rb') as f:
    state_dict = serialization.from_bytes(None, f.read())

bias = state_dict['actor_launch']['bias']
kernel = state_dict['actor_launch']['kernel']

print("Launch Head Bias:", bias)
print("Launch Head Kernel mean:", jnp.mean(kernel))
print("Launch Head Kernel std:", jnp.std(kernel))

# Calculate sigmoid of bias
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("Sigmoid(bias):", sigmoid(bias))
