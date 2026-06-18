import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import optax

logits = jnp.zeros((2, 50, 50))
labels = jnp.zeros((2, 50), dtype=jnp.int32)
ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
print("CE shape:", ce.shape)
print("CE max value:", jnp.max(ce))
