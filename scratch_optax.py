import jax
import jax.numpy as jnp
from flax import nnx
import optax

class SimpleModel(nnx.Module):
    def __init__(self, rngs):
        self.linear = nnx.Linear(10, 10, rngs=rngs)
    def __call__(self, x):
        return jnp.sum(self.linear(x))

rngs = nnx.Rngs(0)
model = SimpleModel(rngs)
optimizer = optax.adam(1e-3)

graphdef, state = nnx.split(model)

# Function to extract raw arrays
def get_raw(s):
    # nnx.State can be flattened
    leaves, treedef = jax.tree_util.tree_flatten(s)
    # leaves are VariableState. We want their .value
    raw_leaves = [l.value if hasattr(l, 'value') else l for l in leaves]
    return jax.tree_util.tree_unflatten(treedef, raw_leaves)

def set_raw(s, raw_s):
    leaves, treedef = jax.tree_util.tree_flatten(s)
    raw_leaves, _ = jax.tree_util.tree_flatten(raw_s)
    new_leaves = []
    for l, rl in zip(leaves, raw_leaves):
        # We can't mutate l inside JIT, we must create a new VariableState
        # Wait, if l is a VariableState, we can replace it?
        # Actually, nnx.State allows creation from raw dicts
        pass

raw_state = get_raw(state)
opt_state = optimizer.init(raw_state)

@jax.jit
def train_step(state, opt_state, x):
    def loss_fn(s):
        m = nnx.merge(graphdef, s)
        return m(x)
    
    loss, grads = jax.value_and_grad(loss_fn)(state)
    
    raw_grads = get_raw(grads)
    raw_state = get_raw(state)
    
    updates, opt_state = optimizer.update(raw_grads, opt_state, raw_state)
    raw_state = optax.apply_updates(raw_state, updates)
    
    # Can we just use nnx.State to reconstruct?
    # No, nnx.update handles this!
    # Wait, nnx.update is NOT pure. It mutates the model.
    # What if we just do:
    # m = nnx.merge(graphdef, state)
    # nnx.update(m, raw_state) # Does nnx.update accept raw dicts? Let's check!
    return loss, raw_state

x = jnp.ones((10,))
try:
    loss1, state1 = train_step(state, opt_state, x)
    print("Success 1:", loss1)
    # The crucial test: Can we merge graphdef with the raw updated dict?
    m2 = nnx.merge(graphdef, state1)
    loss2, state2 = train_step(state1, opt_state, x)
    print("Success 2:", loss2)
except Exception as e:
    print("Error:", e)
