import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from src.models.entity_transformer_flax_v2 import EntityTransformer

model = EntityTransformer(num_features=37, num_classes=5, rngs=nnx.Rngs(0))
state, graphdef = nnx.split(model)

mngr = ocp.CheckpointManager('checkpoints/bc_v2')
raw = mngr.restore(120000)

def safe_merge(path, leaf_state):
    curr = raw.get('model', raw)
    for key_obj in path:
        if isinstance(key_obj, jax.tree_util.DictKey):
            k = key_obj.key
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return leaf_state
        elif isinstance(key_obj, jax.tree_util.SequenceKey):
            idx = key_obj.idx
            if isinstance(curr, (list, tuple)) and idx < len(curr):
                curr = curr[idx]
            elif isinstance(curr, dict) and str(idx) in curr:
                curr = curr[str(idx)]
            else:
                return leaf_state
        else:
            return leaf_state
    return jax.numpy.asarray(curr, dtype=leaf_state.value.dtype) if hasattr(leaf_state, 'value') else curr

state = jax.tree_util.tree_map_with_path(safe_merge, state)
nnx.update(model, state)

dummy_obs = jnp.ones((2, 50, 37), dtype=jnp.float32)
dummy_obs = dummy_obs.at[:, 0, 0].set(-1.0)
valid_mask = jnp.ones((2, 50), dtype=jnp.bool_)

def loss_fn(s_flat):
    merged = nnx.merge(graphdef, s_flat)
    v, l, s, a, ppo = merged(dummy_obs, return_policy=True, valid_launch_mask=valid_mask)
    return jnp.sum(l) + jnp.sum(s) + jnp.sum(a) + jnp.sum(ppo)

grad_fn = jax.value_and_grad(loss_fn)
val, grads = grad_fn(state)

print(f"Value: {val}")
has_nan = False
for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
    if jnp.isnan(leaf.value).any():
        print(f"NaN gradient in: {path}")
        has_nan = True
if not has_nan:
    print("No NaN gradients found!")

