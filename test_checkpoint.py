import jax
import orbax.checkpoint as ocp
from models.entity_transformer_flax_v2 import EntityTransformer
from flax import nnx

model = EntityTransformer(num_features=37, num_classes=5, rngs=nnx.Rngs(0))
state = nnx.split(model)[1]
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
                print(f"Path not found: {path}")
                return leaf_state
        elif isinstance(key_obj, jax.tree_util.SequenceKey):
            idx = key_obj.idx
            if isinstance(curr, (list, tuple)) and idx < len(curr):
                curr = curr[idx]
            elif isinstance(curr, dict) and str(idx) in curr:
                curr = curr[str(idx)]
            else:
                print(f"Path not found: {path}")
                return leaf_state
        else:
            print(f"Path not found: {path}")
            return leaf_state
            
    if hasattr(curr, 'shape') and hasattr(leaf_state, 'value'):
        if curr.shape != leaf_state.value.shape:
            print(f"Shape mismatch at {path}: {curr.shape} vs {leaf_state.value.shape}")
            return leaf_state
    return jax.numpy.asarray(curr, dtype=leaf_state.value.dtype) if hasattr(leaf_state, 'value') else curr

merged = jax.tree_util.tree_map_with_path(safe_merge, state)
print("Merge test complete.")
