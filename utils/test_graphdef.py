import sys
sys.path.append('src')
from models.entity_transformer_flax_small import EntityTransformer
from flax import nnx
import jax

rngs = nnx.Rngs(42)
base_model = EntityTransformer(num_classes=5, rngs=rngs)
graphdef, state_template = nnx.split(base_model)
print("Graphdef leaves:", len(jax.tree_util.tree_leaves(state_template)))
