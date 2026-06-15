import os
import jax
import jax.numpy as jnp
from flax import nnx
from flax import serialization
import optax
import orbax.checkpoint as ocp
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.entity_transformer_flax_small import EntityTransformer

def force_to_pure_dict(node):
    if hasattr(node, 'items'): return {k: force_to_pure_dict(v) for k, v in node.items()}
    elif hasattr(node, 'value'): return force_to_pure_dict(node.value)
    elif isinstance(node, (list, tuple)): return type(node)(force_to_pure_dict(v) for v in node)
    return node

def export_checkpoint(step):
    print(f"Exporting BC checkpoint {step} to .bin...")
    rngs = nnx.Rngs(42)
    model = EntityTransformer(num_classes=5, rngs=rngs)
    
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../checkpoints/bc_light'))
    mngr = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(create=False))
    
    # Use EXACT optimizer from BC script
    lr_schedule = optax.warmup_cosine_decay_schedule(init_value=1e-4, peak_value=3e-4, warmup_steps=1000, decay_steps=10000, end_value=1e-5)
    dummy_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4))
    dummy_opt = nnx.Optimizer(model, dummy_tx)
    try:
        template = {'model': nnx.state(model), 'opt': nnx.state(dummy_opt)}
        restored = mngr.restore(step, args=ocp.args.StandardRestore(template))
        # The restored dict has the raw states. Need to cast to bfloat16.
        restored_casted = jax.tree_util.tree_map(
            lambda r, t: jnp.asarray(r, dtype=t.value.dtype) if hasattr(t, 'value') else r,
            restored['model'], template['model']
        )
        nnx.update(model, restored_casted)
    except Exception as e:
        print(f"Failed to load checkpoint {step}: {e}")
        return
        
    _, state_flat = nnx.split(model)
    pure_dict = force_to_pure_dict(state_flat)
    
    benchmark_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../checkpoints/benchmarks'))
    os.makedirs(benchmark_dir, exist_ok=True)
    bin_path = os.path.join(benchmark_dir, f"bc_light_{step}.bin")
    
    with open(bin_path, "wb") as f:
        f.write(serialization.to_bytes(pure_dict))
    print(f"Successfully exported {bin_path}")

if __name__ == "__main__":
    for step in [55000, 57500, 60000]:
        export_checkpoint(step)
