import os
import sys
import jax
from flax import nnx
from flax import serialization
import orbax.checkpoint as ocp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from models.entity_transformer_flax_v2 import EntityTransformer

def main():
    print("Exporting bc_v2 model to .bin format for League Evaluator...")
    rngs = nnx.Rngs(42)
    model = EntityTransformer(rngs=rngs)
    
    # Exact optimizer from training to satisfy orbax
    import optax
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5, peak_value=1e-3, warmup_steps=1500, decay_steps=60000, end_value=1e-5
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)
    )
    optimizer = nnx.Optimizer(model, tx)
    
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints/bc_v2'))
    mngr = ocp.CheckpointManager(ckpt_dir)
    latest_step = mngr.latest_step()
    
    if latest_step is None:
        print("ERROR: No checkpoints found in directory.")
        return
        
    print(f"Restoring from step {latest_step}...")
    try:
        restored = mngr.restore(
            latest_step, 
            args=ocp.args.StandardRestore({'model': nnx.state(model), 'opt': nnx.state(optimizer)})
        )
        nnx.update(model, restored['model'])
    except Exception as e:
        print(f"Restore failed: {e}")
        return
        
    _, state = nnx.split(model)
    
    def force_to_pure_dict(node):
        if hasattr(node, 'items'): return {k: force_to_pure_dict(v) for k, v in node.items()}
        elif hasattr(node, 'value'): return force_to_pure_dict(node.value)
        elif isinstance(node, (list, tuple)): return type(node)(force_to_pure_dict(v) for v in node)
        return node
        
    state_val = force_to_pure_dict(state)
    
    bin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints/benchmarks'))
    os.makedirs(bin_dir, exist_ok=True)
    
    out_path = os.path.join(bin_dir, "bc_v2.bin")
    raw_bytes = serialization.to_bytes(state_val)
    
    with open(out_path, "wb") as f:
        f.write(raw_bytes)
        
    print(f"Successfully exported bc_v2 to {out_path} ({len(raw_bytes)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    main()
