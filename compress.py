import os
import sys
import tarfile
import argparse
import jax
import jax.numpy as jnp
from flax import nnx
from flax import serialization
import orbax.checkpoint as ocp

sys.path.insert(0, os.path.abspath("src"))
from models.entity_transformer_flax_small import EntityTransformer

def compress_model(checkpoint_dir, step, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_bin = os.path.join(output_dir, "optimized_model.bin")
    output_tar = os.path.join(output_dir, "submission.tar.gz")
    
    print(f"--- 1. Compressing Orbax Checkpoint ({checkpoint_dir}) ---")
    
    rngs = nnx.Rngs(42)
    model = EntityTransformer(num_features=14, num_classes=5, rngs=rngs)
    
    mngr = ocp.CheckpointManager(os.path.abspath(checkpoint_dir))
    
    if step == "latest":
        step = mngr.latest_step()
        if step is None:
            print("ERROR: No checkpoints found!")
            sys.exit(1)
            
    print(f"Loading Checkpoint Step {step} via Orbax...")
    raw = mngr.restore(step)
    
    if isinstance(raw, dict) and '0' in raw:
        print("Detected BC Checkpoint (Orbax tuple format). Extracting model from '0'...")
        populated_state = raw['0']
    elif isinstance(raw, dict) and 'model' in raw:
        print("Detected PPO Checkpoint (Dict format). Extracting model from 'model'...")
        populated_state = raw['model']
    else:
        print("Detected raw model checkpoint.")
        populated_state = raw

    _, state_template = nnx.split(model) 

    print("Applying mixed-precision downcasting (bfloat16)...")
    def cast_to_bfloat16(node):
        if hasattr(node, 'items'):
            return {k: cast_to_bfloat16(v) for k, v in node.items()}
        elif isinstance(node, (list, tuple)):
            return type(node)(cast_to_bfloat16(v) for v in node)
        
        # Orbax might wrap arrays in dicts or objects, try to extract the array
        val = getattr(node, 'value', node)
        if hasattr(val, 'dtype') and jnp.issubdtype(val.dtype, jnp.floating):
            return jnp.asarray(val, dtype=jnp.bfloat16)
        return val
    
    opt_state = cast_to_bfloat16(populated_state)

    def force_to_pure_dict(node):
        if hasattr(node, 'items'):
            return {k: force_to_pure_dict(v) for k, v in node.items()}
        elif hasattr(node, 'value'):
            return force_to_pure_dict(node.value)
        elif isinstance(node, (list, tuple)):
            return type(node)(force_to_pure_dict(v) for v in node)
        return node
        
    pure_dict = force_to_pure_dict(opt_state)

    print(f"Serializing to {output_bin}...")
    with open(output_bin, "wb") as f:
        f.write(serialization.to_bytes(pure_dict))
        
    print(f"Success! Optimized binary size: {os.path.getsize(output_bin) / (1024*1024):.2f} MB")
    
    print("\n--- 2. Packaging Kaggle Submission ---")
    import mctx
    mctx_path = os.path.dirname(mctx.__file__)
    
    files_to_pack = {
        "src/submission/agent.py": "main.py",
        "src/models/entity_transformer_flax_small.py": "src/models/entity_transformer_flax_small.py",
        "src/env_jax/orbit_env.py": "env_jax/orbit_env.py",
        output_bin: "optimized_model.bin",
        mctx_path: "mctx"
    }
    
    with tarfile.open(output_tar, "w:gz") as tar:
        for local_path, archive_path in files_to_pack.items():
            if os.path.exists(local_path):
                print(f"  [+] Added: {local_path} -> {archive_path}")
                tar.add(local_path, arcname=archive_path)
            else:
                print(f"  [!] ERROR: Missing {local_path}")
                sys.exit(1)
                
    print(f"\nDone! Upload {output_tar} to Kaggle.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/ppo_light", help="Directory of checkpoints")
    parser.add_argument("--step", type=str, default="latest", help="Step to compress, or 'latest'")
    parser.add_argument("--out_dir", type=str, default="submissions/v1", help="Output directory for the submission files")
    args = parser.parse_args()
    
    step = args.step if args.step == "latest" else int(args.step)
    compress_model(args.ckpt_dir, step, args.out_dir)
