import os
import sys
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from models.entity_transformer_flax_v2 import EntityTransformer

def main():
    print("Initializing EntityTransformer (v2)...")
    rngs = nnx.Rngs(42)
    model = EntityTransformer(rngs=rngs)
    
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints/bc_v2'))
    if not os.path.exists(ckpt_dir):
        print(f"ERROR: Checkpoint directory not found at {ckpt_dir}")
        return
        
    mngr = ocp.CheckpointManager(ckpt_dir)
    latest_step = mngr.latest_step()
    
    if latest_step is None:
        print("ERROR: No checkpoints found in directory.")
        return
        
    print(f"Loading checkpoint from step {latest_step}...")
    try:
        # Load the raw checkpoint dictionary
        ckpt = mngr.restore(latest_step)
        
        if 'model' in ckpt:
            nnx.update(model, ckpt['model'])
        else:
            nnx.update(model, ckpt)
            
    except Exception as e:
        print(f"Restore failed: {e}")
        return
        
    print("Checkpoint loaded successfully!")
    
    print("Testing forward pass...")
    # Dummy batch of 4 games, 50 planets, 17 features
    dummy_obs = jnp.ones((4, 50, 17))
    dummy_mask = jnp.ones((4, 50), dtype=bool)
    
    try:
        v_logits, launch, angle, ships, ppo_val = model(dummy_obs, return_policy=True, valid_launch_mask=dummy_mask)
        print("Forward pass successful!")
        print(f"  Launch Logits: {launch.shape}")
        print(f"  Angle Logits:  {angle.shape}")
        print(f"  Ships Logits:  {ships.shape}")
        
        # Check for NaNs
        if jnp.isnan(launch).any():
            print("WARNING: NaNs detected in launch logits!")
        else:
            print("No NaNs detected in output.")
            
    except Exception as e:
        print(f"ERROR during forward pass: {e}")

if __name__ == "__main__":
    main()
