import os
import sys
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.entity_transformer_flax import EntityTransformer
from data_pipeline.dataset_grain import build_grain_dataloader

def evaluate():
    print("Initializing Flax NNX Model...")
    rngs = nnx.Rngs(42)
    model = EntityTransformer(d_model=512, n_heads=8, n_layers=12, rngs=rngs)
    
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../checkpoints/flax'))
    print(f"Loading Orbax Checkpoint from {ckpt_dir} (Step 12000)...")
    
    mngr = ocp.CheckpointManager(ckpt_dir)
    
    # Check if 12000 exists, otherwise use the latest
    step = 12000
    if step not in mngr.all_steps():
        if len(mngr.all_steps()) > 0:
            step = mngr.latest_step()
            print(f"Step 12000 not found, using latest step: {step}")
        else:
            print("No checkpoints found!")
            return
            
    # Smart Restoring (Handles BC Tuples and PPO Models)
    raw = mngr.restore(step)
    if isinstance(raw, dict) and '0' in raw and '1' in raw:
        print("Detected BC Checkpoint (Model + Optimizer). Unpacking model...")
        dummy_tx = optax.adamw(learning_rate=3e-4)
        dummy_opt = nnx.Optimizer(model, dummy_tx)
        template = nnx.state((model, dummy_opt))
        restored = mngr.restore(step, args=ocp.args.StandardRestore(template))
        nnx.update(model, restored[0])
    else:
        print("Detected PPO Checkpoint (Model only).")
        _, state = nnx.split(model)
        restore_args = ocp.args.StandardRestore(state)
        restored = mngr.restore(step, args=restore_args)
        nnx.update(model, restored)
    
    print("Model loaded successfully.")
    
    # Load Dataloader
    print("Initializing Grain DataLoader for Validation...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = os.path.join(base_dir, "data", "parquet_db_real")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, "parquet_db_real")
    print(f"Using dataset path: {dataset_path}")
    
    dl = build_grain_dataloader(
        db_path=dataset_path, 
        batch_size=32, 
        worker_count=0,
        grandmaster_only=False,
        subsample_ticks=5
    )
    
    dl_iter = iter(dl)
    batch = next(dl_iter)
    print("Validation Batch loaded.")
    
    state_tokens = jnp.array(batch['state_tokens'])
    target_launch = jnp.array(batch['target_launch'])
    target_angle = jnp.array(batch['target_angle'])
    target_ships = jnp.array(batch['target_ships'])
    
    print("Running Inference...")
    v_logits, launch_logits, angle_logits, ships_logits, ppo_value = model(
        state_tokens, 
        return_policy=True,
        target_launch=target_launch,
        target_angle=target_angle
    )
    
    pred_launch = (launch_logits > 0).astype(jnp.int32)
    true_launch = target_launch.astype(jnp.int32)
    
    valid_mask = batch['is_owned_by_winner'] == 1.0
    
    correct_launches = jnp.sum((pred_launch == true_launch) * valid_mask)
    total_launches = jnp.sum(valid_mask)
    launch_acc = correct_launches / jnp.maximum(total_launches, 1)
    
    print("\n--- Accuracy Metrics (Offline Evaluation) ---")
    print(f"Total Valid Owned Planets in Batch: {total_launches}")
    print(f"Launch Prediction Accuracy: {launch_acc * 100:.2f}%")
    
    angle_mask = true_launch == 1
    if jnp.sum(angle_mask) > 0:
        pred_angle = jnp.argmax(angle_logits, axis=-1)
        correct_angle = jnp.sum((pred_angle == target_angle) * angle_mask)
        angle_acc = correct_angle / jnp.sum(angle_mask)
        print(f"Target Planet Prediction Accuracy (on active launches): {angle_acc * 100:.2f}%")
        
        pred_ships = jnp.argmax(ships_logits, axis=-1)
        correct_ships = jnp.sum((pred_ships == target_ships) * angle_mask)
        ships_acc = correct_ships / jnp.sum(angle_mask)
        print(f"Ships Allocation Accuracy: {ships_acc * 100:.2f}%")
        
        idx = jnp.argmax(angle_mask) 
        flat_idx = jnp.unravel_index(idx, angle_mask.shape)
        b, p = flat_idx
        
        print("\n--- Concrete Prediction Example ---")
        print(f"Planet ID: {p}")
        print(f"True Action: Launch={true_launch[b, p]}, Target={target_angle[b, p]}, Ships={target_ships[b, p]}")
        print(f"Pred Action: Launch={pred_launch[b, p]}, Target={pred_angle[b, p]}, Ships={pred_ships[b, p]}")
    else:
        print("No active launches in this validation batch to evaluate angle/ships.")

if __name__ == "__main__":
    evaluate()