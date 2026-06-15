import os
import sys
import time
import os
import sys
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from tqdm import tqdm
import threading
import queue

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.entity_transformer_flax import EntityTransformer
from data_pipeline.dataset_grain import build_grain_dataloader

def cross_entropy_loss(logits, targets, ignore_index=-1):
    valid_mask = (targets != ignore_index)
    safe_targets = jnp.maximum(targets, 0)
    target_one_hot = jax.nn.one_hot(safe_targets, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits=logits, labels=target_one_hot)
    loss = loss * valid_mask
    num_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)
    return jnp.sum(loss) / num_valid

# 1. Use nnx.jit (wraps jax.jit but natively understands NNX mutable state)
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(m):
        v_logits, launch_logits, angle_logits, ships_logits, _ = m(
            batch['state_tokens'], 
            return_policy=True,
            target_launch=batch['target_launch'],
            target_angle=batch['target_angle']
        )
        
        v_loss = cross_entropy_loss(v_logits, batch['winner'])
        launch_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=launch_logits, labels=batch['target_launch']))
        angle_loss = cross_entropy_loss(angle_logits, batch['target_angle'])
        ships_loss = cross_entropy_loss(ships_logits, batch['target_ships'])

        total_loss = v_loss + launch_loss + angle_loss + ships_loss
        return total_loss, {"loss": total_loss, "v_loss": v_loss}

    # 2. Native NNX grad calculation
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(model)
    
    # 3. Native NNX optimizer step (AdamW applied seamlessly)
    optimizer.update(grads)
    
    return metrics

def train():
    print(f"JAX Devices: {jax.devices()}")
    rngs = nnx.Rngs(42)
    model = EntityTransformer(d_model=512, n_heads=8, n_layers=12, rngs=rngs)
    
    # Setup standard Optax AdamW
    tx = optax.adamw(learning_rate=3e-4, weight_decay=1e-4)
    optimizer = nnx.Optimizer(model, tx)
    
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../checkpoints/flax'))
    mngr = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))
    
    step = 0
    # --- HARDCODED CHECKPOINT RESTORATION (WITH PARTIAL RESTORE) ---
    target_step = 41951
    existing_steps = mngr.all_steps()
    
    if target_step in existing_steps:
        print(f"Restoring from checkpoint step {target_step} with partial_restore=True...")
        
        abstract_state = nnx.state((model, optimizer))
        
        # Use partial_restore=True to allow missing keys in the checkpoint
        restore_args = ocp.args.StandardRestore(abstract_state, partial_restore=True)
        restored = mngr.restore(target_step, args=restore_args)
        
        nnx.update((model, optimizer), restored)
        print(f"Successfully loaded matching weights from step {target_step}.")
        print("NOTE: Missing layers (newly added) have been initialized with random weights.")
    else:
        print(f"ERROR: Checkpoint at step {target_step} not found.")
        sys.exit(1)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = os.path.join(base_dir, "data", "parquet_db_real")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, "parquet_db_real")
    print(f"Using dataset path: {dataset_path}")

    dl = build_grain_dataloader(
        db_path=dataset_path,
        batch_size=256, 
        worker_count=4,
        grandmaster_only=False,
        subsample_ticks=1
    )
    
    def async_prefetcher(iterator):
        q = queue.Queue(maxsize=3)
        def worker():
            for b in iterator:
                try:
                    q.put(jax.tree_util.tree_map(jax.device_put, b))
                except Exception as e:
                    print("Prefetch error:", e)
            q.put(None)
        threading.Thread(target=worker, daemon=True).start()
        while True:
            b = q.get()
            if b is None:
                break
            yield b

    t0 = time.time()
    
    for epoch in range(10):
        print(f"--- Epoch {epoch+1}/10 ---")
        pbar = tqdm(async_prefetcher(dl), desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            metrics = train_step(model, optimizer, batch)
            
            if step % 10 == 0:
                dt = time.time() - t0
                t0 = time.time()
                sps = (256 * 10) / dt
                
                loss_val = float(metrics['loss'])
                v_loss = float(metrics['v_loss'])
                pbar.set_postfix({"Loss": f"{loss_val:.3f}", "V_Loss": f"{v_loss:.3f}", "SPS": f"{sps:.0f}"})
                
            if step > 0 and step % 1000 == 0:
                print(f"\nSaving checkpoint at step {step}...")
                mngr.save(step, args=ocp.args.StandardSave(nnx.state((model, optimizer))))
                
            step += 1

    print("Training Complete!")

if __name__ == "__main__":
    train()
