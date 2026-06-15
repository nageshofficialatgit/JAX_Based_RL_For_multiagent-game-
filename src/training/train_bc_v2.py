import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import multiprocessing as mp

# FORCE SPAWN: Prevents the Linux OOM Killer from duplicating RAM across workers
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from tqdm import tqdm
import threading
import queue
import gc

# Resolution hook to guarantee parental dependencies can be located
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. EXPLICITLY IMPORT THE V2 ARCHITECTURE
from models.entity_transformer_flax_v2 import EntityTransformer
from data_pipeline.dataset_grain_v2 import build_grain_dataloader

def masked_accuracy(logits, targets, mask):
    """Calculates accuracy only on valid tokens defined by the mask."""
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == targets) * mask
    return jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1.0)

@nnx.jit
def train_step(model, optimizer, batch_indices, dataset):
    def loss_fn(m):
        batch_state = dataset['state_tokens'][batch_indices]
        batch_launch = dataset['target_launch'][batch_indices]
        batch_angle = dataset['target_angle'][batch_indices]
        batch_ships = dataset['target_ships'][batch_indices]
        batch_winner = dataset['winner'][batch_indices]
        batch_owned = dataset['is_owned_by_winner'][batch_indices]
        
        # EXTRACT ELO SEPARATELY (Decision Transformer Anti-Pattern fix)
        batch_win_rate = dataset['win_rate'][batch_indices]
        
        # Pass target_ships to enable proper causal teacher-forcing
        v_logits, launch_logits, ships_logits, angle_logits, ppo_value = m(
            batch_state, 
            return_policy=True,
            target_launch=batch_launch,
            target_ships=batch_ships,
            target_angle=batch_angle
        )
        
        # 1. Base Mask: Only learn from planets the winner actually owned
        p1_mask = (batch_owned == 1.0).astype(jnp.float32)

        # 2. Value Loss
        safe_winner = jnp.maximum(batch_winner, 0)
        v_loss = optax.softmax_cross_entropy_with_integer_labels(v_logits, safe_winner).mean()
        
        # Pre-train PPO Value Head (MSE against win_rate) to prevent catastrophic forgetting
        ppo_v_loss = optax.l2_loss(ppo_value, batch_win_rate).mean()
        
        # 3. Safe Cross Entropy
        safe_angle = jnp.maximum(batch_angle, 0)
        safe_ships = jnp.maximum(batch_ships, 0)
        
        launch_bce = optax.sigmoid_binary_cross_entropy(logits=launch_logits, labels=batch_launch)
        angle_ce = optax.softmax_cross_entropy_with_integer_labels(angle_logits, safe_angle)
        ships_ce = optax.softmax_cross_entropy_with_integer_labels(ships_logits, safe_ships)
        
        # --- THE CIRCUIT Breaker: GRADIENT LEAKAGE PROTECTION ---
        chosen_angle_logits = jnp.take_along_axis(angle_logits, safe_angle[..., None], axis=-1)[..., 0]
        is_target_valid = (chosen_angle_logits > -1e8).astype(jnp.float32)
        
        # 4. Strict Valid Target Masks
        pos_launch_mask = p1_mask * (batch_launch == 1.0)
        neg_launch_mask = p1_mask * (batch_launch == 0.0)
        
        valid_target_mask = pos_launch_mask * (batch_angle >= 0) * (batch_ships >= 0) * is_target_valid
        
        # 5. Component Losses
        launch_loss = jnp.sum(launch_bce * p1_mask) / jnp.maximum(jnp.sum(p1_mask), 1e-8)
        angle_loss = jnp.sum(angle_ce * valid_target_mask) / jnp.maximum(jnp.sum(valid_target_mask), 1e-8)
        ships_loss = jnp.sum(ships_ce * valid_target_mask) / jnp.maximum(jnp.sum(valid_target_mask), 1e-8)
        
        # --- ELO-BASED CONTRASTIVE WEIGHTING (Offline RL style) ---
        elo_weight = jnp.maximum(batch_win_rate ** 2, 0.01)
        
        # Unified Policy Loss (Weighted by ELO)
        total_policy_loss = jnp.mean(elo_weight * (launch_loss + angle_loss + ships_loss))
        total_loss = v_loss + ppo_v_loss + total_policy_loss
        
        # 6. Accuracies (For diagnostics)
        launch_preds = (jax.nn.sigmoid(launch_logits) > 0.5).astype(jnp.float32)
        
        # Calculate True Positive Rate (TPR) and True Negative Rate (TNR)
        launch_tpr = jnp.sum((launch_preds == 1.0) * pos_launch_mask) / jnp.maximum(jnp.sum(pos_launch_mask), 1.0)
        launch_tnr = jnp.sum((launch_preds == 0.0) * neg_launch_mask) / jnp.maximum(jnp.sum(neg_launch_mask), 1.0)
        
        target_acc = masked_accuracy(angle_logits, batch_angle, valid_target_mask)
        ships_acc = masked_accuracy(ships_logits, batch_ships, valid_target_mask)

        metrics = {
            "loss": total_loss,
            "v_loss": v_loss,
            "ppo_v_loss": ppo_v_loss,
            "launch_loss": launch_loss,
            "target_loss": angle_loss,
            "ships_loss": ships_loss,
            "launch_tpr": launch_tpr,
            "launch_tnr": launch_tnr,
            "target_acc": target_acc,
            "ships_acc": ships_acc
        }
        return total_loss, metrics

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(model)
    optimizer.update(grads)
    return metrics

def train_bc():
    print(f"JAX Devices Online: {jax.devices()}")
    print("Initializing Lightweight Behavior Cloning Architecture...")
    rngs = nnx.Rngs(42)
    
    # Model inherently loads the smaller defaults defined in entity_transformer_flax_small.py
    model = EntityTransformer(rngs=rngs)
    
    # Cosine Annealing Schedule for rapid, stable convergence
    TOTAL_STEPS = 120000 
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=1e-3, 
        warmup_steps=1500,
        decay_steps=TOTAL_STEPS - 1500,
        end_value=1e-5
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)
    )
    optimizer = nnx.Optimizer(model, tx)
    
    # Save to a dedicated BC directory
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../checkpoints/bc_v2'))
    mngr = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = os.path.join(base_dir, "data", "parquet_db_real")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, "parquet_db_real")
        
    BATCH_SIZE = 8192
    import numpy as np
    import gc

    dataset_cache_dir = os.path.join(dataset_path, "bc_v2_disk_cache")
    os.makedirs(dataset_cache_dir, exist_ok=True)
    flag_file = os.path.join(dataset_cache_dir, "CACHE_READY.flag")

    if os.path.exists(flag_file):
        print(f"Loading dataset directly from Disk Memmap to VRAM...")
        keys = [f.split('.')[0] for f in os.listdir(dataset_cache_dir) if f.endswith('.npy')]
        full_dataset_np = {k: np.load(os.path.join(dataset_cache_dir, f"{k}.npy"), mmap_mode='r') for k in keys}
        print(f"  -> Connected to disk. Shape: {full_dataset_np['state_tokens'].shape}")
    else:
        print(f"Streaming {dataset_path} directly to Hard Disk (Zero RAM Spike)...")
        workers = 8  # 8 workers are fully supported now via mmap and local disk caching
        dl = build_grain_dataloader(
            db_path=dataset_path,
            batch_size=BATCH_SIZE,
            worker_count=workers,
            grandmaster_only=False,
            is_iql_mode=True,
            num_epochs=1
        )

        tmp_dir = os.path.join(dataset_cache_dir, "tmp_chunks")
        os.makedirs(tmp_dir, exist_ok=True)

        batch_files = []
        total_rows = 0
        first_batch_metadata = None

        # Phase 1: Spool chunks one-by-one to disk — RAM stays flat
        for i, batch in enumerate(tqdm(dl, desc="Spooling to Disk")):
            if first_batch_metadata is None:
                first_batch_metadata = {k: (v.shape, v.dtype) for k, v in batch.items()}
            b_path = os.path.join(tmp_dir, f"chunk_{i}.npz")
            np.savez(b_path, **batch)
            batch_files.append(b_path)
            total_rows += batch["state_tokens"].shape[0]

        print(f"Pre-allocating contiguous disk space for {total_rows} rows...")
        memmaps = {}
        for k, (shape, dtype) in first_batch_metadata.items():
            full_shape = (total_rows,) + shape[1:]
            mmap_path = os.path.join(dataset_cache_dir, f"{k}.npy")
            memmaps[k] = np.lib.format.open_memmap(mmap_path, mode='w+', shape=full_shape, dtype=dtype)

        print("Consolidating chunks into final disk arrays...")
        current_idx = 0
        for b_path in tqdm(batch_files, desc="Consolidating"):
            with np.load(b_path) as b:
                b_size = b["state_tokens"].shape[0]
                for k in memmaps.keys():
                    memmaps[k][current_idx:current_idx + b_size] = b[k]
            current_idx += b_size
            os.remove(b_path)

        os.rmdir(tmp_dir)
        for k in memmaps.keys():
            memmaps[k].flush()

        with open(flag_file, 'w') as f:
            f.write("READY")

        print("Consolidation complete. Connecting to new disk cache...")
        full_dataset_np = {k: np.load(os.path.join(dataset_cache_dir, f"{k}.npy"), mmap_mode='r') for k in memmaps.keys()}

    print("Pushing direct DMA Transfer: Disk -> GPU VRAM...")
    full_dataset = jax.device_put(full_dataset_np)
    del full_dataset_np
    gc.collect()

    dataset_size = full_dataset['state_tokens'].shape[0]
    print(f"VRAM Cache ready! {dataset_size} samples. Zero CPU RAM overhead from now on.")



    t0 = time.time()
    
    # RESTORE LOGIC
    latest_step = mngr.latest_step()
    if latest_step is not None:
        print(f"Restoring from checkpoint at step {latest_step}...")
        
        try:
            # Modern Orbax syntax
            restored = mngr.restore(
                latest_step, 
                args=ocp.args.StandardRestore({'model': nnx.state(model), 'opt': nnx.state(optimizer)})
            )
        except Exception:
            # Fallback for older Orbax versions
            restored = mngr.restore(latest_step, items={'model': nnx.state(model), 'opt': nnx.state(optimizer)})
            
        nnx.update(model, restored['model'])
        nnx.update(optimizer, restored['opt'])
        start_step = latest_step
    else:
        start_step = 0
        
    print(f"Starting Continuous BC Training for {TOTAL_STEPS} steps...")
    
    # Single, continuous progress bar mapped to your total steps
    pbar = tqdm(range(start_step, TOTAL_STEPS), desc="BC Training")
    
    rng_key = jax.random.PRNGKey(42)
    step = start_step
    
    if start_step >= TOTAL_STEPS:
        print(f"Training already completed up to {TOTAL_STEPS} steps. Exiting.")
        return

    for step in pbar:
        rng_key, subkey = jax.random.split(rng_key)
        batch_indices = jax.random.randint(subkey, (BATCH_SIZE,), 0, dataset_size)
        
        metrics = train_step(model, optimizer, batch_indices, full_dataset)
        
        if step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            sps = (BATCH_SIZE * 10) / dt
            
            pbar.set_postfix({
                "Loss": f"{float(metrics['loss']):.1f}", 
                "LLoss": f"{float(metrics['launch_loss']):.1f}",
                "TLoss": f"{float(metrics['target_loss']):.1f}",
                "SLoss": f"{float(metrics['ships_loss']):.1f}",
                "TPR": f"{float(metrics['launch_tpr']):.2%}",
                "TNR": f"{float(metrics['launch_tnr']):.2%}",
                "T_Acc": f"{float(metrics['target_acc']):.2%}",
                "S_Acc": f"{float(metrics['ships_acc']):.2%}",
                "SPS": f"{sps:.0f}"
            })
            
        # Guaranteed Periodic Checkpoints
        if step > 0 and step % 2500 == 0:
            print(f"\n[INTERMEDIATE] Saving BC checkpoint at step {step}...")
            mngr.save(step, args=ocp.args.StandardSave({'model': nnx.state(model), 'opt': nnx.state(optimizer)}))
            mngr.wait_until_finished()
            
        step += 1
        if step >= TOTAL_STEPS:
            break

    # FINAL END-POINT CHECKPOINT
    print(f"\n[FINAL END-POINT] Saving final BC checkpoint at step {step}...")
    mngr.save(step, args=ocp.args.StandardSave({'model': nnx.state(model), 'opt': nnx.state(optimizer)}))
    mngr.wait_until_finished() 
    print("Behavior Cloning Completely Finished and Secured!")

if __name__ == "__main__":
    train_bc()