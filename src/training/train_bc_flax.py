import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
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

# Resolution hook to guarantee parental dependencies can be located
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. EXPLICITLY IMPORT THE SMALL ARCHITECTURE
from models.entity_transformer_flax_small import EntityTransformer
from data_pipeline.dataset_grain import build_grain_dataloader

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
        
        v_logits, launch_logits, angle_logits, ships_logits, _ = m(
            batch_state, 
            return_policy=True,
            target_launch=batch_launch,
            target_angle=batch_angle
        )
        
        # 1. Base Mask: Only learn from planets the winner actually owned
        p1_mask = (batch_owned == 1.0).astype(jnp.float32)

        # 2. Value Loss
        safe_winner = jnp.maximum(batch_winner, 0)
        v_loss = optax.softmax_cross_entropy_with_integer_labels(v_logits, safe_winner).mean()
        
        # 3. Safe Cross Entropy
        safe_angle = jnp.maximum(batch_angle, 0)
        safe_ships = jnp.maximum(batch_ships, 0)
        
        launch_bce = optax.sigmoid_binary_cross_entropy(logits=launch_logits, labels=batch_launch)
        angle_ce = optax.softmax_cross_entropy_with_integer_labels(angle_logits, safe_angle)
        ships_ce = optax.softmax_cross_entropy_with_integer_labels(ships_logits, safe_ships)
        
        # --- THE CIRCUIT Breaker: GRADIENT LEAKAGE PROTECTION ---
        # Extract the specific logit for the target chosen by the offline dataset.
        # If it is smaller than -1e8, it means the Transformer masked it.
        chosen_angle_logits = jnp.take_along_axis(angle_logits, safe_angle[..., None], axis=-1)[..., 0]
        is_target_valid = (chosen_angle_logits > -1e8).astype(jnp.float32)
        
        # 4. Strict Valid Target Masks
        pos_launch_mask = p1_mask * (batch_launch == 1.0)
        neg_launch_mask = p1_mask * (batch_launch == 0.0)
        
        # Combine the masks: It must be a launch, have a valid dataset target, AND be allowed by the Transformer.
        valid_target_mask = pos_launch_mask * (batch_angle >= 0) * (batch_ships >= 0) * is_target_valid
        
        # 5. Component Losses (Safely Averaged independently)
        pos_launch_loss = jnp.sum(launch_bce * pos_launch_mask) / jnp.maximum(jnp.sum(pos_launch_mask), 1e-8)
        neg_launch_loss = jnp.sum(launch_bce * neg_launch_mask) / jnp.maximum(jnp.sum(neg_launch_mask), 1e-8)
        balanced_launch_loss = (pos_launch_loss + neg_launch_loss) * 0.5
        
        angle_loss = jnp.sum(angle_ce * valid_target_mask) / jnp.maximum(jnp.sum(valid_target_mask), 1e-8)
        ships_loss = jnp.sum(ships_ce * valid_target_mask) / jnp.maximum(jnp.sum(valid_target_mask), 1e-8)
        
        # Unified Policy Loss
        total_policy_loss = balanced_launch_loss + angle_loss + ships_loss
        total_loss = v_loss + total_policy_loss
        
        # 6. Accuracies (For diagnostics)
        launch_preds = (jax.nn.sigmoid(launch_logits) > 0.5).astype(jnp.float32)
        launch_acc = jnp.sum((launch_preds == batch_launch) * p1_mask) / jnp.maximum(jnp.sum(p1_mask), 1.0)
        target_acc = masked_accuracy(angle_logits, batch_angle, valid_target_mask)
        ships_acc = masked_accuracy(ships_logits, batch_ships, valid_target_mask)

        metrics = {
            "loss": total_loss,
            "v_loss": v_loss,
            "launch_loss": balanced_launch_loss,
            "target_loss": angle_loss,
            "ships_loss": ships_loss,
            "launch_acc": launch_acc,
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
    TOTAL_STEPS = 60000 
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=1e-3, 
        warmup_steps=1500,
        decay_steps=TOTAL_STEPS,
        end_value=1e-5
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)
    )
    optimizer = nnx.Optimizer(model, tx)
    
    # Save to a dedicated BC directory
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../checkpoints/bc_light'))
    mngr = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = os.path.join(base_dir, "data", "parquet_db_real")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, "parquet_db_real")
        
    print(f"Streaming Grandmaster records from: {dataset_path}")
    BATCH_SIZE = 8192
    dl = build_grain_dataloader(
        db_path=dataset_path, 
        batch_size=8192, 
        worker_count=16, 
        grandmaster_only=True, 
        is_iql_mode=True,
        num_epochs=1
    )
    
    print("Pre-baking entire dataset into RAM (This happens once)...")
    full_dataset_list = []
    for batch in tqdm(dl, desc="Extracting to RAM"):
        full_dataset_list.append(batch)
        
    print("Concatenating into monolithic dataset...")
    import numpy as np
    full_dataset_np = {
        key: np.concatenate([b[key] for b in full_dataset_list], axis=0)
        for key in full_dataset_list[0].keys()
    }
    
    del full_dataset_list
    import gc
    gc.collect()
    
    dataset_size = full_dataset_np['state_tokens'].shape[0]
    print(f"Dataset Shape: {full_dataset_np['state_tokens'].shape}")
    print("Uploading massive dataset directly to JAX VRAM...")
    full_dataset = jax.device_put(full_dataset_np)
    del full_dataset_np
    gc.collect()
    print("VRAM Cache established! Zero CPU overhead from now on.")

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
                "L_Acc": f"{float(metrics['launch_acc']):.2%}",
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