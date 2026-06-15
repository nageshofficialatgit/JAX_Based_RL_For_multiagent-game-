import os
import sys

# --- PATH RESOLUTION ---
# Stop JAX from hogging all VRAM during evaluation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 1. Get the absolute path of the current script (src/training/eval.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go up one level to get the 'src' directory
src_dir = os.path.dirname(current_dir)
# 3. Go up one more level to get the project root ('server_deploy')
project_root = os.path.dirname(src_dir)

# 4. Add both to the system path
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

# --- IMPORTS ---
try:
    from src.models.entity_transformer_flax import EntityTransformer
    from src.data_pipeline.dataset_grain import build_grain_dataloader
except ImportError:
    from models.entity_transformer_flax import EntityTransformer
    from data_pipeline.dataset_grain import build_grain_dataloader

import time
import queue
import threading
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from tqdm import tqdm

# --- CONFIGURATION ---
CHECKPOINT_DIR = os.path.join(project_root, "checkpoints", "flax")
CHECKPOINT_STEP = 41191 
EVAL_BATCHES = 50  # How many batches to evaluate (50 batches of 256 = 12,800 game states)

# --- UTILS ---
def cross_entropy_loss(logits, targets, ignore_index=-1):
    valid_mask = (targets != ignore_index)
    safe_targets = jnp.maximum(targets, 0)
    target_one_hot = jax.nn.one_hot(safe_targets, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits=logits, labels=target_one_hot)
    loss = loss * valid_mask
    num_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)
    return jnp.sum(loss) / num_valid

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

# --- METRIC COMPUTATION (JIT) ---
@nnx.jit
def eval_step(model, batch):
    v_logits, launch_logits, angle_logits, ships_logits, _ = model(
        batch['state_tokens'], 
        return_policy=True,
        target_launch=batch['target_launch'],
        target_angle=batch['target_angle']
    )
    
    # 1. Calculate Losses
    v_loss = cross_entropy_loss(v_logits, batch['winner'])
    launch_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=launch_logits, labels=batch['target_launch']))
    angle_loss = cross_entropy_loss(angle_logits, batch['target_angle'])
    ships_loss = cross_entropy_loss(ships_logits, batch['target_ships'])
    total_loss = v_loss + launch_loss + angle_loss + ships_loss

    # 2. Calculate Accuracies
    pred_launch = (launch_logits > 0).astype(jnp.int32)
    true_launch = batch['target_launch'].astype(jnp.int32)
    
    # Only evaluate behavior for planets owned by the eventual winner
    valid_mask = batch['is_owned_by_winner'] == 1.0
    
    correct_launches = jnp.sum((pred_launch == true_launch) * valid_mask)
    total_launches_eval = jnp.sum(valid_mask)
    
    # For angle and ships, only evaluate when the human actually launched a ship
    angle_mask = (true_launch == 1) & valid_mask
    
    pred_angle = jnp.argmax(angle_logits, axis=-1)
    correct_angle = jnp.sum((pred_angle == batch['target_angle']) * angle_mask)
    total_angles_eval = jnp.sum(angle_mask)
    
    pred_ships = jnp.argmax(ships_logits, axis=-1)
    correct_ships = jnp.sum((pred_ships == batch['target_ships']) * angle_mask)

    return {
        "total_loss": total_loss,
        "v_loss": v_loss,
        "launch_loss": launch_loss,
        "angle_loss": angle_loss,
        "ships_loss": ships_loss,
        "correct_launches": correct_launches,
        "total_launches_eval": total_launches_eval,
        "correct_angle": correct_angle,
        "total_angles_eval": total_angles_eval,
        "correct_ships": correct_ships
    }

def run_evaluation():
    print("=== ORBIT WARS BEHAVIOR CLONING EVALUATOR ===")
    
    # 1. Load Model
    rngs = nnx.Rngs(42)
    model = EntityTransformer(d_model=512, n_heads=8, n_layers=12, rngs=rngs)
    graphdef, state = nnx.split(model)
    
    mngr = ocp.CheckpointManager(CHECKPOINT_DIR)
    step_to_load = CHECKPOINT_STEP if CHECKPOINT_STEP is not None else mngr.latest_step()
    
    if step_to_load is None:
        print("ERROR: No checkpoints found!")
        return

    print(f"Loading Checkpoint Step: {step_to_load}")
    
    # =================================================================
    # EXACT LOADING LOGIC FROM train_ppo_flax.py
    # =================================================================
    raw_restored = mngr.restore(step_to_load)
    
    if isinstance(raw_restored, dict) and '0' in raw_restored and '1' in raw_restored:
        print("Detected BC Checkpoint (Model + Optimizer)...")
        dummy_tx = optax.adamw(learning_rate=3e-4)
        dummy_opt = nnx.Optimizer(model, dummy_tx)
        template = nnx.state((model, dummy_opt))
        
        restored_typed = mngr.restore(step_to_load, args=ocp.args.StandardRestore(template))
        restored_casted = jax.tree_util.tree_map(
            lambda r, t: jnp.asarray(r, dtype=t.value.dtype) if hasattr(t, 'value') else r,
            restored_typed['0'], template['0']
        )
        nnx.update(model, restored_casted)
    else:
        print("Detected PPO Checkpoint (Model only)...")
        restored_typed = mngr.restore(step_to_load, args=ocp.args.StandardRestore(state))
        restored_casted = jax.tree_util.tree_map(
            lambda r, t: jnp.asarray(r, dtype=t.value.dtype) if hasattr(t, 'value') else r,
            restored_typed, state
        )
        nnx.update(model, restored_casted)
        
    print("Model successfully loaded into VRAM.")

    # 2. Load Dataset
    dataset_path = os.path.join(project_root, "data", "parquet_db_real")
    if not os.path.exists(dataset_path):
        # Fallback to older path structure if necessary
        dataset_path = os.path.join(project_root, "parquet_db_real")
    
    print(f"Loading Dataset from: {dataset_path}")
    dl = build_grain_dataloader(
        db_path=dataset_path,
        batch_size=256, 
        worker_count=4,
        grandmaster_only=False,
        subsample_ticks=1
    )

    # 3. Evaluation Loop
    metrics_sum = {
        "total_loss": 0.0, "v_loss": 0.0, "launch_loss": 0.0, 
        "angle_loss": 0.0, "ships_loss": 0.0,
        "correct_launches": 0.0, "total_launches_eval": 0.0,
        "correct_angle": 0.0, "total_angles_eval": 0.0,
        "correct_ships": 0.0
    }
    
    print(f"\nEvaluating over {EVAL_BATCHES} batches...")
    pbar = tqdm(async_prefetcher(dl), total=EVAL_BATCHES)
    
    batches_processed = 0
    for batch in pbar:
        if batches_processed >= EVAL_BATCHES:
            break
            
        step_metrics = eval_step(model, batch)
        
        for k in metrics_sum.keys():
            metrics_sum[k] += float(step_metrics[k])
            
        batches_processed += 1
        pbar.set_postfix({"V_Loss": f"{float(step_metrics['v_loss']):.3f}"})

    # 4. Final Report
    avg = {k: v / batches_processed for k, v in metrics_sum.items() if 'loss' in k}
    
    launch_acc = (metrics_sum["correct_launches"] / max(metrics_sum["total_launches_eval"], 1)) * 100
    angle_acc = (metrics_sum["correct_angle"] / max(metrics_sum["total_angles_eval"], 1)) * 100
    ships_acc = (metrics_sum["correct_ships"] / max(metrics_sum["total_angles_eval"], 1)) * 100

    print("\n" + "="*50)
    print(" 📊 EVALUATION REPORT")
    print("="*50)
    print(f"Checkpoint Step : {step_to_load}")
    print(f"States Evaluated: {batches_processed * 256:,.0f}")
    print("-"*50)
    print("--- LOSS METRICS ---")
    print(f"Value Loss      : {avg['v_loss']:.4f}")
    print(f"Launch Loss     : {avg['launch_loss']:.4f}")
    print(f"Angle Loss      : {avg['angle_loss']:.4f}")
    print(f"Ships Loss      : {avg['ships_loss']:.4f}")
    print(f"Total Combined  : {avg['total_loss']:.4f}")
    print("-"*50)
    print("--- BEHAVIORAL ACCURACY (vs Human) ---")
    print(f"Launch Decision : {launch_acc:.2f}%")
    print(f"Target Selection: {angle_acc:.2f}%  (When launching)")
    print(f"Ship Allocation : {ships_acc:.2f}%  (When launching)")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()