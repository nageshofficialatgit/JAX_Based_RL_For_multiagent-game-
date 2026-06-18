import os
import sys
import math
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from models.entity_transformer_flax import EntityTransformer
from data_pipeline.dataset_grain import OrbitWarsDataSource

def evaluate():
    print("Loading Dataset...", flush=True)
    base_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(base_dir, "data", "parquet_db_real")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, "parquet_db_real")
    print(f"Using dataset path: {dataset_path}")
    ds = OrbitWarsDataSource(dataset_path, max_action_history=20, grandmaster_only=False)
    
    print("Finding a complete episode in the dataset...", flush=True)
    target_ep = ds.state_index[50000][0]
    
    indices = []
    for i in range(40000, 60000):
        if ds.state_index[i][0] == target_ep:
            indices.append(i)
            
    # Sort by tick
    indices.sort(key=lambda i: ds.state_index[i][1])
    
    print(f"Selected Episode {target_ep} with {len(indices)} frames.", flush=True)
    
    print("Loading Model...", flush=True)
    rngs = nnx.Rngs(42)
    model = EntityTransformer(d_model=512, n_heads=8, n_layers=12, rngs=rngs)
    graphdef, state = nnx.split(model)
    
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints/flax'))
    mngr = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=3, create=False))
    
    step = 41001
    if not mngr.item_metadata(step):
        step = mngr.latest_step()
    
    print(f"Restoring from step {step}...", flush=True)
    state = mngr.restore(step, args=ocp.args.StandardRestore(state))
    merged_model = nnx.merge(graphdef, state)
    
    for target_ep in [ds.state_index[100000][0], ds.state_index[150000][0]]:
        indices = [i for i in range(len(ds.state_index)) if ds.state_index[i][0] == target_ep]
        indices.sort(key=lambda i: ds.state_index[i][1])
        actual_winner = ds[indices[0]]['winner']
        
        print(f"\n--- Episode {target_ep} | Winner: Player {actual_winner} ---")
        for i in indices[::10]:
            b = ds[i]
            x = jnp.array(b["state_tokens"][None, :, :])
            v_logits, _, _, _ = merged_model(x, return_policy=True)
            p1_prob = float(jax.nn.sigmoid(v_logits[0, 0]))
            print(f"Tick {b['tick']:<4}: P1 Win {p1_prob*100:.1f}%")

if __name__ == "__main__":
    evaluate()
