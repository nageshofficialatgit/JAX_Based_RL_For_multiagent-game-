import os
import sys
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from models.entity_transformer_flax import EntityTransformer

def perform_surgery():
    print("Initializing old model (classes=4)...")
    # Must use num_classes=4 because that's what's in the checkpoint shape
    model_old = EntityTransformer(num_classes=4, rngs=nnx.Rngs(0))
    _, state_old = nnx.split(model_old)
    
    ckpt_dir = os.path.abspath("checkpoints/flax")
    mngr = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=3, create=False))
    
    print("Loading Step 41000...")
    state_old = mngr.restore(41000, args=ocp.args.StandardRestore(state_old))
    
    print("Initializing new model (classes=5)...")
    # The default is now 5
    model_new = EntityTransformer(rngs=nnx.Rngs(1))
    graph_new, state_new = nnx.split(model_new)
    
    print("Performing Neural Surgery...")
    
    # Recursive function to copy and pad weights
    def copy_weights(dict_new, dict_old, path="root"):
        if hasattr(dict_new, 'items'):
            for k, v in dict_new.items():
                copy_weights(v, dict_old[k], f"{path}/{k}")
        elif hasattr(dict_new, 'value'):
            v_new = dict_new.value
            v_old = dict_old.value
            
            if v_new.shape == v_old.shape:
                dict_new.value = v_old
            else:
                print(f"Shape mismatch at {path}: {v_old.shape} -> {v_new.shape}")
                # We expect value_head/kernel [256, 4] -> [256, 5]
                # and value_head/bias [4] -> [5]
                pad_width = [(0, 0) for _ in range(v_old.ndim)]
                # Pad the LAST dimension by 1 (which is the class dimension)
                pad_width[-1] = (0, v_new.shape[-1] - v_old.shape[-1])
                padded = jnp.pad(v_old, pad_width)
                dict_new.value = padded
                
    copy_weights(state_new, state_old)
    
    print("Surgery complete! Saving to 41001...")
    mngr_save = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True))
    mngr_save.save(41001, args=ocp.args.StandardSave(state_new))
    mngr_save.wait_until_finished()
    
    print("Saved 41001! Architecture successfully upgraded to 4-Player support.")

if __name__ == "__main__":
    perform_surgery()
