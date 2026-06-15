import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
import sys
import os
import optax

sys.path.append(os.path.abspath("."))
from src.models.entity_transformer_flax_small import EntityTransformer

def main():
    print("Loading checkpoint 1000...")
    model = EntityTransformer(num_features=14, num_classes=5, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)
    
    state_flat = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, state)
    
    ckpt_dir = os.path.abspath("checkpoints/ppo_light")
    mngr = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=5, create=False))
    
    ckpt = mngr.restore(1000)
    
    if "state_flat" in ckpt:
        state_flat = ckpt["state_flat"]
    elif "model" in ckpt:
        state_flat = ckpt["model"]
    elif "weights" in ckpt:
        state_flat = ckpt["weights"]
    else:
        # Pytree format
        state_flat = ckpt
    
    merged = nnx.merge(graphdef, state_flat)
    
    # Create a dummy observation (batch=1, planets=70, features=14)
    obs = jnp.zeros((1, 70, 14), dtype=jnp.float32)
    obs = obs.at[:, 1:51, 5].set(1.0) # Set p1_mask for planets 1-50
    obs = obs.at[:, 1:51, 6].set(1.0) # Set valid_launch_mask for planets 1-50
    
    valid_mask = jnp.ones((1, 50), dtype=bool)
    
    # Run forward pass
    v_logits, launch_logits, angle_logits, ships_logits, ppo_v = merged(
        obs, return_policy=True, target_launch=None, target_angle=None, valid_launch_mask=valid_mask
    )
    
    # Calculate probabilities and entropies for a single planet (e.g. planet 0)
    l_logits = launch_logits[0, 0]
    a_logits = angle_logits[0, 0]
    s_logits = ships_logits[0, 0]
    
    l_prob = jax.nn.sigmoid(l_logits)
    l_ent = -(l_prob * jnp.log(l_prob + 1e-8) + (1.0 - l_prob) * jnp.log(1.0 - l_prob + 1e-8))
    
    a_probs = jax.nn.softmax(a_logits, axis=-1)
    a_ent = -jnp.sum(a_probs * jnp.log(a_probs + 1e-8), axis=-1)
    
    s_probs = jax.nn.softmax(s_logits, axis=-1)
    s_ent = -jnp.sum(s_probs * jnp.log(s_probs + 1e-8), axis=-1)
    
    print(f"\n--- ENTROPY PROOF (Iteration 1000) ---")
    print(f"Launch Probability:   {l_prob * 100:.2f}%")
    print(f"Launch Entropy:       {l_ent:.5f}")
    print(f"Angle Entropy:        {a_ent:.5f}  (Max theoretical for 50 classes: {jnp.log(50):.5f})")
    print(f"Ships Entropy:        {s_ent:.5f}  (Max theoretical for 10 classes: {jnp.log(10):.5f})")
    
    total_ent = l_ent + 0.5 * a_ent + 0.3 * s_ent
    print(f"\nTotal Policy Entropy: {total_ent:.5f}")

if __name__ == "__main__":
    main()
