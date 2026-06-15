import os
import sys
import jax
import jax.numpy as jnp
from flax import nnx, serialization

# Ensure local imports work from the server_deploy directory
sys.path.append(os.path.abspath("."))

from src.models.entity_transformer_flax import EntityTransformer
from src.env_jax.orbit_env import EnvState, build_observation

def normalize_owners_fixed(owner_list, player_id):
    """
    THE FIX: Explicitly forces the Kaggle player_id to be recognized as 1.
    All other owners become 2 (Enemy) or 0 (Neutral).
    """
    if not owner_list: return []
    arr = jnp.array(owner_list)
    # If the planet belongs to us, map it to 1. If neutral (-1 or 0), map to 0. Else 2.
    return jnp.where(arr == player_id, 1, jnp.where((arr == 0) | (arr == -1), 0, 2))

def parse_kaggle_obs_fixed(obs, player_id):
    """Safely extracts Kaggle's dictionary using the fixed ownership mapping."""
    def pad_array(arr, target_len, dtype=jnp.float32):
        arr = jnp.array(arr, dtype=dtype)
        if len(arr) < target_len:
            padding = jnp.zeros((target_len - len(arr),), dtype=dtype)
            return jnp.concatenate([arr, padding])
        return arr[:target_len]

    return EnvState(
        planet_x=pad_array(obs['planets']['x'], 50),
        planet_y=pad_array(obs['planets']['y'], 50),
        planet_radius=pad_array(obs['planets'].get('radius', [1.0]*50), 50), 
        planet_production=pad_array(obs['planets']['production'], 50),
        planet_owner=pad_array(normalize_owners_fixed(obs['planets']['owner'], player_id), 50, dtype=jnp.int32),
        planet_ships=pad_array(obs['planets']['ships'], 50),
        
        fleet_active=pad_array(obs['fleets'].get('active', []), 200, dtype=jnp.int32),
        fleet_owner=pad_array(normalize_owners_fixed(obs['fleets'].get('owner', []), player_id), 200, dtype=jnp.int32),
        fleet_ships=pad_array(obs['fleets'].get('ships', []), 200),
        fleet_x=pad_array(obs['fleets'].get('x', []), 200),
        fleet_y=pad_array(obs['fleets'].get('y', []), 200),
        fleet_dx=pad_array(obs['fleets'].get('dx', []), 200),
        fleet_dy=pad_array(obs['fleets'].get('dy', []), 200),
        fleet_src_planet=pad_array(obs['fleets'].get('src_planet', []), 200, dtype=jnp.int32),
        
        planet_initial_x=pad_array(obs['planets'].get('initial_x', obs['planets']['x']), 50),
        planet_initial_y=pad_array(obs['planets'].get('initial_y', obs['planets']['y']), 50),
        planet_is_orbiting=pad_array(obs['planets'].get('is_orbiting', [0]*50), 50),
        
        angular_velocity=jnp.array(obs.get('angular_velocity', 0.0)),
        tick=jnp.array(obs.get('tick', 0), dtype=jnp.int32)
    )

def run_diagnostics():
    print("--- 1. Initializing Local Model ---")
    rngs = nnx.Rngs(42)
    model = EntityTransformer(num_classes=5, rngs=rngs)
    
    # FIX: Grab the graphdef here
    graphdef, state_template = nnx.split(model)
    
    bin_path = "optimized_model.bin"
    if not os.path.exists(bin_path):
        print(f"ERROR: {bin_path} not found in current directory.")
        return
        
    with open(bin_path, "rb") as f:
        restored_state = serialization.from_bytes(state_template, f.read())
        
    # THE SURGICAL FIX: Merge instead of update
    model = nnx.merge(graphdef, restored_state)
    graphdef, state_flat = nnx.split(model)

    print("--- 2. Simulating Kaggle Environment Step ---")
    # Simulate Kaggle sending data where YOU are player 0, Enemy is player 1
    # Planet 0 belongs to you, Planet 1 belongs to enemy, Planet 2 is neutral
    simulated_kaggle_obs = {
        'planets': {
            'x': [10.0, 90.0, 50.0],
            'y': [10.0, 90.0, 50.0],
            'owner': [0, 1, -1], # Kaggle's raw ownership format
            'ships': [100.0, 100.0, 20.0],
            'production': [5.0, 5.0, 1.0]
        },
        'fleets': {}
    }
    simulated_player_id = 0 # Kaggle says you are player 0

    print(f"Kaggle Raw Player ID: {simulated_player_id}")
    print(f"Kaggle Raw Planet Owners: {simulated_kaggle_obs['planets']['owner']}")

    # Build the JAX EnvState using the fixed parser
    env_state = parse_kaggle_obs_fixed(simulated_kaggle_obs, simulated_player_id)
    print(f"\nParsed JAX Planet Owners: {env_state.planet_owner[:3]}")

    # Build the observation tensor using your exact local orbit_env.py logic
    # Notice we pass player_id=1 here because normalize_owners_fixed already forced our ID to 1
    obs_tensor = build_observation(env_state, player_id=1, win_rate=1.0)[None, ...]
    
    # Check what the model actually sees in the tensor (Index 5 is ownership)
    model_seen_owners = obs_tensor[0, :3, 5]
    print(f"Tensor Ownership Mapping (What the model sees): {model_seen_owners}")

    print("\n--- 3. Executing Inference ---")
    @jax.jit
    def fast_inference(obs_t, mask):
        merged = nnx.merge(graphdef, state_flat)
        _, launch_logits, _, _, _ = merged(obs_t, return_policy=True)
        return launch_logits[0]

    # Create the exact mask used in your agent.py
    my_planets_mask = (env_state.planet_owner == 1)[:50]
    print(f"My Planets Mask (First 3): {my_planets_mask[:3]}")

    raw_logits = fast_inference(obs_tensor, my_planets_mask)
    masked_logits = jnp.where(my_planets_mask, raw_logits, -1e9)
    
    print(f"\nRaw Logits (First 3): {raw_logits[:3]}")
    print(f"Masked Logits (First 3): {masked_logits[:3]}")
    
    launch_decisions = (masked_logits > 0.0).astype(jnp.int32)
    print(f"Final Launch Decisions (First 3): {launch_decisions[:3]}")
    
    if jnp.sum(launch_decisions) > 0:
        print("\n[SUCCESS] The agent successfully identified its planet and authorized a launch.")
    else:
        print("\n[WARNING] The agent identified its planet, but decided not to launch (Raw logit was < 0).")

if __name__ == "__main__":
    run_diagnostics()