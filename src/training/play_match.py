import os
import sys
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.entity_transformer_flax import EntityTransformer
from env_jax.orbit_env import EnvState, step_physics, apply_actions, build_observation

def initialize_real_map(db_path, episode_id):
    df_ep_planets = pd.read_parquet(os.path.join(db_path, "episode_planets.parquet"))
    df_ep_planets.set_index(['episode_id', 'planet_id'], inplace=True)
    
    path = os.path.join(db_path, "planet_state_coords.parquet")
    if not os.path.exists(path):
        path = os.path.join(db_path, "planet_state.parquet")
    df_states = pd.read_parquet(path)
    df_states.set_index(['episode_id', 'tick', 'planet_id'], inplace=True)
    
    static_planets = df_ep_planets.loc[episode_id]
    dynamic_planets = df_states.loc[(episode_id, 0)]
    planets = static_planets.join(dynamic_planets, how='left').sort_index()
    
    n = len(planets)
    
    is_static = planets['is_static'].values
    is_comet = planets['is_comet'].values
    is_orbiting = (is_static == 0) & (is_comet == 0)
    
    eps = pd.read_parquet(os.path.join(db_path, "episodes.parquet"))
    angular_vel = eps[eps['episode_id'] == episode_id]['angular_velocity'].values[0]
    
    def pad_to_50(arr, val=0.0):
        pad_width = max(0, 50 - len(arr))
        return jnp.pad(jnp.array(arr), (0, pad_width), constant_values=val)[:50]
        
    state = EnvState(
        planet_x=pad_to_50(planets['initial_x'].values, 0.0),
        planet_y=pad_to_50(planets['initial_y'].values, 0.0),
        planet_radius=pad_to_50(planets['radius'].values, 0.0),
        planet_production=pad_to_50(planets['production'].values, 0.0),
        planet_owner=pad_to_50(planets['owner'].values, 0).astype(jnp.int32),
        planet_ships=pad_to_50(planets['ships'].values, 0.0),
        
        fleet_active=jnp.zeros(200, dtype=jnp.bool_),
        fleet_owner=jnp.zeros(200, dtype=jnp.int32),
        fleet_ships=jnp.zeros(200, dtype=jnp.float32),
        fleet_x=jnp.zeros(200, dtype=jnp.float32),
        fleet_y=jnp.zeros(200, dtype=jnp.float32),
        fleet_dx=jnp.zeros(200, dtype=jnp.float32),
        fleet_dy=jnp.zeros(200, dtype=jnp.float32),
        fleet_src_planet=jnp.zeros(200, dtype=jnp.int32),
        
        planet_initial_x=pad_to_50(planets['initial_x'].values, 0.0),
        planet_initial_y=pad_to_50(planets['initial_y'].values, 0.0),
        planet_is_orbiting=pad_to_50(is_orbiting, False).astype(jnp.bool_),
        angular_velocity=jnp.array(angular_vel, dtype=jnp.float32),
        tick=jnp.array(0, dtype=jnp.int32)
    )
    return state

def play_match():
    print("Loading Checkpoint...")
    rngs = nnx.Rngs(42)
    model = EntityTransformer(d_model=512, n_heads=8, n_layers=12, rngs=rngs)
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../checkpoints/flax'))
    mngr = ocp.CheckpointManager(ckpt_dir)
    
    # Try finding best step
    step = 12000
    if step not in mngr.all_steps():
        if len(mngr.all_steps()) > 0:
            step = mngr.latest_step()
        else:
            print("No checkpoints found!")
            return
            
    raw = mngr.restore(step)
    if isinstance(raw, dict) and '0' in raw and '1' in raw:
        dummy_tx = optax.adamw(learning_rate=3e-4)
        dummy_opt = nnx.Optimizer(model, dummy_tx)
        template = nnx.state((model, dummy_opt))
        restored = mngr.restore(step, args=ocp.args.StandardRestore(template))
        nnx.update(model, restored[0])
    else:
        _, state = nnx.split(model)
        restore_args = ocp.args.StandardRestore(state)
        restored = mngr.restore(step, args=restore_args)
        nnx.update(model, restored)
    
    print("Initializing Live JAX Environment from a real map...")
    env_state = initialize_real_map("data/parquet_db_real", episode_id=78775099)
    
    print("--- Starting 100-Tick JAX Self-Play Match ---")
    
    for tick in range(100):
        obs = build_observation(env_state, win_rate=1.0)
        obs_batch = obs[None, :, :]
        
        # Unpack 5 Variables
        v, launch, angle, ships, ppo_value = model(obs_batch, return_policy=True)
        
        launch_mask = (launch[0] > 0).astype(jnp.int32)
        target_buckets = jnp.argmax(angle[0], axis=-1).astype(jnp.int32)
        ships_buckets = jnp.argmax(ships[0], axis=-1).astype(jnp.int32)
        
        env_state = apply_actions(env_state, player_id=1, launch=launch_mask, target=target_buckets, ships=ships_buckets)
        env_state = step_physics(env_state)
        
    print("\nMatch Complete!")
    print(f"Final Fleet Count: {jnp.sum(env_state.fleet_active)}")
    print(f"Final Planets Owned by P1: {jnp.sum(env_state.planet_owner == 1)}")

if __name__ == "__main__":
    play_match()