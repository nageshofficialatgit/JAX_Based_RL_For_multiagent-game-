import jax
import jax.numpy as jnp
from test_ppo_agent import parse_kaggle_obs
from env_jax.orbit_env_v2 import apply_actions

# Dummy obs
obs = {
    'step': 0,
    'planets': [
        [0, 0, 50, 50, 8, 0, 0], # Sun
        [1, 1, 10, 10, 5, 10, 5], # P1 planet
        [2, 2, 90, 90, 5, 10, 5]  # P2 planet
    ],
    'fleets': []
}
env_state = parse_kaggle_obs(obs, 1)

l_acts = jnp.zeros(50, dtype=bool).at[1].set(True)
t_acts = jnp.zeros(50, dtype=jnp.int32).at[1].set(2)
s_acts = jnp.zeros(50, dtype=jnp.int32).at[1].set(9) # bucket 9 = 100%

next_env_state = apply_actions(env_state, 1, l_acts, t_acts, s_acts)

for i in range(200):
    if next_env_state.fleet_active[i] == 1 and env_state.fleet_active[i] == 0:
        print(f"Spawned fleet {i}!")
        print(f"dx: {next_env_state.fleet_dx[i]}")
        print(f"dy: {next_env_state.fleet_dy[i]}")
        print(f"ships: {next_env_state.fleet_ships[i]}")
        
if jnp.sum(next_env_state.fleet_active) == 0:
    print("NO FLEETS SPAWNED!")
