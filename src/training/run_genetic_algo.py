import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import optax
import time
from evosax import SimpleGA
from src.env_jax.orbit_env_v2 import EnvState, reset_env, step_physics, apply_actions
from src.env_jax.jax_heuristic_bot import ConfigJax, get_heuristic_action

def get_random_configs(rng, pop_size=1024):
    rngs = jax.random.split(rng, 12)
    
    # Each parameter is a 3D piecewise array: [pop_size, 3] for [early, mid, late]
    def init_piecewise(r, min_val, max_val):
        return jax.random.uniform(r, (pop_size, 3), minval=min_val, maxval=max_val)

    return ConfigJax(
        max_waves_per_turn=jnp.full((pop_size,), 6, dtype=jnp.int32),
        
        horizon=init_piecewise(rngs[11], 1.0, 50.0),
        roi_threshold=init_piecewise(rngs[0], 0.0, 3.0),
        min_ships_to_launch=init_piecewise(rngs[1], 1.0, 20.0),
        reinforce_size_beta=init_piecewise(rngs[2], 0.0, 2.0),
        reinforce_eta_free=init_piecewise(rngs[3], 0.0, 5.0),
        reinforce_eta_scale=init_piecewise(rngs[4], 1.0, 20.0),
        production_value_multiplier=init_piecewise(rngs[5], 0.0, 5.0),
        attrition_penalty=init_piecewise(rngs[6], 0.0, 3.0),
        harassment_reward_weight=init_piecewise(rngs[7], 0.0, 3.0),
        defend_production_bias=init_piecewise(rngs[8], 0.0, 3.0),
        regroup_weight=init_piecewise(rngs[9], 0.0, 2.0),
        panic_threshold=init_piecewise(rngs[10], 0.0, 3.0)
    )

@jax.jit
def flatten_config(config: ConfigJax):
    return jnp.concatenate([
        config.horizon, config.roi_threshold, config.min_ships_to_launch,
        config.reinforce_size_beta, config.reinforce_eta_free, config.reinforce_eta_scale,
        config.production_value_multiplier, config.attrition_penalty, config.harassment_reward_weight,
        config.defend_production_bias, config.regroup_weight, config.panic_threshold
    ], axis=1)

@jax.jit
def unflatten_config(x_array):
    pop_size = x_array.shape[0]
    return ConfigJax(
        max_waves_per_turn=jnp.full((pop_size,), 6, dtype=jnp.int32),
        horizon=x_array[:, 0:3],
        roi_threshold=x_array[:, 3:6],
        min_ships_to_launch=x_array[:, 6:9],
        reinforce_size_beta=x_array[:, 9:12],
        reinforce_eta_free=x_array[:, 12:15],
        reinforce_eta_scale=x_array[:, 15:18],
        production_value_multiplier=x_array[:, 18:21],
        attrition_penalty=x_array[:, 21:24],
        harassment_reward_weight=x_array[:, 24:27],
        defend_production_bias=x_array[:, 27:30],
        regroup_weight=x_array[:, 30:33],
        panic_threshold=x_array[:, 33:36]
    )


@jax.jit
def simulate_match(rng, config_p1: ConfigJax, opponent_pool: ConfigJax, match_indices: jnp.ndarray):
    config_p2 = jax.tree_util.tree_map(lambda x: x[match_indices[0]], opponent_pool)
    config_p3 = jax.tree_util.tree_map(lambda x: x[match_indices[1]], opponent_pool)
    config_p4 = jax.tree_util.tree_map(lambda x: x[match_indices[2]], opponent_pool)
    
    initial_state = reset_env(rng)
    
    def step_fn(state, tick):
        state = state.replace(tick=tick)
        
        launch1, tgt1, ship1 = get_heuristic_action(state, config_p1, player_id=1, tick=tick)
        launch2, tgt2, ship2 = get_heuristic_action(state, config_p2, player_id=2, tick=tick)
        launch3, tgt3, ship3 = get_heuristic_action(state, config_p3, player_id=3, tick=tick)
        launch4, tgt4, ship4 = get_heuristic_action(state, config_p4, player_id=4, tick=tick)
        
        state = apply_actions(state, 1, launch1, tgt1, ship1)
        state = apply_actions(state, 2, launch2, tgt2, ship2)
        state = apply_actions(state, 3, launch3, tgt3, ship3)
        state = apply_actions(state, 4, launch4, tgt4, ship4)
        
        state = step_physics(state)
        return state, None

    ticks = jnp.arange(250)
    final_state, _ = jax.lax.scan(step_fn, initial_state, ticks, length=250)
    
    p1_ships = jnp.sum(final_state.planet_ships * (final_state.planet_owner == 1))
    p2_ships = jnp.sum(final_state.planet_ships * (final_state.planet_owner == 2))
    p3_ships = jnp.sum(final_state.planet_ships * (final_state.planet_owner == 3))
    p4_ships = jnp.sum(final_state.planet_ships * (final_state.planet_owner == 4))
    enemy_ships = p2_ships + p3_ships + p4_ships
    
    p1_prod = jnp.sum(final_state.planet_production * (final_state.planet_owner == 1))
    p2_prod = jnp.sum(final_state.planet_production * (final_state.planet_owner == 2))
    p3_prod = jnp.sum(final_state.planet_production * (final_state.planet_owner == 3))
    p4_prod = jnp.sum(final_state.planet_production * (final_state.planet_owner == 4))
    enemy_prod = p2_prod + p3_prod + p4_prod
    
    fitness_ships = p1_ships / (p1_ships + enemy_ships + 1e-8)
    fitness_prod = p1_prod / (p1_prod + enemy_prod + 1e-8)
    
    return (fitness_ships * 0.5) + (fitness_prod * 0.5)

vmap_simulate = jax.jit(jax.vmap(simulate_match, in_axes=(None, 0, None, 0)))

def main():
    rng = jax.random.PRNGKey(42)
    pop_size = 512
    hof_size = 50
    num_generations = 500
    
    print(f"Initializing League Training GA: Population={pop_size}, HoF={hof_size}, Generations={num_generations}")
    print("Using 36-Dimensional Piecewise Constant Grid!")
    
    rng, r_init = jax.random.split(rng)
    population = get_random_configs(r_init, pop_size)
    hall_of_fame = get_random_configs(jax.random.PRNGKey(0), hof_size)
    hof_idx = 0
    
    print("Compiling JAX vmap graph... (Expect 3-5 minutes for 36D graph)")
    dummy_pool = jax.tree_util.tree_map(lambda p, h: jnp.concatenate([p, h], axis=0), population, hall_of_fame)
    dummy_indices = jax.random.randint(r_init, shape=(pop_size, 3), minval=0, maxval=pop_size+hof_size)
    _ = vmap_simulate(r_init, population, dummy_pool, dummy_indices)
    
    # Initialize SimpleGA Strategy
    strategy = SimpleGA(popsize=pop_size, num_dims=36, maximize=True, elite_ratio=0.1)
    
    # Define physical bounds for the 12 parameters
    # [min, max] pairs
    bounds = jnp.array([
        [1.0, 50.0],   # 0: horizon
        [0.0, 3.0],    # 1: roi_threshold
        [1.0, 20.0],   # 2: min_ships_to_launch
        [0.0, 2.0],    # 3: reinforce_size_beta
        [0.0, 5.0],    # 4: reinforce_eta_free
        [1.0, 20.0],   # 5: reinforce_eta_scale
        [0.0, 5.0],    # 6: production_value_multiplier
        [0.0, 3.0],    # 7: attrition_penalty
        [0.0, 3.0],    # 8: harassment_reward_weight
        [0.0, 3.0],    # 9: defend_production_bias
        [0.0, 2.0],    # 10: regroup_weight
        [0.0, 3.0]     # 11: panic_threshold
    ])
    
    # Repeat each bound 3 times for [Early, Mid, Late] phases
    clip_min = jnp.repeat(bounds[:, 0], 3)
    clip_max = jnp.repeat(bounds[:, 1], 3)
        
    es_params = strategy.default_params.replace(
        clip_min=clip_min, 
        clip_max=clip_max,
        sigma_init=10.0,
        cross_over_rate=0.5
    )
    state = strategy.initialize(r_init, es_params)
    
    # Override initial mean with a random seed from our pool to give it a good starting point
    flat_pop = flatten_config(population)
    state = state.replace(mean=flat_pop[0])
    
    print("Compilation successful! Starting 500 Generation SimpleGA Evolution...")
    
    for gen in range(num_generations):
        gen_start = time.time()
        
        rng, subkey1, subkey2, subkey3 = jax.random.split(rng, 4)
        
        # Dynamic Resolution Annealing
        # Starts with 10 coarse intervals, scaling down to 1000 fine intervals by the final generation
        current_intervals = 10.0 + (990.0 * (gen / num_generations))
        step_size = (clip_max - clip_min) / current_intervals
        
        # 1. Ask for new continuous candidate parameters
        x, state = strategy.ask(subkey1, state, es_params) 
        
        # 2. Grid Discretization: Snap to dynamic resolution
        x = jnp.round((x - clip_min) / step_size) * step_size + clip_min
        
        # 3. Build full configs
        population = unflatten_config(x)
        
        # 3. Create the Red Queen League Pool (Current Pop + Hall of Fame)
        opponent_pool = jax.tree_util.tree_map(lambda p, h: jnp.concatenate([p, h], axis=0), population, hall_of_fame)
        master_pool_size = pop_size + hof_size
        
        # 4. Play League Matches
        match_indices = jax.random.randint(subkey3, shape=(pop_size, 3), minval=0, maxval=master_pool_size)
        fitness_scores = vmap_simulate(subkey2, population, opponent_pool, match_indices)
        fitness_scores_f32 = fitness_scores.astype(jnp.float32)
        
        # 5. Tell the GA the results so it updates population
        state = strategy.tell(x, fitness_scores_f32, state, es_params)
        
        # Metrics
        best_fit = float(jnp.max(fitness_scores_f32))
        mean_fit = float(jnp.mean(fitness_scores_f32))
        gen_time = time.time() - gen_start
        
        print(f"Gen {gen:03d} | Best Fit: {best_fit:.4f} | Mean Fit: {mean_fit:.4f} | Mut Sigma: {state.sigma:.4f} | Time: {gen_time:.2f}s")
        
        # 6. Update Hall of Fame with the winner of this generation
        best_idx = jnp.argmax(fitness_scores_f32)
        def update_hof(hof_arr, pop_arr):
            best_val = pop_arr[best_idx]
            return hof_arr.at[hof_idx].set(best_val)
            
        hall_of_fame = jax.tree_util.tree_map(update_hof, hall_of_fame, population)
        hof_idx = (hof_idx + 1) % hof_size
        
        if gen == num_generations - 1:
            print("\n=== Best Evolutionary Cubic Config Found ===")
            
            import json
            best_config_dict = {
                "horizon": population.horizon[best_idx].tolist(),
                "max_waves_per_turn": int(population.max_waves_per_turn[best_idx]),
                "roi_threshold": population.roi_threshold[best_idx].tolist(),
                "min_ships_to_launch": population.min_ships_to_launch[best_idx].tolist(),
                "reinforce_size_beta": population.reinforce_size_beta[best_idx].tolist(),
                "reinforce_eta_free": population.reinforce_eta_free[best_idx].tolist(),
                "reinforce_eta_scale": population.reinforce_eta_scale[best_idx].tolist(),
                "production_value_multiplier": population.production_value_multiplier[best_idx].tolist(),
                "attrition_penalty": population.attrition_penalty[best_idx].tolist(),
                "harassment_reward_weight": population.harassment_reward_weight[best_idx].tolist(),
                "defend_production_bias": population.defend_production_bias[best_idx].tolist(),
                "regroup_weight": population.regroup_weight[best_idx].tolist(),
                "panic_threshold": population.panic_threshold[best_idx].tolist(),
            }
            
            for k, v in best_config_dict.items():
                print(f"{k}: {v}")
                
            with open("best_ga_config.json", "w") as f:
                json.dump(best_config_dict, f, indent=4)
                
            print("\nSuccessfully saved full 36-dimensional piecewise config to best_ga_config.json!")
            break

if __name__ == "__main__":
    main()
