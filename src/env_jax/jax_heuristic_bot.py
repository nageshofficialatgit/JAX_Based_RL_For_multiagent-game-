import jax
import jax.numpy as jnp
import chex
from typing import Tuple

from src.env_jax.orbit_env_v2 import EnvState, get_fleet_speed, check_sun_collision

@chex.dataclass
class ConfigJax:
    """Hyperparameters for the JAX Heuristic Bot, tuned by Genetic Algorithm.
       Each major parameter is a [pop_size, 3] array for Piecewise Constant Phases:
       [Early Game, Mid Game, Late Game]
    """
    max_waves_per_turn: jnp.ndarray # Stays [pop_size] as int32
    
    # 3D Piecewise Parameters [pop_size, 3]
    horizon: jnp.ndarray
    roi_threshold: jnp.ndarray
    min_ships_to_launch: jnp.ndarray
    reinforce_size_beta: jnp.ndarray
    reinforce_eta_free: jnp.ndarray
    reinforce_eta_scale: jnp.ndarray
    production_value_multiplier: jnp.ndarray
    attrition_penalty: jnp.ndarray
    harassment_reward_weight: jnp.ndarray
    defend_production_bias: jnp.ndarray
    regroup_weight: jnp.ndarray
    panic_threshold: jnp.ndarray

def get_current_params(config: ConfigJax, tick: int):
    """Evaluates the piecewise constants for the current tick. (Called inside vmap so arrays are [3])"""
    # 0-133: Early (0), 134-266: Mid (1), 267-400: Late (2)
    phase_idx = jnp.where(tick < 133, 0, jnp.where(tick < 266, 1, 2))
    
    def eval_piecewise(param_array):
        return param_array[phase_idx]
        
    return {
        "horizon": jnp.clip(eval_piecewise(config.horizon), 1.0, 50.0),
        "roi_threshold": eval_piecewise(config.roi_threshold),
        "min_ships_to_launch": jnp.maximum(1.0, eval_piecewise(config.min_ships_to_launch)),
        "reinforce_size_beta": eval_piecewise(config.reinforce_size_beta),
        "reinforce_eta_free": eval_piecewise(config.reinforce_eta_free),
        "reinforce_eta_scale": jnp.maximum(0.1, eval_piecewise(config.reinforce_eta_scale)),
        "production_value_multiplier": eval_piecewise(config.production_value_multiplier),
        "attrition_penalty": eval_piecewise(config.attrition_penalty),
        "harassment_reward_weight": eval_piecewise(config.harassment_reward_weight),
        "defend_production_bias": eval_piecewise(config.defend_production_bias),
        "regroup_weight": eval_piecewise(config.regroup_weight),
        "panic_threshold": eval_piecewise(config.panic_threshold)
    }

@jax.jit
def get_enemy_pressure(state: EnvState, player_id: int, horizon) -> jnp.ndarray:
    """Calculates spatial gravity of reachable enemy garrisons."""
    dx_mat = state.planet_x[:, None] - state.planet_x[None, :]
    dy_mat = state.planet_y[:, None] - state.planet_y[None, :]
    dist_mat = jnp.hypot(dx_mat, dy_mat) # [P, P]
    
    enemy_mask = (state.planet_owner != player_id) & (state.planet_owner != 0) & (state.planet_radius > 0.0)
    ships = state.planet_ships
    speeds = get_fleet_speed(ships)
    reach_dist = speeds * horizon # [P]
    
    decay = 1.0 - (dist_mat / (reach_dist[:, None] + 1e-8))
    decay = jnp.clip(decay, 0.0, 1.0)
    
    contrib = jnp.where(enemy_mask[:, None], ships[:, None] * decay, 0.0)
    return jnp.sum(contrib, axis=0) # [P]

@jax.jit
def get_garrison_status(state: EnvState, player_id: int, horizon, params: dict):
    """Calculates safe drain (surplus), capture floor, and enemy mass across a time horizon."""
    f_active = state.fleet_active
    f_ships = state.fleet_ships
    f_owner = state.fleet_owner
    f_target = state.fleet_target_planet
    
    target_mask = jax.nn.one_hot(f_target, 50) * f_active[:, None] # [F, 50]
    
    speed = get_fleet_speed(f_ships)
    eta = jnp.hypot(state.planet_x[f_target] - state.fleet_x, state.planet_y[f_target] - state.fleet_y) / (speed + 1e-8)
    
    eta_matrix_all = jnp.where(target_mask > 0, eta[:, None], jnp.inf)
    is_planet_neutral = (state.planet_owner == 0)[None, :]
    is_same_owner = (f_owner[:, None] == state.planet_owner[None, :]) & ~is_planet_neutral
    
    fleet_impact_matrix = jnp.where(is_same_owner, f_ships[:, None], -f_ships[:, None]) * target_mask
    
    K_steps = jnp.arange(1, 51) # Statically sizing to max horizon 50
    valid_k = (K_steps <= horizon)[:, None]
    
    eta_less_than_k = eta_matrix_all[None, :, :] <= K_steps[:, None, None] # [50, F, 50]
    past_impacts_k = jnp.sum(fleet_impact_matrix[None, :, :] * eta_less_than_k, axis=1) # [50, 50]
    
    is_owned = (state.planet_owner > 0)
    prod_k = state.planet_production[None, :] * K_steps[:, None] * is_owned[None, :] # [50, 50]
    
    G_k = state.planet_ships[None, :] + prod_k + past_impacts_k # [50, 50]
    
    min_G = jnp.min(jnp.where(valid_k, G_k, jnp.inf), axis=0) # [50]
    # CRITICAL FIX: We cannot launch ships from the future. Cap at current ships.
    min_G = jnp.minimum(min_G, state.planet_ships)
    
    my_mask = (state.planet_owner == player_id)
    safe_drain = jnp.maximum(0.0, min_G) * my_mask
    
    k_arr = jnp.arange(1, 51, dtype=jnp.float32)
    rho = (k_arr - params["reinforce_eta_free"]) / params["reinforce_eta_scale"]
    rho = jnp.clip(rho, 0.0, 1.0)
    
    enemy_mass = get_enemy_pressure(state, player_id, horizon)
    reinforcement = params["reinforce_size_beta"] * rho[:, None] * enemy_mass[None, :] # [50, 50]
    
    # Defensive bias: we need less ships to reinforce our own high-production planets
    defense_floor = jnp.ones_like(G_k) - (state.planet_production[None, :] * params["defend_production_bias"])
    defense_floor = jnp.maximum(1.0, defense_floor)
    
    is_mine = (state.planet_owner == player_id)[None, :]
    is_neutral = (state.planet_owner == 0)[None, :]
    # For neutral targets: just beat the garrison (no reinforcement expected)
    # For enemy targets: beat garrison + potential reinforcements
    # For own planets: low defense floor
    floor_k = jnp.where(
        is_mine,
        defense_floor,
        jnp.where(
            is_neutral,
            jnp.maximum(1.0, G_k + 1.0),                          # Neutrals: just beat garrison
            jnp.maximum(1.0, G_k + reinforcement + 1.0)            # Enemies: beat garrison + reinforcements
        )
    )
    
    return safe_drain, floor_k, enemy_mass

@jax.jit
def get_heuristic_action(state: EnvState, config: ConfigJax, player_id: int, tick: int):
    params = get_current_params(config, tick)
    horizon = jnp.round(params["horizon"]).astype(jnp.int32)
    
    safe_drain, floor_k, enemy_mass = get_garrison_status(state, player_id, horizon, params)
    
    # ETA Matrix
    dx = state.planet_x[None, :] - state.planet_x[:, None]
    dy = state.planet_y[None, :] - state.planet_y[:, None]
    dist = jnp.hypot(dx, dy)
    
    sizes = safe_drain[:, None] # [S, 1]
    speed = get_fleet_speed(sizes)
    eta = dist / (speed + 1e-8) # [S, T]
    
    sun_blocked = check_sun_collision(
        state.planet_x[:, None], state.planet_y[:, None],
        state.planet_x[None, :], state.planet_y[None, :]
    )
    
    eta_idx = jnp.clip(jnp.ceil(eta) - 1, 0, horizon - 1).astype(jnp.int32)
    flat_idx = eta_idx * 50 + jnp.arange(50)[None, :]
    flat_floor = floor_k.flatten()
    floor_at_arr = flat_floor[flat_idx] # [S, T]
    
    valid = (
        (sizes >= floor_at_arr) &
        (sizes >= params["min_ships_to_launch"]) &
        # NOTE: sun_blocked removed from hard filter. The Kaggle engine handles collisions.
        # Sun-blocked paths get a ROI penalty below instead.
        (state.planet_radius[:, None] > 0.0) &
        (state.planet_radius[None, :] > 0.0) &
        (jnp.arange(50)[:, None] != jnp.arange(50)[None, :]) &
        (state.planet_owner[:, None] == player_id)
    )
    
    # Attack Scores
    survival = sizes - floor_at_arr
    attrition_score = survival * params["attrition_penalty"]
    prod_value = state.planet_production[None, :] * params["production_value_multiplier"]
    is_enemy = (state.planet_owner[None, :] != player_id) & (state.planet_owner[None, :] != 0)
    harass_score = is_enemy * state.planet_production[None, :] * params["harassment_reward_weight"]
    
    roi_score = ((prod_value * (100.0 - eta)) + attrition_score + harass_score) / (sizes + 1e-8)
    # Apply severe penalty if path is blocked by the sun
    roi_score = jnp.where(sun_blocked, roi_score - 1000.0, roi_score)
    
    # Panic Adjustment (if losing total ships, increase defend bias dynamically)
    my_ships = jnp.sum(state.planet_ships * (state.planet_owner == player_id))
    enemy_ships = jnp.sum(state.planet_ships * is_enemy[0])
    panic = jnp.where(my_ships < enemy_ships * params["panic_threshold"], 1.5, 1.0)
    roi_score = roi_score * jnp.where(is_enemy, 1.0 / panic, 1.0)
    
    # Regroup Scores (NEW)
    is_ally = (state.planet_owner[None, :] == player_id)
    pressure_delta = enemy_mass[None, :] - enemy_mass[:, None] # Target Pressure - Source Pressure
    # Only regroup if target is in higher danger, and distance isn't too extreme
    regroup_score = jnp.where(is_ally, (pressure_delta * params["regroup_weight"]) / (eta + 1e-8), -jnp.inf)
    
    # Unified Score Selection
    master_score = jnp.where(is_ally, regroup_score, roi_score)
    score = jnp.where(valid, master_score, -jnp.inf)
    
    # Filter by ROI threshold
    score = jnp.where(score > params["roi_threshold"], score, -jnp.inf)
    
    launch = jnp.zeros(50, dtype=jnp.bool_)
    target = jnp.zeros(50, dtype=jnp.int32)
    ship_buckets = jnp.zeros(50, dtype=jnp.int32)
    
    def assign_wave(i, carry):
        launch, target, ship_buckets, score_matrix = carry
        
        # Find global max score across all (source, target) pairs
        flat_idx = jnp.argmax(score_matrix)
        best_src = flat_idx // 50
        best_tgt = flat_idx % 50
        max_score = score_matrix[best_src, best_tgt]
        
        is_valid = max_score > -jnp.inf
        
        # If valid, record action
        launch = jnp.where(is_valid, launch.at[best_src].set(True), launch)
        target = jnp.where(is_valid, target.at[best_src].set(best_tgt), target)
        ship_buckets = jnp.where(is_valid, ship_buckets.at[best_src].set(9), ship_buckets)
        
        # Mask out the source and target so they can't be picked again this turn
        score_matrix = jnp.where(is_valid, score_matrix.at[best_src, :].set(-jnp.inf), score_matrix)
        score_matrix = jnp.where(is_valid, score_matrix.at[:, best_tgt].set(-jnp.inf), score_matrix)
        
        return launch, target, ship_buckets, score_matrix

    launch, target, ship_buckets, _ = jax.lax.fori_loop(0, 6, assign_wave, (launch, target, ship_buckets, score))
    
    return launch, target, ship_buckets
