import numpy as np
import math
import jax
import jax.numpy as jnp
import traceback
import os

from src.env_jax.jax_heuristic_bot import ConfigJax, get_heuristic_action, get_garrison_status, get_current_params
from src.env_jax.orbit_env_v2 import EnvState, get_fleet_speed

# The Best Configuration Discovered by the previous 50-Gen JAX Genetic Algorithm
# Manually tuned baseline config for Kaggle environment (flat cubic: a=value, b=c=d=0)
BEST_CONFIG = ConfigJax(
    horizon=jnp.array([16.0, 25.0, 25.0]),                      
    max_waves_per_turn=6, 
    roi_threshold=jnp.array([0.1, 0.5, 0.5]),       
    min_ships_to_launch=jnp.array([4.0, 4.0, 4.0]),
    reinforce_size_beta=jnp.array([2.2, 2.2, 2.2]), 
    reinforce_eta_free=jnp.array([3.0, 3.0, 3.0]),
    reinforce_eta_scale=jnp.array([12.0, 12.0, 12.0]),
    production_value_multiplier=jnp.array([1.0, 1.0, 1.0]),
    attrition_penalty=jnp.array([1.0, 1.0, 1.0]),
    harassment_reward_weight=jnp.array([1.0, 1.0, 1.0]),
    defend_production_bias=jnp.array([1.0, 1.0, 1.0]),
    regroup_weight=jnp.array([1e-3, 1e-3, 1e-3]),
    panic_threshold=jnp.array([1.0, 1.0, 1.0])
)

CENTER_X = 50.0
CENTER_Y = 50.0
ROTATION_LIMIT = 42.0

def log_debug(msg):
    with open("jax_agent_debug.log", "a") as f:
        f.write(msg + "\n")

def agent(obs, config):
    try:
        tick = int(obs.get("step", 0) or 0)
        kaggle_player = int(obs.get("player", 0) or 0)

        if tick == 0:
            for f in ["jax_agent_debug.log", "jax_agent_crash.log"]:
                if os.path.exists(f):
                    os.remove(f)
            log_debug(f"--- Starting New Match | Kaggle Player ID: {kaggle_player} ---")
        
        # --- PARSE PLANETS ---
        # Kaggle: planets[i] = [id, owner, x, y, radius, ships, production]
        # owner: -1 = neutral
        raw_planets = obs.get("planets", []) or []
        raw_initial = obs.get("initial_planets", []) or []
        ang_vel = float(obs.get("angular_velocity", 0.0) or 0.0)
        
        p_x = np.zeros(50, dtype=np.float32)
        p_y = np.zeros(50, dtype=np.float32)
        p_owner = np.zeros(50, dtype=np.int32)
        p_ships = np.zeros(50, dtype=np.float32)
        p_prod = np.zeros(50, dtype=np.float32)
        p_rad = np.zeros(50, dtype=np.float32)
        p_init_x = np.zeros(50, dtype=np.float32)
        p_init_y = np.zeros(50, dtype=np.float32)
        p_is_orbiting = np.zeros(50, dtype=np.float32)
        
        for p_data in raw_planets:
            pid = int(p_data[0])
            if pid >= 50:
                continue
            owner_raw = p_data[1]
            if owner_raw is None or owner_raw == -1:
                p_owner[pid] = 0
            else:
                p_owner[pid] = int(owner_raw) + 1
            p_x[pid] = float(p_data[2])
            p_y[pid] = float(p_data[3])
            p_rad[pid] = float(p_data[4])
            p_ships[pid] = float(p_data[5])
            p_prod[pid] = float(p_data[6])

        # Parse initial planets for orbital detection
        for p_data in raw_initial:
            pid = int(p_data[0])
            if pid >= 50:
                continue
            ix, iy = float(p_data[2]), float(p_data[3])
            p_init_x[pid] = ix
            p_init_y[pid] = iy
            r_from_center = math.hypot(ix - CENTER_X, iy - CENTER_Y)
            ir = float(p_data[4])
            if r_from_center + ir < ROTATION_LIMIT:
                p_is_orbiting[pid] = 1.0

        # --- PARSE FLEETS ---
        # Kaggle: fleets[i] = [id, owner, x, y, angle, from_planet_id, ships]
        raw_fleets = obs.get("fleets", []) or []
        
        f_active = np.zeros(200, dtype=np.bool_)
        f_owner = np.zeros(200, dtype=np.int32)
        f_ships = np.zeros(200, dtype=np.float32)
        f_x = np.zeros(200, dtype=np.float32)
        f_y = np.zeros(200, dtype=np.float32)
        f_dx = np.zeros(200, dtype=np.float32)
        f_dy = np.zeros(200, dtype=np.float32)
        f_src = np.zeros(200, dtype=np.int32)
        f_target = np.zeros(200, dtype=np.int32)
        
        for idx, f_data in enumerate(raw_fleets):
            if idx >= 200:
                break
            f_active[idx] = True
            owner_raw = f_data[1]
            if owner_raw is None or owner_raw == -1:
                f_owner[idx] = 0
            else:
                f_owner[idx] = int(owner_raw) + 1
            f_x[idx] = float(f_data[2])
            f_y[idx] = float(f_data[3])
            angle = float(f_data[4])
            f_src[idx] = int(f_data[5])
            f_ships[idx] = float(f_data[6])
            
            speed = 1.0
            s = float(f_data[6])
            if s > 1.0:
                ratio = math.log(s) / math.log(1000.0)
                ratio = max(0.0, min(1.0, ratio))
                speed = 1.0 + 5.0 * (ratio ** 1.5)
            f_dx[idx] = math.cos(angle) * speed
            f_dy[idx] = math.sin(angle) * speed
            
            # Infer target: find nearest planet in the direction of angle
            dx_dir = math.cos(angle)
            dy_dir = math.sin(angle)
            best_pid = 0
            best_proj = float('inf')
            for p_data_inner in raw_planets:
                tpid = int(p_data_inner[0])
                if tpid >= 50:
                    continue
                dx = float(p_data_inner[2]) - f_x[idx]
                dy = float(p_data_inner[3]) - f_y[idx]
                proj = dx * dx_dir + dy * dy_dir
                if proj <= 0:
                    continue
                perp_sq = max(0, dx**2 + dy**2 - proj**2)
                r = float(p_data_inner[4])
                if perp_sq < r * r * 4 and proj < best_proj:  # generous radius
                    best_proj = proj
                    best_pid = tpid
            f_target[idx] = best_pid

        # --- BUILD JAX STATE ---
        state = EnvState(
            planet_x=jnp.array(p_x),
            planet_y=jnp.array(p_y),
            planet_owner=jnp.array(p_owner),
            planet_ships=jnp.array(p_ships),
            planet_production=jnp.array(p_prod),
            planet_radius=jnp.array(p_rad),
            fleet_active=jnp.array(f_active),
            fleet_owner=jnp.array(f_owner),
            fleet_ships=jnp.array(f_ships),
            fleet_x=jnp.array(f_x),
            fleet_y=jnp.array(f_y),
            fleet_dx=jnp.array(f_dx),
            fleet_dy=jnp.array(f_dy),
            fleet_src_planet=jnp.array(f_src),
            fleet_target_planet=jnp.array(f_target),
            planet_initial_x=jnp.array(p_init_x),
            planet_initial_y=jnp.array(p_init_y),
            planet_is_orbiting=jnp.array(p_is_orbiting),
            angular_velocity=jnp.array(ang_vel),
            planet_dx=jnp.zeros(50),
            planet_dy=jnp.zeros(50),
            comet_starts_x=jnp.zeros((5, 4)),
            comet_starts_y=jnp.zeros((5, 4)),
            comet_dx=jnp.zeros((5, 4)),
            comet_dy=jnp.zeros((5, 4)),
            comet_ships=jnp.zeros((5, 4)),
            tick=jnp.array(tick)
        )
        
        player_id = kaggle_player + 1
        
        if tick == 5:
            my_pids = np.where(p_owner == player_id)[0]
            log_debug(f"DEBUG tick 5: player_id={player_id}, my_planets={my_pids.tolist()}")
            for mp in my_pids:
                log_debug(f"  P{mp}: ships={p_ships[mp]:.0f} prod={p_prod[mp]:.0f} rad={p_rad[mp]:.2f} x={p_x[mp]:.1f} y={p_y[mp]:.1f}")
            neutrals = np.where(p_owner == 0)[0]
            for np_id in neutrals[:5]:
                if p_rad[np_id] > 0:
                    d = np.hypot(p_x[my_pids[0]] - p_x[np_id], p_y[my_pids[0]] - p_y[np_id])
                    log_debug(f"  Neutral P{np_id}: ships={p_ships[np_id]:.0f} prod={p_prod[np_id]:.0f} dist={d:.1f} rad={p_rad[np_id]:.2f}")
        
        # Run the JAX heuristic logic
        launch_mask, target_idx, _ = get_heuristic_action(state, BEST_CONFIG, player_id, tick)
        
        # Get precise safe drain
        params = get_current_params(BEST_CONFIG, tick)
        safe_drain, _, _ = get_garrison_status(state, player_id, 50, params)
        
        launch_mask_np = np.array(launch_mask)
        target_idx_np = np.array(target_idx)
        safe_drain_np = np.array(safe_drain)
        
        actions = []
        for i in range(50):
            if launch_mask_np[i] and p_owner[i] == player_id:
                tgt = int(target_idx_np[i])
                ships_to_send = max(1, int(safe_drain_np[i]))
                if ships_to_send <= p_ships[i] and p_rad[tgt] > 0:
                    # Calculate interception angle
                    if p_is_orbiting[tgt]:
                        speed = 1.0
                        if ships_to_send > 1.0:
                            ratio = math.log(ships_to_send) / math.log(1000.0)
                            ratio = max(0.0, min(1.0, ratio))
                            speed = 1.0 + 5.0 * (ratio ** 1.5)
                        
                        r = math.hypot(p_x[tgt] - 50.0, p_y[tgt] - 50.0)
                        cur_angle = math.atan2(p_y[tgt] - 50.0, p_x[tgt] - 50.0)
                        turns = math.hypot(p_x[tgt] - p_x[i], p_y[tgt] - p_y[i]) / speed
                        for _ in range(5):
                            future_angle = cur_angle + ang_vel * turns
                            fx = 50.0 + r * math.cos(future_angle)
                            fy = 50.0 + r * math.sin(future_angle)
                            turns = math.hypot(fx - p_x[i], fy - p_y[i]) / speed
                        
                        angle = math.atan2(fy - p_y[i], fx - p_x[i])
                    else:
                        angle = math.atan2(p_y[tgt] - p_y[i], p_x[tgt] - p_x[i])
                        
                    actions.append([int(i), float(angle), int(ships_to_send)])
                    log_debug(f"Tick {tick}: P{player_id} launch {ships_to_send} from P{i} -> P{tgt} angle={angle:.2f}")
                
        if tick % 10 == 0:
            my_count = int(np.sum(p_owner == player_id))
            my_ships = int(np.sum(p_ships * (p_owner == player_id)))
            sd_max = float(np.max(safe_drain_np * (p_owner == player_id)))
            log_debug(f"Tick {tick}: Owned {my_count} planets, {my_ships} ships, max_safe_drain={sd_max:.1f}, {len(actions)} actions")
                
        return actions

    except Exception as e:
        with open("jax_agent_crash.log", "a") as f:
            f.write(f"\n--- CRASH AT TICK {obs.get('step', '?')} ---\n")
            traceback.print_exc(file=f)
        return []
