import jax
import jax.numpy as jnp
import chex

@chex.dataclass
class EnvState:
    planet_x: jnp.ndarray          # [N]
    planet_y: jnp.ndarray          # [N]
    planet_radius: jnp.ndarray     # [N]
    planet_production: jnp.ndarray # [N]
    planet_owner: jnp.ndarray      # [N] (0: neutral, 1: p1, 2: p2, 3: p3, 4: p4)
    planet_ships: jnp.ndarray      # [N] float to handle non-integer production
    
    fleet_active: jnp.ndarray      # [F] 1 if active, 0 if inactive slot
    fleet_owner: jnp.ndarray       # [F]
    fleet_ships: jnp.ndarray       # [F]
    fleet_x: jnp.ndarray           # [F]
    fleet_y: jnp.ndarray           # [F]
    fleet_dx: jnp.ndarray          # [F] velocity x
    fleet_dy: jnp.ndarray          # [F] velocity y
    fleet_src_planet: jnp.ndarray  # [F] source planet id
    fleet_target_planet: jnp.ndarray # [F] target planet id
    
    planet_initial_x: jnp.ndarray  # [N]
    planet_initial_y: jnp.ndarray  # [N]
    planet_is_orbiting: jnp.ndarray # [N] bool (1 = rotating, 0 = static)
    angular_velocity: jnp.ndarray  # scalar
    
    planet_dx: jnp.ndarray         # [N]
    planet_dy: jnp.ndarray         # [N]
    
    comet_starts_x: jnp.ndarray    # [5, 4]
    comet_starts_y: jnp.ndarray    # [5, 4]
    comet_dx: jnp.ndarray          # [5, 4]
    comet_dy: jnp.ndarray          # [5, 4]
    comet_ships: jnp.ndarray       # [5, 4]
    
    tick: jnp.ndarray              # scalar

@jax.jit
def get_fleet_speed(ships: jnp.ndarray) -> jnp.ndarray:
    """Non-linear speed calculation ported from Kaggle math_utils.py."""
    safe_ships = jnp.maximum(ships, 1e-8)
    ratio = jnp.log(safe_ships) / jnp.log(1000.0)
    ratio = jnp.clip(ratio, 0.0, 1.0)
    speed = 1.0 + 5.0 * (ratio ** 1.5)
    return jnp.where(ships <= 1.0, 1.0, speed)

@jax.jit
def step_physics(state: EnvState) -> EnvState:
    """Core physics: production and fleet movement."""
    # 1. Planet Production (only for owned planets)
    is_owned = state.planet_owner > 0
    new_ships = state.planet_ships + (state.planet_production * is_owned)
    
    # 2. Fleet Movement
    old_fleet_x = state.fleet_x
    old_fleet_y = state.fleet_y
    new_fleet_x = state.fleet_x + (state.fleet_dx * state.fleet_active)
    new_fleet_y = state.fleet_y + (state.fleet_dy * state.fleet_active)
    
    # 2.5 Planet Orbital Rotation (Fixed Angular Velocity)
    dx_orb = state.planet_initial_x - 50.0
    dy_orb = state.planet_initial_y - 50.0
    orb_r = jnp.sqrt(dx_orb**2 + dy_orb**2 + 1e-8)
    init_angle = jnp.arctan2(dy_orb, dx_orb)
    cur_angle = init_angle + state.angular_velocity * state.tick
    
    new_planet_x = jnp.where(state.planet_is_orbiting, 50.0 + orb_r * jnp.cos(cur_angle), state.planet_x)
    new_planet_y = jnp.where(state.planet_is_orbiting, 50.0 + orb_r * jnp.sin(cur_angle), state.planet_y)
    
    # 2.7 Comets Spawning and Movement (Slots 46-49)
    wave_idx = state.tick // 100
    is_spawn_tick = (state.tick % 100 == 50) & (state.tick < 500)
    
    # Extract wave data safely
    safe_wave_idx = jnp.minimum(wave_idx, 4)
    c_sx = state.comet_starts_x[safe_wave_idx]
    c_sy = state.comet_starts_y[safe_wave_idx]
    c_dx = state.comet_dx[safe_wave_idx]
    c_dy = state.comet_dy[safe_wave_idx]
    c_ships = state.comet_ships[safe_wave_idx]
    
    new_p_dx = jnp.where(is_spawn_tick, state.planet_dx.at[46:50].set(c_dx), state.planet_dx)
    new_p_dy = jnp.where(is_spawn_tick, state.planet_dy.at[46:50].set(c_dy), state.planet_dy)
    new_planet_x = jnp.where(is_spawn_tick, new_planet_x.at[46:50].set(c_sx), new_planet_x)
    new_planet_y = jnp.where(is_spawn_tick, new_planet_y.at[46:50].set(c_sy), new_planet_y)
    new_planet_radius = jnp.where(is_spawn_tick, state.planet_radius.at[46:50].set(1.0), state.planet_radius)
    new_planet_production = jnp.where(is_spawn_tick, state.planet_production.at[46:50].set(1.0), state.planet_production)
    new_ships = jnp.where(is_spawn_tick, new_ships.at[46:50].set(c_ships), new_ships)
    new_owner = jnp.where(is_spawn_tick, state.planet_owner.at[46:50].set(0), state.planet_owner)
    
    # Move active comets
    comet_active = new_planet_radius[46:50] > 0.0
    new_planet_x = new_planet_x.at[46:50].add(jnp.where(comet_active, new_p_dx[46:50], 0.0))
    new_planet_y = new_planet_y.at[46:50].add(jnp.where(comet_active, new_p_dy[46:50], 0.0))
    
    # Comet OOB expiration
    comet_oob = (new_planet_x[46:50] < -5.0) | (new_planet_x[46:50] > 105.0) | (new_planet_y[46:50] < -5.0) | (new_planet_y[46:50] > 105.0)
    new_planet_radius = new_planet_radius.at[46:50].set(jnp.where(comet_oob, 0.0, new_planet_radius[46:50]))
    new_planet_production = new_planet_production.at[46:50].set(jnp.where(comet_oob, 0.0, new_planet_production[46:50]))
    new_ships = new_ships.at[46:50].set(jnp.where(comet_oob, 0.0, new_ships[46:50]))
    new_owner = new_owner.at[46:50].set(jnp.where(comet_oob, 0, new_owner[46:50]))
    
    # 3. Continuous Collision Resolution (Line vs Circle)
    # D vector (Fleet movement)
    dx_f = (new_fleet_x - old_fleet_x)[None, :]
    dy_f = (new_fleet_y - old_fleet_y)[None, :]
    
    # F vector (Old Fleet to Planet)
    px_matrix = new_planet_x[:, None]
    py_matrix = new_planet_y[:, None]
    fx = old_fleet_x[None, :] - px_matrix
    fy = old_fleet_y[None, :] - py_matrix
    
    # Quadratic coefficients
    a = dx_f**2 + dy_f**2
    b = 2.0 * (dx_f * fx + dy_f * fy)
    c = fx**2 + fy**2
    
    # Minimum distance occurs at t = -b / (2a)
    t = jnp.where(a > 0.0, -b / (2.0 * a), 0.0)
    t = jnp.clip(t, 0.0, 1.0)
    
    dist_sq = a * t**2 + b * t + c
    radius_sq = (new_planet_radius + 0.1)**2
    hit_matrix = dist_sq <= radius_sq[:, None]
    hit_matrix = hit_matrix & state.fleet_active[None, :]
    
    # CRITICAL STABILITY FIX: Prevent ship duplication on overlapping planets
    # A fleet can only hit ONE planet per tick. Find the closest valid hit.
    masked_dist = jnp.where(hit_matrix, dist_sq, jnp.inf)
    closest_planet = jnp.argmin(masked_dist, axis=0) # [F]
    closest_hit_mask = jnp.transpose(jax.nn.one_hot(closest_planet, 50, dtype=jnp.bool_)) # [50, F]
    hit_matrix = hit_matrix & closest_hit_mask
    
    # 4. Combat Resolution (Perfect Match to Kaggle)
    # We have 5 possible owners (0: neutral, 1: p1, 2: p2, 3: p3, 4: p4)
    num_owners = 5
    owners_one_hot = jax.nn.one_hot(state.fleet_owner, num_owners) # [F, 5]
    
    # Phase A: Fleet vs Fleet combat in orbit
    # For each planet p, and owner o, sum the incoming fleet ships
    incoming = jnp.einsum('pf,f,fo->po', hit_matrix, state.fleet_ships, owners_one_hot) # [N, 5]
    
    # Find the top 2 incoming fleets
    top2_incoming, top2_incoming_indices = jax.lax.top_k(incoming, 2)
    top1_fleet_ships = top2_incoming[:, 0]
    top2_fleet_ships = top2_incoming[:, 1]
    top1_fleet_owner = top2_incoming_indices[:, 0]
    
    # Survivor of the orbital battle
    surviving_fleet_ships = top1_fleet_ships - top2_fleet_ships
    surviving_fleet_owner = jnp.where(surviving_fleet_ships > 0, top1_fleet_owner, -1)
    
    # Phase B: Surviving Fleet vs Planet Garrison
    # Does the survivor match the planet owner?
    is_friendly_reinforcement = surviving_fleet_owner == new_owner
    
    final_ships = jnp.where(
        is_friendly_reinforcement,
        new_ships + surviving_fleet_ships,
        new_ships - surviving_fleet_ships
    )
    
    # If final_ships < 0, the planet was conquered
    final_owner = jnp.where(
        final_ships < 0,
        surviving_fleet_owner,
        new_owner
    )
    # If the planet is neutral and ties, it stays neutral
    final_owner = jnp.where(
        (new_owner == 0) & (final_ships == 0),
        0,
        final_owner
    )
    
    # NEW: Force ghost planets to remain neutral and empty
    is_ghost = (new_planet_radius == 0.0)
    final_owner = jnp.where(is_ghost, 0, final_owner)
    final_ships = jnp.where(is_ghost, 0.0, final_ships)
    
    final_ships = jnp.abs(final_ships)
    
    # 5. Remove dead fleets
    # A fleet dies if it hits any planet
    fleet_hit_any = jnp.any(hit_matrix, axis=0)
    new_fleet_active = state.fleet_active & (~fleet_hit_any)
    
    # Continuous sun collision
    sun_fx = old_fleet_x - 50.0
    sun_fy = old_fleet_y - 50.0
    sun_a = dx_f[0, :]**2 + dy_f[0, :]**2
    sun_b = 2.0 * (dx_f[0, :] * sun_fx + dy_f[0, :] * sun_fy)
    sun_c = sun_fx**2 + sun_fy**2
    sun_t = jnp.where(sun_a > 0.0, -sun_b / (2.0 * sun_a), 0.0)
    sun_t = jnp.clip(sun_t, 0.0, 1.0)
    sun_dist_sq = sun_a * sun_t**2 + sun_b * sun_t + sun_c
    new_fleet_active = new_fleet_active & (sun_dist_sq > 100.0)
    
    # Out of bounds check
    fleet_oob = (new_fleet_x < 0.0) | (new_fleet_x > 100.0) | (new_fleet_y < 0.0) | (new_fleet_y > 100.0)
    new_fleet_active = new_fleet_active & (~fleet_oob)
    
    new_state = state.replace(
        planet_x=new_planet_x,
        planet_y=new_planet_y,
        planet_dx=new_p_dx,
        planet_dy=new_p_dy,
        planet_radius=new_planet_radius,
        planet_production=new_planet_production,
        planet_ships=final_ships,
        planet_owner=final_owner,
        fleet_x=new_fleet_x,
        fleet_y=new_fleet_y,
        fleet_active=new_fleet_active,
        tick=state.tick + 1
    )
    return new_state

@jax.jit
def apply_actions(state: EnvState, player_id: int, launch: jnp.ndarray, target: jnp.ndarray, ships: jnp.ndarray) -> EnvState:
    """Spawns new fleets from valid planet launches."""
    valid_launch = launch & (state.planet_owner == player_id)
    
    def spawn_planet(carry, p_idx):
        st = carry
        is_valid = valid_launch[p_idx]
        
        bucket = ships[p_idx]
        # MECHANISTIC PATCH: Shift bucket from [0..9] to [0.1 .. 1.0]
        # This completely removes the "0 ships" loophole and allows 100% launches.
        fraction = (bucket + 1) / 10.0
        garrison = st.planet_ships[p_idx]
        send_ships = garrison * fraction
        
        send_ships = jnp.minimum(send_ships, garrison)
        is_valid = is_valid & (send_ships >= 1.0)
        
        empty_slot = jnp.argmin(st.fleet_active)
        has_empty = st.fleet_active[empty_slot] == 0
        is_valid = is_valid & has_empty
        
        target_idx = target[p_idx]
        speed = get_fleet_speed(send_ships)
        
        # --- PERFECT INTERCEPTION MATHEMATICS (STATIC VECTORIZATION) ---
        # Expanded from 150 to 500 to guarantee interception calculations for very slow fleets
        # traveling across the entire board, preventing them from 'missing' and flying out of bounds.
        t_arr = jnp.arange(1, 501, dtype=jnp.float32)
        
        # Calculate Target's orbital parameters
        dx_orb = st.planet_initial_x[target_idx] - 50.0
        dy_orb = st.planet_initial_y[target_idx] - 50.0
        orb_r = jnp.sqrt(dx_orb**2 + dy_orb**2 + 1e-8)
        init_angle = jnp.arctan2(dy_orb, dx_orb)
        
        a_arr = init_angle + st.angular_velocity * (st.tick + t_arr)
        px_orb_arr = 50.0 + orb_r * jnp.cos(a_arr)
        py_orb_arr = 50.0 + orb_r * jnp.sin(a_arr)
        
        px_lin_arr = st.planet_x[target_idx] + st.planet_dx[target_idx] * t_arr
        py_lin_arr = st.planet_y[target_idx] + st.planet_dy[target_idx] * t_arr
        
        tgt_is_orb = st.planet_is_orbiting[target_idx] > 0.5
        px_arr = jnp.where(tgt_is_orb, px_orb_arr, px_lin_arr)
        py_arr = jnp.where(tgt_is_orb, py_orb_arr, py_lin_arr)
        
        req_t_arr = jnp.hypot(px_arr - st.planet_x[p_idx], py_arr - st.planet_y[p_idx]) / speed
        valid_mask = req_t_arr <= t_arr
        
        # Find the first valid t index (fallback to index 0 if none)
        best_idx = jnp.argmax(valid_mask)
        t_approx = t_arr[best_idx]
        
        # --- CONTINUOUS INTERCEPT REFINEMENT (BINARY SEARCH) ---
        low_t = jnp.maximum(0.0, t_approx - 1.0)
        high_t = t_approx
        
        def refine_step(i, carry):
            low, high = carry
            mid = (low + high) / 2.0
            
            a_mid = init_angle + st.angular_velocity * (st.tick + mid)
            px_orb_mid = 50.0 + orb_r * jnp.cos(a_mid)
            py_orb_mid = 50.0 + orb_r * jnp.sin(a_mid)
            
            px_lin_mid = st.planet_x[target_idx] + st.planet_dx[target_idx] * mid
            py_lin_mid = st.planet_y[target_idx] + st.planet_dy[target_idx] * mid
            
            px_mid = jnp.where(tgt_is_orb, px_orb_mid, px_lin_mid)
            py_mid = jnp.where(tgt_is_orb, py_orb_mid, py_lin_mid)
            
            dist = jnp.hypot(px_mid - st.planet_x[p_idx], py_mid - st.planet_y[p_idx])
            # Kaggle engine physics: fleets spawn at edge + 0.1, and hit at target edge
            fleet_dist = speed * mid + st.planet_radius[p_idx] + 0.1 + st.planet_radius[target_idx]
            
            new_low = jnp.where(fleet_dist < dist, mid, low)
            new_high = jnp.where(fleet_dist < dist, high, mid)
            return (new_low, new_high)
            
        final_low, final_high = jax.lax.fori_loop(0, 15, refine_step, (low_t, high_t))
        best_t = final_high
        
        final_a = init_angle + st.angular_velocity * (st.tick + best_t)
        best_px = jnp.where(tgt_is_orb, 50.0 + orb_r * jnp.cos(final_a), st.planet_x[target_idx] + st.planet_dx[target_idx] * best_t)
        best_py = jnp.where(tgt_is_orb, 50.0 + orb_r * jnp.sin(final_a), st.planet_y[target_idx] + st.planet_dy[target_idx] * best_t)
        
        angle_rad = jnp.arctan2(best_py - st.planet_y[p_idx], best_px - st.planet_x[p_idx])
        
        # --- SUN COLLISION HARNESS ---
        sun_collision = check_sun_collision(st.planet_x[p_idx], st.planet_y[p_idx], best_px, best_py)
        is_valid = is_valid & (~sun_collision)
        
        dx = jnp.cos(angle_rad) * speed
        dy = jnp.sin(angle_rad) * speed
        
        # Spawn exactly like Kaggle: at edge + 0.1
        start_x = st.planet_x[p_idx] + jnp.cos(angle_rad) * (st.planet_radius[p_idx] + 0.1)
        start_y = st.planet_y[p_idx] + jnp.sin(angle_rad) * (st.planet_radius[p_idx] + 0.1)
        
        new_active = jnp.where(is_valid, st.fleet_active.at[empty_slot].set(1), st.fleet_active)
        new_f_owner = jnp.where(is_valid, st.fleet_owner.at[empty_slot].set(player_id), st.fleet_owner)
        new_f_ships = jnp.where(is_valid, st.fleet_ships.at[empty_slot].set(send_ships), st.fleet_ships)
        new_f_x = jnp.where(is_valid, st.fleet_x.at[empty_slot].set(start_x), st.fleet_x)
        new_f_y = jnp.where(is_valid, st.fleet_y.at[empty_slot].set(start_y), st.fleet_y)
        new_f_dx = jnp.where(is_valid, st.fleet_dx.at[empty_slot].set(dx), st.fleet_dx)
        new_f_dy = jnp.where(is_valid, st.fleet_dy.at[empty_slot].set(dy), st.fleet_dy)
        new_f_src = jnp.where(is_valid, st.fleet_src_planet.at[empty_slot].set(p_idx), st.fleet_src_planet)
        new_f_tgt = jnp.where(is_valid, st.fleet_target_planet.at[empty_slot].set(target_idx), st.fleet_target_planet)
        
        new_p_ships = jnp.where(is_valid, st.planet_ships.at[p_idx].set(garrison - send_ships), st.planet_ships)
        
        new_st = st.replace(
            fleet_active=new_active,
            fleet_owner=new_f_owner,
            fleet_ships=new_f_ships,
            fleet_x=new_f_x,
            fleet_y=new_f_y,
            fleet_dx=new_f_dx,
            fleet_dy=new_f_dy,
            fleet_src_planet=new_f_src,
            fleet_target_planet=new_f_tgt,
            planet_ships=new_p_ships
        )
        return new_st, None

    num_planets = state.planet_x.shape[0]
    final_state, _ = jax.lax.scan(spawn_planet, state, jnp.arange(num_planets))
    return final_state

@jax.jit
def build_observation_v1(state: EnvState, player_id: int, win_rate: float = 0.5) -> jnp.ndarray:
    """Builds the exact (70, 14) tensor expected by V1 EntityTransformer, from the perspective of player_id."""
    n_planets = state.planet_x.shape[0]
    
    p_tensor = jnp.zeros((50, 14))
    p_tensor = p_tensor.at[:, 0].set(-1.0)
    
    # Active planets & Local frame centering
    is_active = state.planet_radius > 0.0
    p_tensor = p_tensor.at[:, 0].set(jnp.where(is_active, 0.0, -1.0))
    
    # Filter ghosts for centering
    active_count = jnp.sum(is_active) + 1e-8
    center_x = jnp.sum(state.planet_x * is_active) / active_count
    center_y = jnp.sum(state.planet_y * is_active) / active_count
    rel_planet_x = state.planet_x - center_x
    rel_planet_y = state.planet_y - center_y

    norm_px = rel_planet_x / 50.0
    norm_py = rel_planet_y / 50.0
    norm_p_rad = state.planet_radius / 10.0
    norm_p_prod = state.planet_production / 5.0
    norm_p_ships = jnp.log1p(state.planet_ships) / 7.0

    p_tensor = p_tensor.at[:n_planets, 1].set(norm_px)
    p_tensor = p_tensor.at[:n_planets, 2].set(norm_py)
    p_tensor = p_tensor.at[:n_planets, 3].set(norm_p_rad)
    p_tensor = p_tensor.at[:n_planets, 4].set(norm_p_prod)
    
    # --- UNIVERSAL EGOCENTRIC MAPPING FOR N-PLAYERS ---
    def map_owner(raw_owner, p_id):
        return jnp.where(
            raw_owner == 0, 0.0,
            jnp.where(
                raw_owner == p_id, 1.0,
                jnp.where(raw_owner < p_id, raw_owner + 1.0, raw_owner.astype(jnp.float32))
            )
        )
    
    ego_planet_owner = map_owner(state.planet_owner, player_id)
    p_tensor = p_tensor.at[:n_planets, 5].set(ego_planet_owner[:n_planets])
    p_tensor = p_tensor.at[:n_planets, 6].set(norm_p_ships)

    # Static & Comet masks
    is_static = (1.0 - state.planet_is_orbiting).astype(jnp.float32)
    p_tensor = p_tensor.at[:n_planets, 7].set(is_static)
    
    # Feature 8: is_comet
    is_comet_slot = jnp.arange(50) >= 46
    p_tensor = p_tensor.at[:n_planets, 8].set(is_comet_slot.astype(jnp.float32)) 

    p_tensor = p_tensor.at[:n_planets, 9].set(state.tick / 1000.0)
    p_tensor = p_tensor.at[:n_planets, 10].set(win_rate)
    p_tensor = p_tensor.at[:n_planets, 11].set(state.angular_velocity)
    
    # NEW: Features 12, 13 for comet velocity (dx, dy)
    p_tensor = p_tensor.at[:n_planets, 12].set(state.planet_dx / 6.0)
    p_tensor = p_tensor.at[:n_planets, 13].set(state.planet_dy / 6.0)
    
    # --- FLEET OBSERVATIONS ---
    a_tensor = jnp.zeros((20, 14))
    a_tensor = a_tensor.at[:, 0].set(-2.0)
    
    # OPTIMIZATION: Use top_k to efficiently find the 20 largest active fleets
    fleet_scores = jnp.where(state.fleet_active > 0, state.fleet_ships, -1.0)
    _, sort_idx = jax.lax.top_k(fleet_scores, 20)
    f_active = state.fleet_active[sort_idx]
    f_owner = state.fleet_owner[sort_idx]
    f_ships = state.fleet_ships[sort_idx]
    f_x = state.fleet_x[sort_idx]
    f_y = state.fleet_y[sort_idx]
    f_dx = state.fleet_dx[sort_idx]
    f_dy = state.fleet_dy[sort_idx]
    f_src = state.fleet_src_planet[sort_idx]
    
    f_angle = jnp.arctan2(f_dy, f_dx)
    src_x = jnp.where(f_src < 50, state.planet_x[f_src] - center_x, 50.0 - center_x)
    src_y = jnp.where(f_src < 50, state.planet_y[f_src] - center_y, 50.0 - center_y)

    f_x_rel = f_x - center_x
    f_y_rel = f_y - center_y
    dist_flown = jnp.hypot(f_x_rel - src_x, f_y_rel - src_y)
    speed = get_fleet_speed(f_ships)
    ticks_flown = jnp.floor(dist_flown / speed)
    
    norm_fx = f_x_rel / 50.0
    norm_fy = f_y_rel / 50.0
    norm_f_ships = jnp.log1p(f_ships) / 7.0

    a_tensor = a_tensor.at[:, 0].set(jnp.where(f_active > 0, 1.0, -2.0))
    a_tensor = a_tensor.at[:, 1].set(norm_fx)
    a_tensor = a_tensor.at[:, 2].set(norm_fy)
    a_tensor = a_tensor.at[:, 3].set(norm_f_ships)
    
    # Egocentric mapping for fleets
    ego_f_owner = map_owner(f_owner, player_id)
    a_tensor = a_tensor.at[:, 4].set(ego_f_owner)
    
    a_tensor = a_tensor.at[:, 5].set(ticks_flown / 150.0)
    a_tensor = a_tensor.at[:, 6].set(f_angle / jnp.pi)
    a_tensor = a_tensor.at[:, 9].set(state.tick / 1000.0)
    a_tensor = a_tensor.at[:, 10].set(win_rate)
    a_tensor = a_tensor.at[:, 11].set(state.angular_velocity)
    a_tensor = a_tensor.at[:, 12].set(f_dx / 6.0)
    a_tensor = a_tensor.at[:, 13].set(f_dy / 6.0)
    
    obs = jnp.concatenate([p_tensor, a_tensor], axis=0)
    return obs

def check_sun_collision(src_x, src_y, tgt_x, tgt_y):
    """Vectorized check if a line segment crosses the Sun at (50, 50) with radius 10."""
    dx = tgt_x - src_x
    dy = tgt_y - src_y
    L2 = dx**2 + dy**2 + 1e-8
    
    fx = 50.0 - src_x
    fy = 50.0 - src_y
    
    t = (fx * dx + fy * dy) / L2
    t_clamped = jnp.clip(t, 0.0, 1.0)
    
    closest_x = src_x + t_clamped * dx
    closest_y = src_y + t_clamped * dy
    
    dist_sq = (closest_x - 50.0)**2 + (closest_y - 50.0)**2
    return dist_sq < 100.0

@jax.jit
def build_observation(state: EnvState, player_id: int, win_rate: float = 0.5) -> jnp.ndarray:
    """Builds the exact (50, 35) tensor expected by EntityTransformer, from the perspective of player_id."""
    n_planets = 50
    p_tensor = jnp.zeros((n_planets, 37), dtype=jnp.float32)
    p_tensor = p_tensor.at[:, 0].set(-1.0)
    
    # Active planets & Local frame centering
    is_active = state.planet_radius > 0.0
    p_tensor = p_tensor.at[:, 0].set(jnp.where(is_active, 0.0, -1.0))
    
    # Polar Coordinates from center (50, 50)
    dx = state.planet_x - 50.0
    dy = state.planet_y - 50.0
    polar_r = jnp.hypot(dx, dy) / 50.0
    theta = jnp.arctan2(dy, dx)
    
    # Player's center of mass for rotation invariance
    my_mask = (state.planet_owner == player_id) & is_active
    my_count = jnp.sum(my_mask)
    my_cx = jnp.sum(dx * my_mask) / (my_count + 1e-8)
    my_cy = jnp.sum(dy * my_mask) / (my_count + 1e-8)
    
    player_theta = jnp.where(my_count > 0, jnp.arctan2(my_cy, my_cx), 0.0)
    
    rel_theta = theta - player_theta
    # Wrap to [-pi, pi]
    rel_theta = (rel_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    polar_theta = rel_theta 

    norm_p_rad = state.planet_radius / 10.0
    norm_p_prod = state.planet_production / 5.0
    norm_p_ships = jnp.log1p(state.planet_ships) / 7.0

    p_tensor = p_tensor.at[:n_planets, 1].set(polar_r)
    p_tensor = p_tensor.at[:n_planets, 2].set(jnp.sin(polar_theta))
    p_tensor = p_tensor.at[:n_planets, 3].set(jnp.cos(polar_theta))
    p_tensor = p_tensor.at[:n_planets, 4].set(norm_p_rad)
    p_tensor = p_tensor.at[:n_planets, 5].set(norm_p_prod)
    
    # --- UNIVERSAL EGOCENTRIC MAPPING FOR N-PLAYERS ---
    def map_owner(raw_owner, p_id):
        return jnp.where(
            raw_owner == 0, 0.0,
            jnp.where(
                raw_owner == p_id, 1.0,
                jnp.where(raw_owner < p_id, raw_owner + 1.0, raw_owner.astype(jnp.float32))
            )
        )
    
    ego_planet_owner = map_owner(state.planet_owner, player_id)
    p_tensor = p_tensor.at[:n_planets, 6].set(ego_planet_owner[:n_planets])
    p_tensor = p_tensor.at[:n_planets, 7].set(norm_p_ships)

    # Static & Comet masks
    is_static = (1.0 - state.planet_is_orbiting).astype(jnp.float32)
    p_tensor = p_tensor.at[:n_planets, 8].set(is_static)
    
    is_comet_slot = jnp.arange(50) >= 46
    is_comet = is_comet_slot.astype(jnp.float32)
    p_tensor = p_tensor.at[:n_planets, 9].set(is_comet) 

    p_tensor = p_tensor.at[:n_planets, 10].set(state.tick / 1000.0)
    
    # Feature 11: True Local Angular Velocity (matches dataset_grain_v2)
    is_orbiting_mask = state.planet_is_orbiting
    is_comet_mask = (jnp.arange(50) >= 46).astype(jnp.float32)
    true_ang_vel = state.angular_velocity * is_orbiting_mask * (1.0 - is_comet_mask)
    p_tensor = p_tensor.at[:n_planets, 11].set(true_ang_vel)
    
    # Feature 12, 13 for comet velocity (dx, dy)
    p_tensor = p_tensor.at[:n_planets, 12].set(state.planet_dx / 6.0)
    p_tensor = p_tensor.at[:n_planets, 13].set(state.planet_dy / 6.0)
    
    # --- PLANET-CENTRIC FLEET AGGREGATION ---
    f_active = state.fleet_active
    f_ships = state.fleet_ships
    f_owner = state.fleet_owner
    f_target = state.fleet_target_planet
    f_x = state.fleet_x
    f_y = state.fleet_y
    
    # Expand properties to [F, 50] to map to all planets
    target_mask = jax.nn.one_hot(f_target, 50) # [F, 50]
    target_mask = target_mask * f_active[:, None] # Only count active fleets
    
    # Allied vs Enemy fleets for general aggregation
    ego_fleet_owner = map_owner(f_owner, player_id)
    
    # 4-Player Fleet Splitting (Features 14, 15, 16, 17)
    inc_1 = jnp.sum(f_ships[:, None] * (ego_fleet_owner == 1.0)[:, None] * target_mask, axis=0) # [50]
    inc_2 = jnp.sum(f_ships[:, None] * (ego_fleet_owner == 2.0)[:, None] * target_mask, axis=0)
    inc_3 = jnp.sum(f_ships[:, None] * (ego_fleet_owner == 3.0)[:, None] * target_mask, axis=0)
    inc_4 = jnp.sum(f_ships[:, None] * (ego_fleet_owner == 4.0)[:, None] * target_mask, axis=0)
    
    incoming_allied = inc_1
    incoming_enemy = inc_2 + inc_3 + inc_4
    
    # Calculate perfect interception ETA
    p_x_target = state.planet_x[f_target]
    p_y_target = state.planet_y[f_target]
    speed = get_fleet_speed(f_ships)
    
    t0_eta = jnp.hypot(p_x_target - f_x, p_y_target - f_y) / speed
    tgt_is_orb = state.planet_is_orbiting[f_target]
    dx_orb = state.planet_initial_x[f_target] - 50.0
    dy_orb = state.planet_initial_y[f_target] - 50.0
    orb_r = jnp.hypot(dx_orb, dy_orb)
    cur_angle_eta = jnp.arctan2(dy_orb, dx_orb) + state.angular_velocity * state.tick
    
    a1_eta = cur_angle_eta + state.angular_velocity * t0_eta
    px1_eta = jnp.where(tgt_is_orb > 0.5, 50.0 + orb_r * jnp.cos(a1_eta), p_x_target)
    py1_eta = jnp.where(tgt_is_orb > 0.5, 50.0 + orb_r * jnp.sin(a1_eta), p_y_target)
    t1_eta = jnp.hypot(px1_eta - f_x, py1_eta - f_y) / speed
    
    a2_eta = cur_angle_eta + state.angular_velocity * t1_eta
    px2_eta = jnp.where(tgt_is_orb > 0.5, 50.0 + orb_r * jnp.cos(a2_eta), p_x_target)
    py2_eta = jnp.where(tgt_is_orb > 0.5, 50.0 + orb_r * jnp.sin(a2_eta), p_y_target)
    
    # Sun occlusion check
    path_blocked = check_sun_collision(f_x, f_y, px2_eta, py2_eta)
    eta = jnp.where(path_blocked, jnp.inf, jnp.hypot(px2_eta - f_x, py2_eta - f_y) / speed) # [F]
    
    # Feature 18: Imminent Attacker ETA
    is_planet_neutral = (state.planet_owner == 0)[None, :]
    is_attacker = (ego_fleet_owner[:, None] != state.planet_owner[None, :]) | is_planet_neutral
    valid_attacker_eta = jnp.where(is_attacker & (f_active > 0)[:, None] & (target_mask > 0), eta[:, None], jnp.inf) # [F, 50]
    min_attacker_eta = jnp.min(valid_attacker_eta, axis=0) # [50]
    
    # Normalize features
    p_tensor = p_tensor.at[:, 14].set(jnp.log1p(inc_1) / 7.0)
    p_tensor = p_tensor.at[:, 15].set(jnp.log1p(inc_2) / 7.0)
    p_tensor = p_tensor.at[:, 16].set(jnp.log1p(inc_3) / 7.0)
    p_tensor = p_tensor.at[:, 17].set(jnp.log1p(inc_4) / 7.0)
    
    inv_attacker_eta = jnp.where(min_attacker_eta < jnp.inf, 1.0 / (1.0 + min_attacker_eta / 10.0), 0.0)
    p_tensor = p_tensor.at[:, 18].set(inv_attacker_eta)
    
    # Feature 19 & 20: Quadrant Binary Encodings
    p_tensor = p_tensor.at[:n_planets, 19].set((state.planet_x > 50.0).astype(jnp.float32))
    p_tensor = p_tensor.at[:n_planets, 20].set((state.planet_y > 50.0).astype(jnp.float32))
    
    # Feature 21: Comet Spawn Countdown Clock
    wave_idx = state.tick // 100
    next_spawn = (wave_idx * 100) + 50
    next_spawn = jnp.where(state.tick > next_spawn, next_spawn + 100, next_spawn)
    ticks_to_spawn = jnp.maximum(0, next_spawn - state.tick)
    comet_urgency = 1.0 - (ticks_to_spawn / 100.0)
    comet_urgency = jnp.where(state.tick > 450, 0.0, comet_urgency)
    p_tensor = p_tensor.at[:n_planets, 21].set(comet_urgency)
    
    # Feature 22 & 23: Global Economic & Military Share
    my_ships_total = jnp.sum(state.planet_ships * (ego_planet_owner == 1.0)) + jnp.sum(incoming_allied)
    all_ships_total = jnp.sum(state.planet_ships) + jnp.sum(incoming_allied) + jnp.sum(incoming_enemy) + 1e-8
    my_ship_share = my_ships_total / all_ships_total

    my_prod_total = jnp.sum(state.planet_production * (ego_planet_owner == 1.0))
    all_prod_total = jnp.sum(state.planet_production) + 1e-8
    my_prod_share = my_prod_total / all_prod_total

    p_tensor = p_tensor.at[:n_planets, 22].set(my_ship_share)
    p_tensor = p_tensor.at[:n_planets, 23].set(my_prod_share)
    
    # FEATURE 24: True Capture Cost (Net Garrison) via Timeline Simulation
    eta_matrix_all = jnp.where(target_mask > 0, eta[:, None], jnp.inf) # [F, 50]
    is_planet_neutral = (state.planet_owner == 0)[None, :] # [1, 50]
    is_same_owner = (state.fleet_owner[:, None] == state.planet_owner[None, :]) & ~is_planet_neutral # [F, 50]
    fleet_impact_matrix = jnp.where(is_same_owner, f_ships[:, None], -f_ships[:, None]) * target_mask # [F, 50]
    time_mask = eta_matrix_all[:, None, :] <= eta_matrix_all[None, :, :] # [F_j, F_i, 50]
    past_impacts = jnp.sum(fleet_impact_matrix[:, None, :] * time_mask, axis=0) # [F_i, 50]
    safe_eta = jnp.where(eta_matrix_all == jnp.inf, 0.0, eta_matrix_all)
    G_matrix = state.planet_ships[None, :] + state.planet_production[None, :] * safe_eta + past_impacts
    is_valid_eval = (target_mask > 0.5) & ~is_same_owner
    valid_G = jnp.where(is_valid_eval & (eta_matrix_all != jnp.inf), G_matrix, jnp.inf)
    min_G = jnp.min(valid_G, axis=0) # [50]
    
    true_capture_cost = jnp.where(min_G == jnp.inf, state.planet_ships, min_G)
    p_tensor = p_tensor.at[:n_planets, 24].set(jnp.clip(true_capture_cost / 100.0, -1.0, 1.0))
    
    # Feature 25: Enemy Proximity
    enemy_planet_mask = (ego_planet_owner >= 2.0)
    dx_mat = state.planet_x[:, None] - state.planet_x[None, :]
    dy_mat = state.planet_y[:, None] - state.planet_y[None, :]
    dist_mat = jnp.hypot(dx_mat, dy_mat)
    
    enemy_dist = jnp.where(enemy_planet_mask, dist_mat, jnp.inf)
    min_enemy_dist = jnp.min(enemy_dist, axis=1) # [50]
    p_tensor = p_tensor.at[:n_planets, 25].set(1.0 / (1.0 + min_enemy_dist / 10.0))
    
    # Feature 26: Sun Shadow Angular Width (matches dataset_grain_v2)
    dist_to_sun = jnp.hypot(state.planet_x - 50.0, state.planet_y - 50.0)
    safe_dist = jnp.maximum(dist_to_sun, 10.1)  # Prevent arcsin > 1
    angular_width = 2.0 * jnp.arcsin(10.0 / safe_dist)
    p_tensor = p_tensor.at[:n_planets, 26].set(angular_width / jnp.pi)

    # Feature 27: Threat Density (local, matches dataset_grain_v2 15-tick horizon approximation)
    enemy_ships_total = state.planet_ships * enemy_planet_mask
    density_local = jnp.sum(enemy_ships_total[None, :] * (dist_mat <= 20.0), axis=1)
    p_tensor = p_tensor.at[:n_planets, 27].set(jnp.clip(density_local / 200.0, 0.0, 1.0))
    
    # Feature 28: Economic Momentum (matches dataset_grain_v2)
    my_prod_f28 = jnp.sum(state.planet_production * (ego_planet_owner == 1.0))
    enemy_prod_f28 = jnp.sum(state.planet_production * (ego_planet_owner >= 2.0))
    net_production_advantage = my_prod_f28 - enemy_prod_f28
    p_tensor = p_tensor.at[:n_planets, 28].set(jnp.clip(net_production_advantage / 25.0, -1.0, 1.0))
    
    # Feature 29: Angular Convergence (matches dataset_grain_v2)
    is_orbiting_f29 = state.planet_is_orbiting
    dx_orb_f29 = state.planet_x - 50.0
    dy_orb_f29 = state.planet_y - 50.0
    orb_vx = -dy_orb_f29 * state.angular_velocity
    orb_vy = dx_orb_f29 * state.angular_velocity
    # Vector from each planet to player's center of mass
    my_mask_f29 = (state.planet_owner == player_id) & is_active
    my_count_f29 = jnp.sum(my_mask_f29)
    my_cx_f29 = jnp.sum((state.planet_x - 50.0) * my_mask_f29) / (my_count_f29 + 1e-8) + 50.0
    my_cy_f29 = jnp.sum((state.planet_y - 50.0) * my_mask_f29) / (my_count_f29 + 1e-8) + 50.0
    to_me_x = my_cx_f29 - state.planet_x
    to_me_y = my_cy_f29 - state.planet_y
    mag_v = jnp.hypot(orb_vx, orb_vy) + 1e-8
    mag_me = jnp.hypot(to_me_x, to_me_y) + 1e-8
    convergence = (orb_vx * to_me_x + orb_vy * to_me_y) / (mag_v * mag_me)
    p_tensor = p_tensor.at[:n_planets, 29].set(convergence * is_orbiting_f29)
    
    # FEATURE 30: The Endgame Horizon (True Remaining Yield)
    min_transit_ticks = 10.0 # Simplified constant
    remaining_ticks_after_arrival = jnp.maximum(0.0, 500.0 - state.tick - min_transit_ticks)
    true_remaining_yield = remaining_ticks_after_arrival * state.planet_production
    p_tensor = p_tensor.at[:n_planets, 30].set(true_remaining_yield / 2500.0)
    
    # FEATURE 31: Logistical Gravity (Territory Control)
    my_prod_array = state.planet_production * (ego_planet_owner == 1.0)
    enemy_prod_array = state.planet_production * (ego_planet_owner >= 2.0)
    dx_all = state.planet_x[:, None] - state.planet_x[None, :]
    dy_all = state.planet_y[:, None] - state.planet_y[None, :]
    dist_all = jnp.hypot(dx_all, dy_all) + 1.0
    my_gravity = jnp.sum(my_prod_array[None, :] / dist_all, axis=1)
    enemy_gravity = jnp.sum(enemy_prod_array[None, :] / dist_all, axis=1)
    p_tensor = p_tensor.at[:n_planets, 31].set(jnp.clip((my_gravity - enemy_gravity) / 10.0, -1.0, 1.0))
    
    # FEATURE 32: The "Kingmaker" (Leader Mask)
    # Correctly sum both planet and fleet ships using the EGOCENTRIC mapping
    ego_planet_owner_int = ego_planet_owner.astype(jnp.int32)
    ego_fleet_owner_int = ego_fleet_owner.astype(jnp.int32)
    
    total_ego_planet_ships = jnp.bincount(ego_planet_owner_int, weights=state.planet_ships, length=5)
    total_ego_fleet_ships = jnp.bincount(ego_fleet_owner_int * (f_active > 0), weights=f_ships, length=5)
    total_ego_ships = total_ego_planet_ships + total_ego_fleet_ships
    
    # We only care about the Kingmaker among enemies (Ego IDs 2, 3, 4)
    enemy_ships = total_ego_ships[2:5]
    leader_ego_id = jnp.argmax(enemy_ships) + 2 # Resolves to 2, 3, or 4
    
    # If all enemies have 0 ships, zero out the leader mask
    leader_ego_id = jnp.where(jnp.max(enemy_ships) > 0, leader_ego_id, 0)
    
    is_leader = (ego_planet_owner == leader_ego_id).astype(jnp.float32)
    p_tensor = p_tensor.at[:n_planets, 32].set(is_leader)
    
    # FEATURE 33: Safe Surplus (Exportable Economy)
    safe_surplus = jnp.maximum(0.0, jnp.minimum(state.planet_ships, min_G))
    p_tensor = p_tensor.at[:n_planets, 33].set((safe_surplus / 100.0) * (ego_planet_owner == 1.0))
    
    # FEATURE 34: The Evacuation Protocol (Doomed Planet Mask)
    deficit = jnp.where(min_G < 0, -min_G, 0.0)
    is_doomed_by_combat = (deficit > 0)
    evacuate_mask = is_doomed_by_combat & (ego_planet_owner == 1.0)
    p_tensor = p_tensor.at[:n_planets, 34].set(evacuate_mask.astype(jnp.float32))
    
    # FEATURE 35: The Cry For Help (Local Deficit Heatmap)
    my_planets_mask = (ego_planet_owner == 1.0)
    deficit_allied = deficit * my_planets_mask
    close_allies_mask = (dist_mat <= 40.0) * my_planets_mask[None, :]
    local_allied_deficit = jnp.sum(close_allies_mask * deficit_allied[None, :], axis=1)
    p_tensor = p_tensor.at[:n_planets, 35].set(jnp.clip(local_allied_deficit / 100.0, 0.0, 1.0))
    
    # FEATURE 36: Allied Proximity (Distance to P1's closest border)
    # This solves the "Targeting Closer Planets" issue by explicitly providing distance
    my_planet_mask = (ego_planet_owner == 1.0)
    my_dist = jnp.where(my_planet_mask, dist_mat, jnp.inf)
    min_my_dist = jnp.min(my_dist, axis=1) # [50]
    p_tensor = p_tensor.at[:n_planets, 36].set(1.0 / (1.0 + min_my_dist / 10.0))
    
    return p_tensor

def reset_env(rng):
    rng, r_groups, r_prod, r_ships, r_ang, r_home = jax.random.split(rng, 6)
    
    # 1. Decide number of groups (5 to 10)
    num_groups = jax.random.randint(r_groups, (), minval=5, maxval=11)
    
    # Generate up to 10 base planets in Q1 (x > 50, y > 50)
    rng, r_x, r_y = jax.random.split(rng, 3)
    base_x = jax.random.uniform(r_x, (10,), minval=55.0, maxval=95.0)
    base_y = jax.random.uniform(r_y, (10,), minval=55.0, maxval=95.0)
    
    # Apply 4-fold mirror symmetry
    px = jnp.concatenate([base_x, 100.0 - base_x, base_x, 100.0 - base_x])
    py = jnp.concatenate([base_y, base_y, 100.0 - base_y, 100.0 - base_y])
    
    # 2. Production (1 to 5) and Radius
    base_prod = jax.random.randint(r_prod, (10,), minval=1, maxval=6).astype(jnp.float32)
    prod = jnp.tile(base_prod, 4)
    radius = 1.0 + jnp.log(jnp.maximum(prod, 1.0))
    
    # 3. Ships (5 to 99)
    base_ships = jax.random.uniform(r_ships, (10,), minval=5.0, maxval=99.0)
    ships = jnp.tile(base_ships, 4)
    
    # 4. Mask out padded groups
    group_indices = jnp.tile(jnp.arange(10), 4)
    active_mask = (group_indices < num_groups)
    
    px = jnp.where(active_mask, px, 0.0)
    py = jnp.where(active_mask, py, 0.0)
    radius = jnp.where(active_mask, radius, 0.0)
    prod = jnp.where(active_mask, prod, 0.0)
    ships = jnp.where(active_mask, ships, 0.0)
    
    # Randomly choose between 2-player and 4-player game
    rng, r_mode = jax.random.split(rng)
    is_4p = jax.random.bernoulli(r_mode, 0.5)
    
    # Select one active group to be the home planets (0 to num_groups-1)
    home_group = jax.random.randint(r_home, (), minval=0, maxval=num_groups)
    
    # Randomize which quadrant P1 starts in for training diversity
    # Quadrant offsets: 0=Q1, 10=Q2, 20=Q3, 30=Q4
    rng, r_quad = jax.random.split(rng)
    quad_rotation = jax.random.randint(r_quad, (), minval=0, maxval=4) * 10
    offsets = jnp.array([0, 30, 10, 20])  # P1, P2, P3, P4 base offsets
    rotated = (offsets + quad_rotation) % 40
    
    p1_idx = rotated[0] + home_group
    p2_idx = rotated[1] + home_group
    p3_idx = rotated[2] + home_group
    p4_idx = rotated[3] + home_group
    
    owner = jnp.zeros(40, dtype=jnp.int32)
    owner = owner.at[p1_idx].set(1)
    owner = owner.at[p2_idx].set(2)
    owner = jnp.where(is_4p, owner.at[p3_idx].set(3), owner)
    owner = jnp.where(is_4p, owner.at[p4_idx].set(4), owner)
    
    ships = ships.at[p1_idx].set(10.0)
    ships = ships.at[p2_idx].set(10.0)
    ships = jnp.where(is_4p, ships.at[p3_idx].set(10.0), ships)
    ships = jnp.where(is_4p, ships.at[p4_idx].set(10.0), ships)
    
    # 6. Orbiting vs Static
    dist_to_center = jnp.sqrt((px - 50.0)**2 + (py - 50.0)**2)
    is_orbiting = ((dist_to_center + radius) < 50.0).astype(jnp.float32)
    is_orbiting = jnp.where(active_mask, is_orbiting, 0.0)
    
    ang_vel = jax.random.uniform(r_ang, (), minval=0.025, maxval=0.05)
    rng, r_dir = jax.random.split(rng, 2)
    ang_vel = ang_vel * jax.random.choice(r_dir, jnp.array([1.0, -1.0]))
    
    # 7. Pad to 50 for static JAX shapes (adding 6 empty ghost slots + 4 comet slots)
    pad_fn = lambda x, fill: jnp.concatenate([x, jnp.full((10,), fill, dtype=x.dtype)])
    px = pad_fn(px, 0.0)
    py = pad_fn(py, 0.0)
    p_dx = jnp.zeros(50, dtype=jnp.float32)
    p_dy = jnp.zeros(50, dtype=jnp.float32)
    radius = pad_fn(radius, 0.0)
    prod = pad_fn(prod, 0.0)
    ships = pad_fn(ships, 0.0)
    owner = pad_fn(owner, 0)
    is_orbiting = pad_fn(is_orbiting, 0.0)
    
    # 8. Pre-generate 5 waves of comets
    # Comets start at edges and fly through the board
    rng, rc_x, rc_y, rc_sh = jax.random.split(rng, 4)
    c_start_x = jax.random.choice(rc_x, jnp.array([-5.0, 105.0]), shape=(5,))
    c_start_y = jax.random.uniform(rc_y, shape=(5,), minval=20.0, maxval=80.0)
    c_target_x = jnp.where(c_start_x < 0, 105.0, -5.0)
    c_target_y = 100.0 - c_start_y
    
    c_angle = jnp.arctan2(c_target_y - c_start_y, c_target_x - c_start_x)
    c_dx_base = jnp.cos(c_angle) * 4.0
    c_dy_base = jnp.sin(c_angle) * 4.0
    
    # Apply 4-fold symmetry for comets
    c_sx = jnp.stack([c_start_x, 100.0 - c_start_x, c_start_x, 100.0 - c_start_x], axis=1)
    c_sy = jnp.stack([c_start_y, c_start_y, 100.0 - c_start_y, 100.0 - c_start_y], axis=1)
    c_dx = jnp.stack([c_dx_base, -c_dx_base, c_dx_base, -c_dx_base], axis=1)
    c_dy = jnp.stack([c_dy_base, c_dy_base, -c_dy_base, -c_dy_base], axis=1)
    
    c_sh_base = jax.random.uniform(rc_sh, shape=(5,), minval=1.0, maxval=99.0)
    c_ships = jnp.tile(c_sh_base[:, None], (1, 4))
    
    # Fleets
    f_active = jnp.zeros(200, dtype=jnp.int32)
    f_owner = jnp.zeros(200, dtype=jnp.int32)
    f_ships = jnp.zeros(200, dtype=jnp.float32)
    f_x = jnp.zeros(200, dtype=jnp.float32)
    f_y = jnp.zeros(200, dtype=jnp.float32)
    f_dx = jnp.zeros(200, dtype=jnp.float32)
    f_dy = jnp.zeros(200, dtype=jnp.float32)
    f_src = jnp.zeros(200, dtype=jnp.int32)
    f_target = jnp.zeros(200, dtype=jnp.int32)
    
    return EnvState(
        planet_x=px, planet_y=py, planet_initial_x=px, planet_initial_y=py,
        planet_is_orbiting=is_orbiting, angular_velocity=ang_vel,
        planet_dx=p_dx, planet_dy=p_dy,
        planet_radius=radius, planet_production=prod,
        planet_owner=owner, planet_ships=ships,
        comet_starts_x=c_sx, comet_starts_y=c_sy, comet_dx=c_dx, comet_dy=c_dy, comet_ships=c_ships,
        fleet_active=f_active, fleet_owner=f_owner, fleet_ships=f_ships,
        fleet_x=f_x, fleet_y=f_y, fleet_dx=f_dx, fleet_dy=f_dy, fleet_src_planet=f_src,
        fleet_target_planet=f_target,
        tick=jnp.array(0, dtype=jnp.int32)
    )