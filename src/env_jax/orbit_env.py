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
        
        # --- PERFECT INTERCEPTION MATHEMATICS ---
        # Step 0: Current Distance approximation
        dx0 = st.planet_x[target_idx] - st.planet_x[p_idx]
        dy0 = st.planet_y[target_idx] - st.planet_y[p_idx]
        t0 = jnp.sqrt(dx0**2 + dy0**2 + 1e-8) / speed
        
        # Calculate Target's orbital parameters
        dx_orb = st.planet_initial_x[target_idx] - 50.0
        dy_orb = st.planet_initial_y[target_idx] - 50.0
        orb_r = jnp.sqrt(dx_orb**2 + dy_orb**2 + 1e-8)
        init_angle = jnp.arctan2(dy_orb, dx_orb)
        cur_angle = init_angle + st.angular_velocity * st.tick
        
        # Step 1: Future position after t0
        a1 = cur_angle + st.angular_velocity * t0
        px_orb1 = 50.0 + orb_r * jnp.cos(a1)
        py_orb1 = 50.0 + orb_r * jnp.sin(a1)
        px_lin1 = st.planet_x[target_idx] + st.planet_dx[target_idx] * t0
        py_lin1 = st.planet_y[target_idx] + st.planet_dy[target_idx] * t0
        px1 = jnp.where(st.planet_is_orbiting[target_idx] > 0.5, px_orb1, px_lin1)
        py1 = jnp.where(st.planet_is_orbiting[target_idx] > 0.5, py_orb1, py_lin1)
        
        t1 = jnp.sqrt((px1 - st.planet_x[p_idx])**2 + (py1 - st.planet_y[p_idx])**2 + 1e-8) / speed
        
        # Step 2: Refined Future position after t1
        a2 = cur_angle + st.angular_velocity * t1
        px_orb2 = 50.0 + orb_r * jnp.cos(a2)
        py_orb2 = 50.0 + orb_r * jnp.sin(a2)
        px_lin2 = st.planet_x[target_idx] + st.planet_dx[target_idx] * t1
        py_lin2 = st.planet_y[target_idx] + st.planet_dy[target_idx] * t1
        px2 = jnp.where(st.planet_is_orbiting[target_idx] > 0.5, px_orb2, px_lin2)
        py2 = jnp.where(st.planet_is_orbiting[target_idx] > 0.5, py_orb2, py_lin2)
        
        angle_rad = jnp.arctan2(py2 - st.planet_y[p_idx], px2 - st.planet_x[p_idx])
        
        dx = jnp.cos(angle_rad) * speed
        dy = jnp.sin(angle_rad) * speed
        
        new_active = jnp.where(is_valid, st.fleet_active.at[empty_slot].set(1), st.fleet_active)
        new_f_owner = jnp.where(is_valid, st.fleet_owner.at[empty_slot].set(player_id), st.fleet_owner)
        new_f_ships = jnp.where(is_valid, st.fleet_ships.at[empty_slot].set(send_ships), st.fleet_ships)
        new_f_x = jnp.where(is_valid, st.fleet_x.at[empty_slot].set(st.planet_x[p_idx]), st.fleet_x)
        new_f_y = jnp.where(is_valid, st.fleet_y.at[empty_slot].set(st.planet_y[p_idx]), st.fleet_y)
        new_f_dx = jnp.where(is_valid, st.fleet_dx.at[empty_slot].set(dx), st.fleet_dx)
        new_f_dy = jnp.where(is_valid, st.fleet_dy.at[empty_slot].set(dy), st.fleet_dy)
        new_f_src = jnp.where(is_valid, st.fleet_src_planet.at[empty_slot].set(p_idx), st.fleet_src_planet)
        
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
            planet_ships=new_p_ships
        )
        return new_st, None

    num_planets = state.planet_x.shape[0]
    final_state, _ = jax.lax.scan(spawn_planet, state, jnp.arange(num_planets))
    return final_state

@jax.jit
def build_observation(state: EnvState, player_id: int, win_rate: float = 0.5) -> jnp.ndarray:
    """Builds the exact (70, 14) tensor expected by EntityTransformer, from the perspective of player_id."""
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
    
    p1_idx = home_group
    p2_idx = 30 + home_group # Q4
    p3_idx = 10 + home_group # Q2
    p4_idx = 20 + home_group # Q3
    
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
    
    return EnvState(
        planet_x=px, planet_y=py, planet_initial_x=px, planet_initial_y=py,
        planet_is_orbiting=is_orbiting, angular_velocity=ang_vel,
        planet_dx=p_dx, planet_dy=p_dy,
        planet_radius=radius, planet_production=prod,
        planet_owner=owner, planet_ships=ships,
        comet_starts_x=c_sx, comet_starts_y=c_sy, comet_dx=c_dx, comet_dy=c_dy, comet_ships=c_ships,
        fleet_active=f_active, fleet_owner=f_owner, fleet_ships=f_ships,
        fleet_x=f_x, fleet_y=f_y, fleet_dx=f_dx, fleet_dy=f_dy, fleet_src_planet=f_src,
        tick=jnp.array(0, dtype=jnp.int32)
    )