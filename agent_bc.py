import os
import sys
import jax
import jax.numpy as jnp
from flax import nnx
from flax import serialization
import numpy as np
import traceback
import math

# Adjust paths if running from subdirectories or Kaggle
try:
    from src.models.entity_transformer_flax_v2 import EntityTransformer
    from src.env_jax.orbit_env_v2 import EnvState, build_observation, apply_actions, step_physics
except ImportError:
    pass

import typing

USE_MCTX = False
MCTX_SIMULATIONS = 32
MCTX_CANDIDATES = 8

try:
    import mctx
    MCTX_AVAILABLE = True
except ImportError:
    MCTX_AVAILABLE = False
    USE_MCTX = False

class SearchEmbedding(typing.NamedTuple):
    env_state: EnvState
    candidate_actions: jnp.ndarray

def get_candidate_actions(l_logits, a_logits, s_logits, valid_mask, rng_key):
    l_act0 = (l_logits > 0.0) & valid_mask
    a_act0 = jnp.argmax(a_logits, axis=-1)
    s_act0 = jnp.argmax(s_logits, axis=-1)
    act0 = jnp.stack([l_act0, a_act0, s_act0], axis=-1)
    
    def sample_fn(key):
        k1, k2, k3 = jax.random.split(key, 3)
        l_prob = jax.nn.sigmoid(l_logits)
        l_act = (jax.random.uniform(k1, l_logits.shape) < l_prob) & valid_mask
        a_act = jax.random.categorical(k2, a_logits, axis=-1)
        s_act = jax.random.categorical(k3, s_logits, axis=-1)
        return jnp.stack([l_act, a_act, s_act], axis=-1)
        
    keys = jax.random.split(rng_key, MCTX_CANDIDATES - 1)
    act_sampled = jax.vmap(sample_fn)(keys)
    return jnp.concatenate([act0[None, ...], act_sampled], axis=0)

@jax.jit
def run_mctx_search(rng_key, env_state, player_id, graphdef, model_state):
    merged_model = nnx.merge(graphdef, model_state)
    
    obs = build_observation(env_state, player_id, win_rate=1.0)
    valid_mask = (env_state.planet_owner == player_id)[:50] & (env_state.planet_ships[:50] >= 1.0)
    _, l_logits, a_logits, s_logits, value = merged_model(obs[None], return_policy=True, valid_launch_mask=valid_mask[None])
    
    rng_key, subkey = jax.random.split(rng_key)
    candidates = get_candidate_actions(l_logits[0], a_logits[0], s_logits[0], valid_mask, subkey)
    
    embedding = SearchEmbedding(env_state, candidates)
    
    root = mctx.RootFnOutput(
        value=value[:, 0] if value.ndim > 1 else value,
        prior_logits=jnp.zeros((1, MCTX_CANDIDATES)),
        embedding=jax.tree_util.tree_map(lambda x: x[None, ...], embedding)
    )
    
    def single_step(k, a, emb):
        chosen_act = emb.candidate_actions[a]
        l_act = chosen_act[:, 0]
        target_act = chosen_act[:, 1]
        s_act = chosen_act[:, 2]
        
        next_env_state = apply_actions(emb.env_state, player_id, l_act, target_act, s_act)
        next_env_state = step_physics(next_env_state)
        
        n_obs = build_observation(next_env_state, player_id, win_rate=1.0)
        n_valid_mask = (next_env_state.planet_owner == player_id)[:50] & (next_env_state.planet_ships[:50] >= 1.0)
        
        _, n_l_logits, n_a_logits, n_s_logits, n_value = merged_model(n_obs[None], return_policy=True, valid_launch_mask=n_valid_mask[None])
        
        n_candidates = get_candidate_actions(n_l_logits[0], n_a_logits[0], n_s_logits[0], n_valid_mask, k)
        next_emb = SearchEmbedding(next_env_state, n_candidates)
        
        rec_out = mctx.RecurrentFnOutput(
            reward=jnp.zeros((), dtype=jnp.float32),
            discount=jnp.where(next_env_state.tick >= 500, 0.0, 1.0).astype(jnp.float32),
            prior_logits=jnp.zeros(MCTX_CANDIDATES, dtype=jnp.float32),
            value=n_value[0, 0] if n_value.ndim > 1 else n_value[0]
        )
        return rec_out, next_emb

    def recurrent_fn(params, key, action, emb):
        B = action.shape[0]
        keys = jax.random.split(key, B)
        return jax.vmap(single_step)(keys, action, emb)

    rng_key, search_key = jax.random.split(rng_key)
    policy_output = mctx.muzero_policy(
        params=None,
        rng_key=search_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=MCTX_SIMULATIONS,
        max_depth=10,
        qtransform=mctx.qtransform_by_parent_and_siblings
    )
    
    best_action_idx = policy_output.action[0]
    best_joint_action = candidates[best_action_idx]
    
    return best_joint_action[:, 0], best_joint_action[:, 1], best_joint_action[:, 2], policy_output

def get_speed(s):
    import math
    s = max(s, 1e-8)
    ratio = min(max(math.log(s) / math.log(1000.0), 0.0), 1.0)
    return 1.0 if s <= 1.0 else 1.0 + 5.0 * (ratio ** 1.5)

def check_sun_collision(src_x, src_y, tgt_x, tgt_y):
    import math
    dx = tgt_x - src_x
    dy = tgt_y - src_y
    L2 = dx**2 + dy**2
    if L2 < 1e-8:
        return False
    
    fx = 50.0 - src_x
    fy = 50.0 - src_y
    
    t = (fx * dx + fy * dy) / L2
    t_clamped = max(0.0, min(1.0, t))
    
    closest_x = src_x + t_clamped * dx
    closest_y = src_y + t_clamped * dy
    
    dist_sq = (closest_x - 50.0)**2 + (closest_y - 50.0)**2
    return dist_sq < 100.001

def get_intercept(src_x, src_y, target_id, speed, env_state, initial_x, initial_y, tick, angular_velocity):
    import math
    dist_to_sun = math.hypot(initial_x[target_id] - 50.0, initial_y[target_id] - 50.0)
    is_orbital = 1.0 < dist_to_sun < 49.0
    
    tgt_x = float(env_state.planet_x[target_id])
    tgt_y = float(env_state.planet_y[target_id])
    
    if not is_orbital:
        return math.atan2(tgt_y - src_y, tgt_x - src_x), math.hypot(tgt_x - src_x, tgt_y - src_y) / speed, tgt_x, tgt_y
        
    init_px = initial_x[target_id]
    init_py = initial_y[target_id]
    init_angle = math.atan2(init_py - 50.0, init_px - 50.0)
    
    best_t = float('inf')
    best_angle = 0.0
    best_px = tgt_x
    best_py = tgt_y
    
    for t_sim in range(1, 2000):
        a = init_angle + angular_velocity * (tick + t_sim)
        px = 50.0 + dist_to_sun * math.cos(a)
        py = 50.0 + dist_to_sun * math.sin(a)
        
        req_t = math.hypot(px - src_x, py - src_y) / speed
        if req_t <= float(t_sim):
            best_t = req_t
            best_angle = math.atan2(py - src_y, px - src_x)
            best_px = px
            best_py = py
            break
            
    if best_t == float('inf'):
        best_t = math.hypot(tgt_x - src_x, tgt_y - src_y) / speed
        best_angle = math.atan2(tgt_y - src_y, tgt_x - src_x)
        
    return best_angle, best_t, best_px, best_py

def parse_kaggle_obs(obs_dict, raw_player=0, initial_x=None, initial_y=None):
    """Safely extracts Kaggle's observation lists and pads them to JAX static shapes."""
    raw_planets = obs_dict.get('planets', [])
    raw_fleets = obs_dict.get('fleets', [])
    
    p_x, p_y, p_radius, p_prod, p_owner, p_ships, p_orbiting = ([0.0]*50 for _ in range(7))
    p_owner = [-1] * 50
    
    for p in raw_planets:
        pid, owner, x, y, radius, ships, prod = p
        pid = int(pid)
        p_x[pid] = float(x)
        p_y[pid] = float(y)
        p_radius[pid] = float(radius)
        p_owner[pid] = int(owner)
        p_ships[pid] = float(ships)
        p_prod[pid] = float(prod)
        
        dist = ((float(x) - 50.0)**2 + (float(y) - 50.0)**2)**0.5
        if 1.0 < dist < 49.0:
            p_orbiting[pid] = 1.0

    f_active, f_owner, f_ships, f_x, f_y, f_dx, f_dy, f_src = [], [], [], [], [], [], [], []
    
    for f in raw_fleets:
        fid, owner, x, y, angle, src, ships = f
        f_active.append(1)
        f_owner.append(int(owner))
        f_ships.append(float(ships))
        f_x.append(float(x))
        f_y.append(float(y))
        f_src.append(int(src))
        
        import math
        speed = get_speed(float(ships))
        f_dx.append(float(math.cos(float(angle)) * speed))
        f_dy.append(float(math.sin(float(angle)) * speed))
        
    f_target = []
    angular_velocity = float(obs_dict.get('angular_velocity', 0.02))
    tick = int(obs_dict.get('step', 0))
    
    # Raycast Fleet Targets (Fundamental Fix)
    for f in raw_fleets:
        fid, owner, fx, fy, fangle, src, ships = f
        speed = get_speed(float(ships))
        dx_dir = math.cos(fangle)
        dy_dir = math.sin(fangle)
        
        best_t = float('inf')
        best_p = 0 # Default to 0 if no target found (shouldn't happen)
        
        for p in raw_planets:
            pid, _, px, py, radius, _, _ = p
            pid = int(pid)
            radius = float(radius)
            
            # Check if orbital
            init_px = initial_x[pid] if initial_x is not None else float(px)
            init_py = initial_y[pid] if initial_y is not None else float(py)
            dist_to_sun = math.hypot(init_px - 50.0, init_py - 50.0)
            
            is_orbital = dist_to_sun > 1.0 and dist_to_sun < 49.0
            
            if not is_orbital:
                # Static intersection
                proj = (float(px) - float(fx)) * dx_dir + (float(py) - float(fy)) * dy_dir
                if proj >= 0:
                    perp_sq = (float(px) - float(fx))**2 + (float(py) - float(fy))**2 - proj**2
                    if perp_sq < radius**2:
                        hit_dist = proj - math.sqrt(radius**2 - perp_sq)
                        t = max(0.0, hit_dist / speed)
                        if t < best_t:
                            best_t = t
                            best_p = pid
            else:
                # Orbital forward simulation
                # Find intersection by simulating up to 2000 ticks
                orb_r = dist_to_sun
                init_angle = math.atan2(init_py - 50.0, init_px - 50.0)
                
                # We simulate forward turn by turn
                for t in range(1, 2001):
                    if float(t) > best_t:
                        break # Already found a faster hit
                    
                    cur_angle = init_angle + angular_velocity * (tick + t)
                    pos_x = 50.0 + orb_r * math.cos(cur_angle)
                    pos_y = 50.0 + orb_r * math.sin(cur_angle)
                    
                    proj = (pos_x - float(fx)) * dx_dir + (pos_y - float(fy)) * dy_dir
                    if proj >= 0:
                        perp_sq = (pos_x - float(fx))**2 + (pos_y - float(fy))**2 - proj**2
                        if perp_sq < radius**2:
                            hit_dist = proj - math.sqrt(radius**2 - perp_sq)
                            hit_t = hit_dist / speed
                            # If the fleet arrives at this position around time t
                            if abs(hit_t - t) <= 1.5:
                                if hit_t < best_t:
                                    best_t = hit_t
                                    best_p = pid
                                break
        f_target.append(best_p)

    def pad_array(arr, target_len, dtype=jnp.float32):
        arr = jnp.array(arr, dtype=dtype)
        if len(arr) < target_len:
            padding = jnp.zeros((target_len - len(arr),), dtype=dtype)
            return jnp.concatenate([arr, padding])
        return arr[:target_len]

    def normalize_owners(arr, ego_id):
        if len(arr) == 0: return jnp.array([], dtype=jnp.int32)
        arr = jnp.array(arr, dtype=jnp.float32)
        new_owner = jnp.where(
            arr == -1.0, 0.0,
            jnp.where(
                arr == float(ego_id), 1.0,
                jnp.where(arr < float(ego_id), arr + 2.0, arr + 1.0)
            )
        )
        return new_owner.astype(jnp.int32)

    return EnvState(
        planet_x=pad_array(p_x, 50),
        planet_y=pad_array(p_y, 50),
        planet_radius=pad_array(p_radius, 50), 
        planet_production=pad_array(p_prod, 50),
        planet_owner=pad_array(normalize_owners(p_owner, raw_player), 50, dtype=jnp.int32),
        planet_ships=pad_array(p_ships, 50),
        
        fleet_active=pad_array(f_active, 200, dtype=jnp.int32),
        fleet_owner=pad_array(normalize_owners(f_owner, raw_player), 200, dtype=jnp.int32),
        fleet_ships=pad_array(f_ships, 200),
        fleet_x=pad_array(f_x, 200),
        fleet_y=pad_array(f_y, 200),
        fleet_dx=pad_array(f_dx, 200),
        fleet_dy=pad_array(f_dy, 200),
        fleet_src_planet=pad_array(f_src, 200, dtype=jnp.int32),
        fleet_target_planet=pad_array(f_target, 200, dtype=jnp.int32), # Target Raycasted
        
        planet_initial_x=pad_array(p_x, 50),
        planet_initial_y=pad_array(p_y, 50),
        planet_is_orbiting=pad_array(p_orbiting, 50),
        
        comet_starts_x=jnp.zeros((5, 4), dtype=jnp.float32),
        comet_starts_y=jnp.zeros((5, 4), dtype=jnp.float32),
        comet_dx=jnp.zeros((5, 4), dtype=jnp.float32),
        comet_dy=jnp.zeros((5, 4), dtype=jnp.float32),
        comet_ships=jnp.zeros((5, 4), dtype=jnp.float32),
        
        planet_dx=jnp.zeros(50, dtype=jnp.float32),
        planet_dy=jnp.zeros(50, dtype=jnp.float32),
        
        angular_velocity=jnp.array(obs_dict.get('angular_velocity', 0.0)),
        tick=jnp.array(0, dtype=jnp.int32) 
    )

class BCAgent:
    def __init__(self):
        # Update to 37 features!
        self.model = EntityTransformer(num_features=37, num_classes=5, rngs=nnx.Rngs(0))
        self.graphdef, self.state = nnx.split(self.model)
        
        import inspect
        base_dir = os.path.dirname(os.path.abspath(inspect.getfile(lambda: None))) if '__file__' not in globals() else os.path.dirname(__file__)
        if not os.path.exists(os.path.join(base_dir, "checkpoints")):
            base_dir = os.getcwd()
            
        ckpt_path = os.path.join(base_dir, "checkpoints", "ppo_v2_3050.bin")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing Checkpoint: {ckpt_path}")
            
        with open(ckpt_path, "rb") as f:
            raw_bytes = f.read()
            
        state_flat = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, self.state)
        self.state_restored = serialization.from_bytes(state_flat, raw_bytes)
        
        self.merged_model = nnx.merge(self.graphdef, self.state_restored)
        self.rng = jax.random.PRNGKey(42)
        
        # Initialize log file
        self.log_file = os.path.join(base_dir, "agent_brain.log")
        # Don't overwrite the log file here, local_kaggle_eval clears it
            
        self.initial_x = None
        self.initial_y = None
        
        self.prev_fleets = {}
        self.prev_planets = {}

    def __call__(self, obs):
        try:
            self.rng, rng_sample = jax.random.split(self.rng)
            obs_dict = obs if isinstance(obs, dict) else obs.__dict__
            
            raw_player = obs_dict.get('player', 0)
            player_id = raw_player + 1 if raw_player == 0 or raw_player == 1 else raw_player
            
            tick = obs_dict.get('step', 0)
            
            if self.initial_x is None or tick == 0:
                # Capture initial coordinates for orbital math
                self.initial_x = [0.0]*50
                self.initial_y = [0.0]*50
                for p in obs_dict.get('planets', []):
                    pid = int(p[0])
                    self.initial_x[pid] = float(p[2])
                    self.initial_y[pid] = float(p[3])
                    
            # Event Tracking
            curr_fleets = {f[0]: f for f in obs_dict.get('fleets', [])}
            curr_planets = {p[0]: p for p in obs_dict.get('planets', [])}
            
            if tick > 0:
                events = []
                for fid, f in curr_fleets.items():
                    if fid not in self.prev_fleets:
                        events.append(f"[EVENT: LAUNCH] P{f[1]} launched {f[6]} ships from Planet {f[5]} (Fleet {fid})")
                        
                for fid, prev_f in self.prev_fleets.items():
                    if fid not in curr_fleets:
                        events.append(f"[EVENT: FLEET_END] Fleet {fid} ({prev_f[6]} ships of P{prev_f[1]}) disappeared.")
                
                for pid, curr_p in curr_planets.items():
                    prev_p = self.prev_planets.get(pid)
                    if prev_p:
                        if curr_p[1] != prev_p[1]:
                            events.append(f"[EVENT: CAPTURE] Planet {pid} captured by P{curr_p[1]} (from P{prev_p[1]}) with {curr_p[5]} ships!")
                        elif curr_p[5] < prev_p[5] - 2.0: # Filter natural decay/growth
                            events.append(f"[EVENT: COMBAT] Planet {pid} attacked! Ships dropped from {prev_p[5]} to {curr_p[5]}.")
                
                if events:
                    with open(self.log_file, "a") as logf:
                        logf.write(f"\n--- TICK {tick} EVENTS ---\n")
                        for e in events:
                            logf.write(f"{e}\n")
                            
            self.prev_fleets = curr_fleets
            self.prev_planets = curr_planets
            
            # 1. Feature Extraction entirely in JAX using exact mathematical formulas
            env_state = parse_kaggle_obs(obs_dict, raw_player=raw_player, initial_x=self.initial_x, initial_y=self.initial_y)
            # The network was trained with player_id=1.0 for Ego! So we must pass 1!
            obs_tensor = build_observation(env_state, player_id=1, win_rate=1.0)[None, ...]
            
            # 2. Build Validity Mask (Ego is ALWAYS 1 in env_state now)
            my_planets_mask = (env_state.planet_owner == 1)[:50]
            valid_launch_mask = my_planets_mask & (env_state.planet_ships[:50] >= 1.0)
            
            if USE_MCTX:
                try:
                    state_flat = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, self.state_restored)
                    self.rng, search_rng = jax.random.split(self.rng)
                    import time
                    t0 = time.time()
                    l_acts, t_acts, s_acts, policy_output = run_mctx_search(search_rng, env_state, player_id, self.graphdef, state_flat)
                    t1 = time.time()
                    
                    l_acts = np.array(l_acts)
                    t_acts = np.array(t_acts)
                    s_acts = np.array(s_acts)
                    
                    actions = []
                    for src_idx in range(50):
                        if l_acts[src_idx] and my_planets_mask[src_idx]:
                            tgt_idx = int(t_acts[src_idx])
                            s_idx = int(s_acts[src_idx])
                            
                            ship_pct = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][min(s_idx, 9)]
                            available = float(env_state.planet_ships[src_idx])
                            ships_to_send = max(1, int(available * ship_pct))
                            
                            actions.append(f"SHIP {src_idx} {ships_to_send} {tgt_idx}")
                            
                    with open(self.log_file, "a") as f:
                        f.write(f"\n[TICK {tick}] Player {player_id} MCTX Search Complete in {t1-t0:.3f}s\n")
                        f.write(f"  Selected Candidate Index: {policy_output.action[0]}\n")
                        f.write(f"  Actions generated: {actions}\n")
                            
                    return actions

                except Exception as e:
                    with open(self.log_file, "a") as f:
                        f.write(f"MCTX CRASH: {e}\\n")
                    # Fallback to empty if MCTX fails
                    return []
            else:
                # RAW MODEL EXECUTION (No Search)
                _, l_logits, a_logits, s_logits, _ = self.merged_model(obs_tensor, return_policy=True, valid_launch_mask=valid_launch_mask[None])
                
                TEMPERATURE = 2.0
                LAUNCH_PENALTY = 0.0 # Higher penalty makes launch less likely
                
                self.rng, k1, k2, k3 = jax.random.split(self.rng, 4)
                
                # Temperature scaled sampling
                l_prob = jax.nn.sigmoid((l_logits[0] - LAUNCH_PENALTY) / TEMPERATURE)
                l_acts = (jax.random.uniform(k1, l_logits[0].shape) < l_prob) & valid_launch_mask
                
                with open(self.log_file, "a") as f:
                    f.write(f"  l_logits max: {np.max(l_logits[0]):.3f}, min: {np.min(l_logits[0]):.3f}\n")
                    f.write(f"  l_prob max: {np.max(l_prob):.3f}\n")
                    f.write(f"  Valid planets: {np.sum(valid_launch_mask)}\n")
                
                t_acts = jax.random.categorical(k2, a_logits[0] / TEMPERATURE, axis=-1)
                s_acts = jax.random.categorical(k3, s_logits[0] / TEMPERATURE, axis=-1)
                
                l_acts = np.array(l_acts)
                t_acts = np.array(t_acts)
                s_acts = np.array(s_acts)
                
                actions = []
                for src_idx in range(50):
                    if l_acts[src_idx] and my_planets_mask[src_idx]:
                        tgt_idx = int(t_acts[src_idx])
                        s_idx = int(s_acts[src_idx])
                        
                        ship_pct = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][min(s_idx, 9)]
                        available = float(env_state.planet_ships[src_idx])
                        ships_to_send = max(1, int(available * ship_pct))
                        
                        actions.append(f"SHIP {src_idx} {ships_to_send} {tgt_idx}")
                        
                with open(self.log_file, "a") as f:
                    f.write(f"\n[TICK {tick}] Player {player_id} RAW MODEL EXECUTION (Temp={TEMPERATURE}, Penalty={LAUNCH_PENALTY})\n")
                    f.write(f"  Actions generated: {actions}\n")
                        
                return actions

            
        except Exception as e:
            with open(self.log_file, "a") as f:
                f.write(f"CRITICAL ERROR in agent: {e}\\n")
            return []

agent_instance = BCAgent()

def agent(obs):
    return agent_instance(obs)