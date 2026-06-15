import sys
import os
import traceback
import jax
import jax.numpy as jnp
from flax import nnx
from flax import serialization
import mctx

# --- UNBUFFERED LOGGING ---
class Unbuffered:
    def __init__(self, stream): self.stream = stream
    def write(self, data): self.stream.write(data); self.stream.flush()
    def writelines(self, datas): self.stream.writelines(datas); self.stream.flush()
    def __getattr__(self, attr): return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout); sys.stderr = Unbuffered(sys.stderr)

# --- PATH RESOLUTION FOR KAGGLE ---
KAGGLE_PATH = "/kaggle_simulations/agent/"
if os.path.exists(KAGGLE_PATH):
    sys.path.insert(0, KAGGLE_PATH)
    BASE_DIR = KAGGLE_PATH
else:
    # Safely handle execution via exec() where __file__ is undefined
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.path.abspath(".")
    
    # Ensure the base directory is in sys.path so local imports work
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

from src.models.entity_transformer_flax_v2 import EntityTransformer
from env_jax.orbit_env_v2 import EnvState, build_observation

# --- AGENT RUNTIME STATE ---
class AgentEngine:
    def __init__(self):
        self.compiled_step = None
        self.ready = False

GLOBAL_AGENT = AgentEngine()

def parse_kaggle_obs(obs):
    """Safely extracts Kaggle's observation lists and pads them to JAX static shapes."""
    
    # Handle Kaggle environment wrapping
    obs_dict = obs if isinstance(obs, dict) else obs.__dict__
    
    raw_planets = obs_dict.get('planets', [])
    raw_fleets = obs_dict.get('fleets', [])
    
    # 1. Parse Planets: [id, owner, x, y, radius, ships, production]
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
        
        # Planets with a distance > 1 and < 49 from the sun (50,50) are rotating
        dist = ((float(x) - 50.0)**2 + (float(y) - 50.0)**2)**0.5
        if 1.0 < dist < 49.0:
            p_orbiting[pid] = 1.0

    # 2. Parse Fleets: [id, owner, x, y, angle, from_planet_id, ships]
    f_active, f_owner, f_ships, f_x, f_y, f_dx, f_dy, f_src = [], [], [], [], [], [], [], []
    
    # Port the JAX get_speed logic locally so we can calculate dx/dy
    def get_speed(s):
        import math
        s = max(s, 1e-8)
        ratio = min(max(math.log(s) / math.log(1000.0), 0.0), 1.0)
        return 1.0 if s <= 1.0 else 1.0 + 5.0 * (ratio ** 1.5)
        
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

    # Helper padding functions
    def pad_array(arr, target_len, dtype=jnp.float32):
        arr = jnp.array(arr, dtype=dtype)
        if len(arr) < target_len:
            padding = jnp.zeros((target_len - len(arr),), dtype=dtype)
            return jnp.concatenate([arr, padding])
        return arr[:target_len]

    def normalize_owners(arr):
        if len(arr) == 0: return jnp.array([], dtype=jnp.int32)
        arr = jnp.array(arr)
        # Shift Kaggle -1/0/1 to JAX 0/1/2
        if jnp.min(arr) == -1 or (jnp.max(arr) <= 1 and jnp.min(arr) == 0):
            return arr + 1
        return arr

    return EnvState(
        planet_x=pad_array(p_x, 50),
        planet_y=pad_array(p_y, 50),
        planet_radius=pad_array(p_radius, 50), 
        planet_production=pad_array(p_prod, 50),
        planet_owner=pad_array(normalize_owners(p_owner), 50, dtype=jnp.int32),
        planet_ships=pad_array(p_ships, 50),
        
        fleet_active=pad_array(f_active, 200, dtype=jnp.int32),
        fleet_owner=pad_array(normalize_owners(f_owner), 200, dtype=jnp.int32),
        fleet_ships=pad_array(f_ships, 200),
        fleet_x=pad_array(f_x, 200),
        fleet_y=pad_array(f_y, 200),
        fleet_dx=pad_array(f_dx, 200),
        fleet_dy=pad_array(f_dy, 200),
        fleet_src_planet=pad_array(f_src, 200, dtype=jnp.int32),
        
        # We set initial_x/y to the CURRENT position and set tick to 0.
        # This guarantees the Beam Search physics step rotates it perfectly by 1 tick.
        planet_initial_x=pad_array(p_x, 50),
        planet_initial_y=pad_array(p_y, 50),
        planet_is_orbiting=pad_array(p_orbiting, 50),
        
        angular_velocity=jnp.array(obs_dict.get('angular_velocity', 0.0)),
        tick=jnp.array(0, dtype=jnp.int32) 
    )

# --- STRATEGY TOGGLE ---
# Options: "GREEDY", "BEAM_SEARCH", "MCTS"
STRATEGY = "MCTS"  

def initialize_agent():
    print(f"Initializing Model... (Strategy: {'BEAM SEARCH' if USE_BEAM_SEARCH else 'GREEDY POLICY'})")
    
    # 1. Load Blueprint
    rngs = nnx.Rngs(42)
    model = EntityTransformer(num_features=14, num_classes=5, rngs=rngs)
    
    # FIX: Grab the graphdef here before loading weights
    graphdef, state_template = nnx.split(model)
    
    # 2. Inject Binary Weights
    bin_path = os.path.join(BASE_DIR, "optimized_model.bin")
    if not os.path.exists(bin_path):
        print(f"CRITICAL: {bin_path} not found. Agent will crash.")
        return
        
    with open(bin_path, "rb") as f:
        restored_state_dict = serialization.from_bytes(state_template, f.read())
        
    # THE SURGICAL FIX: Bypass nnx.update() and rebuild the model directly
    model = nnx.merge(graphdef, restored_state_dict)

    # --- STRATEGY 1: PURE POLICY (GREEDY) ---
    @nnx.jit
    def pure_policy_inference(m, obs_tensor, my_planets_mask, env_state, player_id, rng_key):
        valid_launch_mask = my_planets_mask & (env_state.planet_ships >= 1.0)
        _, launch_logits, angle_logits, ships_logits, _ = m(
            obs_tensor, return_policy=True, valid_launch_mask=valid_launch_mask[None, :]
        )
        
        launch_decisions = (launch_logits[0] > 0.0).astype(jnp.int32)
        target_decisions = jnp.argmax(angle_logits[0], axis=-1)
        ship_decisions = jnp.argmax(ships_logits[0], axis=-1)
        
        return launch_decisions, target_decisions, ship_decisions

    # --- STRATEGY 2: DEEP BEAM SEARCH ---
    @nnx.jit
    def beam_search_inference(m, obs_tensor, my_planets_mask, env_state, player_id, rng_key):
        NUM_CANDIDATES = 64  # INCREASED WIDTH: Test 64 distinct action combinations
        ROLLOUT_DEPTH = 100   # INCREASED DEPTH: Fast-forward physics 15 ticks so fleets land!
        
        valid_launch_mask = my_planets_mask & (env_state.planet_ships >= 1.0)
        
        # 1. Evaluate current state to get policy logits
        _, launch_logits, angle_logits, ships_logits, _ = m(
            obs_tensor, return_policy=True, valid_launch_mask=valid_launch_mask[None, :]
        )
        
        # 2. Generate Candidate Actions (Stochastic Sampling)
        # We use jax.vmap to sample NUM_CANDIDATES different action combinations
        def sample_action(r_key):
            r1, r2, r3 = jax.random.split(r_key, 3)
            # Sample Launch (Bernoulli)
            l_prob = jax.nn.sigmoid(launch_logits[0])
            l_act = jax.random.bernoulli(r1, l_prob).astype(jnp.int32)
            # Sample Target (Categorical)
            t_act = jax.random.categorical(r2, angle_logits[0], axis=-1)
            # Sample Ships (Categorical)
            s_act = jax.random.categorical(r3, ships_logits[0], axis=-1)
            return l_act, t_act, s_act
            
        rng_keys = jax.random.split(rng_key, NUM_CANDIDATES)
        # c_l_acts shape: [64, 50]
        c_l_acts, c_t_acts, c_s_acts = jax.vmap(sample_action)(rng_keys)
        
        # Inject the "Greedy" action as Candidate 0 to guarantee we never do worse than greedy
        greedy_l = (launch_logits[0] > 0.0).astype(jnp.int32)
        greedy_t = jnp.argmax(angle_logits[0], axis=-1)
        greedy_s = jnp.argmax(ships_logits[0], axis=-1)
        
        c_l_acts = c_l_acts.at[0].set(greedy_l)
        c_t_acts = c_t_acts.at[0].set(greedy_t)
        c_s_acts = c_s_acts.at[0].set(greedy_s)
        
        # Inject the "Passive" action as Candidate 1 to guarantee we can safely hold
        c_l_acts = c_l_acts.at[1].set(jnp.zeros(50, dtype=jnp.int32))
        
        # 3. Simulate Parallel Futures
        from env_jax.orbit_env_v2 import apply_actions, step_physics, build_observation
        
        def simulate_future(l_act, t_act, s_act):
            # Step 1: Launch fleets directly to target planet IDs
            state_next = apply_actions(env_state, player_id, l_act, t_act, s_act)
            
            # Step 2: Rollout physics loop into the future!
            def physics_loop(i, state_carry):
                return step_physics(state_carry)
            
            state_future = jax.lax.fori_loop(0, ROLLOUT_DEPTH, physics_loop, state_next)
            
            return build_observation(state_future, player_id=player_id, win_rate=1.0)
            
        # vmap the simulation over all candidates
        # resulting_obs shape: [16, 70, 12]
        resulting_obs = jax.vmap(simulate_future)(c_l_acts, c_t_acts, c_s_acts)
        
        # 4. Evaluate all futures at once
        _, _, _, _, scores = m(resulting_obs, return_policy=True)
        
        # 5. Pick the Winner
        best_cand_idx = jnp.argmax(scores)
        
        final_launch = c_l_acts[best_cand_idx]
        final_target = c_t_acts[best_cand_idx]
        final_ships = c_s_acts[best_cand_idx]
        
        return final_launch, final_target, final_ships
    # --- STRATEGY 3: MCTS (MuZero Policy) ---
    @nnx.jit
    def mctx_inference(m, obs_tensor, my_planets_mask, env_state, player_id, rng_key):
        from env_jax.orbit_env_v2 import apply_actions, step_physics, build_observation
        
        r_sim, r_root = jax.random.split(rng_key)
        
        def generate_candidates(state, obs, rng):
            v_mask = (state.planet_owner == player_id) & (state.planet_ships >= 1.0)
            _, l_logits, a_logits, s_logits, _ = m(
                obs, return_policy=True, valid_launch_mask=v_mask[None, :]
            )
            def sample_one(r):
                r1, r2, r3 = jax.random.split(r, 3)
                l_act = jax.random.bernoulli(r1, jax.nn.sigmoid(l_logits[0])).astype(jnp.int32)
                t_act = jax.random.categorical(r2, a_logits[0], axis=-1)
                s_act = jax.random.categorical(r3, s_logits[0], axis=-1)
                return l_act, t_act, s_act
            keys = jax.random.split(rng, 8)
            return jax.vmap(sample_one)(keys)

        l, t, s = generate_candidates(env_state, obs_tensor, r_root)

        from flax import struct
        @struct.dataclass
        class MCTSEmbedding:
            env_state: EnvState
            l_acts: jnp.ndarray
            t_acts: jnp.ndarray
            s_acts: jnp.ndarray

        def root_fn(rng_key):
            _, _, _, _, value = m(obs_tensor, return_policy=True)
            batched_env_state = jax.tree_util.tree_map(lambda x: x[None, ...], env_state)
            embedding = MCTSEmbedding(env_state=batched_env_state, l_acts=l[None, ...], t_acts=t[None, ...], s_acts=s[None, ...])
            return mctx.RootFnOutput(prior_logits=jnp.zeros((1, 8)), value=value, embedding=embedding)
            
        def recurrent_fn(params, rng_key, action, embedding):
            env_state_unbatched = jax.tree_util.tree_map(lambda x: x[0], embedding.env_state)
            l_acts_unbatched = embedding.l_acts[0]
            t_acts_unbatched = embedding.t_acts[0]
            s_acts_unbatched = embedding.s_acts[0]
            
            act_idx = action[0]
            l_act = l_acts_unbatched[act_idx]
            t_act = t_acts_unbatched[act_idx]
            s_act = s_acts_unbatched[act_idx]
            
            st = apply_actions(env_state_unbatched, player_id, l_act, t_act, s_act)
            st = step_physics(st)
            
            new_obs = build_observation(st, player_id=player_id, win_rate=1.0)[None, ...]
            _, _, _, _, value = m(new_obs, return_policy=True)
            
            new_batched_env_state = jax.tree_util.tree_map(lambda x: x[None, ...], st)
            new_embedding = MCTSEmbedding(env_state=new_batched_env_state, l_acts=embedding.l_acts, t_acts=embedding.t_acts, s_acts=embedding.s_acts)
            
            return mctx.RecurrentFnOutput(
                reward=jnp.zeros((1,)), discount=jnp.ones((1,)), 
                prior_logits=jnp.zeros((1, 8)), value=value
            ), new_embedding

        policy_output = mctx.muzero_policy(
            params=None, rng_key=r_sim, root=root_fn(r_root), recurrent_fn=recurrent_fn,
            num_simulations=16, dirichlet_fraction=0.0, temperature=0.0
        )
        
        final_launch = l[policy_output.action[0]]
        final_target = t[policy_output.action[0]]
        final_ships = s[policy_output.action[0]]
        
        final_launch = jnp.where(my_planets_mask, final_launch, 0)
        return final_launch, final_target, final_ships

    # --- BIND THE SELECTED ENGINE ---
    if STRATEGY == "MCTS":
        GLOBAL_AGENT.compiled_step = lambda o, m_mask, s, pid, rng: mctx_inference(model, o, m_mask, s, pid, rng)
    elif STRATEGY == "BEAM_SEARCH":
        GLOBAL_AGENT.compiled_step = lambda o, m_mask, s, pid, rng: beam_search_inference(model, o, m_mask, s, pid, rng)
    else:
        GLOBAL_AGENT.compiled_step = lambda o, m_mask, s, pid, rng: pure_policy_inference(model, o, m_mask, s, pid, rng)
    
    # 4. Trigger Hardware Compilation (Warmup)
    print("Triggering XLA compilation... (This will take slightly longer if Beam Search depth is high)")
    dummy_obs = jnp.zeros((1, 70, 14), dtype=jnp.float32)
    dummy_mask = jnp.zeros((50,), dtype=bool)
    
    # Create dummy state to fulfill the expected types
    dummy_state = EnvState(
        planet_x=jnp.zeros(50, dtype=jnp.float32),
        planet_y=jnp.zeros(50, dtype=jnp.float32),
        planet_radius=jnp.ones(50, dtype=jnp.float32),
        planet_production=jnp.zeros(50, dtype=jnp.float32),
        planet_owner=jnp.zeros(50, dtype=jnp.int32),
        planet_ships=jnp.zeros(50, dtype=jnp.float32),
        fleet_active=jnp.zeros(200, dtype=jnp.int32),
        fleet_owner=jnp.zeros(200, dtype=jnp.int32),
        fleet_ships=jnp.zeros(200, dtype=jnp.float32),
        fleet_x=jnp.zeros(200, dtype=jnp.float32),
        fleet_y=jnp.zeros(200, dtype=jnp.float32),
        fleet_dx=jnp.zeros(200, dtype=jnp.float32),
        fleet_dy=jnp.zeros(200, dtype=jnp.float32),
        fleet_src_planet=jnp.zeros(200, dtype=jnp.int32),
        planet_initial_x=jnp.zeros(50, dtype=jnp.float32),
        planet_initial_y=jnp.zeros(50, dtype=jnp.float32),
        planet_is_orbiting=jnp.zeros(50, dtype=jnp.float32),
        angular_velocity=jnp.array(0.0),
        tick=jnp.array(0, dtype=jnp.int32)
    )
    dummy_rng = jax.random.PRNGKey(0)
    _ = GLOBAL_AGENT.compiled_step(dummy_obs, dummy_mask, dummy_state, 1, dummy_rng)
    
    GLOBAL_AGENT.ready = True
    print("Agent Online and Ready.")

# Trigger initialization instantly during Kaggle container spin-up
try:
    initialize_agent()
except Exception as e:
    print("CRITICAL INIT FAILURE", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)

def get_kaggle_player(obs, config):
    """Extracts player ID safely from Kaggle dicts and maps to 1-based indexing."""
    raw_player = None
    if hasattr(obs, 'player'): raw_player = obs.player
    elif isinstance(obs, dict) and 'player' in obs: raw_player = obs['player']
    elif hasattr(config, 'agentIndex'): raw_player = config.agentIndex
    elif isinstance(config, dict) and 'agentIndex' in config: raw_player = config['agentIndex']
    
    return raw_player + 1 if raw_player is not None else 1

# --- KAGGLE MAIN LOOP ---
def act(obs, config):
    try:
        if not GLOBAL_AGENT.ready:
            print("[DEBUG] Agent not ready, returning []")
            return []

        # 1. Determine Correct Player ID
        player_id = get_kaggle_player(obs, config)

        # 2. Build the actual JAX State
        env_state = parse_kaggle_obs(obs)
        
        # Get the real turn number for the wiretap logs
        obs_dict = obs if isinstance(obs, dict) else obs.__dict__
        tick = int(obs_dict.get('step', 0))
        
        # 3. Process exactly as seen in training
        obs_tensor = build_observation(env_state, player_id=player_id, win_rate=1.0)[None, ...]
        
        # 4. Create action mask (only own planets)
        my_planets_mask = (env_state.planet_owner == player_id)[:50]
        owned_count = int(jnp.sum(my_planets_mask))
        
        # --- WIRETAP 1: Ownership ---
        if tick % 50 == 0:  # Print every 50 ticks to avoid spamming the console
            print(f"\n--- TICK {tick} ---")
            print(f"[WIRETAP 1] Player ID: {player_id} | Planets Owned: {owned_count}")
            
            if owned_count == 0:
                print(f"[!] FATAL: Agent thinks it owns 0 planets. Masking is broken.")
                raw_owners = [p[1] for p in obs.get('planets', [])] if isinstance(obs, dict) else 'Unknown'
                print(f"Raw Kaggle Owners: {raw_owners}")

        # (Add a PRNG key to your GLOBAL_AGENT initialization if you don't have one)
        if not hasattr(GLOBAL_AGENT, 'rng_key'):
            tick_val = int(obs.get('tick', 0)) if isinstance(obs, dict) else getattr(obs, 'tick', 0)
            GLOBAL_AGENT.rng_key = jax.random.PRNGKey(tick_val)
            
        # Generate a new key for this turn
        GLOBAL_AGENT.rng_key, subkey = jax.random.split(GLOBAL_AGENT.rng_key)

        # 5. Execute Highly-Optimized JIT Step with Beam Search
        launch_act, target_act, ships_act = GLOBAL_AGENT.compiled_step(
            obs_tensor, my_planets_mask, env_state, player_id, subkey
        )

        # --- WIRETAP 2: Model Intent ---
        if tick % 50 == 0 and owned_count > 0:
            # Extract just the launch decisions for the planets we actually own
            my_launches = launch_act[my_planets_mask]
            print(f"[WIRETAP 2] Launch Decisions for my planets (1=Launch, 0=Hold): {my_launches}")

        # 6. Format Actions for Kaggle
        actions = []
        
        # We need the current planet ships to calculate absolute ship counts
        current_garrisons = env_state.planet_ships
        
        # We need planet coordinates to calculate the angle to the target
        px = env_state.planet_x
        py = env_state.planet_y
        
        for p_idx in range(50):
            if launch_act[p_idx] == 1:
                target_idx = int(target_act[p_idx])
                
                # 1. Calculate Absolute Ships
                fraction = float(ships_act[p_idx] + 1.0) / 10.0
                absolute_ships = int(current_garrisons[p_idx] * fraction)
                
                # --- WIRETAP 3: Formatting ---
                if tick % 50 == 0:
                    print(f"[WIRETAP 3] Planet {p_idx} attempting launch. Garrison: {current_garrisons[p_idx]}, Fraction: {fraction}, Calc Ships: {absolute_ships}")

                # Only launch if we have at least 1 ship to send
                if absolute_ships < 1:
                    continue

                # 2. Calculate Angle in Radians
                dx = px[target_idx] - px[p_idx]
                dy = py[target_idx] - py[p_idx]
                angle_in_radians = float(jnp.arctan2(dy, dx))
                
                # 3. Format strictly as [source_id, angle_radians, ship_count]
                actions.append([
                    int(p_idx), 
                    angle_in_radians, 
                    absolute_ships
                ])
                
        if len(actions) > 0 and tick % 50 == 0:
            print(f"[WIRETAP 4] Successfully yielding actions to Kaggle: {actions}")

        return actions

    except Exception as e:
        tick_val = obs.get('step', 'Unknown') if isinstance(obs, dict) else getattr(obs, 'step', 'Unknown')
        print(f"STEP EXECUTION CRASH AT TICK {tick_val}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return []