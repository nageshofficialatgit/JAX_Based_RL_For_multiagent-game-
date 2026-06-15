import os
import sys
import math
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

# Add src to path so we can import the model
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from models.entity_transformer_flax import EntityTransformer

# Global state to prevent reloading
_GLOBAL_STATE = None

def _init_global_state():
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        return
        
    print("Initializing Agent Flax Model...", file=sys.stderr)
    rngs = nnx.Rngs(42)
    # Using our 38M parameter architecture
    model = EntityTransformer(d_model=512, n_heads=8, n_layers=12, rngs=rngs)
    
    graphdef, state = nnx.split(model)
    
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints/flax'))
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=False)
    mngr = ocp.CheckpointManager(ckpt_dir, options=options)
    
    # Hardcode step 41001 (Post-Surgery 4-Player Architecture)
    step = 41001
    if not mngr.item_metadata(step):
        # Fallback to latest if 41001 is deleted for some reason
        step = mngr.latest_step()
        
    print(f"Loading checkpoint {step}...", file=sys.stderr)
    restore_args = ocp.args.StandardRestore(state)
    state = mngr.restore(step, args=restore_args)
    
    @jax.jit
    def predict_step(state_flat, x):
        merged_model = nnx.merge(graphdef, state_flat)
        v_logits, launch_logits, angle_logits, ships_logits = merged_model(x, return_policy=True)
        return launch_logits, angle_logits, ships_logits
        
    _GLOBAL_STATE = {
        'state': state,
        'predict_fn': predict_step
    }
    print("Model initialized successfully.", file=sys.stderr)

def build_features(obs):
    planets = obs.get("planets", [])
    tick = obs.get("step", 0)
    
    planet_tensors = np.zeros((50, 12), dtype=np.float32)
    planet_tensors[:, 0] = -1.0 # padding
    
    ang_vel = obs.get("angular_velocity", 0.0)
    for p in planets:
        pid = p[0]
        if pid >= 50: continue
        x, y = p[1], p[2]
        owner = p[3]
        ships = p[4]
        radius = p[5]
        production = p[6]
        
        # Derived
        dx = x - 50.0
        dy = y - 50.0
        dist_sun = math.hypot(dx, dy)
        is_static = 1.0 if dist_sun + radius >= 50.0 else 0.0
        
        planet_tensors[pid, 0] = 0.0
        planet_tensors[pid, 1] = x
        planet_tensors[pid, 2] = y
        planet_tensors[pid, 3] = radius
        planet_tensors[pid, 4] = production
        planet_tensors[pid, 5] = float(owner)
        planet_tensors[pid, 6] = float(ships)
        planet_tensors[pid, 7] = is_static
        planet_tensors[pid, 8] = 0.0 # is_comet
        planet_tensors[pid, 9] = tick / 1000.0
        planet_tensors[pid, 10] = 0.5 # unknown win rate
        planet_tensors[pid, 11] = ang_vel

    action_tensors = np.zeros((20, 12), dtype=np.float32)
    action_tensors[:, 0] = -2.0 # padded action history
    
    fleets = obs.get("fleets", [])
    n_act = min(len(fleets), 20)
    
    def get_fleet_speed(ships):
        if ships <= 1: return 1.0
        ratio = math.log(ships) / math.log(1000.0)
        ratio = max(0.0, min(1.0, ratio))
        return 1.0 + 5.0 * (ratio ** 1.5)
        
    for i in range(n_act):
        f = fleets[i]
        owner = f[1]
        fx = f[2]
        fy = f[3]
        angle = f[4]
        src_pid = f[5]
        ships = f[6]
        
        # Approximate ticks since launch
        src_x = planet_tensors[src_pid, 1] if src_pid < 50 else 50.0
        src_y = planet_tensors[src_pid, 2] if src_pid < 50 else 50.0
        dist_flown = math.hypot(fx - src_x, fy - src_y)
        speed = get_fleet_speed(ships)
        ticks_flown = int(dist_flown / speed)
        
        action_tensors[i, 0] = 1.0
        action_tensors[i, 1] = float(src_pid)
        action_tensors[i, 2] = angle
        action_tensors[i, 3] = float(ships)
        action_tensors[i, 4] = float(owner)
        action_tensors[i, 5] = float(ticks_flown)
        action_tensors[i, 9] = tick / 1000.0
        action_tensors[i, 10] = 0.5
        action_tensors[i, 11] = ang_vel
    
    tokens = np.vstack([planet_tensors, action_tensors])
    return tokens[None, :, :] # Add batch dim

def agent(observation, configuration=None):
    try:
        _init_global_state()
        
        x = build_features(observation)
        x_jax = jnp.array(x)
        
        launch_logits, angle_logits, ships_logits = _GLOBAL_STATE['predict_fn'](_GLOBAL_STATE['state'], x_jax)
        
        launch_logits = np.array(launch_logits[0]) # [50]
        angle_logits = np.array(angle_logits[0]) # [50, 72]
        ships_logits = np.array(ships_logits[0]) # [50, 10]
        
        actions = []
        my_player_id = observation.get("player")
        planets = observation.get("planets", [])
        
        for p in planets:
            pid = p[0]
            owner = p[3]
            garrison = p[4]
            
            # The "God View": We simply ignore the model's predictions for the enemy player!
            if owner != my_player_id:
                continue
                
            if launch_logits[pid] > 0:
                angle_bin = np.argmax(angle_logits[pid])
                ships_bin = np.argmax(ships_logits[pid])
                
                angle_rad = (angle_bin / 72.0) * (2 * math.pi) - math.pi
                
                # ships_bin is 0-9 mapping to 0%-100%
                frac = (ships_bin + 0.5) / 10.0
                ships_to_send = max(1, int(garrison * frac))
                
                if ships_to_send > garrison:
                    ships_to_send = int(garrison)
                
                if ships_to_send > 0:
                    actions.append(f"SHIP {pid} {ships_to_send} {angle_rad:.3f}")
                    
        return actions
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []
