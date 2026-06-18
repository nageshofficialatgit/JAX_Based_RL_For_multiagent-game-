import os
import sys
import time
import random

# MUST BE SET BEFORE IMPORTING JAX FOR TRITON ACCELERATION
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from flax import nnx
from flax import serialization
import optax
import orbax.checkpoint as ocp
from clu import metric_writers
from clu import periodic_actions
import datetime
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)
    
from models.entity_transformer_flax_v2 import EntityTransformer
from env_jax.orbit_env_v2 import EnvState, step_physics, apply_actions, build_observation, reset_env

# --- Stable Shared-A100 Configuration ---
NUM_ENVS = 2048
ROLLOUT_STEPS = 128
PPO_EPOCHS = 2
BATCH_SIZE = 16384
LEARNING_RATE = 3e-4
GAMMA = 0.999
GAE_LAMBDA = 0.95
CLIP_EPS = 0.1  # Tightened MAPPO clip for BC Fine-tuning
VF_COEF = 0.5
MAX_POOL_SIZE = 50

# --- SITUATION-BASED STORAGE TOGGLE ---
POOL_DEVICE = "gpu" 

_last_config_mtime = 0
_cached_config = jnp.array([1.0, -1.0, 0.2, 0.5, 0.1, 0.02, -0.01, 0.05, 0.5], dtype=jnp.float32)

def load_reward_config():
    """Hot-reloadable reward config. Indices:
    0: base_win_reward        - Terminal reward for winning
    1: base_loss_reward       - Terminal reward for losing
    2: dense_ship_dominance   - p1_ships / total_ships (military power ratio)
    3: dense_planet_capture_delta - Reward per planet GAINED, penalty per planet LOST this tick
    4: dense_production_share - p1_production / total_production (economic dominance)
    5: dense_fleet_activity   - Reward for each fleet launched (encourages engagement)
    6: dense_no_op_penalty    - Penalty when zero fleets launched
    7: dense_planet_holding   - Small per-tick reward for owned_planets / 50
    8: terminal_dominance_bonus - Extra ship_dominance multiplier at game end
    """
    global _last_config_mtime, _cached_config
    config_path = os.path.join(os.path.dirname(__file__), "ppo_reward_config.json")
    try:
        mtime = os.path.getmtime(config_path)
        if mtime > _last_config_mtime:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            _cached_config = jnp.array([
                cfg.get("base_win_reward", 1.0),
                cfg.get("base_loss_reward", -1.0),
                cfg.get("dense_ship_dominance", 0.2),
                cfg.get("dense_planet_capture_delta", 0.5),
                cfg.get("dense_production_share", 0.1),
                cfg.get("dense_fleet_activity", 0.02),
                cfg.get("dense_no_op_penalty", -0.01),
                cfg.get("dense_planet_holding", 0.05),
                cfg.get("terminal_dominance_bonus", 0.5)
            ], dtype=jnp.float32)
            _last_config_mtime = mtime
    except Exception:
        pass
    return _cached_config


def train_ppo():
    print(f"Initializing PPO True Self-Play Pipeline (League Device: {POOL_DEVICE.upper()})...")
    rngs = nnx.Rngs(42)
    model = EntityTransformer(num_features=37, num_classes=5, rngs=rngs)
    graphdef, state = nnx.split(model)
    
    # Load BC pre-trained weights from bc_light checkpoint
    bc_ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'checkpoints/bc_v2'))
    # Save PPO weights to a separate ppo_v2 directory
    ppo_ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'checkpoints/ppo_v2'))
    os.makedirs(ppo_ckpt_dir, exist_ok=True)
    
    bc_mngr = ocp.CheckpointManager(bc_ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=3, create=False))
    mngr = ocp.CheckpointManager(ppo_ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True))

    graphdef, state = nnx.split(model)
    tx = optax.chain(optax.clip_by_global_norm(1), optax.adamw(LEARNING_RATE, weight_decay=1e-4))
    opt_state = tx.init(jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, state))

    # --- Checkpoint Restoration Priority: PPO_LIGHT > BC_LIGHT > Fresh Init ---
    start_iteration = 0
    restored = False
    
    # 1. Try resuming from existing PPO checkpoint
    try:
        ppo_step = mngr.latest_step()
        if ppo_step is not None:
            print(f"Resuming PPO from ppo_light checkpoint step {ppo_step}...")
            raw = mngr.restore(ppo_step)
            restored_model = raw.get('model', raw)
            
            def safe_merge_ppo(path, leaf_state):
                curr = restored_model
                for key_obj in path:
                    if isinstance(key_obj, jax.tree_util.DictKey):
                        k = key_obj.key
                        if isinstance(curr, dict) and k in curr:
                            curr = curr[k]
                        else:
                            return leaf_state
                    elif isinstance(key_obj, jax.tree_util.SequenceKey):
                        idx = key_obj.idx
                        if isinstance(curr, (list, tuple)) and idx < len(curr):
                            curr = curr[idx]
                        elif isinstance(curr, dict) and str(idx) in curr:
                            curr = curr[str(idx)]
                        else:
                            return leaf_state
                    else:
                        return leaf_state
                try:
                    tgt_dtype = leaf_state.value.dtype if hasattr(leaf_state, 'value') else getattr(leaf_state, 'dtype', None)
                    return jnp.asarray(curr, dtype=tgt_dtype) if tgt_dtype is not None else curr
                except Exception:
                    return leaf_state
            
            restored_model_casted = jax.tree_util.tree_map_with_path(safe_merge_ppo, state)
            nnx.update(model, restored_model_casted)
            
            # Intentionally NOT restoring opt_state to reset momentum and switch to AdamW safely
            print("Model restored. Optimizer state reset for AdamW transition.")
            
            start_iteration = ppo_step
            restored = True
            print(f"PPO resume successful at iteration {ppo_step}.")
    except Exception as e:
        print(f"No PPO checkpoint found: {e}")
    
    # 2. Fall back to BC pre-trained weights
    if not restored:
        try:
            bc_step = bc_mngr.latest_step()
            if bc_step is not None:
                print(f"Loading BC pre-trained weights from bc_v2 checkpoint step {bc_step}...")
                template = {'model': state}
                try:
                    restored_raw = bc_mngr.restore(bc_step)
                    # Manually merge only matching keys
                    def safe_merge(path, leaf_state):
                        curr = restored_raw.get('model', restored_raw)
                        for key_obj in path:
                            if isinstance(key_obj, jax.tree_util.DictKey):
                                k = key_obj.key
                                if k in ('value', 'raw_value'):
                                    continue
                                if isinstance(curr, dict) and k in curr:
                                    curr = curr[k]
                                else:
                                    return leaf_state
                            elif isinstance(key_obj, jax.tree_util.SequenceKey):
                                idx = key_obj.idx
                                if isinstance(curr, (list, tuple)) and idx < len(curr):
                                    curr = curr[idx]
                                elif isinstance(curr, dict) and str(idx) in curr:
                                    curr = curr[str(idx)]
                                else:
                                    return leaf_state
                            else:
                                return leaf_state
                        # Attempt to cast the loaded weights
                        try:
                            # For array leaves, jax.numpy.asarray safely casts to the exact shape and dtype expected by the model
                            return jnp.asarray(curr, dtype=leaf_state.dtype) if hasattr(leaf_state, 'dtype') else curr
                        except Exception:
                            return leaf_state
                    
                    restored_model_casted = jax.tree_util.tree_map_with_path(safe_merge, state)
                        
                    nnx.update(model, restored_model_casted)
                    print("Successfully loaded BC v2 weights.")
                except Exception as e2:
                    print(f"Manual merge failed: {e2}")
            else:
                print("No BC v2 weights found. Starting from scratch.")
                start_iteration = 0
        except Exception as e:
            print(f"Failed to load BC v2 weights: {e}. Starting from scratch.")
            start_iteration = 0
    
    def env_step(env_state, rng_key, merged_model, opponent_model, reward_config):
        obs_p1 = build_observation(env_state, player_id=1, win_rate=1.0)
        
        r1, r2, rng_key = jax.random.split(rng_key, 3)
        
        # P1 valid mask
        p1_mask = (env_state.planet_owner == 1)
        valid_launch_mask1 = p1_mask & (env_state.planet_ships >= 1.0)

        _, l_logits1, s_logits1, a_logits1, ppo_value, l_act1, s_act1, target_act1 = merged_model(
            obs_p1[None, :, :], return_policy=True, sample_rng=r1, valid_launch_mask=valid_launch_mask1[None, :]
        )
        l_act1, target_act1, s_act1 = l_act1[0], target_act1[0], s_act1[0]
        l_prob1 = jax.nn.sigmoid(l_logits1[0])
        
        obs_p2 = build_observation(env_state, player_id=2, win_rate=1.0)
        obs_p3 = build_observation(env_state, player_id=3, win_rate=1.0)
        obs_p4 = build_observation(env_state, player_id=4, win_rate=1.0)
        
        # Batch opponents
        obs_opps = jnp.stack([obs_p2, obs_p3, obs_p4], axis=0)
        vl_m_opps = jnp.stack([
            (env_state.planet_owner == 2) & (env_state.planet_ships >= 1.0),
            (env_state.planet_owner == 3) & (env_state.planet_ships >= 1.0),
            (env_state.planet_owner == 4) & (env_state.planet_ships >= 1.0)
        ], axis=0)

        _, l_logits_opps, s_logits_opps, a_logits_opps, _, l_act_opps, s_act_opps, target_act_opps = opponent_model(
            obs_opps, return_policy=True, sample_rng=r2, valid_launch_mask=vl_m_opps
        )

        # P1 Transition Logging
        launch_lp = jnp.log(jnp.where(l_act1, l_prob1, 1.0 - l_prob1) + 1e-8)
        angle_lp = jax.nn.log_softmax(a_logits1[0])
        angle_lp = jnp.take_along_axis(angle_lp, target_act1[..., None], axis=-1)[..., 0]
        ships_lp = jax.nn.log_softmax(s_logits1[0])
        ships_lp = jnp.take_along_axis(ships_lp, s_act1[..., None], axis=-1)[..., 0]

        # PATCH 1: Remove division and sum to allow per-planet PPO ratio calculation
        p1_mask = (env_state.planet_owner == 1)
        total_lp = (launch_lp + jnp.where(l_act1, angle_lp + ships_lp, 0.0)) * p1_mask
        value = ppo_value[0] if hasattr(ppo_value, '__len__') else ppo_value

        # P1 Diagnostic Calculations
        no_op = (jnp.sum(l_act1) == 0).astype(jnp.float32)
        launch_prob_mean = jnp.sum(l_prob1 * p1_mask) / (jnp.sum(p1_mask) + 1e-8)
        
        # Track the average ship payload bucket (1-10) sent by the agent
        avg_fleet_size = jnp.sum(jnp.where(l_act1, s_act1 + 1.0, 0.0)) / (jnp.sum(l_act1) + 1e-8)

        # --- PRE-ACTION SNAPSHOT (for capture delta) ---
        pre_p1_planet_mask = (env_state.planet_owner == 1)
        pre_p1_planets = jnp.sum(pre_p1_planet_mask.astype(jnp.float32))

        # Execute Actions for all 4 players (Angles are calculated inside orbit_env.py)
        # Measure exact number of fleets before launch
        active_fleets_before = jnp.sum(env_state.fleet_active)
        env_state = apply_actions(env_state, 1, l_act1, target_act1, s_act1)
        # Measure exact number of fleets after launch to get perfectly nuanced count
        active_fleets_after = jnp.sum(env_state.fleet_active)
        actual_launches_p1 = active_fleets_after - active_fleets_before
        
        env_state = apply_actions(env_state, 2, l_act_opps[0], target_act_opps[0], s_act_opps[0])
        env_state = apply_actions(env_state, 3, l_act_opps[1], target_act_opps[1], s_act_opps[1])
        env_state = apply_actions(env_state, 4, l_act_opps[2], target_act_opps[2], s_act_opps[2])
        env_state = step_physics(env_state)

        # --- POST-ACTION METRICS ---
        p1_planet_ships = jnp.sum(jnp.where(env_state.planet_owner == 1, env_state.planet_ships, 0.0))
        p1_fleet_ships = jnp.sum(jnp.where((env_state.fleet_owner == 1) & env_state.fleet_active, env_state.fleet_ships, 0.0))
        p1_ships = p1_planet_ships + p1_fleet_ships
        
        p2_planet_ships = jnp.sum(jnp.where(env_state.planet_owner == 2, env_state.planet_ships, 0.0))
        p2_fleet_ships = jnp.sum(jnp.where((env_state.fleet_owner == 2) & env_state.fleet_active, env_state.fleet_ships, 0.0))
        p2_ships = p2_planet_ships + p2_fleet_ships
        
        p3_planet_ships = jnp.sum(jnp.where(env_state.planet_owner == 3, env_state.planet_ships, 0.0))
        p3_fleet_ships = jnp.sum(jnp.where((env_state.fleet_owner == 3) & env_state.fleet_active, env_state.fleet_ships, 0.0))
        p3_ships = p3_planet_ships + p3_fleet_ships
        
        p4_planet_ships = jnp.sum(jnp.where(env_state.planet_owner == 4, env_state.planet_ships, 0.0))
        p4_fleet_ships = jnp.sum(jnp.where((env_state.fleet_owner == 4) & env_state.fleet_active, env_state.fleet_ships, 0.0))
        p4_ships = p4_planet_ships + p4_fleet_ships
        
        max_enemy_ships = jnp.maximum(p2_ships, jnp.maximum(p3_ships, p4_ships))
        done = (env_state.tick >= 500) | (p1_ships == 0) | (max_enemy_ships == 0)
        
        win_condition = p1_ships > max_enemy_ships
        base_reward = jnp.where(win_condition, reward_config[0], reward_config[1])
        total_ships = p1_ships + p2_ships + p3_ships + p4_ships + 1e-8
        
        # ZERO-SUM: Net Advantage instead of Absolute Share
        ship_advantage = (p1_ships - max_enemy_ships) / total_ships
        
        # --- PLANET CAPTURE DELTA (the key missing reward) ---
        post_p1_planet_mask = (env_state.planet_owner == 1)
        post_p1_planets = jnp.sum(post_p1_planet_mask.astype(jnp.float32))
        planet_delta = post_p1_planets - pre_p1_planets  # +N = captured, -N = lost
        
        # HIGH-VALUE CAPTURE MULTIPLIER
        newly_captured_mask = post_p1_planet_mask & ~pre_p1_planet_mask
        newly_captured_prod = jnp.sum(jnp.where(newly_captured_mask, env_state.planet_production, 0.0))
        
        # --- PRODUCTION SHARE (economic dominance) ---
        p1_prod = jnp.sum(jnp.where(env_state.planet_owner == 1, env_state.planet_production, 0.0))
        p2_prod = jnp.sum(jnp.where(env_state.planet_owner == 2, env_state.planet_production, 0.0))
        p3_prod = jnp.sum(jnp.where(env_state.planet_owner == 3, env_state.planet_production, 0.0))
        p4_prod = jnp.sum(jnp.where(env_state.planet_owner == 4, env_state.planet_production, 0.0))
        max_enemy_prod = jnp.maximum(p2_prod, jnp.maximum(p3_prod, p4_prod))
        total_prod = jnp.sum(env_state.planet_production) + 1e-8
        
        # ZERO-SUM: Net Production Advantage
        prod_advantage = (p1_prod - max_enemy_prod) / total_prod
        
        # --- FLEET ACTIVITY ---
        # Use purely mechanistic count of successfully spawned fleets instead of network intent
        fleet_count = actual_launches_p1.astype(jnp.float32)
        no_op_val = (fleet_count == 0).astype(jnp.float32)

        # Dense shaping rewards (all coefficients from JSON config)
        dense_reward = (
            ship_advantage * reward_config[2] +                 # [2] Military power advantage (Zero-Sum)
            (planet_delta * reward_config[3]) + (newly_captured_prod * 0.1 * reward_config[3]) + # [3] Conquest delta + Prod Multiplier
            prod_advantage * reward_config[4] +                 # [4] Economic advantage (Zero-Sum)
            fleet_count * reward_config[5] +                    # [5] Engagement activity
            no_op_val * reward_config[6] +                      # [6] No-op penalty
            (post_p1_planets / 50.0) * reward_config[7]         # [7] Gentle holding nudge (normalized)
        )
        
        terminal_bonus = ship_advantage * reward_config[8]    # [8] Configurable terminal dominance
        reward = jnp.where(done, base_reward + terminal_bonus + dense_reward, dense_reward)
        
        def _reset(r): return reset_env(r)
        def _keep(r): return env_state
        env_state = jax.lax.cond(done, _reset, _keep, rng_key)

        launch_count = jnp.sum(l_act1.astype(jnp.float32))
        launch_rate = launch_count / (pre_p1_planets + 1e-8)
        terminal_win = jnp.where(done & (p1_ships > max_enemy_ships), 1.0, 0.0)
        terminal_loss = jnp.where(done & (max_enemy_ships > p1_ships), 1.0, 0.0)
        terminal_ship_gap = jnp.where(done, p1_ships - max_enemy_ships, 0.0)
        terminal_done = done.astype(jnp.float32)
        
        p1_planet_count = post_p1_planets

        transition = {
            "obs": obs_p1, "action_launch": l_act1, "action_angle": target_act1,  
            "action_ships": s_act1, "log_prob": total_lp, "value": value,
            "reward": reward, "done": done, "launch_count": launch_count,
            "planet_count": p1_planet_count,
            "launch_rate": launch_rate, "terminal_win": terminal_win,
            "terminal_loss": terminal_loss, "terminal_ship_gap": terminal_ship_gap,
            "terminal_done": terminal_done,
            "no_op": no_op, "launch_prob_mean": launch_prob_mean, 
            "avg_fleet_size": avg_fleet_size, "p1_ships": p1_ships
        }
        return env_state, transition, rng_key

    @jax.jit
    def rollout(env_states, rngs, state_flat, opponent_state_flat, reward_config):
        merged_model = nnx.merge(graphdef, state_flat)
        opponent_model = nnx.merge(graphdef, opponent_state_flat)
        
        def _step(carry, _):
            es, rs = carry
            es, trans, rs = jax.vmap(env_step, in_axes=(0, 0, None, None, None))(es, rs, merged_model, opponent_model, reward_config)
            return (es, rs), trans
        
        (env_states, rngs), transitions = jax.lax.scan(_step, (env_states, rngs), None, length=ROLLOUT_STEPS)
        
        final_obs = jax.vmap(lambda s: build_observation(s, player_id=1, win_rate=1.0))(env_states)
        final_v, _, _, _, final_ppo = merged_model(final_obs, return_policy=True)
        return env_states, rngs, transitions, final_ppo

    @jax.jit
    def compute_advantages(transitions, final_value, ema_mean, ema_var):
        advantages = jnp.zeros_like(transitions['reward'])
        returns = jnp.zeros_like(transitions['reward'])
        last_adv = jnp.zeros(NUM_ENVS)
        
        # Use the CURRENT EMA state to unscale the values
        running_std = jnp.sqrt(ema_var + 1e-8)
        last_v_raw = final_value * running_std + ema_mean
        
        def adv_step(i, carry):
            advs, rets, l_adv, l_v_raw = carry
            idx = ROLLOUT_STEPS - 1 - i
            r = transitions['reward'][idx]
            d = transitions['done'][idx]
            
            # Unscale the network's value prediction back to the Raw Domain
            v_raw = transitions['value'][idx] * running_std + ema_mean
            
            delta = r + GAMMA * l_v_raw * (1.0 - d) - v_raw
            l_adv = delta + GAMMA * GAE_LAMBDA * (1.0 - d) * l_adv
            
            advs = advs.at[idx].set(l_adv)
            rets = rets.at[idx].set(l_adv + v_raw)
            return advs, rets, l_adv, v_raw
            
        advantages, returns, _, _ = jax.lax.fori_loop(0, ROLLOUT_STEPS, adv_step, (advantages, returns, last_adv, last_v_raw))
        flat_advs = advantages.flatten()
        flat_advs = (flat_advs - flat_advs.mean()) / (flat_advs.std() + 1e-8)
        
        flat_rets = returns.flatten()
        
        # --- THE KAGGLE GLOBAL EMA SCALING FIX ---
        # Update EMA with current batch (Alpha = 0.001 is roughly a 1000-batch sliding window)
        batch_mean = jnp.mean(flat_rets)
        batch_var = jnp.var(flat_rets)
        
        new_ema_mean = (1.0 - 0.001) * ema_mean + 0.001 * batch_mean
        new_ema_var = (1.0 - 0.001) * ema_var + 0.001 * batch_var
        
        running_std = jnp.sqrt(new_ema_var + 1e-8)
        
        # Scale globally and clip to the [-5, +5] range
        scaled_rets = (flat_rets - new_ema_mean) / running_std
        scaled_rets = jnp.clip(scaled_rets, -5.0, 5.0)
        
        dataset = {
            'obs': transitions['obs'].reshape(-1, 50, 37),
            'a_launch': transitions['action_launch'].reshape(-1, 50) if 'action_launch' in transitions else transitions['a_launch'].reshape(-1, 50),
            'a_angle': transitions['action_angle'].reshape(-1, 50) if 'action_angle' in transitions else transitions['a_angle'].reshape(-1, 50),
            'a_ships': transitions['action_ships'].reshape(-1, 50) if 'action_ships' in transitions else transitions['a_ships'].reshape(-1, 50),
            'old_lp': transitions['log_prob'].reshape(-1, 50) if 'log_prob' in transitions else transitions['old_lp'].reshape(-1, 50),
            'old_v': transitions['value'].reshape(-1) if 'value' in transitions else transitions['old_v'].reshape(-1),
            'adv': flat_advs,
            'ret': scaled_rets
        }
        return dataset, new_ema_mean, new_ema_var

    @jax.jit
    def shuffle_and_batch(dataset, rng_key):
        total_samples = ROLLOUT_STEPS * NUM_ENVS
        total_batches = total_samples // BATCH_SIZE
        indices = jax.random.permutation(rng_key, total_samples)
        
        return {
            k: v[indices].reshape((total_batches, BATCH_SIZE) + v.shape[1:]) 
            for k, v in dataset.items()
        }

    def loss_fn(s_flat, batch, current_entropy, current_clip):
        merged = nnx.merge(graphdef, s_flat)
        p1_mask = (batch['obs'][:, :50, 6] == 1.0)
        valid_launch_mask = p1_mask & (batch['obs'][:, :50, 7] >= (jnp.log1p(1.0)/7.0 - 1e-4))
        v_logits, launch_logits, ships_logits, angle_logits, ppo_v = merged(
            batch['obs'], return_policy=True, target_launch=batch['a_launch'], target_ships=batch['a_ships'], target_angle=batch['a_angle'], valid_launch_mask=valid_launch_mask)
        
        launch_prob = jax.nn.sigmoid(launch_logits)
        p_safe = jnp.clip(launch_prob, 1e-7, 1.0 - 1e-7)
        launch_lp = jnp.log(jnp.where(batch['a_launch'], p_safe, 1.0 - p_safe))
        
        angle_lp = jax.nn.log_softmax(angle_logits)
        angle_lp = jnp.take_along_axis(angle_lp, batch['a_angle'][..., None], axis=-1)[..., 0]
        
        ships_lp = jax.nn.log_softmax(ships_logits)
        ships_lp = jnp.take_along_axis(ships_lp, batch['a_ships'][..., None], axis=-1)[..., 0]
        
        p1_mask = (batch['obs'][:, :50, 6] == 1.0)
        
        # PATCH 3: Compute MAPPO ratio per-planet (independently)
        new_lp = (launch_lp + jnp.where(batch['a_launch'], angle_lp + ships_lp, 0.0)) * valid_launch_mask
        
        old_lp_safe = jnp.nan_to_num(batch['old_lp'], neginf=-100.0)
        new_lp_safe = jnp.nan_to_num(new_lp, neginf=-100.0)
        
        log_diff = jnp.clip(new_lp_safe - old_lp_safe, -10.0, 10.0)
        ratio = jnp.clip(jnp.exp(log_diff), 0.0, 5.0) 
        
        adv_broadcast = batch['adv'][:, None]
        surr1 = jnp.where(valid_launch_mask, ratio * adv_broadcast, 0.0)
        surr2 = jnp.where(valid_launch_mask, jnp.clip(ratio, 1.0 - current_clip, 1.0 + current_clip) * adv_broadcast, 0.0)
        
        policy_loss = -jnp.sum(jnp.minimum(surr1, surr2)) / (jnp.sum(valid_launch_mask) + 1e-8)
        
        value_pred = ppo_v[..., 0] if len(ppo_v.shape) > 1 else ppo_v
        v_clipped = batch['old_v'] + jnp.clip(value_pred - batch['old_v'], -current_clip, current_clip)
        v_loss_unclipped = jnp.square(value_pred - batch['ret'])
        v_loss_clipped = jnp.square(v_clipped - batch['ret'])
        value_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))
        
        # PATCH 4: Masked Entropy Calculation
        launch_ent_raw = -(launch_prob * jnp.log(launch_prob + 1e-8) + (1.0 - launch_prob) * jnp.log(1.0 - launch_prob + 1e-8))
        
        angle_probs = jnp.nan_to_num(jax.nn.softmax(angle_logits, axis=-1))  
        target_ent_raw = -jnp.sum(angle_probs * jnp.log(angle_probs + 1e-8), axis=-1)
        
        ships_probs = jnp.nan_to_num(jax.nn.softmax(ships_logits, axis=-1)) 
        ships_ent_raw = -jnp.sum(ships_probs * jnp.log(ships_probs + 1e-8), axis=-1)
        
        # FIX: Mask unused head entropy by valid_launch_mask to prevent entropy farming
        valid_entropies = (launch_ent_raw + launch_prob * (0.5 * target_ent_raw + 0.3 * ships_ent_raw)) * valid_launch_mask
        entropy = jnp.sum(valid_entropies) / (jnp.sum(valid_launch_mask) + 1e-8)
        
        # Calculate Explained Variance to track Value Network health
        explained_var = 1.0 - jnp.var(batch['ret'] - value_pred) / (jnp.var(batch['ret']) + 1e-8)
        explained_var = jnp.nan_to_num(explained_var)
        
        total_loss = policy_loss + VF_COEF * value_loss - current_entropy * entropy
        
        metrics = {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'kl': jnp.sum((old_lp_safe - new_lp_safe) * valid_launch_mask) / (jnp.sum(valid_launch_mask) + 1e-8),
            'clip_frac': jnp.sum((jnp.abs(ratio - 1.0) > current_clip) * valid_launch_mask) / (jnp.sum(valid_launch_mask) + 1e-8),
            'ratio': jnp.sum(ratio * valid_launch_mask) / (jnp.sum(valid_launch_mask) + 1e-8),
            'explained_var': explained_var
        }
        return total_loss, metrics

    @jax.jit
    def train_batch(s_flat, o_st, batch, current_entropy, current_clip):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(s_flat, batch, current_entropy, current_clip)
        
        # BULLETPROOF FIX: Prevent transient NaNs from infecting weights
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        
        metrics['grad_norm'] = optax.global_norm(grads)
        updates, o_st = tx.update(grads, o_st, s_flat)
        s_flat = optax.apply_updates(s_flat, updates)
        return s_flat, o_st, metrics

    def move_pytree(pytree, target_device_str):
        target_device = jax.devices("gpu")[0] if target_device_str == "gpu" else jax.devices("cpu")[0]
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, target_device), pytree)
        
    # Re-split to ensure state contains the loaded BC weights!
    graphdef, state = nnx.split(model)
    state_flat = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, state)
    
    # Create a purely random bot state for the "Rand Baseline"
    model_rand = EntityTransformer(num_features=37, num_classes=5, rngs=nnx.Rngs(999))
    _, state_rand = nnx.split(model_rand)
    state_flat_rand = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, state_rand)
    
    # --- Matrix Initialization ---
    import numpy as np
    import pickle
    
    rule_bot_state = {"iter": 0, "weights": move_pytree(state_flat_rand, POOL_DEVICE)}
    best_agent_state = {"iter": 0, "weights": move_pytree(state_flat, POOL_DEVICE)}
    league_pool = [{"iter": 0, "weights": move_pytree(state_flat, POOL_DEVICE)}]
    elo_matrix = {"Rand Baseline": 0.5, "Self-Play": 0.5, "Historical Meta": 0.5}
    league_win_rates = jnp.full(MAX_POOL_SIZE, 0.5, dtype=jnp.float32)

    # Attempt to load saved League State
    league_state_path = os.path.join(ppo_ckpt_dir, "league_state.pkl")
    if restored and os.path.exists(league_state_path):
        try:
            with open(league_state_path, "rb") as f:
                saved_state = pickle.load(f)
            
            # Reconstruct to device arrays
            def restore_pytree(pt):
                return jax.tree_util.tree_map(lambda x: jax.device_put(jnp.array(x), jax.devices(POOL_DEVICE)[0]) if isinstance(x, np.ndarray) else x, pt)
                
            league_pool = [restore_pytree(agent) for agent in saved_state["league_pool"]]
            best_agent_state = restore_pytree(saved_state["best_agent_state"])
            elo_matrix = saved_state["elo_matrix"]
            if "Rule/BC" in elo_matrix:
                elo_matrix["Rand Baseline"] = elo_matrix.pop("Rule/BC")
            print(f"Restored League State: Pool Size {len(league_pool)}, Best Agent Iter {best_agent_state['iter']}")
        except Exception as e:
            print(f"Failed to restore league state: {e}")

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, NUM_ENVS)
    env_states = jax.vmap(reset_env)(rngs)
    
    # --- PFSP Helper Functions ---
    @jax.jit
    def compute_pfsp_probs(win_rates_array, active_count):
        mask = jnp.arange(MAX_POOL_SIZE) < active_count
        weights = jnp.where(mask, (1.0 - win_rates_array) ** 2.0, 0.0)
        return weights / (jnp.sum(weights) + 1e-8)

    @jax.jit
    def update_historical_win_rate(win_rates_array, idx, terminal_wr, alpha=0.1):
        old_wr = win_rates_array[idx]
        new_wr = (1.0 - alpha) * old_wr + alpha * terminal_wr
        return win_rates_array.at[idx].set(new_wr)
    
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = metric_writers.create_default_writer(f"runs/ppo_league_{run_name}")
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=start_iteration + 10000, writer=writer)
    print("Starting League Training Loop...")
    
    # Initialize SAC-style auto-tuner state
    import math
    log_alpha = math.log(0.05)
    current_entropy_val = 0.05
    
    # Global EMA trackers for Return Scaling (Kaggle Top 10 Fix)
    return_ema_mean = jnp.array(0.0, dtype=jnp.float32)
    return_ema_var = jnp.array(1.0, dtype=jnp.float32)
    
    for iteration in range(start_iteration, start_iteration + 10000):
        # 1. Dynamic Config Load
        reward_config = load_reward_config()

        # 2. Prioritized Fictitious Self-Play (PFSP) Selection
        rng, opp_rng = jax.random.split(rng)
        opp_choice = float(jax.random.uniform(opp_rng))
        
        if opp_choice < 0.10:
            match_type = "Rand Baseline"
            opp = rule_bot_state
            opp_pool_idx = -1
        elif opp_choice < 0.30:
            match_type = "Best Agent"
            opp = best_agent_state
            opp_pool_idx = -1
        elif opp_choice < 0.75 or len(league_pool) == 0:
            # 45% Pure Self-Play
            match_type = "Self-Play"
            opp = {"iter": iteration, "weights": state_flat} 
            opp_pool_idx = -1
        else:
            # 25% PFSP Historical
            match_type = "Historical Meta"
            probs = compute_pfsp_probs(league_win_rates, len(league_pool))
            opp_pool_idx = int(jax.random.choice(opp_rng, MAX_POOL_SIZE, p=probs))
            opp = league_pool[opp_pool_idx]

        opp_iter = opp["iter"]
        opponent_state_flat = move_pytree(opp["weights"], "gpu")
        
        # --- 3. Run Trajectories ---
        t0 = time.time()
        env_states, rngs, transitions, next_value = rollout(
            env_states, rngs, state_flat, opponent_state_flat, reward_config
        )
        t_rollout = time.time() - t0
        
        t1 = time.time()
        dataset, return_ema_mean, return_ema_var = compute_advantages(
            transitions, next_value, return_ema_mean, return_ema_var
        )
        total_batches = (ROLLOUT_STEPS * NUM_ENVS) // BATCH_SIZE
        
        all_metrics = []
        for epoch in range(PPO_EPOCHS):
            rng, subkey = jax.random.split(rng)
            batches = shuffle_and_batch(dataset, subkey)
            
            # MAPPO clip (fixed at 0.2, no longer needs dynamic loosening)
            current_clip_eps = 0.2
            current_entropy_jnp = jnp.array(current_entropy_val, dtype=jnp.float32)
            current_clip_jnp = jnp.array(current_clip_eps, dtype=jnp.float32)

            for b in range(total_batches):
                # Extract single batch natively
                batch = {k: v[b] for k, v in batches.items()}
                state_flat, opt_state, step_metrics = train_batch(state_flat, opt_state, batch, current_entropy_jnp, current_clip_jnp)
                
                # Keep metrics ON DEVICE natively (no CPU blocking!)
                all_metrics.append(step_metrics)
            
            # CRITICAL: Prevent asynchronous dispatch queue from exploding memory
            # by forcing JAX to materialize the model state at the end of each epoch!
            state_flat = jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, state_flat)
                
        # Aggregate metrics over all batches ON THE DEVICE natively
        metrics = {k: jnp.mean(jnp.stack([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}
        t_update = time.time() - t1
        
        # --- Extract Diagnostics ---
        raw_metrics = {
            'a_loss': jnp.mean(metrics['policy_loss']),
            'v_loss': jnp.mean(metrics['value_loss']),
            'ent': jnp.mean(metrics['entropy']),
            'kl_mean': jnp.mean(metrics['kl']),
            'kl_max': jnp.max(metrics['kl']),
            'clip_frac': jnp.mean(metrics['clip_frac']),
            'expl_var': jnp.mean(metrics['explained_var']),
            'grad_norm_mean': jnp.mean(metrics['grad_norm']),
            'mean_reward': jnp.mean(transitions['reward']),
            'avg_launch_rate': jnp.mean(transitions['launch_rate']),
            'avg_attacks_per_turn': jnp.mean(transitions['launch_count']),
            'peak_planets': jnp.mean(jnp.max(transitions['planet_count'], axis=0)),
            'avg_ships_bucket': jnp.mean(transitions['action_ships']),
            'terminal_games': jnp.sum(transitions['terminal_done']),
            'terminal_win': jnp.sum(transitions['terminal_win']),
            'terminal_loss': jnp.sum(transitions['terminal_loss']),
            'terminal_ship_gap': jnp.sum(transitions['terminal_ship_gap']),
            'no_op': jnp.mean(transitions['no_op']),
            'launch_prob_mean': jnp.mean(transitions['launch_prob_mean']),
            'avg_fleet_size': jnp.mean(transitions['avg_fleet_size']),
            'planet_count': jnp.mean(transitions['planet_count']),
            'p1_ships': jnp.mean(transitions['p1_ships'])
        }
        
        # SAC-style Alpha Update
        target_entropy = 0.10 
        
        # We need raw_metrics['ent'] as a pure python float to update log_alpha
        # JAX array item extraction
        actual_entropy = float(raw_metrics['ent'])
        
        # If actual > target, explore less (alpha decreases)
        # If actual < target, explore more (alpha increases)
        log_alpha -= 0.05 * (actual_entropy - target_entropy)
        current_entropy_val = min(math.exp(log_alpha), 0.05)
        
        # Single Device-to-Host Synchronization
        host_metrics = jax.device_get(raw_metrics)
        
        a_loss = host_metrics['a_loss']
        v_loss = host_metrics['v_loss']
        ent = host_metrics['ent']
        kl_mean = host_metrics['kl_mean']
        kl_max = host_metrics['kl_max']
        clip_frac = host_metrics['clip_frac']
        expl_var = host_metrics['expl_var']
        grad_norm_mean = host_metrics['grad_norm_mean']

        mean_reward = host_metrics['mean_reward']
        avg_launch_rate = host_metrics['avg_launch_rate']
        avg_attacks_per_turn = host_metrics['avg_attacks_per_turn']
        peak_planets = host_metrics['peak_planets']
        avg_ships_bucket = host_metrics['avg_ships_bucket']
        
        # Removed due to high python loop/sync cost
        # unique_targets_per_step = 0.0
        
        terminal_games = host_metrics['terminal_games']
        terminal_win_rate = host_metrics['terminal_win'] / (terminal_games + 1e-8)
        terminal_loss_rate = host_metrics['terminal_loss'] / (terminal_games + 1e-8)
        terminal_gap = host_metrics['terminal_ship_gap'] / (terminal_games + 1e-8)
        
        total_steps = NUM_ENVS * ROLLOUT_STEPS
        total_sps = total_steps / (t_rollout + t_update + 1e-8)

        # --- PFSP Win-Rate Matrix Update ---
        if terminal_games > 0:
            alpha = 0.1 
            if match_type == "Rand Baseline":
                elo_matrix["Rand Baseline"] = (1 - alpha) * elo_matrix["Rand Baseline"] + alpha * terminal_win_rate
            elif match_type == "Historical Meta" and opp_pool_idx != -1:
                league_win_rates = update_historical_win_rate(league_win_rates, opp_pool_idx, terminal_win_rate, alpha)
                elo_matrix["Historical Meta"] = float(jnp.mean(league_win_rates[:len(league_pool)]))
            elif match_type == "Self-Play":
                elo_matrix["Self-Play"] = (1 - alpha) * elo_matrix.get("Self-Play", 0.5) + alpha * terminal_win_rate

        # ====================================================================
        # RICH MONITOR FORMATTING (DYNAMIC ALIGNMENT & TERMINAL COLORING)
        # ====================================================================
        # ANSI Escape Colors
        CLR_RESET = "\033[0m"
        CLR_HEADER= "\033[1;36m"  # Cyan Bold
        CLR_LABEL = "\033[0;37m"  # Dim White
        CLR_VAL   = "\033[1;32m"  # Green Bold
        CLR_WARN  = "\033[1;31m"  # Red Bold
        CLR_DIV   = "\033[0;34m"  # Light Blue for Borders

        opp_tag = f"ITER {opp_iter:03d}" if opp_iter > 0 else "RAND_START"
        match_ctx = f"{match_type} ({opp_tag})"
        
        # Determine color for critical stability indicators
        kl_color = CLR_WARN if kl_mean > 0.05 else CLR_VAL
        clip_color = CLR_WARN if clip_frac > 0.25 else CLR_VAL
        ev_color = CLR_WARN if expl_var < 0.20 else CLR_VAL

        # Constructing dynamically spaced rows to prevent box shattering
        box_width = 86
        
        print(f"{CLR_DIV}┌" + "─" * (box_width - 2) + f"┐{CLR_RESET}")
        
        row1 = f" {CLR_HEADER}ITERATION {iteration:03d}{CLR_RESET} │ Target: {match_ctx:<23} │ Speed: {total_sps:>5.0f} SPS │ R: {t_rollout:.2f}s | U: {t_update:.2f}s"
        print(f"{CLR_DIV}│{CLR_RESET}{row1:<{box_width + 18}}{CLR_DIV}│{CLR_RESET}") # +18 compensates for hidden ANSI chars
        
        print(f"{CLR_DIV}├─ HISTORICAL COMPETITIVE BENCHMARKS (EMA) ──────────────────────────────────────────┤{CLR_RESET}")
        row2 = f"   vs Rand Baseline: {CLR_VAL}{elo_matrix['Rand Baseline']:>5.1%}{CLR_RESET} │  vs Self-Play: {CLR_VAL}{elo_matrix.get('Self-Play', 0.5):>5.1%}{CLR_RESET}  │  vs Meta Pool: {CLR_VAL}{elo_matrix['Historical Meta']:>5.1%}{CLR_RESET}"
        print(f"{CLR_DIV}│{CLR_RESET}{row2:<{box_width + 27}}{CLR_DIV}│{CLR_RESET}")
        
        print(f"{CLR_DIV}├─ OPTIMIZATION ENGINE & POLICY STABILITY ───────────────────────────────────────────┤{CLR_RESET}")
        row3 = f"   Actor Loss:     {CLR_VAL}{a_loss:+.4f}{CLR_RESET} │ Mean KL Div:   {kl_color}{kl_mean:.4f}{CLR_RESET} │ Policy Entropy: {CLR_VAL}{ent:.3f}{CLR_RESET}"
        row4 = f"   Value MSE Loss: {CLR_VAL}{v_loss:.4f}{CLR_RESET} │ Peak Max KL:   {CLR_VAL}{kl_max:.4f}{CLR_RESET} │ Update Clip %:  {clip_color}{clip_frac:>5.1%}{CLR_RESET}"
        row5 = f"   Explained Var:  {ev_color}{expl_var:>.3f}{CLR_RESET} │ Mean GradNorm: {CLR_VAL}{grad_norm_mean:.3f}{CLR_RESET} │ Active Pool:    {CLR_VAL}{len(league_pool):<3}{CLR_RESET}"
        print(f"{CLR_DIV}│{CLR_RESET}{row3:<{box_width + 27}}{CLR_DIV}│{CLR_RESET}")
        print(f"{CLR_DIV}│{CLR_RESET}{row4:<{box_width + 27}}{CLR_DIV}│{CLR_RESET}")
        print(f"{CLR_DIV}│{CLR_RESET}{row5:<{box_width + 27}}{CLR_DIV}│{CLR_RESET}")
        
        print(f"{CLR_DIV}├─ ACTOR BEHAVIORAL PROFILE ────────────────────────────────────────────────────────┤{CLR_RESET}")
        row6 = f"   Fleet Launch Rate:  {CLR_VAL}{avg_launch_rate:>5.1%}{CLR_RESET} │"
        row7 = f"   Avg Payload Vol:    {CLR_VAL}{avg_ships_bucket:>5.2f}/10.00{CLR_RESET} │ Mean Step Reward:      {CLR_VAL}{mean_reward:+.4f}{CLR_RESET}"
        row7a = f"   Attacks Per Turn:   {CLR_VAL}{avg_attacks_per_turn:>5.2f}{CLR_RESET} │ Peak Planets Held:     {CLR_VAL}{peak_planets:>5.1f}/50.0{CLR_RESET}"
        print(f"{CLR_DIV}│{CLR_RESET}{row6:<{box_width + 18}}{CLR_DIV}│{CLR_RESET}")
        print(f"{CLR_DIV}│{CLR_RESET}{row7:<{box_width + 18}}{CLR_DIV}│{CLR_RESET}")
        print(f"{CLR_DIV}│{CLR_RESET}{row7a:<{box_width + 18}}{CLR_DIV}│{CLR_RESET}")
        
        print(f"{CLR_DIV}├─ ACTIVE MATCH METRICS ────────────────────────────────────────────────────────────┤{CLR_RESET}")
        row8 = f"   Epoch Win Rate:  {CLR_VAL}{terminal_win_rate:>5.1%}{CLR_RESET} │ Completed Games: {CLR_VAL}{int(terminal_games):<5}{CLR_RESET} │ Net Ship Margin: {CLR_VAL}{terminal_gap:+.1f}{CLR_RESET}"
        print(f"{CLR_DIV}│{CLR_RESET}{row8:<{box_width + 27}}{CLR_DIV}│{CLR_RESET}")
        
        print(f"{CLR_DIV}└" + "─" * (box_width - 2) + f"┘{CLR_RESET}")

        # --- CLU Asynchronous Logging ---
        writer.write_scalars(iteration, {
            "League/WinRate_vs_Rand_EMA": elo_matrix["Rand Baseline"],
            "League/WinRate_vs_SelfPlay_EMA": elo_matrix.get("Self-Play", 0.5),
            "League/WinRate_vs_Historical_EMA": elo_matrix["Historical Meta"],
            "League/Active_Match_WinRate": terminal_win_rate,
            "League/Opponent_Snapshot_Iteration": opp_iter,
            "Game/Mean_Reward": mean_reward,
            "Game/Terminal_Ship_Margin": terminal_gap,
            "Behavior/Fleet_Launch_Rate": avg_launch_rate,
            "Behavior/Avg_Attacks_Per_Turn": avg_attacks_per_turn,
            "Behavior/Peak_Planets_Held": peak_planets,
            "Behavior/Avg_Ships_Bucket": avg_ships_bucket,
            "Losses/Actor": a_loss,
            "Losses/Value": v_loss,
            "PPO/KL_Divergence": kl_mean,
            "PPO/Clip_Fraction": clip_frac,
            "PPO/Grad_Norm": grad_norm_mean,
            "Speed/Total_SPS": total_sps,
            "Behavior/No_Op_Rate": host_metrics['no_op'],
            "Behavior/Mean_Launch_Prob": host_metrics['launch_prob_mean'],
            "Behavior/Avg_Fleet_Size": host_metrics['avg_fleet_size'],
            "Game/Owned_Planet_Count": host_metrics['planet_count'],
            "Game/Total_Owned_Ships": host_metrics['p1_ships'],
            "PPO/Entropy": ent,
            "PPO/Explained_Variance": expl_var
        })
        
        # Trigger CLU periodic progress reports
        report_progress(iteration, time.time())
        
        if iteration % 50 == 0 and iteration > 0:
            merged_model = nnx.merge(graphdef, state_flat)
            _, updated_state = nnx.split(merged_model)
            save_payload = {'model': updated_state, 'opt': opt_state}
            mngr.save(iteration, args=ocp.args.StandardSave(save_payload))
            mngr.wait_until_finished()
            
            # Save League State via Pickle
            try:
                def force_numpy(pt):
                    return jax.tree_util.tree_map(lambda x: np.array(x) if hasattr(x, 'shape') else x, pt)
                    
                league_state_dump = {
                    "league_pool": [force_numpy(agent) for agent in league_pool],
                    "best_agent_state": force_numpy(best_agent_state),
                    "elo_matrix": elo_matrix
                }
                with open(league_state_path, "wb") as f:
                    pickle.dump(league_state_dump, f)
                print(f"Saved League State: Pool Size {len(league_pool)}")
            except Exception as e:
                print(f"Failed to save league state: {e}")
            
        # --- High-Performance Benchmark Dumping ---
        if iteration % 50 == 0 and iteration > 0:
            benchmark_dir = os.path.abspath(os.path.join(ppo_ckpt_dir, "..", "benchmarks"))
            os.makedirs(benchmark_dir, exist_ok=True)
            tmp_bin = os.path.join(benchmark_dir, f"benchmark_{iteration}.bin.tmp")
            final_bin = os.path.join(benchmark_dir, f"benchmark_{iteration}.bin")
            
            def force_to_pure_dict(node):
                if hasattr(node, 'items'): return {k: force_to_pure_dict(v) for k, v in node.items()}
                elif hasattr(node, 'value'): return force_to_pure_dict(node.value)
                elif isinstance(node, (list, tuple)): return type(node)(force_to_pure_dict(v) for v in node)
                return node
                
            # We can directly serialize the state_flat since it has no optimizer state!
            pure_dict = force_to_pure_dict(state_flat)
            with open(tmp_bin, "wb") as f:
                f.write(serialization.to_bytes(pure_dict))
            os.rename(tmp_bin, final_bin)
            print(f"Dumped Benchmark Binary: {final_bin}")
            
        # --- Gatekeeper Promotion Logic ---
        if match_type == "Best Agent" and terminal_games > 50 and terminal_win_rate > 0.55:
            print(f"*** GATEKEEPER DEFEATED (WinRate {terminal_win_rate:.2f}). PROMOTING ITERATION {iteration} TO BEST AGENT! ***")
            best_agent_state = {"iter": iteration, "weights": move_pytree(state_flat, POOL_DEVICE)}

        if iteration % 25 == 0 and iteration > 0:
            league_pool.append({"iter": iteration, "weights": move_pytree(state_flat, POOL_DEVICE)})
            if len(league_pool) > MAX_POOL_SIZE: league_pool.pop(1) 

if __name__ == "__main__":
    train_ppo()