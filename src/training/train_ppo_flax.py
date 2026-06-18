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
import optax
import orbax.checkpoint as ocp
from clu import metric_writers
from clu import periodic_actions
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)
    
from models.entity_transformer_flax import EntityTransformer
from env_jax.orbit_env import EnvState, step_physics, apply_actions, build_observation

# --- Stable Shared-A100 Configuration ---
NUM_ENVS = 384
ROLLOUT_STEPS = 128
PPO_EPOCHS = 4
BATCH_SIZE = 1024
LEARNING_RATE = 3e-6
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.005
VF_COEF = 0.5
MAX_POOL_SIZE = 50

# --- SITUATION-BASED STORAGE TOGGLE ---
POOL_DEVICE = "gpu" 

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
    
    # 4. Mask out padded groups (so we have exactly num_groups * 4 planets)
    group_indices = jnp.tile(jnp.arange(10), 4)
    active_mask = (group_indices < num_groups)
    
    # Set ghost slots to 0 radius, 0 prod, 0 ships
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
    
    # Home planets start with exactly 10 ships
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
    
    # 7. Pad to 50 for static JAX shapes (adding 10 empty ghost slots)
    pad_fn = lambda x, fill: jnp.concatenate([x, jnp.full((10,), fill, dtype=x.dtype)])
    px = pad_fn(px, 0.0)
    py = pad_fn(py, 0.0)
    radius = pad_fn(radius, 0.0)
    prod = pad_fn(prod, 0.0)
    ships = pad_fn(ships, 0.0)
    owner = pad_fn(owner, 0)
    is_orbiting = pad_fn(is_orbiting, 0.0)
    
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
        planet_radius=radius, planet_production=prod,
        planet_owner=owner, planet_ships=ships,
        fleet_active=f_active, fleet_owner=f_owner, fleet_ships=f_ships,
        fleet_x=f_x, fleet_y=f_y, fleet_dx=f_dx, fleet_dy=f_dy, fleet_src_planet=f_src,
        tick=jnp.array(0, dtype=jnp.int32)
    )



def train_ppo():
    print(f"Initializing PPO True Self-Play Pipeline (League Device: {POOL_DEVICE.upper()})...")
    rngs = nnx.Rngs(42)
    model = EntityTransformer(num_classes=5, rngs=rngs)
    graphdef, state = nnx.split(model)
    
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'checkpoints/flax'))
    mngr = ocp.CheckpointManager(ckpt_dir, options=ocp.CheckpointManagerOptions(max_to_keep=3, create=False))

    step = None
    try:
        step = mngr.latest_step()
    except FileNotFoundError:
        pass

    start_iteration = 0
    if step is not None:
        start_iteration = max(0, step - 41000)
        print(f"Restoring weights from {step} (Starting at Iteration {start_iteration})...")
        try:
            raw_restored = mngr.restore(step)
            if isinstance(raw_restored, dict) and '0' in raw_restored and '1' in raw_restored:
                dummy_tx = optax.adamw(learning_rate=3e-4)
                dummy_opt = nnx.Optimizer(model, dummy_tx)
                template = nnx.state((model, dummy_opt))
                restored_typed = mngr.restore(step, args=ocp.args.StandardRestore(template))
                restored_casted = jax.tree_util.tree_map(
                    lambda r, t: jnp.asarray(r, dtype=t.value.dtype) if hasattr(t, 'value') else r,
                    restored_typed['0'], template['0']
                )
                nnx.update(model, restored_casted)
            else:
                restored_typed = mngr.restore(step, args=ocp.args.StandardRestore(state))
                restored_casted = jax.tree_util.tree_map(
                    lambda r, t: jnp.asarray(r, dtype=t.value.dtype) if hasattr(t, 'value') else r,
                    restored_typed, state
                )
                nnx.update(model, restored_casted)
            graphdef, state = nnx.split(model)
            print(f"Restored successfully.")
        except Exception as e:
            print(f"Restore failed: {e}. Fresh init.")
    
    tx = optax.chain(optax.clip_by_global_norm(1), optax.adam(LEARNING_RATE))
    opt_state = tx.init(jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, state))
    
    def env_step(env_state, rng_key, merged_model, opponent_model):
        obs_p1 = build_observation(env_state, player_id=1, win_rate=1.0)
        _, l_logits1, a_logits1, s_logits1, ppo_value = merged_model(obs_p1[None, :, :], return_policy=True)
        
        obs_p2 = build_observation(env_state, player_id=2, win_rate=1.0)
        obs_p3 = build_observation(env_state, player_id=3, win_rate=1.0)
        obs_p4 = build_observation(env_state, player_id=4, win_rate=1.0)
        
        # Batch all 3 opponent inferences together for extreme GPU efficiency
        obs_opps = jnp.stack([obs_p2, obs_p3, obs_p4], axis=0)
        _, l_logits_opps, a_logits_opps, s_logits_opps, _ = opponent_model(obs_opps, return_policy=True)

        r1, r2, r3, r4, r5, r6, rng_key = jax.random.split(rng_key, 7)

        # Action Sampling P1
        l_prob1 = jax.nn.sigmoid(l_logits1[0])
        l_act1 = jax.random.bernoulli(r1, l_prob1).astype(jnp.int32)
        target_act1 = jax.random.categorical(r2, a_logits1[0], axis=-1)
        s_act1 = jax.random.categorical(r3, s_logits1[0], axis=-1)

        # Action Sampling Opponents (P2, P3, P4)
        l_prob_opps = jax.nn.sigmoid(l_logits_opps)
        l_act_opps = jax.random.bernoulli(r4, l_prob_opps).astype(jnp.int32)
        target_act_opps = jax.random.categorical(r5, a_logits_opps, axis=-1)
        s_act_opps = jax.random.categorical(r6, s_logits_opps, axis=-1)

        # P1 Transition Logging
        launch_lp = jnp.log(jnp.where(l_act1, l_prob1, 1.0 - l_prob1) + 1e-8)
        angle_lp = jax.nn.log_softmax(a_logits1[0])
        angle_lp = jnp.take_along_axis(angle_lp, target_act1[..., None], axis=-1)[..., 0]
        ships_lp = jax.nn.log_softmax(s_logits1[0])
        ships_lp = jnp.take_along_axis(ships_lp, s_act1[..., None], axis=-1)[..., 0]

        p1_mask = (env_state.planet_owner == 1)
        num_active_planets = jnp.maximum(1.0, jnp.sum(p1_mask))
        total_lp = jnp.sum((launch_lp + jnp.where(l_act1, angle_lp + ships_lp, 0.0)) * p1_mask) / num_active_planets
        value = ppo_value[0] if hasattr(ppo_value, '__len__') else ppo_value

        # P1 Diagnostic Calculations
        no_op = (jnp.sum(l_act1) == 0).astype(jnp.float32)
        launch_prob_mean = jnp.sum(l_prob1 * p1_mask) / (jnp.sum(p1_mask) + 1e-8)
        valid_launch_mask = (l_act1 == 1) & p1_mask & (env_state.planet_ships > 1.0)
        valid_launch_rate = jnp.sum(valid_launch_mask) / (jnp.sum(l_act1) + 1e-8)
        
        # Calculate actual ships sent: (bucket + 1) / 10 * garrison
        ships_sent_per_planet = env_state.planet_ships * ((s_act1 + 1.0) / 10.0)
        avg_fleet_size = jnp.sum(jnp.where(l_act1, ships_sent_per_planet, 0.0)) / (jnp.sum(l_act1) + 1e-8)

        # Execute Actions for all 4 players
        env_state = apply_actions(env_state, 1, l_act1, target_act1, s_act1)
        env_state = apply_actions(env_state, 2, l_act_opps[0], target_act_opps[0], s_act_opps[0])
        env_state = apply_actions(env_state, 3, l_act_opps[1], target_act_opps[1], s_act_opps[1])
        env_state = apply_actions(env_state, 4, l_act_opps[2], target_act_opps[2], s_act_opps[2])
        env_state = step_physics(env_state)

        p1_ships = jnp.sum(jnp.where(env_state.planet_owner == 1, env_state.planet_ships, 0.0))
        p2_ships = jnp.sum(jnp.where(env_state.planet_owner == 2, env_state.planet_ships, 0.0))
        p3_ships = jnp.sum(jnp.where(env_state.planet_owner == 3, env_state.planet_ships, 0.0))
        p4_ships = jnp.sum(jnp.where(env_state.planet_owner == 4, env_state.planet_ships, 0.0))

        max_enemy_ships = jnp.maximum(p2_ships, jnp.maximum(p3_ships, p4_ships))
        done = (env_state.tick >= 500) | (p1_ships == 0) | (max_enemy_ships == 0)
        
        win_condition = p1_ships > max_enemy_ships
        base_reward = jnp.where(win_condition, 1.0, -1.0)
        total_ships = p1_ships + p2_ships + p3_ships + p4_ships + 1e-8
        ship_dominance = p1_ships / total_ships
        reward = jnp.where(done, base_reward + ship_dominance, 0.0)
        
        def _reset(r): return reset_env(r)
        def _keep(r): return env_state
        env_state = jax.lax.cond(done, _reset, _keep, rng_key)

        launch_count = jnp.sum(l_act1.astype(jnp.float32))
        p1_planet_count = jnp.sum((env_state.planet_owner == 1).astype(jnp.float32))
        launch_rate = launch_count / 50.0
        terminal_win = jnp.where(done & (p1_ships > max_enemy_ships), 1.0, 0.0)
        terminal_loss = jnp.where(done & (max_enemy_ships > p1_ships), 1.0, 0.0)
        terminal_ship_gap = jnp.where(done, p1_ships - max_enemy_ships, 0.0)
        terminal_done = done.astype(jnp.float32)

        transition = {
            "obs": obs_p1, "action_launch": l_act1, "action_angle": target_act1,  
            "action_ships": s_act1, "log_prob": total_lp, "value": value,
            "reward": reward, "done": done, "launch_count": launch_count,
            "planet_count": p1_planet_count,
            "launch_rate": launch_rate, "terminal_win": terminal_win,
            "terminal_loss": terminal_loss, "terminal_ship_gap": terminal_ship_gap,
            "terminal_done": terminal_done,
            "no_op": no_op, "launch_prob_mean": launch_prob_mean, 
            "valid_launch_rate": valid_launch_rate, "avg_fleet_size": avg_fleet_size,
            "p1_ships": p1_ships
        }
        return env_state, transition, rng_key

    @jax.jit
    def rollout(env_states, rngs, state_flat, opponent_state_flat):
        merged_model = nnx.merge(graphdef, state_flat)
        opponent_model = nnx.merge(graphdef, opponent_state_flat)
        
        def _step(carry, _):
            es, rs = carry
            es, trans, rs = jax.vmap(env_step, in_axes=(0, 0, None, None))(es, rs, merged_model, opponent_model)
            return (es, rs), trans
        
        (env_states, rngs), transitions = jax.lax.scan(_step, (env_states, rngs), None, length=ROLLOUT_STEPS)
        
        final_obs = jax.vmap(lambda s: build_observation(s, player_id=1, win_rate=1.0))(env_states)
        final_v, _, _, _, final_ppo = merged_model(final_obs, return_policy=True)
        return env_states, rngs, transitions, final_ppo

    @jax.jit
    def compute_advantages(transitions, final_value):
        advantages = jnp.zeros_like(transitions['reward'])
        returns = jnp.zeros_like(transitions['reward'])
        last_adv = jnp.zeros(NUM_ENVS)
        last_v = final_value
        
        def adv_step(i, carry):
            advs, rets, l_adv, l_v = carry
            idx = ROLLOUT_STEPS - 1 - i
            r = transitions['reward'][idx]
            v = transitions['value'][idx]
            d = transitions['done'][idx]
            
            delta = r + GAMMA * l_v * (1.0 - d) - v
            l_adv = delta + GAMMA * GAE_LAMBDA * (1.0 - d) * l_adv
            
            advs = advs.at[idx].set(l_adv)
            rets = rets.at[idx].set(l_adv + v)
            return advs, rets, l_adv, v
            
        advantages, returns, _, _ = jax.lax.fori_loop(0, ROLLOUT_STEPS, adv_step, (advantages, returns, last_adv, last_v))
        flat_advs = advantages.flatten()
        flat_advs = (flat_advs - flat_advs.mean()) / (flat_advs.std() + 1e-8)
        
        flat_rets = returns.flatten()
        flat_rets = (flat_rets - flat_rets.mean()) / (flat_rets.std() + 1e-8)
        
        return {
            'obs': transitions['obs'].reshape(-1, 70, 12),
            'a_launch': transitions['action_launch'].reshape(-1, 50) if 'action_launch' in transitions else transitions['a_launch'].reshape(-1, 50),
            'a_angle': transitions['action_angle'].reshape(-1, 50) if 'action_angle' in transitions else transitions['a_angle'].reshape(-1, 50),
            'a_ships': transitions['action_ships'].reshape(-1, 50) if 'action_ships' in transitions else transitions['a_ships'].reshape(-1, 50),
            'old_lp': transitions['log_prob'].flatten() if 'log_prob' in transitions else transitions['old_lp'].flatten(),
            'adv': flat_advs,
            'ret': flat_rets
        }

    @jax.jit
    def shuffle_and_batch(dataset, rng_key):
        total_samples = ROLLOUT_STEPS * NUM_ENVS
        total_batches = total_samples // BATCH_SIZE
        indices = jax.random.permutation(rng_key, total_samples)
        
        return {
            k: v[indices].reshape((total_batches, BATCH_SIZE) + v.shape[1:]) 
            for k, v in dataset.items()
        }

    def loss_fn(s_flat, batch):
        merged = nnx.merge(graphdef, s_flat)
        v_logits, launch_logits, angle_logits, ships_logits, ppo_v = merged(
            batch['obs'], return_policy=True, target_launch=batch['a_launch'], target_angle=batch['a_angle'])
        
        launch_prob = jax.nn.sigmoid(launch_logits)
        p_safe = jnp.clip(launch_prob, 1e-7, 1.0 - 1e-7)
        launch_lp = jnp.log(jnp.where(batch['a_launch'], p_safe, 1.0 - p_safe))
        
        angle_lp = jax.nn.log_softmax(angle_logits)
        angle_lp = jnp.take_along_axis(angle_lp, batch['a_angle'][..., None], axis=-1)[..., 0]
        
        ships_lp = jax.nn.log_softmax(ships_logits)
        ships_lp = jnp.take_along_axis(ships_lp, batch['a_ships'][..., None], axis=-1)[..., 0]
        
        p1_mask = (batch['obs'][:, 1:51, 5] == 1.0)
        num_active_planets = jnp.maximum(1.0, jnp.sum(p1_mask, axis=1))
        new_lp = jnp.sum((launch_lp + jnp.where(batch['a_launch'], angle_lp + ships_lp, 0.0)) * p1_mask, axis=1) / num_active_planets
        
        old_lp_safe = jnp.nan_to_num(batch['old_lp'], neginf=-100.0)
        new_lp_safe = jnp.nan_to_num(new_lp, neginf=-100.0)
        
        log_diff = jnp.clip(new_lp_safe - old_lp_safe, -10.0, 10.0)
        ratio = jnp.clip(jnp.exp(log_diff), 0.0, 5.0) 
        
        surr1 = ratio * batch['adv']
        surr2 = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * batch['adv']
        
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        
        value_pred = ppo_v[..., 0] if len(ppo_v.shape) > 1 else ppo_v
        value_loss = jnp.mean(jnp.square(value_pred - batch['ret']))
        
        entropy = -jnp.mean(launch_prob * jnp.log(launch_prob + 1e-8) + (1.0 - launch_prob) * jnp.log(1.0 - launch_prob + 1e-8))
        
        # Calculate Explained Variance to track Value Network health
        explained_var = 1.0 - jnp.var(batch['ret'] - value_pred) / (jnp.var(batch['ret']) + 1e-8)
        
        total_loss = policy_loss + VF_COEF * value_loss - ENTROPY_COEF * entropy
        
        metrics = {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'kl': jnp.mean(old_lp_safe - new_lp_safe),
            'clip_frac': jnp.mean(jnp.abs(ratio - 1.0) > CLIP_EPS),
            'ratio': jnp.mean(ratio),
            'explained_var': explained_var
        }
        return total_loss, metrics

    @jax.jit
    def train_batch(s_flat, o_st, batch):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(s_flat, batch)
        metrics['grad_norm'] = optax.global_norm(grads)
        updates, o_st = tx.update(grads, o_st, s_flat)
        s_flat = optax.apply_updates(s_flat, updates)
        return s_flat, o_st, metrics

    def move_pytree(pytree, target_device_str):
        target_device = jax.devices("gpu")[0] if target_device_str == "gpu" else jax.devices("cpu")[0]
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, target_device), pytree)
        
    state_flat = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, state)
    
    # --- Matrix Initialization ---
    rule_bot_state = {"iter": 0, "weights": move_pytree(state_flat, POOL_DEVICE)}
    best_agent_state = {"iter": 0, "weights": move_pytree(state_flat, POOL_DEVICE)}
    league_pool = [{"iter": 0, "weights": move_pytree(state_flat, POOL_DEVICE)}]
    
    elo_matrix = {"Rule/BC": 0.5, "Best Agent": 0.5, "Historical Meta": 0.5}

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, NUM_ENVS)
    env_states = jax.vmap(reset_env)(rngs)
    
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = metric_writers.create_default_writer(f"runs/ppo_league_{run_name}")
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=start_iteration + 10000, writer=writer)
    print("Starting League Training Loop...")
    
    for iteration in range(start_iteration, start_iteration + 10000):
        match_rand = random.random()
        if match_rand < 0.10:
            opp = rule_bot_state
            match_type = "Rule/BC"
        elif match_rand < 0.30:
            opp = best_agent_state
            match_type = "Best Agent"
        elif match_rand < 0.80:
            recent_pool = league_pool[-10:] if len(league_pool) >= 10 else league_pool
            opp = random.choice(recent_pool)
            match_type = "Recent Pool"
        else:
            opp = random.choice(league_pool)
            match_type = "Historical"

        opp_state = opp["weights"]
        opp_iter = opp["iter"]

        t0 = time.time()
        env_states, rngs, transitions, next_value = rollout(env_states, rngs, state_flat, opp_state)
        t_rollout = time.time() - t0
        
        t1 = time.time()
        dataset = compute_advantages(transitions, next_value)
        total_batches = (ROLLOUT_STEPS * NUM_ENVS) // BATCH_SIZE
        
        all_metrics = []
        for epoch in range(PPO_EPOCHS):
            rng, subkey = jax.random.split(rng)
            batches = shuffle_and_batch(dataset, subkey)
            
            for b in range(total_batches):
                # Extract single batch natively
                batch = {k: v[b] for k, v in batches.items()}
                state_flat, opt_state, step_metrics = train_batch(state_flat, opt_state, batch)
                
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
            'valid_launch_rate': jnp.mean(transitions['valid_launch_rate']),
            'avg_fleet_size': jnp.mean(transitions['avg_fleet_size']),
            'planet_count': jnp.mean(transitions['planet_count']),
            'p1_ships': jnp.mean(transitions['p1_ships'])
        }
        
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
        unique_targets_per_step = 0.0
        
        terminal_games = host_metrics['terminal_games']
        terminal_win_rate = host_metrics['terminal_win'] / (terminal_games + 1e-8)
        terminal_loss_rate = host_metrics['terminal_loss'] / (terminal_games + 1e-8)
        terminal_gap = host_metrics['terminal_ship_gap'] / (terminal_games + 1e-8)
        
        total_steps = NUM_ENVS * ROLLOUT_STEPS
        total_sps = total_steps / (t_rollout + t_update + 1e-8)

        # --- Update Win-Rate Matrix (Exponential Moving Average) ---
        if terminal_games > 0:
            alpha = 0.1 # Smoothing factor
            if match_type == "Rule/BC":
                elo_matrix["Rule/BC"] = (1 - alpha) * elo_matrix["Rule/BC"] + alpha * terminal_win_rate
            elif match_type == "Best Agent":
                elo_matrix["Best Agent"] = (1 - alpha) * elo_matrix["Best Agent"] + alpha * terminal_win_rate
            else:
                elo_matrix["Historical Meta"] = (1 - alpha) * elo_matrix["Historical Meta"] + alpha * terminal_win_rate

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

        opp_tag = f"ITER {opp_iter:03d}" if opp_iter > 0 else "BC_START"
        match_ctx = f"{match_type} ({opp_tag})"
        
        # Determine color for critical stability indicators
        kl_color = CLR_WARN if kl_mean > 0.05 else CLR_VAL
        clip_color = CLR_WARN if clip_frac > 0.25 else CLR_VAL
        ev_color = CLR_WARN if expl_var < 0.20 else CLR_VAL

        # Constructing dynamically spaced rows to prevent box shattering
        box_width = 86
        
        print(f"{CLR_DIV}┌" + "─" * (box_width - 2) + f"┐{CLR_RESET}")
        
        row1 = f" {CLR_HEADER}ITERATION {iteration:03d}{CLR_RESET} │ Target: {match_ctx:<23} │ Speed: {total_sps:>5.0f} SPS"
        print(f"{CLR_DIV}│{CLR_RESET}{row1:<{box_width + 18}}{CLR_DIV}│{CLR_RESET}") # +18 compensates for hidden ANSI chars
        
        print(f"{CLR_DIV}├─ HISTORICAL COMPETITIVE BENCHMARKS (EMA) ──────────────────────────────────────────┤{CLR_RESET}")
        row2 = f"   vs BC Baseline: {CLR_VAL}{elo_matrix['Rule/BC']:>5.1%}{CLR_RESET}  │  vs Best Gatekeeper: {CLR_VAL}{elo_matrix['Best Agent']:>5.1%}{CLR_RESET}  │  vs Meta Pool: {CLR_VAL}{elo_matrix['Historical Meta']:>5.1%}{CLR_RESET}"
        print(f"{CLR_DIV}│{CLR_RESET}{row2:<{box_width + 27}}{CLR_DIV}│{CLR_RESET}")
        
        print(f"{CLR_DIV}├─ OPTIMIZATION ENGINE & POLICY STABILITY ───────────────────────────────────────────┤{CLR_RESET}")
        row3 = f"   Actor Loss:     {CLR_VAL}{a_loss:+.4f}{CLR_RESET} │ Mean KL Div:   {kl_color}{kl_mean:.4f}{CLR_RESET} │ Policy Entropy: {CLR_VAL}{ent:.3f}{CLR_RESET}"
        row4 = f"   Value MSE Loss: {CLR_VAL}{v_loss:.4f}{CLR_RESET} │ Peak Max KL:   {CLR_VAL}{kl_max:.4f}{CLR_RESET} │ Update Clip %:  {clip_color}{clip_frac:>5.1%}{CLR_RESET}"
        row5 = f"   Explained Var:  {ev_color}{expl_var:>.3f}{CLR_RESET} │ Mean GradNorm: {CLR_VAL}{grad_norm_mean:.3f}{CLR_RESET} │ Active Pool:    {CLR_VAL}{len(league_pool):<3}{CLR_RESET}"
        print(f"{CLR_DIV}│{CLR_RESET}{row3:<{box_width + 27}}{CLR_DIV}│{CLR_RESET}")
        print(f"{CLR_DIV}│{CLR_RESET}{row4:<{box_width + 27}}{CLR_DIV}│{CLR_RESET}")
        print(f"{CLR_DIV}│{CLR_RESET}{row5:<{box_width + 27}}{CLR_DIV}│{CLR_RESET}")
        
        print(f"{CLR_DIV}├─ ACTOR BEHAVIORAL PROFILE ────────────────────────────────────────────────────────┤{CLR_RESET}")
        row6 = f"   Fleet Launch Rate:  {CLR_VAL}{avg_launch_rate:>5.1%}{CLR_RESET} │ Spatial Target Spread: {CLR_VAL}{unique_targets_per_step:>5.2f}/50.00{CLR_RESET}"
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
            "League/WinRate_vs_BC_EMA": elo_matrix["Rule/BC"],
            "League/WinRate_vs_Best_EMA": elo_matrix["Best Agent"],
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
            "Behavior/Valid_Launch_Rate": host_metrics['valid_launch_rate'],
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
            mngr.save(41001 + iteration, args=ocp.args.StandardSave(updated_state))
            mngr.wait_until_finished()
            
        # --- Gatekeeper Promotion Logic ---
        if match_type == "Best Agent" and terminal_games > 50 and terminal_win_rate > 0.55:
            print(f"*** GATEKEEPER DEFEATED (WinRate {terminal_win_rate:.2f}). PROMOTING ITERATION {iteration} TO BEST AGENT! ***")
            best_agent_state = {"iter": iteration, "weights": move_pytree(state_flat, POOL_DEVICE)}

        if iteration % 25 == 0 and iteration > 0:
            league_pool.append({"iter": iteration, "weights": move_pytree(state_flat, POOL_DEVICE)})
            if len(league_pool) > MAX_POOL_SIZE: league_pool.pop(1) 

if __name__ == "__main__":
    train_ppo()