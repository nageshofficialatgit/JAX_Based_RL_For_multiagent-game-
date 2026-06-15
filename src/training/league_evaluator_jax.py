import chex
chex.disable_asserts()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import glob
import pandas as pd
import jax
import jax.numpy as jnp
from flax import nnx
from flax import serialization
import trueskill
import mctx
import flax.struct as struct

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import Master Environment (V2) which contains both build_observation and build_observation_v1
from env_jax.orbit_env_v2 import EnvState, step_physics, apply_actions, build_observation, build_observation_v1, reset_env

# Import BOTH Architectures
from models.entity_transformer_flax_small import EntityTransformer as EntityTransformerV1
from models.entity_transformer_flax_v2 import EntityTransformer as EntityTransformerV2

trueskill.setup(draw_probability=0.01)

CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"))
BENCHMARK_DIR = os.path.join(CHECKPOINT_DIR, "benchmarks")
LEADERBOARD_CSV = os.path.join(CHECKPOINT_DIR, "leaderboard.csv")

NUM_ENVS = 512
MAX_STEPS = 500

def get_action(env_state, player_id, obs, merged_model, rng):
    valid_mask = (env_state.planet_owner == player_id) & (env_state.planet_ships >= 1.0)
    _, l_logits, a_logits, s_logits, _, l_act, t_act, s_act = merged_model(
        obs[None, ...], return_policy=True, sample_rng=rng, valid_launch_mask=valid_mask[None, :]
    )
    return l_act[0], t_act[0], s_act[0]

def get_action_greedy(env_state, player_id, obs, merged_model, rng):
    valid_mask = (env_state.planet_owner == player_id) & (env_state.planet_ships >= 1.0)
    _, l_logits, a_logits, s_logits, _ = merged_model(
        obs[None, ...], return_policy=True, valid_launch_mask=valid_mask[None, :]
    )
    l_act = (jax.nn.sigmoid(l_logits[0]) > 0.3).astype(jnp.int32)
    t_act = jnp.argmax(a_logits[0], axis=-1).astype(jnp.int32)
    s_act = jnp.argmax(s_logits[0], axis=-1).astype(jnp.int32)
    return l_act, t_act, s_act

@struct.dataclass
class MCTSEmbedding:
    env_state: EnvState
    l_acts: jnp.ndarray
    t_acts: jnp.ndarray
    s_acts: jnp.ndarray

def generate_candidates(env_state, player_id, obs, merged_model, rng):
    valid_mask = (env_state.planet_owner == player_id) & (env_state.planet_ships >= 1.0)
    _, l_logits, a_logits, s_logits, _ = merged_model(
        obs[None, ...], return_policy=True, valid_launch_mask=valid_mask[None, :]
    )
    def sample_one(r):
        r1, r2, r3 = jax.random.split(r, 3)
        l_act = jax.random.bernoulli(r1, jax.nn.sigmoid(l_logits[0])).astype(jnp.int32)
        t_act = jax.random.categorical(r2, a_logits[0], axis=-1)
        s_act = jax.random.categorical(r3, s_logits[0], axis=-1)
        return l_act, t_act, s_act
    keys = jax.random.split(rng, 8)
    return jax.vmap(sample_one)(keys)

def get_action_mctx(env_state, player_id, obs, merged_model, rng, is_v2: bool):
    r_root, r_sim = jax.random.split(rng)
    
    def root_fn(rng_key):
        _, _, _, _, value = merged_model(obs[None, ...], return_policy=True)
        l, t, s = generate_candidates(env_state, player_id, obs, merged_model, rng_key)
        
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
        
        new_st = apply_actions(env_state_unbatched, player_id, l_act, t_act, s_act)
        new_st = step_physics(new_st)
        
        new_obs = build_observation(new_st, player_id, 1.0) if is_v2 else build_observation_v1(new_st, player_id, 1.0)
        _, _, _, _, value = merged_model(new_obs[None, ...], return_policy=True)
        new_l, new_t, new_s = generate_candidates(new_st, player_id, new_obs, merged_model, rng_key)
        
        batched_new_st = jax.tree_util.tree_map(lambda x: x[None, ...], new_st)
        new_embedding = MCTSEmbedding(env_state=batched_new_st, l_acts=new_l[None, ...], t_acts=new_t[None, ...], s_acts=new_s[None, ...])
        
        return mctx.RecurrentFnOutput(
            reward=jnp.zeros((1,)), discount=jnp.ones((1,)), 
            prior_logits=jnp.zeros((1, 8)), value=value
        ), new_embedding

    policy_output = mctx.muzero_policy(
        params=None, rng_key=r_sim, root=root_fn(r_root), recurrent_fn=recurrent_fn,
        num_simulations=16, dirichlet_fraction=0.0, temperature=0.0
    )
    l, t, s = generate_candidates(env_state, player_id, obs, merged_model, r_root)
    return l[policy_output.action[0]], t[policy_output.action[0]], s[policy_output.action[0]]

def get_action_mctx_v1(env_state, player_id, obs, merged_model, rng):
    return get_action_mctx(env_state, player_id, obs, merged_model, rng, False)
def get_action_mctx_v2(env_state, player_id, obs, merged_model, rng):
    return get_action_mctx(env_state, player_id, obs, merged_model, rng, True)


@jax.jit
def batch_simulate(rng_keys, models_flat_v1, models_flat_v2, graphdef_v1, graphdef_v2, agent_types):
    m1_v1 = nnx.merge(graphdef_v1, models_flat_v1[0])
    m2_v1 = nnx.merge(graphdef_v1, models_flat_v1[1])
    m3_v1 = nnx.merge(graphdef_v1, models_flat_v1[2])
    m4_v1 = nnx.merge(graphdef_v1, models_flat_v1[3])
    
    m1_v2 = nnx.merge(graphdef_v2, models_flat_v2[0])
    m2_v2 = nnx.merge(graphdef_v2, models_flat_v2[1])
    m3_v2 = nnx.merge(graphdef_v2, models_flat_v2[2])
    m4_v2 = nnx.merge(graphdef_v2, models_flat_v2[3])
    
    def single_env_loop(r_key):
        r_reset, r_loop = jax.random.split(r_key)
        state = reset_env(r_reset)
        
        def step_fn(i, val):
            st, rk = val
            rk, r1, r2, r3, r4 = jax.random.split(rk, 5)
            
            o1_v1 = build_observation_v1(st, 1, 1.0)
            o2_v1 = build_observation_v1(st, 2, 1.0)
            o3_v1 = build_observation_v1(st, 3, 1.0)
            o4_v1 = build_observation_v1(st, 4, 1.0)
            
            o1_v2 = build_observation(st, 1, 1.0)
            o2_v2 = build_observation(st, 2, 1.0)
            o3_v2 = build_observation(st, 3, 1.0)
            o4_v2 = build_observation(st, 4, 1.0)

            def get_action_slot(agent_type, st, player_id, r, o_v1, o_v2, m_v1, m_v2):
                def do_v1_temp(args): return get_action(st, player_id, o_v1, m_v1, r)
                def do_v1_greedy(args): return get_action_greedy(st, player_id, o_v1, m_v1, r)
                
                def do_v2_temp(args): return get_action(st, player_id, o_v2, m_v2, r)
                def do_v2_greedy(args): return get_action_greedy(st, player_id, o_v2, m_v2, r)
                
                return jax.lax.switch(
                    agent_type,
                    [do_v1_temp, do_v1_greedy, do_v2_temp, do_v2_greedy],
                    None
                )
            
            l1, t1, s1 = get_action_slot(agent_types[0], st, 1, r1, o1_v1, o1_v2, m1_v1, m1_v2)
            l2, t2, s2 = get_action_slot(agent_types[1], st, 2, r2, o2_v1, o2_v2, m2_v1, m2_v2)
            l3, t3, s3 = get_action_slot(agent_types[2], st, 3, r3, o3_v1, o3_v2, m3_v1, m3_v2)
            l4, t4, s4 = get_action_slot(agent_types[3], st, 4, r4, o4_v1, o4_v2, m4_v1, m4_v2)

            st = apply_actions(st, 1, l1, t1, s1)
            st = apply_actions(st, 2, l2, t2, s2)
            st = apply_actions(st, 3, l3, t3, s3)
            st = apply_actions(st, 4, l4, t4, s4)
            st = step_physics(st)
            
            return st, rk
            
        final_state, _ = jax.lax.fori_loop(0, MAX_STEPS, step_fn, (state, r_loop))
        
        s1 = jnp.sum(jnp.where(final_state.planet_owner == 1, final_state.planet_ships, 0.0))
        s2 = jnp.sum(jnp.where(final_state.planet_owner == 2, final_state.planet_ships, 0.0))
        s3 = jnp.sum(jnp.where(final_state.planet_owner == 3, final_state.planet_ships, 0.0))
        s4 = jnp.sum(jnp.where(final_state.planet_owner == 4, final_state.planet_ships, 0.0))
        
        is_4p = (s3 + s4 > 0)
        
        return jnp.array([s1, s2, s3, s4]), is_4p
        
    return jax.vmap(single_env_loop)(rng_keys)

def load_ratings():
    if os.path.exists(LEADERBOARD_CSV):
        df = pd.read_csv(LEADERBOARD_CSV)
        return {row['Model']: trueskill.Rating(mu=row['Mu'], sigma=row['Sigma']) for _, row in df.iterrows()}
    return {}

def save_ratings(ratings_dict):
    data = []
    for model, r in ratings_dict.items():
        data.append({"Model": model, "Mu": r.mu, "Sigma": r.sigma, "TrueSkill": r.mu - 3 * r.sigma})
    df = pd.DataFrame(data).sort_values("TrueSkill", ascending=False)
    df.to_csv(LEADERBOARD_CSV, index=False)
    print("\n=== LEADERBOARD UPDATED ===")
    print(df.to_string(index=False))

def main():
    print("Initializing Ultra-Fast JAX League Evaluator (V1 & V2) on GPU...")
    rngs = nnx.Rngs(42)
    base_v1 = EntityTransformerV1(num_features=14, num_classes=5, rngs=rngs)
    base_v2 = EntityTransformerV2(num_features=37, num_classes=5, rngs=rngs)
    
    graphdef_v1, state_template_v1 = nnx.split(base_v1)
    graphdef_v2, state_template_v2 = nnx.split(base_v2)
    
    # Pre-extract pure tree structures
    template_val_v1 = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, state_template_v1)
    template_val_v2 = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, state_template_v2)
    
    # Create zeroed dummies
    dummy_v1 = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), template_val_v1)
    dummy_v2 = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), template_val_v2)
    
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    ratings = load_ratings()
    
    def get_model_state(model_name):
        if model_name.startswith("random"):
            _, s = nnx.split(EntityTransformerV2(num_features=37, num_classes=5, rngs=nnx.Rngs(int(time.time()))))
            state_val = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, s)
            return (None, state_val) # (v1_state, v2_state)
            
        base_name = model_name
        if base_name.endswith("_temp"): base_name = base_name[:-5]
        elif base_name.endswith("_greedy"): base_name = base_name[:-7]
        elif base_name.endswith("_mctx"): base_name = base_name[:-5]
        
        is_v2 = "v2" in base_name
        
        bin_path = os.path.join(BENCHMARK_DIR, f"{base_name}.bin")
        if not os.path.exists(bin_path): return None, None
        
        with open(bin_path, "rb") as f:
            raw = f.read()
            
        if is_v2:
            state_val = serialization.from_bytes(template_val_v2, raw)
            return (None, state_val)
        else:
            state_val = serialization.from_bytes(template_val_v1, raw)
            return (state_val, None)

    iteration = 0
    while True:
        iteration += 1
        bin_files = glob.glob(os.path.join(BENCHMARK_DIR, "*.bin"))
        
        # Prune old models: Keep only the 10 newest benchmarks
        bin_files.sort(key=os.path.getmtime, reverse=True)
        MAX_BENCHMARKS = 10
        if len(bin_files) > MAX_BENCHMARKS:
            for old_file in bin_files[MAX_BENCHMARKS:]:
                try: os.remove(old_file)
                except Exception: pass
            bin_files = bin_files[:MAX_BENCHMARKS]
            
        models_available = ["random", "random_2", "random_3", "random_4"]
        
        random_ts = -9999.0
        if "random" in ratings:
            r = ratings["random"]
            random_ts = r.mu - 3 * r.sigma
            
        for f in bin_files:
            base = os.path.basename(f).replace(".bin", "")
            for suffix in ["_greedy"]:
                model_name = f"{base}{suffix}"
                
                # Check if it's worse than random
                if model_name in ratings:
                    mr = ratings[model_name]
                    model_ts = mr.mu - 3 * mr.sigma
                    if model_ts < random_ts:
                        continue
                models_available.append(model_name)
        
        if len(models_available) < 4:
            print("Waiting for at least 4 models to be available...")
            time.sleep(10)
            continue
            
        import random
        selected_models = []
        if "bc_v2_greedy" in models_available:
            selected_models.append("bc_v2_greedy")
            models_available.remove("bc_v2_greedy")
        
        needed = 4 - len(selected_models)
        selected_models.extend(random.sample(models_available, needed))
        print(f"\n[{iteration}] Simulating Arena: {selected_models}")
        
        states_v1 = []
        states_v2 = []
        valid_agents = []
        
        for m in selected_models:
            sv1, sv2 = get_model_state(m)
            if sv1 is None and sv2 is None: break
            
            states_v1.append(sv1 if sv1 is not None else dummy_v1)
            states_v2.append(sv2 if sv2 is not None else dummy_v2)
            valid_agents.append(m)
            
        if len(valid_agents) < 4: continue
        
        rng_key = jax.random.PRNGKey(int(time.time()))
        keys = jax.random.split(rng_key, NUM_ENVS)
        
        t0 = time.time()
        def get_type_id(name):
            is_v2 = name.startswith("random") or ("v2" in name)
            offset = 2 if is_v2 else 0
            if name.endswith("_greedy"): return offset + 1
            return offset + 0
            
        agent_types = jnp.array([get_type_id(m) for m in selected_models], dtype=jnp.int32)
        
        final_ships, is_4p = batch_simulate(keys, states_v1, states_v2, graphdef_v1, graphdef_v2, agent_types)
        final_ships = np.array(final_ships)
        is_4p = np.array(is_4p)
        t_sim = time.time() - t0
        
        print(f"Simulated {NUM_ENVS} games in {t_sim:.2f}s ({(NUM_ENVS*MAX_STEPS)/t_sim:,.0f} steps/sec)")
        
        for m in selected_models:
            if m not in ratings: ratings[m] = trueskill.Rating()
            
        for i in range(NUM_ENVS):
            ships = final_ships[i]
            if is_4p[i]:
                ranks = [3, 3, 3, 3]
                order = np.argsort(-ships)
                for rank, p_idx in enumerate(order):
                    ranks[p_idx] = rank
                team_ratings = [(ratings[selected_models[0]],), (ratings[selected_models[1]],), (ratings[selected_models[2]],), (ratings[selected_models[3]],)]
                new_ratings = trueskill.rate(team_ratings, ranks=ranks)
                ratings[selected_models[0]] = new_ratings[0][0]
                ratings[selected_models[1]] = new_ratings[1][0]
                ratings[selected_models[2]] = new_ratings[2][0]
                ratings[selected_models[3]] = new_ratings[3][0]
            else:
                ranks = [1, 1]
                if ships[0] > ships[1]: ranks = [0, 1]
                elif ships[1] > ships[0]: ranks = [1, 0]
                team_ratings = [(ratings[selected_models[0]],), (ratings[selected_models[1]],)]
                new_ratings = trueskill.rate(team_ratings, ranks=ranks)
                ratings[selected_models[0]] = new_ratings[0][0]
                ratings[selected_models[1]] = new_ratings[1][0]
                
        save_ratings(ratings)
        time.sleep(2)

if __name__ == "__main__":
    import numpy as np
    main()
