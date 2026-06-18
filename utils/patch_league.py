import re

with open("src/training/league_evaluator_jax.py", "r") as f:
    content = f.read()

# 1. Add imports
content = content.replace("import trueskill\n", "import trueskill\nimport mctx\nimport flax.struct as struct\n")

# 2. Change NUM_ENVS
content = content.replace("NUM_ENVS = 1024", "NUM_ENVS = 512")

# 3. Add get_action_greedy and get_action_mctx
mctx_code = """
def get_action_greedy(obs, merged_model, rng):
    _, l_logits, a_logits, s_logits, _ = merged_model(obs[None, ...], return_policy=True)
    l_act = jnp.round(jax.nn.sigmoid(l_logits[0])).astype(jnp.int32)
    t_act = jnp.argmax(a_logits[0], axis=-1).astype(jnp.int32)
    s_act = jnp.argmax(s_logits[0], axis=-1).astype(jnp.int32)
    return l_act, t_act, s_act

@struct.dataclass
class MCTSEmbedding:
    env_state: EnvState
    l_acts: jnp.ndarray
    t_acts: jnp.ndarray
    s_acts: jnp.ndarray

def generate_candidates(obs, merged_model, rng):
    _, l_logits, a_logits, s_logits, _ = merged_model(obs[None, ...], return_policy=True)
    def sample_one(r):
        r1, r2, r3 = jax.random.split(r, 3)
        l_act = jax.random.bernoulli(r1, jax.nn.sigmoid(l_logits[0])).astype(jnp.int32)
        t_act = jax.random.categorical(r2, a_logits[0], axis=-1)
        s_act = jax.random.categorical(r3, s_logits[0], axis=-1)
        return l_act, t_act, s_act
    keys = jax.random.split(rng, 8)
    return jax.vmap(sample_one)(keys)

def get_action_mctx(obs, env_state, merged_model, rng, player_id):
    r_root, r_sim = jax.random.split(rng)
    
    def root_fn(rng_key):
        _, _, _, _, value = merged_model(obs[None, ...], return_policy=True)
        l, t, s = generate_candidates(obs, merged_model, rng_key)
        embedding = MCTSEmbedding(env_state=env_state, l_acts=l, t_acts=t, s_acts=s)
        return mctx.RootFnOutput(prior_logits=jnp.zeros(8), value=value[0], embedding=embedding)
        
    def recurrent_fn(params, rng_key, action, embedding):
        l_act = embedding.l_acts[action]
        t_act = embedding.t_acts[action]
        s_act = embedding.s_acts[action]
        
        new_st = apply_actions(embedding.env_state, player_id, l_act, t_act, s_act)
        new_st = step_physics(new_st)
        
        new_obs = build_observation(new_st, player_id, 1.0)
        _, _, _, _, value = merged_model(new_obs[None, ...], return_policy=True)
        new_l, new_t, new_s = generate_candidates(new_obs, merged_model, rng_key)
        new_embedding = MCTSEmbedding(env_state=new_st, l_acts=new_l, t_acts=new_t, s_acts=new_s)
        
        return mctx.RecurrentFnOutput(reward=jnp.zeros_like(value[0]), discount=jnp.ones_like(value[0]), prior_logits=jnp.zeros(8), value=value[0]), new_embedding

    policy_output = mctx.muzero_policy(
        params=None, rng_key=r_sim, root=root_fn(r_root), recurrent_fn=recurrent_fn,
        num_simulations=16, dirichlet_fraction=0.0, temperature=0.0
    )
    l, t, s = generate_candidates(obs, merged_model, r_root)
    return l[policy_output.action], t[policy_output.action], s[policy_output.action]

"""

content = content.replace("@jax.jit\ndef batch_simulate(", mctx_code + "\n@jax.jit\ndef batch_simulate(")

# 4. Update batch_simulate signature and switch logic
content = content.replace("def batch_simulate(rng_keys, models_flat, graphdef):", "def batch_simulate(rng_keys, models_flat, graphdef, agent_types):")

switch_logic = """
            def do_temp(args): o, m, r, s, p = args; return get_action(o, m, r)
            def do_greedy(args): o, m, r, s, p = args; return get_action_greedy(o, m, r)
            def do_mctx(args): o, m, r, s, p = args; return get_action_mctx(o, s, m, r, p)
            branches = [do_temp, do_greedy, do_mctx]
            
            l1, t1, s1 = jax.lax.switch(agent_types[0], branches, (o1, m1, r1, st, 1))
            l2, t2, s2 = jax.lax.switch(agent_types[1], branches, (o2, m2, r2, st, 2))
            l3, t3, s3 = jax.lax.switch(agent_types[2], branches, (o3, m3, r3, st, 3))
            l4, t4, s4 = jax.lax.switch(agent_types[3], branches, (o4, m4, r4, st, 4))
"""

old_logic = """            l1, t1, s1 = get_action(o1, m1, r1)
            l2, t2, s2 = get_action(o2, m2, r2)
            l3, t3, s3 = get_action(o3, m3, r3)
            l4, t4, s4 = get_action(o4, m4, r4)"""

content = content.replace(old_logic, switch_logic)

# 5. Update main get_model_state
new_get_model_state = """    def get_model_state(model_name):
        if model_name.startswith("random"):
            _, s = nnx.split(EntityTransformer(num_classes=5, rngs=nnx.Rngs(int(time.time()))))
            return jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, s)
        
        base_name = model_name
        if base_name.endswith("_temp"): base_name = base_name[:-5]
        elif base_name.endswith("_greedy"): base_name = base_name[:-7]
        elif base_name.endswith("_mctx"): base_name = base_name[:-5]
        
        bin_path = os.path.join(BENCHMARK_DIR, f"{base_name}.bin")
        if not os.path.exists(bin_path): return None
        
        with open(bin_path, "rb") as f:
            raw = f.read()
        return serialization.from_bytes(
            jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, state_template), raw
        )"""

old_get_model_state = """    def get_model_state(model_name):
        if model_name == "random":
            _, s = nnx.split(EntityTransformer(num_classes=5, rngs=nnx.Rngs(int(time.time()))))
            return jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, s)
        
        bin_path = os.path.join(BENCHMARK_DIR, f"{model_name}.bin")
        if not os.path.exists(bin_path): return None
        
        with open(bin_path, "rb") as f:
            raw = f.read()
        return serialization.from_bytes(
            jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, state_template), raw
        )"""

content = content.replace(old_get_model_state, new_get_model_state)

# 6. Update models_available
new_models = """        models_available = ["random", "random_2"]
        for f in bin_files:
            base = os.path.basename(f).replace(".bin", "")
            models_available.extend([f"{base}_temp", f"{base}_greedy", f"{base}_mctx"])"""

old_models = """        models_available = [os.path.basename(f).replace(".bin", "") for f in bin_files] + ["random", "random_2"]"""

content = content.replace(old_models, new_models)

# 7. Update batch_simulate call in main
new_call = """        def get_type_id(name):
            if name.endswith("_greedy"): return 1
            if name.endswith("_mctx"): return 2
            return 0
        agent_types = jnp.array([get_type_id(m) for m in selected_models], dtype=jnp.int32)
        final_ships, is_4p = batch_simulate(keys, states, graphdef, agent_types)"""

old_call = """        final_ships, is_4p = batch_simulate(keys, states, graphdef)"""

content = content.replace(old_call, new_call)

with open("src/training/league_evaluator_jax.py", "w") as f:
    f.write(content)
