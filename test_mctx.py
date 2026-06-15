import jax
import jax.numpy as jnp
from flax import nnx
import typing

try:
    import mctx
    MCTX_AVAILABLE = True
except ImportError:
    MCTX_AVAILABLE = False

from src.models.entity_transformer_flax_v2 import EntityTransformer
from src.env_jax.orbit_env_v2 import EnvState, build_observation, apply_actions, step_physics

MCTX_SIMULATIONS = 8
MCTX_CANDIDATES = 4

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
    
    return best_joint_action[:, 0], best_joint_action[:, 1], best_joint_action[:, 2]

if __name__ == "__main__":
    from src.env_jax.orbit_env_v2 import reset_env
    rngs = nnx.Rngs(0)
    model = EntityTransformer(num_features=37, num_classes=5, rngs=rngs)
    graphdef, state = nnx.split(model)
    
    env_state = reset_env(jax.random.PRNGKey(42))
    state_flat = jax.tree_util.tree_map(lambda x: x.value if hasattr(x, 'value') else x, state)
    
    l_act, t_act, s_act = run_mctx_search(jax.random.PRNGKey(1), env_state, 1, graphdef, state_flat)
    print("Search Output Shapes:", l_act.shape, t_act.shape, s_act.shape)
    print("MCTX Setup Successful!")
