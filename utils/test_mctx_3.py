import jax
import jax.numpy as jnp
import mctx

def test(dummy_arg):
    def root_fn(rng_key):
        return mctx.RootFnOutput(prior_logits=jnp.zeros((1, 8)), value=jnp.zeros(1), embedding=jnp.zeros((1, 5)))
        
    def recurrent_fn(params, rng_key, action, embedding):
        return mctx.RecurrentFnOutput(
            reward=jnp.zeros((1,)), 
            discount=jnp.ones((1,)), 
            prior_logits=jnp.zeros((1, 8)), 
            value=jnp.zeros(1)
        ), embedding

    def do_mctx(args):
        policy_output = mctx.muzero_policy(
            params=None,
            rng_key=jax.random.PRNGKey(0),
            root=root_fn(jax.random.PRNGKey(0)),
            recurrent_fn=recurrent_fn,
            num_simulations=16,
            dirichlet_fraction=0.0,
            temperature=0.0
        )
        return policy_output.action
        
    def do_greedy(args):
        return jnp.zeros((1,), dtype=jnp.int32)
        
    return jax.lax.switch(0, [do_greedy, do_mctx], None)

print(jax.vmap(test)(jnp.ones(10)))
