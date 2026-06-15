import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from src.models.entity_transformer_flax_v2 import EntityTransformer

def loss_fn(s_flat, graphdef, batch, current_entropy, current_clip):
    merged = nnx.merge(graphdef, s_flat)
    
    v_logits, launch_logits, ships_logits, angle_logits, ppo_v = merged(
        batch['obs'], return_policy=True, target_launch=batch['a_launch'], target_angle=batch['a_angle'], valid_launch_mask=batch['valid_launch_mask'])
        
    launch_prob = jax.nn.sigmoid(launch_logits)
    p_safe = jnp.clip(launch_prob, 1e-7, 1.0 - 1e-7)
    launch_lp = jnp.log(jnp.where(batch['a_launch'], p_safe, 1.0 - p_safe))
    
    angle_lp = jax.nn.log_softmax(angle_logits)
    angle_lp = jnp.take_along_axis(angle_lp, batch['a_angle'][..., None], axis=-1)[..., 0]
    
    ships_lp = jax.nn.log_softmax(ships_logits)
    ships_lp = jnp.take_along_axis(ships_lp, batch['a_ships'][..., None], axis=-1)[..., 0]
    
    valid_launch_mask = batch['valid_launch_mask']
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
    value_loss = jnp.mean(jnp.square(value_pred - batch['ret']))
    
    launch_ent_raw = -(launch_prob * jnp.log(launch_prob + 1e-8) + (1.0 - launch_prob) * jnp.log(1.0 - launch_prob + 1e-8))
    
    angle_probs = jnp.nan_to_num(jax.nn.softmax(angle_logits, axis=-1))  
    target_ent_raw = -jnp.sum(angle_probs * jnp.log(angle_probs + 1e-8), axis=-1)
    
    ships_probs = jnp.nan_to_num(jax.nn.softmax(ships_logits, axis=-1)) 
    ships_ent_raw = -jnp.sum(ships_probs * jnp.log(ships_probs + 1e-8), axis=-1)
    
    valid_entropies = (launch_ent_raw + launch_prob * (0.5 * target_ent_raw + 0.3 * ships_ent_raw)) * valid_launch_mask
    entropy = jnp.sum(valid_entropies) / (jnp.sum(valid_launch_mask) + 1e-8)
    
    total_loss = policy_loss + 0.5 * value_loss - current_entropy * entropy

    jax.debug.print("policy_loss: {}", policy_loss)
    jax.debug.print("value_loss: {}", value_loss)
    jax.debug.print("entropy: {}", entropy)
    
    metrics = {
        'loss': total_loss, 'policy_loss': policy_loss, 'value_loss': value_loss, 'entropy': entropy
    }
    return total_loss, metrics

# Setup dummy batch
batch_size = 2
batch = {
    'obs': jnp.zeros((batch_size, 50, 37)),
    'a_launch': jnp.zeros((batch_size, 50), dtype=jnp.int32),
    'a_angle': jnp.zeros((batch_size, 50), dtype=jnp.int32),
    'a_ships': jnp.zeros((batch_size, 50), dtype=jnp.int32),
    'old_lp': jnp.zeros((batch_size, 50)),
    'adv': jnp.zeros((batch_size,)),
    'ret': jnp.zeros((batch_size,)),
    'valid_launch_mask': jnp.ones((batch_size, 50), dtype=jnp.bool_)
}

rngs = nnx.Rngs(0)
model = EntityTransformer(num_features=37, rngs=rngs)
graphdef, s_flat = nnx.split(model)

print("Calling value_and_grad...")
grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
(loss, metrics), grads = grad_fn(s_flat, graphdef, batch, 0.01, 0.2)

def check_nan(x):
    if hasattr(x, 'value'):
        x = x.value
    return jnp.isnan(x).any()

has_nan = jax.tree_util.tree_map(check_nan, grads)
nan_count = sum(jax.tree_util.tree_leaves(has_nan))
print(f"Number of NaN gradient arrays: {nan_count}")
print("Done.")
