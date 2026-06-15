import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import optax
from src.data_pipeline.dataset_grain import build_grain_dataloader
from src.models.entity_transformer_flax_small import EntityTransformer
from flax import nnx

model = EntityTransformer(rngs=nnx.Rngs(0))
loader = build_grain_dataloader(batch_size=10, worker_count=1, num_epochs=1)
it = iter(loader)
for _ in range(1000):
    batch = next(it)
    
    batch_state = batch['state_tokens']
    batch_launch = batch['target_launch']
    batch_angle = batch['target_angle']
    batch_ships = batch['target_ships']
    batch_owned = batch['is_owned_by_winner']
    
    v_logits, launch_logits, angle_logits, ships_logits, _ = model(
        batch_state, return_policy=True,
        target_launch=batch_launch, target_angle=batch_angle
    )
    
    p1_mask = (batch_owned == 1.0).astype(jnp.float32)
    safe_angle = jnp.maximum(batch_angle, 0)
    angle_ce = optax.softmax_cross_entropy_with_integer_labels(angle_logits, safe_angle)
    
    pos_launch_mask = p1_mask * (batch_launch == 1.0)
    valid_target_mask = pos_launch_mask * (batch_angle >= 0) * (batch_ships >= 0)
    
    # Check if angle_ce * valid_target_mask > 100
    high_loss = (angle_ce * valid_target_mask) > 100
    if jnp.any(high_loss):
        b, p = jnp.where(high_loss)
        for i in range(len(b)):
            print(f"\nBINGO! High loss at Batch {b[i]}, Source Planet {p[i]}")
            print(f"angle_ce: {angle_ce[b[i], p[i]]}")
            print(f"batch_angle (target): {batch_angle[b[i], p[i]]}")
            print(f"valid_target_mask: {valid_target_mask[b[i], p[i]]}")
            target = batch_angle[b[i], p[i]]
            print(f"angle_logit for target {target}: {angle_logits[b[i], p[i], target]}")
            print(f"is target {target} padded? state[b, target, 0] = {batch_state[b[i], target+1, 0]}")
            print(f"is source {p[i]} padded? state[b, source, 0] = {batch_state[b[i], p[i]+1, 0]}")
        exit(0)

print("No high loss found.")
