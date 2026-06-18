import jax
import jax.numpy as jnp
import time

@jax.jit
def intercept_vmap(src_x, src_y, dist_to_sun, init_angle, angular_velocity, tick, speed):
    t_arr = jnp.arange(1, 151, dtype=jnp.float32)
    a_arr = init_angle + angular_velocity * (tick + t_arr)
    px = 50.0 + dist_to_sun * jnp.cos(a_arr)
    py = 50.0 + dist_to_sun * jnp.sin(a_arr)
    
    req_t = jnp.hypot(px - src_x, py - src_y) / speed
    valid_mask = req_t <= t_arr
    
    # We want the FIRST true index
    best_idx = jnp.argmax(valid_mask)
    best_px = px[best_idx]
    best_py = py[best_idx]
    best_angle = jnp.arctan2(best_py - src_y, best_px - src_x)
    return best_angle, t_arr[best_idx]

# Test cases
print(intercept_vmap(10.0, 10.0, 30.0, 0.0, 0.02, 0, 2.0))
print(intercept_vmap(90.0, 90.0, 45.0, 3.14, -0.05, 100, 1.5))
print(intercept_vmap(50.0, 50.0, 20.0, 1.0, 0.01, 50, 5.0))
