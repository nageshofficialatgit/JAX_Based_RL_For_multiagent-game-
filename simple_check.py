import orbax.checkpoint as ocp
import jax.numpy as jnp
import os

mngr = ocp.CheckpointManager(os.path.abspath('checkpoints/ppo_v2'))
step = mngr.latest_step()
if step is None:
    print("No checkpoint found")
    exit(0)
    
print(f"Loading step {step}")
raw = mngr.restore(step)

if 'model' in raw:
    state = raw['model']
else:
    state = raw
    
has_nan = False
has_inf = False
max_val = 0.0

def traverse(d, path=""):
    global has_nan, has_inf, max_val
    if isinstance(d, dict):
        for k, v in d.items():
            traverse(v, path + "." + str(k))
    else:
        try:
            val = jnp.asarray(d)
            if jnp.isnan(val).any():
                print(f"NaN in {path}")
                has_nan = True
            if jnp.isinf(val).any():
                print(f"Inf in {path}")
                has_inf = True
            m_val = float(jnp.max(jnp.abs(val)))
            if m_val > max_val:
                max_val = m_val
        except Exception as e:
            pass

traverse(state)
print(f"Overall Max Abs Weight: {max_val:.2f}")
if has_nan:
    print("Model CONTAINS NaNs!")
elif has_inf:
    print("Model CONTAINS Infs!")
else:
    print("Model is clean of NaNs and Infs.")
