from flax import nnx
import jax

class Small(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(1)

class Big(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(1)
        self.w2 = nnx.Param(2)

s = Small()
b = Big()

g_s, state_s = nnx.split(s)
g_b, state_b = nnx.split(b)

try:
    nnx.merge(g_s, state_b)
except Exception as e:
    print("Merge g_s with state_b:", str(e))

try:
    nnx.merge(g_b, state_s)
except Exception as e:
    print("Merge g_b with state_s:", str(e))
