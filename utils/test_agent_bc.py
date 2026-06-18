import jax
import jax.numpy as jnp
from agent_bc import BCAgent

obs = {
    "planets": [
        [0, 1, 10.0, 10.0, 5.0, 100, 2],
        [1, 2, 90.0, 90.0, 5.0, 100, 2],
        [2, -1, 50.0, 50.0, 10.0, 0, 5],
    ],
    "player": 1,
    "angular_velocity": 0.0,
    "step": 0
}

agent = BCAgent()
moves = agent(obs)
print("Moves:")
for m in moves:
    print(m)
