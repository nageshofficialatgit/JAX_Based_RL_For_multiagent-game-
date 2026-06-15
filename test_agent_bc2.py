import jax
import jax.numpy as jnp
import numpy as np
from agent_bc import agent

obs = {
    "planets": [[i, 1 if i < 5 else 2 if i < 10 else -1, float(np.random.rand()*100), float(np.random.rand()*100), 5.0, 100, 2] for i in range(50)],
    "fleets": [],
    "player": 1,
    "angular_velocity": 0.05,
    "step": 100
}

print("Running test inference with new BC Agent...")
moves = agent(obs)

print(f"Moves generated: {moves}")

print("\nReading agent_brain.log output:")
try:
    with open("agent_brain.log", "r") as f:
        print(f.read()[-1000:])
except Exception as e:
    print(f"Error reading log: {e}")
