import sys
import os
from kaggle_environments import make

sys.path.insert(0, os.getcwd())

print("Initializing Orbit Environment...")
env = make("orbit_wars", configuration={"size": 1000, "episodeSteps": 400})

print("Simulating GA Evolved Bot vs Braniac V2...")
# JAX agent plays as Player 1, Braniac plays as Player 2
env.run(["heuristic_bots/jax_agent.py", "braniac_v2.py"])

output_path = "ga_vs_braniac_replay.html"
with open(output_path, "w") as f:
    f.write(env.render(mode="html"))
    
print(f"Match saved to {output_path}")

rewards = env.steps[-1][0]["reward"], env.steps[-1][1]["reward"]
print(f"Final Rewards: GA Agent (P1): {rewards[0]} | Braniac V2 (P2): {rewards[1]}")
if rewards[0] > rewards[1]:
    print("GA Agent won!")
elif rewards[0] < rewards[1]:
    print("Braniac V2 won!")
else:
    print("Draw!")
