import os
import sys
import json
from kaggle_environments import make

def main():
    env = make("orbit_wars", configuration={"seed": 789792789}, debug=True)
    
    agents = ["test_ppo_agent.py", "test_ppo_agent.py"]
    
    print("Starting match between test_ppo_agent.py vs test_ppo_agent.py...")
    steps = env.run(agents)
    
    print("Match finished!")
    final_state = steps[-1]
    for i, state in enumerate(final_state):
        print(f"Player {i+1} status: {state.status}, reward: {state.reward}")
        
    out_file = "ppo_vs_braniac_test_replay.html"
    with open(out_file, "w") as f:
        f.write(env.render(mode="html"))
    print(f"Replay saved to {out_file}")

if __name__ == "__main__":
    main()
