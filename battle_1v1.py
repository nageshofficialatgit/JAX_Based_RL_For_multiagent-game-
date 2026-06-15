import sys
import time
from training.arena import load_agent
from training.fast_sim import simulate
import numpy as np

def run_1v1():
    print("Loading agents...")
    agent_flax, _ = load_agent("agent_flax.py", "agent_flax")
    agent_v4, _ = load_agent("braniac_v4.py", "agent_v4")
    
    agents = [agent_flax, agent_v4]
    flax_wins = 0
    v4_wins = 0
    draws = 0
    
    n_games = 10
    print(f"Running {n_games} 1v1 games...")
    t0 = time.time()
    
    for seed in range(n_games):
        # fast_sim handles 2-player games if we pass exactly 2 agents!
        scores = simulate(agents, seed=seed+1000, max_steps=500)
        
        if scores[0] > scores[1]:
            flax_wins += 1
        elif scores[1] > scores[0]:
            v4_wins += 1
        else:
            draws += 1
            
        sys.stdout.write(f"\rCompleted: {seed+1}/{n_games} | Flax Wins: {flax_wins} | V4 Wins: {v4_wins} | Draws: {draws}")
        sys.stdout.flush()
        
    print(f"\n\nResults in {time.time()-t0:.1f}s:")
    print(f"Flax Agent (Step 41000) Winrate: {(flax_wins/n_games)*100:.1f}%")
    print(f"Braniac V4 Winrate: {(v4_wins/n_games)*100:.1f}%")

if __name__ == "__main__":
    run_1v1()
