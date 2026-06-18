import sys
import os
import argparse
import random
import multiprocessing
from kaggle_environments import make

def play_match(seed):
    env = make("orbit_wars", configuration={"seed": seed, "episodeSteps": 500, "actTimeout": 10.0}, debug=False)
    # Play our agent against braniac
    env.run(["agent_bc.py", "braniac_v2.py", "braniac_v2.py", "braniac_v2.py"])
    
    html = env.render(mode="html")
    with open("artifacts/ppo_vs_braniac_replay.html", "w") as f:
        f.write(html)
        
    steps = env.steps
    final_step = steps[-1]
    
    r1 = final_step[0].reward
    r2 = final_step[1].reward
    
    if r1 is None: r1 = -1
    if r2 is None: r2 = -1
    
    if r1 > r2:
        return 1
    elif r1 < r2:
        return 0
    else:
        return 0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()
    
    print(f"Starting {args.games} matches of BC Agent vs Random in true Kaggle Environment...")
    
    # Clear the brain log
    with open("agent_brain.log", "w") as f:
        f.write("=== ORBIT WARS AGENT BRAIN LOG ===\n")
    
    seeds = [random.randint(0, 10000) for _ in range(args.games)]
    
    # We can run sequentially for now to catch errors
    wins = 0
    losses = 0
    ties = 0
    
    for i, seed in enumerate(seeds):
        try:
            res = play_match(seed)
            if res == 1:
                wins += 1
                print(f"Game {i+1}: BC Agent WON! (Score: W {wins} - L {losses} - T {ties})")
            elif res == 0:
                losses += 1
                print(f"Game {i+1}: BC Agent LOST. (Score: W {wins} - L {losses} - T {ties})")
            else:
                ties += 1
                print(f"Game {i+1}: TIE. (Score: W {wins} - L {losses} - T {ties})")
        except Exception as e:
            print(f"Game {i+1} Failed with error: {e}")
            
    win_rate = wins / args.games
    print(f"\nFinal Win Rate: {win_rate * 100:.2f}% ({wins}W, {losses}L, {ties}T)")

if __name__ == "__main__":
    main()
