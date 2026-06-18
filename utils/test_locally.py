import os
import sys
import tarfile
import tempfile
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from kaggle_environments import make

def run_single_match(match_args):
    seed, agents, mode = match_args
    try:
        # We must disable debug printing when running in parallel to avoid terminal spam
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)
        
        final_step = env.steps[-1]
        
        # Player 0 is our submission
        p0_reward = final_step[0].reward
        
        if mode == "2p":
            max_enemy = final_step[1].reward
        else:
            max_enemy = max(final_step[1].reward, final_step[2].reward, final_step[3].reward)
            
        is_win = p0_reward is not None and max_enemy is not None and p0_reward > max_enemy
        is_tie = p0_reward is not None and max_enemy is not None and p0_reward == max_enemy
        
        return {"seed": seed, "win": is_win, "tie": is_tie, "reward": p0_reward, "enemy_reward": max_enemy}
    except Exception as e:
        return {"seed": seed, "error": str(e)}

def run_local_evaluation(submission_tar, opponent_script, num_games, mode):
    if opponent_script != "random" and not os.path.exists(opponent_script):
        print(f"ERROR: Opponent {opponent_script} not found!")
        sys.exit(1)

    agent_path = submission_tar
            
    agents = [agent_path, opponent_script]
    if mode == "4p":
        agents = [agent_path, opponent_script, opponent_script, opponent_script]
        
    print(f"--- 2. Starting {num_games}x {mode.upper()} Arena Matches ---")
    print(f"Submission vs {opponent_script}")
    
    match_args = [(42 + i, agents, mode) for i in range(num_games)]
        
    results = []
    wins = 0
    ties = 0
    losses = 0
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=min(num_games, os.cpu_count() or 4)) as executor:
        futures = {executor.submit(run_single_match, args): args for args in match_args}
        
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if "error" in res:
                print(f"Game {i+1}/{num_games} (Seed {res['seed']}) - ERROR: {res['error']}")
                continue
                
            if res["win"]:
                wins += 1
                status = "WIN 🏆"
            elif res["tie"]:
                ties += 1
                status = "TIE 🤝"
            else:
                losses += 1
                status = "LOSS 💀"
                
            print(f"Game {i+1}/{num_games} (Seed {res['seed']}) - {status} | Score: {res['reward']} vs {res['enemy_reward']}")
            
    print("\n--- 3. Final Arena Results ---")
    print(f"Total Games : {num_games}")
    print(f"Wins        : {wins} ({(wins/num_games)*100:.1f}%)")
    print(f"Losses      : {losses} ({(losses/num_games)*100:.1f}%)")
    print(f"Ties        : {ties} ({(ties/num_games)*100:.1f}%)")
    
    if mode == "4p":
        print("Note: In 4P mode, ties or 2nd place are considered losses on Kaggle.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default="submissions/v1/submission.tar.gz", help="Path to submission tarball")
    parser.add_argument("--opp", type=str, default="braniac_v2.py", help="Path to opponent script")
    parser.add_argument("--games", type=int, default=10, help="Number of games to simulate")
    parser.add_argument("--mode", type=str, choices=["2p", "4p"], default="2p", help="Match type (2p or 4p)")
    args = parser.parse_args()
    
    run_local_evaluation(args.sub, args.opp, args.games, args.mode)