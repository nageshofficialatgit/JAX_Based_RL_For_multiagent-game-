import os
from kaggle_environments import make

def generate_replay():
    print("Initializing Orbit Wars environment...")
    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    
    print("Running match: agent_bc.py vs random...")
    # Change 'random' to 'braniac_v2.py' if you want to see it lose to the advanced bot!
    env.run(["agent_bc.py", "random"])
    
    print("Saving replay to bc_v2_replay.html...")
    out_html = env.render(mode="html")
    with open("bc_v2_replay.html", "w", encoding="utf-8") as f:
        f.write(out_html)
        
    print("Done! You can open 'bc_v2_replay.html' in your browser to watch the match.")

if __name__ == "__main__":
    generate_replay()
