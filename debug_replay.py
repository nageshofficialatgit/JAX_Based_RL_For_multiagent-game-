import json

with open("ppo_vs_braniac_test_replay.html", "r") as f:
    content = f.read()

# The json is typically in a script tag or something. Let's find it.
start_idx = content.find('"steps": [')
if start_idx != -1:
    # Find the start of the JSON object
    obj_start = content.rfind('{', 0, start_idx)
    # Find the end of the JSON object (roughly)
    # Actually, it's easier to just parse the lines
    pass

import re
match = re.search(r'window\.kaggle = (.*?);</script>', content, re.DOTALL)
if match:
    data = json.loads(match.group(1))
    steps = data['steps']
    print(f"Total steps: {len(steps)}")
    
    fleets_spawned = {}
    fleets_hit = set()
    fleets_out = set()
    
    for step_idx, step in enumerate(steps):
        obs = step[0]['observation']
        fleets = obs.get('fleets', [])
        
        # Track fleets
        for f in fleets:
            fid = f[0]
            if fid not in fleets_spawned:
                fleets_spawned[fid] = {
                    'step': step_idx,
                    'owner': f[1],
                    'x': f[2],
                    'y': f[3],
                    'angle': f[4],
                    'src': f[5],
                    'ships': f[6]
                }
    print(f"Total fleets spawned: {len(fleets_spawned)}")
    
    # We can't easily track hits from just the observations, but if a fleet disappears, it hit something.
    
    # Let's just print the first 10 fleets spawned by player 1 (owner 1)
    print("Player 1 fleets:")
    count = 0
    for fid, f in fleets_spawned.items():
        if f['owner'] == 1:
            print(f"Fleet {fid}: src {f['src']} -> angle {f['angle']:.2f}, ships {f['ships']}")
            count += 1
            if count > 10: break

