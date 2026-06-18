import json

with open("ppo_vs_braniac_test_replay.html", "r") as f:
    content = f.read()

import re
# The json is inside window.kaggle = {...};
match = re.search(r'window\.kaggle\s*=\s*(\{.*?\});\s*</script>', content, re.DOTALL)
if match:
    data = json.loads(match.group(1))
    steps = data['steps']
    
    # Let's find fleets that are active around tick 60-80
    for tick in range(50, 100):
        if tick >= len(steps): break
        obs = steps[tick][0]['observation']
        fleets = obs.get('fleets', [])
        for f in fleets:
            if f[1] == 0: # Player 1 (index 0)
                # fid, owner, x, y, angle, src, ships
                print(f"Tick {tick}: Fleet {f[0]} at ({f[2]:.1f}, {f[3]:.1f}), angle={f[4]:.2f}, ships={f[6]}")
