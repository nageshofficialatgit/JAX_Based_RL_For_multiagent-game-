import json
import re

with open("ppo_vs_braniac_test_replay.html", "r") as f:
    content = f.read()

match = re.search(r'window\.kaggle\s*=\s*(\{.*?\});\s*</script>', content, re.DOTALL)
if match:
    data = json.loads(match.group(1))
    steps = data['steps']
    
    # Get planet 4 position at tick 66
    obs = steps[66][0]['observation']
    for p in obs['planets']:
        if p[0] == 4:
            print(f"Planet 4 at tick 66: x={p[2]:.1f}, y={p[3]:.1f}, r={p[4]:.1f}")
            
    # Get fleet info
    for tick in range(65, 75):
        obs = steps[tick][0]['observation']
        for f in obs.get('fleets', []):
            if f[0] in [92, 94, 96, 105]:
                print(f"Tick {tick}: Fleet {f[0]} at ({f[2]:.1f}, {f[3]:.1f}), angle={f[4]:.2f}, ships={f[6]}")
