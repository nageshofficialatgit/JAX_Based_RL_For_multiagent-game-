import json
import re

with open('ppo_vs_braniac_test_replay.html', 'r') as f:
    html = f.read()

# The JSON is assigned to window.kaggleEnv
match = re.search(r'window\.kaggleEnv\s*=\s*(\{.*?\});\n', html, re.DOTALL)
if match:
    kaggle_env = json.loads(match.group(1))
    steps = kaggle_env.get('steps', [])
    if len(steps) > 398:
        obs = steps[398][0]['observation']
        print("Fleets at tick 398:")
        for f in obs.get('fleets', []):
            print(f"  ID={f[0]}, Owner={f[1]}, Pos=({f[2]:.2f}, {f[3]:.2f}), Angle={f[4]:.3f}, Source={f[5]}, Ships={f[6]}")
    else:
        print("Tick 398 not found")
else:
    print("Could not find window.kaggleEnv")
