import json

with open("ppo_vs_braniac_test_replay.html", "r") as f:
    content = f.read()

start = content.find('window.kaggle = {') + 16
end = content.find('};</script>', start) + 1
json_str = content[start:end]

data = json.loads(json_str)
steps = data['steps']

for tick in range(100, 200):
    if tick >= len(steps): break
    obs = steps[tick][0]['observation']
    fleets = obs.get('fleets', [])
    for f in fleets:
        if f[1] == 0: # Player 1 (us)
            print(f"Tick {tick}: Fleet {f[0]} at ({f[2]:.1f}, {f[3]:.1f}), angle={f[4]:.2f}, ships={f[6]}")
