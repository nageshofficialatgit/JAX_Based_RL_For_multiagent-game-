import json

with open('ppo_vs_braniac_test_replay.html', 'r') as f:
    content = f.read()

start = content.find('window.KaggleState.steps = ')
end = content.find(';', start)
json_data = content[start + len('window.KaggleState.steps = '):end]
steps = json.loads(json_data)
for tick in range(380, 410):
    if tick < len(steps):
        obs = steps[tick][0]['observation']
        for f in obs.get('fleets', []):
            if f[6] > 100:  # Any large fleet
                print(f"TICK {tick}: Fleet ID={f[0]}, Owner={f[1]}, Pos=({f[2]:.2f}, {f[3]:.2f}), Angle={f[4]:.3f}, Ships={f[6]}")
