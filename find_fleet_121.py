import json

with open('ppo_vs_braniac_test_replay.html', 'r') as f:
    content = f.read()

# find the JSON payload inside the HTML
start = content.find('window.KaggleState.steps = ')
if start != -1:
    end = content.find(';', start)
    json_data = content[start + len('window.KaggleState.steps = '):end]
    steps = json.loads(json_data)
    for tick, step in enumerate(steps):
        obs = step[0]['observation']
        for f in obs.get('fleets', []):
            if f[0] == 121:
                print(f"TICK {tick}: Fleet 121 exists! Pos: x={f[2]:.2f}, y={f[3]:.2f}, Ships={f[6]}")
