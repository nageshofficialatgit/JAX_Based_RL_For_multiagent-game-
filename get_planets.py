import json

with open("ppo_vs_braniac_test_replay.html", "r") as f:
    content = f.read()

start = content.find('"planets": [')
if start != -1:
    end = content.find('],', start) + 1
    planets_str = "{" + content[start:end] + "}"
    try:
        data = json.loads(planets_str)
        for p in data['planets']:
            print(f"Planet {p[0]}: x={p[2]:.1f}, y={p[3]:.1f}, r={p[4]:.1f}")
    except Exception as e:
        print("Failed", e)
