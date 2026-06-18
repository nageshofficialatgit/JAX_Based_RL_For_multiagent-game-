import sys, os, json
from kaggle_environments import make

sys.path.insert(0, os.getcwd())

env = make("orbit_wars", configuration={"size": 1000, "episodeSteps": 400})
env.run(["braniac_v2.py", "braniac_v2.py"])

# Dump tick 0 and tick 5 observation for player 0
for t in [0, 5]:
    obs = env.steps[t][0]["observation"]
    print(f"\n=== TICK {t} ===")
    print(f"Type of obs: {type(obs)}")
    print(f"Keys: {list(obs.keys()) if isinstance(obs, dict) else 'NOT A DICT'}")
    planets = obs.get("planets", [])
    print(f"Type of planets: {type(planets)}")
    if isinstance(planets, list) and len(planets) > 0:
        print(f"  planets[0] = {planets[0]}")
        print(f"  len(planets) = {len(planets)}")
    elif isinstance(planets, dict):
        first_key = list(planets.keys())[0]
        print(f"  planets[{first_key}] = {planets[first_key]}")
    fleets = obs.get("fleets", [])
    print(f"Type of fleets: {type(fleets)}")
    if isinstance(fleets, list) and len(fleets) > 0:
        print(f"  fleets[0] = {fleets[0]}")
    print(f"player = {obs.get('player')}")
    print(f"step = {obs.get('step')}")
