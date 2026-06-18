import re

with open('ppo_vs_braniac_test_replay.html', 'r') as f:
    html = f.read()

# The json starts somewhere after "window.kaggleEnv = {" or something similar.
# Let's just find "fleets":[  and parse it line by line
matches = re.finditer(r'"fleets": \[\s*([\s\S]*?)\s*\]', html)
for tick, match in enumerate(matches):
    fleet_str = match.group(1)
    # find all arrays inside it
    fleets = re.findall(r'\[(.*?)\]', fleet_str.replace('\n', ''))
    for f in fleets:
        parts = [p.strip() for p in f.split(',')]
        if len(parts) >= 7:
            try:
                ships = float(parts[6])
                if ships > 115 and ships < 125:
                    print(f"TICK {tick}: Fleet Owner={parts[1]}, pos={parts[2]},{parts[3]}, angle={parts[4]}, ships={parts[6]}")
            except:
                pass
