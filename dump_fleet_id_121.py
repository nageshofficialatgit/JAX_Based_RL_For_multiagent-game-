import re

with open('ppo_vs_braniac_test_replay.html', 'r') as f:
    html = f.read()

matches = re.finditer(r'"fleets": \[\s*([\s\S]*?)\s*\]', html)
for tick, match in enumerate(matches):
    fleet_str = match.group(1)
    fleets = re.findall(r'\[(.*?)\]', fleet_str.replace('\n', ''))
    for f in fleets:
        parts = [p.strip() for p in f.split(',')]
        if len(parts) >= 7:
            try:
                fid = int(parts[0])
                if fid == 121:
                    print(f"TICK {tick}: Fleet Owner={parts[1]}, pos={parts[2]},{parts[3]}, angle={parts[4]}, ships={parts[6]}")
            except:
                pass
