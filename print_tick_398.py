import json

with open('ppo_vs_braniac_test_replay.html', 'r') as f:
    content = f.read()

start = content.find('window.KaggleState.steps = ')
end = content.find(';', start)
json_data = content[start + len('window.KaggleState.steps = '):end]

try:
    steps = json.loads(json_data)
except json.JSONDecodeError as e:
    # try manual regex parsing just for step 398
    pass

import re
matches = list(re.finditer(r'"fleets": \[\s*([\s\S]*?)\s*\]', content))
if len(matches) > 398:
    fleet_str = matches[398].group(1)
    fleets = re.findall(r'\[(.*?)\]', fleet_str.replace('\n', ''))
    print("Fleets at tick 398:")
    for f in fleets:
        print(f"  {f}")
else:
    print("Not enough ticks")
