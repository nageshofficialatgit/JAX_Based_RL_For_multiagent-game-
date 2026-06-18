import re

with open('ppo_vs_braniac_test_replay.html', 'r') as f:
    for line in f:
        # Each step is typically serialized as a JSON array inside the HTML, or we can just regex for the fleets arrays
        # Kaggle state is: window.KaggleState = { ... };
        # The steps are inside window.kaggleEnv.steps
        pass

# simpler approach: just find all occurrences of "121" in the file and see context
import subprocess
output = subprocess.check_output(['grep', '-n', '-A', '2', '-B', '2', '121', 'ppo_vs_braniac_test_replay.html']).decode()
for i, block in enumerate(output.split('--')):
    if '121' in block:
        print(f"Block {i}:\n{block}")

