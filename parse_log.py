import re

with open("agent_brain.log", "r") as f:
    lines = f.readlines()

tick = 0
for line in lines:
    if "[TICK" in line:
        m = re.search(r"\[TICK (\d+)\]", line)
        if m:
            tick = int(m.group(1))
    if 380 <= tick <= 400 and "121" in line:
        print(f"TICK {tick}: {line.strip()}")
