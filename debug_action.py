import json
import sys
import jax.numpy as jnp
import math

# Try to parse the replay JSON to get the first tick state
try:
    with open("ppo_vs_braniac_test_replay.html", "r") as f:
        content = f.read()
        json_start = content.find("var replayData = ") + len("var replayData = ")
        json_end = content.find(";\n", json_start)
        data = json.loads(content[json_start:json_end])
        
    steps = data["steps"]
    # get step 0
    obs_0 = steps[0][0]["observation"]
    
    # We will simulate test_ppo_agent.py logic for step 0
    # Let's import parse_kaggle_obs
    from test_ppo_agent import parse_kaggle_obs, check_if_fleet_hits, get_speed
    
    initial_x = [0.0]*50
    initial_y = [0.0]*50
    for p in obs_0['planets']:
        pid = int(p[0])
        initial_x[pid] = float(p[2])
        initial_y[pid] = float(p[3])
        
    env_state = parse_kaggle_obs(obs_0, raw_player=0, initial_x=initial_x, initial_y=initial_y)
    
    print("Env State parsed.")
    # In agent_brain.log, tick 0 player 1 generated: [[20, -1.6374884850542049, 1.0]]
    # This means src_idx = 20, target_idx = ? 
    # Let's manually run check_if_fleet_hits for src_idx 20 and angle -1.637488
    
    src_idx = 20
    angle = -1.6374884850542049
    ships = 1.0 # wait, if it was overridden, what was the original bucket? 
    # The agent log says l_logits max: 8.049. So it launched.
    # It probably targeted the Sun or another planet. 
    
    fx = float(env_state.planet_x[src_idx])
    fy = float(env_state.planet_y[src_idx])
    speed = get_speed(ships)
    angular_velocity = float(obs_0.get('angular_velocity', 0.02))
    raw_planets = obs_0.get('planets', [])
    
    will_hit = check_if_fleet_hits(
        fx, fy, angle, speed, 0, raw_planets, 
        initial_x, initial_y, angular_velocity
    )
    print(f"Angle: {angle}, will_hit: {will_hit}")
    
    # Let's find which planet it was aiming for!
    for t_idx in range(50):
        if t_idx == src_idx: continue
        px = initial_x[t_idx]
        py = initial_y[t_idx]
        a = math.atan2(py - fy, px - fx)
        # print diff
        if abs(a - angle) < 0.1:
            print(f"Likely aimed at {t_idx} (angle {a})")
            
except Exception as e:
    import traceback
    traceback.print_exc()
