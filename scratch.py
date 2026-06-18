import math
import numpy as np

def get_speed(s):
    s = max(s, 1e-8)
    ratio = min(max(math.log(s) / math.log(1000.0), 0.0), 1.0)
    return 1.0 if s <= 1.0 else 1.0 + 5.0 * (ratio ** 1.5)

def check_if_fleet_hits(fx, fy, fangle, speed, tick, raw_planets, initial_x, initial_y, angular_velocity):
    dx_dir = math.cos(fangle)
    dy_dir = math.sin(fangle)
    best_t = float('inf')
    best_pid = -1
    for p in raw_planets:
        pid, owner, px, py, radius, ships, prod = p
        pid = int(pid)
        radius = float(radius)
        init_px = initial_x[pid] if initial_x is not None else float(px)
        init_py = initial_y[pid] if initial_y is not None else float(py)
        dist_to_sun = math.hypot(init_px - 50.0, init_py - 50.0)
        is_orbital = dist_to_sun > 1.0 and (dist_to_sun + radius) < 50.0
        if not is_orbital:
            proj = (float(px) - float(fx)) * dx_dir + (float(py) - float(fy)) * dy_dir
            if proj >= 0:
                perp_sq = (float(px) - float(fx))**2 + (float(py) - float(fy))**2 - proj**2
                if perp_sq < radius**2:
                    hit_dist = proj - math.sqrt(radius**2 - perp_sq)
                    t = max(0.0, hit_dist / speed)
                    if t < best_t:
                        best_t = t
                        best_pid = pid
        else:
            orb_r = dist_to_sun
            init_angle = math.atan2(init_py - 50.0, init_px - 50.0)
            for t in range(1, 2001):
                if float(t) > best_t:
                    break
                cur_angle = init_angle + angular_velocity * (tick + t)
                pos_x = 50.0 + orb_r * math.cos(cur_angle)
                pos_y = 50.0 + orb_r * math.sin(cur_angle)
                proj = (pos_x - float(fx)) * dx_dir + (pos_y - float(fy)) * dy_dir
                if proj >= 0:
                    perp_sq = (pos_x - float(fx))**2 + (pos_y - float(fy))**2 - proj**2
                    if perp_sq < radius**2:
                        hit_dist = proj - math.sqrt(radius**2 - perp_sq)
                        hit_t = hit_dist / speed
                        if abs(hit_t - t) <= 1.5:
                            if hit_t < best_t:
                                best_t = hit_t
                                best_pid = pid
                            break
    return best_t != float('inf'), best_pid, best_t

# Dummy data based on the first action in log: 
# P0 launched 1 ships from Planet 20 (Fleet 0)
# Actions generated: [[20, -1.6374884850542049, 1.0]]
# Let's say fx, fy is position of planet 20.
print("Script ready")
