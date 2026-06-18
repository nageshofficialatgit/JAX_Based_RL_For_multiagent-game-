import math

def test_intercept():
    src_x, src_y = 10.0, 10.0
    dist_to_sun = 30.0
    angular_velocity = 0.02
    tick = 0
    speed = 2.0
    radius = 2.0
    
    init_angle = 0.0 # planet starts at (80, 50)
    
    # 1. get_intercept logic
    best_t = float('inf')
    best_angle = 0.0
    for t_sim in range(1, 2000):
        a = init_angle + angular_velocity * (tick + t_sim)
        px = 50.0 + dist_to_sun * math.cos(a)
        py = 50.0 + dist_to_sun * math.sin(a)
        
        req_t = math.hypot(px - src_x, py - src_y) / speed
        if req_t <= float(t_sim):
            best_t = req_t
            best_angle = math.atan2(py - src_y, px - src_x)
            break
            
    print(f"Launched at angle {best_angle} to hit at t={best_t}")
    
    # 2. parse_kaggle_obs raycast logic
    fx, fy = src_x, src_y
    dx_dir = math.cos(best_angle)
    dy_dir = math.sin(best_angle)
    
    hit_best_t = float('inf')
    for t in range(1, 2001):
        if float(t) > hit_best_t: break
        
        cur_angle = init_angle + angular_velocity * (tick + t)
        pos_x = 50.0 + dist_to_sun * math.cos(cur_angle)
        pos_y = 50.0 + dist_to_sun * math.sin(cur_angle)
        
        proj = (pos_x - fx) * dx_dir + (pos_y - fy) * dy_dir
        if proj >= 0:
            perp_sq = (pos_x - fx)**2 + (pos_y - fy)**2 - proj**2
            if perp_sq < radius**2:
                hit_dist = proj - math.sqrt(radius**2 - perp_sq)
                hit_t = hit_dist / speed
                if abs(hit_t - t) <= 1.5:
                    if hit_t < hit_best_t:
                        hit_best_t = hit_t
                        print(f"Raycast HIT at t={t}, hit_t={hit_t}")
                    break
                    
    print(f"Raycast final hit_t={hit_best_t}")

test_intercept()
