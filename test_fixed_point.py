import math

def test_intercept(src_x, src_y, dist_to_sun, init_angle, angular_velocity, tick, speed):
    # 1. Linear search (as in agent_bc.py)
    best_t_lin = float('inf')
    best_angle_lin = 0.0
    for t_sim in range(1, 2000):
        a = init_angle + angular_velocity * (tick + t_sim)
        px = 50.0 + dist_to_sun * math.cos(a)
        py = 50.0 + dist_to_sun * math.sin(a)
        
        req_t = math.hypot(px - src_x, py - src_y) / speed
        if req_t <= float(t_sim):
            best_t_lin = req_t
            best_angle_lin = math.atan2(py - src_y, px - src_x)
            break
            
    # 2. Fixed point iteration (JAX friendly)
    t_fixed = 0.0
    for i in range(20):
        a = init_angle + angular_velocity * (tick + t_fixed)
        px = 50.0 + dist_to_sun * math.cos(a)
        py = 50.0 + dist_to_sun * math.sin(a)
        t_fixed = math.hypot(px - src_x, py - src_y) / speed
        
    best_angle_fixed = math.atan2(py - src_y, px - src_x)
    
    print(f"Linear: t={best_t_lin:.4f}, angle={best_angle_lin:.6f}")
    print(f"FixedP: t={t_fixed:.4f}, angle={best_angle_fixed:.6f}")
    print(f"Diff  : {abs(best_angle_lin - best_angle_fixed):.6e}")

test_intercept(10.0, 10.0, 30.0, 0.0, 0.02, 0, 2.0)
test_intercept(90.0, 90.0, 45.0, 3.14, -0.05, 100, 1.5)
test_intercept(50.0, 50.0, 20.0, 1.0, 0.01, 50, 5.0)
