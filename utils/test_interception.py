import numpy as np

def calculate_interception(source_x, source_y, target_r, target_theta_0, ang_vel, speed):
    low, high = 0.0, 1000.0
    for _ in range(50):
        T = (low + high) / 2
        theta_T = target_theta_0 + ang_vel * T
        tx = 50 + target_r * np.cos(theta_T)
        ty = 50 + target_r * np.sin(theta_T)
        dist = np.sqrt((tx - source_x)**2 + (ty - source_y)**2)
        if dist > speed * T:
            # We haven't reached it yet at time T
            low = T
        else:
            # We can reach it at time T or earlier
            high = T
            
    if high > 990.0:
        return None, None # Cannot reach
        
    T = high
    theta_T = target_theta_0 + ang_vel * T
    tx = 50 + target_r * np.cos(theta_T)
    ty = 50 + target_r * np.sin(theta_T)
    angle = float(np.arctan2(ty - source_y, tx - source_x))
    return angle, T

print(calculate_interception(50, 50, 30, 0, 0.02, 1.0))
print(calculate_interception(20, 20, 45, 0, 0.05, 1.0)) # Impossible to catch if moving away?

