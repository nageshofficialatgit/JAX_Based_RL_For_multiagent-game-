import math
def simulate_hit(src_x, src_y, src_r, tgt_x, tgt_y, tgt_r, tgt_a, speed, angle):
    # Discrete simulation
    fx = src_x + (src_r + 0.1) * math.cos(angle)
    fy = src_y + (src_r + 0.1) * math.sin(angle)
    dx = speed * math.cos(angle)
    dy = speed * math.sin(angle)
    
    for tick in range(1, 100):
        fx += dx
        fy += dy
        
        # Target moves
        cur_a = tgt_a + 0.02 * tick
        px = 50.0 + math.hypot(tgt_x - 50, tgt_y - 50) * math.cos(cur_a)
        py = 50.0 + math.hypot(tgt_x - 50, tgt_y - 50) * math.sin(cur_a)
        
        dist = math.hypot(fx - px, fy - py)
        if dist <= tgt_r:
            return True, tick
    return False, -1
