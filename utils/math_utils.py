import math
from numba import njit

@njit(cache=True, fastmath=True)
def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

@njit(cache=True, fastmath=True)
def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + 5.0 * (ratio ** 1.5)

@njit(cache=True, fastmath=True)
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    seg_sq = dx * dx + dy * dy
    if seg_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_sq))
    return dist(px, py, x1 + t * dx, y1 + t * dy)

@njit(cache=True, fastmath=True)
def segment_hits_sun(x1, y1, x2, y2):
    return point_to_segment_distance(50.0, 50.0, x1, y1, x2, y2) < 11.5

@njit(cache=True, fastmath=True)
def launch_point(sx, sy, sr, angle):
    c = sr + 0.1
    return sx + math.cos(angle) * c, sy + math.sin(angle) * c
