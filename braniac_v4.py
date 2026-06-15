import math
import os
import time
from collections import defaultdict, namedtuple
import numpy as np
from numba import njit
@njit(fastmath=True)
def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

@njit(fastmath=True)
def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + 5.0 * (ratio ** 1.5)

@njit(fastmath=True)
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    seg_sq = dx * dx + dy * dy
    if seg_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_sq))
    return dist(px, py, x1 + t * dx, y1 + t * dy)

@njit(fastmath=True)
def segment_hits_sun(x1, y1, x2, y2):
    return point_to_segment_distance(50.0, 50.0, x1, y1, x2, y2) < 11.5

@njit(fastmath=True)
def launch_point(sx, sy, sr, angle):
    c = sr + 0.1
    return sx + math.cos(angle) * c, sy + math.sin(angle) * c

def _apply_training_overrides(overrides):
    """Called by training/arena.py before each game to set tunable params."""
    global EVAL_WEIGHTS, CANDIDATE_WEIGHTS
    g = globals()
    for key, value in overrides.items():
        if key in EVAL_WEIGHTS:
            EVAL_WEIGHTS[key] = type(EVAL_WEIGHTS[key])(value)
        elif key in CANDIDATE_WEIGHTS:
            CANDIDATE_WEIGHTS[key] = type(CANDIDATE_WEIGHTS[key])(value)
        elif key in g:
            g[key] = type(g[key])(value)
            
    # Fallback for playing against older generations
    if "my_production" in overrides and "my_production_base" not in overrides:
        EVAL_WEIGHTS["my_production_base"] = float(overrides["my_production"])
        EVAL_WEIGHTS["my_production_decay"] = 0.0
        
    if "retreat_threshold" in overrides:
        global HAMMER_ABORT_OVERRUN_RATIO
        HAMMER_ABORT_OVERRUN_RATIO = float(overrides["retreat_threshold"])

EVAL_WEIGHTS = {
    "my_ships": 1.0,
    "my_production": 15.0, # Placeholder, updated dynamically
    "my_production_base": 15.0,
    "my_production_decay": -0.05,
    "my_planets": 5.0,
    "enemy_ships_total": -0.2,
    "enemy_prod_total": -2.0,
    "enemy_leader_ships": -1.0,
    "my_spread_penalty": -2.0,
    "my_spread_exponent": 1.0,
    "enemy_spread_bonus": 1.0,
    "enemy_spread_exponent": 1.0,
    "weight_t30": 0.5,
    "weight_t60": 0.5,
    "weight_t100": 0.5,
    "retreat_threshold": 1.5,
}


F14_4A_2P_FOCUS_ENABLED = True
F14_4A_2P_FOCUS_DIST_BONUS = 18.0   
F14_4A_2P_FOCUS_HAMMER_BONUS = 20.0
F14_4A_2P_FOCUS_MEGA_BONUS = 100

BOARD = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_R = 10.0
SUN_SAFETY = 1.5
ROTATION_LIMIT = 50.0
LAUNCH_CLEARANCE = 0.1
MAX_SPEED = 6.0
TOTAL_STEPS = 500
SIM_HORIZON = 110
FWD_SIM_FILTER_ENABLED = True   
FWD_SIM_HORIZON = 7             
FWD_SIM_DEFENSE_CHECK = True    
FWD_SIM_RANK_BONUS_4P = 0.0     
                                
SEARCH_EXPAND_4P_ENABLED = True 
                                
                                
SEARCH_EXPAND_2P_ENABLED = True 
SEARCH_MAX_PER_SOURCE = 3       
SEARCH_MAX_ACTIONS_TO_PICK = 5    
SEARCH_MAX_ACTIONS_TO_PICK_2P = 7 
SEARCH_DISABLES_CHEAP_PICKUP = True  
HAMMER_MELIS_VERIFY = True      
SEARCH_DEPTH2_ENABLED = False   


NEUTRAL_CAP_USES_EFFECTIVE_GARRISON = True
NEUTRAL_CAP_LOOKAHEAD = 10       

N6_USE_EFFECTIVE_PRE_GARRISON = True

TERMINAL_PHASE_ENABLED = True
TERMINAL_PHASE_TURNS = 30

FLEET_INTENT_ENABLED = True
FLEET_INTENT_MIN_DROP = 8       
FLEET_INTENT_HAMMER_BONUS = 5.0 





F1B_EXPAND_BONUS_ENABLED = True
F1B_EXPAND_BONUS = 3.0   


R1_RECAPTURE_PRIORITY_ENABLED = True
R1_RECAPTURE_HAMMER_BONUS = 8.0

E2_USE_GARRISON_THRESHOLD = True


SO1_STATIC_PREFERENCE_ENABLED = True
SO1_STATIC_BONUS = 2.179862   
SO1_STATIC_BONUS_2P = 2.179862    
SO1_STATIC_BONUS_4P = 2.95474    


SP1_SPEED_AWARE_ENABLED = True
SP1_LONG_DIST_THRESHOLD = 27.637375  
SP1_LONG_DIST_SHIPS = 22         






TI1_TIE_FOR_WIN_ENABLED = True
TI1_HORIZON_TURNS = 25           
TI1_REQUIRED_EXTRA_MARGIN = 5    
TI1_TRAILING_GAP_MIN = 10        





AS1_ANTI_SECOND_ENABLED = True





FAILTOLERANT_ENABLED = True




MELIS_SANITY_ENABLED = True
MELIS_SANITY_THETA = 3.0





F16_DIVERSITY_ENABLED = True
F16_CLOSEST_PICKS = 2   
F16_PROD_PICKS = 1      



FWD_SCORE_AGG_ENABLED = True
FWD_SCORE_AGG_TURNS = (4, 8, 14, 20)





PSM_OPENING_TURN = 14
PSM_OPENING_TURN_2P = 14    
PSM_OPENING_TURN_4P = 10    


ABSORB_MIN_THREAT = 3            
ABSORB_PROJECTION_MARGIN = 0     


DEFENSE_OVERSEND = 1             
DEFENSE_OVERSEND_2P = 1    
DEFENSE_OVERSEND_4P = 0    
DEFENSE_COALITION_MAX = 2        





MIN_DISPATCH_SHIPS = 8           



F3_THREE_BUCKET_ENABLED = True
F3_SAFE_FLOOR = 5
F3_SAFE_DIST = 12.0
F3_HARD_FLOOR = 14
F3_HARD_GARRISON = 14


EXPAND_K_OPENING = 2             
EXPAND_K_MID = 1                 
EXPAND_MAX_TRAVEL_OPENING = 20
EXPAND_MAX_TRAVEL_MID = 14
EXPAND_MIN_MARGIN = 0            
EXPAND_MIN_MARGIN_4P = 3  


X8B_2P_EXTRA = 3
EXPAND_MIN_SHIPS = MIN_DISPATCH_SHIPS


EXPAND_MIN_PROD_2P = 2





TIEBREAK_ENABLED = True
TIEBREAK_EPS_FRAC = 0.005   
TIEBREAK_EPS_MIN = 1.439234      







ROT_AWARE_RANK_ENABLED = os.environ.get("V124_ROT_AWARE", "1") != "0"






VALUE_WEIGHT_2P = 4.86118
VALUE_WEIGHT_4P = float(os.environ.get("V126_VALUE_WEIGHT_4P", "2.0"))







ANTI_SNIPE_ENABLED = os.environ.get("V124_ANTI_SNIPE", "1") != "0"
ANTI_SNIPE_HORIZON = 25          
ANTI_SNIPE_2P_ONLY = False       






REACTIVE_SNIPE_PROJECTION_ENABLED = True
REACTIVE_EMIT_FRAC = 0.49629        
REACTIVE_MIN_ENEMY_SHIPS = 5     
REACTIVE_MIN_PROJECTED = 3       



SUN_SHADOW_REACTIVE_FILTER = True






COUNTER_SNIPE_ENABLED = os.environ.get("V124_COUNTER_SNIPE", "1") != "0"
COUNTER_SNIPE_2P_ONLY = False    
COUNTER_SNIPE_MAX_COST = 30
COUNTER_SNIPE_MIN_DELAY = 1
COUNTER_SNIPE_MAX_DELAY = 12








CHEAP_PICKUP_ENABLED = os.environ.get("V124_CHEAP_PICKUP", "1") != "0"
CHEAP_PICKUP_4P_ONLY = True
CHEAP_PICKUP_MAX_GARRISON = 25

CHEAP_PICKUP_MIN_PROD = int(os.environ.get("F32_CP_MIN_PROD", "2"))









ENDGAME_ROI_ENABLED = os.environ.get("V128_ENDGAME_ROI", "1") != "0"
ENDGAME_ROI_TURNS = 30






NEUTRAL_TEMPO_FILTER_ENABLED = os.environ.get("V128_TEMPO_FILTER", "1") != "0"
NEUTRAL_TEMPO_THRESHOLD = 10     






LAUNCH_BLACKOUT_ENABLED = os.environ.get("V128_LAUNCH_BLACKOUT", "1") != "0"
LAUNCH_BLACKOUT_TURNS = 10







NEUTRAL_HARD_CAP_ENABLED = os.environ.get("V128_NEUTRAL_CAP", "1") != "0"
NEUTRAL_HARD_CAP_4P = 40          
NEUTRAL_HARD_CAP_2P = 61          
NEUTRAL_WATCHLIST_MIN_DROP = 5  





LOW_PROD_NEUTRAL_SKIP_ENABLED = True
LOW_PROD_NEUTRAL_SKIP_PROD = 1       
LOW_PROD_NEUTRAL_SKIP_GARRISON = 14  







WEAKEST_TARGET_ENABLED = os.environ.get("V128_WEAKEST_TARGET", "1") != "0"
WEAKEST_TARGET_BONUS = 2.0      
WEAKEST_TARGET_MIN_STEP = 60    
WEAKEST_DONT_FINISH_SHARE = 0.05
WEAKEST_DONT_FINISH_PENALTY = 12.0  





LEADER_BASH_ENABLED = os.environ.get("V128_LEADER_BASH", "1") != "0"
LEADER_BASH_RATIO = 1.3
LEADER_BASH_BONUS = 4.0
LEADER_BASH_MIN_STEP = 60   





COALITION_ENABLED = True
COALITION_MAX_PARTICIPANTS = 3   
COALITION_NEUTRALS_ONLY = False  
COALITION_MAX_TRAVEL_BONUS = 2   
COALITION_MIN_PER_CONTRIBUTOR = 15   
COALITION_MIN_PER_CONTRIBUTOR_2P = 15    
COALITION_MIN_PER_CONTRIBUTOR_4P = 5    
COALITION_MIN_TARGET_SHIPS = 20      


HAMMER_ENABLED = True
HAMMER_STOCKPILE_MIN = 50
HAMMER_TARGET_PROD_MIN = 2
HAMMER_PROD_SHARE_TRIGGER = 0.40
HAMMER_OVERKILL_RATIO = 1.30
HAMMER_SURROUNDED_PROMOTE_TURNS = 10  
HAMMER_MAX_TRAVEL = 24                
HAMMER_ABORT_OVERRUN_RATIO = 1.329521     
HAMMER_PLAN_REVALIDATE_INTERVAL = 1   
HAMMER_MIN_PER_CONTRIBUTOR = 9        









MEGA_HAMMER_ENABLED = True



MEGA_HAMMER_4P_ONLY = True
MEGA_HAMMER_SHIPS_MIN = 300           
MEGA_HAMMER_TARGET_GARRISON_MAX = 80  
MEGA_HAMMER_MAX_TRAVEL = 40           









PROD_RESERVE_ENABLED = False          


MEGA_HAMMER_THRESHOLD_BY_PROD = {5: 200, 4: 250, 3: 300, 2: 350, 1: 400}







FRESH_CAPTURE_INHERITANCE_ENABLED = True
FRESH_CAPTURE_MAX_AGE = 5                  
MEGA_HAMMER_SHIPS_MIN_FRESH = 200          





MEGA_HAMMER_CONCENTRATE_ENABLED = True
MEGA_HAMMER_MAX_PER_TURN = 1               




MEGA_HAMMER_MELIS_VERIFY = True




MEGA_HAMMER_VERIFY_OPP_EMIT = 0.30









HAMMER_NO_THREAT_OVERSEND_ENABLED = True
HAMMER_NO_THREAT_OVERSEND_2P_ONLY = True


HAMMER_ALWAYS_OVERSEND_2P = False





HAMMER_SAFE_SURPLUS_OVERSEND_ENABLED = True
HAMMER_SAFE_SURPLUS_RATIO = 2.0  
HAMMER_OVERSEND_MAX_THREAT_RATIO = 0.3  






ACCUMULATOR_ENABLED = True
ACCUMULATOR_4P_ONLY = True                  
ACCUMULATOR_TURN_MIN = 15                   
ACCUMULATOR_LEAD_MIN_SHIPS = 100            
ACCUMULATOR_LEAD_THREAT_RATIO = 0.5         
ACCUMULATOR_FEEDER_MIN_SURPLUS = 30         
ACCUMULATOR_FEEDER_KEEP_RESERVE = 30        
ACCUMULATOR_FEEDER_MAX_TRAVEL = 30          
ACCUMULATOR_MAX_FEEDS_PER_TURN = 3          





BRAIN_LEAD_RESERVE_ENABLED = True
BRAIN_LEAD_RESERVE_4P_ONLY = True            






BRAIN_LEAD_RESERVE_MIN_SHIPS = 200




BRAIN_LEAD_RESERVE_REQUIRE_TARGET = False



BRAIN_LEAD_PREFER_FRONTIER = False
BRAIN_LEAD_FRONTIER_WEIGHT = 2.0


MEGA_HAMMER_TARGET_GARRISON_MAX_ITER_H = 100  





MULTIPRONG_ENABLED = False  
MULTIPRONG_2P_ONLY = True


MULTIPRONG_REINFORCER_MIN_RATIO = 1.0


MULTIPRONG_E_OVERKILL = 1.05

MULTIPRONG_CREDIBILITY_FACTOR = 0.6
MULTIPRONG_MAX_TRAVEL = 40       
MULTIPRONG_MIN_PER_CONTRIBUTOR = 8
MULTIPRONG_MAX_PARTICIPANTS = 3


LATE_FLUSH_REMAINING_TURNS = 25  
LATE_FLUSH_OVERKILL_RATIO = 1.05      


SOFT_DEADLINE_FRACTION = 0.82


RACE_ENABLED = True
RACE_HORIZON_TURNS = 18          
RACE_MAX_NEUTRAL_DIST = 20     
RACE_TIE_GOES_TO_LARGER = True   


PERSONALITY_ENABLED = True
PERSONALITY_AGG_HIGH = 0.30      
PERSONALITY_AGG_LOW = 0.10       
PERSONALITY_MIN_SAMPLE = 50      

MODE_PARAMS = {
    "patient": {
        "expand_k_opening": 2,            
        "expand_max_travel_opening": 22,  
        "expand_k_mid": 1,
        "expand_max_travel_mid": 14,
        "hammer_prod_share": 0.2,
        "hammer_overkill": 1.30,
        "hammer_stockpile_min": 50,       
    },
    "opportunistic": {
        "expand_k_opening": 3,            
        "expand_max_travel_opening": 22,  
        "expand_k_mid": 2,                
        "expand_max_travel_mid": 18,      
        "hammer_prod_share": 0.35,        
        "hammer_overkill": 1.30,
        "hammer_stockpile_min": 50,
    },
    "pressure": {
        "expand_k_opening": 3,            
        "expand_max_travel_opening": 22,  
        "expand_k_mid": 0,
        "expand_max_travel_mid": 9,      
        "hammer_prod_share": 0.30,        
        "hammer_overkill": 1.20,          
        "hammer_stockpile_min": 50,
    },
}










MODE_PARAMS_2P = {
    "patient": {
        "expand_k_opening": 5,            
        "expand_max_travel_opening": 35,  
        "expand_k_mid": 4,                
        "expand_max_travel_mid": 28,      
        "hammer_prod_share": 0.30,        
        "hammer_overkill": 1.15,          
        "hammer_stockpile_min": 25,       
    },
    "opportunistic": {
        "expand_k_opening": 5,
        "expand_max_travel_opening": 35,
        "expand_k_mid": 6,
        "expand_max_travel_mid": 30,
        "hammer_prod_share": 0.28,
        "hammer_overkill": 1.15,
        "hammer_stockpile_min": 25,
    },
    "pressure": {
        "expand_k_opening": 5,
        "expand_max_travel_opening": 35,
        "expand_k_mid": 2,
        "expand_max_travel_mid": 52,      
        "hammer_prod_share": 0.25,        
        "hammer_overkill": 1.177645,
        "hammer_stockpile_min": 25,
    },
}




TWO_P_PATIENT_NUDGE_TURNS = 10
TWO_P_PATIENT_ESCALATE_TURNS = 20
TWO_P_PROD_SHARE_HISTORY = 10
TWO_P_PROD_SHARE_PROGRESS_EPS = 0.005   





STOP_EXPAND_2P_ENABLED = True




STOP_EXPAND_PROD_SHARE_2P = 0.65    
STOP_EXPAND_TURN_MIN_2P = 30        






COMBAT_STOP_EXPAND_ENABLED = False      
COMBAT_STOP_EXPAND_4P_ONLY = True
COMBAT_STOP_EXPAND_TURN_MIN = 25
COMBAT_CONTACT_MIN_SHIPS = 15
COMBAT_CHEAP_GARRISON = 10              
COMBAT_CHEAP_DIST = 12.0






PROD_LAG_STOP_EXPAND_ENABLED = True
PROD_LAG_STOP_EXPAND_TURN_MIN = 25
PROD_LAG_STOP_EXPAND_THRESH_2P = 0.40   
PROD_LAG_STOP_EXPAND_THRESH_4P = 0.22   





ENEMY_TEMPO_STOP_EXPAND_ENABLED = True
ENEMY_TEMPO_STOP_EXPAND_TURN_MIN = 20
ENEMY_TEMPO_STOP_EXPAND_MIN_LAUNCHES = 2






EASY_ENEMY_STOP_EXPAND_ENABLED = False
EASY_ENEMY_STOP_EXPAND_TURN_MIN = 15
EASY_ENEMY_MAX_GARRISON = 20
EASY_ENEMY_MAX_DIST = 25.0
EASY_ENEMY_MIN_COUNT = 1





TURN_CUTOFF_STOP_EXPAND_ENABLED = True
TURN_CUTOFF_STOP_EXPAND_TURN = 80   







PROD_LEAD_STOP_EXPAND_4P_ENABLED = True
PROD_LEAD_STOP_EXPAND_4P_TURN_MIN = 25
PROD_LEAD_STOP_EXPAND_4P_THRESH = 0.35   






STOCKPILE_STOP_EXPAND_ENABLED = True
STOCKPILE_STOP_EXPAND_TURN_MIN = 20
STOCKPILE_STOP_EXPAND_MAX_GARRISON = 250  







NEUTRAL_SATURATION_STOP_EXPAND_ENABLED = False  
NEUTRAL_SATURATION_2P_ONLY = True
NEUTRAL_SATURATION_TURN_MIN = 20
NEUTRAL_SATURATION_CHEAP_GARRISON = 10
NEUTRAL_SATURATION_REACH_DIST = 30.0






Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])






def orbital_radius(p):
    return dist(p.x, p.y, CENTER_X, CENTER_Y)


def is_static_planet(p):
    return orbital_radius(p) + p.radius >= ROTATION_LIMIT


def safe_geometry(sx, sy, sr, tx, ty, tr):
    """Direct-line angle + clear travel distance, or None if the path crosses the sun."""
    angle = math.atan2(ty - sy, tx - sx)
    lx, ly = launch_point(sx, sy, sr, angle)
    hit_d = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    ex = lx + math.cos(angle) * hit_d
    ey = ly + math.sin(angle) * hit_d
    if segment_hits_sun(lx, ly, ex, ey):
        return None
    return angle, hit_d


def estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    safe = safe_geometry(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns


def predict_planet_position(planet, initial_by_id, ang_vel, turns):
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    cur = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new = cur + ang_vel * turns
    return CENTER_X + r * math.cos(new), CENTER_Y + r * math.sin(new)






R4_BEHIND_SUN_WAIT_ENABLED = True
R4_FUTURE_HORIZON = 10   







def predict_comet_position(planet_id, comets, turns):
    for group in comets:
        pids = group.get("planet_ids", []) if isinstance(group, dict) else []
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", []) if isinstance(group, dict) else []
        path_index = group.get("path_index", 0) if isinstance(group, dict) else 0
        if idx >= len(paths):
            return None
        path = paths[idx]
        future_idx = int(path_index) + int(turns)
        if 0 <= future_idx < len(path):
            return float(path[future_idx][0]), float(path[future_idx][1])
        return None
    return None


def predict_target_position(target, world, turns):
    """Dispatch: comets use their precomputed path; orbital planets use angular
    extrapolation; static planets stay put. Returns (x, y) or None if a comet
    has expired by `turns`."""
    if target.id in world.comet_ids:
        pos = predict_comet_position(target.id, world.comets, turns)
        if pos is not None:
            return pos
        
    return predict_planet_position(target, world.initial_by_id, world.ang_vel, turns)


AIM_MAX_ITERS = 6          
AIM_CONVERGE_TURNS = 2
AIM_CONVERGE_DIST = 0.6


def aim_at_target(src, target, ships, initial_by_id, ang_vel, world=None):
    """Returns (angle, turns) for sending `ships` from src to hit target.
    Iterates orbital prediction. Returns None if the path is blocked by the
    sun OR if convergence isn't reached — better to skip a target than fire
    a fleet that wanders past it because our aim didn't settle.

    V13.3 Q1: when target is a comet AND world is passed, use comet path for
    future-position; otherwise existing orbital extrapolation.

    V13.3 R4 (behind-sun wait): if the FIRST estimate fails (current path
    blocked by sun), try aiming at projected future positions of the target
    where the orbital motion may have cleared the path. We launch NOW aiming
    at where the target WILL be — fleet flies straight, target swings into
    place. Better than rejecting the shot entirely."""
    est = estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    if est is None and R4_BEHIND_SUN_WAIT_ENABLED and world is not None:
        
        for future_t in range(2, R4_FUTURE_HORIZON, 2):
            if target.id in world.comet_ids:
                pos = predict_comet_position(target.id, world.comets, future_t)
            else:
                init = initial_by_id.get(target.id)
                if init is None:
                    pos = None
                elif dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius >= ROTATION_LIMIT:
                    pos = None  
                else:
                    pos = predict_planet_position(target, initial_by_id, ang_vel, future_t)
            if pos is None:
                continue
            est = estimate_arrival(src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships)
            if est is not None:
                break
    if est is None:
        return None
    
    is_comet = world is not None and target.id in world.comet_ids
    if not is_comet:
        init = initial_by_id.get(target.id)
        if init is None:
            return est
        if dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius >= ROTATION_LIMIT:
            return est

    angle, turns = est
    tx, ty = target.x, target.y
    for _ in range(AIM_MAX_ITERS):
        if is_comet:
            pos = predict_comet_position(target.id, world.comets, turns)
            if pos is None:
                
                return None
            ntx, nty = pos
        else:
            ntx, nty = predict_planet_position(target, initial_by_id, ang_vel, turns)
        nest = estimate_arrival(src.x, src.y, src.radius, ntx, nty, target.radius, ships)
        if nest is None:
            return None
        nangle, nturns = nest
        if (abs(ntx - tx) < AIM_CONVERGE_DIST
                and abs(nty - ty) < AIM_CONVERGE_DIST
                and abs(nturns - turns) <= AIM_CONVERGE_TURNS):
            return nangle, nturns
        angle, turns = nangle, nturns
        tx, ty = ntx, nty
    
    return None


def fleet_target_planet(fleet, planets, initial_by_id=None, ang_vel=0.0):
    """Which planet this in-flight fleet hits, and when (in turns from now).

    Two-pass: static planets via cheap straight-line intersection, orbital
    planets via per-turn forward simulation. The naive straight-line check
    against the planet's CURRENT position misses orbital targets — the
    planet has rotated since the fleet launched, so the ray won't intersect
    its current XY but WILL intersect its future orbital position. Without
    accounting for this, incoming hostile fleets at our orbital planets
    don't show up in arrivals_by_planet, and the reservation walk wrongly
    decides our planet is safe and lets it fire offensively.
    """
    dx_dir = math.cos(fleet.angle)
    dy_dir = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)

    def _is_orbital(p):
        if initial_by_id is None:
            return False
        init = initial_by_id.get(p.id)
        if init is None:
            return False
        return dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius < ROTATION_LIMIT

    static_planets = []
    orbital_planets = []
    for p in planets:
        if _is_orbital(p):
            orbital_planets.append(p)
        else:
            static_planets.append(p)

    best_p, best_t = None, float(SIM_HORIZON) + 1.0

    for p in static_planets:
        dx = p.x - fleet.x
        dy = p.y - fleet.y
        proj = dx * dx_dir + dy * dy_dir
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        rr = p.radius * p.radius
        if perp_sq >= rr:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, rr - perp_sq)))
        t = hit_d / speed
        if t <= SIM_HORIZON and t < best_t:
            best_t, best_p = t, p

    if initial_by_id is not None and orbital_planets:
        best_dsq = None
        max_t = int(math.ceil(min(best_t, float(SIM_HORIZON))))
        for t in range(1, max_t + 1):
            fx = fleet.x + dx_dir * speed * t
            fy = fleet.y + dy_dir * speed * t
            for p in orbital_planets:
                px, py = predict_planet_position(p, initial_by_id, ang_vel, t)
                rr = p.radius * p.radius
                dsq = (fx - px) ** 2 + (fy - py) ** 2
                if dsq < rr:
                    if t < best_t or (t == best_t and (best_dsq is None or dsq < best_dsq)):
                        best_t, best_p, best_dsq = float(t), p, dsq
            if best_p is not None and best_t <= t:
                break

    if best_p is None:
        return None, None
    return best_p, max(1, int(math.ceil(best_t)))






def garrison_at_arrival(target, travel_turns):
    """Defender ship count at the moment our fleet lands."""
    if target.owner == -1:
        return int(target.ships)  
    return int(target.ships) + int(target.production) * int(travel_turns)


def needed_to_capture(target, travel_turns):
    """Ships required at arrival to flip ownership (combat: survivor > garrison)."""
    return garrison_at_arrival(target, travel_turns) + 1









EFFECTIVE_GARRISON_ENABLED = True

def effective_garrison_at_arrival(target, travel_turns, world):
    """Defender count at our arrival, accounting for pre-arrival enemy fleets.
    Returns (projected_owner, projected_ships) at travel_turns."""
    if not EFFECTIVE_GARRISON_ENABLED:
        return target.owner, garrison_at_arrival(target, travel_turns)
    arrivals = world.arrivals_by_planet.get(target.id, [])
    
    
    
    if world.is_2p:
        relevant = sorted(
            ((eta, owner, ships) for eta, owner, ships in arrivals
             if 1 <= eta <= travel_turns and ships > 0 and owner != -1),
            key=lambda x: x[0],
        )
    else:
        relevant = sorted(
            ((eta, owner, ships) for eta, owner, ships in arrivals
             if 1 <= eta <= travel_turns and owner != world.player and ships > 0
             and owner != -1),
            key=lambda x: x[0],
        )
    if not relevant:
        return target.owner, garrison_at_arrival(target, travel_turns)
    owner = int(target.owner)
    ships = int(target.ships)
    prod = max(0, int(target.production))
    last_t = 0
    for eta, fleet_owner, fleet_ships in relevant:
        
        if owner != -1:
            ships += prod * (eta - last_t)
        if fleet_owner == owner:
            ships += fleet_ships  
        else:
            if fleet_ships > ships:
                owner = int(fleet_owner)
                ships = fleet_ships - ships
            elif fleet_ships < ships:
                ships -= fleet_ships
            else:
                ships = 0  
        last_t = eta
    
    if owner != -1:
        ships += prod * (travel_turns - last_t)
    return owner, ships


def effective_needed_to_capture(target, travel_turns, world):
    """needed_to_capture with effective_garrison_at_arrival projection."""
    _, defender_ships = effective_garrison_at_arrival(target, travel_turns, world)
    return defender_ships + 1






def collect_arrivals(planet_id, fleets, planets, initial_by_id=None, ang_vel=0.0):
    """For a given planet, return [(eta, owner, ships)] of all fleets converging on it."""
    out = []
    for f in fleets:
        if int(f.ships) <= 0:
            continue
        target, eta = fleet_target_planet(f, planets, initial_by_id, ang_vel)
        if target is None or target.id != planet_id:
            continue
        out.append((eta, int(f.owner), int(f.ships)))
    return out


def compute_planet_reserve(planet, arrivals, player):
    """The minimum ships we must keep on the surface so the running balance never
    dips below ABSORB_PROJECTION_MARGIN through every incoming fleet's arrival,
    factoring production growth and friendly reinforcements.

    Returns (reserve, holds, deficit, deadline).
        reserve   int, ships that must NOT be sent out this turn.
        holds     True if reserve <= planet.ships (planet survives on its own).
        deficit   ships we still need from outside if !holds (else 0).
        deadline  earliest turn balance dips below margin if !holds (else None).

    V12.3c4 (2.4 redesign): per-fleet ABSORB_MIN_THREAT filter replaced
    with window-aggregated check. Window = garrison/production (the
    planet's natural absorb cycle). If sum(hostile_in_window) < threshold,
    ignore all hostile fleets within the window. Hostile fleets outside
    the window are always counted (they're far out enough that natural
    growth doesn't cover them and they aren't simple noise). Closes the
    Stackelberg-leader exploit (firing many sub-threshold fleets) without
    triggering absorb on transient noise the planet would have absorbed.
    """
    if planet.owner != player:
        return 0, True, 0, None

    prod = max(0, int(planet.production))
    ships_now = max(0, int(planet.ships))
    if prod > 0:
        absorb_window = max(1, ships_now // prod)
    else:
        absorb_window = SIM_HORIZON

    hostile_in_window = 0
    for eta, owner, ships in arrivals:
        if ships <= 0 or owner == player or owner == -1:
            continue
        if int(eta) <= absorb_window:
            hostile_in_window += int(ships)
    
    
    
    
    
    absorb_min_threat = max(1, min(ABSORB_MIN_THREAT, ships_now // 3))
    skip_in_window_hostiles = hostile_in_window < absorb_min_threat

    
    
    
    
    
    friendly_events = defaultdict(int)
    hostile_by_owner = defaultdict(lambda: defaultdict(int))
    for eta, owner, ships in arrivals:
        if ships <= 0:
            continue
        if owner == player:
            friendly_events[eta] += ships
        elif owner == -1:
            continue
        else:
            if skip_in_window_hostiles and int(eta) <= absorb_window:
                continue
            hostile_by_owner[eta][owner] += int(ships)

    events = defaultdict(int)
    for eta, ships in friendly_events.items():
        events[eta] += ships
    for eta, owner_totals in hostile_by_owner.items():
        
        
        sorted_h = sorted(owner_totals.values(), reverse=True)
        if len(sorted_h) == 1:
            survivor = sorted_h[0]
        elif sorted_h[0] == sorted_h[1]:
            survivor = 0
        else:
            survivor = sorted_h[0] - sorted_h[1]
        events[eta] -= survivor

    if not events:
        return 0, True, 0, None

    growth = int(planet.production)
    bal = int(planet.ships)
    last_t = 0
    min_bal = bal
    deadline = None

    for turn in sorted(events):
        bal += growth * (turn - last_t)
        bal += events[turn]
        if bal < min_bal:
            min_bal = bal
        if bal < ABSORB_PROJECTION_MARGIN and deadline is None:
            deadline = turn
        last_t = turn

    if min_bal >= ABSORB_PROJECTION_MARGIN:
        excess = min_bal - ABSORB_PROJECTION_MARGIN
        reserve = max(0, int(planet.ships) - excess)
        return reserve, True, 0, None

    deficit = ABSORB_PROJECTION_MARGIN - min_bal
    return int(planet.ships), False, int(deficit), deadline






def forward_project(world, our_capture_target=None, our_capture_turn=None,
                    our_capture_ships=None, horizon=20,
                    project_opponent_moves=False,
                    opponent_emit_fraction=0.4,
                    snapshot_turns=None):
    """Project every planet's owner+ship count forward `horizon` turns.

    Inputs:
      world — current World snapshot.
      our_capture_target/turn/ships — optional our planned capture (treated
        as a hypothetical friendly fleet arrival).
      horizon — how many turns to project.
      project_opponent_moves — if True, each enemy planet launches a fraction
        of its CURRENT surplus toward its closest non-friendly target every
        few turns. Increases accuracy at cost of pessimism for our holdings.
      opponent_emit_fraction — fraction of surplus the projected launch sends.
    Returns:
      dict planet_id -> (owner_at_H, ships_at_H).

    Model:
      - Existing in-flight fleets arrive at their projected ETA (engine
        combat math: attackers fight each other top-minus-second, then
        survivor reinforces or attacks defender garrison).
      - Production accumulates each turn for owned planets.
      - Phantom launches: each enemy planet within max-speed reach of
        our_capture_target projects a fleet of size phantom_factor*ships
        with optimistic ETA. This catches the dominant snipe risk that
        existing arrivals_by_planet misses (the enemy hasn't launched yet
        but COULD before our planet stabilises).
    """
    
    by_pid = defaultdict(list)
    for pid, arrs in world.arrivals_by_planet.items():
        for eta, owner, ships in arrs:
            if 0 < eta <= horizon:
                by_pid[pid].append((int(eta), int(owner), int(ships)))

    

    
    if our_capture_target is not None and our_capture_turn is not None:
        by_pid[our_capture_target].append(
            (int(our_capture_turn), int(world.player), int(our_capture_ships))
        )

    
    state = {}
    for p in world.planets:
        state[p.id] = [int(p.owner), int(p.ships), int(p.production)]

    
    
    
    planet_pos_map = {p.id: (float(p.x), float(p.y)) for p in world.planets}
    pid_list = list(state.keys())

    
    prod_by_pid = {p.id: max(0, int(p.production)) for p in world.planets}

    snapshots = {} if snapshot_turns else None
    snapshot_set = set(snapshot_turns) if snapshot_turns else None
    for t in range(1, horizon + 1):
        
        for pid, st in state.items():
            if st[0] != -1:
                st[1] += st[2]
        
        
        
        if project_opponent_moves and t % 4 == 0:
            for pid, st in state.items():
                if st[0] == -1 or st[1] < 10:
                    continue
                src_x, src_y = planet_pos_map[pid]
                src_owner = st[0]
                best_d = float("inf")
                best_op = None
                for opid, ost in state.items():
                    if opid == pid or ost[0] == src_owner:
                        continue
                    ox, oy = planet_pos_map[opid]
                    d = ((src_x - ox) ** 2 + (src_y - oy) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best_op = d, opid
                if best_op is None:
                    continue
                
                if src_owner == world.player:
                    frac = opponent_emit_fraction * 0.5
                else:
                    frac = opponent_emit_fraction
                emit = int(st[1] * frac)
                if emit < 5:
                    continue
                ratio = math.log(max(2, emit)) / math.log(1000.0)
                speed = 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)
                eta_arrive = max(1, int(math.ceil(best_d / speed)))
                arrival_t = t + eta_arrive
                if arrival_t > horizon:
                    continue
                by_pid[best_op].append((arrival_t, src_owner, emit))
                st[1] -= emit
        
        
        
        
        
        for pid, arrs in by_pid.items():
            this_turn = [(o, s) for et, o, s in arrs if et == t]
            if not this_turn:
                continue
            st = state[pid]
            defender_owner, garrison = st[0], st[1]
            from_owner = defaultdict(int)
            for o, s in this_turn:
                from_owner[o] += s
            sorted_owners = sorted(from_owner.items(), key=lambda x: -x[1])
            top_owner, top_ships = sorted_owners[0]
            if len(sorted_owners) >= 2:
                second_ships = sorted_owners[1][1]
                if top_ships == second_ships:
                    survivor_ships = 0
                    survivor_owner = -1
                else:
                    survivor_ships = top_ships - second_ships
                    survivor_owner = top_owner
            else:
                survivor_ships = top_ships
                survivor_owner = top_owner
            if survivor_ships > 0:
                if defender_owner == survivor_owner:
                    st[1] = garrison + survivor_ships
                else:
                    new_garrison = garrison - survivor_ships
                    if new_garrison < 0:
                        st[0] = survivor_owner
                        st[1] = -new_garrison
                    else:
                        st[1] = new_garrison
        if snapshot_set is not None and t in snapshot_set:
            snapshots[t] = {pid: (st[0], st[1]) for pid, st in state.items()}

    final = {pid: (st[0], st[1]) for pid, st in state.items()}
    if snapshot_turns is not None:
        return final, snapshots
    return final


def _depth2_penalty(world, our_action, top_opp_actions=2):
    """For our action, project worst-case opponent reply.
    Each enemy planet within reach of our_action's target tries to launch a
    counter-snipe. Returns the WORST (lowest from our POV) Melis score among
    those counter-snipe scenarios.

    Used to penalize our actions that invite easy counter-snipes.
    """
    target_id = our_action["target_id"]
    tgt = world.planet_by_id.get(target_id)
    if tgt is None:
        return 0.0
    worst_delta = 0.0
    candidates_evaluated = 0
    for ep in world.planets:
        if ep.owner == world.player or ep.owner == -1:
            continue
        if int(ep.ships) < 9:
            continue
        d = ((tgt.x - ep.x) ** 2 + (tgt.y - ep.y) ** 2) ** 0.5
        if d > 30.0:
            continue
        
        opp_ships = max(8, int(ep.ships) - 5)
        ratio = math.log(max(2, opp_ships)) / math.log(1000.0)
        speed = 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)
        opp_eta = max(1, int(math.ceil(d / speed)))
        if opp_eta > FWD_SIM_HORIZON + 4:
            continue
        
        proj = forward_project(
            world,
            our_capture_target=our_action["target_id"],
            our_capture_turn=our_action["arrival_turn"],
            our_capture_ships=our_action["ships"],
            horizon=FWD_SIM_HORIZON + 6,
            project_opponent_moves=True,
            opponent_emit_fraction=0.30,
        )
        
        
        
        end_owner, end_ships = proj.get(target_id, (-1, 0))
        
        if end_owner != world.player and opp_ships > end_ships:
            worst_delta = min(worst_delta, -opp_ships)
        candidates_evaluated += 1
        if candidates_evaluated >= top_opp_actions:
            break
    return worst_delta


def search_step_action(world, max_per_source=3, max_actions_to_eval=10,
                       use_depth2=False):
    """Depth-1 alpha-beta over step actions.

    1. Generate candidate step actions via generate_step_actions.
    2. Evaluate each via melis_evaluate (sim+score).
    3. Return list sorted by score (highest first), up to `max_actions_to_eval`.

    Each action has additional key "score". Caller picks top action(s) and
    commits via _commit_fleet.
    """
    actions = generate_step_actions(world, max_per_source=max_per_source)
    if not actions:
        return []
    baseline_score = melis_evaluate(world, our_step_action=None)
    
    
    
    apply_decay = world.is_2p
    scored = []
    for act in actions[:max_actions_to_eval]:
        act_score = melis_evaluate(world, our_step_action=act)
        gain = act_score - baseline_score
        if apply_decay and gain > 0:
            gain *= 0.97 ** int(act["arrival_turn"])
        act["score"] = gain
        scored.append(act)
    scored.sort(key=lambda a: (-a["score"], a.get("raw_dist", 0.0)))
    if use_depth2:
        
        for act in scored[:3]:
            act["score"] += _depth2_penalty(world, act)
        scored.sort(key=lambda a: (-a["score"], a.get("raw_dist", 0.0)))
    
    
    if MELIS_SANITY_ENABLED and world.is_2p and scored and scored[0]["score"] < MELIS_SANITY_THETA:
        return []
    return scored


def generate_step_actions(world, max_per_source=3):
    """Generate candidate "step actions" — Melis style. Each step action is
    a single capture targeting one planet, sourced from one of our planets.

    Returns list of dicts: {"target_id", "source_id", "angle", "arrival_turn",
                            "ships", "raw_dist"}.

    Pruning:
      - Skip targets that aren't reachable within max_travel + 4
      - Skip neutral targets blocked by NEUTRAL_HARD_CAP
      - Take top `max_per_source` per source (closest by raw distance)
    """
    actions = []
    if not world.my_planets:
        return actions
    
    is_opening = world.is_opening
    if is_opening:
        max_travel = world.mode_params.get(
            "expand_max_travel_opening", EXPAND_MAX_TRAVEL_OPENING)
    else:
        max_travel = world.mode_params["expand_max_travel_mid"]

    for src in world.my_planets:
        avail = max(0, int(src.ships))
        if avail < MIN_DISPATCH_SHIPS:
            continue
        targets = []
        for t in world.planets:
            if t.owner == world.player:
                continue
            if not is_targetable(world, t):
                continue
            if _neutral_blocked_by_cap(world, t):
                continue
            raw = dist(src.x, src.y, t.x, t.y)
            tt = raw / MAX_SPEED
            if tt > max_travel + 4:
                continue
                
            # --- PROJECT BRANIAC: META-HEURISTIC DYNAMIC WEIGHTS ---
            opp_agg = world.opponent_model.aggression_index[t.owner] if t.owner != -1 else 0.5
            opp_tempo = world.opponent_model.tempo[t.owner] if t.owner != -1 else 0.0
            my_econ = world.opponent_model.economy[world.player]
            their_econ = world.opponent_model.economy[t.owner] if t.owner != -1 else 0.0

            alpha, beta, gamma = get_meta_weights(my_econ, their_econ, opp_agg, opp_tempo)

            # --- PROJECT BRANIAC: EFSC MACRO-DIRECTOR INJECTION ---
            R = max(1, 500 - (world.step + tt))
            P = t.production * gamma
            # Approximate cost including defender premium
            C = t.ships + (P * tt if t.owner != -1 else 0) + 1 
            
            if t.owner == -1:
                swing = (R * P) - C
            else:
                swing = alpha * (R * P) - C
                
            # --- PROJECT BRANIAC: DEATH TRAP AVOIDANCE INJECTION ---
            enemy_threat = 0.0
            for e in world.planets:
                if e.owner not in (-1, world.player):
                    ed = dist(t.x, t.y, e.x, e.y)
                    if ed < 30.0:
                        enemy_threat += (e.ships / max(1.0, ed)) * 2.0 * beta
            swing -= enemy_threat
            
            roi = swing / max(1.0, tt)
            targets.append((roi, raw, t))
            
        # Sort by highest EFSC ROI
        targets.sort(key=lambda x: x[0], reverse=True)
        
        # Take the absolute best EFSC targets
        picks = [(x[1], x[2]) for x in targets[:max_per_source]]
        
        for raw, t in picks:
            plan = plan_solo_capture(world, src, t, avail, max_travel)
            if plan is None:
                continue
            angle, turns, ships = plan
            
            # --- PROJECT BRANIAC: LOCOMOTION (CPA HOLD) INJECTION ---
            if not is_static_planet(src) and is_static_planet(t):
                # src is orbital, target is static
                next_src_x, next_src_y = predict_planet_position(src, world.initial_by_id, world.ang_vel, 1)
                d_now = dist(src.x, src.y, t.x, t.y)
                d_next = dist(next_src_x, next_src_y, t.x, t.y)
                if d_next < d_now - 0.2 and turns > 15:
                    continue # HOLD! Taxi is approaching target.
                    
            actions.append({
                "target_id": int(t.id),
                "source_id": int(src.id),
                "angle": float(angle),
                "arrival_turn": int(turns),
                "ships": int(ships),
                "raw_dist": float(raw),
            })
    return actions


def melis_evaluate(world, our_step_action=None, horizon=12, future_horizon=8,
                   opp_emit=0.20):
    """Melis full-attack-future evaluator.

    Inputs:
      world — current World snapshot.
      our_step_action — optional dict {"target_id", "arrival_turn", "ships"}.
        If provided, simulates our planned capture as part of the projection.
      horizon — short-term sim horizon for our action's effect.
      future_horizon — additional "all-attack-future" projection turns where
        every planet (us + opponents) keeps emitting surplus toward closest
        non-friendly. Captures position quality beyond the immediate move.
      opp_emit — fraction of surplus opponents launch in projection. 0.30
        is the calibrated default; lower = more capture-friendly.

    Returns: scalar score from our player's POV (higher = better).
    """
    target = arrival = ships = None
    if our_step_action is not None:
        target = our_step_action.get("target_id")
        arrival = our_step_action.get("arrival_turn")
        ships = our_step_action.get("ships")
    H = horizon + future_horizon
    n = 2 if world.is_2p else 4
    if FWD_SCORE_AGG_ENABLED:
        snap_turns = tuple(t for t in FWD_SCORE_AGG_TURNS if t <= H)
        if not snap_turns:
            snap_turns = (H,)
        final, snaps = forward_project(
            world,
            our_capture_target=target,
            our_capture_turn=arrival,
            our_capture_ships=ships,
            horizon=H,
            project_opponent_moves=True,
            opponent_emit_fraction=opp_emit,
            snapshot_turns=snap_turns,
        )
        total = 0.0
        count = 0
        for t in snap_turns:
            snap = snaps.get(t)
            if snap is None:
                continue
            total += forward_score(snap, world.player, n, world)
            count += 1
        if H not in snap_turns:
            total += forward_score(final, world.player, n, world)
            count += 1
        return total / max(1, count)
    state = forward_project(
        world,
        our_capture_target=target,
        our_capture_turn=arrival,
        our_capture_ships=ships,
        horizon=H,
        project_opponent_moves=True,
        opponent_emit_fraction=opp_emit,
    )
    return forward_score(state, world.player, n, world)


def forward_score(state, player, n_seats, world=None):
    """Score a forward-projected state from `player`'s POV.

    Combines: ship advantage + 5×planet-count advantage + 8×production advantage.
    Weights chosen so an extra owned planet is worth ~5 ships (a typical garrison)
    and an extra production unit is worth ~8 ships (≈2 turns of growth)."""
    n_planets = [0] * n_seats
    n_prod = [0] * n_seats
    n_ships = [0] * n_seats
    for pid, (o, s) in state.items():
        if 0 <= o < n_seats:
            n_ships[o] += s
            n_planets[o] += 1
            if world is not None:
                p = world.planet_by_id.get(pid)
                if p is not None:
                    n_prod[o] += int(p.production)
    if n_seats <= 1:
        return n_ships[player]
    others = [i for i in range(n_seats) if i != player]
    leader_ships = max(n_ships[i] for i in others)
    leader_planets = max(n_planets[i] for i in others)
    leader_prod = max(n_prod[i] for i in others)
    return ((n_ships[player] - leader_ships)
            + 5 * (n_planets[player] - leader_planets)
            + 8 * (n_prod[player] - leader_prod))






# Distilled weights from braniac_meta_weights.pth
W0 = [[-0.0721510648727417, -0.46015065908432007, 0.22417038679122925, -0.0029929280281066895], [0.30876004695892334, 0.3069007992744446, 0.20922881364822388, -0.37780988216400146], [-0.3610629439353943, 0.10812503844499588, 0.19018077850341797, 0.11765532940626144], [0.14291304349899292, -0.20402973890304565, -0.24913311004638672, -0.1371721625328064], [0.3303300738334656, -0.49334973096847534, -0.41011011600494385, -0.3552170395851135], [-0.32615765929222107, 0.33027103543281555, -0.0050007374957203865, 0.20454855263233185], [0.4379097819328308, -0.11733770370483398, 0.3420795798301697, -0.2550702691078186], [-0.4722561240196228, 0.33654314279556274, -0.21710360050201416, -0.127252459526062], [-0.2494123876094818, 0.3382261395454407, -0.08571568131446838, 0.036351628601551056], [-0.009871060959994793, 0.34935715794563293, 0.0797690823674202, 0.4235169291496277], [-0.3370939791202545, -0.1423516571521759, 0.28576552867889404, 0.383149653673172], [0.3373764157295227, 0.32955777645111084, -0.12814655900001526, 0.007838874123990536], [-0.17407387495040894, 0.4347558617591858, -0.4562719166278839, 0.30396488308906555], [0.1661437749862671, 0.06459343433380127, 0.31822216510772705, -0.4137645363807678], [0.1867067515850067, -0.19730278849601746, 0.3336526155471802, 0.09000878036022186], [-0.1466686874628067, 0.45817065238952637, 0.2505894601345062, 0.2125098556280136], [0.2554819583892822, -0.1895713210105896, -0.40459388494491577, -0.27455562353134155], [-0.3855161964893341, 0.35946252942085266, 0.4699520170688629, 0.373046875], [0.2820066809654236, 0.46810340881347656, -0.00955701619386673, -0.2607405185699463], [0.1679542064666748, 0.2841372787952423, 0.08554815500974655, -0.11402081698179245], [0.04538949579000473, -0.24977363646030426, 0.07604195922613144, 0.4383382201194763], [0.15255245566368103, -0.024845950305461884, 0.37859538197517395, -0.038908157497644424], [-0.4517601728439331, 0.12900060415267944, -0.12645357847213745, -0.14140117168426514], [-0.040090978145599365, -0.420135498046875, -0.16079258918762207, -0.2560268044471741], [0.10824467986822128, 0.0704825147986412, 0.1474820375442505, -0.0047714589163661], [0.2786419987678528, -0.4232577681541443, -0.28516507148742676, 0.21994270384311676], [-0.24187791347503662, -0.2945214509963989, -0.19373303651809692, 0.0526922345161438], [-0.15410465002059937, -0.48305654525756836, 0.13486123085021973, -0.416253924369812], [-0.0856664776802063, 0.46032220125198364, 0.46090972423553467, -0.45136916637420654], [-0.17942607402801514, 0.2607951760292053, -0.43680858612060547, -0.49993884563446045], [-0.02823108434677124, -0.08560115098953247, 0.33362656831741333, -0.2267540693283081], [0.30491551756858826, 0.3588602840900421, 0.10907187312841415, 0.07239554077386856], [-0.233742356300354, -0.26801520586013794, -0.21588033437728882, -0.16100692749023438], [-0.15916089713573456, -0.32915371656417847, 0.2783679962158203, 0.43776169419288635], [0.30418235063552856, -0.381952702999115, 0.2998979687690735, -0.3288038969039917], [-0.20505350828170776, -0.42957448959350586, 0.3181421160697937, -0.09029805660247803], [-0.04044795781373978, -0.5047661662101746, -0.38551726937294006, 0.09898801147937775], [-0.08745720982551575, -0.09296521544456482, -0.22868862748146057, 0.22441436350345612], [-0.06307609379291534, 0.02618955448269844, -0.24099606275558472, 0.2670927941799164], [-0.35118207335472107, 0.01956729218363762, -0.45304402709007263, 0.4542540907859802], [-0.41991305351257324, -0.17791098356246948, -0.26729995012283325, -0.45510441064834595], [0.3468277156352997, 0.02376958541572094, -0.4811966121196747, 0.41144946217536926], [0.1781454086303711, -0.10964131355285645, 0.11957180500030518, -0.29802650213241577], [0.4973897635936737, 0.11474566161632538, -0.02432159148156643, -0.2955278158187866], [-0.046838343143463135, -0.327994167804718, 0.16899770498275757, -0.15019261837005615], [-0.13350719213485718, 0.4671579599380493, 0.3963382840156555, -0.4813874363899231], [0.060914456844329834, -0.024345099925994873, -0.2838249206542969, -0.03664290904998779], [0.08165206760168076, -0.43640491366386414, -0.331677109003067, 0.29215946793556213], [0.4676274061203003, -0.21034152805805206, 0.02161978930234909, 0.2689489722251892], [0.2808328866958618, -0.016071515157818794, -0.1883973628282547, -0.09803236275911331], [-0.24503108859062195, -0.37901991605758667, 0.25450435280799866, 0.32315874099731445], [0.23658418655395508, -0.0417141355574131, 0.09120333939790726, 0.4307798743247986], [0.15245990455150604, -0.18353886902332306, -0.4118525981903076, 0.4919555187225342], [-0.4117491841316223, 0.08964085578918457, -0.2294864058494568, -0.32129132747650146], [0.25887033343315125, 0.3311873972415924, -0.47350573539733887, -0.24908696115016937], [0.4354543387889862, 0.2465215027332306, -0.19835637509822845, 0.468307763338089], [-0.06642460823059082, -0.43427735567092896, -0.29970988631248474, 0.32126230001449585], [0.2567046582698822, -0.07378105819225311, 0.3572936952114105, 0.13605687022209167], [0.011222709901630878, -0.23337030410766602, -0.4500943422317505, 0.1612471044063568], [0.21444205939769745, 0.056550223380327225, 0.3056902289390564, 0.21835200488567352], [-0.14785772562026978, 0.09095625579357147, 0.4887697696685791, 0.1095564067363739], [0.06730800867080688, 0.2952638268470764, 0.026876211166381836, -0.352527379989624], [0.18443739414215088, 0.12903541326522827, 0.10888016223907471, -0.22550368309020996], [0.4428365230560303, -0.42224055528640747, 0.1752091348171234, 0.27116629481315613]]
B0 = [0.3226701617240906, 0.26424866914749146, 0.228600412607193, 0.009352266788482666, 0.03374391794204712, -0.3999057710170746, -0.3769133687019348, 0.07799762487411499, -0.4042845666408539, 0.3419051170349121, 0.04367706552147865, 0.3778257966041565, -0.281479150056839, 0.18277281522750854, -0.3831523656845093, 0.27174219489097595, -0.22495180368423462, 0.19086647033691406, -0.1114431843161583, 0.35591164231300354, 0.05685563012957573, 0.47794368863105774, -0.26978373527526855, 0.44360148906707764, 0.025993412360548973, -0.3568671941757202, -0.29636895656585693, 0.35494083166122437, -0.11327922344207764, -0.15031278133392334, 0.30301475524902344, -0.16966810822486877, 0.11606526374816895, 0.17107154428958893, -0.3906296491622925, 0.42774999141693115, 0.11988810449838638, 0.07564422488212585, 0.1324225664138794, -0.05991928279399872, 0.38323312997817993, -0.3407285809516907, -0.21432864665985107, 0.015116319060325623, -0.24499660730361938, 0.10785883665084839, -0.4927676320075989, 0.11028013378381729, -0.40439295768737793, 0.467523992061615, 0.4260919988155365, 0.381174772977829, 0.449929416179657, -0.42710256576538086, -0.05573372170329094, -0.25151970982551575, -0.1186545118689537, 0.32457825541496277, 0.2674205005168915, -0.27744054794311523, -0.31286680698394775, 0.09840410947799683, 0.49617695808410645, 0.44003692269325256]
W2 = [[0.07288473844528198, 0.06575627624988556, 0.008139527402818203, 0.03802558779716492, -0.11499835550785065, 0.05248045176267624, 0.05026380717754364, -0.04571056365966797, 0.1107817217707634, 0.08954107761383057, -0.0826612040400505, -0.0616324320435524, -0.04837839677929878, -0.12270279228687286, -0.02497086301445961, -0.07933539152145386, -0.026994869112968445, 0.024637684226036072, -0.11139906197786331, 0.09580247104167938, -0.061602093279361725, -0.053333356976509094, -0.02203182876110077, -0.01006530225276947, -0.13766241073608398, 0.06652887165546417, 0.06473572552204132, 0.028256744146347046, 0.05425146222114563, 0.06671927869319916, -0.11744709312915802, -0.04161335900425911, 0.020664885640144348, 0.09194330126047134, 0.11694081127643585, -0.09574630856513977, -0.06149483472108841, 0.10457256436347961, 0.09483926743268967, 0.0748993381857872, 0.07118077576160431, -0.10807263106107712, 0.10358062386512756, -0.13035409152507782, -0.0026109665632247925, -0.0708581954240799, 0.08287246525287628, -0.05140981823205948, -0.05052989721298218, 0.1026773750782013, -0.11362796276807785, 0.09457022696733475, -0.11101428419351578, 0.051670148968696594, -0.003575884969905019, -0.05116681009531021, 0.07640494406223297, 0.011547812260687351, 0.0847349688410759, 0.03713490813970566, -0.09154768288135529, 0.04012884199619293, -0.018907785415649414, -0.09326542168855667], [-0.08205951750278473, 0.0472618043422699, 0.15752850472927094, -0.09935776889324188, 0.04902578890323639, -0.08317705988883972, 0.06211747229099274, -0.07023490965366364, 0.037858348339796066, 0.08915100246667862, -0.10620676726102829, 0.02285103313624859, 0.006122390739619732, -0.11554461717605591, 0.09154549241065979, -0.033032383769750595, 0.05446043610572815, 0.009185773320496082, -0.002178948139771819, 0.010387513786554337, 0.08793435245752335, -0.12817704677581787, -0.016205474734306335, -0.06604206562042236, 0.07607412338256836, 0.04320260137319565, 0.11097736656665802, 0.052146852016448975, 0.017518892884254456, -0.12482577562332153, -0.1241602748632431, -0.08431379497051239, 0.0672307163476944, 0.06196023151278496, 0.07688647508621216, 0.017804041504859924, -0.1077997088432312, -0.02255357801914215, 0.05107470601797104, -0.125998392701149, 0.12492464482784271, -0.09969669580459595, 0.022067099809646606, 0.12164264172315598, 0.04498985409736633, 0.07203000783920288, -0.05008387565612793, 0.044696491211652756, 0.007607961073517799, 0.023298032581806183, 0.0353727862238884, -0.00036616012221202254, 0.01707700453698635, -0.021366193890571594, -0.13406381011009216, 0.08052173256874084, -0.11559031158685684, 0.04590369388461113, -0.012103630229830742, 0.017465487122535706, 0.05440682917833328, -0.08432838320732117, -0.11723785102367401, 0.0072218007408082485], [0.05134454369544983, -0.031994596123695374, -0.014648040756583214, 0.11959496140480042, 0.09429323673248291, -0.017094986513257027, 0.04698123037815094, 0.002056136727333069, -0.09910948574542999, -0.051786888390779495, -0.12450984120368958, 0.06609366834163666, 0.04435493052005768, 0.1121794730424881, -0.10068544000387192, 0.06385403126478195, 0.04024915397167206, 0.08534210175275803, -0.014062034897506237, -0.07204686850309372, 0.054679352790117264, 0.0318135991692543, 0.09277699887752533, -0.07503852248191833, 0.07896039634943008, -0.03282255306839943, 0.057749807834625244, 0.11281189322471619, -0.02472253143787384, -0.05728556215763092, -0.10587595403194427, 0.11563549935817719, -0.03215062618255615, 0.05306524783372879, 0.04785121977329254, -0.0027093887329101562, -0.12990057468414307, -0.08047040551900864, 0.010444275103509426, -0.013398763723671436, 0.12073394656181335, -0.0077766780741512775, -0.010846629738807678, -0.10948613286018372, 0.021724238991737366, -0.023288920521736145, 0.08528071641921997, -0.09804308414459229, 0.0037693381309509277, -0.07787153124809265, -0.0019191476749256253, 0.06493710726499557, -0.044548507779836655, 0.0029104799032211304, 0.014074045233428478, 0.008943852037191391, 0.049251675605773926, -0.04037150740623474, 0.09756267815828323, -0.048760756850242615, -0.07604613155126572, -0.03220604360103607, -0.05468203127384186, -0.03272852674126625], [0.0332566499710083, 0.0955287367105484, -0.0499308779835701, 0.09980954229831696, -0.03116627037525177, 0.08158379793167114, -0.07233193516731262, -0.057739630341529846, -0.008746901527047157, -0.02765020728111267, -0.06741798669099808, 0.09338536113500595, 0.04287315905094147, -0.11852887272834778, -0.00967301893979311, -0.11392858624458313, -0.045700669288635254, -0.027625838294625282, 0.0934760570526123, -0.014685877598822117, 0.05882873386144638, 0.09257932007312775, -0.04606299102306366, 0.07801862061023712, 0.1137688159942627, 0.07057145982980728, 0.08551637828350067, -0.04429878294467926, 0.11064563691616058, -0.11378642916679382, -0.0941622406244278, 0.08768141269683838, 0.02335970103740692, 0.12141865491867065, 0.0894245058298111, 0.04608488082885742, -0.07676612585783005, -0.018552163615822792, 0.09176834672689438, -0.003403238020837307, -0.059792518615722656, -0.0845554769039154, -0.06094534695148468, -0.07869239896535873, 0.09942439198493958, 0.08937887847423553, 0.03770892322063446, -0.09740006178617477, 0.05741136148571968, 0.03837496414780617, -0.13737338781356812, -0.06292583048343658, 0.06469576805830002, 0.08147422969341278, 0.03805842995643616, 0.018165089190006256, 0.10160180926322937, -0.06293295323848724, 0.11189713329076767, -0.011487334966659546, -0.02276850864291191, -0.031955599784851074, 0.10451455414295197, -0.04983844980597496], [-0.034228548407554626, -0.013876691460609436, -0.014754544012248516, -0.00915217399597168, -0.07495386898517609, -0.022874919697642326, 0.024932950735092163, 0.012610509991645813, 0.1008298471570015, 0.09822025150060654, 0.011420024558901787, 0.004869557451456785, -0.09924861043691635, -0.010906606912612915, 0.008952214382588863, -0.07143884152173996, -0.041169315576553345, -0.08475162833929062, -0.10113697499036789, -0.06415250152349472, -0.023071279749274254, 0.05955111235380173, -0.042665690183639526, -0.05369500815868378, -0.0220203697681427, 0.11279671639204025, 0.09502235054969788, -0.10484287142753601, -0.1116136908531189, 0.028022825717926025, -0.0635746419429779, -0.058349426835775375, -0.1174589991569519, -0.0902290940284729, 0.11511936783790588, -0.042334288358688354, 0.08549127727746964, -0.05560462176799774, 0.11755581200122833, 0.08782976865768433, 0.09529155492782593, 0.1285136193037033, 0.07090483605861664, -0.07655878365039825, -0.06418806314468384, 0.10528802871704102, -0.030601397156715393, -0.11046004295349121, -0.017120197415351868, -0.09657920151948929, -0.07052230834960938, 0.07281383126974106, 0.02172360010445118, 0.06704951822757721, 0.0159621462225914, 0.003184662200510502, 0.03156363219022751, 0.1265513002872467, -0.11448164284229279, 0.09565626084804535, 0.0023295111022889614, -0.04321424663066864, 0.09766589105129242, -0.062227219343185425], [-0.1015084832906723, -0.03794541954994202, 0.06457096338272095, 0.07430724799633026, -0.08891309797763824, 0.09475283324718475, 0.11848391592502594, 0.026500195264816284, 0.08770781755447388, -0.0863436758518219, 0.015499085187911987, 0.07715795934200287, -0.02996933087706566, -0.07784521579742432, 0.08271358907222748, 0.10533349215984344, 0.10920828580856323, 0.05464643985033035, -0.11479631066322327, 0.007873967289924622, -0.038319144397974014, 0.03928113728761673, 0.08806027472019196, -0.018090322613716125, -0.004658008459955454, -0.05176704376935959, -0.06018424034118652, -0.050365954637527466, -0.04227516055107117, 0.03458692133426666, -0.018753930926322937, -0.09320570528507233, -0.047708913683891296, -0.07733452320098877, 0.09944210946559906, -0.041870638728141785, -0.05455206334590912, -0.00313611445017159, 0.027460576966404915, 0.0646573007106781, -0.031255483627319336, -0.12395824491977692, -0.07207709550857544, -0.0013974905014038086, -0.007413506507873535, 0.003922000527381897, -0.033744558691978455, 0.07279212772846222, -0.13030683994293213, -0.051875412464141846, 0.05570428818464279, 0.03980504721403122, 0.11119776964187622, -0.03249466419219971, 0.09520155191421509, -0.12407033145427704, -0.0284498929977417, -0.09278695285320282, 0.059066008776426315, -0.05115813761949539, 0.07393313944339752, -0.018063515424728394, 0.08428902924060822, 0.0771990567445755], [0.06757161021232605, -0.07925814390182495, 0.04333744943141937, -0.09348942339420319, -0.041911110281944275, 0.014438226819038391, -0.06088702380657196, -0.1129978746175766, -0.11672204732894897, -0.022370576858520508, 0.09846058487892151, -0.02120053768157959, -0.008072569966316223, 0.0999889075756073, 0.02642759680747986, -0.12112843990325928, -0.01247735321521759, -0.11870867013931274, 0.06306290626525879, 0.06041190028190613, 0.029754742980003357, -0.07539087533950806, -0.015525221824645996, -0.11115157604217529, -0.012322977185249329, 0.022541582584381104, 0.045628949999809265, -0.04299174249172211, -0.024231895804405212, 0.04896529018878937, -0.083757683634758, -0.06585212051868439, 0.05895555019378662, 0.10637925565242767, -0.04090581834316254, -0.009520918130874634, 0.039834484457969666, -0.0075865983963012695, 0.01953941583633423, 0.01474587619304657, -0.026907682418823242, -0.061259761452674866, -0.08422607183456421, -0.011878103017807007, 0.0033280104398727417, 0.04516439139842987, 0.012798607349395752, 0.1248721033334732, 0.03752177953720093, -0.07499787211418152, 0.007541626691818237, -0.06322188675403595, 0.015553146600723267, -0.023118942975997925, -0.013612881302833557, -0.11838468909263611, 0.12430065870285034, 0.103742316365242, -0.028857439756393433, -0.07760708034038544, -0.04763782024383545, -0.013245925307273865, 0.1004573404788971, -0.02035883069038391], [-0.11647908389568329, -0.10816448926925659, -0.08036753535270691, -0.07266591489315033, -0.09774310886859894, -0.025129184126853943, 0.08784109354019165, -0.03822411596775055, -0.009082749485969543, -0.041713446378707886, -0.1210724413394928, -0.06193874776363373, 0.024151921272277832, 0.08558596670627594, 0.07502391934394836, 0.07438530027866364, -0.00894087553024292, -0.09963907301425934, -0.07817471027374268, -0.05304490029811859, 0.04119144380092621, 0.01208716630935669, -0.009826883673667908, -0.07865719497203827, -0.01607850193977356, -0.020373165607452393, -0.0720398873090744, 0.02112935483455658, -0.08248043060302734, -0.041867271065711975, -0.03789323568344116, -0.09161429107189178, -0.02858683466911316, -0.10944929718971252, -0.024495884776115417, 0.045426174998283386, -0.05731235444545746, 0.06319817900657654, 0.024471387267112732, -0.009636744856834412, -0.08006878197193146, 0.0011235177516937256, 0.08151763677597046, -0.01293259859085083, 0.0636012852191925, 0.10863551497459412, 0.02311970293521881, -0.10442353785037994, 0.032462671399116516, -0.10059638321399689, -0.07233671844005585, -0.09033410251140594, -0.05075769126415253, -0.0069651007652282715, 0.04754680395126343, -0.11588174104690552, -0.10431176424026489, -0.058828577399253845, -0.12107324600219727, 0.09054389595985413, 0.07021096348762512, 0.01309138536453247, 0.02648012340068817, 0.030592918395996094], [-0.08977387845516205, 0.012376397848129272, -0.0803520455956459, -0.02973315119743347, 0.11518825590610504, 0.08377265930175781, 0.12139678001403809, 0.08507533371448517, 0.0634213462471962, -0.0887245461344719, -0.12880361080169678, -0.025946777313947678, -0.13154900074005127, -0.11005479097366333, 0.08042845875024796, -0.060632217675447464, -0.06244131922721863, -0.054088294506073, 0.0398414209485054, 0.09617152810096741, 0.053173378109931946, 0.04881179332733154, 0.03401723504066467, 0.04832012951374054, 0.09769381582736969, 0.0699370801448822, 0.073906809091568, 0.075738325715065, -0.05535343289375305, -0.08914956450462341, -0.11462028324604034, 0.004768328741192818, 0.06089898943901062, 0.03284885734319687, -0.11789928376674652, 0.0801956057548523, -0.00550033338367939, -0.14019258320331573, -0.018431534990668297, -0.01931910589337349, 0.06437027454376221, 0.04156291112303734, -0.09896402060985565, 0.04692928493022919, 0.10246412456035614, 0.11667564511299133, -0.019989609718322754, -0.024122869595885277, -0.031151216477155685, 0.0669456496834755, -0.09196440875530243, 0.09510516375303268, -0.02714115008711815, -0.07287934422492981, 0.03516097739338875, 0.06320870667695999, 0.0009684953256510198, -0.014559066854417324, 0.074598029255867, -0.07725443691015244, -0.09333612769842148, -0.035992592573165894, -0.07668489217758179, -0.10152260959148407], [0.04496419429779053, 0.029538080096244812, -0.015991494059562683, 0.07553507387638092, -0.01656554639339447, 0.012456417083740234, -0.009944379329681396, 0.04056526720523834, -0.007075637578964233, -0.03130519390106201, -0.09548540413379669, -0.09356805682182312, -0.08439590036869049, 0.01115851104259491, -0.006119906902313232, -0.04178686439990997, 0.019427716732025146, 0.06473739445209503, -0.09102672338485718, -0.08749735355377197, 0.04253348708152771, -0.11537370085716248, -0.09336178004741669, -0.06596289575099945, -0.10911442339420319, -0.032907336950302124, -0.006786927580833435, 0.006062701344490051, -0.018518850207328796, -0.12090699374675751, 0.0694665014743805, 0.06166870892047882, -0.12296976149082184, 0.0648605078458786, -0.04464203119277954, -0.03885498642921448, -0.09244239330291748, 0.046855196356773376, 0.05600278079509735, -0.11626605689525604, 0.03448683023452759, -0.01780550181865692, 0.08228853344917297, 0.05760420858860016, -0.12041865289211273, -0.026821285486221313, 0.03467261791229248, -0.03666427731513977, -0.06547406315803528, 0.0048135071992874146, 0.03741709887981415, -0.03598250448703766, 0.0921730101108551, 0.09187525510787964, -0.08567927777767181, 0.05139407515525818, -0.05137114226818085, -0.06538686156272888, -0.09906423091888428, -0.007981643080711365, 0.11684960126876831, -0.08859264850616455, -0.11582747101783752, -0.0038530677556991577], [0.012064963579177856, -0.11336414515972137, -0.07192329317331314, 0.1147453784942627, -0.02860894799232483, -0.06594625860452652, 0.07037433981895447, 0.0953613817691803, 0.0016641481779515743, 0.07556341588497162, -0.03912452235817909, -0.011926168575882912, -0.03604764863848686, 0.08936131000518799, 0.01888991892337799, 0.1157822385430336, -0.0005915462970733643, 0.07645615935325623, -0.037609901279211044, -0.08411242067813873, 0.0654272511601448, 0.09611526131629944, 0.021999835968017578, 0.05551409721374512, -0.053382061421871185, -0.08569563180208206, 0.11054816842079163, -0.08795090019702911, -0.08925579488277435, -0.10755111277103424, -0.09371411800384521, 0.08526280522346497, -0.11458244919776917, 0.10740053653717041, -0.024529099464416504, 0.08162753283977509, 0.09162916243076324, -0.09847445040941238, -0.08230817317962646, -0.003138813888654113, 0.11706046760082245, -0.03365626186132431, -0.06744559109210968, 0.03805443271994591, -0.043333977460861206, -0.015273123979568481, -0.04246781766414642, -0.11450129002332687, 0.11407333612442017, 0.134222149848938, 0.012711819261312485, 0.0618075430393219, -0.021543996408581734, -0.09664203226566315, -0.016778841614723206, 0.0720786526799202, 0.09039850533008575, -0.031018555164337158, -0.017631303519010544, -0.0906146988272667, -0.09380293637514114, -0.10838255286216736, -0.01089392602443695, 0.02863171137869358], [-0.009796276688575745, -0.10800348222255707, 0.08130129426717758, 0.01895749568939209, -0.10262833535671234, -0.004236018750816584, -0.008013278245925903, 0.10594071447849274, -0.001675863517448306, 0.05604181066155434, 0.027336420491337776, 0.0755564272403717, -0.11041431874036789, -0.014269769191741943, -0.034423649311065674, 0.02383897267282009, -0.10483027994632721, 0.036073192954063416, -0.06607271730899811, 0.062205683439970016, 0.012571524828672409, -0.1300283968448639, -0.0732039213180542, -0.07465240359306335, 0.07229280471801758, -0.10120800137519836, -0.0007216483354568481, 0.09965819120407104, -0.012122824788093567, -0.04464501142501831, 0.09383851289749146, 0.00866401381790638, -0.005288019776344299, 0.011081146076321602, 0.10641776025295258, 0.11962270736694336, -0.08705989271402359, -0.07233838737010956, -0.005510498769581318, 0.06675121188163757, 0.10667230188846588, 0.0248834528028965, -0.03467485308647156, 0.036052156239748, -0.08464892208576202, 0.0013870447874069214, 0.0008598119020462036, -0.12692086398601532, -0.08937336504459381, 0.07643430680036545, 0.045887745916843414, 0.034441977739334106, 0.026035159826278687, -0.04722701013088226, 0.0030268384143710136, -0.12044068425893784, 0.002459252020344138, 0.06503559648990631, -0.05246470496058464, -0.10298945754766464, -0.1335044652223587, 0.07619324326515198, 0.06683114171028137, 0.04001519829034805], [0.0033948123455047607, 0.04088762402534485, -0.08801738917827606, -0.04244624078273773, -0.03470452129840851, -0.07866912335157394, -0.08473372459411621, -0.06534497439861298, -0.10594410449266434, 0.1048789769411087, -0.03326836973428726, -0.09890469908714294, 0.07133866846561432, -0.06472565233707428, -0.0637756958603859, 0.04188429191708565, 0.0724734514951706, -0.08027760684490204, 0.010230115614831448, -0.10669370740652084, 0.056632719933986664, -0.09740632027387619, 0.09119027853012085, 0.02046850323677063, 0.03341450169682503, 0.08155595511198044, -0.03521303832530975, -0.019471198320388794, 0.0018915235996246338, 0.06012621521949768, -0.10238493978977203, 0.07009845972061157, -0.031322672963142395, 0.06514989584684372, -0.0222921222448349, -0.057720646262168884, -0.10159502178430557, 0.02105538360774517, -0.04229645058512688, 0.06758565455675125, 0.018211722373962402, 0.051993828266859055, 0.014886602759361267, -0.019816407933831215, 0.016137734055519104, -0.08672662079334259, -0.08602070808410645, -0.025470836088061333, -0.07097917795181274, 0.07193649560213089, -0.09782005101442337, 0.09512859582901001, -0.13918118178844452, 0.09802716970443726, -0.0356164425611496, -0.13034075498580933, -0.04228491336107254, 0.0074828555807471275, -0.04566504433751106, 0.052927207201719284, -0.1179502084851265, 0.060421690344810486, -0.06599593162536621, -0.06562546640634537], [-0.08125010132789612, 0.10482409596443176, -0.03684071823954582, -0.035290539264678955, -0.09482409060001373, -0.04467909783124924, 0.08734023571014404, 0.04654558002948761, 0.08425024151802063, 0.0395064502954483, 0.09141648560762405, -0.050649430602788925, 0.10722257196903229, -0.024991944432258606, 0.0954512283205986, 0.0526730902493, -0.08553807437419891, 0.040126774460077286, -0.0294057447463274, -0.021051308140158653, -0.00947467889636755, 0.10111574828624725, 0.016502514481544495, -0.01222287118434906, -0.02216758392751217, -0.05432378500699997, -0.07724566757678986, -0.04145325720310211, 0.07228142023086548, -0.08422081172466278, 0.07682308554649353, 0.021252142265439034, -0.035318657755851746, -0.04452570155262947, 0.028552860021591187, -0.11035729944705963, 0.021335579454898834, 0.04455002397298813, -0.05945243686437607, -0.12556445598602295, -0.036747708916664124, -0.08905404806137085, -0.02423940598964691, -0.031702689826488495, -0.08120740950107574, 0.097795769572258, 0.05619068443775177, -0.12057732790708542, -0.06029724329710007, -0.05828651413321495, 0.024945862591266632, -0.08326174318790436, 0.07363077253103256, 0.023604586720466614, -0.0729730874300003, 0.1109115481376648, -0.03400905802845955, 0.08453243970870972, -0.061025623232126236, 0.01638513244688511, -0.054979708045721054, -0.09248878061771393, -0.10941243171691895, -0.024813203141093254], [-0.10580708086490631, -0.042739540338516235, -0.08817396312952042, 0.11404572427272797, -0.08486424386501312, -0.08295974880456924, 0.02347525954246521, -0.06043633818626404, 0.07672207802534103, 0.02386837638914585, 0.03721122071146965, -0.12621745467185974, -0.0477224737405777, 0.05919468402862549, 0.052977122366428375, 0.09981151670217514, -0.017117157578468323, 0.10221336781978607, -0.025991927832365036, 0.11236312240362167, 0.06512927263975143, 0.1187412366271019, 0.08051049709320068, 0.05291543900966644, 0.030856288969516754, 0.027386130765080452, -0.01143452525138855, 0.03942345082759857, 0.06396786868572235, 0.0004823654890060425, -0.07262519001960754, 0.09395372122526169, -0.09465797245502472, 0.06089046597480774, 0.08099596202373505, -0.07245086133480072, -0.05690956115722656, -0.017302656546235085, -0.004670944530516863, -0.10529037564992905, 0.03763577342033386, 0.09095622599124908, 0.08990468084812164, 0.12831038236618042, 0.09158657491207123, 0.036883026361465454, -0.09877780079841614, 0.06008652597665787, -0.06411904096603394, -0.04571332409977913, -0.033305574208498, 0.09547369927167892, 0.0030108352657407522, 0.08735787868499756, 0.07738693058490753, -0.031652115285396576, 0.05504904314875603, 0.09470590204000473, -0.07166863977909088, -0.10281327366828918, 0.0010751623194664717, 0.08260396122932434, 0.007406547665596008, -0.048458319157361984], [-0.09770239889621735, -0.007751584053039551, -0.10930398851633072, 0.0195624977350235, 0.004240885376930237, 0.06021181121468544, 0.11857928335666656, 0.01920129358768463, -0.0535828173160553, -0.08693190664052963, -0.07361935079097748, -0.002311469055712223, 0.10177505016326904, 0.0005956143140792847, 0.06886889040470123, -0.07964649051427841, -0.03657141327857971, -0.10184542089700699, -0.1450214385986328, 0.06413490325212479, -0.01733209379017353, 0.045134931802749634, -0.014003202319145203, 0.07069575786590576, -0.01069165300577879, 0.01467883586883545, -0.0011542737483978271, 0.07447677850723267, -0.06572473049163818, 0.04442591965198517, 0.036259979009628296, -0.01135068479925394, -0.11543993651866913, -0.0030295997858047485, 0.1043853759765625, 0.014103055000305176, 0.047675419598817825, -0.07823853194713593, 0.058771636337041855, -0.12274622917175293, 0.08498810231685638, 0.0634482353925705, -0.09568794071674347, -0.009900837205350399, -0.02089349925518036, -0.11361508071422577, -0.12173062562942505, -0.04145701229572296, 0.04946163296699524, -0.12746188044548035, -0.03406168892979622, 0.09042681008577347, -0.12323439121246338, -0.09923318028450012, 0.03503712639212608, 0.012226146645843983, 0.06303189694881439, 0.05829861760139465, -0.056055910885334015, 0.045901160687208176, 0.04825516417622566, 0.03166620433330536, -0.013522088527679443, 0.07431970536708832], [0.02190171182155609, 0.08349190652370453, -0.08227604627609253, -0.08651560544967651, -0.03657519817352295, -0.019362062215805054, -0.07199262082576752, 0.10091835260391235, 0.09412442147731781, 0.0493704229593277, 0.0721236914396286, -0.07876165211200714, 0.046116068959236145, 0.08283142745494843, -0.0010620206594467163, -0.11552876234054565, -0.052770212292671204, 0.048442766070365906, -0.09952749311923981, 0.01089489459991455, 0.04151099920272827, 0.06869064271450043, -0.08429345488548279, -0.11402663588523865, 0.033438727259635925, 0.04833829402923584, 0.11476758122444153, -0.04737049341201782, 0.06395253539085388, 0.0696769654750824, 0.06833788752555847, 0.07450000941753387, 0.11785149574279785, -0.12438845634460449, 0.10930889844894409, 0.06365765631198883, -0.0123366117477417, -0.03431329131126404, 0.03828166425228119, -0.09429064393043518, 0.11947430670261383, 0.007939711213111877, -0.10891093313694, 0.024644136428833008, -0.11133845150470734, -0.04355272650718689, 0.04051579535007477, 0.05187797546386719, -0.02858671545982361, 0.08628010749816895, -0.06289759278297424, -0.11432166397571564, -0.11724366247653961, 0.07534871995449066, 0.07634378969669342, -0.07298654317855835, 0.03880620002746582, 0.12489642202854156, -0.053867727518081665, -0.05838222801685333, -0.11117501556873322, 0.06724473834037781, -0.04697781801223755, 0.06733694672584534], [-0.08907712996006012, 0.012905105948448181, 0.05753962695598602, 0.02504734694957733, -0.008778244256973267, 0.039720166474580765, 0.10526591539382935, -0.0882619172334671, 0.07886514812707901, 0.10735306888818741, -0.004743094556033611, -0.0951235368847847, -0.06427344679832458, 0.05466872453689575, -0.04336150363087654, 0.11040353775024414, 0.10861530900001526, -0.10440169274806976, 0.12917809188365936, -0.08591865748167038, 0.05918320640921593, -0.12280883640050888, -0.12334740161895752, -0.03262878954410553, 0.02634088695049286, -0.012472682632505894, -0.01373426616191864, 0.06703150272369385, 0.021126985549926758, 0.0202152281999588, 0.0743030458688736, -0.10617615282535553, 0.04067358374595642, 0.007340190000832081, -0.032101258635520935, 0.11985793709754944, 0.0005799560458399355, 0.05760212242603302, -0.128987655043602, -0.10660916566848755, 0.042668357491493225, 0.046367086470127106, 0.01250407099723816, 0.15184959769248962, -0.11442174017429352, 0.12156075239181519, -0.10613341629505157, -0.012152674607932568, 0.05417210981249809, 0.02711583487689495, 0.03418487682938576, 0.042460858821868896, -0.05459677428007126, -0.0035630017518997192, -0.07743147015571594, 0.034116800874471664, 0.0867789089679718, -0.016186993569135666, -0.05739256367087364, -0.009614301845431328, 0.08103304356336594, 0.06933537125587463, 0.09720748662948608, -0.12738896906375885], [-0.02282017469406128, -0.02010643482208252, 0.06517219543457031, 0.11673001945018768, 0.04183037579059601, -0.0042798519134521484, 0.08705362677574158, -0.016258493065834045, -0.0405379980802536, 0.09153588116168976, -0.09510962665081024, 0.09158167243003845, -0.06266617774963379, -0.06189711391925812, -0.04301080107688904, -0.1241680383682251, -0.02506357431411743, 0.020285606384277344, 0.010129347443580627, 0.07618480920791626, 0.08756217360496521, 0.006413578987121582, 0.030247226357460022, 0.07937373220920563, 0.0016431808471679688, 0.07291600108146667, 0.09007814526557922, 0.06136003136634827, 0.09876537322998047, -0.10419188439846039, 0.028663381934165955, 0.080876424908638, 0.10642765462398529, 0.026937901973724365, -0.01618880033493042, 0.11096851527690887, 0.047916561365127563, -0.07894556224346161, -0.0463484525680542, -0.12157894670963287, -0.01815137267112732, -0.122065469622612, -0.11411997675895691, 0.0742688775062561, 0.05377432703971863, -0.05594688653945923, -0.11746446788311005, 0.11201663315296173, 0.1180146187543869, 0.0018592923879623413, 0.08866578340530396, -0.12423259019851685, -0.09270758926868439, 0.020982369780540466, 0.08202019333839417, -0.1033136248588562, 0.04990978538990021, 0.08251260221004486, -0.06675887107849121, -0.10013124346733093, 0.09007453918457031, 0.060403913259506226, -0.09410639107227325, 0.10045163333415985], [0.06352561712265015, -0.09094369411468506, -0.023895911872386932, -0.03630676865577698, 0.060765743255615234, 0.10381574928760529, -0.04607228934764862, 0.025498464703559875, -0.02794155478477478, -0.11143962293863297, -0.02626107633113861, 0.04569998383522034, -0.002370313974097371, 0.047024667263031006, 0.05791987478733063, 0.052809763699769974, 0.05790145695209503, 0.04446392133831978, -0.036038804799318314, -0.019796276465058327, -0.03067900985479355, 0.030831193551421165, -0.07563579082489014, -0.004359692335128784, 0.10343053936958313, -0.026715194806456566, 0.0022387057542800903, -5.614757537841797e-05, 0.03487128019332886, 0.036881059408187866, -0.0405011922121048, -0.016113324090838432, 0.09277626872062683, -0.015032908879220486, 0.07160195708274841, 0.0933023989200592, -0.035713061690330505, -0.013363952748477459, 0.005144309252500534, -0.08793334662914276, 0.07272863388061523, 0.11574872583150864, -0.027370944619178772, 0.030744114890694618, -0.08781212568283081, -0.02510926127433777, -0.0471072643995285, 0.06421389430761337, 0.04998953640460968, -0.03321661055088043, 0.058015428483486176, -0.11149531602859497, 0.08489641547203064, 0.08073541522026062, 0.00847667083144188, 0.07216015458106995, -0.033763229846954346, -0.0854867547750473, -0.04852067679166794, 0.0634002834558487, 0.016794003546237946, -0.11503255367279053, 0.06706815958023071, 0.11810562759637833], [-0.058707207441329956, -0.12238547205924988, 0.05083833634853363, -0.01394730806350708, -0.11634217202663422, 0.031096726655960083, -0.0023963451385498047, 0.07040543854236603, -0.06654074788093567, 0.016450300812721252, -0.013972282409667969, 0.1090870052576065, -0.03432558476924896, 0.0212392657995224, -0.0942649096250534, -0.11884328722953796, -0.0024573057889938354, 0.021310314536094666, -0.09782832860946655, -0.08932118117809296, 0.09341408312320709, -0.048637256026268005, -0.015302613377571106, 0.025937288999557495, 0.11381004750728607, -0.017704695463180542, -0.0431990772485733, -0.030132144689559937, -0.11703401803970337, -0.052937448024749756, -0.022702276706695557, -0.057411253452301025, -0.07834842801094055, -0.018338128924369812, 0.0626605898141861, -0.07385672628879547, -0.11752866208553314, -0.10242536664009094, 0.08817274868488312, 0.04847162961959839, -0.04732581973075867, 0.07314316928386688, -0.11108355224132538, 0.09861677885055542, -0.07676120102405548, -0.11063045263290405, 0.07189498841762543, 0.021066665649414062, -0.08427149057388306, -0.016071319580078125, -0.07535037398338318, -0.020219966769218445, 0.012606903910636902, -0.07435557246208191, -0.09397678077220917, -0.10732042789459229, -0.008752644062042236, 0.10580354928970337, 0.10704725980758667, 0.006763160228729248, 0.06802244484424591, 0.00021034479141235352, 0.11016286909580231, -0.042541444301605225], [-0.12070640921592712, 0.0872788280248642, -0.01650846004486084, 0.032515689730644226, 0.05043956637382507, -0.05134119093418121, -0.0843421220779419, 0.12007270753383636, 0.05800287425518036, -0.11917266249656677, -0.06014031171798706, -0.027893245220184326, -0.12394928932189941, -0.00045102834701538086, -0.03160648047924042, -0.09537734091281891, 0.017080873250961304, 0.07804779708385468, -0.023596197366714478, 0.06922760605812073, -0.08112987875938416, 0.09272347390651703, 0.026577100157737732, -0.058554038405418396, 0.0515502393245697, -0.09425650537014008, -0.052221253514289856, 0.10786589980125427, -0.049473583698272705, -0.0866498053073883, -0.03210541605949402, -0.10164818167686462, 0.07088492810726166, -0.11230769753456116, 0.04205133020877838, 0.03529606759548187, 0.028294801712036133, -0.007317349314689636, -0.0854349434375763, 0.07087251543998718, 0.11161907017230988, 0.09599354863166809, 0.008662834763526917, 0.02695341408252716, 0.016462117433547974, -0.08854825794696808, 0.08937261998653412, 0.04637317359447479, 0.03929133713245392, 0.03982660174369812, 0.027532577514648438, -0.032009899616241455, 0.007184058427810669, -0.02251988649368286, -0.07479645311832428, -0.03599543869495392, -0.06619484722614288, -0.016357481479644775, 0.06976625323295593, -0.07971552014350891, -0.011486276984214783, 0.07248036563396454, -0.10243447124958038, -0.08394969999790192], [-0.08669817447662354, 0.0015238523483276367, 0.022727321833372116, 0.0007374584674835205, -0.007342725992202759, 0.05955038592219353, -0.012278184294700623, 0.08481158316135406, 0.0033757886849343777, 0.014746751636266708, 0.005859515629708767, 0.10439618676900864, 0.11017452925443649, 0.11974595487117767, 0.10824211686849594, 0.028518350794911385, -0.0234498530626297, -0.057735465466976166, 0.05432157590985298, -0.1377197951078415, -0.08002135902643204, -0.06554986536502838, 0.06069864332675934, 0.11628496646881104, 0.036382511258125305, 0.025216568261384964, 0.01709325611591339, 0.07044966518878937, -0.06493091583251953, -0.08419130742549896, -0.06276130676269531, -0.01029151864349842, -0.005654260516166687, -0.048869673162698746, 0.019604459404945374, 0.0863163024187088, -0.04358227178454399, -0.01869155466556549, 0.08235348761081696, 0.07543428987264633, 0.11835357546806335, 0.11035902053117752, 0.1225678026676178, -0.04801226034760475, -0.058458030223846436, -0.07265110313892365, -0.06810477375984192, 0.11408305168151855, 0.10695929080247879, 0.06119518354535103, 0.06837490200996399, 0.019620155915617943, 0.011018817313015461, -0.002110779285430908, -0.10097122937440872, -0.014412567019462585, 0.026858773082494736, -0.005792602431029081, -0.056073833256959915, -0.050916146486997604, 0.022718897089362144, 0.05022938549518585, 0.06705963611602783, -0.03626024350523949], [-0.11558708548545837, -0.04651254415512085, 0.1080089658498764, -0.07035574316978455, 0.06718629598617554, 0.043121710419654846, -0.016396939754486084, -0.04815450310707092, -0.1243847981095314, 0.06415627151727676, 0.07993125915527344, 0.0719468742609024, -0.00990759115666151, 0.0390988290309906, 0.08865147083997726, 0.07061081379652023, 0.12081088125705719, 0.02565777860581875, -0.05426664277911186, 0.04938492551445961, -0.0678005963563919, -0.08195234090089798, 0.030397355556488037, -0.03977721929550171, -0.055260565131902695, -0.10166022926568985, -0.08718101680278778, 0.06994713842868805, 0.03605981171131134, -0.010612204670906067, 0.0776936411857605, 0.08639640361070633, -0.013667717576026917, -0.09289766103029251, 0.05218420922756195, 0.03517450392246246, -0.08540309220552444, -0.10596583038568497, -0.003186560468748212, 0.02649826370179653, 0.004372656345367432, -0.011508564464747906, 0.07838273048400879, -0.046862464398145676, 0.08099238574504852, 0.013281092047691345, 0.07894828915596008, -0.1286192238330841, 0.05012498423457146, 0.032600052654743195, -0.08735000342130661, -0.006279982626438141, 0.0755690261721611, -0.06036670506000519, -0.0227203406393528, 0.1063336580991745, -0.023474324494600296, -0.10578995198011398, -0.08640818297863007, -0.015099640004336834, 0.052513688802719116, -0.08259047567844391, 0.015022680163383484, 0.01798408478498459], [-0.05835658311843872, 0.031766340136528015, 0.0499056875705719, 0.03930202126502991, -0.0358055979013443, -0.09918001294136047, 0.004398569464683533, 0.08000880479812622, 0.06505727767944336, 0.017543572932481766, 0.10942437499761581, -0.042658474296331406, -0.013508989475667477, 0.07878045737743378, 0.0754246935248375, 0.045679450035095215, 0.019749119877815247, 0.10825653374195099, 0.014081725850701332, 0.006690938025712967, -0.1224246621131897, 0.02459852583706379, 0.05516126751899719, -0.08529351651668549, -0.08768793195486069, -0.08616539090871811, -0.10815343260765076, -0.08225561678409576, 0.07169683277606964, 0.0828680694103241, -0.11979767680168152, -0.035022586584091187, 0.029420480132102966, -0.11056370288133621, 0.03155778348445892, -0.0308973491191864, -0.03949885442852974, 0.020873090252280235, 0.035100098699331284, -0.059725482016801834, 0.06544415652751923, -0.13018672168254852, 0.04807506501674652, -0.026392966508865356, -0.08519801497459412, -0.1027127057313919, -0.052874743938446045, 0.05018806830048561, 0.007754996884614229, -0.04598117247223854, 0.012678381986916065, 0.08171084523200989, 0.06369423866271973, 0.08036172389984131, -0.0971931740641594, 0.07289475947618484, 0.12048640847206116, -0.010524389334022999, 0.04949691891670227, 0.09515567123889923, -0.01051576342433691, 0.043633416295051575, 0.02247518301010132, 0.08252598345279694], [-0.05428558588027954, 0.04531586170196533, -0.08505608141422272, 0.11816146969795227, -0.11734446883201599, 0.030375711619853973, 0.039525046944618225, 0.002439752221107483, 0.05662964656949043, -4.533661558525637e-05, -0.046095166355371475, -0.08947423845529556, 0.08701661974191666, 0.013062819838523865, 0.03066009283065796, -0.043136823922395706, 0.0527702271938324, 0.10790298879146576, 0.053605131804943085, -0.006972995121032, 0.04636732488870621, -0.00566372973844409, 0.06063137948513031, -0.07935433089733124, -0.03840552642941475, -0.11131737381219864, 0.11853286623954773, -0.011674314737319946, -0.08626197278499603, 0.07266810536384583, -0.1080416589975357, -0.05138972029089928, 0.02217249572277069, 0.04370834678411484, 0.023434773087501526, -0.04422716796398163, -0.0031239758245646954, -0.10387320816516876, 0.03493701294064522, -0.0641101747751236, -0.04699210822582245, -0.11944153904914856, -0.025260433554649353, -0.061734363436698914, -0.11411997675895691, 0.006115809082984924, 0.057200804352760315, -0.03564683347940445, -0.03823363408446312, -0.13142655789852142, -0.003007270395755768, 0.01448907982558012, -0.050808850675821304, -0.030355706810951233, -0.012766270898282528, 0.05233212560415268, -0.07575477659702301, 0.07856354117393494, 0.0864761546254158, -0.10467706620693207, 0.030364694073796272, 0.1114271730184555, -0.09779438376426697, 0.03514499217271805], [-0.11868046224117279, -0.12176325917243958, 0.1314930021762848, -0.03942781686782837, 0.10170494019985199, 0.03816106915473938, 0.06641124188899994, 0.09443829953670502, -0.11132451146841049, 0.09799160063266754, -0.08275061845779419, 0.0588761605322361, -0.0215204618871212, 0.06677369773387909, 0.11063934117555618, -0.056716348975896835, 0.042829930782318115, 0.050461653620004654, 0.03700372949242592, 0.07967717200517654, -0.050330158323049545, 0.08217492699623108, -0.07975487411022186, 0.04519924521446228, 0.10296954214572906, 0.016695639118552208, -0.05082225799560547, 0.019310876727104187, 0.11555631458759308, -0.04560767114162445, 0.013928711414337158, -0.04631458595395088, -0.06414930522441864, 0.07412064075469971, 0.1176726222038269, -0.004740402102470398, -0.11591511964797974, 0.03791792690753937, -0.10558073967695236, 0.013814439065754414, -0.11583718657493591, 0.10922019928693771, 0.05028940737247467, 0.020757967606186867, -0.05345222353935242, -0.07006905972957611, 0.07848513126373291, -0.006646696478128433, 0.08514420688152313, -0.11801107227802277, 0.053865570574998856, -0.10280710458755493, -0.11877619475126266, 0.1174851655960083, -0.1003701239824295, 0.045636434108018875, 0.0424652174115181, 0.08736354857683182, 0.0010858782334253192, -0.024191392585635185, 0.05434373393654823, 0.04709823429584503, -0.08827650547027588, -0.036559101194143295], [0.1181415468454361, 0.09359100461006165, -0.05097123607993126, 0.03144678473472595, 0.047621190547943115, 0.0035215974785387516, -0.0776127427816391, 0.06903152167797089, -0.12906180322170258, 0.01716626062989235, 0.08882015198469162, 0.06232929602265358, 0.08809702098369598, -0.04342585802078247, 0.10413367301225662, 0.002618701197206974, -0.07802872359752655, 0.024522533640265465, 0.07528629153966904, -0.013286596164107323, -0.05988455191254616, 0.018672112375497818, 0.09939606487751007, -0.049166128039360046, -0.11587832123041153, -0.1303645521402359, 0.11092868447303772, 0.08301195502281189, -0.11009076237678528, -0.025453969836235046, -0.04656778275966644, -0.12664580345153809, 0.0878593772649765, 0.06900793313980103, -0.10338005423545837, 0.10811519622802734, -0.09969441592693329, 0.004224639385938644, 0.08489241451025009, 0.00519714318215847, 0.08933021128177643, 0.04613811895251274, -0.1119920015335083, 0.10697204619646072, -0.10521608591079712, 0.0547085702419281, 0.05712983012199402, 0.001064844662323594, 0.044153861701488495, -0.07190189510583878, -0.061299074441194534, 0.04907160624861717, -0.0856865718960762, -0.017040491104125977, 0.049274992197752, -0.07281786948442459, -0.08406426757574081, -0.03944139555096626, 0.08596280217170715, -0.03461918607354164, 0.11138363182544708, -0.03319229185581207, 0.0023832321166992188, 0.09166588634252548], [0.015606790781021118, -0.06603692471981049, 0.09506151080131531, 0.10053087770938873, -0.056433796882629395, 0.021525397896766663, 0.027616292238235474, 0.006207689642906189, 0.0719098299741745, 0.09402459859848022, 0.049098044633865356, -0.09418821334838867, 0.01607450842857361, -0.013474315404891968, -0.10669784247875214, 0.10729970037937164, 0.023932158946990967, -0.06336835026741028, 0.008780598640441895, 0.02482028305530548, 0.060386061668395996, 0.10589025914669037, -0.11733567714691162, 0.09290090203285217, 0.1084241271018982, 0.003820553421974182, -0.05166587233543396, 0.10725550353527069, 0.023886755108833313, -0.08048510551452637, 0.027817249298095703, -0.05240972340106964, 0.018189311027526855, 0.046633630990982056, 0.07567670941352844, 0.0442134290933609, -0.06556443870067596, 0.025294333696365356, -0.03132624924182892, -0.009313732385635376, -0.11120438575744629, -0.057287752628326416, -0.11112545430660248, -0.005926787853240967, -0.029653385281562805, 0.0639159232378006, -0.0723496824502945, 0.04457099735736847, -0.12115554511547089, 0.042977944016456604, -0.009844467043876648, 0.007885664701461792, -0.107625812292099, 0.05816587805747986, -0.07694214582443237, -0.026088669896125793, 0.0008677691221237183, 0.0008855462074279785, -0.01470804214477539, 0.11771561205387115, 0.11783745884895325, 0.09305794537067413, -0.06533423066139221, -0.08050371706485748], [-0.06455942988395691, 0.041720300912857056, -0.06965699791908264, -0.1011255532503128, -0.09198404848575592, 0.029199257493019104, -0.033294543623924255, 0.0014499276876449585, -0.08579988777637482, -0.03895549476146698, 0.03858408331871033, -0.06347556412220001, 0.03325510025024414, 0.05061569809913635, -0.022940605878829956, -0.0029578953981399536, -0.0793769508600235, 0.11264730989933014, -0.08291563391685486, 0.006590709090232849, 0.09236855804920197, 0.10242237150669098, 0.03482025861740112, 0.0705430805683136, 0.08803367614746094, 0.016606181859970093, -0.06666585803031921, -0.09107953310012817, -0.09759865701198578, 0.060481488704681396, -0.11057083308696747, 0.0841493010520935, -0.063743457198143, 0.09705734252929688, 0.0888119786977768, -0.0004047602415084839, 0.011897563934326172, -0.0006133168935775757, -0.09904490411281586, -0.12202110886573792, -0.11691655218601227, -0.11272737383842468, 0.0679355263710022, 0.028984293341636658, 0.007706359028816223, -0.05028115212917328, -0.05404117703437805, -0.07405601441860199, -0.0697748064994812, -0.0177268385887146, -0.029152706265449524, -0.01728855073451996, 0.10915069282054901, -0.0010934770107269287, 0.08381624519824982, -0.06232798099517822, -0.07322216033935547, 0.035424113273620605, -0.10892324149608612, -0.12039989233016968, -0.08070160448551178, -0.07463136315345764, -0.018564268946647644, 0.007116153836250305], [0.10146154463291168, -0.03501904010772705, -0.05953308939933777, 0.0005461126565933228, -0.09606221318244934, -0.1196785569190979, 0.029571890830993652, -0.09383577108383179, 0.05095106363296509, -0.04246111214160919, -0.060275062918663025, 0.09144248068332672, -0.10391266644001007, -0.021834418177604675, -0.07626095414161682, 0.088208869099617, -0.0039685070514678955, -0.07241898775100708, -0.12416544556617737, -0.03755585849285126, -0.01717868447303772, -0.09907537698745728, 0.09470000863075256, 0.06211434304714203, -0.08142077922821045, -0.05179452896118164, -0.009187623858451843, -0.045085206627845764, 0.007480189204216003, 0.0060212016105651855, -0.10919752717018127, 0.04326353967189789, 0.0824359655380249, -0.02527555823326111, -0.0629459023475647, 0.0364222377538681, 0.013046041131019592, 0.11458754539489746, 0.029352053999900818, 0.030324921011924744, -0.11989687383174896, 0.009180009365081787, -0.07476328313350677, -0.01728813350200653, 0.07001166045665741, 0.015499338507652283, 0.0723077654838562, 0.004755303263664246, 0.016428008675575256, 0.0411720871925354, 0.011344701051712036, -0.09718850255012512, 0.005083486437797546, -0.11921714246273041, 0.0427766889333725, 0.06167791783809662, 0.09102044999599457, 0.0992506593465805, 0.11224587261676788, -0.1229080855846405, -0.01855359971523285, 0.05514509975910187, -0.06012625992298126, 0.03078119456768036], [-0.09207527339458466, -0.09860830008983612, -0.09416051208972931, -0.007667332887649536, -0.07571089267730713, 0.07190580666065216, -0.05969156324863434, 0.003693521022796631, 0.11543889343738556, -0.0035533607006073, -0.008975863456726074, -0.08061036467552185, -0.05500802397727966, -0.08350345492362976, 0.05468904972076416, -0.03850138187408447, 0.016970962285995483, -0.09280532598495483, 0.05728289484977722, 0.0038486123085021973, 0.07042399048805237, -0.09942635893821716, 0.1067533791065216, -0.05681632459163666, -0.05288761854171753, -0.07925558090209961, 0.05723181366920471, 0.05273912847042084, 0.05702202022075653, -0.07966355979442596, -0.06978975236415863, -0.05264592170715332, -0.10495904088020325, 0.04775318503379822, 0.038182854652404785, 0.03319372236728668, 0.12434497475624084, -0.08658565580844879, -0.03316514194011688, -0.0012696236371994019, -0.007285937666893005, 0.02865225076675415, 0.02582612633705139, 0.11787034571170807, 0.03435397148132324, -0.05981813371181488, -0.07595568895339966, -0.09617237746715546, 0.055839017033576965, 0.09757007658481598, 0.09729962050914764, 0.02845895290374756, 0.03171689808368683, 0.030780404806137085, 0.10053487122058868, -0.004004284739494324, 0.03655445575714111, -0.048237428069114685, -0.12192262709140778, -0.055020496249198914, -0.02601693570613861, -0.107928067445755, 0.029568418860435486, 0.06283174455165863]]
B2 = [-0.08101758360862732, -0.05305541679263115, -0.08862370997667313, -0.09202994406223297, 0.05147264152765274, 0.06829319894313812, 0.06627976894378662, -0.06846976280212402, 0.07621879130601883, -0.0886269062757492, 0.0706150084733963, -0.034150630235672, -0.12281510978937149, 0.06637096405029297, 0.027712425217032433, -0.010017634369432926, 0.05541098117828369, -0.03497416153550148, 0.05431362986564636, 0.08419470489025116, -0.010651066899299622, 0.12173759937286377, -0.0675736591219902, 0.11278843134641647, -0.09397862106561661, 0.06238703429698944, 0.07887831330299377, -0.13204985857009888, 0.10438311100006104, -0.013812661170959473, -0.028848588466644287, 0.027096152305603027]
W4 = [[-0.018915215507149696, -0.03236912190914154, -0.13070742785930634, 0.0019644484855234623, -0.030464092269539833, -0.08412665128707886, 0.07192115485668182, -0.1193876713514328, 0.0036434424109756947, -0.07332390546798706, -0.11579199135303497, 0.08419772237539291, 0.06984970718622208, -0.09489506483078003, 0.090788334608078, 0.06767331808805466, 0.02221280336380005, 0.016067884862422943, 0.01694345474243164, 0.0875026285648346, -0.07812373340129852, -0.08587135374546051, -0.11051836609840393, 0.09330648928880692, 0.138202965259552, 0.1345936506986618, 0.01639530248939991, 0.04824892804026604, 0.12407819926738739, 0.025612741708755493, 0.14676640927791595, 0.07654358446598053], [0.08232176303863525, -0.12302899360656738, 0.03019225038588047, 0.10755700618028641, -0.030511127784848213, 0.14169703423976898, -0.04109026491641998, 0.07732759416103363, -0.12719383835792542, 0.07323159277439117, -0.038989339023828506, -0.03739514946937561, -0.04101330414414406, 0.16165271401405334, 0.15805524587631226, 0.1290135383605957, 0.17465384304523468, -0.04077305272221565, 0.15256516635417938, 0.14192569255828857, 0.1375454217195511, -0.11450247466564178, 0.0006413582595996559, -0.07562731951475143, 0.0027935642283409834, -0.12983712553977966, -0.15827049314975739, -0.1510619819164276, -0.01959805190563202, 0.015479519963264465, -0.06736410409212112, -0.11963385343551636], [-0.12791085243225098, -0.13973818719387054, -0.09000938385725021, 0.1087704598903656, 0.020063357427716255, 0.11723580211400986, 0.05174221098423004, -0.15265756845474243, 0.08011941611766815, -0.03238779306411743, 0.040593214333057404, -0.11745240539312363, 0.14710576832294464, -0.06984493881464005, -0.132686585187912, -0.13805028796195984, -0.03271713852882385, -0.07936329394578934, -0.11893688887357712, 0.12274443358182907, -0.020144715905189514, 0.06073114275932312, -0.09624062478542328, 0.04418148472905159, 0.13345244526863098, -0.059790123254060745, 0.021258648484945297, 0.0496135875582695, 0.17169596254825592, 0.07393528521060944, -0.1541878581047058, 0.14112277328968048], [0.04859939590096474, 0.05638422071933746, -0.05069437250494957, -0.020763874053955078, 0.024685250595211983, 0.006979957222938538, 0.09849773347377777, -0.1315261423587799, -0.04451720044016838, 0.11927933990955353, -0.08328770101070404, -0.0950513407588005, -0.07735198736190796, -0.06324745714664459, 0.1223316639661789, -0.10037922114133835, 0.08176626265048981, 0.1631438434123993, -0.08680568635463715, -0.057811979204416275, 0.0019960105419158936, 0.043246567249298096, -0.09492959827184677, 0.09751292318105698, 0.03937390074133873, -0.11726129800081253, 0.022844817489385605, 0.1109568253159523, -0.168124258518219, -0.09815298765897751, 0.14349247515201569, 0.01069052517414093], [0.03368454799056053, 0.026364482939243317, 0.013829710893332958, 0.10129594057798386, 0.09771044552326202, -0.13082636892795563, -0.05004611611366272, -0.08204741775989532, 0.035716891288757324, -0.12086988985538483, -0.13445574045181274, 0.14262139797210693, -0.049974311143159866, -0.10814705491065979, -0.11618567258119583, 0.055117737501859665, 0.05913999676704407, -0.07538948953151703, -0.11339889466762543, -0.055065009742975235, 0.039345934987068176, 0.004778698086738586, 0.14307861030101776, 0.03922678902745247, 0.03410704806447029, 0.04562002792954445, -0.10494880378246307, -0.09204177558422089, -0.05112820863723755, -0.14051809906959534, 0.02451510727405548, 0.012784019112586975]]
B4 = [-0.06762168556451797, -0.02569984458386898, 0.031054353341460228, -0.0241722259670496, -0.014328154735267162]

ACTIONS = [
    (2.0, 1.0, 1.0), # Baseline
    (3.0, 1.0, 1.0), # Hyper Aggressive (PVP focused)
    (1.5, 2.0, 1.0), # Defensive (Fear traps)
    (2.0, 1.0, 1.5), # Greedy Expansion (Neutral focused)
    (3.0, 0.5, 0.5), # Berserker (Ignore threats, rush PVP)
]

def relu(x):
    return [max(0.0, val) for val in x]

def matmul_add(x, w, b):
    # x is of shape (N,)
    # w is of shape (M, N)
    # b is of shape (M,)
    # returns shape (M,)
    out = []
    for row, bias in zip(w, b):
        val = sum(xi * wi for xi, wi in zip(x, row)) + bias
        out.append(val)
    return out

def get_meta_weights(my_econ, opp_econ, opp_agg, opp_tempo):
    # State: [My Econ, Opp Econ, Opp Aggression, Opp Tempo]
    state = [float(my_econ), float(opp_econ), float(opp_agg), float(opp_tempo)]
    
    h1 = relu(matmul_add(state, W0, B0))
    h2 = relu(matmul_add(h1, W2, B2))
    q_values = matmul_add(h2, W4, B4)
    
    # Argmax
    action_idx = q_values.index(max(q_values))
    return ACTIONS[action_idx]


# --- PROJECT BRANIAC: META-HEURISTIC OPPONENT TRACKER ---
class OpponentModel:
    def __init__(self, num_players=4):
        self.aggression_index = {i: 0.5 for i in range(num_players)}
        self.economy = {i: 0.0 for i in range(num_players)}
        self.tempo = {i: 0.0 for i in range(num_players)}
        
    def update(self, planets, fleets):
        # 1. Calculate Economy (Ships + Production * 10)
        for i in self.economy.keys():
            self.economy[i] = 0.0
            
        for p in planets:
            if p.owner != -1:
                self.economy[p.owner] += p.ships + (p.production * 10)
                
        for f in fleets:
            if f.owner != -1:
                self.economy[f.owner] += f.ships
                
        # 2. Calculate Aggression Index & Tempo
        fleet_counts = {i: 0.0 for i in self.economy.keys()}
        for f in fleets:
            fleet_counts[f.owner] += f.ships
            
        for i in self.economy.keys():
            if self.economy[i] > 0:
                self.aggression_index[i] = min(1.0, fleet_counts[i] / max(1.0, self.economy[i]))
                self.tempo[i] = fleet_counts[i]

class World:
    def __init__(self, obs, inferred_step=None):
        
        
        global COALITION_MIN_PER_CONTRIBUTOR, DEFENSE_OVERSEND, PSM_OPENING_TURN, SO1_STATIC_BONUS
        self.player = _read(obs, "player", 0)
        obs_step = _read(obs, "step", 0) or 0
        self.step = max(obs_step, inferred_step or 0)
        raw_planets = _read(obs, "planets", []) or []
        raw_fleets = _read(obs, "fleets", []) or []
        raw_init = _read(obs, "initial_planets", []) or []
        self.ang_vel = _read(obs, "angular_velocity", 0.0) or 0.0

        self.planets = [Planet(*p) for p in raw_planets]
        self.fleets = [Fleet(*f) for f in raw_fleets]
        self.initial_by_id = {Planet(*p).id: Planet(*p) for p in raw_init}
        
        # --- PROJECT BRANIAC: INITIALIZE TRACKER ---
        self.opponent_model = OpponentModel(4)
        self.opponent_model.update(self.planets, self.fleets)

        
        
        
        raw_comet_ids = _read(obs, "comet_planet_ids", []) or []
        self.comet_ids = set(int(x) for x in raw_comet_ids)
        
        
        
        self.comet_remaining = {}
        raw_comet_groups = _read(obs, "comets", []) or []
        
        
        
        self.comets = raw_comet_groups
        for grp in raw_comet_groups:
            try:
                idx = int(grp.get("path_index", 0))
                pids = grp.get("planet_ids", []) or []
                paths = grp.get("paths", []) or []
                for i, pid in enumerate(pids):
                    if i < len(paths):
                        rem = max(0, len(paths[i]) - idx)
                        self.comet_remaining[int(pid)] = rem
            except (AttributeError, TypeError, IndexError):
                continue

        self.planet_by_id = {p.id: p for p in self.planets}
        self.my_planets = [p for p in self.planets if p.owner == self.player]
        self.enemy_planets = [p for p in self.planets if p.owner not in (-1, self.player)]
        self.neutral_planets = [p for p in self.planets if p.owner == -1]

        self.remaining_steps = max(1, TOTAL_STEPS - self.step)
        self.is_opening = self.step < PSM_OPENING_TURN
        self.is_late = self.remaining_steps < LATE_FLUSH_REMAINING_TURNS

        
        self.owner_strength = defaultdict(int)
        self.owner_production = defaultdict(int)
        for p in self.planets:
            if p.owner != -1:
                self.owner_strength[p.owner] += int(p.ships)
                self.owner_production[p.owner] += int(p.production)
        for f in self.fleets:
            self.owner_strength[f.owner] += int(f.ships)

        self.my_prod = self.owner_production.get(self.player, 0)
        self.total_prod = sum(self.owner_production.values())
        self.my_prod_share = (self.my_prod / self.total_prod) if self.total_prod else 0.0
        
        if self.remaining_steps < 80 and self.my_prod_share > 0.55:
            self.is_late = True

        
        self.leader_id = None
        self.contest_leader = False

        
        
        
        
        self.owner_planet_count = defaultdict(int)
        for p in self.planets:
            if p.owner not in (-1,):
                self.owner_planet_count[p.owner] += 1
        self.weakest_enemy = None
        self.weakest_enemy_prod_share = 0.0
        if self.total_prod > 0:
            best_score = None
            for owner in self.owner_production.keys():
                if owner in (-1, self.player):
                    continue
                score = (
                    self.owner_production.get(owner, 0) * 0.5
                    + self.owner_strength.get(owner, 0) * 0.3
                    + self.owner_planet_count.get(owner, 0) * 0.2
                )
                if best_score is None or score < best_score:
                    best_score = score
                    self.weakest_enemy = owner
            if self.weakest_enemy is not None:
                their_prod = self.owner_production.get(self.weakest_enemy, 0)
                self.weakest_enemy_prod_share = (
                    their_prod / self.total_prod if self.total_prod else 0.0
                )

        
        
        
        
        
        self.arrivals_by_planet = defaultdict(list)
        for f in self.fleets:
            target, eta = fleet_target_planet(f, self.planets, self.initial_by_id, self.ang_vel)
            if target is None:
                continue
            self.arrivals_by_planet[target.id].append((eta, int(f.owner), int(f.ships)))

        
        
        
        
        self.enemy_race_eta = _compute_enemy_race_eta(self) if RACE_ENABLED else {}

        
        
        
        
        global _game_num_players
        if _game_num_players is None and self.planets:
            _game_num_players = self.num_players
        self.is_2p = (_game_num_players == 2)

        
        
        if self.is_2p:
            COALITION_MIN_PER_CONTRIBUTOR = COALITION_MIN_PER_CONTRIBUTOR_2P
            DEFENSE_OVERSEND = DEFENSE_OVERSEND_2P
            PSM_OPENING_TURN = PSM_OPENING_TURN_2P
            SO1_STATIC_BONUS = SO1_STATIC_BONUS_2P
        else:
            COALITION_MIN_PER_CONTRIBUTOR = COALITION_MIN_PER_CONTRIBUTOR_4P
            DEFENSE_OVERSEND = DEFENSE_OVERSEND_4P
            PSM_OPENING_TURN = PSM_OPENING_TURN_4P
            SO1_STATIC_BONUS = SO1_STATIC_BONUS_4P
        
        
        
        if LEADER_BASH_ENABLED and not self.is_2p:
            lead_scores = {}
            for owner in self.owner_production.keys():
                if owner == -1:
                    continue
                lead_scores[owner] = (
                    self.owner_strength.get(owner, 0) * 0.5
                    + self.owner_production.get(owner, 0) * 0.5
                )
            if lead_scores:
                top_owner = max(lead_scores, key=lambda k: lead_scores[k])
                self.leader_id = top_owner
                my_score = lead_scores.get(self.player, 0)
                top_score = lead_scores.get(top_owner, 0)
                if (
                    top_owner != self.player
                    and my_score > 0
                    and (top_score / my_score) >= LEADER_BASH_RATIO
                ):
                    self.contest_leader = True

        
        
        
        
        self.mode = _detect_mode(self) if PERSONALITY_ENABLED else "patient"
        
        
        
        if TERMINAL_PHASE_ENABLED and self.remaining_steps < TERMINAL_PHASE_TURNS:
            self.mode = "pressure"
        params_table = MODE_PARAMS_2P if self.is_2p else MODE_PARAMS
        self.mode_params = params_table[self.mode]

        
        
        self.stop_expanding_2p = (
            STOP_EXPAND_2P_ENABLED
            and self.is_2p
            and self.step >= STOP_EXPAND_TURN_MIN_2P
            and self.my_prod_share >= STOP_EXPAND_PROD_SHARE_2P
        )

        
        
        
        
        self.in_combat_contact = False
        if COMBAT_STOP_EXPAND_ENABLED:
            my_ids = {p.id for p in self.my_planets}
            enemy_ids = {p.id for p in self.enemy_planets}
            for pid, arrs in self.arrivals_by_planet.items():
                if pid in my_ids:
                    for _eta, owner, ships in arrs:
                        if owner != self.player and owner != -1 and ships >= COMBAT_CONTACT_MIN_SHIPS:
                            self.in_combat_contact = True
                            break
                elif pid in enemy_ids:
                    for _eta, owner, ships in arrs:
                        if owner == self.player and ships >= COMBAT_CONTACT_MIN_SHIPS:
                            self.in_combat_contact = True
                            break
                if self.in_combat_contact:
                    break
        self.combat_stop_expand = (
            COMBAT_STOP_EXPAND_ENABLED
            and self.in_combat_contact
            and self.step >= COMBAT_STOP_EXPAND_TURN_MIN
            and (not COMBAT_STOP_EXPAND_4P_ONLY or not self.is_2p)
        )

        
        
        
        prod_lag_thresh = (
            PROD_LAG_STOP_EXPAND_THRESH_2P if self.is_2p
            else PROD_LAG_STOP_EXPAND_THRESH_4P
        )
        self.prod_lag_stop_expand = (
            PROD_LAG_STOP_EXPAND_ENABLED
            and self.step >= PROD_LAG_STOP_EXPAND_TURN_MIN
            and self.my_prod_share < prod_lag_thresh
        )

        
        self.enemy_tempo_stop_expand = (
            ENEMY_TEMPO_STOP_EXPAND_ENABLED
            and self.step >= ENEMY_TEMPO_STOP_EXPAND_TURN_MIN
            and FLEET_INTENT_ENABLED
            and len(_enemy_recently_launched) >= ENEMY_TEMPO_STOP_EXPAND_MIN_LAUNCHES
        )

        
        self.easy_enemy_stop_expand = False
        if EASY_ENEMY_STOP_EXPAND_ENABLED and self.step >= EASY_ENEMY_STOP_EXPAND_TURN_MIN:
            easy_count = 0
            for ep in self.enemy_planets:
                if int(ep.ships) > EASY_ENEMY_MAX_GARRISON:
                    continue
                for mp in self.my_planets:
                    if dist(mp.x, mp.y, ep.x, ep.y) <= EASY_ENEMY_MAX_DIST:
                        easy_count += 1
                        break
                if easy_count >= EASY_ENEMY_MIN_COUNT:
                    break
            self.easy_enemy_stop_expand = (easy_count >= EASY_ENEMY_MIN_COUNT)

        
        self.stockpile_stop_expand = False
        if STOCKPILE_STOP_EXPAND_ENABLED and self.step >= STOCKPILE_STOP_EXPAND_TURN_MIN:
            for mp in self.my_planets:
                if int(mp.ships) >= STOCKPILE_STOP_EXPAND_MAX_GARRISON:
                    self.stockpile_stop_expand = True
                    break

        
        self.prod_lead_stop_expand_4p = (
            PROD_LEAD_STOP_EXPAND_4P_ENABLED
            and not self.is_2p
            and self.step >= PROD_LEAD_STOP_EXPAND_4P_TURN_MIN
            and self.my_prod_share >= PROD_LEAD_STOP_EXPAND_4P_THRESH
        )

        
        self.turn_cutoff_stop_expand = (
            TURN_CUTOFF_STOP_EXPAND_ENABLED
            and self.step >= TURN_CUTOFF_STOP_EXPAND_TURN
        )

        
        self.neutral_saturation_stop_expand = False
        if (
            NEUTRAL_SATURATION_STOP_EXPAND_ENABLED
            and self.step >= NEUTRAL_SATURATION_TURN_MIN
            and (not NEUTRAL_SATURATION_2P_ONLY or self.is_2p)
        ):
            any_cheap = False
            for n in self.planets:
                if n.owner != -1 or n.id in self.comet_ids:
                    continue
                if int(n.ships) > NEUTRAL_SATURATION_CHEAP_GARRISON:
                    continue
                for mp in self.my_planets:
                    if dist(mp.x, mp.y, n.x, n.y) <= NEUTRAL_SATURATION_REACH_DIST:
                        any_cheap = True
                        break
                if any_cheap:
                    break
            self.neutral_saturation_stop_expand = not any_cheap

        
        
        self.stop_expand_lax = (
            self.combat_stop_expand
            or self.prod_lag_stop_expand
            or self.enemy_tempo_stop_expand
            or self.easy_enemy_stop_expand
            or self.neutral_saturation_stop_expand
            or self.stockpile_stop_expand
        )

        
        
        
        self.focus_enemy_2p = None
        if F14_4A_2P_FOCUS_ENABLED and self.is_2p:
            for o in self.owner_production.keys():
                if o not in (-1, self.player):
                    self.focus_enemy_2p = o
                    break

    @property
    def num_players(self):
        owners = set()
        for p in self.planets:
            if p.owner != -1:
                owners.add(p.owner)
        for f in self.fleets:
            owners.add(f.owner)
        return max(2, len(owners))


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _compute_enemy_race_eta(world):
    """For each neutral, return earliest turn an enemy could land a capturing
    fleet. Considers (a) enemy fleets already in flight aimed at this neutral,
    and (b) enemy planets that have enough ships and are within reach.
    Returns {neutral_id: eta_int}. Neutrals with no credible threat omitted.

    Used to prioritize uncontested-but-soon-to-be-contested neutrals AND to
    skip targets we'd lose the race for (saving ships for next turn).
    """
    out = {}
    if not world.neutral_planets:
        return out

    for n in world.neutral_planets:
        needed = int(n.ships) + 1
        earliest = None

        
        for eta, owner, ships in world.arrivals_by_planet.get(n.id, []):
            if owner == world.player or owner == -1:
                continue
            if ships < needed:
                continue
            if earliest is None or eta < earliest:
                earliest = int(eta)

        
        for ep in world.enemy_planets:
            if int(ep.ships) < needed:
                continue
            d = dist(ep.x, ep.y, n.x, n.y)
            if d > RACE_MAX_NEUTRAL_DIST:
                continue
            
            
            
            
            
            if safe_geometry(ep.x, ep.y, ep.radius, n.x, n.y, n.radius) is None:
                continue
            
            
            
            min_turns = max(1, int(math.ceil(d / fleet_speed(int(ep.ships)))))
            if min_turns > RACE_HORIZON_TURNS:
                continue
            if earliest is None or min_turns < earliest:
                earliest = min_turns

        if earliest is not None:
            out[n.id] = earliest
    return out


def _detect_mode(world):
    """Pick a personality mode from the current snapshot.

    Aggression score = (enemy ships in flight) / (total enemy ships, in flight
    or on planets). A high ratio means enemies are committing to attacks; a
    low ratio means they're stockpiling / quiet. We stay PATIENT during the
    opening since initial expansions look like aggression but aren't.

    V12.2 R2: in 2P, sustained PATIENT with no production-share gain forces
    escalation (10 turns → OPPORTUNISTIC, 20 turns → PRESSURE). This is the
    Bocsimacko "value action over inaction" principle — patient-vs-patient
    1v1 is a stable equilibrium the bot otherwise can't leave.
    """
    if world.is_opening:
        if world.is_2p:
            _record_2p_progress(world.my_prod_share, intended_patient=True, reset=True)
        return "patient"

    enemy_planet_ships = 0
    for p in world.planets:
        if p.owner not in (-1, world.player):
            enemy_planet_ships += int(p.ships)
    enemy_fleet_ships = 0
    for f in world.fleets:
        if f.owner != world.player and f.owner != -1:
            enemy_fleet_ships += int(f.ships)

    enemy_total = enemy_planet_ships + enemy_fleet_ships
    if enemy_total < PERSONALITY_MIN_SAMPLE:
        intended = "patient"
    else:
        aggression = enemy_fleet_ships / float(enemy_total)
        if aggression >= PERSONALITY_AGG_HIGH:
            intended = "pressure"
        elif aggression <= PERSONALITY_AGG_LOW:
            intended = "opportunistic"
        else:
            intended = "patient"

    if not world.is_2p:
        return intended

    
    
    
    _record_2p_progress(world.my_prod_share, intended_patient=(intended == "patient"))
    return "pressure"


def _record_2p_progress(my_prod_share, intended_patient, reset=False):
    """Track production-share trend in 2P. Increment streak whenever the bot
    intends to stay PATIENT and prod-share hasn't grown >EPS over the rolling
    window. Reset streak on opening, on non-PATIENT intent, or on real progress.
    Returns current streak length.
    """
    global _2p_patient_streak, _2p_prod_share_history
    if reset:
        _2p_patient_streak = 0
        _2p_prod_share_history = []
        return 0
    _2p_prod_share_history.append(float(my_prod_share))
    if len(_2p_prod_share_history) > TWO_P_PROD_SHARE_HISTORY:
        _2p_prod_share_history.pop(0)
    if not intended_patient:
        _2p_patient_streak = 0
        return 0
    if len(_2p_prod_share_history) >= TWO_P_PROD_SHARE_HISTORY:
        delta = _2p_prod_share_history[-1] - _2p_prod_share_history[0]
        if delta > TWO_P_PROD_SHARE_PROGRESS_EPS:
            _2p_patient_streak = 0
            return 0
    _2p_patient_streak += 1
    return _2p_patient_streak






_agent_step = 0
_hammer_plan = None             
_planet_idle_counts = {}        
_promoted_stockpiles = set()    
_game_num_players = None        
_2p_patient_streak = 0          
_2p_prod_share_history = []     






_neutral_prev_ships = {}
_neutral_wounded = set()




_enemy_prev_ships = {}
_enemy_recently_launched = set()  



_planet_prev_owner = {}        
_freshly_lost_planets = set()  

_freshly_captured_planets = set()  
_planet_capture_age = {}       






_pending_commitments = []






OPP_PROFILE_WINDOW = 20
_opp_profile = {}


def _update_opp_profile_4p(world):
    """V12.8et: collect rolling per-enemy behavioral signals. 4P-only;
    caller must check world.is_2p first to avoid 2P side effects.
    """
    global _opp_profile
    if world.step == 0:
        _opp_profile = {}

    plan_ships = defaultdict(int)
    plan_max = defaultdict(int)
    plan_count = defaultdict(int)
    for p in world.planets:
        if p.owner == world.player or p.owner == -1:
            continue
        s = int(p.ships)
        plan_ships[p.owner] += s
        plan_count[p.owner] += 1
        if s > plan_max[p.owner]:
            plan_max[p.owner] = s
    fleet_ships = defaultdict(int)
    for f in world.fleets:
        if f.owner == world.player or f.owner == -1:
            continue
        fleet_ships[f.owner] += int(f.ships)

    enemies = set(plan_count.keys()) | set(fleet_ships.keys())
    for owner in enemies:
        ps = plan_ships.get(owner, 0)
        fs = fleet_ships.get(owner, 0)
        total = ps + fs
        emit = (fs / total) if total else 0.0
        prof = _opp_profile.setdefault(owner, {"emit": [], "stock": [], "plan": []})
        prof["emit"].append(emit)
        prof["stock"].append(plan_max.get(owner, 0))
        prof["plan"].append(plan_count.get(owner, 0))
        if len(prof["emit"]) > OPP_PROFILE_WINDOW:
            prof["emit"] = prof["emit"][-OPP_PROFILE_WINDOW:]
            prof["stock"] = prof["stock"][-OPP_PROFILE_WINDOW:]
            prof["plan"] = prof["plan"][-OPP_PROFILE_WINDOW:]

    world.opp_profile = _opp_profile






def predict_defender_at_arrival(world, target, arrival_turn):
    """Owner + ship count on `target` at `arrival_turn` (turns from now), using
    the same combat rules as the env: each turn growth, then resolve arrivals."""
    arrivals = world.arrivals_by_planet.get(target.id, [])
    by_turn = defaultdict(list)
    for eta, owner, ships in arrivals:
        if ships <= 0:
            continue
        by_turn[eta].append((owner, ships))

    owner = target.owner
    garrison = float(target.ships)
    horizon = max(1, int(math.ceil(arrival_turn)))

    for t in range(1, horizon + 1):
        if owner != -1:
            garrison += int(target.production)
        group = by_turn.get(t)
        if group:
            owner, garrison = _resolve_combat(owner, garrison, group)
    return owner, max(0.0, garrison)


def _resolve_combat(owner, garrison, arrivals):
    """Match the env's resolve rule: top-attacker minus second-attacker wins; ties = neutral."""
    by_owner = defaultdict(int)
    for o, s in arrivals:
        by_owner[o] += s
    if not by_owner:
        return owner, max(0.0, garrison)
    sorted_o = sorted(by_owner.items(), key=lambda kv: kv[1], reverse=True)
    top_o, top_s = sorted_o[0]
    if len(sorted_o) > 1 and top_s == sorted_o[1][1]:
        survivor_o, survivor_s = -1, 0
    elif len(sorted_o) > 1:
        survivor_o, survivor_s = top_o, top_s - sorted_o[1][1]
    else:
        survivor_o, survivor_s = top_o, top_s

    if survivor_s <= 0:
        return owner, max(0.0, garrison)
    if owner == survivor_o:
        return owner, garrison + survivor_s
    garrison -= survivor_s
    if garrison < 0:
        return survivor_o, -garrison
    return owner, garrison





FWD_SIM_ENABLED = os.environ.get("V128_FWD_SIM", "1") != "0"
FWD_LOOKAHEAD_HORIZON = 25
FWD_LOOKAHEAD_TOP_K = 6          
FWD_MAX_FLEETS = 80


def _fwd_clone(world):
    planet_ids = []
    planet_owner = {}
    planet_ships = {}
    planet_xy = {}
    planet_radius = {}
    planet_prod = {}
    orbital = {}
    for p in world.planets:
        if p.id in world.comet_ids:
            continue
        planet_ids.append(p.id)
        planet_owner[p.id] = int(p.owner)
        planet_ships[p.id] = float(p.ships)
        planet_xy[p.id] = (float(p.x), float(p.y))
        planet_radius[p.id] = float(p.radius)
        planet_prod[p.id] = int(p.production)
        init = world.initial_by_id.get(p.id)
        if init is not None:
            dx = float(init.x) - CENTER_X
            dy = float(init.y) - CENTER_Y
            r = math.sqrt(dx * dx + dy * dy)
            if r + p.radius < ROTATION_LIMIT:
                orbital[p.id] = (r, math.atan2(dy, dx))
    fleets = []
    next_id = 0
    for f in world.fleets:
        fleets.append([int(f.id), int(f.owner), float(f.x), float(f.y),
                       float(f.angle), int(f.ships)])
        next_id = max(next_id, int(f.id))
    return {
        "planet_ids": planet_ids,
        "planet_owner": planet_owner,
        "planet_ships": planet_ships,
        "planet_xy": planet_xy,
        "planet_radius": planet_radius,
        "planet_prod": planet_prod,
        "orbital": orbital,
        "fleets": fleets,
        "step": int(world.step),
        "ang_vel": float(world.ang_vel),
        "next_fleet_id": next_id + 1,
    }


def _fwd_inject_launch(state, src_id, angle, ships):
    if src_id not in state["planet_xy"]:
        return False
    if state["planet_ships"][src_id] < ships:
        return False
    state["planet_ships"][src_id] -= ships
    radius = state["planet_radius"][src_id]
    sx, sy = state["planet_xy"][src_id]
    fx = sx + math.cos(angle) * (radius + 0.1)
    fy = sy + math.sin(angle) * (radius + 0.1)
    owner = state["planet_owner"][src_id]
    state["fleets"].append([state["next_fleet_id"], int(owner), fx, fy,
                            float(angle), int(ships)])
    state["next_fleet_id"] += 1
    return True


def _fwd_step(state):
    for pid in state["planet_ids"]:
        if state["planet_owner"][pid] != -1:
            state["planet_ships"][pid] += state["planet_prod"][pid]
    combat = {pid: [] for pid in state["planet_ids"]}
    surviving = []
    radii = state["planet_radius"]
    xy = state["planet_xy"]
    pids = state["planet_ids"]
    for fl in state["fleets"]:
        ships = fl[5]
        if ships <= 0:
            continue
        speed = fleet_speed(ships)
        old_x, old_y = fl[2], fl[3]
        new_x = old_x + math.cos(fl[4]) * speed
        new_y = old_y + math.sin(fl[4]) * speed
        fl[2] = new_x
        fl[3] = new_y
        if not (0.0 <= new_x <= BOARD and 0.0 <= new_y <= BOARD):
            continue
        if point_to_segment_distance(CENTER_X, CENTER_Y, old_x, old_y, new_x, new_y) < SUN_R:
            continue
        hit_pid = -1
        for pid in pids:
            px, py = xy[pid]
            if point_to_segment_distance(px, py, old_x, old_y, new_x, new_y) < radii[pid]:
                hit_pid = pid
                break
        if hit_pid >= 0:
            combat[hit_pid].append(fl)
        else:
            surviving.append(fl)
    state["step"] += 1
    new_xy = dict(xy)
    for pid, (r, a0) in state["orbital"].items():
        a = a0 + state["ang_vel"] * state["step"]
        new_xy[pid] = (CENTER_X + r * math.cos(a), CENTER_Y + r * math.sin(a))
    still = []
    for fl in surviving:
        hit_pid = -1
        for pid in pids:
            if pid not in state["orbital"]:
                continue
            old_px, old_py = xy[pid]
            new_px, new_py = new_xy[pid]
            if point_to_segment_distance(fl[2], fl[3], old_px, old_py, new_px, new_py) < radii[pid]:
                hit_pid = pid
                break
        if hit_pid >= 0:
            combat[hit_pid].append(fl)
        else:
            still.append(fl)
    state["planet_xy"] = new_xy
    state["fleets"] = still
    for pid, arrivals in combat.items():
        if not arrivals:
            continue
        per_owner = defaultdict(int)
        for fl in arrivals:
            per_owner[fl[1]] += fl[5]
        sorted_o = sorted(per_owner.items(), key=lambda kv: kv[1], reverse=True)
        top_o, top_s = sorted_o[0]
        if len(sorted_o) > 1:
            second_s = sorted_o[1][1]
            if top_s == second_s:
                surv_s, surv_o = 0, -1
            else:
                surv_s, surv_o = top_s - second_s, top_o
        else:
            surv_o, surv_s = top_o, top_s
        if surv_s > 0:
            cur = state["planet_owner"][pid]
            if cur == surv_o:
                state["planet_ships"][pid] += surv_s
            else:
                state["planet_ships"][pid] -= surv_s
                if state["planet_ships"][pid] < 0:
                    state["planet_owner"][pid] = surv_o
                    state["planet_ships"][pid] = -state["planet_ships"][pid]


def _fwd_simulate(state, horizon):
    for _ in range(horizon):
        if len(state["fleets"]) > FWD_MAX_FLEETS:
            break
        _fwd_step(state)
    return state


def _fwd_my_score(state, player):
    total = 0.0
    for pid in state["planet_ids"]:
        if state["planet_owner"][pid] == player:
            total += state["planet_ships"][pid]
    for fl in state["fleets"]:
        if fl[1] == player:
            total += fl[5]
    return total


def _fwd_marginal(world, src_id, angle, ships, player, horizon):
    """V12.8ay: Δ score (with-launch − without-launch) at horizon."""
    state_no = _fwd_clone(world)
    _fwd_simulate(state_no, horizon)
    base = _fwd_my_score(state_no, player)
    state_yes = _fwd_clone(world)
    if not _fwd_inject_launch(state_yes, src_id, angle, int(ships)):
        return 0.0
    _fwd_simulate(state_yes, horizon)
    return _fwd_my_score(state_yes, player) - base


def _fwd_capture_holds_2p(world, src, target, angle, turns, ships, my_player):
    """V12.8av: simulate launching this fleet now; verify the captured
    target is still ours `turns + FWD_STAB_HORIZON` turns later. Returns
    True if capture sticks, False if predicted to flip."""
    state = _fwd_clone(world)
    if not _fwd_inject_launch(state, src.id, angle, int(ships)):
        return True  
    horizon = int(turns) + 15  
    _fwd_simulate(state, horizon)
    return state["planet_owner"].get(target.id) == my_player






def is_targetable(world, target):
    """Comets travel along non-orbital elliptical paths that aim_at_target can't
    predict. Aiming at them produces fleets that wander and often hit the sun.
    Skip them entirely as expansion / hammer targets.

    V12.9 redundant-launch fix: also skip NEUTRAL targets where one of OUR
    fleets is already in flight with enough ships to flip the planet on
    arrival. Prevents wasted small follow-up fleets piling on a neutral that
    is already being captured.

    V12.9 cap55: enforce the neutral hard cap (2P >=55, 4P legacy) here so
    every targeting code path obeys it — the previous per-call check at
    generate_step_actions/handle_expand missed cheap-pickup, multiprong, and
    other paths."""
    if target.id in world.comet_ids:
        return False
    if target.owner == -1:
        
        
        
        my_arrivals = sorted(
            ((eta, ships) for eta, owner, ships
             in world.arrivals_by_planet.get(target.id, [])
             if owner == world.player),
            key=lambda x: x[0],
        )
        if my_arrivals:
            total_ships = sum(s for _, s in my_arrivals)
            last_eta = my_arrivals[-1][0]
            if total_ships > garrison_at_arrival(target, last_eta):
                return False
        if _neutral_blocked_by_cap(world, target):
            return False
        
        
        
        
        
        
        
        if (LOW_PROD_NEUTRAL_SKIP_ENABLED
                and int(target.production) <= LOW_PROD_NEUTRAL_SKIP_PROD
                and int(target.ships) >= LOW_PROD_NEUTRAL_SKIP_GARRISON):
            return False
    return True


def _update_neutral_watchlist(world):
    """V12.8c: rebuild the wounded-neutral set from this turn's deltas.
    A neutral that lost >= NEUTRAL_WATCHLIST_MIN_DROP ships since last
    turn is considered wounded — someone else attacked it, so it's now
    cheaper for us to take. _neutral_prev_ships is then refreshed.

    V13.3 F1: also track enemy planet ship-drops as 'recently launched'
    signal. A drop > FLEET_INTENT_MIN_DROP indicates the source committed
    a fleet outward; the source is in a brief vulnerable state."""
    _neutral_wounded.clear()
    if NEUTRAL_HARD_CAP_ENABLED:
        for p in world.neutral_planets:
            prev = _neutral_prev_ships.get(p.id)
            cur = int(p.ships)
            if prev is not None and (prev - cur) >= NEUTRAL_WATCHLIST_MIN_DROP:
                _neutral_wounded.add(p.id)
    _neutral_prev_ships.clear()
    for p in world.neutral_planets:
        _neutral_prev_ships[p.id] = int(p.ships)
    
    if FLEET_INTENT_ENABLED:
        _enemy_recently_launched.clear()
        for p in world.enemy_planets:
            prev = _enemy_prev_ships.get(p.id)
            cur = int(p.ships)
            if prev is not None:
                
                
                expected = prev + int(p.production)
                if expected - cur >= FLEET_INTENT_MIN_DROP:
                    _enemy_recently_launched.add(p.id)
        _enemy_prev_ships.clear()
        for p in world.enemy_planets:
            _enemy_prev_ships[p.id] = int(p.ships)
    
    
    
    
    if R1_RECAPTURE_PRIORITY_ENABLED:
        _freshly_lost_planets.clear()
        _freshly_captured_planets.clear()
        for p in world.planets:
            prev_owner = _planet_prev_owner.get(p.id)
            if prev_owner == world.player and p.owner != -1 and p.owner != world.player:
                _freshly_lost_planets.add(p.id)
            
            if (
                FRESH_CAPTURE_INHERITANCE_ENABLED
                and prev_owner is not None
                and prev_owner != world.player
                and p.owner == world.player
            ):
                _freshly_captured_planets.add(p.id)
                _planet_capture_age[p.id] = 0
        
        if FRESH_CAPTURE_INHERITANCE_ENABLED:
            for pid in list(_planet_capture_age.keys()):
                if pid in _freshly_captured_planets:
                    continue
                pp = world.planet_by_id.get(pid)
                if pp is None or pp.owner != world.player:
                    del _planet_capture_age[pid]
                else:
                    _planet_capture_age[pid] += 1
                    if _planet_capture_age[pid] > FRESH_CAPTURE_MAX_AGE:
                        del _planet_capture_age[pid]
        _planet_prev_owner.clear()
        for p in world.planets:
            _planet_prev_owner[p.id] = int(p.owner)


def _neutral_blocked_by_cap(world, target):
    """V12.9 cap55: ignore neutrals with high garrison. V13.3 N4: use
    effective_garrison_at_arrival projection (estimated 10-turn lookahead)
    so a 60-ship neutral about to be hit by enemy 8 → effective 52 → unblocks."""
    if not NEUTRAL_HARD_CAP_ENABLED:
        return False
    if target.owner != -1:
        return False
    
    if NEUTRAL_CAP_USES_EFFECTIVE_GARRISON:
        eff_owner, eff_ships = effective_garrison_at_arrival(target, NEUTRAL_CAP_LOOKAHEAD, world)
        if eff_owner != -1:
            
            return False
        if world.is_2p:
            return eff_ships >= NEUTRAL_HARD_CAP_2P
        if eff_ships <= NEUTRAL_HARD_CAP_4P:
            return False
        return target.id not in _neutral_wounded
    
    if world.is_2p:
        return int(target.ships) >= NEUTRAL_HARD_CAP_2P
    if int(target.ships) <= NEUTRAL_HARD_CAP_4P:
        return False
    return target.id not in _neutral_wounded


def _neutral_tempo_ok(world, target, ships, turns):
    """V12.8cq: skip neutral captures whose expected production gain over
    remaining turns doesn't beat the ship cost by NEUTRAL_TEMPO_THRESHOLD.
    4P-only (2P duels make every neutral worth it). Refuses captures that
    repay slowly even if technically positive (kovi-inspired patience)."""
    if not NEUTRAL_TEMPO_FILTER_ENABLED:
        return True
    if world.is_2p:
        return True
    if target.owner != -1:
        return True
    remaining_after = max(0, int(world.remaining_steps) - int(turns))
    net = float(target.production) * remaining_after - float(ships)
    return net >= NEUTRAL_TEMPO_THRESHOLD


def _ti1_extra_margin(world):
    """V13.3 TI1: returns extra margin to require on captures when we're
    trailing the leader in the late game. Tie counts as a win (engine reward=1
    for max-sum players); low-margin failed attacks drop our absolute sum but
    not our enemies' enough to help. Conserve when behind."""
    if not TI1_TIE_FOR_WIN_ENABLED:
        return 0
    if world.remaining_steps > TI1_HORIZON_TURNS:
        return 0
    my_sum = world.owner_strength.get(world.player, 0)
    leader_sum = my_sum
    for owner, ships in world.owner_strength.items():
        if owner == world.player or owner == -1:
            continue
        if ships > leader_sum:
            leader_sum = ships
    if leader_sum - my_sum < TI1_TRAILING_GAP_MIN:
        return 0  
    return TI1_REQUIRED_EXTRA_MARGIN


def _endgame_roi_ok(world, target, ships, turns):
    """V12.8b: in the last ENDGAME_ROI_TURNS (4P only), refuse neutral captures
    whose expected production growth doesn't repay the ships spent. 4P-only
    because in 2P the differential-value of denying the neutral to the single
    opponent makes marginal late grabs still net-positive at this threshold;
    n=384 test of the un-gated version showed -38 wins 2P, +17pp 4P. Hostile
    targets always allowed. Returns True if firing is OK."""
    if not ENDGAME_ROI_ENABLED:
        return True
    if world.is_2p:
        return True
    if target.owner != -1:
        return True
    if world.step < TOTAL_STEPS - ENDGAME_ROI_TURNS:
        return True
    remaining_after = max(0, int(world.remaining_steps) - int(turns))
    expected_growth = float(target.production) * remaining_after
    
    
    
    
    threshold = float(target.ships) if E2_USE_GARRISON_THRESHOLD else float(ships)
    return expected_growth > threshold


def friendly_already_committed(world, target_id):
    """Patient ethos: ONE main fleet per target — UNLESS the target is enemy
    and our in-flight fleet undershoots its growing garrison.

    Neutrals don't grow, so a correctly-sized fleet wins or loses on arrival;
    a follow-up there is wasted ships (Bocsimacko/zvold canonical rule). For
    enemy targets, the planet grows by its production rate every turn the
    fleet is in flight, so a single source from long range can fail to
    capture; allow a sequenced follow-up only when no single pending fleet
    is sufficient at its own arrival turn.
    """
    target = world.planet_by_id.get(target_id)
    if target is None:
        return False
    pending = [c for c in _pending_commitments if c["target_id"] == target_id]
    if not pending:
        return False
    
    if target.owner == -1 or target.owner == world.player:
        return sum(c["ships"] for c in pending) > 0
    
    
    
    for c in pending:
        eta = int(c["arrival_abs"]) - int(world.step)
        if eta <= 0:
            continue
        if int(c["ships"]) >= needed_to_capture(target, eta):
            return True
    return False


def _commit_fleet(world, moves, spent, target_locked,
                  src_id, target_id, angle, turns, ships):
    """Single point of truth for firing a fleet: appends move, charges spent,
    locks target this turn, and records the persistent commitment so future
    turns know we already engaged this target."""
    moves.append([src_id, float(angle), int(ships)])
    spent[src_id] += int(ships)
    target_locked.add(target_id)
    
    
    target_obj = world.planet_by_id.get(int(target_id))
    owner_at_commit = int(target_obj.owner) if target_obj is not None else -2
    _pending_commitments.append({
        "target_id": int(target_id),
        "ships": int(ships),
        "arrival_abs": int(world.step) + int(turns),
        "owner_at_commit": owner_at_commit,
    })
    if os.environ.get("ORBIT_TRACE"):
        try:
            with open(os.environ["ORBIT_TRACE"], "a") as fh:
                fh.write(
                    f"t={world.step} src={src_id} tgt={target_id} ships={ships} eta={turns}\n"
                )
        except Exception:
            pass


def plan_solo_capture(world, src, tgt, max_avail, max_travel):
    """Plan a single-fleet capture (angle, turns, ships) honoring all the
    fleet-quality rules. Returns None if no viable shot exists.

    Critical: aiming uses fleet_speed(ships), so a different ship count than
    we end up sending produces a wrong angle and the fleet wanders / hits the
    sun. We aim, decide ships, then RE-AIM with the exact ship count.
    """
    
    
    raw_dist = dist(src.x, src.y, tgt.x, tgt.y)
    if F3_THREE_BUCKET_ENABLED:
        if tgt.owner == -1 and raw_dist < F3_SAFE_DIST:
            min_floor = F3_SAFE_FLOOR
        elif (tgt.owner != -1 and tgt.owner != world.player
              and int(tgt.ships) >= F3_HARD_GARRISON):
            min_floor = F3_HARD_FLOOR
        else:
            min_floor = MIN_DISPATCH_SHIPS
    else:
        min_floor = 5 if (world.is_2p and raw_dist < 12.0) else MIN_DISPATCH_SHIPS
    if max_avail < min_floor:
        return None
    aim = aim_at_target(src, tgt, max_avail, world.initial_by_id, world.ang_vel, world=world)
    if aim is None:
        return None
    angle, turns = aim
    if turns > max_travel:
        return None
    need = effective_needed_to_capture(tgt, turns, world)  
    margin = EXPAND_MIN_MARGIN_4P if not world.is_2p else EXPAND_MIN_MARGIN
    
    
    
    
    extra = X8B_2P_EXTRA if world.is_2p else 0
    
    
    
    
    extra += _ti1_extra_margin(world)
    preferred = max(min_floor, need + margin + extra)
    
    
    
    if SP1_SPEED_AWARE_ENABLED:
        raw_dist = dist(src.x, src.y, tgt.x, tgt.y)
        if raw_dist >= SP1_LONG_DIST_THRESHOLD:
            preferred = max(preferred, min(SP1_LONG_DIST_SHIPS, max_avail))
    if preferred <= max_avail:
        ships = preferred
    else:
        ships = max(min_floor, need + margin)
        if ships > max_avail:
            ships = max(min_floor, need)  
    if ships < min_floor or ships > max_avail:
        return None
    aim2 = aim_at_target(src, tgt, ships, world.initial_by_id, world.ang_vel, world=world)
    if aim2 is None:
        return None
    angle, turns = aim2
    if turns > max_travel:
        return None
    need2 = effective_needed_to_capture(tgt, turns, world)  
    if ships < need2 + margin:
        ships = need2 + margin
        if ships > max_avail:
            return None
        aim3 = aim_at_target(src, tgt, ships, world.initial_by_id, world.ang_vel, world=world)
        if aim3 is None:
            return None
        angle, turns = aim3
        if turns > max_travel:
            return None
    
    
    
    
    
    if AS1_ANTI_SECOND_ENABLED and not world.is_2p:
        for eta, owner, e_ships in world.arrivals_by_planet.get(tgt.id, []):
            if int(eta) != int(turns):
                continue
            if owner == world.player or owner == -1:
                continue
            if int(e_ships) >= int(ships):
                return None  
    
    
    
    
    
    if FWD_SIM_FILTER_ENABLED and not world.is_2p and tgt.owner == -1:
        proj = forward_project(
            world,
            our_capture_target=tgt.id,
            our_capture_turn=int(turns),
            our_capture_ships=int(ships),
            horizon=FWD_SIM_HORIZON,
            project_opponent_moves=True,
            opponent_emit_fraction=0.30,
        )
        end_owner, end_ships = proj.get(tgt.id, (-1, 0))
        
        
        
        if end_owner != world.player and end_owner != -1 and end_ships > 5:
            return None
    return angle, turns, int(ships)






def handle_defense(world, rescue_needs, available, spent, target_locked,
                   moves, mode_log):
    """Rescue siblings flagged by absorb. Single source preferred; 2-source
    coalition fallback. Each rescuer respects its own reserve and arrives by
    deadline. Locked rescue targets prevent over-rescue.

    V14.2 (Phase 3.8): preemptive doom-evac. When total incoming enemy
    ships overwhelm garrison+future_production, the planet is definitely
    doomed even with rescue. Skip rescue (which wastes ships) and evac
    directly. User-observed scenario: 40 garrison, 10+49 incoming → solo
    rescue would send a sub-need fleet and still lose; better to evac.
    """
    if not rescue_needs:
        return

    for victim_id, (deficit, deadline, victim) in rescue_needs.items():
        if victim_id in target_locked:
            continue
        need = deficit + DEFENSE_OVERSEND

        
        
        
        
        
        if PREEMPTIVE_DOOM_EVAC_ENABLED and (not PREEMPTIVE_DOOM_EVAC_2P_ONLY or world.is_2p):
            enemy_arrivals = [
                (eta, owner, int(ships)) for eta, owner, ships
                in world.arrivals_by_planet.get(victim_id, [])
                if owner != world.player and owner != -1
            ]
            if world.is_2p or not PREEMPTIVE_EVAC_USE_LARGEST_SINGLE_ENEMY_4P:
                threat_metric = sum(ships for _eta, _owner, ships in enemy_arrivals)
            else:
                
                by_owner = defaultdict(int)
                for _eta, owner, ships in enemy_arrivals:
                    by_owner[owner] += ships
                threat_metric = max(by_owner.values()) if by_owner else 0
            window = deadline if deadline is not None else PREEMPTIVE_EVAC_DEFAULT_WINDOW
            garrison_at_deadline = int(victim.ships) + int(victim.production) * int(window)
            if threat_metric > garrison_at_deadline * PREEMPTIVE_EVAC_DOOM_RATIO:
                if _try_doom_evac(world, victim, available, spent, target_locked, moves, mode_log):
                    continue
                
                

        
        solo = []
        for src in world.my_planets:
            if src.id == victim_id:
                continue
            avail = available[src.id] - spent[src.id]
            if avail < need:
                continue
            aim = aim_at_target(src, victim, avail, world.initial_by_id, world.ang_vel, world=world)
            if aim is None:
                continue
            angle, turns = aim
            if deadline is not None and turns > deadline:
                continue
            solo.append((turns, src.id, src, angle, avail))

        if solo:
            solo.sort()  
            
            
            
            fired_solo = False
            last_fail = None
            for _t, src_id, src, _angle_est, avail in solo:
                send = min(avail, need)
                send = max(send, deficit + 1)
                if send < MIN_DISPATCH_SHIPS:
                    send = MIN_DISPATCH_SHIPS if avail >= MIN_DISPATCH_SHIPS else 0
                if send <= 0:
                    last_fail = "doomed-too-poor"
                    continue
                aim_final = aim_at_target(src, victim, send, world.initial_by_id, world.ang_vel, world=world)
                if aim_final is None:
                    last_fail = "doomed-aim-blocked"
                    continue
                angle, turns = aim_final
                if deadline is not None and turns > deadline:
                    last_fail = "doomed-too-slow"
                    continue
                
                
                if FWD_SIM_DEFENSE_CHECK and not world.is_2p:
                    proj = forward_project(
                        world,
                        our_capture_target=victim_id,
                        our_capture_turn=int(turns),
                        our_capture_ships=int(send),
                        horizon=FWD_SIM_HORIZON,
                        project_opponent_moves=True,
                        opponent_emit_fraction=0.30,
                    )
                    end_owner, _ = proj.get(victim_id, (-1, 0))
                    if end_owner != world.player:
                        last_fail = "fwd-sim-victim-still-lost"
                        continue
                _commit_fleet(world, moves, spent, target_locked,
                              src_id, victim_id, angle, turns, int(send))
                mode_log[victim_id] = "defended-by-solo"
                mode_log[src_id] = "defense"
                fired_solo = True
                break
            if fired_solo:
                continue
            if last_fail is not None:
                mode_log[victim_id] = last_fail
                

        
        if not COALITION_ENABLED:
            
            if _try_doom_evac(world, victim, available, spent, target_locked, moves, mode_log):
                continue
            mode_log[victim_id] = "doomed"
            continue
        coalition = _find_defense_coalition(
            world, victim, deadline, need, available, spent
        )
        if coalition is None:
            
            if _try_doom_evac(world, victim, available, spent, target_locked, moves, mode_log):
                continue
            mode_log[victim_id] = "doomed"
            continue
        for src_id, src, angle, ships, turns in coalition:
            _commit_fleet(world, moves, spent, target_locked,
                          src_id, victim_id, angle, turns, int(ships))
            mode_log[src_id] = "defense-coalition"
        mode_log[victim_id] = "defended-by-coalition"


def _try_doom_evac(world, victim, available, spent, target_locked, moves, mode_log):
    """V14.1b (Phase 3.2 V2): doomed planet evacuation.

    When rescue attempts have failed and the planet is about to flip, send
    its garrison to our highest-production friendly within reach. Preserves
    ships that would otherwise be captured. Returns True if a fleet was
    committed.

    V14.2 (Phase 3.6, Idea 5): attack-fallback. If no friendly destination,
    try sending the garrison to a winnable enemy/neutral target instead of
    letting the ships die with the planet. Prioritizes enemy planets in
    _enemy_recently_launched (they just emptied → weakly defended).
    """
    if not DOOM_EVAC_ENABLED:
        return False
    garrison = available[victim.id] - spent[victim.id]
    if garrison < DOOM_EVAC_MIN_SHIPS:
        return False

    
    
    
    friendly_candidates = []
    for dst in world.my_planets:
        if dst.id == victim.id:
            continue
        aim = aim_at_target(victim, dst, garrison, world.initial_by_id,
                            world.ang_vel, world=world)
        if aim is None:
            continue
        angle, turns = aim
        if turns > DOOM_EVAC_MAX_TRAVEL:
            continue
        
        
        score = int(dst.ships) + int(dst.production) * 5
        friendly_candidates.append((-score, int(turns), dst, angle))
    if friendly_candidates:
        friendly_candidates.sort()
        _score, turns, dst, angle = friendly_candidates[0]
        _commit_fleet(world, moves, spent, target_locked,
                      victim.id, dst.id, angle, turns, int(garrison))
        mode_log[victim.id] = "doom-evac-launched"
        mode_log[dst.id] = "doom-evac-recipient"
        return True

    
    if not DOOM_EVAC_ATTACK_FALLBACK_ENABLED:
        return False
    if DOOM_EVAC_ATTACK_FALLBACK_4P_ONLY and world.is_2p:
        return False
    attack_candidates = []
    for dst in world.planets:
        if dst.id == victim.id or dst.owner == world.player:
            continue
        if dst.id in target_locked:
            continue
        if not is_targetable(world, dst):
            continue
        aim = aim_at_target(victim, dst, garrison, world.initial_by_id,
                            world.ang_vel, world=world)
        if aim is None:
            continue
        angle, turns = aim
        if turns > DOOM_EVAC_MAX_TRAVEL:
            continue
        
        
        
        is_enemy = dst.owner != -1
        prod = int(dst.production) if is_enemy else 0
        arrival_garrison = int(dst.ships) + prod * int(turns)
        required = arrival_garrison + DOOM_EVAC_ATTACK_OVERKILL
        if int(garrison) < required:
            continue
        
        recently_launched_bonus = (
            -DOOM_EVAC_ATTACK_PREFER_LAUNCHED_BONUS
            if (is_enemy and dst.id in _enemy_recently_launched) else 0
        )
        rank = (
            recently_launched_bonus,
            -int(dst.production),
            int(turns),
            int(required),
        )
        attack_candidates.append((rank, dst, angle, turns))
    if not attack_candidates:
        return False
    attack_candidates.sort(key=lambda x: x[0])
    _rank, dst, angle, turns = attack_candidates[0]
    _commit_fleet(world, moves, spent, target_locked,
                  victim.id, dst.id, angle, turns, int(garrison))
    mode_log[victim.id] = "doom-evac-attack"
    mode_log[dst.id] = "doom-evac-attack-target"
    return True


def _find_defense_coalition(world, victim, deadline, need, available, spent):
    """Pick the closest pair of siblings whose combined ships meet `need`, both
    arrive by `deadline`, AND each contributes >= COALITION_MIN_PER_CONTRIBUTOR.
    Re-aims each contributor with its exact ship count.
    Returns [(src_id, src, angle, ships), ...] or None.
    """
    options = []
    for src in world.my_planets:
        if src.id == victim.id:
            continue
        avail = available[src.id] - spent[src.id]
        if avail < COALITION_MIN_PER_CONTRIBUTOR:
            continue
        aim = aim_at_target(src, victim, avail, world.initial_by_id, world.ang_vel, world=world)
        if aim is None:
            continue
        _angle_est, turns = aim
        if deadline is not None and turns > deadline:
            continue
        options.append((turns, src.id, src, avail))

    if len(options) < 2:
        return None
    options.sort()  

    for i in range(len(options)):
        for j in range(i + 1, len(options)):
            t_i, sid_i, s_i, a_i = options[i]
            t_j, sid_j, s_j, a_j = options[j]
            if a_i + a_j < need:
                continue
            ratio = a_i / float(a_i + a_j)
            ship_i = max(COALITION_MIN_PER_CONTRIBUTOR,
                         min(a_i, int(round(need * ratio))))
            ship_j = max(COALITION_MIN_PER_CONTRIBUTOR,
                         min(a_j, need - ship_i))
            while ship_i + ship_j < need:
                if ship_i < a_i:
                    ship_i += 1
                elif ship_j < a_j:
                    ship_j += 1
                else:
                    break
            if (ship_i + ship_j < need
                    or ship_i < COALITION_MIN_PER_CONTRIBUTOR
                    or ship_j < COALITION_MIN_PER_CONTRIBUTOR):
                continue
            
            aim_i = aim_at_target(s_i, victim, ship_i, world.initial_by_id, world.ang_vel, world=world)
            aim_j = aim_at_target(s_j, victim, ship_j, world.initial_by_id, world.ang_vel, world=world)
            if aim_i is None or aim_j is None:
                continue
            ang_i, turns_i = aim_i
            ang_j, turns_j = aim_j
            if (deadline is not None
                    and (turns_i > deadline or turns_j > deadline)):
                continue
            return [
                (sid_i, s_i, ang_i, ship_i, turns_i),
                (sid_j, s_j, ang_j, ship_j, turns_j),
            ]
    return None






COMET_EVAC_REMAINING_TURNS = 3   
COMET_EVAC_MIN_SHIPS = 5          






DOOM_EVAC_ENABLED = True
DOOM_EVAC_MIN_SHIPS = 5           
DOOM_EVAC_MAX_TRAVEL = 40         







DOOM_EVAC_ATTACK_FALLBACK_ENABLED = True
DOOM_EVAC_ATTACK_FALLBACK_4P_ONLY = True  
DOOM_EVAC_ATTACK_OVERKILL = 2     
DOOM_EVAC_ATTACK_PREFER_LAUNCHED_BONUS = 3  






PREEMPTIVE_DOOM_EVAC_ENABLED = True
PREEMPTIVE_DOOM_EVAC_2P_ONLY = False  

PREEMPTIVE_EVAC_DOOM_RATIO = 1.20  
PREEMPTIVE_EVAC_DEFAULT_WINDOW = 15  





PREEMPTIVE_EVAC_USE_LARGEST_SINGLE_ENEMY_4P = True


def handle_comet_evac(world, available, spent, target_locked, moves, mode_log):
    """For each owned comet about to expire, send ALL its ships to the nearest
    non-comet friendly planet (or neutral fallback). Ships left on a comet
    that exits the system are lost permanently — evacuation preserves them.
    """
    if not world.comet_remaining:
        return
    
    
    own_non_comet = [p for p in world.my_planets if p.id not in world.comet_ids]
    if not own_non_comet:
        
        
        own_non_comet = [p for p in world.planets
                         if p.owner == -1 and p.id not in world.comet_ids]
        if not own_non_comet:
            return
    for src in world.my_planets:
        rem = world.comet_remaining.get(src.id)
        if rem is None or rem > COMET_EVAC_REMAINING_TURNS:
            continue
        if src.id in mode_log:
            continue
        avail = max(0, available[src.id] - spent.get(src.id, 0))
        if avail < COMET_EVAC_MIN_SHIPS:
            continue
        
        
        
        
        
        best = None
        best_d = float("inf")
        for dst in own_non_comet:
            if dst.id == src.id:
                continue
            d_now = dist(src.x, src.y, dst.x, dst.y)
            est_turns = max(1, int(math.ceil(d_now / fleet_speed(max(1, int(avail))))))
            dst_px, dst_py = predict_target_position(dst, world, est_turns)
            d = dist(src.x, src.y, dst_px, dst_py)
            if d < best_d:
                best_d = d
                best = dst
        if best is None:
            continue
        aim = aim_at_target(src, best, avail, world.initial_by_id, world.ang_vel, world=world)
        if aim is None:
            continue
        angle, turns = aim
        
        
        if turns >= rem:
            
            
            pass
        _commit_fleet(world, moves, spent, target_locked,
                      src.id, best.id, angle, turns, int(avail))
        mode_log[src.id] = "comet-evac"






def handle_cheap_pickup(world, available, spent, target_locked, moves, mode_log):
    """V12.4d (4P-only): each idle source fires on the cheapest reachable
    low-garrison neutral if it can solo it. Bypasses the K=1 mid-game
    starvation where small free planets sit ignored because the source's
    K=1 nearest is a higher-garrison target. 4P-only — see CHEAP_PICKUP_4P_ONLY.
    """
    if not CHEAP_PICKUP_ENABLED:
        return
    if CHEAP_PICKUP_4P_ONLY and world.is_2p:
        return
    
    
    if LAUNCH_BLACKOUT_ENABLED and world.step >= TOTAL_STEPS - LAUNCH_BLACKOUT_TURNS:
        return
    if world.is_opening:
        max_travel = world.mode_params.get("expand_max_travel_opening", EXPAND_MAX_TRAVEL_OPENING)
    else:
        max_travel = world.mode_params["expand_max_travel_mid"]

    cheap_neutrals = [
        p for p in world.neutral_planets
        if int(p.ships) <= CHEAP_PICKUP_MAX_GARRISON
        and p.id not in target_locked
        and is_targetable(world, p)
    ]
    if not cheap_neutrals:
        return
    
    if CHEAP_PICKUP_MIN_PROD >= 2 and any(int(p.production) >= CHEAP_PICKUP_MIN_PROD for p in cheap_neutrals):
        cheap_neutrals = [p for p in cheap_neutrals if int(p.production) >= CHEAP_PICKUP_MIN_PROD]

    sources = sorted(world.my_planets,
                     key=lambda s: -(available[s.id] - spent[s.id]))
    for src in sources:
        avail = available[src.id] - spent[src.id]
        if avail < MIN_DISPATCH_SHIPS:
            continue
        if mode_log.get(src.id):
            continue
        candidates = []
        for n in cheap_neutrals:
            if n.id in target_locked:
                continue
            if friendly_already_committed(world, n.id):
                continue
            cost = int(n.ships) + 1
            if cost > avail:
                continue
            raw = dist(src.x, src.y, n.x, n.y)
            if raw / MAX_SPEED > max_travel + 4:
                continue
            eff = _effective_target_dist(src, n, world)
            candidates.append((cost, eff, n))
        if not candidates:
            continue
        candidates.sort(key=lambda kv: (kv[0], kv[1]))
        for _cost, _eff, n in candidates:
            plan = plan_solo_capture(world, src, n, avail, max_travel)
            if plan is None:
                continue
            angle, turns, ships = plan
            if RACE_ENABLED:
                enemy_eta = world.enemy_race_eta.get(n.id)
                if enemy_eta is not None and turns > enemy_eta:
                    continue
            if not _capture_holds_against_snipe(world, n, turns, int(ships)):
                continue
            if not _endgame_roi_ok(world, n, int(ships), turns):
                continue
            if not _neutral_tempo_ok(world, n, int(ships), turns):
                continue
            _commit_fleet(world, moves, spent, target_locked,
                          src.id, n.id, angle, turns, int(ships))
            mode_log[src.id] = "cheap-pickup"
            break


def _is_cheap_neutral_pick(world, target):
    """V14.1f (Phase 3.5, Idea 4): cheap-pick predicate for combat-contact gate.

    Returns True if the neutral target has small garrison AND there exists
    one of our planets within COMBAT_CHEAP_DIST. Used to preserve free
    pickups while dropping expensive neutrals during active combat.
    """
    if target.owner != -1:
        return True  
    if int(target.ships) > COMBAT_CHEAP_GARRISON:
        return False
    for mp in world.my_planets:
        if dist(mp.x, mp.y, target.x, target.y) <= COMBAT_CHEAP_DIST:
            return True
    return False


def _handle_search_expand_4p(world, available, spent, target_locked, moves, mode_log):
    """V12.9 Melis search-based expansion (4P only). Generates candidate step
    actions via generate_step_actions, ranks by melis_evaluate gain, commits
    top SEARCH_MAX_ACTIONS_TO_PICK that don't conflict (different targets +
    sources). Returns list of committed source ids so caller can skip them.
    """
    actions = search_step_action(
        world, max_per_source=SEARCH_MAX_PER_SOURCE,
        max_actions_to_eval=12,
        use_depth2=SEARCH_DEPTH2_ENABLED,
    )
    committed_sources = set()
    committed_targets = set()
    for act in actions[:SEARCH_MAX_ACTIONS_TO_PICK * 2]:
        if act["score"] <= 0:
            continue
        src_id = act["source_id"]
        tgt_id = act["target_id"]
        if src_id in committed_sources or tgt_id in committed_targets:
            continue
        if tgt_id in target_locked:
            continue
        
        src_status = mode_log.get(src_id)
        if src_status == "brain-reserved-lead":
            continue
        avail = available[src_id] - spent[src_id]
        if avail < act["ships"]:
            continue
        
        
        
        
        tgt = world.planet_by_id.get(tgt_id)
        
        if (world.stop_expanding_2p or world.prod_lead_stop_expand_4p or world.turn_cutoff_stop_expand) and tgt is not None and tgt.owner == -1:
            continue
        
        if world.stop_expand_lax and tgt is not None and tgt.owner == -1:
            if not _is_cheap_neutral_pick(world, tgt):
                continue
        if tgt is not None and tgt.owner == -1:
            turns_act = int(act["arrival_turn"])
            ships_act = int(act["ships"])
            if not _capture_holds_against_snipe(world, tgt, turns_act, ships_act):
                continue
            if not _endgame_roi_ok(world, tgt, ships_act, turns_act):
                continue
            if not _neutral_tempo_ok(world, tgt, ships_act, turns_act):
                continue
        _commit_fleet(world, moves, spent, target_locked,
                      src_id, tgt_id, act["angle"], act["arrival_turn"], act["ships"])
        mode_log[src_id] = "search-expand"
        committed_sources.add(src_id)
        committed_targets.add(tgt_id)
        if len(committed_sources) >= SEARCH_MAX_ACTIONS_TO_PICK:
            break
    return committed_sources


def handle_expand(world, available, spent, target_locked, moves, mode_log):
    
    if LAUNCH_BLACKOUT_ENABLED and world.step >= TOTAL_STEPS - LAUNCH_BLACKOUT_TURNS:
        return
    
    if (SEARCH_EXPAND_4P_ENABLED and not world.is_2p) or \
       (SEARCH_EXPAND_2P_ENABLED and world.is_2p):
        _handle_search_expand_4p(world, available, spent, target_locked, moves, mode_log)
        
    if world.is_opening:
        
        
        K = world.mode_params.get("expand_k_opening", EXPAND_K_OPENING)
        max_travel = world.mode_params.get("expand_max_travel_opening", EXPAND_MAX_TRAVEL_OPENING)
    else:
        K = world.mode_params["expand_k_mid"]
        max_travel = world.mode_params["expand_max_travel_mid"]

    nonfriendly = [
        p for p in world.planets
        if p.owner != world.player and is_targetable(world, p)
    ]
    
    
    
    if world.stop_expanding_2p or world.prod_lead_stop_expand_4p or world.turn_cutoff_stop_expand:
        nonfriendly = [p for p in nonfriendly if p.owner != -1]
    
    elif world.stop_expand_lax:
        nonfriendly = [
            p for p in nonfriendly
            if p.owner != -1 or _is_cheap_neutral_pick(world, p)
        ]
    if not nonfriendly:
        return

    def frontier_key(src):
        return min(dist(src.x, src.y, t.x, t.y) for t in nonfriendly)

    sources = sorted(world.my_planets, key=frontier_key)

    for src in sources:
        
        avail = _routine_avail(world, src, available[src.id] - spent[src.id])
        if avail < MIN_DISPATCH_SHIPS:
            continue
        
        
        
        status = mode_log.get(src.id)
        if status and status != "cheap-pickup":
            continue  

        candidates = _nearest_targets(src, world, K, max_travel, target_locked)
        fired_solo = False
        for tgt, _approx_dist in candidates:
            if friendly_already_committed(world, tgt.id):
                continue
            plan = plan_solo_capture(world, src, tgt, avail, max_travel)
            if plan is None:
                continue
            angle, turns, ships = plan
            if RACE_ENABLED and tgt.owner == -1:
                enemy_eta = world.enemy_race_eta.get(tgt.id)
                if enemy_eta is not None and turns > enemy_eta:
                    snipe = _plan_counter_snipe(world, src, tgt, avail, max_travel)
                    if snipe is None:
                        continue
                    angle, turns, ships = snipe
            if tgt.owner == -1 and not _capture_holds_against_snipe(world, tgt, turns, int(ships)):
                continue
            if not _endgame_roi_ok(world, tgt, int(ships), turns):
                continue
            if not _neutral_tempo_ok(world, tgt, int(ships), turns):
                continue
            
            
            
            if (
                FWD_SIM_ENABLED
                and world.is_2p
                and tgt.owner != world.player
                and not _fwd_capture_holds_2p(world, src, tgt, angle, turns, int(ships), world.player)
            ):
                continue
            _commit_fleet(world, moves, spent, target_locked,
                          src.id, tgt.id, angle, turns, int(ships))
            mode_log[src.id] = "expand-solo"
            fired_solo = True
            break

        if fired_solo:
            continue
        if not COALITION_ENABLED:
            continue

        coalition_max_travel = max_travel + COALITION_MAX_TRAVEL_BONUS
        for tgt, _ in candidates:
            if tgt.id in target_locked:
                continue
            if COALITION_NEUTRALS_ONLY and tgt.owner != -1:
                continue
            if friendly_already_committed(world, tgt.id):
                continue
            ok = _try_coalition_expand(
                world, src, tgt, coalition_max_travel, available, spent,
                target_locked, moves, mode_log,
            )
            if ok:
                break


def _effective_target_dist(src, tgt, world):
    """V12.4a rotation-aware distance proxy for target prefilter ranking.

    Predicts target position at expected travel time and returns distance
    to that future position. Static planets unchanged. Orbital planets
    rotating toward us get a shorter effective distance (promote);
    rotating away get longer (demote). One-step approximation — cheap;
    real arrival is computed later by aim_at_target inside plan_solo_capture.
    Affects WHICH targets get inspected when K is small, not which fleets fly.
    """
    raw = dist(src.x, src.y, tgt.x, tgt.y)
    if not ROT_AWARE_RANK_ENABLED:
        return raw
    init = world.initial_by_id.get(tgt.id)
    if init is None:
        return raw
    if dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius >= ROTATION_LIMIT:
        return raw
    speed = fleet_speed(50)
    travel = max(1, int(math.ceil(raw / speed)))
    if travel > 60:
        return raw
    px, py = predict_planet_position(tgt, world.initial_by_id, world.ang_vel, travel)
    return dist(src.x, src.y, px, py)


def _counter_snipe_candidates(world, src, max_travel, target_locked):
    """V12.4c: neutrals where a known enemy fleet will capture before us, and
    we can re-flip cheaply on a short follow-up. Returns [(target, raw_dist)]
    sorted by re-flip cost ascending. 2P-only — see COUNTER_SNIPE_2P_ONLY note.
    """
    if not COUNTER_SNIPE_ENABLED:
        return []
    if COUNTER_SNIPE_2P_ONLY and not world.is_2p:
        return []
    out = []
    for n in world.neutral_planets:
        if n.id in target_locked:
            continue
        if not is_targetable(world, n):
            continue
        enemy_eta = None
        enemy_remaining = None
        needed = int(n.ships) + 1
        for eta, owner, ships in world.arrivals_by_planet.get(n.id, []):
            if owner == world.player or owner == -1:
                continue
            if ships < needed:
                continue
            if enemy_eta is None or eta < enemy_eta:
                enemy_eta = int(eta)
                enemy_remaining = ships - int(n.ships)
        if enemy_eta is None:
            continue
        d = dist(src.x, src.y, n.x, n.y)
        speed = fleet_speed(50)
        my_eta_est = max(1, int(math.ceil(d / speed)))
        if my_eta_est > max_travel + 4:
            continue
        delay = my_eta_est - enemy_eta
        if delay < COUNTER_SNIPE_MIN_DELAY or delay > COUNTER_SNIPE_MAX_DELAY:
            continue
        prod = max(0, int(n.production))
        defender_at_my_arrival = max(0, int(enemy_remaining)) + prod * delay
        flip_cost = defender_at_my_arrival + 1
        if flip_cost > COUNTER_SNIPE_MAX_COST:
            continue
        out.append((flip_cost, n, d))
    out.sort(key=lambda kv: kv[0])
    return [(n, d) for _cost, n, d in out]


def _plan_counter_snipe(world, src, tgt, max_avail, max_travel):
    """V12.4c: size a small fleet to re-flip a neutral AFTER a known enemy
    fleet captures it. Returns (angle, turns, ships) or None. 2P-only.
    """
    if not COUNTER_SNIPE_ENABLED or tgt.owner != -1:
        return None
    if COUNTER_SNIPE_2P_ONLY and not world.is_2p:
        return None
    if max_avail < MIN_DISPATCH_SHIPS:
        return None
    enemy_eta = None
    enemy_remaining = None
    needed_to_take = int(tgt.ships) + 1
    for eta, owner, ships in world.arrivals_by_planet.get(tgt.id, []):
        if owner == world.player or owner == -1:
            continue
        if ships < needed_to_take:
            continue
        if enemy_eta is None or eta < enemy_eta:
            enemy_eta = int(eta)
            enemy_remaining = ships - int(tgt.ships)
    if enemy_eta is None:
        return None

    aim = aim_at_target(src, tgt, max_avail, world.initial_by_id, world.ang_vel, world=world)
    if aim is None:
        return None
    angle, turns = aim
    if turns > max_travel:
        return None
    delay = turns - enemy_eta
    if delay < COUNTER_SNIPE_MIN_DELAY or delay > COUNTER_SNIPE_MAX_DELAY:
        return None
    prod = max(0, int(tgt.production))
    defender = max(0, int(enemy_remaining)) + prod * delay
    ships = max(MIN_DISPATCH_SHIPS, defender + 1)
    if ships > max_avail or ships > COUNTER_SNIPE_MAX_COST:
        return None
    aim2 = aim_at_target(src, tgt, ships, world.initial_by_id, world.ang_vel, world=world)
    if aim2 is None:
        return None
    angle, turns = aim2
    if turns > max_travel:
        return None
    delay2 = turns - enemy_eta
    if delay2 < COUNTER_SNIPE_MIN_DELAY or delay2 > COUNTER_SNIPE_MAX_DELAY:
        return None
    defender2 = max(0, int(enemy_remaining)) + prod * delay2
    if ships < defender2 + 1:
        ships = defender2 + 1
        if ships > max_avail or ships > COUNTER_SNIPE_MAX_COST:
            return None
        aim3 = aim_at_target(src, tgt, ships, world.initial_by_id, world.ang_vel, world=world)
        if aim3 is None:
            return None
        angle, turns = aim3
        if turns > max_travel:
            return None
    return angle, turns, int(ships)


def _capture_holds_against_snipe(world, target, arrival_turn, ships_sent):
    """V12.4b: returns True if our post-capture garrison stays >0 through every
    KNOWN enemy fleet arriving within ANTI_SNIPE_HORIZON. Walks surplus +
    production growth between events; subtracts each enemy fleet at its eta;
    refuses if balance ever drops <=0. Friendly follow-ups credited.

    Gated to 2P only (ANTI_SNIPE_2P_ONLY): in 4P with 3 enemies the veto
    fires too often, starving expansion (192-game test: 55 third-place
    finishes vs 12_4a's 4). 2P has only one snipe source so the veto
    targets actual snipe traps without paralyzing expansion.
    """
    if not ANTI_SNIPE_ENABLED:
        return True
    if ANTI_SNIPE_2P_ONLY and not world.is_2p:
        return True
    if target.owner != -1:
        return True
    arrivals = world.arrivals_by_planet.get(target.id, [])
    enemy_after = []
    friendly_after = []
    for eta, owner, ships in arrivals:
        if ships <= 0:
            continue
        if eta <= arrival_turn:
            continue
        if eta - arrival_turn > ANTI_SNIPE_HORIZON:
            continue
        if owner == world.player:
            friendly_after.append((eta, ships))
        elif owner != -1:
            enemy_after.append((eta, ships))

    
    
    
    
    
    if REACTIVE_SNIPE_PROJECTION_ENABLED:
        for enemy_p in world.enemy_planets:
            e_ships = int(enemy_p.ships)
            if e_ships < REACTIVE_MIN_ENEMY_SHIPS:
                continue
            
            
            
            
            
            if SUN_SHADOW_REACTIVE_FILTER and not world.is_2p and segment_hits_sun(
                enemy_p.x, enemy_p.y, target.x, target.y
            ):
                continue
            d = dist(enemy_p.x, enemy_p.y, target.x, target.y)
            projected_force = max(REACTIVE_MIN_PROJECTED, int(e_ships * REACTIVE_EMIT_FRAC))
            speed = fleet_speed(projected_force)
            travel = max(1, int(math.ceil(d / speed)))
            
            snipe_eta = travel
            if snipe_eta <= arrival_turn:
                continue  
            if snipe_eta - arrival_turn > ANTI_SNIPE_HORIZON:
                continue
            enemy_after.append((snipe_eta, projected_force))

    if not enemy_after:
        return True

    
    if N6_USE_EFFECTIVE_PRE_GARRISON:
        _, pre_garrison = effective_garrison_at_arrival(target, arrival_turn, world)
    else:
        pre_garrison = garrison_at_arrival(target, arrival_turn)
    if ships_sent <= pre_garrison:
        return True
    surplus = ships_sent - pre_garrison
    prod = max(0, int(target.production))
    by_turn = defaultdict(int)
    for eta, ships in enemy_after:
        by_turn[eta] -= ships
    for eta, ships in friendly_after:
        by_turn[eta] += ships

    bal = surplus
    last_t = arrival_turn
    for eta in sorted(by_turn):
        bal += prod * (eta - last_t)
        bal += by_turn[eta]
        if bal <= 0:
            return False
        last_t = eta
    return True


def _tiebreak_hash(world, src_id, target_id):
    """Deterministic, replayable hash for breaking near-equal-distance ties.
    Salts on (player, step, src, target) so different turns / sources don't
    produce identical perturbations. Multiplicative mix instead of Python's
    hash() because PYTHONHASHSEED randomizes hash() across processes."""
    h = (int(world.player) * 2654435761) & 0xFFFFFFFF
    h ^= (int(world.step) * 1664525) & 0xFFFFFFFF
    h ^= (int(src_id) * 16777619) & 0xFFFFFFFF
    h ^= (int(target_id) * 2246822519) & 0xFFFFFFFF
    return h & 0xFFFF


def _nearest_targets(src, world, K, max_travel, target_locked):
    """Top-K nearest non-friendly, non-comet planets, plus any race-winnable
    contested neutrals appended at the FRONT regardless of K (V12.1a).

    Final travel-time and capture cost happen inside plan_solo_capture; the
    race-loss skip in handle_expand vetoes any target where we'd arrive after
    the enemy.

    V12.3c5 (2.5): in 2P, near-equal-distance candidates (within
    TIEBREAK_EPS_FRAC of best) are reordered by a deterministic
    (player, step, src, target) hash. Cracks symmetric-Nash mirror lock
    where two PATIENT bots otherwise pick the same target deterministically.
    Replayable via hash construction.
    """
    
    
    
    _f31_has_better = (
        world.is_2p
        and EXPAND_MIN_PROD_2P >= 2
        and any(int(n.production) >= EXPAND_MIN_PROD_2P for n in world.neutral_planets
                if n.id not in target_locked)
    )
    candidates = []
    for t in world.planets:
        if t.owner == world.player:
            continue
        if t.id in target_locked:
            continue
        if not is_targetable(world, t):
            continue
        if _neutral_blocked_by_cap(world, t):
            continue
        
        if _f31_has_better and t.owner == -1 and int(t.production) < EXPAND_MIN_PROD_2P:
            continue
        
        
        
        raw = dist(src.x, src.y, t.x, t.y)
        if raw / MAX_SPEED > max_travel + 4:
            continue
        eff = _effective_target_dist(src, t, world)
        
        
        weight = VALUE_WEIGHT_2P if world.is_2p else VALUE_WEIGHT_4P
        weighted = eff - max(0, int(t.production)) * weight
        
        
        
        if F1B_EXPAND_BONUS_ENABLED and t.owner != world.player and t.owner != -1:
            if t.id in _enemy_recently_launched:
                weighted -= F1B_EXPAND_BONUS
        
        
        
        if SO1_STATIC_PREFERENCE_ENABLED:
            init_t = world.initial_by_id.get(t.id)
            if init_t is not None:
                r_t = dist(init_t.x, init_t.y, CENTER_X, CENTER_Y)
                if r_t + init_t.radius >= ROTATION_LIMIT:
                    weighted -= SO1_STATIC_BONUS
        
        
        if (
            LEADER_BASH_ENABLED
            and not world.is_2p
            and world.contest_leader
            and world.step >= LEADER_BASH_MIN_STEP
            and world.leader_id is not None
            and t.owner == world.leader_id
        ):
            weighted -= LEADER_BASH_BONUS
        
        
        
        
        
        if (
            not world.is_2p
            and t.owner != -1
            and t.owner != world.player
            and world.opp_profile
            and t.owner in world.opp_profile
        ):
            prof = world.opp_profile[t.owner]
            if len(prof["emit"]) >= 5:
                avg_emit = sum(prof["emit"]) / len(prof["emit"])
                if avg_emit > 0.35:
                    weighted -= 5.0  
        
        
        
        if (
            WEAKEST_TARGET_ENABLED
            and not world.is_2p
            and world.step >= WEAKEST_TARGET_MIN_STEP
            and world.mode == "pressure"
            and world.weakest_enemy is not None
            and t.owner == world.weakest_enemy
        ):
            if world.weakest_enemy_prod_share < WEAKEST_DONT_FINISH_SHARE:
                weighted += WEAKEST_DONT_FINISH_PENALTY
            else:
                weighted -= WEAKEST_TARGET_BONUS
        
        if (
            F14_4A_2P_FOCUS_ENABLED
            and world.is_2p
            and world.focus_enemy_2p is not None
            and t.owner == world.focus_enemy_2p
        ):
            weighted -= F14_4A_2P_FOCUS_DIST_BONUS
        candidates.append((t, weighted, raw))
    if not candidates:
        return []
    candidates.sort(key=lambda kv: kv[1])
    
    
    
    
    if (FWD_SIM_RANK_BONUS_4P > 0 and not world.is_2p and len(candidates) > 1):
        baseline_proj = forward_project(
            world, horizon=FWD_SIM_HORIZON,
            project_opponent_moves=True, opponent_emit_fraction=0.30
        )
        baseline_score = forward_score(baseline_proj, world.player, 4, world)
        rerank = []
        topN = min(K + 2, len(candidates))
        for idx, (t, w, raw) in enumerate(candidates[:topN]):
            est_eta = max(1, int(math.ceil(raw / MAX_SPEED)))
            est_ships = needed_to_capture(t, est_eta) + 1
            proj = forward_project(
                world, our_capture_target=t.id, our_capture_turn=est_eta,
                our_capture_ships=est_ships, horizon=FWD_SIM_HORIZON,
                project_opponent_moves=True, opponent_emit_fraction=0.30
            )
            score_gain = forward_score(proj, world.player, 4, world) - baseline_score
            adjusted = w - FWD_SIM_RANK_BONUS_4P * score_gain
            rerank.append((t, adjusted, raw))
        candidates = rerank + candidates[topN:]
        candidates.sort(key=lambda kv: kv[1])
    if world.is_2p and TIEBREAK_ENABLED and len(candidates) > 1:
        best_d = candidates[0][1]
        eps = max(TIEBREAK_EPS_MIN, TIEBREAK_EPS_FRAC * best_d)
        def _k(kv):
            tgt, weighted_d, _raw = kv
            bucket = int(weighted_d / eps) if eps > 0 else 0
            return (bucket, _tiebreak_hash(world, src.id, tgt.id), weighted_d)
        candidates.sort(key=_k)

    counter_snipe = _counter_snipe_candidates(world, src, max_travel, target_locked)

    if not RACE_ENABLED or not world.enemy_race_eta:
        head = counter_snipe + [(t, raw) for t, _eff, raw in candidates[:K]]
        return _dedupe_targets(head)

    race_priority = []
    normal = []
    for t, _eff, raw in candidates:
        enemy_eta = world.enemy_race_eta.get(t.id)
        if enemy_eta is None or t.owner != -1:
            normal.append((t, raw))
            continue
        my_min = max(1, int(math.ceil(raw / fleet_speed(max(1, int(src.ships))))))
        if my_min <= enemy_eta:
            race_priority.append((t, raw))
        else:
            normal.append((t, raw))

    return _dedupe_targets(counter_snipe + race_priority + normal[:K])


def _dedupe_targets(seq):
    """V12.4c: preserve order, drop duplicates by target id (counter-snipe and
    race-priority can overlap with the K window)."""
    seen = set()
    out = []
    for tgt, d in seq:
        if tgt.id in seen:
            continue
        seen.add(tgt.id)
        out.append((tgt, d))
    return out


def _aim_partner(world, partner, tgt, ships, max_travel):
    """Aim a coalition partner with EXACT `ships` count. Returns (angle, turns) or None."""
    if ships < COALITION_MIN_PER_CONTRIBUTOR:
        return None
    aim = aim_at_target(partner, tgt, ships, world.initial_by_id, world.ang_vel, world=world)
    if aim is None:
        return None
    angle, turns = aim
    if turns > max_travel:
        return None
    return angle, turns


def _try_coalition_expand(world, src, tgt, max_travel, available, spent,
                          target_locked, moves, mode_log):
    """src can't take tgt alone; find a partner whose combined ships flip it.
    Each contributor must send >= COALITION_MIN_PER_CONTRIBUTOR (no tiny
    pieces). For tiny targets we DON'T split — the patient ethos prefers
    waiting for a solo fleet over showering a small target with two halves.
    """
    src_avail = available[src.id] - spent[src.id]
    if src_avail < COALITION_MIN_PER_CONTRIBUTOR:
        return False
    
    
    if int(tgt.ships) < COALITION_MIN_TARGET_SHIPS:
        return False

    
    
    
    
    partners = []
    for p in world.my_planets:
        if p.id == src.id:
            continue
        avail = available[p.id] - spent[p.id]
        if avail < COALITION_MIN_PER_CONTRIBUTOR:
            continue
        
        est = aim_at_target(p, tgt, avail, world.initial_by_id, world.ang_vel, world=world)
        if est is None:
            continue
        _, est_turns = est
        if est_turns > max_travel:
            continue
        partners.append((est_turns, p, avail))
    if not partners:
        return False
    partners.sort(key=lambda kv: kv[0])

    
    for est_turns, p, p_avail in partners:
        combined = src_avail + p_avail
        
        
        
        
        
        
        
        est_src = aim_at_target(src, tgt, src_avail, world.initial_by_id, world.ang_vel, world=world)
        if est_src is None:
            continue
        worst = max(est_src[1], est_turns)
        total_needed = needed_to_capture(tgt, worst)
        if combined < total_needed:
            continue

        
        ratio = src_avail / float(combined)
        s_src = max(COALITION_MIN_PER_CONTRIBUTOR,
                    min(src_avail, int(round(total_needed * ratio))))
        s_p = max(COALITION_MIN_PER_CONTRIBUTOR,
                  min(p_avail, total_needed - s_src))
        
        while s_src + s_p < total_needed:
            if s_src < src_avail:
                s_src += 1
            elif s_p < p_avail:
                s_p += 1
            else:
                break
        if s_src + s_p < total_needed:
            continue
        if s_src < COALITION_MIN_PER_CONTRIBUTOR or s_p < COALITION_MIN_PER_CONTRIBUTOR:
            continue
        if s_src > src_avail or s_p > p_avail:
            continue

        
        aim_src = aim_at_target(src, tgt, s_src, world.initial_by_id, world.ang_vel, world=world)
        aim_p = aim_at_target(p, tgt, s_p, world.initial_by_id, world.ang_vel, world=world)
        if aim_src is None or aim_p is None:
            continue
        a_src, t_src = aim_src
        a_p, t_p = aim_p
        if t_src > max_travel or t_p > max_travel:
            continue

        
        
        
        
        
        
        
        
        
        
        if world.is_2p and abs(t_src - t_p) > 1:
            continue

        
        
        
        post_eta = max(t_src, t_p)
        post_needed = needed_to_capture(tgt, post_eta)
        if s_src + s_p < post_needed:
            continue

        _commit_fleet(world, moves, spent, target_locked,
                      src.id, tgt.id, a_src, t_src, int(s_src))
        _commit_fleet(world, moves, spent, target_locked,
                      p.id, tgt.id, a_p, t_p, int(s_p))
        mode_log[src.id] = "expand-coalition"
        mode_log[p.id] = "expand-coalition"
        return True

    return False






def _routine_avail(world, planet, base_avail):
    """V14.1d iter g: production-tier reserve. Subtract a fraction of high-prod
    planet garrison from routine expand/hammer spending. The reserve grows
    naturally via production and is available to mega-hammer.
    """
    if not PROD_RESERVE_ENABLED:
        return base_avail
    if PROD_RESERVE_4P_ONLY and world.is_2p:
        return base_avail
    if world.step < PROD_RESERVE_TURN_MIN:
        return base_avail
    if int(planet.production) < PROD_RESERVE_MIN_PROD:
        return base_avail
    reserve = int(int(planet.ships) * PROD_RESERVE_FRAC)
    return max(0, base_avail - reserve)


def _brain_pick_lead(world, available, spent, mode_log, min_ships=None):
    """Shared lead-picker used by both _brain_reserve_lead (pre-pass) and
    handle_accumulator (post-defense). Returns Planet or None.

    Identical logic to handle_accumulator's original lead-selection so the
    reservation and the actual feeder-target agree. min_ships defaults to
    the accumulator's threshold; the brain pre-pass passes a higher value.

    B3b: when BRAIN_LEAD_PREFER_FRONTIER, score = avail - frontier_dist*weight
    so a frontier planet beats a deep-back-corner one even if the back has
    slightly more ships — a closer lead delivers strikes faster.
    """
    if min_ships is None:
        min_ships = ACCUMULATOR_LEAD_MIN_SHIPS
    enemies = world.enemy_planets
    candidates = []
    for p in world.my_planets:
        status = mode_log.get(p.id)
        
        if status and status != "brain-reserved-lead":
            continue
        avail = available[p.id] - spent[p.id]
        if avail < min_ships:
            continue
        threat = sum(int(ships) for eta, owner, ships
                     in world.arrivals_by_planet.get(p.id, [])
                     if owner != world.player and owner != -1)
        if threat >= avail * ACCUMULATOR_LEAD_THREAT_RATIO:
            continue
        if BRAIN_LEAD_PREFER_FRONTIER and enemies:
            frontier_dist = min(dist(p.x, p.y, e.x, e.y) for e in enemies)
            score = float(avail) - frontier_dist * BRAIN_LEAD_FRONTIER_WEIGHT
        else:
            score = float(avail)
        candidates.append((score, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _brain_reserve_lead(world, available, spent, mode_log):
    """B1 (one-brain pre-pass): mark the future accumulator-lead with a
    sentinel so handle_expand can't drain it into small-ship pickups before
    handle_accumulator / handle_mega_hammer run.

    Runs after defense (which doesn't gate on mode_log status of sources)
    and before the expand → accumulator → mega-hammer chain. If defense
    later commits the same planet, defense overwrites mode_log[p.id] = 'defense'
    and the chain naturally skips it — life beats lead."""
    if not BRAIN_LEAD_RESERVE_ENABLED:
        return
    if not ACCUMULATOR_ENABLED:
        return
    if BRAIN_LEAD_RESERVE_4P_ONLY and world.is_2p:
        return
    if ACCUMULATOR_4P_ONLY and world.is_2p:
        return
    if world.step < ACCUMULATOR_TURN_MIN:
        return
    lead = _brain_pick_lead(world, available, spent, mode_log,
                            min_ships=BRAIN_LEAD_RESERVE_MIN_SHIPS)
    if lead is None:
        return
    
    
    if BRAIN_LEAD_RESERVE_REQUIRE_TARGET:
        has_target = False
        for tgt in world.enemy_planets:
            if int(tgt.ships) > MEGA_HAMMER_TARGET_GARRISON_MAX_ITER_H:
                continue
            aim = aim_at_target(lead, tgt, available[lead.id] - spent[lead.id],
                                world.initial_by_id, world.ang_vel, world=world)
            if aim is None:
                continue
            _, turns = aim
            if turns > MEGA_HAMMER_MAX_TRAVEL:
                continue
            has_target = True
            break
        if not has_target:
            return
    mode_log[lead.id] = "brain-reserved-lead"


def handle_accumulator(world, available, spent, target_locked, moves, mode_log):
    """V14.2 (Phase 3.7, Idea 6c): accumulator — feed surplus from safe
    backline planets to the lead stockpile each turn.

    Engine: fleet speed = 1 + 5×(log(ships)/log(1000))^1.5. One big fleet
    (1000 ships, speed 6) arrives faster AND survives tied-combat better
    than 4 fleets of 250 ships. Concentration > spread.

    Strategy: each turn, identify our planet with the most ships ("lead").
    For other planets in the safe backline (no incoming enemy threat AND
    surplus above reserve), send their surplus TO the lead. Over multiple
    turns, the lead accumulates a massive stockpile and handle_mega_hammer
    fires it as one overwhelming strike.

    Runs BEFORE handle_mega_hammer so accumulated ships are visible to
    mega-hammer this turn (but in-flight feeds arrive on later turns).
    """
    if not ACCUMULATOR_ENABLED:
        return
    if ACCUMULATOR_4P_ONLY and world.is_2p:
        return
    if world.step < ACCUMULATOR_TURN_MIN:
        return

    
    
    
    
    
    lead_candidates = []
    for p in world.my_planets:
        status = mode_log.get(p.id)
        if status and status != "brain-reserved-lead":
            continue
        avail = available[p.id] - spent[p.id]
        if avail < ACCUMULATOR_LEAD_MIN_SHIPS:
            continue
        
        threat = sum(int(ships) for eta, owner, ships
                     in world.arrivals_by_planet.get(p.id, [])
                     if owner != world.player and owner != -1)
        if threat >= avail * ACCUMULATOR_LEAD_THREAT_RATIO:
            continue
        lead_candidates.append((avail, p))
    if not lead_candidates:
        return
    lead_candidates.sort(key=lambda x: -x[0])
    lead_avail, lead = lead_candidates[0]

    
    feeders = []
    for p in world.my_planets:
        if p.id == lead.id or p.id in mode_log:
            continue
        
        threat = sum(int(ships) for eta, owner, ships
                     in world.arrivals_by_planet.get(p.id, [])
                     if owner != world.player and owner != -1)
        if threat > 0:
            continue
        avail = available[p.id] - spent[p.id]
        surplus = avail - ACCUMULATOR_FEEDER_KEEP_RESERVE
        if surplus < ACCUMULATOR_FEEDER_MIN_SURPLUS:
            continue
        
        aim = aim_at_target(p, lead, surplus, world.initial_by_id,
                            world.ang_vel, world=world)
        if aim is None:
            continue
        angle, turns = aim
        if turns > ACCUMULATOR_FEEDER_MAX_TRAVEL:
            continue
        feeders.append((turns, surplus, p, angle))

    if not feeders:
        return
    
    feeders.sort(key=lambda x: (x[0], -x[1]))
    fed_count = 0
    for turns, surplus, src, angle in feeders:
        if fed_count >= ACCUMULATOR_MAX_FEEDS_PER_TURN:
            break
        _commit_fleet(world, moves, spent, target_locked,
                      src.id, lead.id, angle, turns, int(surplus))
        mode_log[src.id] = "accumulator-feeder"
        fed_count += 1
    if fed_count > 0:
        
        if lead.id not in mode_log:
            mode_log[lead.id] = "accumulator-lead"


def handle_mega_hammer(world, available, spent, target_locked, moves, mode_log):
    """V14.1c (Phase 3.3): single-source overwhelming strike.

    For each of our planets with avail >= MEGA_HAMMER_SHIPS_MIN, find an
    enemy target whose garrison (after projected arrivals) is <=
    MEGA_HAMMER_TARGET_GARRISON_MAX and is within MEGA_HAMMER_MAX_TRAVEL
    turns. Launch the ENTIRE garrison as a single huge fleet — exploits
    the fleet-speed log formula (bigger = faster) and overwhelms
    reactive defense.

    Runs BEFORE handle_hammer so a successful mega strike isn't dissolved
    into the multi-stockpile coalition logic.
    """
    if not MEGA_HAMMER_ENABLED:
        return
    if MEGA_HAMMER_4P_ONLY and world.is_2p:
        return
    
    
    sources = sorted(world.my_planets,
                     key=lambda p: -(available[p.id] - spent[p.id]))
    fired_targets = set()
    fired_count = 0
    for src in sources:
        
        if MEGA_HAMMER_CONCENTRATE_ENABLED and fired_count >= MEGA_HAMMER_MAX_PER_TURN:
            break
        avail = available[src.id] - spent[src.id]
        
        
        
        prod = int(src.production)
        if FRESH_CAPTURE_INHERITANCE_ENABLED and src.id in _planet_capture_age:
            threshold = MEGA_HAMMER_SHIPS_MIN_FRESH
        else:
            threshold = MEGA_HAMMER_THRESHOLD_BY_PROD.get(prod, MEGA_HAMMER_SHIPS_MIN)
        if avail < threshold:
            continue  
        
        
        status = mode_log.get(src.id)
        if status and status not in ("cheap-pickup", "brain-reserved-lead"):
            continue
        
        best = None
        for tgt in world.enemy_planets:
            if tgt.id in target_locked or tgt.id in fired_targets:
                continue
            if int(tgt.ships) > MEGA_HAMMER_TARGET_GARRISON_MAX_ITER_H:
                continue
            aim = aim_at_target(src, tgt, avail, world.initial_by_id,
                                world.ang_vel, world=world)
            if aim is None:
                continue
            angle, turns = aim
            if turns > MEGA_HAMMER_MAX_TRAVEL:
                continue
            
            
            focus_bonus = 0
            if (F14_4A_2P_FOCUS_ENABLED and world.is_2p
                    and getattr(world, "focus_enemy_2p", None) is not None
                    and tgt.owner == world.focus_enemy_2p):
                focus_bonus = F14_4A_2P_FOCUS_MEGA_BONUS
            score = (int(tgt.production) + focus_bonus, -int(turns))
            if best is None or score > best[0]:
                best = (score, tgt, angle, turns)
        if best is None:
            continue
        _, tgt, angle, turns = best
        
        
        
        
        if MEGA_HAMMER_MELIS_VERIFY and turns > 0:
            proj = forward_project(
                world,
                our_capture_target=tgt.id,
                our_capture_turn=int(turns),
                our_capture_ships=int(avail),
                horizon=FWD_SIM_HORIZON + int(turns),
                project_opponent_moves=True,
                opponent_emit_fraction=MEGA_HAMMER_VERIFY_OPP_EMIT,
            )
            end_owner, _ = proj.get(tgt.id, (-1, 0))
            if end_owner != world.player:
                continue
        _commit_fleet(world, moves, spent, target_locked,
                      src.id, tgt.id, angle, turns, int(avail))
        mode_log[src.id] = "mega-hammer-launched"
        mode_log[tgt.id] = "mega-hammer-target"
        fired_targets.add(tgt.id)
        fired_count += 1


def handle_hammer(world, available, spent, target_locked, moves, mode_log):
    """One persistent plan at a time. Plan picks a strong-production enemy
    target and a set of stockpiles whose combined fleet arriving simultaneously
    beats defender_at_arrival × overkill. Launches stagger so all fleets land
    on the same turn. Plan aborts if defender reinforces past committed strength.
    """
    global _hammer_plan
    if not HAMMER_ENABLED:
        return
    if not world.enemy_planets:
        _hammer_plan = None
        return

    if _hammer_plan is not None:
        
        target = world.planet_by_id.get(_hammer_plan["target_id"])
        if target is None or target.owner == world.player:
            _hammer_plan = None
        else:
            
            arrival_rel = _hammer_plan["target_arrival_abs"] - world.step
            if arrival_rel <= 0:
                _hammer_plan = None
            else:
                d_owner, d_ships = predict_defender_at_arrival(world, target, arrival_rel)
                if d_ships > _hammer_plan["committed_strength"] / HAMMER_ABORT_OVERRUN_RATIO:
                    _hammer_plan = None

    if _hammer_plan is None:
        
        if not _hammer_should_fire(world):
            return
        plan = _build_hammer_plan(world, available, spent)
        if plan is None:
            return
        
        
        
        if HAMMER_MELIS_VERIFY:
            target = world.planet_by_id.get(plan["target_id"])
            if target is not None:
                arrival_rel = plan["target_arrival_abs"] - world.step
                if arrival_rel > 0:
                    proj = forward_project(
                        world,
                        our_capture_target=plan["target_id"],
                        our_capture_turn=int(arrival_rel),
                        our_capture_ships=int(plan["committed_strength"]),
                        horizon=FWD_SIM_HORIZON + arrival_rel,
                        project_opponent_moves=True,
                        opponent_emit_fraction=0.30,
                    )
                    end_owner, _ = proj.get(plan["target_id"], (-1, 0))
                    if end_owner != world.player:
                        return  
        _hammer_plan = plan

    
    plan = _hammer_plan
    completed_launches = []
    for src_id, launch in list(plan["launches"].items()):
        if launch.get("fired"):
            continue
        if launch["fire_turn_abs"] > world.step:
            continue  
        src = world.planet_by_id.get(src_id)
        if src is None or src.owner != world.player:
            completed_launches.append(src_id)
            continue
        ships = launch["ships"]
        if ships < HAMMER_MIN_PER_CONTRIBUTOR:
            completed_launches.append(src_id)
            continue
        avail = available[src_id] - spent[src_id]
        if avail < ships:
            completed_launches.append(src_id)
            continue
        target = world.planet_by_id[plan["target_id"]]
        
        aim = aim_at_target(src, target, ships, world.initial_by_id, world.ang_vel, world=world)
        if aim is None:
            completed_launches.append(src_id)
            continue
        angle, turns = aim
        _commit_fleet(world, moves, spent, target_locked,
                      src_id, plan["target_id"], angle, turns, int(ships))
        mode_log[src_id] = "hammer"
        launch["fired"] = True

    
    for sid in completed_launches:
        plan["launches"].pop(sid, None)
    if not plan["launches"] or all(l.get("fired") for l in plan["launches"].values()):
        _hammer_plan = None


def _hammer_should_fire(world):
    """Trigger condition: my prod share >= mode-specific threshold AND a strong
    enemy production target is reachable, OR we're in late-flush mode."""
    if world.is_late:
        return True
    threshold = world.mode_params["hammer_prod_share"]
    if world.my_prod_share < threshold:
        return False
    return True


def _build_hammer_plan(world, available, spent):
    """Pick best target + stockpile set. Stockpiles are planets with ships >= MIN
    or promoted-by-idle. Combined arrival fleet must beat defender × overkill.
    Returns plan dict or None."""
    
    
    
    stockpile_min = world.mode_params.get("hammer_stockpile_min", HAMMER_STOCKPILE_MIN)
    stockpiles = []
    for p in world.my_planets:
        
        avail = _routine_avail(world, p, available[p.id] - spent[p.id])
        if avail < HAMMER_MIN_PER_CONTRIBUTOR:
            continue
        promoted = p.id in _promoted_stockpiles
        if avail < stockpile_min and not promoted:
            continue
        stockpiles.append((p, avail))
    if not stockpiles:
        return None

    overkill = LATE_FLUSH_OVERKILL_RATIO if world.is_late else world.mode_params["hammer_overkill"]

    targets = [
        p for p in world.enemy_planets
        if is_targetable(world, p) and p.production >= HAMMER_TARGET_PROD_MIN
    ]
    if not targets:
        if world.is_late:
            targets = [p for p in world.enemy_planets if is_targetable(world, p)]
        if not targets:
            return None

    best = None
    for tgt in targets:
        
        per_src = []
        for src, avail in stockpiles:
            aim = aim_at_target(src, tgt, max(1, avail), world.initial_by_id, world.ang_vel, world=world)
            if aim is None:
                continue
            angle, turns = aim
            if turns > HAMMER_MAX_TRAVEL:
                continue
            per_src.append((turns, src, avail, angle))
        if not per_src:
            continue
        
        per_src.sort()  
        target_arrival = per_src[-1][0]
        d_owner, d_ships = predict_defender_at_arrival(world, tgt, target_arrival)
        if d_owner == world.player:
            continue
        required = int(math.ceil(d_ships * overkill)) + 1

        
        accum = 0
        chosen = []
        for turns, src, avail, angle in per_src:
            chosen.append((turns, src, avail, angle))
            accum += avail
            if accum >= required:
                break
        if accum < required:
            continue

        
        
        
        
        
        
        
        slack = accum - required
        if slack > 0 and chosen:
            last_turn, last_src, last_avail, last_angle = chosen[-1]
            
            
            
            
            
            
            oversend_active = (
                HAMMER_NO_THREAT_OVERSEND_ENABLED
                and (not HAMMER_NO_THREAT_OVERSEND_2P_ONLY or world.is_2p)
            )
            
            last_src_threat = sum(
                int(ships) for eta, owner, ships
                in world.arrivals_by_planet.get(last_src.id, [])
                if owner != world.player and owner != -1
            )
            
            
            safe_surplus_ok = (
                HAMMER_SAFE_SURPLUS_OVERSEND_ENABLED
                and last_avail >= required * HAMMER_SAFE_SURPLUS_RATIO
                and last_src_threat <= last_avail * HAMMER_OVERSEND_MAX_THREAT_RATIO
            )
            if safe_surplus_ok:
                
                pass
            elif oversend_active and HAMMER_ALWAYS_OVERSEND_2P and world.is_2p:
                
                pass
            elif oversend_active and last_src_threat == 0:
                
                pass
            else:
                trimmed = last_avail - slack
                if trimmed < HAMMER_MIN_PER_CONTRIBUTOR:
                    chosen.pop()
                    if not chosen or sum(c[2] for c in chosen) < required - last_avail:
                        chosen.append((last_turn, last_src, last_avail, last_angle))
                else:
                    chosen[-1] = (last_turn, last_src, trimmed, last_angle)

        score = required - target_arrival * 0.5  
        
        if (F14_4A_2P_FOCUS_ENABLED and world.is_2p
                and getattr(world, "focus_enemy_2p", None) is not None
                and tgt.owner == world.focus_enemy_2p):
            score += F14_4A_2P_FOCUS_HAMMER_BONUS
        
        
        if FLEET_INTENT_ENABLED and tgt.id in _enemy_recently_launched:
            score += FLEET_INTENT_HAMMER_BONUS
        
        
        
        if R1_RECAPTURE_PRIORITY_ENABLED and tgt.id in _freshly_lost_planets:
            score += R1_RECAPTURE_HAMMER_BONUS
        
        
        
        
        
        
        
        if not world.is_2p:
            my_strength = world.owner_strength.get(world.player, 0)
            enemy_strengths = [
                (world.owner_strength[o], o)
                for o in world.owner_strength
                if o not in (-1, world.player) and world.owner_strength[o] > 0
            ]
            if enemy_strengths:
                max_enemy_strength, max_enemy_owner = max(enemy_strengths)
                if max_enemy_strength > my_strength and tgt.owner == max_enemy_owner:
                    score = score - abs(score) * 0.3
        cand = {
            "target_id": tgt.id,
            "target_arrival_abs": world.step + target_arrival,
            "committed_strength": sum(c[2] for c in chosen),
            "score": score,
            "launches": {},
        }
        for turns, src, ships, angle in chosen:
            fire_turn_rel = target_arrival - turns
            cand["launches"][src.id] = {
                "fire_turn_abs": world.step + fire_turn_rel,
                "ships": int(ships),
                "angle": float(angle),
                "fired": False,
            }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best






def handle_multiprong(world, available, spent, target_locked, moves, mode_log):
    """If a hammer is committed at target T and a credible enemy reinforcer E
    is pumping ships into T, open a same-turn second prong at E using surplus
    ships. Strict credibility gates: 2P only, real-reinforcement gate, post-
    launch garrison gate, prong-credibility gate.

    The picture-1 failure: bot fed all output into one stream against an
    actively-reinforced target. Two prongs force the opponent to choose:
    defend T -> we take E (no more reinforcements -> hammer lands clean);
    defend E -> they pull ships off T (hammer lands clean).
    """
    if not MULTIPRONG_ENABLED:
        return
    if MULTIPRONG_2P_ONLY and not world.is_2p:
        return
    if _hammer_plan is None:
        return

    target_id = _hammer_plan.get("target_id")
    target = world.planet_by_id.get(target_id)
    if target is None or target.owner == world.player or target.owner == -1:
        return
    arrival_rel = _hammer_plan.get("target_arrival_abs", world.step) - world.step
    if arrival_rel <= 0:
        return
    committed = int(_hammer_plan.get("committed_strength", 0))
    if committed <= 0:
        return

    
    
    reinforcer_ships = defaultdict(int)
    for f in world.fleets:
        if int(f.ships) <= 0:
            continue
        if f.owner == world.player or f.owner == -1:
            continue
        ftarget, _eta = fleet_target_planet(
            f, world.planets, world.initial_by_id, world.ang_vel
        )
        if ftarget is None or ftarget.id != target_id:
            continue
        reinforcer_ships[int(f.from_planet_id)] += int(f.ships)
    if not reinforcer_ships:
        return

    
    _, defender_at_arrival = predict_defender_at_arrival(world, target, arrival_rel)
    needed_t = int(math.ceil(defender_at_arrival)) + 1
    deficit = max(0, needed_t - committed)

    
    
    
    
    min_reinforce = max(1, int(math.ceil(deficit * MULTIPRONG_REINFORCER_MIN_RATIO)))

    
    candidates = []
    for src_id, ship_count in reinforcer_ships.items():
        src = world.planet_by_id.get(src_id)
        if src is None:
            continue
        if src.owner == world.player or src.owner == -1:
            continue
        if ship_count < min_reinforce:
            continue
        candidates.append((src, ship_count))
    if not candidates:
        return
    
    candidates.sort(key=lambda kv: kv[1], reverse=True)

    
    for reinforcer, in_flight in candidates:
        if reinforcer.id in target_locked:
            continue
        if not is_targetable(world, reinforcer):
            continue
        
        prong = _build_multiprong_attack(
            world, reinforcer, available, spent, target_locked
        )
        if prong is None:
            continue
        prong_strength, prong_arrival, prong_landings, e_at_arrival = prong

        
        
        
        
        if prong_strength <= e_at_arrival * MULTIPRONG_E_OVERKILL:
            continue
        
        needed_e = int(math.ceil(e_at_arrival)) + 1
        if committed + prong_strength < needed_t + int(round(needed_e * MULTIPRONG_CREDIBILITY_FACTOR)):
            continue

        
        for src_id, src, angle, ships, turns in prong_landings:
            _commit_fleet(
                world, moves, spent, target_locked,
                src_id, reinforcer.id, angle, turns, int(ships),
            )
            mode_log[src_id] = "multiprong"
        mode_log[reinforcer.id] = "multiprong-target"
        return  


def _build_multiprong_attack(world, target, available, spent, target_locked):
    """Plan a 1-3 source attack on `target` from surplus ships (post-hammer,
    post-expand, post-defense). Returns (strength, arrival_turn, landings, e_at_arrival) or None.

    Each landing: (src_id, src, angle, ships, turns).
    """
    sources = []
    for src in world.my_planets:
        avail = available[src.id] - spent[src.id]
        if avail < MULTIPRONG_MIN_PER_CONTRIBUTOR:
            continue
        
        aim = aim_at_target(src, target, max(MULTIPRONG_MIN_PER_CONTRIBUTOR, avail), world.initial_by_id, world.ang_vel, world=world)
        if aim is None:
            continue
        _angle, est_turns = aim
        if est_turns > MULTIPRONG_MAX_TRAVEL:
            continue
        sources.append((est_turns, src, avail))
    if not sources:
        return None
    sources.sort(key=lambda kv: kv[0])  

    
    
    chosen = []
    for est_turns, src, avail in sources[:MULTIPRONG_MAX_PARTICIPANTS]:
        chosen.append((est_turns, src, avail))
        common_arrival = max(t for t, _, _ in chosen)
        _, e_at_arrival = predict_defender_at_arrival(world, target, common_arrival)
        total_avail = sum(a for _, _, a in chosen)
        required = int(math.ceil(e_at_arrival * MULTIPRONG_E_OVERKILL)) + 1
        if total_avail >= required:
            break
    common_arrival = max(t for t, _, _ in chosen)
    _, e_at_arrival = predict_defender_at_arrival(world, target, common_arrival)
    required = int(math.ceil(e_at_arrival * MULTIPRONG_E_OVERKILL)) + 1
    total_avail = sum(a for _, _, a in chosen)
    if total_avail < required:
        return None

    
    slack = total_avail - required
    if slack > 0 and chosen:
        last_turn, last_src, last_avail = chosen[-1]
        trimmed = last_avail - slack
        if trimmed >= MULTIPRONG_MIN_PER_CONTRIBUTOR:
            chosen[-1] = (last_turn, last_src, trimmed)

    
    landings = []
    final_strength = 0
    for est_turns, src, ships in chosen:
        if ships < MULTIPRONG_MIN_PER_CONTRIBUTOR:
            return None
        aim = aim_at_target(src, target, ships, world.initial_by_id, world.ang_vel, world=world)
        if aim is None:
            return None
        angle, turns = aim
        if turns > MULTIPRONG_MAX_TRAVEL:
            return None
        landings.append((src.id, src, angle, int(ships), int(turns)))
        final_strength += int(ships)

    
    final_arrival = max(turns for _, _, _, _, turns in landings)
    _, final_defender = predict_defender_at_arrival(world, target, final_arrival)
    final_required = int(math.ceil(final_defender * MULTIPRONG_E_OVERKILL)) + 1
    if final_strength < final_required:
        return None

    return final_strength, final_arrival, landings, final_defender






# The CMA-ES Optimizer will tune these values
CANDIDATE_WEIGHTS = {
    "dist_penalty": -1.5,
    "dist_exponent": 1.0,
    "target_enemy_prod_bonus": 5.0,
    "target_neutral_cost_penalty": -0.5,
    "friendly_defense_bonus": 10.0,
    "retreat_threshold": 1.5,
}

def evaluate_board(state_dict, in_flight_counts, world, my_player_id):
    score = 0.0
    my_ships = in_flight_counts.get(my_player_id, 0)
    my_prod = 0
    my_planets = 0
    
    enemy_ships_total = 0
    enemy_prod_total = 0
    owner_ship_counts = defaultdict(int)
    for o, s in in_flight_counts.items():
        owner_ship_counts[o] += s
        if o != my_player_id and o != -1:
            enemy_ships_total += s
            
    # Spatial Moments Tracking
    my_x_sum = 0.0
    my_y_sum = 0.0
    enemy_x_sum = 0.0
    enemy_y_sum = 0.0
    
    for p in world.planets:
        state = state_dict.get(p.id)
        if not state:
            continue
            
        owner = state[0]
        ships = state[1] + in_flight_counts.get(p.id, 0)
        
        if owner == my_player_id:
            my_ships += ships
            my_prod += p.production
            my_planets += 1
            owner_ship_counts[my_player_id] += ships
            my_x_sum += p.x * ships
            my_y_sum += p.y * ships
        elif owner != -1:
            enemy_ships_total += ships
            enemy_prod_total += p.production
            owner_ship_counts[owner] += ships
            enemy_x_sum += p.x * ships
            enemy_y_sum += p.y * ships
            
    # Calculate Spatial Variance (Spread)
    my_spread = 0.0
    enemy_spread = 0.0
    
    if my_ships > 0:
        my_cx = my_x_sum / my_ships
        my_cy = my_y_sum / my_ships
        for p in world.planets:
            state = state_dict.get(p.id)
            if state and state[0] == my_player_id:
                s = state[1] + in_flight_counts.get(p.id, 0)
                my_spread += s * ((p.x - my_cx)**2 + (p.y - my_cy)**2)
        my_spread /= my_ships
        my_spread = my_spread ** 0.5  # Convert variance to standard deviation
        
    if enemy_ships_total > 0:
        enemy_cx = enemy_x_sum / enemy_ships_total
        enemy_cy = enemy_y_sum / enemy_ships_total
        for p in world.planets:
            state = state_dict.get(p.id)
            if state and state[0] != my_player_id and state[0] != -1:
                s = state[1] + in_flight_counts.get(p.id, 0)
                enemy_spread += s * ((p.x - enemy_cx)**2 + (p.y - enemy_cy)**2)
        enemy_spread /= enemy_ships_total
        enemy_spread = enemy_spread ** 0.5  # Convert variance to standard deviation
            
    leader_ships = 0
    if not world.is_2p:
        for o, s in owner_ship_counts.items():
            if o != my_player_id and o != -1:
                if s > leader_ships:
                    leader_ships = s

    score += my_ships * EVAL_WEIGHTS["my_ships"]
    score += my_prod * EVAL_WEIGHTS["my_production"]
    score += my_planets * EVAL_WEIGHTS["my_planets"]
    score += enemy_ships_total * EVAL_WEIGHTS["enemy_ships_total"]
    score += enemy_prod_total * EVAL_WEIGHTS["enemy_prod_total"]
    score += leader_ships * EVAL_WEIGHTS["enemy_leader_ships"]
        
    score += (my_spread ** EVAL_WEIGHTS["my_spread_exponent"]) * EVAL_WEIGHTS["my_spread_penalty"]
    score += (enemy_spread ** EVAL_WEIGHTS["enemy_spread_exponent"]) * EVAL_WEIGHTS["enemy_spread_bonus"]
    
    return score

@njit(fastmath=True)
def fast_score_action_via_delta(
    p_owners, p_ships, p_prods, p_x, p_y,
    arrival_pid, arrival_eta, arrival_owner, arrival_ships,
    max_h, my_player_id, is_2p, eval_weights, arrivals_table
):
    num_planets = len(p_owners)
    max_players = 6
    
    for i in range(len(arrival_pid)):
        eta = arrival_eta[i]
        if 0 < eta <= max_h:
            arrivals_table[eta, arrival_pid[i], arrival_owner[i] + 1] += arrival_ships[i]
            
    state_owners = p_owners.copy()
    state_ships = p_ships.copy()
    
    total_score = 0.0
    
    for t in range(1, max_h + 1):
        for pid in range(num_planets):
            if state_owners[pid] != -1:
                state_ships[pid] += p_prods[pid]
                
        # Resolve arrivals
        for pid in range(num_planets):
            has_arrivals = False
            for p_idx in range(max_players):
                if arrivals_table[t, pid, p_idx] > 0:
                    has_arrivals = True
                    break
            if not has_arrivals:
                continue
                
            defender_owner = state_owners[pid]
            garrison = state_ships[pid]
            
            top_ships = 0.0
            top_owner = -2
            second_ships = 0.0
            
            for p_idx in range(max_players):
                s = arrivals_table[t, pid, p_idx]
                if s == 0:
                    continue
                o = p_idx - 1
                if s > top_ships:
                    second_ships = top_ships
                    top_ships = s
                    top_owner = o
                elif s > second_ships:
                    second_ships = s
                    
            if top_ships == second_ships:
                survivor_ships = 0.0
                survivor_owner = -1
            else:
                survivor_ships = top_ships - second_ships
                survivor_owner = top_owner
                
            if survivor_ships > 0:
                if defender_owner == survivor_owner:
                    state_ships[pid] = garrison + survivor_ships
                else:
                    new_garrison = garrison - survivor_ships
                    if new_garrison < 0:
                        state_owners[pid] = survivor_owner
                        state_ships[pid] = -new_garrison
                    else:
                        state_ships[pid] = new_garrison
                        
        if t == 30 or t == 60 or t == 100:
            my_ships = 0.0
            my_prod = 0.0
            my_planets = 0.0
            enemy_ships_total = 0.0
            enemy_prod_total = 0.0
            
            owner_ships = np.zeros(max_players, dtype=np.float64)
            
            my_x_sum = 0.0
            my_y_sum = 0.0
            enemy_x_sum = 0.0
            enemy_y_sum = 0.0
            
            for pid in range(num_planets):
                o = state_owners[pid]
                s = state_ships[pid]
                
                # Add future arrivals to owner_ships
                for i in range(len(arrival_pid)):
                    if arrival_pid[i] == pid and arrival_eta[i] > t:
                        arr_o = arrival_owner[i]
                        p_idx = arr_o + 1
                        future_s = arrival_ships[i]
                        owner_ships[p_idx] += future_s
                        if arr_o == my_player_id:
                            my_ships += future_s
                            my_x_sum += p_x[pid] * future_s
                            my_y_sum += p_y[pid] * future_s
                        elif arr_o != -1:
                            enemy_ships_total += future_s
                            enemy_x_sum += p_x[pid] * future_s
                            enemy_y_sum += p_y[pid] * future_s
                
                owner_ships[o + 1] += s
                if o == my_player_id:
                    my_ships += s
                    my_prod += p_prods[pid]
                    my_planets += 1.0
                    my_x_sum += p_x[pid] * s
                    my_y_sum += p_y[pid] * s
                elif o != -1:
                    enemy_ships_total += s
                    enemy_prod_total += p_prods[pid]
                    enemy_x_sum += p_x[pid] * s
                    enemy_y_sum += p_y[pid] * s
                    
            my_spread = 0.0
            enemy_spread = 0.0
            
            if my_ships > 0:
                my_cx = my_x_sum / my_ships
                my_cy = my_y_sum / my_ships
                for pid in range(num_planets):
                    o = state_owners[pid]
                    if o == my_player_id:
                        s = state_ships[pid]
                        my_spread += s * ((p_x[pid] - my_cx)**2 + (p_y[pid] - my_cy)**2)
                    for i in range(len(arrival_pid)):
                        if arrival_pid[i] == pid and arrival_eta[i] > t and arrival_owner[i] == my_player_id:
                            future_s = arrival_ships[i]
                            my_spread += future_s * ((p_x[pid] - my_cx)**2 + (p_y[pid] - my_cy)**2)
                my_spread = (my_spread / my_ships) ** 0.5
                
            if enemy_ships_total > 0:
                enemy_cx = enemy_x_sum / enemy_ships_total
                enemy_cy = enemy_y_sum / enemy_ships_total
                for pid in range(num_planets):
                    o = state_owners[pid]
                    if o != my_player_id and o != -1:
                        s = state_ships[pid]
                        enemy_spread += s * ((p_x[pid] - enemy_cx)**2 + (p_y[pid] - enemy_cy)**2)
                    for i in range(len(arrival_pid)):
                        if arrival_pid[i] == pid and arrival_eta[i] > t:
                            arr_o = arrival_owner[i]
                            if arr_o != my_player_id and arr_o != -1:
                                future_s = arrival_ships[i]
                                enemy_spread += future_s * ((p_x[pid] - enemy_cx)**2 + (p_y[pid] - enemy_cy)**2)
                enemy_spread = (enemy_spread / enemy_ships_total) ** 0.5

            leader_ships = 0.0
            we_are_leader = False
            if not is_2p:
                for p_idx in range(1, max_players):
                    o = p_idx - 1
                    if o != my_player_id and o != -1:
                        if owner_ships[p_idx] > leader_ships:
                            leader_ships = owner_ships[p_idx]
                if my_ships > leader_ships and leader_ships > 0:
                    we_are_leader = True
                    
            score = 0.0
            score += my_ships * eval_weights[0]
            score += my_prod * eval_weights[1]
            score += my_planets * eval_weights[2]
            score += enemy_ships_total * eval_weights[3]
            score += enemy_prod_total * eval_weights[4]
            score += leader_ships * eval_weights[5]
            
            score += (my_spread ** eval_weights[7]) * eval_weights[6]
            score += (enemy_spread ** eval_weights[9]) * eval_weights[8]
            
            if t == 30:
                total_score += score * eval_weights[10]
            elif t == 60:
                total_score += score * eval_weights[11]
            elif t == 100:
                total_score += score * eval_weights[12]

    # Zero out arrivals_table for reuse
    for i in range(len(arrival_pid)):
        eta = arrival_eta[i]
        if 0 < eta <= max_h:
            arrivals_table[eta, arrival_pid[i], arrival_owner[i] + 1] = 0.0

    return total_score

def prepare_world_state(world):
    max_h = 100
    max_pid = max((p.id for p in world.planets), default=-1)
    num_planets = max_pid + 1
    
    p_owners = np.full(num_planets, -1, dtype=np.int32)
    p_ships = np.zeros(num_planets, dtype=np.float64)
    p_prods = np.zeros(num_planets, dtype=np.float64)
    p_x = np.zeros(num_planets, dtype=np.float64)
    p_y = np.zeros(num_planets, dtype=np.float64)
    
    for p in world.planets:
        p_owners[p.id] = int(p.owner)
        p_ships[p.id] = float(p.ships)
        p_prods[p.id] = float(p.production)
        p_x[p.id] = float(p.x)
        p_y[p.id] = float(p.y)
        
    arrival_pid_list = []
    arrival_eta_list = []
    arrival_owner_list = []
    arrival_ships_list = []
    
    for pid, arrs in world.arrivals_by_planet.items():
        for eta, owner, ships in arrs:
            if 0 < eta <= max_h:
                arrival_pid_list.append(int(pid))
                arrival_eta_list.append(int(eta))
                arrival_owner_list.append(int(owner))
                arrival_ships_list.append(float(ships))
                
    eval_weights = np.array([
        EVAL_WEIGHTS["my_ships"],              # 0
        EVAL_WEIGHTS["my_production"],         # 1
        EVAL_WEIGHTS["my_planets"],            # 2
        EVAL_WEIGHTS["enemy_ships_total"],     # 3
        EVAL_WEIGHTS["enemy_prod_total"],      # 4
        EVAL_WEIGHTS["enemy_leader_ships"],    # 5
        EVAL_WEIGHTS["my_spread_penalty"],     # 6
        EVAL_WEIGHTS["my_spread_exponent"],    # 7
        EVAL_WEIGHTS["enemy_spread_bonus"],    # 8
        EVAL_WEIGHTS["enemy_spread_exponent"], # 9
        EVAL_WEIGHTS["weight_t30"],            # 10
        EVAL_WEIGHTS["weight_t60"],            # 11
        EVAL_WEIGHTS["weight_t100"]            # 12
    ], dtype=np.float64)
    
    arrivals_table = np.zeros((max_h + 1, num_planets, 6), dtype=np.float64)
    
    return (p_owners, p_ships, p_prods, p_x, p_y, 
            arrival_pid_list, arrival_eta_list, arrival_owner_list, arrival_ships_list, 
            eval_weights, arrivals_table, max_h)

def score_actions_given_state(base_state, world, actions):
    (b_owners, b_ships, b_prods, b_x, b_y, 
     b_arr_pid, b_arr_eta, b_arr_owner, b_arr_ships, 
     eval_weights, arrivals_table, max_h) = base_state
     
    p_owners = b_owners.copy()
    p_ships = b_ships.copy()
    
    a_pid = b_arr_pid.copy()
    a_eta = b_arr_eta.copy()
    a_owner = b_arr_owner.copy()
    a_ships = b_arr_ships.copy()
    
    if actions:
        for action in actions:
            src = action['src']
            if p_owners[src] == world.player:
                p_ships[src] = max(0.0, p_ships[src] - float(action['ships']))
            
            tgt = action['tgt']
            eta = action['eta']
            ships = action['ships']
            if 0 < eta <= max_h:
                a_pid.append(int(tgt))
                a_eta.append(int(eta))
                a_owner.append(int(world.player))
                a_ships.append(float(ships))
                
            tgt_planet = world.planet_by_id[tgt]
            for ep in world.enemy_planets:
                if ep.id == tgt: continue
                if ep.ships < 10: continue
                d = ((tgt_planet.x - ep.x)**2 + (tgt_planet.y - ep.y)**2)**0.5
                if d > 30.0: continue
                
                opp_ships = int(ep.ships * 0.3)
                ratio = math.log(max(2, opp_ships)) / math.log(1000.0)
                speed = 1.0 + 2.0 * (ratio ** 1.5)
                opp_eta = max(1, int(math.ceil(d / speed)))
                
                if 0 < opp_eta <= max_h:
                    a_pid.append(int(tgt))
                    a_eta.append(int(opp_eta))
                    a_owner.append(int(ep.owner))
                    a_ships.append(float(opp_ships))
                    
    arr_pid_np = np.array(a_pid, dtype=np.int32)
    arr_eta_np = np.array(a_eta, dtype=np.int32)
    arr_owner_np = np.array(a_owner, dtype=np.int32)
    arr_ships_np = np.array(a_ships, dtype=np.float64)
    
    return fast_score_action_via_delta(
        p_owners, p_ships, b_prods, b_x, b_y,
        arr_pid_np, arr_eta_np, arr_owner_np, arr_ships_np,
        max_h, world.player, world.is_2p, eval_weights, arrivals_table
    )

def score_action_via_delta(world, actions=None):
    base_state = prepare_world_state(world)
    return score_actions_given_state(base_state, world, actions)

def generate_candidate_missions(world, available):
    """
    Generate mathematically scored mission candidates and feed top 20 to Beam Search & Oracle.
    """
    moves = []
    base_state = prepare_world_state(world)
    baseline_score = score_actions_given_state(base_state, world, None)
    
    candidates = []
    
    # 1. Generate all possible source -> target transfers
    for src in world.my_planets:
        avail = available.get(src.id, 0)
        if avail < 5:
            continue
            
        for tgt in world.planets:
            if tgt.id == src.id:
                continue
                
            aim = aim_at_target(src, tgt, avail, world.initial_by_id, world.ang_vel, world=world)
            if aim is None:
                continue
            angle, turns = aim
            if turns > 60:
                continue
                
            # Quick Mathematical Scorer
            score = (turns ** CANDIDATE_WEIGHTS["dist_exponent"]) * CANDIDATE_WEIGHTS["dist_penalty"]
            if tgt.owner != world.player and tgt.owner != -1:
                score += tgt.production * CANDIDATE_WEIGHTS["target_enemy_prod_bonus"]
            elif tgt.owner == -1:
                score += tgt.ships * CANDIDATE_WEIGHTS["target_neutral_cost_penalty"]
            elif tgt.owner == world.player:
                # Basic defense heuristic: if a friendly planet is under attack
                score += CANDIDATE_WEIGHTS["friendly_defense_bonus"]
                
            action = {'src': src.id, 'tgt': tgt.id, 'ships': avail, 'eta': turns, 'angle': angle}
            candidates.append((score, [action]))
                
    if not candidates:
        return moves
        
    candidates.sort(key=lambda x: -x[0])
    # Keep top 10 bundles to reduce branching factor for Beam Search
    top_candidates = [bundle for delta, bundle in candidates[:10]]
    
    # Beam Search
    beam = [([], baseline_score)] # List of (combination_of_bundles, joint_score)
    beam_width = 3
    max_depth = 2
    
    def is_valid_combo(combo, new_bundle):
        spent = defaultdict(int)
        for bundle in combo:
            for act in bundle:
                spent[act['src']] += act['ships']
        for act in new_bundle:
            if available.get(act['src'], 0) - spent[act['src']] < act['ships']:
                return False
        return True
        
    for depth in range(max_depth):
        new_beam = []
        for combo, _ in beam:
            for new_bundle in top_candidates:
                # Enforce ordering to prevent duplicate permutations (A+B vs B+A)
                if combo and top_candidates.index(new_bundle) <= top_candidates.index(combo[-1]):
                    continue
                    
                if is_valid_combo(combo, new_bundle):
                    new_combo = combo + [new_bundle]
                    # Flatten the combo to score
                    flattened = [act for b in new_combo for act in b]
                    score = score_actions_given_state(base_state, world, flattened)
                    new_beam.append((new_combo, score))
                    
        if not new_beam:
            break
            
        # Keep the best combinations
        combined = beam + new_beam
        combined.sort(key=lambda x: -x[1])
        # Deduplicate scores just in case
        seen_scores = set()
        dedup_beam = []
        for c, s in combined:
            # simple deduplication by score to ensure diversity in the beam
            if round(s, 2) not in seen_scores:
                seen_scores.add(round(s, 2))
                dedup_beam.append((c, s))
                if len(dedup_beam) >= beam_width:
                    break
        beam = dedup_beam

    # Execute the best combination found
    best_combo = beam[0][0]
    for bundle in best_combo:
        for act in bundle:
            moves.append([act['src'], float(act['angle']), int(act['ships'])])
            
    return moves

def plan_moves(world, deadline=None):
    global _planet_idle_counts, _promoted_stockpiles, _pending_commitments

    
    
    
    
    
    
    def _commitment_viable(c):
        if c["arrival_abs"] <= world.step:
            return False
        target = world.planet_by_id.get(c["target_id"])
        if target is None:
            return False
        if target.owner == world.player:
            return False
        if FAILTOLERANT_ENABLED:
            owner_at_commit = c.get("owner_at_commit")
            if owner_at_commit is not None and int(target.owner) != int(owner_at_commit):
                return False
        return True
    _pending_commitments[:] = [c for c in _pending_commitments if _commitment_viable(c)]

    
    _update_neutral_watchlist(world)

    moves = []
    spent = defaultdict(int)
    target_locked = set()
    mode_log = {}

    
    rescue_needs = {}
    available = {}
    for p in world.my_planets:
        arrivals = world.arrivals_by_planet.get(p.id, [])
        reserve, holds, deficit, dline = compute_planet_reserve(
            p, arrivals, world.player
        )
        available[p.id] = max(0, int(p.ships) - reserve)
        if not holds:
            rescue_needs[p.id] = (deficit, dline, p)
            mode_log[p.id] = "absorb-need-rescue"
        elif arrivals:
            mode_log[p.id] = "absorb"

    
    
    
    def _over_budget():
        return deadline is not None and time.perf_counter() >= deadline

    # PHASE 1: Hook the new delta scoring system into the main agent logic
    # We replace all the disjointed heuristics (expand, hammer, pickup) with our unified delta evaluator!
    if not _over_budget():
        moves.extend(generate_candidate_missions(world, available))

    return moves






def agent(obs, config=None):
    global _agent_step, _hammer_plan, _planet_idle_counts, _promoted_stockpiles, _pending_commitments
    global _game_num_players, _2p_patient_streak, _2p_prod_share_history

    global _opp_profile  
    obs_step = _read(obs, "step", 0) or 0
    if obs_step == 0:
        _agent_step = 0
        _hammer_plan = None
        _planet_idle_counts = {}
        _promoted_stockpiles = set()
        _pending_commitments = []
        _game_num_players = None
        _2p_patient_streak = 0
        _2p_prod_share_history = []
        _neutral_prev_ships.clear()
        _neutral_wounded.clear()
        _enemy_prev_ships.clear()
        _enemy_recently_launched.clear()
        _planet_prev_owner.clear()
        _freshly_lost_planets.clear()
        _opp_profile = {}
    _agent_step += 1
    
    # Apply dynamic phase transitions
    EVAL_WEIGHTS["my_production"] = EVAL_WEIGHTS.get("my_production_base", 15.0) + EVAL_WEIGHTS.get("my_production_decay", -0.05) * _agent_step

    start = time.perf_counter()
    world = World(obs, inferred_step=_agent_step - 1)
    if not world.my_planets:
        return []

    
    if not world.is_2p:
        _update_opp_profile_4p(world)

    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    soft_budget = max(0.5, act_timeout * SOFT_DEADLINE_FRACTION)
    deadline = start + soft_budget

    return plan_moves(world, deadline=deadline)


__all__ = ["agent", "Planet", "Fleet"]