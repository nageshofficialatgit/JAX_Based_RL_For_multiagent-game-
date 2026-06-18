import os
import gc
import numpy as np
import grain.python as grain
from numba import njit

@njit(cache=True)
def check_sun_collision_scalar(src_x, src_y, tgt_x, tgt_y):
    """LLVM-compiled scalar sun collision check."""
    dx = tgt_x - src_x
    dy = tgt_y - src_y
    L2 = dx * dx + dy * dy + 1e-8
    fx = 50.0 - src_x
    fy = 50.0 - src_y
    t = (fx * dx + fy * dy) / L2
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    cx = src_x + t * dx
    cy = src_y + t * dy
    return (cx - 50.0) * (cx - 50.0) + (cy - 50.0) * (cy - 50.0) < 100.0

def check_sun_collision(src_x, src_y, tgt_x, tgt_y):
    """Vectorized check if a line segment crosses the Sun at (50, 50) with radius 10."""
    dx = tgt_x - src_x
    dy = tgt_y - src_y
    L2 = dx**2 + dy**2 + 1e-8
    fx = 50.0 - src_x
    fy = 50.0 - src_y
    t = (fx * dx + fy * dy) / L2
    t_clamped = np.clip(t, 0.0, 1.0)
    closest_x = src_x + t_clamped * dx
    closest_y = src_y + t_clamped * dy
    dist_sq = (closest_x - 50.0)**2 + (closest_y - 50.0)**2
    return dist_sq < 100.0

@njit(cache=True)
def get_fleet_speed_numba(ships):
    """LLVM-compiled fleet speed calculation."""
    if ships <= 1.0:
        return 1.0
    ratio = np.log(ships) / np.log(1000.0)
    if ratio > 1.0: ratio = 1.0
    if ratio < 0.0: ratio = 0.0
    return 1.0 + 5.0 * (ratio ** 1.5)

class OrbitWarsDataSource(grain.RandomAccessDataSource):
    """
    High-throughput Google Grain DataSource for Orbit Wars Parquet Database.
    Replaces the PyTorch Dataset for the Pure JAX/Flax Stack.
    """
    def __init__(self, db_path="parquet_db_real", max_action_history=20, grandmaster_only=True, subsample_ticks=5, is_iql_mode=True):
        self.db_path = db_path
        self.max_action_history = max_action_history
        self.subsample_ticks = subsample_ticks
        self.is_iql_mode = is_iql_mode
        
        print("Loading Parquet Database with Polars (Rust Engine)...")
        import polars as pl
        import pandas as pd
        
        self.df_episodes = pd.read_parquet(os.path.join(db_path, "episodes.parquet"))
        self.df_player_eps = pd.read_parquet(os.path.join(db_path, "player_episodes.parquet"))
        
        # We only need the index from planet_state
        planet_state_path = os.path.join(db_path, "planet_state_coords.parquet")
        if not os.path.exists(planet_state_path):
            planet_state_path = os.path.join(db_path, "planet_state.parquet")
        df_planet_state = pd.read_parquet(planet_state_path)
        
        winners = self.df_player_eps[self.df_player_eps['is_winner'] == 1]
        self.episode_to_winner = dict(zip(winners['episode_id'], winners['slot']))
        
        # Load leaderboard to map winner -> win_rate (0.0 to 1.0 continuous gradient)
        lb_path = os.path.join(db_path, "leaderboard.csv")
        player_to_win_rate = {}
        if os.path.exists(lb_path):
            lb = pd.read_csv(lb_path)
            player_to_win_rate = dict(zip(lb['name'], lb['win_rate']))
        else:
            print("WARNING: leaderboard.csv not found. Defaulting to 0.5 win rate.")
        
        self.episode_to_win_rate = {}
        for _, row in winners.iterrows():
            name = row['name']
            self.episode_to_win_rate[row['episode_id']] = player_to_win_rate.get(name, 0.5)
            
        # Load episodes to get angular_velocity for physics calculation
        eps = pd.read_parquet(os.path.join(db_path, "episodes.parquet"))
        self.episode_to_angular_vel = dict(zip(eps['episode_id'], eps['angular_velocity']))
        
        # Filter by Grandmaster Episodes if enabled
        self.gm_ids = None
        if grandmaster_only:
            gm_file = os.path.join(db_path, "grandmaster_episodes.csv")
            if os.path.exists(gm_file):
                print("Filtering dataset to Grandmaster episodes only...")
                gm_df = pd.read_csv(gm_file)
                self.gm_ids = set(gm_df['episode_id'].values)
        print("Collecting Polars Dataframe (Zero Pandas Overhead)...")
        df_planet_state_pl = pl.scan_parquet(os.path.join(self.db_path, "planet_state.parquet"))
        
        if self.gm_ids is not None:
            df_planet_state_pl = df_planet_state_pl.filter(pl.col('episode_id').is_in(self.gm_ids))
            
        if self.subsample_ticks > 1:
            df_planet_state_pl = df_planet_state_pl.filter(pl.col('tick') % self.subsample_ticks == 0)
            
        df_pl_mem = df_planet_state_pl.collect()
        
        # --- THE ZERO-COPY MULTIPROCESSING FIX ---
        print("Building Memory-Efficient NumPy Index for planet_state...")
        mmap_cache_path = os.path.join(self.db_path, "planet_state_mmap_cache.npy")
        
        # Only write the 6GB chunk to disk once
        if not os.path.exists(mmap_cache_path):
            print("Writing planet_state to disk for Zero-Copy IPC sharing...")
            p_id = df_pl_mem["planet_id"].to_numpy().astype(np.float32)
            owner = df_pl_mem["owner"].to_numpy().astype(np.float32)
            ships = df_pl_mem["ships"].to_numpy().astype(np.float32)
            temp_arr = np.stack([p_id, owner, ships], axis=1)
            np.save(mmap_cache_path, temp_arr)
            del temp_arr, p_id, owner, ships
            gc.collect()

        # Connect to the memory-mapped file. 
        # When Grain pickles this object, it only sends the file handle to the workers!
        print("Connecting to Shared Memory Map...")
        self.planet_state_arr = np.load(mmap_cache_path, mmap_mode='r')
        
        # Build dict mapping (episode_id, tick) -> (start_idx, end_idx)
        print("Extracting Episode/Tick keys...")
        ep = df_pl_mem["episode_id"].to_numpy()
        ticks = df_pl_mem["tick"].to_numpy()
        
        self.state_lookup = {}
        curr_key = None
        start_idx = 0
        for i in range(len(ep)):
            key = (ep[i], ticks[i])
            if key != curr_key:
                if curr_key is not None:
                    self.state_lookup[curr_key] = (start_idx, i)
                curr_key = key
                start_idx = i
        if curr_key is not None:
            self.state_lookup[curr_key] = (start_idx, len(ep))
            
        # CRITICAL FIX: Delete the massive 20GB Polars frame and arrays from RAM before doing anything else
        del df_pl_mem
        del ep
        del ticks
        gc.collect()
            
        print("Memory-Efficient NumPy Index built.")
        
        self.state_index = list(self.state_lookup.keys())
        # Subsample again if needed
        if self.subsample_ticks > 1:
            self.state_index = [k for k in self.state_index if k[1] % self.subsample_ticks == 0]
        
        # Filter out unused episodes for later stages using the state_lookup keys
        self.target_episodes = np.unique([k[0] for k in self.state_index])
        
        # Load episode_planets with Polars (Rust multi-threaded I/O)
        print("Loading episode_planets with Polars (Rust Engine)...")
        df_ep_planets_pl = pl.read_parquet(os.path.join(self.db_path, "episode_planets.parquet"))
        if hasattr(self, 'target_episodes'):
            df_ep_planets_pl = df_ep_planets_pl.filter(pl.col('episode_id').is_in(self.target_episodes.tolist()))
        
        self.episode_planets_dict_np = {}
        for group in df_ep_planets_pl.partition_by("episode_id", as_dict=False):
            group = group.sort("planet_id")
            ep_id = int(group["episode_id"][0])
            self.episode_planets_dict_np[ep_id] = {
                'initial_x': group["initial_x"].to_numpy().astype(np.float32),
                'initial_y': group["initial_y"].to_numpy().astype(np.float32),
                'radius': group["radius"].to_numpy().astype(np.float32),
                'production': group["production"].to_numpy().astype(np.float32),
                'is_static': group["is_static"].to_numpy().astype(np.float32),
                'is_comet': group["is_comet"].to_numpy().astype(np.float32),
                'planet_ids': group["planet_id"].to_numpy().astype(np.int32)
            }
        del df_ep_planets_pl
        gc.collect()
            
        # Load actions with Polars (Rust multi-threaded I/O)
        print("Loading actions with Polars (Rust Engine)...")
        df_actions_pl = pl.read_parquet(os.path.join(self.db_path, "actions.parquet"))
        if hasattr(self, 'target_episodes'):
            df_actions_pl = df_actions_pl.filter(pl.col('episode_id').is_in(self.target_episodes.tolist()))
        
        self.actions_dict_np = {}
        for group in df_actions_pl.sort("tick").partition_by("episode_id", as_dict=False):
            group = group.sort("tick")
            ep_id = int(group["episode_id"][0])
            
            if "target_planet_id" in group.columns:
                target_ids = group["target_planet_id"].fill_null(-1).to_numpy().astype(np.int32)
            else:
                target_ids = np.full(len(group), -1, dtype=np.int32)
                
            self.actions_dict_np[ep_id] = {
                'ticks': group["tick"].to_numpy().astype(np.int32),
                'slot': group["slot"].to_numpy().astype(np.float32),
                'src_planet_id': group["src_planet_id"].to_numpy().astype(np.int32),
                'n_ships': group["n_ships"].to_numpy().astype(np.float32),
                'angle': group["angle"].to_numpy().astype(np.float32),
                'target_planet_id': target_ids
            }
        del df_actions_pl
        gc.collect()
            
        print("All memory-efficient indexes built.")
        
        # --- THE MULTIPROCESSING IPC BYPASS ---
        # cache_dict_path = os.path.join(self.db_path, "metadata_dicts_cache.pkl")
        # if not os.path.exists(cache_dict_path):
        #     print("Writing metadata dictionaries to disk for fast worker loading...")
        #     import pickle
        #     with open(cache_dict_path, 'wb') as f:
        #         pickle.dump({
        #             'state_lookup': self.state_lookup,
        #             'episode_planets_dict_np': self.episode_planets_dict_np,
        #             'actions_dict_np': self.actions_dict_np,
        #             'episode_to_winner': self.episode_to_winner,
        #             'episode_to_win_rate': self.episode_to_win_rate,
        #             'episode_to_angular_vel': self.episode_to_angular_vel,
        #         }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # We are using workers=0, so keep them in memory to avoid the 46GB unpickling memory spike.
        # self.state_lookup = None
        # self.episode_planets_dict_np = None
        # self.actions_dict_np = None
        # self.episode_to_winner = None
        # self.episode_to_win_rate = None
        # self.episode_to_angular_vel = None
        self.planet_state_arr = None  # CRITICAL: pickle reads memmap bytes into RAM. We must bypass this.
        
        print(f"Grain DataSource ready. Total states to train on: {len(self.state_index)}")



    def __len__(self):
        return len(self.state_index)

    def __getitem__(self, idx):
        # WORKER LAZY LOAD (Disabled for workers=0 to avoid pickle RAM explosion)
        # if self.state_lookup is None:
        #     import pickle
        #     cache_dict_path = os.path.join(self.db_path, "metadata_dicts_cache.pkl")
        #     with open(cache_dict_path, 'rb') as f:
        #         d = pickle.load(f)
        #         self.state_lookup = d['state_lookup']
        #         self.episode_planets_dict_np = d['episode_planets_dict_np']
        #         self.actions_dict_np = d['actions_dict_np']
        #         self.episode_to_winner = d['episode_to_winner']
        #         self.episode_to_win_rate = d['episode_to_win_rate']
        #         self.episode_to_angular_vel = d['episode_to_angular_vel']
                
        if self.planet_state_arr is None:
            mmap_cache_path = os.path.join(self.db_path, "planet_state_mmap_cache.npy")
            self.planet_state_arr = np.load(mmap_cache_path, mmap_mode='r')

        episode_id, tick_val = self.state_index[idx]
        
        # STRICT CASTING: remove any Pandas metadata wrappers and keep shapes stable
        episode_id = int(episode_id)
        tick = int(tick_val)
        winner_slot = int(self.episode_to_winner.get(episode_id, -1))
        win_rate = float(self.episode_to_win_rate.get(episode_id, 0.5))
        angular_vel = float(self.episode_to_angular_vel.get(episode_id, 0.0))
        
        try:
            static_planets = self.episode_planets_dict_np[episode_id]
            if (episode_id, tick) not in self.state_lookup:
                raise KeyError(f"State not found: {(episode_id, tick)}")
                
            start_idx, end_idx = self.state_lookup[(episode_id, tick)]
            dynamic_arr = self.planet_state_arr[start_idx:end_idx]
            
            # PURE NUMPY (O(1) indexing, 100x faster than Pandas join)
            raw_owner = np.zeros(50, dtype=np.float32)
            raw_ships = np.zeros(50, dtype=np.float32)
            p_ids = dynamic_arr[:, 0].astype(int)
            valid_p_ids = p_ids < 50
            raw_owner[p_ids[valid_p_ids]] = dynamic_arr[valid_p_ids, 1]
            raw_ships[p_ids[valid_p_ids]] = dynamic_arr[valid_p_ids, 2]
            
        except KeyError as e:
            # Grain expects deterministic reads, so we fallback to index 0 safely
            return self.__getitem__(0)
            
        n_planets = len(static_planets['planet_ids'])
        
        # FIX: Ensure n_planets never exceeds the maximum sequence length of 50
        n_planets = min(n_planets, 50)
        
        # Compute exact orbital physics for current_x and current_y
        init_x = static_planets['initial_x']
        init_y = static_planets['initial_y']
        is_static = static_planets['is_static']
        is_comet = static_planets['is_comet']
        
        current_x = np.copy(init_x)
        current_y = np.copy(init_y)
        
        orbiting_mask = (is_static == 0) & (is_comet == 0)
        if np.any(orbiting_mask):
            dx = init_x[orbiting_mask] - 50.0
            dy = init_y[orbiting_mask] - 50.0
            orb_r = np.sqrt(dx**2 + dy**2)
            init_angle = np.arctan2(dy, dx)
            cur_angle = init_angle + angular_vel * tick
            current_x[orbiting_mask] = 50.0 + orb_r * np.cos(cur_angle)
            current_y[orbiting_mask] = 50.0 + orb_r * np.sin(cur_angle)
        
        # Calculate Polar Coordinates from center (50, 50)
        polar_r = np.zeros(50, dtype=np.float32)
        polar_theta = np.zeros(50, dtype=np.float32)
        
        if n_planets > 0:
            dx = current_x[:n_planets] - 50.0
            dy = current_y[:n_planets] - 50.0
            polar_r[:n_planets] = np.sqrt(dx**2 + dy**2) / 50.0
            theta = np.arctan2(dy, dx)
            
            # Find the center of mass of 'Me' to establish rotation-invariant angle 0
            my_mask = (raw_owner[:n_planets] == winner_slot)
            if np.any(my_mask):
                my_cx = np.mean(current_x[:n_planets][my_mask]) - 50.0
                my_cy = np.mean(current_y[:n_planets][my_mask]) - 50.0
                player_theta = np.arctan2(my_cy, my_cx)
            else:
                my_cx = 0.0
                my_cy = 0.0
                player_theta = 0.0
                
            rel_theta = theta - player_theta
            # Wrap to [-pi, pi]
            rel_theta = (rel_theta + np.pi) % (2 * np.pi) - np.pi
            # DO NOT DIVIDE BY PI! Keep in radians for Sin/Cos
            polar_theta[:n_planets] = rel_theta 
            
        # 1. State Features (Egocentric Owners)
        raw_owner_sliced = raw_owner[:n_planets]
        
        # FIX: Parquet stores Neutral as -1. Players are 0, 1, 2, 3.
        # We must map Neutral (-1) to 0.0.
        # We must map winner_slot to 1.0 (Ego).
        # Other players mapped to 2.0, 3.0, etc.
        ego_owner = np.where(
            raw_owner_sliced == -1.0, 0.0,
            np.where(
                raw_owner_sliced == float(winner_slot), 1.0,
                np.where(raw_owner_sliced < float(winner_slot), raw_owner_sliced + 2.0, raw_owner_sliced + 1.0)
            )
        )

        planet_tensors = np.zeros((50, 37), dtype=np.float32)
        planet_tensors[:, 0] = -1.0
        planet_tensors[:n_planets, 0] = 0.0
        planet_tensors[:n_planets, 1] = polar_r[:n_planets]
        
        # THE FIX: Euclidean Manifold Mapping for Attention Heads
        planet_tensors[:n_planets, 2] = np.sin(polar_theta[:n_planets])
        planet_tensors[:n_planets, 3] = np.cos(polar_theta[:n_planets])
        
        planet_tensors[:n_planets, 4] = static_planets['radius'][:n_planets] / 10.0
        planet_tensors[:n_planets, 5] = static_planets['production'][:n_planets] / 5.0
        planet_tensors[:n_planets, 6] = ego_owner 
        planet_tensors[:n_planets, 7] = np.log1p(raw_ships[:n_planets]) / 7.0
        planet_tensors[:n_planets, 8] = is_static[:n_planets].astype(np.float32)
        
        # Feature 9: Comet Mask
        planet_tensors[:n_planets, 9] = is_comet[:n_planets].astype(np.float32)
        
        # Feature 10: Normalized Game Tick
        planet_tensors[:n_planets, 10] = tick / 1000.0
        
        # Feature 11: True Local Angular Velocity
        true_ang_vel = angular_vel * (1.0 - is_static[:n_planets]) * (1.0 - is_comet[:n_planets])
        planet_tensors[:n_planets, 11] = true_ang_vel
        
        # 12 and 13 remain 0.0 placeholders for trajectory dx/dy
        # 2. V2 Planet-Centric Fleet Aggregation (Features 14-18)
        inc_1 = np.zeros(50, dtype=np.float32)
        inc_2 = np.zeros(50, dtype=np.float32)
        inc_3 = np.zeros(50, dtype=np.float32)
        inc_4 = np.zeros(50, dtype=np.float32)
        min_attacker_eta = np.full(50, np.inf, dtype=np.float32)
        min_G = np.full(50, np.inf, dtype=np.float32)
        
        # Initialize fleet variables to avoid UnboundLocalError if no recent fleets exist
        r_slots = np.array([], dtype=np.int32)
        r_ships = np.array([], dtype=np.float32)
        
        if episode_id in self.actions_dict_np:
            try:
                ep_actions = self.actions_dict_np[episode_id]
                ticks = ep_actions['ticks']
                
                # We only care about fleets launched recently (max 200 ticks ago is safe)
                recent_start = np.searchsorted(ticks, tick - 200, side='left')
                recent_end = np.searchsorted(ticks, tick, side='left')
                
                if recent_end > recent_start:
                    r_slots = ep_actions['slot'][recent_start:recent_end]
                    r_src = ep_actions['src_planet_id'][recent_start:recent_end]
                    r_ships = ep_actions['n_ships'][recent_start:recent_end]
                    r_angles = ep_actions['angle'][recent_start:recent_end]
                    r_ticks = ticks[recent_start:recent_end]
                    r_targets = ep_actions['target_planet_id'][recent_start:recent_end]
                    
                    # Calculate Fleet Speeds
                    safe_ships = np.maximum(r_ships, 1e-8)
                    ratio = np.clip(np.log(safe_ships) / np.log(1000.0), 0.0, 1.0)
                    speed = 1.0 + 5.0 * (ratio ** 1.5)
                    speed = np.where(r_ships <= 1.0, 1.0, speed)
                    
                    ticks_flown = tick - r_ticks
                    
                    # Approximated original source coordinates
                    # (We use current_x as an approximation since orbital drift over <200 ticks is small)
                    src_x = np.zeros_like(r_src, dtype=np.float32)
                    src_y = np.zeros_like(r_src, dtype=np.float32)
                    for i, sid in enumerate(r_src):
                        if sid < n_planets:
                            src_x[i] = current_x[sid]
                            src_y[i] = current_y[sid]
                            
                    f_x = src_x + ticks_flown * speed * np.cos(r_angles)
                    f_y = src_y + ticks_flown * speed * np.sin(r_angles)
                    
                    f_active_mask = np.zeros(len(r_targets), dtype=bool)
                    f_etas = np.full(len(r_targets), np.inf, dtype=np.float32)
                    
                    # For each fleet, check if it's still flying to its target
                    for i, tgt in enumerate(r_targets):
                        if tgt != -1 and tgt < 50:
                            tgt_x = current_x[tgt]
                            tgt_y = current_y[tgt]
                            
                            # Use exact 2-step Newton intercept time
                            tgt_init_x = init_x[tgt]
                            tgt_init_y = init_y[tgt]
                            tgt_is_orb = is_static[tgt] == 0 and is_comet[tgt] == 0
                            
                            t0 = np.hypot(tgt_x - f_x[i], tgt_y - f_y[i]) / speed[i]
                            
                            dx_orb = tgt_init_x - 50.0
                            dy_orb = tgt_init_y - 50.0
                            orb_r = np.hypot(dx_orb, dy_orb)
                            cur_angle = np.arctan2(dy_orb, dx_orb) + angular_vel * tick
                            
                            a1 = cur_angle + angular_vel * t0
                            px1 = 50.0 + orb_r * np.cos(a1) if tgt_is_orb else tgt_x
                            py1 = 50.0 + orb_r * np.sin(a1) if tgt_is_orb else tgt_y
                            t1 = np.hypot(px1 - f_x[i], py1 - f_y[i]) / speed[i]
                            
                            a2 = cur_angle + angular_vel * t1
                            px2 = 50.0 + orb_r * np.cos(a2) if tgt_is_orb else tgt_x
                            py2 = 50.0 + orb_r * np.sin(a2) if tgt_is_orb else tgt_y
                            
                            path_blocked = check_sun_collision(f_x[i], f_y[i], px2, py2)
                            eta = np.inf if path_blocked else np.hypot(px2 - f_x[i], py2 - f_y[i]) / speed[i]
                            
                            # Has it arrived? Compare the elapsed distance against distance to intercept
                            orig_dist = np.hypot(px2 - src_x[i], py2 - src_y[i])
                            if ticks_flown * speed[i] < orig_dist:
                                f_active_mask[i] = True
                                f_etas[i] = eta
                                # Still active
                                f_owner = float(r_slots[i])
                                if f_owner == -1.0:
                                    ego_f_owner_val = 0.0
                                elif f_owner == float(winner_slot):
                                    ego_f_owner_val = 1.0
                                elif f_owner < float(winner_slot):
                                    ego_f_owner_val = f_owner + 2.0
                                else:
                                    ego_f_owner_val = f_owner + 1.0
                                    
                                if ego_f_owner_val == 1.0:
                                    inc_1[tgt] += r_ships[i]
                                elif ego_f_owner_val == 2.0:
                                    inc_2[tgt] += r_ships[i]
                                elif ego_f_owner_val == 3.0:
                                    inc_3[tgt] += r_ships[i]
                                elif ego_f_owner_val == 4.0:
                                    inc_4[tgt] += r_ships[i]
                                    
                                is_attacker = (ego_f_owner_val != ego_owner[tgt]) or (ego_owner[tgt] == 0.0)
                                if is_attacker and eta < min_attacker_eta[tgt]:
                                    min_attacker_eta[tgt] = eta
                                        
                    # Vectorized Timeline Simulation for Deep Game Math
                    if np.any(f_active_mask):
                        target_mask = (r_targets[:, None] == np.arange(50)[None, :]) # [F, 50]
                        active_target_mask = target_mask * f_active_mask[:, None] # [F, 50]
                        
                        eta_matrix = np.where(active_target_mask, f_etas[:, None], np.inf) # [F, 50]
                        
                        is_planet_neutral = (ego_owner == 0.0)[None, :]
                        is_same_owner = (ego_f_owner[:, None] == ego_owner[None, :]) & ~is_planet_neutral
                        fleet_impact_matrix = np.where(is_same_owner, r_ships[:, None], -r_ships[:, None]) * active_target_mask # [F, 50]
                        
                        time_mask = eta_matrix[:, None, :] <= eta_matrix[None, :, :] # [F, F, 50]
                        past_impacts = np.sum(fleet_impact_matrix[:, None, :] * time_mask, axis=0) # [F, 50]
                        
                        padded_prod = np.zeros(50)
                        padded_prod[:n_planets] = static_planets['production'][:n_planets]
                        padded_raw = np.zeros(50)
                        padded_raw[:n_planets] = raw_ships[:n_planets]
                        
                        G_matrix = padded_raw[None, :] + padded_prod[None, :] * eta_matrix + past_impacts
                        
                        is_valid_eval = active_target_mask & ~is_same_owner
                        valid_G = np.where(is_valid_eval, G_matrix, np.inf)
                        min_G = np.min(valid_G, axis=0) # [50]
            except Exception:
                pass
                
        # Normalize features
        planet_tensors[:, 14] = np.log1p(inc_1) / 7.0
        planet_tensors[:, 15] = np.log1p(inc_2) / 7.0
        planet_tensors[:, 16] = np.log1p(inc_3) / 7.0
        planet_tensors[:, 17] = np.log1p(inc_4) / 7.0
        
        inv_attacker_eta = np.where(min_attacker_eta < np.inf, 1.0 / (1.0 + min_attacker_eta / 10.0), 0.0)
        planet_tensors[:, 18] = inv_attacker_eta
        
        # Feature 19 & 20: Quadrant Binary Encodings
        q_x = (current_x[:n_planets] > 50.0).astype(np.float32)
        q_y = (current_y[:n_planets] > 50.0).astype(np.float32)
        endgame_factor = np.clip((500 - tick) / 100.0, 0.0, 1.0)
        # Note: endgame_factor is embedded into true_remaining_yield (Feature 30) later
        
        # Feature 31: Orbital Gravity PhaseCountdown Clock
        wave_idx = tick // 100
        next_spawn = (wave_idx * 100) + 50
        if tick > next_spawn:
            next_spawn += 100
        ticks_to_spawn = max(0, next_spawn - tick)
        
        planet_tensors[:n_planets, 19] = q_x
        planet_tensors[:n_planets, 20] = q_y
        
        # Feature 21: Comet Spawn Countdown Clock
        comet_urgency = 1.0 - (ticks_to_spawn / 100.0) 
        comet_urgency = 0.0 if tick > 450 else comet_urgency
        planet_tensors[:n_planets, 21] = comet_urgency
        
        # Feature 22 & 23: Global Economic & Military Share
        incoming_allied = inc_1
        incoming_enemy = inc_2 + inc_3 + inc_4
        
        my_ships_total = np.sum(raw_ships[:n_planets][ego_owner == 1.0]) + np.sum(incoming_allied)
        all_ships_total = np.sum(raw_ships[:n_planets]) + np.sum(incoming_allied) + np.sum(incoming_enemy) + 1e-8
        my_ship_share = my_ships_total / all_ships_total
        
        my_prod_total = np.sum(static_planets['production'][:n_planets][ego_owner == 1.0])
        all_prod_total = np.sum(static_planets['production'][:n_planets]) + 1e-8
        my_prod_share = my_prod_total / all_prod_total

        planet_tensors[:n_planets, 22] = my_ship_share
        planet_tensors[:n_planets, 23] = my_prod_share
        
        # FEATURE 24: True Capture Cost (Net Garrison) via Timeline Simulation
        true_capture_cost = np.where(min_G[:n_planets] == np.inf, raw_ships[:n_planets], min_G[:n_planets])
        planet_tensors[:n_planets, 24] = np.clip(true_capture_cost / 100.0, -1.0, 1.0)
        
        # Feature 25: Enemy Proximity
        enemy_planet_mask = (ego_owner >= 2.0)
        if np.any(enemy_planet_mask):
            e_x = current_x[:n_planets][enemy_planet_mask]
            e_y = current_y[:n_planets][enemy_planet_mask]
            
            # Use exact 2-step Newton intercept matrix for speed = 6.0
            tgt_is_orb = ((is_static[:n_planets] == 0) & (is_comet[:n_planets] == 0)).astype(np.float32)[:, None]
            dx_matrix = current_x[:n_planets, None] - e_x[None, :]
            dy_matrix = current_y[:n_planets, None] - e_y[None, :]
            t0_23 = np.hypot(dx_matrix, dy_matrix) / 6.0
            
            dx_orb_23 = init_x[:n_planets, None] - 50.0
            dy_orb_23 = init_y[:n_planets, None] - 50.0
            orb_r_23 = np.hypot(dx_orb_23, dy_orb_23)
            cur_angle_23 = np.arctan2(dy_orb_23, dx_orb_23) + angular_vel * tick
            
            a1_23 = cur_angle_23 + angular_vel * t0_23
            px1_23 = np.where(tgt_is_orb, 50.0 + orb_r_23 * np.cos(a1_23), current_x[:n_planets, None])
            py1_23 = np.where(tgt_is_orb, 50.0 + orb_r_23 * np.sin(a1_23), current_y[:n_planets, None])
            t1_23 = np.hypot(px1_23 - e_x[None, :], py1_23 - e_y[None, :]) / 6.0
            
            a2_23 = cur_angle_23 + angular_vel * t1_23
            px2_23 = np.where(tgt_is_orb, 50.0 + orb_r_23 * np.cos(a2_23), current_x[:n_planets, None])
            py2_23 = np.where(tgt_is_orb, 50.0 + orb_r_23 * np.sin(a2_23), current_y[:n_planets, None])
            
            path_blocked_23 = check_sun_collision(current_x[:n_planets, None], current_y[:n_planets, None], px2_23, py2_23)
            t2_23 = np.where(path_blocked_23, np.inf, np.hypot(px2_23 - e_x[None, :], py2_23 - e_y[None, :]) / 6.0)
            
            min_transit_ticks = np.min(t2_23, axis=1)
            p_frontline = np.clip(1.0 - (min_transit_ticks / 20.0), 0.0, 1.0)
            planet_tensors[:n_planets, 25] = p_frontline
        else:
            min_transit_ticks = np.full(n_planets, 50.0)
            planet_tensors[:n_planets, 25] = 0.0
            
        # Feature 26: The "Sun Shadow" (Occlusion Mask / Vulnerability)
        dist_to_sun = np.hypot(current_x[:n_planets] - 50.0, current_y[:n_planets] - 50.0)
        safe_dist = np.maximum(dist_to_sun, 10.1) # Prevent arcsin > 1
        angular_width = 2.0 * np.arcsin(10.0 / safe_dist)
        planet_tensors[:n_planets, 26] = angular_width / np.pi  # FIX: Was 24, overwrote True Capture Cost
        
        # FEATURE 27: Threat Density (15-Tick Horizon)
        if np.any(enemy_planet_mask):
            e_ships = raw_ships[:n_planets][enemy_planet_mask]
            
            # Use exact Newton intercept for speed = 4.0
            t0_25 = np.hypot(dx_matrix, dy_matrix) / 4.0
            a1_25 = cur_angle_23 + angular_vel * t0_25
            px1_25 = np.where(tgt_is_orb, 50.0 + orb_r_23 * np.cos(a1_25), current_x[:n_planets, None])
            py1_25 = np.where(tgt_is_orb, 50.0 + orb_r_23 * np.sin(a1_25), current_y[:n_planets, None])
            t1_25 = np.hypot(px1_25 - e_x[None, :], py1_25 - e_y[None, :]) / 4.0
            
            a2_25 = cur_angle_23 + angular_vel * t1_25
            px2_25 = np.where(tgt_is_orb, 50.0 + orb_r_23 * np.cos(a2_25), current_x[:n_planets, None])
            py2_25 = np.where(tgt_is_orb, 50.0 + orb_r_23 * np.sin(a2_25), current_y[:n_planets, None])
            
            path_blocked_25 = check_sun_collision(current_x[:n_planets, None], current_y[:n_planets, None], px2_25, py2_25)
            t2_25 = np.where(path_blocked_25, np.inf, np.hypot(px2_25 - e_x[None, :], py2_25 - e_y[None, :]) / 4.0)
            
            can_reach_mask = (t2_25 <= 15.0).astype(np.float32)
            local_threat_density = np.sum(can_reach_mask * e_ships[None, :], axis=1)
        else:
            local_threat_density = np.zeros(n_planets)

        planet_tensors[:n_planets, 27] = np.clip(local_threat_density / 200.0, 0.0, 1.0)
        
        # FEATURE 28: Economic Momentum (Net Ship Delta per Tick)
        my_prod = np.sum(static_planets['production'][:n_planets][ego_owner == 1.0])
        enemy_prod = np.sum(static_planets['production'][:n_planets][ego_owner >= 2.0])
        
        net_production_advantage = my_prod - enemy_prod
        planet_tensors[:n_planets, 28] = np.clip(net_production_advantage / 25.0, -1.0, 1.0)
        
        # FEATURE 29: Angular Convergence
        is_orbiting = ((is_static[:n_planets] == 0) & (is_comet[:n_planets] == 0)).astype(np.float32)
        
        dx_orb = current_x[:n_planets] - 50.0
        dy_orb = current_y[:n_planets] - 50.0
        
        orb_vx = -dy_orb * angular_vel
        orb_vy = dx_orb * angular_vel
        
        to_me_x = (my_cx + 50.0) - current_x[:n_planets]
        to_me_y = (my_cy + 50.0) - current_y[:n_planets]
        
        mag_v = np.hypot(orb_vx, orb_vy) + 1e-8
        mag_me = np.hypot(to_me_x, to_me_y) + 1e-8
        
        convergence = ((orb_vx * to_me_x) + (orb_vy * to_me_y)) / (mag_v * mag_me)
        planet_tensors[:n_planets, 29] = convergence * is_orbiting
        
        # FEATURE 30: The Endgame Horizon (True Remaining Yield)
        remaining_ticks_after_arrival = np.maximum(0.0, 500.0 - tick - min_transit_ticks)
        true_remaining_yield = remaining_ticks_after_arrival * static_planets['production'][:n_planets]
        planet_tensors[:n_planets, 30] = true_remaining_yield / 2500.0
        
        # FEATURE 31: Logistical Gravity (Territory Control)
        my_prod_array = static_planets['production'][:n_planets] * (ego_owner == 1.0)
        enemy_prod_array = static_planets['production'][:n_planets] * (ego_owner >= 2.0)
        dx_all = current_x[:n_planets, None] - current_x[:n_planets][None, :]
        dy_all = current_y[:n_planets, None] - current_y[:n_planets][None, :]
        dist_all = np.hypot(dx_all, dy_all) + 1.0 
        my_gravity = np.sum(my_prod_array[None, :] / dist_all, axis=1)
        enemy_gravity = np.sum(enemy_prod_array[None, :] / dist_all, axis=1)
        planet_tensors[:n_planets, 31] = np.clip((my_gravity - enemy_gravity) / 10.0, -1.0, 1.0)
        
        # FEATURE 32: The "Kingmaker" (Leader Mask)
        ego_f_owner = np.where(
            r_slots == -1.0, 0.0,
            np.where(
                r_slots == float(winner_slot), 1.0,
                np.where(r_slots < float(winner_slot), r_slots + 2.0, r_slots + 1.0)
            )
        ) if len(r_slots) > 0 else np.array([])
        
        p_ships = np.zeros(5)
        enemy_ids = np.zeros(5)
        for p in range(1, 5):
            fleet_ships_for_p = np.sum(r_ships[ego_f_owner == float(p)]) if len(r_slots) > 0 else 0.0
            p_ships[p] = np.sum(raw_ships[:n_planets][ego_owner == float(p)]) + fleet_ships_for_p
            enemy_ids[p] = float(p)
        enemy_ships_arr = p_ships[2:5]
        if np.max(enemy_ships_arr) > 0:
            leader_id = enemy_ids[np.argmax(enemy_ships_arr) + 2] if np.max(enemy_ships_arr) > 0 else 0.0
        else:
            leader_id = 0.0
        is_leader = (ego_owner == leader_id).astype(np.float32)
        planet_tensors[:n_planets, 32] = is_leader
        
        # FEATURE 33: Safe Surplus (Exportable Economy)
        safe_surplus = np.maximum(0.0, np.minimum(raw_ships[:n_planets], min_G[:n_planets]))
        planet_tensors[:n_planets, 33] = (safe_surplus / 100.0) * (ego_owner == 1.0)
        
        # FEATURE 34: The Evacuation Protocol (Doomed Planet Mask)
        deficit = np.where(min_G[:n_planets] < 0, -min_G[:n_planets], 0.0)
        is_doomed_by_combat = (deficit > 0)
        evacuate_mask = is_doomed_by_combat & (ego_owner == 1.0)
        planet_tensors[:n_planets, 34] = evacuate_mask.astype(np.float32)
        
        # FEATURE 35: The Cry For Help (Local Deficit Heatmap)
        my_planets_mask = (ego_owner == 1.0)
        deficit_allied = deficit * my_planets_mask
        if np.any(my_planets_mask):
            close_allies_mask = (dist_all <= 40.0) * my_planets_mask[None, :]
            local_allied_deficit = np.sum(close_allies_mask * deficit_allied[None, :], axis=1)
        else:
            local_allied_deficit = np.zeros(n_planets)
        planet_tensors[:n_planets, 35] = np.clip(local_allied_deficit / 100.0, 0.0, 1.0)
        
        # Padding
        planet_tensors[:n_planets, 36] = 0.0
        
        output = {
            "state_tokens": planet_tensors.astype(np.float32), # Now [50, 35]
            "winner": np.array(winner_slot, dtype=np.int32),
            "tick": np.array(tick, dtype=np.int32),
            "win_rate": np.array(win_rate, dtype=np.float32) # Export explicitly!
        }
        
        # 3. IQL Targets
        if self.is_iql_mode:
            target_launch = np.zeros(50, dtype=np.float32).reshape(50)
            target_angle = np.full(50, -1, dtype=np.int32).reshape(50)
            target_ships = np.full(50, -1, dtype=np.int32).reshape(50)
            
            # THE FIX: Check against 1.0 (the egocentric 'Me' token), NOT winner_slot
            is_owned_by_winner = (planet_tensors[:50, 5] == 1.0).astype(np.float32)
            
            if episode_id in self.actions_dict_np:
                try:
                    ep_actions = self.actions_dict_np[episode_id]
                    ticks = ep_actions['ticks']
                    
                    # --- TARGET LABELS ---
                    future_start = np.searchsorted(ticks, tick, side='left')
                    future_end = np.searchsorted(ticks, tick + self.subsample_ticks, side='left')
                    
                    if future_end > future_start:
                        f_slots = ep_actions['slot'][future_start:future_end]
                        f_src = ep_actions['src_planet_id'][future_start:future_end]
                        f_ships = ep_actions['n_ships'][future_start:future_end]
                        f_angles = ep_actions['angle'][future_start:future_end]
                        f_targets = ep_actions['target_planet_id'][future_start:future_end]
                        
                        my_launches = (f_slots == winner_slot)
                        
                        if np.any(my_launches):
                            my_src = f_src[my_launches]
                            my_ships = f_ships[my_launches]
                            my_angles = f_angles[my_launches]
                            my_targets = f_targets[my_launches]
                            
                            for sid, ships, angle, target in zip(my_src, my_ships, my_angles, my_targets):
                                pid = int(sid)
                                if pid < 50:
                                    target_launch[pid] = 1.0
                                    
                                    best_target = None
                                    if target != -1:
                                        try:
                                            best_target = int(np.where(static_planets['planet_ids'] == target)[0][0])
                                            # CRITICAL FIX: Ignore self-launches which crash the Pointer Network
                                            if best_target == pid:
                                                best_target = None
                                        except IndexError:
                                            pass
                                            
                                    if best_target is None:
                                        # Fast orbit_env style calculation
                                        dx = current_x[:n_planets] - current_x[pid]
                                        dy = current_y[:n_planets] - current_y[pid]
                                        planet_angles = np.arctan2(dy, dx)
                                        diff = np.abs(planet_angles - angle)
                                        diff = np.minimum(diff, 2 * np.pi - diff)
                                        if pid < len(diff):  # FIX: Prevent IndexError for comet launches (pid >= n_planets)
                                            diff[pid] = 1e9 # Mask self
                                        inferred = int(np.argmin(diff))
                                        if diff[inferred] < 0.1: # Strict 5.7 degree tolerance
                                            best_target = inferred
                                        
                                    if best_target is None:
                                        target_angle[pid] = -1
                                    else:
                                        target_angle[pid] = best_target
                                        
                                    garrison = raw_ships[pid] if pid < n_planets else ships
                                    fraction = ships / max(1, garrison)
                                    ships_bin = int(fraction * 10)
                                    target_ships[pid] = np.clip(ships_bin, 0, 9)
                except Exception:
                    pass
            
            output["target_launch"] = target_launch.astype(np.float32)
            output["target_angle"] = target_angle.astype(np.int32)
            output["target_ships"] = target_ships.astype(np.int32)
            output["is_owned_by_winner"] = is_owned_by_winner.astype(np.float32)
            
        return output

def build_grain_dataloader(db_path="parquet_db_real", batch_size=256, worker_count=4, grandmaster_only=True, subsample_ticks=5, is_iql_mode=True, num_epochs=None):
    if not os.path.isabs(db_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates = [
            os.path.join(repo_root, db_path),
            os.path.join(repo_root, "parquet_db_real"),
            os.path.join(repo_root, "data", "parquet_db_real"),
        ]
        resolved = None
        for candidate in candidates:
            if os.path.exists(candidate):
                resolved = candidate
                break
        db_path = resolved if resolved is not None else os.path.abspath(os.path.join(repo_root, db_path))
        print(f"Resolved dataset path to {db_path}")

    source = OrbitWarsDataSource(db_path=db_path, grandmaster_only=grandmaster_only, subsample_ticks=subsample_ticks, is_iql_mode=is_iql_mode)
    
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=num_epochs,
        shard_options=grain.NoSharding(),
        shuffle=True,
        seed=42
    )
    
    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)],
        worker_count=worker_count
    )
    
    return dataloader

if __name__ == "__main__":
    print("Testing Grain DataLoader...")
    dl = build_grain_dataloader(batch_size=4, worker_count=1)
    
    for batch in dl:
        print("Batch received via Grain:")
        print("State shape:", batch["state_tokens"].shape)
        print("Winner shape:", batch["winner"].shape)
        if "target_launch" in batch:
            print("Target Launch shape:", batch["target_launch"].shape)
        break
    print("Grain DataLoader works perfectly!")
