import os
import json
import math
import pathlib
import time
import zipfile
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SUN_X, SUN_Y = 50.0, 50.0

def get_episode_id(stem):
    m = re.match(r'episode-(\d+)', stem)
    if m: return int(m.group(1))
    m = re.match(r'^(\d+)$', stem)
    if m: return int(m.group(1))
    return hash(stem) % 100000

def parse_episode(json_str, dataset_label, stem):
    try:
        data = json.loads(json_str)
    except:
        return None

    rewards = data.get('rewards', [])
    steps = data.get('steps', [])
    info = data.get('info', {})
    agents = info.get('TeamNames', [])
    n_players = len(rewards)
    n_steps = len(steps)

    if n_steps < 2 or n_players < 2:
        return None

    episode_id = get_episode_id(stem)
    
    valid_rewards = [r for r in rewards if r is not None]
    if valid_rewards:
        max_r = max(valid_rewards)
        winners = [i for i, r in enumerate(rewards) if r == max_r]
    else:
        winners = [0]
    winner = winners[0] if len(winners) == 1 else -1

    obs0 = steps[0][0].get('observation', {})
    av = float(obs0.get('angular_velocity', 0.0))
    comet_ids = set(obs0.get('comet_planet_ids', []))
    init_planets = obs0.get('initial_planets', obs0.get('planets', []))

    episode_row = {
        'episode_id': episode_id, 'dataset': dataset_label,
        'n_players': n_players, 'n_steps': n_steps,
        'seed': info.get('seed', 0), 'angular_velocity': av,
        'n_planets': len(init_planets), 'n_comets': len(comet_ids),
        'winner_slot': winner,
    }

    player_rows = []
    for slot in range(n_players):
        name = agents[slot] if slot < len(agents) else f'player_{slot}'
        player_rows.append({
            'episode_id': episode_id, 'slot': slot, 'name': name,
            'reward': float(rewards[slot] or 0), 'is_winner': 1 if slot == winner else 0,
        })

    ep_planet_rows = []
    for p in init_planets:
        pid, owner, x, y, r, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        orbit_r = math.sqrt((x - SUN_X)**2 + (y - SUN_Y)**2)
        is_static = (orbit_r + r >= 50.0) or (av == 0)
        ep_planet_rows.append({
            'episode_id': episode_id, 'planet_id': pid,
            'initial_x': x, 'initial_y': y, 'radius': r,
            'production': prod, 'orbit_radius': orbit_r,
            'is_static': is_static, 'is_comet': pid in comet_ids,
            'initial_ships': ships, 'initial_owner': owner,
        })

    tick_rows, action_rows = [], []
    ps_eid, ps_tick, ps_pid, ps_owner, ps_ships = [], [], [], [], []

    for tick in range(n_steps):
        step = steps[tick]
        if not isinstance(step, list) or not step:
            continue
        obs = step[0].get('observation', {})
        planets = obs.get('planets', [])
        fleets = obs.get('fleets', [])

        for slot in range(n_players):
            sp = sum(p[5] for p in planets if p[1] == slot)
            sf = sum(f[6] for f in fleets  if f[1] == slot)
            tick_rows.append({
                'episode_id': episode_id, 'tick': tick, 'slot': slot,
                'ships_planets': sp, 'ships_fleets': sf, 'total_ships': sp + sf,
                'production': sum(p[6] for p in planets if p[1] == slot),
                'n_planets':  sum(1   for p in planets if p[1] == slot),
                'n_fleets':   sum(1   for f in fleets  if f[1] == slot),
            })
            if tick >= 1 and slot < len(step):
                for a in (step[slot].get('action', []) or []):
                    if isinstance(a, (list, tuple)) and len(a) >= 3:
                        action_rows.append({
                            'episode_id': episode_id, 'tick': tick, 'slot': slot,
                            'src_planet_id': int(a[0]),
                            'angle': float(a[1]), 'n_ships': int(a[2]),
                        })

        for p in planets:
            ps_eid.append(episode_id); ps_tick.append(tick)
            ps_pid.append(p[0]); ps_owner.append(p[1]); ps_ships.append(p[5])

    return {
        'episode': episode_row,
        'players': player_rows,
        'ticks': tick_rows,
        'actions': action_rows,
        'episode_planets': ep_planet_rows,
        'planet_state': {'episode_id': ps_eid, 'tick': ps_tick,
                         'planet_id': ps_pid, 'owner': ps_owner, 'ships': ps_ships},
    }

SCHEMAS = {
    'episodes': pa.schema([
        ('episode_id', pa.int32()), ('dataset', pa.string()),
        ('n_players', pa.int8()), ('n_steps', pa.int16()),
        ('seed', pa.int32()), ('angular_velocity', pa.float32()),
        ('n_planets', pa.int8()), ('n_comets', pa.int8()), ('winner_slot', pa.int8()),
    ]),
    'player_episodes': pa.schema([
        ('episode_id', pa.int32()), ('slot', pa.int8()), ('name', pa.string()),
        ('reward', pa.float32()), ('is_winner', pa.int8()),
    ]),
    'tick_summary': pa.schema([
        ('episode_id', pa.int32()), ('tick', pa.int16()), ('slot', pa.int8()),
        ('ships_planets', pa.int32()), ('ships_fleets', pa.int32()),
        ('total_ships', pa.int32()), ('production', pa.int16()),
        ('n_planets', pa.int8()), ('n_fleets', pa.int16()),
    ]),
    'actions': pa.schema([
        ('episode_id', pa.int32()), ('tick', pa.int16()), ('slot', pa.int8()),
        ('src_planet_id', pa.int16()), ('angle', pa.float32()), ('n_ships', pa.int32()),
    ]),
    'episode_planets': pa.schema([
        ('episode_id', pa.int32()), ('planet_id', pa.int16()),
        ('initial_x', pa.float32()), ('initial_y', pa.float32()),
        ('radius', pa.float32()), ('production', pa.int8()),
        ('orbit_radius', pa.float32()), ('is_static', pa.bool_()),
        ('is_comet', pa.bool_()), ('initial_ships', pa.int16()), ('initial_owner', pa.int8()),
    ]),
    'planet_state': pa.schema([
        ('episode_id', pa.int32()), ('tick', pa.int16()),
        ('planet_id', pa.int16()), ('owner', pa.int8()), ('ships', pa.int32()),
    ]),
}

import sys
if len(sys.argv) < 2:
    print("Usage: python build_parquet_from_zip.py <zip_file>")
    exit(1)
zip_path = sys.argv[1]
if not os.path.exists(zip_path):
    print(f"Zip file {zip_path} not found.")
    exit(1)

OUT_DIR = pathlib.Path(f'parquet_db_real/{pathlib.Path(zip_path).stem}')
OUT_DIR.mkdir(parents=True, exist_ok=True)
buffers = {k: [] for k in SCHEMAS}

def flush_all():
    for name, schema in SCHEMAS.items():
        if not buffers[name]:
            continue
        if name == 'planet_state':
            combined = {k: [] for k in ('episode_id','tick','planet_id','owner','ships')}
            for chunk in buffers[name]:
                for k in combined: combined[k].extend(chunk[k])
            df = pd.DataFrame(combined)
        else:
            df = pd.DataFrame(buffers[name])
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        out = OUT_DIR / f'{name}.parquet'
        if out.exists():
            table = pa.concat_tables([pq.read_table(out), table])
        pq.write_table(table, out, compression='snappy')
        buffers[name].clear()

# zip_path is defined above

# Set Max Episodes (None for all episodes)
MAX_EPISODES = None

print(f"Opening {zip_path} and streaming JSONs...")
t0 = time.time()
ok = skip = err = 0
FLUSH_EVERY = 50

with zipfile.ZipFile(zip_path, 'r') as z:
    json_files = [f for f in z.namelist() if f.endswith('.json')]
    print(f"Found {len(json_files)} json files in the zip.")
    
    files_to_process = json_files[:MAX_EPISODES]
    print(f'Processing {len(files_to_process)} episodes...')
    
    for i, json_filename in enumerate(files_to_process):
        try:
            with z.open(json_filename) as f:
                json_bytes = f.read()
            
            stem = pathlib.Path(json_filename).stem
            result = parse_episode(json_bytes, "2026-06-05", stem)
            if result is None:
                skip += 1; continue
            buffers['episodes'].append(result['episode'])
            buffers['player_episodes'].extend(result['players'])
            buffers['tick_summary'].extend(result['ticks'])
            buffers['actions'].extend(result['actions'])
            buffers['episode_planets'].extend(result['episode_planets'])
            buffers['planet_state'].append(result['planet_state'])
            ok += 1
        except Exception as e:
            err += 1
            if err <= 3: print(f'  ERROR {json_filename}: {e}')

        if (i + 1) % FLUSH_EVERY == 0:
            flush_all()
            elapsed = time.time() - t0
            rate = ok / elapsed if elapsed > 0 else 0
            print(f'  [{i+1:>4}/{len(files_to_process)}] ok={ok} skip={skip} err={err}'
                  f'  ({elapsed:.0f}s, {rate:.1f} ep/s)')

flush_all()
elapsed = time.time() - t0
print(f'\nDone: {ok} episodes in {elapsed:.0f}s ({ok/elapsed:.1f} ep/s)')
print(f'Skipped: {skip}  Errors: {err}')
print()
for name in SCHEMAS:
    out = OUT_DIR / f'{name}.parquet'
    if out.exists():
        n = len(pq.read_table(out, columns=[pq.read_schema(out).names[0]]))
        size_mb = out.stat().st_size / 1e6
        print(f'  {name+".parquet":<30} {n:>10,} rows  {size_mb:>7.1f} MB')
