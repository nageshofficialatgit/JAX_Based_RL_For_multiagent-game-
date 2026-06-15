import os
import json
import math
import pathlib
import time
import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zipfile
import gc

# Set input/output directories based on environment
INPUT_DIR = pathlib.Path('/DATA/tauseef_2121cs04/medhasree_temp/input')
OUT_DIR = pathlib.Path('/DATA/tauseef_2121cs04/medhasree_temp/working')

# Fallback for local windows/linux testing
if not INPUT_DIR.exists():
    INPUT_DIR = pathlib.Path('./input')
if not OUT_DIR.exists():
    OUT_DIR = pathlib.Path('./working')
    os.makedirs(OUT_DIR, exist_ok=True)

print(f'Input dir: {INPUT_DIR}')
print(f'Output dir: {OUT_DIR}')

def find_json_files(root_dir, last_n_days=10):
    all_zips = list(root_dir.rglob('*.zip'))
    print(f'Total .zip files found: {len(all_zips)}')
    
    by_day = {}
    for p in all_zips:
        # Expected format: orbit-wars-episodes-2026-06-05.zip
        m = re.search(r'(\d{4}-\d{2}-\d{2})', p.stem)
        if m:
            day = m.group(1)
            by_day.setdefault(day, []).append(p)
        else:
            day = p.parent.name
            by_day.setdefault(day, []).append(p)

    days = sorted(by_day.keys())
    print(f'All days found: {days}')
    
    # Restrict to last N days
    days = days[-last_n_days:]
    print(f'Selected last {last_n_days} days: {days}')
    
    result = []
    for d in days:
        for zip_path in sorted(by_day[d]):
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    for name in z.namelist():
                        if name.endswith('.json'):
                            result.append((zip_path, name, d))
            except zipfile.BadZipFile:
                print(f"Skipping corrupt zip: {zip_path}")
    return result, days

SUN_X, SUN_Y = 50.0, 50.0

def get_episode_id(name):
    path = pathlib.Path(name)
    stem = path.stem
    m = re.match(r'episode-(\d+)', stem)
    if m: return int(m.group(1))
    m = re.match(r'^(\d+)$', stem)
    if m: return int(m.group(1))
    raise ValueError(f'Cannot extract episode ID from {name}')

def parse_episode(zip_path, json_name, dataset_label):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(json_name, 'r') as f:
            data = json.loads(f.read())

    rewards = data.get('rewards', [])
    steps = data.get('steps', [])
    info = data.get('info', {})
    agents = info.get('Agents', [])
    n_players = len(rewards)
    n_steps = len(steps)

    if n_steps < 2 or n_players < 2:
        return None

    episode_id = get_episode_id(json_name)
    max_r = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == max_r]
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
        name = agents[slot]['Name'] if slot < len(agents) else f'player_{slot}'
        player_rows.append({
            'episode_id': episode_id, 'slot': slot, 'name': name,
            'reward': rewards[slot], 'is_winner': 1 if slot == winner else 0,
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

buffers = {k: [] for k in SCHEMAS}
writers = {}

def get_writer(name, schema):
    out = OUT_DIR / f'{name}.parquet'
    if name not in writers:
        if out.exists():
            # If exists, we should ideally append. Unfortunately, basic PyArrow ParquetWriter 
            # doesn't natively support appending to an existing file in-place safely without rewrites.
            # However, we can write chunked files (e.g. actions_0.parquet, actions_1.parquet)
            # OR we can just write one continuous stream for this script run.
            # Here we assume a fresh start or we overwrite for this specific Kaggle run.
            pass 
        writers[name] = pq.ParquetWriter(out, schema, compression='snappy')
    return writers[name]

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
        
        # Free up the buffer dictionary immediately to prevent Memory spike during Arrow conversion
        buffers[name].clear()
        
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        
        # MEMORY FIX: Use ParquetWriter to append the stream. 
        # This completely avoids loading historical data back into RAM.
        writer = get_writer(name, schema)
        writer.write_table(table)
        
        # Force garbage collection
        del df
        del table
    
    gc.collect()

def close_writers():
    for writer in writers.values():
        writer.close()

import subprocess

def main():
    print("Listing Kaggle datasets for 'orbit-wars-episodes'...")
    try:
        # Use kaggle CLI to list datasets, getting CSV output
        output = subprocess.check_output(
            ["orbit/bin/kaggle", "datasets", "list", "-s", "orbit-wars-episodes-2026", "-v"], 
            text=True
        )
    except subprocess.CalledProcessError as e:
        print("Error listing datasets:", e)
        return

    datasets = []
    lines = output.strip().split('\n')
    for line in lines[1:]: # skip header
        if not line: continue
        parts = line.split(',')
        if len(parts) >= 1:
            ref = parts[0]
            m = re.search(r'orbit-wars-episodes-(\d{4}-\d{2}-\d{2})', ref)
            if m:
                date_str = m.group(1)
                # Filter for dates between May 1st and May 15th
                if "2026-05-01" <= date_str <= "2026-05-15":
                    datasets.append((date_str, ref))
                
    # Sort by date descending
    datasets.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\nSelected {len(datasets)} datasets to download (2026-05-01 to 2026-05-15):")
    for date, ref in datasets:
        print(f"  {date}: {ref}")
        
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    t0 = time.time()
    ok = skip = err = 0

    for d, ref in datasets:
        print(f"\n--- Downloading {ref} ---")
        subprocess.run(["orbit/bin/kaggle", "datasets", "download", ref, "-p", str(INPUT_DIR)], check=True)
        
        # Expected zip path
        zip_name = ref.split('/')[-1] + '.zip'
        zip_path = INPUT_DIR / zip_name
        
        print(f"\nProcessing ZIP: {zip_path.name}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                json_names = [n for n in z.namelist() if n.endswith('.json')]
        except zipfile.BadZipFile:
            print(f"Skipping corrupt zip: {zip_path}")
            if zip_path.exists():
                os.remove(zip_path)
            continue
            
        eps_in_zip = 0
        for json_name in json_names:
            try:
                result = parse_episode(zip_path, json_name, d)
                if result is None:
                    skip += 1
                    continue
                buffers['episodes'].append(result['episode'])
                buffers['player_episodes'].extend(result['players'])
                buffers['tick_summary'].extend(result['ticks'])
                buffers['actions'].extend(result['actions'])
                buffers['episode_planets'].extend(result['episode_planets'])
                buffers['planet_state'].append(result['planet_state'])
                ok += 1
                eps_in_zip += 1
                
                # Flush every 1000 episodes to prevent RAM bloat on 1.44GB zips
                if len(buffers['episodes']) >= 1000:
                    flush_all()
                    
            except Exception as e:
                err += 1
                if err <= 3: print(f'  ERROR {json_name} in {zip_path.name}: {e}')
        
        # Flush any remaining rows in the buffer for this zip
        flush_all()
        
        print(f"Deleting {zip_path.name} to free disk space...")
        if zip_path.exists():
            os.remove(zip_path)

    close_writers()
    
    elapsed = time.time() - t0
    print(f'\nDone: {ok} episodes in {elapsed:.0f}s ({ok/elapsed if elapsed else 0:.1f} ep/s)')
    print(f'Skipped: {skip}  Errors: {err}\n')
    
    print('Output files written to:', OUT_DIR)
    for name in SCHEMAS:
        out = OUT_DIR / f'{name}.parquet'
        if out.exists():
            n = len(pq.read_table(out, columns=[pq.read_schema(out).names[0]]))
            size_mb = out.stat().st_size / 1e6
            print(f'  {name+".parquet":<30} {n:>10,} rows  {size_mb:>7.1f} MB')

if __name__ == '__main__':
    main()
