import os
import re
import json
import time

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def update_config(config_file, updates, phase_name):
    try:
        with open(config_file, 'r') as cf:
            config = json.load(cf)
            
        for k, v in updates.items():
            config[k] = v
            
        # Write to a temp file first to prevent PPO from reading a half-written JSON
        tmp_file = config_file + ".tmp"
        with open(tmp_file, 'w') as cf:
            json.dump(config, cf, indent=4)
        os.replace(tmp_file, config_file)
            
        print(f"\n[✓] Successfully updated config for {phase_name}!")
    except Exception as e:
        print(f"[!] Error updating config: {e}")

def main():
    # Adjust paths if your directory structure differs
    log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ppo_v2.log"))
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "ppo_reward_config.json"))
    
    while not os.path.exists(log_file):
        print(f"Waiting for {log_file} to be created...")
        time.sleep(5)

    print("===================================================")
    print("  ORBIT WARS: AUTONOMOUS REWARD CURRICULUM DAEMON  ")
    print("===================================================")
    print("Current State: PHASE 1 (The Defibrillator)")
    print("Waiting for Agent to learn basic launches and captures...\n")
    
    # Regex Parsers
    rate_regex = re.compile(r"Fleet Launch Rate:\s*([\d\.]+)%")
    peak_regex = re.compile(r"Peak Planets Held:\s*([\d\.]+)/50\.0")
    win_regex = re.compile(r"Epoch Win Rate:\s*([\d\.]+)%")
    margin_regex = re.compile(r"Net Ship Margin:\s*([\+\-\d\.]+)")
    
    # History Buffers (Requires 3 consecutive epochs to trigger to ensure stability)
    history = {'rate': [], 'peak': [], 'win': [], 'margin': []}
    required_epochs = 3
    
    current_phase = 1

    with open(log_file, 'r') as f:
        # Jump to the end of the file so we only read new output
        f.seek(0, os.SEEK_END)
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(1.0)
                continue
                
            clean_line = strip_ansi(line)
            
            # --- PARSE METRICS ---
            m_rate = rate_regex.search(clean_line)
            m_peak = peak_regex.search(clean_line)
            m_win = win_regex.search(clean_line)
            m_margin = margin_regex.search(clean_line)
            
            if m_rate: history['rate'].append(float(m_rate.group(1)))
            if m_peak: history['peak'].append(float(m_peak.group(1)))
            if m_win: history['win'].append(float(m_win.group(1)))
            if m_margin: history['margin'].append(float(m_margin.group(1)))
            
            # Keep buffers at max length
            for k in history:
                if len(history[k]) > required_epochs:
                    history[k].pop(0)
            
            # Only evaluate if we have a full buffer of metrics
            if all(len(v) == required_epochs for v in history.values()):
                avg_rate = sum(history['rate']) / required_epochs
                avg_peak = sum(history['peak']) / required_epochs
                avg_win = sum(history['win']) / required_epochs
                avg_margin = sum(history['margin']) / required_epochs
                
                # ==============================================================
                # PHASE 1 -> PHASE 2 TRANSITION
                # ==============================================================
                if current_phase == 1:
                    # Targets: Launch > 10%, Planets > 2.5
                    if avg_rate >= 10.0 and avg_peak >= 2.5:
                        print("\n*** PHASE 1 CLEAR: BASIC BEHAVIOR EMERGED ***")
                        print(f"Metrics | Rate: {avg_rate:.1f}% | Peak: {avg_peak:.1f}")
                        print("Transitioning to PHASE 2: Force Conservation...")
                        
                        updates = {
                            "dense_fleet_activity": 0.0,
                            "dense_no_op_penalty": 0.0,
                            "dense_planet_holding": 0.0,
                            "dense_planet_capture_delta": 0.5,
                            "dense_production_share": 0.2
                        }
                        update_config(config_file, updates, "PHASE 2")
                        current_phase = 2
                        
                        # Clear buffers so we don't instantly trigger the next phase
                        history = {k: [] for k in history}
                        print("\nWaiting for Agent to learn Force Conservation (Margin > -100, Win > 45%)...")

                # ==============================================================
                # PHASE 2 -> PHASE 3 TRANSITION
                # ==============================================================
                elif current_phase == 2:
                    # Targets: Ship Margin > -100, Win Rate > 45%
                    if avg_margin >= -100.0 and avg_win >= 55.0:
                        print("\n*** PHASE 2 CLEAR: TACTICAL COMPETENCE ACHIEVED ***")
                        print(f"Metrics | Margin: {avg_margin:.1f} | Win Rate: {avg_win:.1f}%")
                        print("Transitioning to PHASE 3: Pure Self-Play Meta...")
                        
                        updates = {
                            "dense_planet_capture_delta": 0.0,
                            "dense_production_share": 0.0,
                            "dense_ship_dominance": 0.0,
                            "terminal_dominance_bonus": 0.5,
                            "base_win_reward": 1.0,
                            "base_loss_reward": -1.0
                        }
                        update_config(config_file, updates, "PHASE 3")
                        
                        print("\n[DAEMON COMPLETE] Agent is now running in 100% sparse competitive mode.")
                        print("Shutting down daemon. Let the training run overnight!")
                        return

if __name__ == "__main__":
    main()
