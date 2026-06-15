import os
import json
import re

import sys

def build_submission():
    if len(sys.argv) < 2:
        print("Usage: python build_submission.py <path_to_best_json>")
        return
        
    best_file = sys.argv[1]
    if not os.path.exists(best_file):
        print(f"Error: {best_file} not found.")
        return

    print(f"Building submission using weights from {best_file}")
    with open(best_file, "r") as f:
        best_params = json.load(f)

    # Read the agent file
    with open("braniac_v4.py", "r") as f:
        code = f.read()

    # Extract original EVAL_WEIGHTS to know which keys to replace
    # We use a simple evaluation trick or regex to replace the values
    
    for key, value in best_params.items():
        if key == "_avg_rank":
            continue
            
        # Try to replace in EVAL_WEIGHTS or CANDIDATE_WEIGHTS
        # Regex to find: "key": <number>,
        pattern = r'("' + key + r'"\s*:\s*)-?[0-9]*\.?[0-9]+'
        code, count = re.subn(pattern, r'\g<1>' + str(value), code)
        if count == 0:
            print(f"Warning: Could not find key '{key}' in source code to update.")

    with open("submission.py", "w") as f:
        f.write(code)

    print("Successfully built submission.py! You can now upload it to Kaggle.")

if __name__ == "__main__":
    build_submission()
