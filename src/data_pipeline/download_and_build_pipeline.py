import os
import subprocess
import pandas as pd
import time

def main():
    # Load manifest and select the LATEST 3 days (the last 3 rows of the file)
    df = pd.read_csv('manifest.csv')
    
    # THE EDIT: Grab the last 3 rows instead of the first 36
    target_days = df.tail(3)
    
    print(f"Starting pipeline for the latest {len(target_days)} days...")
    
    for idx, row in target_days.iterrows():
        date = row['date']
        slug = row['daily_dataset_slug']
        zip_filename = f"{slug}.zip"
        
        print(f"\n{'='*50}")
        print(f"Processing Data for {date}")
        print(f"{'='*50}")
        
        # Check if already processed
        expected_dir = f"parquet_db_real/{slug}"
        if os.path.exists(expected_dir):
            print(f"[{date}] Directory {expected_dir} already exists. Skipping to avoid duplicates.")
            continue
            
        # 1. Download using your environment's kaggle CLI
        print(f"[{date}] Downloading {slug}...")
        download_cmd = ["./orbit/bin/kaggle", "datasets", "download", "-d", f"kaggle/{slug}"]
        subprocess.run(download_cmd, check=True)
        
        # 2. Extract & Build Parquet
        print(f"[{date}] Building Parquet from {zip_filename}...")
        build_cmd = ["./orbit/bin/python", "build_parquet_from_zip.py", zip_filename]
        subprocess.run(build_cmd, check=True)
        
        # 3. Cleanup
        print(f"[{date}] Cleaning up {zip_filename}...")
        os.remove(zip_filename)
        print(f"[{date}] Done!")
        
    print("\nLatest target days processed successfully!")

if __name__ == "__main__":
    main()