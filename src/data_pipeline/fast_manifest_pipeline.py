import os
import shutil
import kagglehub

def main():
    target_dir = "/home/medhasree_2121cs05/2201cs50_nagesh/server_deploy/parquet_db_real"
    os.makedirs(target_dir, exist_ok=True)
    
    print("Downloading nbridelancetb/orbit-wars-replay-parquet...")
    path = kagglehub.dataset_download("nbridelancetb/orbit-wars-replay-parquet")
    print(f"Downloaded to {path}")
    
    print(f"Moving files to {target_dir}...")
    for filename in os.listdir(path):
        src = os.path.join(path, filename)
        dst = os.path.join(target_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"Copied {filename}")
            
    print("Dataset successfully set up!")

if __name__ == "__main__":
    main()