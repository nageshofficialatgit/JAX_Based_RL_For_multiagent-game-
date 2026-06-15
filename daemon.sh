#!/bin/bash
# A simple daemon to wait for the BC training to finish, then start PPO training.
# Usage: ./daemon.sh <bc_pid>

BC_PID=$1

if [ -z "$BC_PID" ]; then
    echo "Usage: ./daemon.sh <bc_pid>"
    exit 1
fi

echo "Monitoring BC Spooler/Training process PID: $BC_PID..."

# Loop until the BC process finishes
while kill -0 $BC_PID 2> /dev/null; do
    sleep 60
done

echo ""
echo "============================================="
echo "BC Training Process ($BC_PID) has completed!"
echo "============================================="

echo "Waiting for process_replays.py to complete dataset extraction..."
# Loop until process_replays.py finishes
while pgrep -f process_replays.py > /dev/null; do
    sleep 60
done

echo ""
echo "============================================="
echo "Dataset extraction has completed!"
echo "Backing up checkpoints..."
echo "============================================="

BACKUP_DIR="checkpoints/backup/bc_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r checkpoints/bc_v2/* "$BACKUP_DIR/"
echo "Checkpoints backed up to $BACKUP_DIR"

echo "============================================="
echo "Initializing NEXT PHASE BC Training..."
echo "Backing up old dataset to prevent data loss..."
echo "Migrating fresh dataset from working directory to production..."
echo "============================================="

# Backup the old database and its cache instead of deleting them
OLD_DB_BACKUP="data_backup/parquet_db_real_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OLD_DB_BACKUP"
if [ -d "parquet_db_real" ]; then
    mv parquet_db_real/* "$OLD_DB_BACKUP/"
fi
if [ -d "data/parquet_db_real" ]; then
    mv data/parquet_db_real/* "$OLD_DB_BACKUP/"
fi
echo "Old dataset and cache safely moved to $OLD_DB_BACKUP"

# Move new dataset from the working extraction directory into the production DB
mkdir -p parquet_db_real
mv working/*.parquet parquet_db_real/

nohup env PYTHONUNBUFFERED=1 orbit/bin/python src/training/train_bc_v2.py > bc_training_v2_phase2.log 2>&1 &
NEXT_BC_PID=$!

echo "BC Phase 2 started successfully in the background! PID: $NEXT_BC_PID"
echo "You can check progress using: tail -f bc_training_v2_phase2.log"
