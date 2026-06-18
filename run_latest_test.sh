#!/bin/bash
source orbit/bin/activate
echo "Exporting latest checkpoint..."
python export_ckpt.py
echo "Running 1v1 test..."
JAX_PLATFORMS=cpu python run_1v1_test.py
echo "Done! Replay saved to ppo_vs_braniac_test_replay.html"
