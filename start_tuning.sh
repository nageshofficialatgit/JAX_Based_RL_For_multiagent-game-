#!/bin/bash
if [ ! -d "orbit" ]; then
    python3 -m venv orbit
fi
source orbit/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python training/cma_tuner.py
