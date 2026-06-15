import numpy as np
from src.data_pipeline.dataset_grain import build_grain_dataloader

loader = build_grain_dataloader(batch_size=10, worker_count=1, num_epochs=1)
for i, batch in enumerate(loader):
    state = batch['state_tokens'] # [B, 70, 14]
    launch = batch['target_launch'] # [B, 50]
    angle = batch['target_angle'] # [B, 50]
    
    # Check valid planets
    valid_planets = (state[:, 1:51, 0] == 0.0) # [B, 50]
    
    for b in range(10):
        for p in range(50):
            if launch[b, p] == 1.0 and angle[b, p] != -1:
                t = angle[b, p]
                if not valid_planets[b, t]:
                    print(f"BINGO! Launch at {p} targets padded planet {t}!")
                    exit(0)
    
    if i > 50:
        break

print("No padded planets targeted.")
