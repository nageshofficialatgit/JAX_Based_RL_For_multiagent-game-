import time
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure `server_deploy/src` is on sys.path so local modules resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_pipeline.dataset import OrbitWarsDataset
from models.entity_transformer import EntityTransformer

def train_iql():
    print("Initializing IQL Dataset...")
    # TODO: Dataset needs to be updated to yield target actions for each of the 50 planets
    dataset = OrbitWarsDataset(max_action_history=20)
    train_dl = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the 38M Parameter Model
    model = EntityTransformer().to(device)
    
    # Loss Functions for Discretized Action Space
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # IQL Hyperparameters
    beta = 3.0 # Advantage weighting temperature
    
    model.train()
    print("\nStarting IQL Offline RL Training...")
    
    for i, batch in enumerate(train_dl):
        x = batch["state_tokens"].to(device)
        y_winner = batch["winner"].to(device)
        
        # MOCK TARGETS (Dataset update needed to supply these)
        # target_launch = [B, 50] (1 if planet launched a fleet, 0 otherwise)
        # target_angle = [B, 50] (0 to 71 class index)
        # target_ships = [B, 50] (0 to 9 class index)
        # is_owned_by_winner = [B, 50] (Boolean mask to only train on winner's actions)
        
        # Forward Pass
        v_logits, launch_logits, angle_logits, ships_logits, ppo_val = model(x, return_policy=True)
        
        # 1. Critic Loss (Value Network)
        # Using CrossEntropy since winner is a discrete slot
        v_loss = ce_loss(v_logits, y_winner).mean()
        
        # 2. Advantage Calculation (Q - V)
        # For simplicity in this fast experiment, we use the probability of the winner 
        # as the state value V(s), and assume Q(s,a) is implicitly 1.0 since we only 
        # clone the actions of the eventual winner.
        with torch.no_grad():
            v_probs = torch.softmax(v_logits, dim=-1)
            v_s = v_probs[torch.arange(v_probs.shape[0]), y_winner] # Prob of winner winning
            
            # IQL Advantage Weighting: exp(beta * (Q - V))
            # Since Q is 1.0 (they won), advantage is (1.0 - V(s))
            advantage = 1.0 - v_s
            weights = torch.exp(beta * advantage)
            # Clip weights to prevent explosion
            weights = torch.clamp(weights, max=100.0).unsqueeze(1) # [B, 1]
            
        # 3. Actor Loss (Policy Extraction via Behavioral Cloning + Advantage Weighting)
        # We only apply loss to planets owned by the winner
        # loss_launch = bce_loss(launch_logits, target_launch)
        # loss_angle = ce_loss(angle_logits.transpose(1, 2), target_angle)
        # loss_ships = ce_loss(ships_logits.transpose(1, 2), target_ships)
        
        # actor_loss = (loss_launch + loss_angle + loss_ships) * weights * is_owned_by_winner
        # total_actor_loss = actor_loss.mean()
        
        # total_loss = v_loss + total_actor_loss
        
        # optimizer.zero_grad()
        # total_loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # optimizer.step()
        
        if (i + 1) % 50 == 0:
            print(f"Batch {i+1} | V-Loss: {v_loss.item():.4f}")
            break # Just a structural dry run

if __name__ == "__main__":
    train_iql()
