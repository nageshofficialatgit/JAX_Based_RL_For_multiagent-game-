import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import OrbitWarsDataset
from entity_transformer import EntityTransformer

def train():
    print("Initializing Dataset...")
    dataset = OrbitWarsDataset(max_action_history=20)
    
    # Let's do a 90/10 split just to be proper
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    batch_size = 512
    # Setting num_workers to 8 for max IO throughput on 32-core server
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Instantiate the Entity Transformer (Scaled up to ~38M parameters)
    model = EntityTransformer().to(device)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    print("\nStarting Full Training Run...")
    
    num_epochs = 10
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        t0 = time.time()
        moving_loss = 0.0
        moving_acc = 0.0
        
        for i, batch in enumerate(train_dl):
            x = batch["state_tokens"].to(device)
            y = batch["winner"].to(device) # Shape [B]
            
            optimizer.zero_grad()
            logits = model(x) # [B, 4]
            
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accuracy
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean().item()
            
            moving_loss = 0.9 * moving_loss + 0.1 * loss.item() if i > 0 else loss.item()
            moving_acc = 0.9 * moving_acc + 0.1 * acc if i > 0 else acc
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                batches_per_sec = (i + 1) / elapsed
                items_per_sec = batches_per_sec * batch_size
                
                vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0
                
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {i+1}/{len(train_dl)} | "
                      f"Loss: {moving_loss:.4f} | Acc: {moving_acc*100:.1f}% | "
                      f"Speed: {items_per_sec:.0f} states/s | VRAM: {vram_mb:.0f}MB", flush=True)

        print(f"\n--- Epoch {epoch+1} Complete | Final Acc: {moving_acc*100:.1f}% ---")
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"value_network_ep{epoch+1}.pt")
        if moving_acc > best_acc:
            best_acc = moving_acc
            torch.save(model.state_dict(), "value_network_best.pt")
            print("=> Saved new best checkpoint!")
            
    print("\nTraining Complete!")



if __name__ == "__main__":
    train()
