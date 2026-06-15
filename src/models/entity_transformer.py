import torch
import torch.nn as nn

class EntityTransformer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_layers=12, num_features=10, num_classes=4):
        super().__init__()
        
        self.feature_embed = nn.Linear(num_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # --- GLOBAL HEADS (For state value / winner prediction) ---
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes) # Predicts V(s) effectively
        )
        
        # --- LOCAL POLICY HEADS (For IQL / Actor) ---
        # Each planet token outputs an independent action
        # 1. Will this planet launch a fleet this tick? (Binary)
        self.actor_launch = nn.Linear(d_model, 1)
        
        # 2. What angle? (Discretized into 72 bins of 5 degrees for stable learning)
        self.actor_angle = nn.Linear(d_model, 72)
        
        # 3. What fraction of ships? (Discretized into 10 bins: 10%, 20%... 100%)
        self.actor_ships = nn.Linear(d_model, 10)

    def forward(self, x, return_policy=False):
        """
        x: [batch_size, num_tokens, num_features]
        """
        batch_size = x.shape[0]
        x_emb = self.feature_embed(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)
        
        out = self.transformer(x_emb)
        
        # Global Value
        cls_out = out[:, 0, :]
        v_logits = self.value_head(cls_out)
        
        if not return_policy:
            return v_logits
            
        # Local Policy (Extract only the first 50 tokens which are the Planets)
        # Note: out is [B, T+1, D] because of CLS at index 0.
        planet_out = out[:, 1:51, :] # [B, 50, D]
        
        launch_logits = self.actor_launch(planet_out).squeeze(-1) # [B, 50]
        angle_logits = self.actor_angle(planet_out) # [B, 50, 72]
        ships_logits = self.actor_ships(planet_out) # [B, 50, 10]
        
        return v_logits, launch_logits, angle_logits, ships_logits
