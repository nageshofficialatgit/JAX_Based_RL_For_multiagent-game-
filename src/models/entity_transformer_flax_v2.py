import jax
import jax.numpy as jnp
from flax import nnx

class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.mha = nnx.MultiHeadAttention(
            num_heads=n_heads, 
            in_features=d_model, 
            qkv_features=d_model, 
            out_features=d_model, 
            decode=False,
            rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(d_model, dim_feedforward, rngs=rngs),
            nnx.gelu,
            nnx.Linear(dim_feedforward, d_model, rngs=rngs)
        )
        
    def __call__(self, x, mask=None):
        # Pre-LayerNorm Architecture
        h = self.norm1(x)
        h = self.mha(h, h, h, mask=mask)
        x = x + h
        
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

class EntityTransformer(nnx.Module):
    # CHANGED DEFAULTS: num_features=37 for Grandmaster Meta-Engineering V4.1
    def __init__(self, d_model=128, n_heads=4, n_layers=4, num_features=37, num_classes=5, rngs: nnx.Rngs = None):
        super().__init__()
        # Input Embeddings
        self.feature_embed = nnx.Linear(num_features, d_model, rngs=rngs)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, d_model)))
        
        self.num_planets = 50
        self.type_embed = nnx.Embed(3, d_model, rngs=rngs)
        
        blocks_list = [TransformerBlock(d_model, n_heads, d_model * 4, rngs) for _ in range(n_layers)]
        self.blocks = nnx.List(blocks_list) if hasattr(nnx, 'List') else blocks_list
        
        self.norm_f = nnx.LayerNorm(d_model, rngs=rngs)
        
        self.value_head = nnx.Sequential(
            nnx.Linear(d_model, 64, rngs=rngs), 
            nnx.gelu,
            nnx.Linear(64, num_classes, rngs=rngs)
        )
        self.ppo_value_head = nnx.Linear(d_model, 1, rngs=rngs)
        
        # --- SHRUNK LOCAL POLICY HEADS ---
        # 1. Launch
        self.actor_launch = nnx.Linear(d_model, 1, rngs=rngs)
        self.launch_embed = nnx.Embed(2, 16, rngs=rngs) 
        
        # 2. Ships (Moved UP)
        self.actor_ships = nnx.Sequential(
            nnx.Linear(d_model + 16, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, 10, rngs=rngs)
        )
        self.ships_embed = nnx.Embed(10, 16, rngs=rngs) # NEW: Ship choice embedding
        
        # 3. Target Query (Moved DOWN)
        self.target_query = nnx.Sequential(
            nnx.Linear(d_model + 16 + 16, 64, rngs=rngs), # Takes Launch AND Ships
            nnx.gelu,
            nnx.Linear(64, 32, rngs=rngs) 
        )
        self.target_key = nnx.Sequential(
            nnx.Linear(d_model, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, 32, rngs=rngs)
        )

    def __call__(self, x, return_policy=False, target_launch=None, target_ships=None, target_angle=None, valid_launch_mask=None, sample_rng=None):
        batch_size = x.shape[0]
        x = x.astype(jnp.bfloat16)
        x_emb = self.feature_embed(x)
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (batch_size, 1, x_emb.shape[-1]))
        x_emb = jnp.concatenate((cls_tokens, x_emb), axis=1)
        
        seq_len = x_emb.shape[1]
        type_ids = jnp.zeros((seq_len,), dtype=jnp.int32)
        type_ids = type_ids.at[1:1+self.num_planets].set(1)
        type_ids = type_ids.at[1+self.num_planets:].set(2)
        x_emb = x_emb + self.type_embed(type_ids)[None, :, :]
        
        valid_tokens = (x[:, :, 0] != -1.0)
        cls_valid = jnp.ones((batch_size, 1), dtype=bool)
        valid_mask = jnp.concatenate((cls_valid, valid_tokens), axis=1) 
        attention_mask = valid_mask[:, None, None, :]
        
        for block in self.blocks:
            x_emb = block(x_emb, mask=attention_mask)
            
        out = self.norm_f(x_emb).astype(jnp.float32)
        
        cls_out = out[:, 0, :]
        v_logits = self.value_head(cls_out)

        if not return_policy:
            return v_logits
            
        planet_out = out[:, 1:1+self.num_planets, :] 
        
        # 1. Autoregressive Launch
        launch_logits = jnp.squeeze(self.actor_launch(planet_out), axis=-1).astype(jnp.float32)
        if valid_launch_mask is not None:
            launch_logits = jnp.where(valid_launch_mask, launch_logits, -1e4)
        
        if target_launch is not None:
            launch_choices = jnp.minimum(jnp.maximum(target_launch, 0), 1).astype(jnp.int32)
        elif sample_rng is not None:
            rng1, sample_rng = jax.random.split(sample_rng)
            launch_prob = jax.nn.sigmoid(launch_logits)
            launch_choices = jax.random.bernoulli(rng1, launch_prob).astype(jnp.int32)
        else:
            launch_choices = (launch_logits > 0).astype(jnp.int32)

        launch_emb = self.launch_embed(launch_choices)

        # 2. Autoregressive Ships (NEW CAUSALITY ORDER)
        ships_input = jnp.concatenate([planet_out, launch_emb], axis=-1)
        ships_logits = self.actor_ships(ships_input).astype(jnp.float32)
        
        if target_ships is not None:
            ships_choices = jnp.minimum(jnp.maximum(target_ships, 0), 9).astype(jnp.int32)
        elif sample_rng is not None:
            rng2, sample_rng = jax.random.split(sample_rng)
            ships_choices = jax.random.categorical(rng2, ships_logits, axis=-1).astype(jnp.int32)
        else:
            ships_choices = jnp.argmax(ships_logits, axis=-1).astype(jnp.int32)
            
        ships_emb = self.ships_embed(ships_choices)

        # 3. Permutation-Invariant Targeting via Pointer Network
        # The query now contains the 'ships_emb', effectively knowing the fleet speed!
        query_input = jnp.concatenate([planet_out, launch_emb, ships_emb], axis=-1) 
        queries = self.target_query(query_input) 
        keys = self.target_key(planet_out) 

        angle_logits = jnp.einsum('bqd,bkd->bqk', queries, keys) / jnp.sqrt(queries.shape[-1])

        mask = jnp.eye(self.num_planets, dtype=bool)[None, :, :]
        mask = mask | (~valid_tokens[:, None, :])
        
        # Prevent NaN in softmax by unmasking self if ALL targets are masked
        all_masked = jnp.all(mask, axis=-1, keepdims=True)
        mask = jnp.where(all_masked, False, mask)
        
        angle_logits = jnp.where(mask, -1e4, angle_logits).astype(jnp.float32)

        if target_angle is not None:
            target_choices = jnp.minimum(jnp.maximum(target_angle, 0), self.num_planets-1).astype(jnp.int32)
        elif sample_rng is not None:
            rng3, sample_rng = jax.random.split(sample_rng)
            target_choices = jax.random.categorical(rng3, angle_logits, axis=-1).astype(jnp.int32)
        else:
            target_choices = jnp.argmax(angle_logits, axis=-1).astype(jnp.int32)

        ppo_value = jnp.squeeze(self.ppo_value_head(cls_out), axis=-1).astype(jnp.float32)

        if sample_rng is not None:
            return v_logits, launch_logits, ships_logits, angle_logits, ppo_value, launch_choices, ships_choices, target_choices

        return v_logits, launch_logits, ships_logits, angle_logits, ppo_value

if __name__ == "__main__":
    # Quick shape verification
    rngs = nnx.Rngs(0)
    model = EntityTransformer(d_model=64, n_heads=2, n_layers=2, rngs=rngs, num_features=35)
    
    dummy_x = jnp.ones((4, 50, 35)) # [Batch, Tokens, Features]
    v_logits, launch, angle, ships, ppo_val = model(dummy_x, return_policy=True)
    
    print("Flax NNX Compilation Successful with Mixed Precision (bfloat16)!")
    print(f"Value Logits: {v_logits.shape} (Expected [4, 4])")
    print(f"Launch Logits: {launch.shape} (Expected [4, 50])")
    print(f"Angle Logits: {angle.shape} (Expected [4, 50, 50])")
    print(f"Ships Logits: {ships.shape} (Expected [4, 50, 10])")
    print(f"PPO Value: {ppo_val.shape} (Expected [4])")