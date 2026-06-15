import jax
import jax.numpy as jnp
from flax import nnx

class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, rngs: nnx.Rngs):
        # Cast all internal weights and activations to bfloat16
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs, dtype=jnp.bfloat16)
        self.mha = nnx.MultiHeadAttention(
            num_heads=n_heads, 
            in_features=d_model, 
            qkv_features=d_model, 
            out_features=d_model, 
            decode=False,
            rngs=rngs,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16
        )
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs, dtype=jnp.bfloat16)
        self.mlp = nnx.Sequential(
            nnx.Linear(d_model, dim_feedforward, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16),
            nnx.gelu,
            nnx.Linear(dim_feedforward, d_model, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
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
    # CHANGED DEFAULTS: num_features=14 for comets (dx, dy)
    def __init__(self, d_model=128, n_heads=4, n_layers=4, num_features=14, num_classes=5, rngs: nnx.Rngs = None):
        super().__init__()
        # Input Embeddings
        self.feature_embed = nnx.Linear(num_features, d_model, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, d_model), dtype=jnp.bfloat16))
        
        self.num_planets = 50
        self.num_actions = 20
        
        self.type_embed = nnx.Embed(3, d_model, rngs=rngs, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        
        blocks_list = [TransformerBlock(d_model, n_heads, d_model * 4, rngs) for _ in range(n_layers)]
        self.blocks = nnx.List(blocks_list) if hasattr(nnx, 'List') else blocks_list
        
        self.norm_f = nnx.LayerNorm(d_model, rngs=rngs, dtype=jnp.bfloat16)
        
        # --- SHRUNK GLOBAL HEADS ---
        self.value_head = nnx.Sequential(
            nnx.Linear(d_model, 64, rngs=rngs), # Shrunk intermediate dim
            nnx.gelu,
            nnx.Linear(64, num_classes, rngs=rngs)
        )
        self.ppo_value_head = nnx.Linear(d_model, 1, rngs=rngs)
        
        # --- SHRUNK LOCAL POLICY HEADS ---
        self.actor_launch = nnx.Linear(d_model, 1, rngs=rngs)
        self.launch_embed = nnx.Embed(2, 16, rngs=rngs) # Reduced from 32 to 16
        
        self.target_query = nnx.Sequential(
            nnx.Linear(d_model + 16, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, 32, rngs=rngs) # Pointer latent space reduced to 32
        )
        self.target_key = nnx.Sequential(
            nnx.Linear(d_model, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, 32, rngs=rngs)
        )

        self.actor_ships = nnx.Sequential(
            nnx.Linear(d_model + 16 + d_model, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, 10, rngs=rngs)
        )

    def __call__(self, x, return_policy=False, target_launch=None, target_angle=None, valid_launch_mask=None, sample_rng=None):
        """
        x: [batch_size, num_tokens, num_features]
        """
        batch_size = x.shape[0]
        
        # Input features are typically float32 from the environment, 
        # so we cast them to bfloat16 as they enter the network.
        x = x.astype(jnp.bfloat16)
        x_emb = self.feature_embed(x)
        
        # Broadcast CLS token to match batch size
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (batch_size, 1, x_emb.shape[-1]))
        x_emb = jnp.concatenate((cls_tokens, x_emb), axis=1)
        
        # Generate Token Type IDs
        seq_len = x_emb.shape[1]
        type_ids = jnp.zeros((seq_len,), dtype=jnp.int32)
        type_ids = type_ids.at[1:1+self.num_planets].set(1)
        type_ids = type_ids.at[1+self.num_planets:].set(2)
        
        # Inject Token Type Embeddings
        x_emb = x_emb + self.type_embed(type_ids)[None, :, :]
        
        # Generate Attention Mask to hide zero-padded empty planets/actions
        # Feature 0: -1.0 is padded planet, -2.0 is padded action
        valid_tokens = (x[:, :, 0] != -1.0) & (x[:, :, 0] != -2.0)
        # Prepend True for the CLS token
        cls_valid = jnp.ones((batch_size, 1), dtype=bool)
        valid_mask = jnp.concatenate((cls_valid, valid_tokens), axis=1) # [B, SeqLen]
        # MHA expects mask shape [B, num_heads, SeqLen, SeqLen] or broadcastable
        attention_mask = valid_mask[:, None, None, :]
        
        for block in self.blocks:
            x_emb = block(x_emb, mask=attention_mask)
            
        out = self.norm_f(x_emb)
        
        # ====================================================================
        # CRITICAL BRIDGE: Cast the bfloat16 representations back to float32 
        # before passing them to the final layers to prevent NaN losses.
        # ====================================================================
        out = out.astype(jnp.float32)
        
        # Global Value Predictor (classification)
        cls_out = out[:, 0, :]
        v_logits = self.value_head(cls_out)

        if not return_policy:
            return v_logits
            
        # Local Policy (Extract only the active planet tokens)
        planet_out = out[:, 1:1+self.num_planets, :] # [B, num_planets, D]
        
        # 1. Autoregressive Launch
        launch_logits = jnp.squeeze(self.actor_launch(planet_out), axis=-1) # [B, 50]
        
        # Action Masking: prevent invalid launches
        if valid_launch_mask is not None:
            launch_logits = jnp.where(valid_launch_mask, launch_logits, -1e9)
        
        # 2. Permutation-Invariant Targeting via Pointer Network
        if target_launch is not None:
            launch_choices = jnp.minimum(jnp.maximum(target_launch, 0), 1).astype(jnp.int32)
        elif sample_rng is not None:
            rng1, sample_rng = jax.random.split(sample_rng)
            launch_prob = jax.nn.sigmoid(launch_logits)
            launch_choices = jax.random.bernoulli(rng1, launch_prob).astype(jnp.int32)
        else:
            launch_choices = (launch_logits > 0).astype(jnp.int32)

        launch_emb = self.launch_embed(launch_choices) # [B, 50, 32]

        # Queries: produced by each launching planet (depends on its launch choice)
        query_input = jnp.concatenate([planet_out, launch_emb], axis=-1) # [B,50,D+32]
        queries = self.target_query(query_input) # [B,50,Dk]

        # Keys: produced by every candidate target planet (purely from planet embedding)
        keys = self.target_key(planet_out) # [B,50,Dk]

        # Dot-product scores between each source-query and every target-key
        angle_logits = jnp.einsum('bqd,bkd->bqk', queries, keys) / jnp.sqrt(queries.shape[-1])

        # Prevent self-targeting by masking diagonal AND prevent targeting ghost planets
        valid_planets = valid_tokens[:, 1:1+self.num_planets] # [B, 50]
        mask = jnp.eye(self.num_planets, dtype=bool)[None, :, :]
        mask = mask | (~valid_planets[:, None, :])
        angle_logits = jnp.where(mask, -1e9, angle_logits)

        # 3. Autoregressive Ships: gather chosen target's actual transformer embedding
        if target_angle is not None:
            # target_angle encodes chosen target planet id per source
            target_choices = jnp.minimum(jnp.maximum(target_angle, 0), self.num_planets-1).astype(jnp.int32)
        elif sample_rng is not None:
            rng2, sample_rng = jax.random.split(sample_rng)
            target_choices = jax.random.categorical(rng2, angle_logits, axis=-1).astype(jnp.int32)
        else:
            target_choices = jnp.argmax(angle_logits, axis=-1).astype(jnp.int32)

        # Gather the actual target embeddings
        batch_idx = jnp.arange(batch_size)[:, None]
        target_emb = planet_out[batch_idx, target_choices, :] # [B, 50, D]

        ships_input = jnp.concatenate([planet_out, launch_emb, target_emb], axis=-1)
        ships_logits = self.actor_ships(ships_input) # [B, 50, 10]

        # PPO continuous scalar value (for advantage baselines)
        ppo_value = jnp.squeeze(self.ppo_value_head(cls_out), axis=-1)

        if sample_rng is not None:
            rng3, _ = jax.random.split(sample_rng)
            ships_choices = jax.random.categorical(rng3, ships_logits, axis=-1).astype(jnp.int32)
            return v_logits, launch_logits, angle_logits, ships_logits, ppo_value, launch_choices, target_choices, ships_choices

        return v_logits, launch_logits, angle_logits, ships_logits, ppo_value

if __name__ == "__main__":
    # Quick shape verification
    rngs = nnx.Rngs(0)
    model = EntityTransformer(d_model=64, n_heads=2, n_layers=2, rngs=rngs, num_features=14)
    
    dummy_x = jnp.ones((4, 70, 14)) # [Batch, Tokens, Features]
    v_logits, launch, angle, ships, ppo_val = model(dummy_x, return_policy=True)
    
    print("Flax NNX Compilation Successful with Mixed Precision (bfloat16)!")
    print(f"Value Logits: {v_logits.shape} (Expected [4, 4])")
    print(f"Launch Logits: {launch.shape} (Expected [4, 50])")
    print(f"Angle Logits: {angle.shape} (Expected [4, 50, 50])")
    print(f"Ships Logits: {ships.shape} (Expected [4, 50, 10])")
    print(f"PPO Value: {ppo_val.shape} (Expected [4])")