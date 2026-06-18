# Orbit Wars Server Deploy Reference Index

## 1. Project Purpose

This workspace is building and training an agent for the Kaggle `orbit_wars` environment.
The problem is a continuous 2D real-time strategy game where the agent must:
- control planets and fleets,
- capture neutral and enemy planets,
- handle orbiting planets and comets,
- avoid the sun,
- maximize ships on owned assets at game end.

The core training goal is to produce a strong policy for launching fleets from owned planets using a transformer-based model.

## 2. Key Game Concepts

- `planets`: `[id, owner, x, y, radius, ships, production]`
- `fleets`: `[id, owner, x, y, angle, from_planet_id, ships]`
- `owner`: `-1` or `0` means neutral, `0..3` are players
- `sun`: centered at `(50, 50)`, radius `10`; fleets crossing it die
- `fleet speed`: nonlinear, depends on ship count
- `orbiting planets`: rotate around sun at constant angular velocity
- `comets`: temporary moving planets, spawn at fixed steps and leave the board
- action: `[from_planet_id, angle, num_ships]`

## 3. Main Files and Structure

### Documentation
- `server_deploy/README.md`
  - Full Orbit Wars rules and observation/action format
- `server_deploy/agents.md`
  - Kaggle submission guide and example agent code

### Submission wrapper
- `server_deploy/src/submission/agent.py`
  - Kaggle-compatible entrypoint
  - Handles import path resolution and unbuffered logging
  - Currently returns an empty action list as a scaffold

### Environment implementation
- `server_deploy/src/env_jax/orbit_env.py`
  - `EnvState`: JAX dataclass holding planet and fleet state
  - `get_fleet_speed()`: uses Kaggle-style nonlinear speed
  - `step_physics()`: production, fleet movement, rotation, collision, combat, sun removal
  - `apply_actions()`: spawns new fleets from launch decisions
  - `build_observation()`: builds the model input tensor of shape `[70, 12]`

### Model architecture
- `server_deploy/src/models/entity_transformer_flax.py`
  - Transformer-based policy/value network in Flax NNX
  - Input tokens: 50 planets + 20 fleets + 1 CLS token
  - Outputs:
    - value head (`num_classes=5` for winner/state score)
    - launch logits for each planet
    - angle logits for each planet
    - ship amount logits for each planet
  - Designed for autoregressive local policy decisions

- `server_deploy/src/models/entity_transformer.py`
  - Likely PyTorch analog used by prototype training

### Training and evaluation
- `server_deploy/src/training/train_iql.py`
  - Offline RL / IQL prototype using PyTorch
  - Loads `OrbitWarsDataset` and `EntityTransformer`
  - Contains commented placeholders for action targets
  - Indicates dataset target labeling is not fully implemented yet

- `server_deploy/src/training/train_ppo_flax.py`
  - JAX/Flax PPO training pipeline
  - Builds self-play environment and model restoration
  - Uses `optax` + `orbax.checkpoint`
  - Implements rollout, advantage/GAE, PPO loss, and update logic
  - Appears to optimize player 1 policy in a simplified self-play loop

- `server_deploy/src/training/train_value.py`
  - Value training script (exact details in file)

- `server_deploy/src/training/train_value_flax.py`
  - JAX/Flax value model training

- `server_deploy/src/training/play_match.py`
  - Match execution / evaluation harness

- `server_deploy/src/training/eval_checkpoint.py`
  - Checkpoint evaluation utilities

## 4. Data and Pipeline

- `server_deploy/src/data_pipeline/dataset.py`
  - Dataset construction for orbit wars data
- `server_deploy/src/data_pipeline/dataset_grain.py`
  - Fine-grained dataset utilities
- `server_deploy/src/data_pipeline/download_and_build_pipeline.py`
  - Download/build data pipeline support
- `server_deploy/src/data_pipeline/filter_grandmasters.py`
  - Data filtering utility

## 5. Supporting utilities

- `server_deploy/build_submission.py`
  - Builds tarball for Kaggle submission
- `server_deploy/create_zoo.py`
  - Likely creates agent zoo or environment collection
- `server_deploy/profile_game.py`
  - Profiling utility for the game
- `server_deploy/plot_metrics.py`
  - Visualization of training/evaluation metrics

## 6. What has been done so far

- Core game rules and Kaggle action format fully documented in `server_deploy/README.md`
- A Kaggle agent wrapper scaffold exists in `server_deploy/src/submission/agent.py`
- A JAX/Flax environment with physics and combat is implemented in `server_deploy/src/env_jax/orbit_env.py`
- A transformer policy/value architecture is implemented in `server_deploy/src/models/entity_transformer_flax.py`
- Two training directions exist:
  - PyTorch prototype / IQL (`train_iql.py`)
  - JAX/Flax PPO self-play (`train_ppo_flax.py`)
- Data pipeline and dataset modules are present for offline training

## 7. Future reference notes

- Use `server_deploy/src/env_jax/orbit_env.py` as the canonical environment when building training loops.
- Use `server_deploy/src/models/entity_transformer_flax.py` for model input/output expectations.
- `train_iql.py` is a useful prototype but needs dataset target action labels to be completed.
- `train_ppo_flax.py` is the main RL training path for a JAX-based agent.
- `src/submission/agent.py` should be the final Kaggle export once inference is implemented.

## 8. Recommended next tasks

1. finish dataset labeling for action targets and complete `OrbitWarsDataset`
2. implement inference in `src/submission/agent.py` using the trained Flax model
3. confirm training/evaluation flow via `play_match.py` and `eval_checkpoint.py`
4. add README pointers to script usage and experiment commands

---

This document is intended as the indexed reference for the current `server_deploy` Orbit Wars training project.