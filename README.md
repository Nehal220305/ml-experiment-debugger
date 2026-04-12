---
title: ML Experiment Debugger
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - machine-learning
  - debugging
  - ml-debugging
  - rl-environment
---

# ML Experiment Debugger Environment

An OpenEnv environment where an AI agent debugs broken ML training experiments.
The agent receives a faulty training configuration and log, identifies the bug,
and proposes a fix. Designed for RL training and agent evaluation.

> No existing OpenEnv environment covers ML experiment debugging.
> This fills a direct gap in the RL agent evaluation ecosystem.

**Live API Docs:** https://Nehal-2203-ml-experiment-debugger.hf.space/docs

## What is this?

Real ML engineers debug broken training runs every day — exploding loss,
data leakage, silent label corruption, vanishing gradients. This environment
simulates exactly those scenarios using real PyTorch training loops so AI
agents can learn to diagnose and fix them programmatically.

## Tasks

| Task | Bug | Difficulty | Description |
|------|-----|------------|-------------|
| easy | `learning_rate_too_high` | Easy | Loss explodes to NaN — real gradient explosion via PyTorch |
| medium | `data_leakage` | Medium | Val set == train set, accuracy misleadingly perfect |
| hard | `label_noise` | Hard | 30% labels flipped + bad LR, model appears fine but fails on real data |
| very_hard | `wrong_loss_function` | Very Hard | MSE loss for classification — accuracy plateaus near 52% despite converging loss |
| expert_1 | `vanishing_gradients` | Expert | Deep network with sigmoid activations — gradients vanish, loss stalls |
| expert_2 | `missing_normalization` | Expert | Unnormalized inputs (values >1000) cause wildly oscillating loss |

## Action Space

The agent sends a JSON action with a free-text diagnosis:
```json
{
  "action_type": "diagnose",
  "response": "The learning rate of 50.0 is too high causing gradient explosion. The loss explodes to NaN within 5 steps. Fix: reduce learning rate to 0.01."
}
```

### Action types

- `diagnose` — agent provides free-text diagnosis and fix. Scored by LLM-as-judge on bug identification, fix quality, and explanation depth.
- `request_more_steps` — agent requests 10 more training steps to observe patterns before diagnosing.
- `inspect_gradients` — agent requests gradient norm statistics to identify gradient-related bugs.

## Observation Space
```json
{
  "task_id": "easy | medium | hard | very_hard | expert_1 | expert_2",
  "training_log": ["step 1: loss=0.745 train_acc=0.459", "step 2: loss=11.34 train_acc=0.512", "..."],
  "current_config": {"learning_rate": 50.0, "max_iter": 20, "optimizer": "sgd", "..."},
  "hint": "optional hint after wrong attempts",
  "message": "feedback string",
  "reward": 0.01,
  "done": false
}
```

## Example Agent Interaction

### 1. Reset — start a new episode
```bash
POST /reset {"task_id": "easy"}
```
```json
{
  "training_log": ["Training config: lr=100.0, optimizer=sgd, loss=bce", "step 1: loss=0.745 train_acc=0.306 val_acc=0.262 grad_norm=4.30e-01", "step 2: loss=63.52 train_acc=0.641 val_acc=0.625 grad_norm=2.56e+01", "step 5: loss=nan (EXPLODING GRADIENTS, grad_norm=1.59e+06)", "..."],
  "current_config": {"learning_rate": 100.0, "max_iter": 20, "optimizer": "sgd"},
  "message": "Task 'easy': Training loss explodes to NaN. Diagnose and fix the config."
}
```

### 2. Submit free-text diagnosis — get scored
```bash
POST /step {"action": {"action_type": "diagnose", "response": "The learning rate of 100.0 is too high causing exploding gradients. The loss jumps from 0.745 to NaN within 5 steps and gradient norm exceeds 1e6. Fix: reduce learning rate to 0.01."}}
```
```json
{
  "reward": 0.99,
  "done": true,
  "feedback": "✓ Correctly identified the bug | ✓ Suggested a valid fix | ✓ Provided detailed explanation | ✓ Thorough analysis",
  "message": "Score: 0.99. Episode complete."
}
```

## Reward Function

| Event | Reward |
|-------|--------|
| Correctly identifies bug | +0.40 |
| Suggests valid fix with specific values | +0.40 |
| Explains why bug causes symptoms | +0.20 |
| Score range | 0.01 – 0.99 |
| Easy task cap | 0.95 |
| Medium task cap | 0.85 |
| Hard task cap | 0.80 |
| Very Hard task cap | 0.75 |
| Expert tasks cap | 0.55-0.60 |

Score range: 0.01 – 0.99 (strictly exclusive of 0 and 1).

## Setup & Usage

### Local (Python)
```bash
# Install dependencies
pip install openenv-core fastapi uvicorn scikit-learn numpy pydantic openai torch

# Run server
cd server
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Local (Docker)
```bash
docker build -t ml-experiment-debugger -f server/Dockerfile .
docker run -p 7860:7860 ml-experiment-debugger
```

### Run inference (using HF router — no paid API key needed)
```bash
export HF_TOKEN=your_huggingface_token
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
python inference.py --host https://Nehal-2203-ml-experiment-debugger.hf.space
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode, returns initial observation |
| `/step` | POST | Send action, returns observation + reward |
| `/state` | GET | Current episode state |
| `/health` | GET | Server health check |
| `/baseline` | GET | Trigger baseline script, returns scores |
| `/grader` | GET | Returns grader score for current episode |
| `/tasks` | GET | Lists all tasks and action schema |
| `/docs` | GET | Interactive API documentation |

## Baseline Scores

Evaluated using `llama-3.3-70b-versatile` (Groq) with zero-shot prompting:

| Task | Score |
|------|-------|
| easy | 0.95 |
| medium | 0.85 |
| hard | 0.80 |
| very_hard | 0.75 |
| expert_1 | 0.50 |
| expert_2 | 0.55 |
| **avg** | **0.73** |

## Why This Environment Matters

Debugging broken ML experiments is a task every ML engineer faces daily.
No existing OpenEnv environment covers this domain. This environment enables:

- Training RL agents to diagnose ML failures automatically
- Evaluating LLMs on realistic engineering debugging tasks
- Benchmarking agent performance across 6 difficulty levels
- Testing agent ability to reason about gradient flow, data quality, and loss design

## Environment Details

- **Framework:** OpenEnv + FastAPI + Docker
- **Grader:** LLM-as-judge (Llama-3.3-70B) with keyword fallback — evaluates bug identification, fix quality, and explanation depth
- **Training:** Real inline PyTorch execution — all loss curves and metrics are genuine PyTorch output
- **Randomized:** Bug parameters vary per episode — agents cannot memorize answers
- **Multi-agent:** True concurrent session isolation per episode_id
- **Session Management:** Auto-expiry of abandoned sessions after 1 hour
- **Reproducible:** Fixed random seed (42) ensures consistent bug symptoms
- **Live Space:** https://huggingface.co/spaces/Nehal-2203/ml-experiment-debugger