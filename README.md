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

The agent sends a JSON action with these fields:
```json
{
  "action_type": "identify_bug | fix_config | submit_fix",
  "bug_identified": "learning_rate_too_high | data_leakage | label_noise | wrong_loss_function | vanishing_gradients | missing_normalization",
  "config_changes": {"learning_rate": 0.001},
  "explanation": "optional reasoning string"
}
```

### Action types

- `identify_bug` — agent names the bug. +0.3 reward if correct.
- `fix_config` — agent proposes config changes. Returns updated score.
- `submit_fix` — final submission. Triggers grader, ends episode.

## Observation Space
```json
{
  "task_id": "easy | medium | hard | very_hard | expert_1 | expert_2",
  "training_log": ["step 1: loss=0.745 train_acc=0.459", "step 2: loss=11.34 train_acc=0.512", "..."],
  "current_config": {"learning_rate": 50.0, "max_iter": 20, "optimizer": "sgd", "..."},
  "hint": "optional hint after wrong attempts",
  "message": "feedback string",
  "reward": 0.0,
  "done": false
}
```

## Reward Function

| Event | Reward |
|-------|--------|
| Correctly identifies bug type | +0.3 |
| Fix makes training converge | +0.7 |
| Wrong bug identification | 0.0 |
| Partial fix (some improvement) | +0.35 |

Reward is given at every step — not just at episode end.
Score range: 0.0 – 1.0.

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
| easy | 1.00 |
| medium | 1.00 |
| hard | 0.35 |
| very_hard | 0.30 |
| expert_1 | 0.60 |
| expert_2 | 1.00 |
| **avg** | **0.71** |

## Why This Environment Matters

Debugging broken ML experiments is a task every ML engineer faces daily.
No existing OpenEnv environment covers this domain. This environment enables:

- Training RL agents to diagnose ML failures automatically
- Evaluating LLMs on realistic engineering debugging tasks
- Benchmarking agent performance across 6 difficulty levels
- Testing agent ability to reason about gradient flow, data quality, and loss design

## Environment Details

- **Framework:** OpenEnv + FastAPI + Docker
- **Training:** Real PyTorch training loops — all loss curves and metrics are genuine
- **Grader:** Runs actual PyTorch training with the agent's fix — scores are objective, not heuristic
- **Reproducible:** Fixed random seed (42) ensures identical results every run
- **Live Space:** https://huggingface.co/spaces/Nehal-2203/ml-experiment-debugger