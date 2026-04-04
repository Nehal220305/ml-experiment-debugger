import sys
import os
import uuid

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")

from fastapi import FastAPI
from fastapi import Request, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
from server.ml_experiment_debugger_environment import MlExperimentDebuggerEnvironment
from models import MLAction

import asyncio
from functools import partial

app = FastAPI(
    title="ML Experiment Debugger",
    description="OpenEnv environment for debugging broken ML training experiments.",
)

env = MlExperimentDebuggerEnvironment()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    
    model_config = {"extra": "allow"}


class StepRequest(BaseModel):
    action: MLAction


@app.post("/reset")
async def reset(
    request: Request,
    task_id: Optional[str] = Query(default=None),
    seed: Optional[int] = Query(default=None),
    episode_id: Optional[str] = Query(default=None),
):
    try:
        body = await request.json()
    except:
        body = {}

    final_task_id = task_id or (body.get("task_id") if body else None) or "easy"
    final_seed = seed or (body.get("seed") if body else None)
    final_episode_id = episode_id or (body.get("episode_id") if body else None)

    loop = asyncio.get_event_loop()
    obs = await loop.run_in_executor(
        None,
        partial(env.reset, task_id=final_task_id, seed=final_seed, episode_id=final_episode_id)
    )
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.post("/step")
async def step(request: StepRequest):
    loop = asyncio.get_event_loop()
    obs = await loop.run_in_executor(None, partial(env.step, request.action))
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/state")
def state():
    return env.state.model_dump()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/schema")
def schema():
    return {
        "action": MLAction.model_json_schema(),
    }


@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Exploding Loss",
                "difficulty": "easy",
                "description": "Learning rate is too high — loss goes to NaN immediately.",
                "bug": "learning_rate_too_high",
                "action_schema": {
                    "action_type": "submit_fix",
                    "bug_identified": "learning_rate_too_high",
                    "config_changes": {"learning_rate": 0.01},
                    "explanation": "string (optional)",
                },
            },
            {
                "id": "medium",
                "name": "Data Leakage",
                "difficulty": "medium",
                "description": "Validation set is same as training set — accuracy is misleadingly perfect.",
                "bug": "data_leakage",
                "action_schema": {
                    "action_type": "submit_fix",
                    "bug_identified": "data_leakage",
                    "config_changes": {"fix_train_val_split": True},
                    "explanation": "string (optional)",
                },
            },
            {
                "id": "hard",
                "name": "Silent Label Noise",
                "difficulty": "hard",
                "description": "30% of labels are flipped — model trains fine but fails on real data.",
                "bug": "label_noise",
                "action_schema": {
                    "action_type": "submit_fix",
                    "bug_identified": "label_noise",
                    "config_changes": {"label_noise_pct": 0.0},
                    "explanation": "string (optional)",
                },
            },
            {
                "id": "very_hard",
                "name": "Wrong Loss Function",
                "difficulty": "very_hard",
                "description": "Model uses regression loss for classification — accuracy plateaus near random.",
                "bug": "wrong_loss_function",
                "action_schema": {
                    "action_type": "submit_fix",
                    "bug_identified": "wrong_loss_function",
                    "config_changes": {"loss": "log_loss"},
                    "explanation": "string (optional)",
                },
            },
            {
                "id": "expert_1",
                "name": "Vanishing Gradients",
                "difficulty": "expert",
                "description": "Deep network with sigmoid activations — gradients vanish, loss stalls.",
                "bug": "vanishing_gradients",
                "action_schema": {
                    "action_type": "submit_fix",
                    "bug_identified": "vanishing_gradients",
                    "config_changes": {"activation": "relu"},
                    "explanation": "string (optional)",
                },
            },
            {
                "id": "expert_2",
                "name": "Missing Normalization",
                "difficulty": "expert",
                "description": "Unnormalized inputs cause oscillating loss and poor convergence.",
                "bug": "missing_normalization",
                "action_schema": {
                    "action_type": "submit_fix",
                    "bug_identified": "missing_normalization",
                    "config_changes": {"normalize_input": True},
                    "explanation": "string (optional)",
                },
            },
        ]
    }


@app.get("/grader")
def get_grader():
    return {
        "grader": "ML Experiment Debugger Grader",
        "scoring": {
            "identify_bug_correct": 0.3,
            "fix_converges": 0.7,
            "partial_fix": 0.35,
            "wrong_identification": 0.0,
        },
        "score_range": "0.0 to 1.0",
        "description": "Scores agent on bug identification and config fix quality.",
    }


@app.get("/baseline")
def get_baseline():
    return {
        "baseline_scores": {
            "easy": 1.00,
            "medium": 1.00,
            "hard": 0.35,
            "very_hard": 0.30,
            "expert_1": 1.00,
            "expert_2": 1.00,
            "average": 0.78,
        },
        "model": "llama-3.3-70b-versatile (Groq)",
        "description": "Run baseline.py with GROQ_API_KEY to reproduce these scores.",
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()