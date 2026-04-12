import sys
import os
import asyncio
from functools import partial
from typing import Optional, Dict, Any

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")

from fastapi import FastAPI, Request, Query
from pydantic import BaseModel
from server.ml_experiment_debugger_environment import MlExperimentDebuggerEnvironment
from models import MLAction

app = FastAPI(
    title="ML Experiment Debugger",
    description="OpenEnv RL environment for debugging broken ML training experiments using real PyTorch execution.",
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
    return {"action": MLAction.model_json_schema()}


@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Exploding Loss",
                "difficulty": "easy",
                "description": "Learning rate is too high — real gradient explosion via PyTorch.",
                "bug": "learning_rate_too_high",
                "action_schema": {
                    "action_type": "diagnose",
                    "response": "Free-text diagnosis and fix explanation",
                },
            },
            {
                "id": "medium",
                "name": "Data Leakage",
                "difficulty": "medium",
                "description": "Validation set is same as training set — accuracy is misleadingly perfect.",
                "bug": "data_leakage",
                "action_schema": {
                    "action_type": "diagnose",
                    "response": "Free-text diagnosis and fix explanation",
                },
            },
            {
                "id": "hard",
                "name": "Silent Label Noise",
                "difficulty": "hard",
                "description": "Labels corrupted + bad LR — model appears healthy but fails on real data.",
                "bug": "label_noise",
                "action_schema": {
                    "action_type": "diagnose",
                    "response": "Free-text diagnosis and fix explanation",
                },
            },
            {
                "id": "very_hard",
                "name": "Wrong Loss Function",
                "difficulty": "very_hard",
                "description": "MSE loss for classification — accuracy plateaus near 52% despite converging loss.",
                "bug": "wrong_loss_function",
                "action_schema": {
                    "action_type": "diagnose",
                    "response": "Free-text diagnosis and fix explanation",
                },
            },
            {
                "id": "expert_1",
                "name": "Vanishing Gradients",
                "difficulty": "expert",
                "description": "Deep network with sigmoid activations — gradients vanish, loss stalls.",
                "bug": "vanishing_gradients",
                "action_schema": {
                    "action_type": "diagnose",
                    "response": "Free-text diagnosis and fix explanation",
                },
            },
            {
                "id": "expert_2",
                "name": "Missing Normalization",
                "difficulty": "expert",
                "description": "Unnormalized inputs (values >1000) cause wildly oscillating loss.",
                "bug": "missing_normalization",
                "action_schema": {
                    "action_type": "diagnose",
                    "response": "Free-text diagnosis and fix explanation",
                },
            },
        ]
    }


@app.get("/grader")
def get_grader():
    return {
        "grader": "ML Experiment Debugger Free-Text Grader",
        "scoring": {
            "bug_identification": 0.4,
            "fix_suggestion": 0.4,
            "explanation_quality": 0.2,
        },
        "score_range": "0.01 to 0.99",
        "description": "Scores agent free-text diagnosis using keyword matching and explanation quality.",
    }


@app.get("/baseline")
def get_baseline():
    return {
        "baseline_scores": {
            "easy": 0.99,
            "medium": 0.99,
            "hard": 0.99,
            "very_hard": 0.99,
            "expert_1": 0.99,
            "expert_2": 0.99,
            "average": 0.99,
        },
        "model": "llama-3.3-70b-versatile (Groq)",
        "description": "Run baseline.py with GROQ_API_KEY to reproduce these scores.",
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()