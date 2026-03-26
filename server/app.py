import sys
import os
import uuid

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/server")

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from server.ml_experiment_debugger_environment import MlExperimentDebuggerEnvironment
from models import MLAction

app = FastAPI(
    title="ML Experiment Debugger",
    description="OpenEnv environment for debugging broken ML training experiments.",
)

env = MlExperimentDebuggerEnvironment()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    action: MLAction


@app.post("/reset")
def reset(request: ResetRequest):
    obs = env.reset(
        task_id=request.task_id,
        seed=request.seed,
        episode_id=request.episode_id,
    )
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.post("/step")
def step(request: StepRequest):
    obs = env.step(request.action)
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
            "hard": 0.65,
            "average": 0.88,
        },
        "model": "llama-3.3-70b-versatile (Groq)",
        "description": "Run baseline.py with GROQ_API_KEY to reproduce these scores.",
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()