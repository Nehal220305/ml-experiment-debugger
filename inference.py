"""
Inference Script for ML Experiment Debugger Environment
MANDATORY env variables:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier
    HF_TOKEN       Your Hugging Face / API key
"""

import os
import json
import argparse
import requests
from typing import List, Optional
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") or "dummy"
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
BENCHMARK = "ml-experiment-debugger"

parser = argparse.ArgumentParser(description="ML Experiment Debugger inference")
parser.add_argument("--host", type=str, default="http://localhost:7860")
args = parser.parse_args()
BASE_URL = args.host.rstrip("/")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def reset_env(task_id: str) -> dict:
    response = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    response.raise_for_status()
    return response.json()


def step_env(action: dict) -> dict:
    response = requests.post(f"{BASE_URL}/step", json={"action": action}, timeout=30)
    response.raise_for_status()
    return response.json()


def ask_agent(observation: dict, task_id: str) -> dict:
    training_log = "\n".join(observation.get("training_log", []))
    current_config = json.dumps(observation.get("current_config", {}), indent=2)
    message = observation.get("message", "")
    hint = observation.get("hint", "")

    system_prompt = """You are an expert ML engineer debugging broken training experiments.
Available bugs — use EXACTLY these strings:
- learning_rate_too_high: learning rate causes NaN/exploding loss
- data_leakage: validation set is same as training set
- label_noise: training labels are corrupted
- wrong_loss_function: wrong loss function for classification task
- vanishing_gradients: deep network with sigmoid causes gradients to vanish
- missing_normalization: unnormalized inputs cause unstable training

Respond with a valid JSON object only:
{
  "bug_identified": "<bug name>",
  "config_changes": {<key-value fixes>},
  "explanation": "<reasoning>"
}
No markdown, no backticks, no extra text."""

    user_prompt = f"""Task: {task_id}
Message: {message}
Hint: {hint if hint else 'None'}
Training log:
{training_log}
Current config:
{current_config}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def run_task(task_id: str) -> float:
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = reset_env(task_id)
        observation = result.get("observation", {})

        try:
            action = ask_agent(observation, task_id)
            action_str = action.get("bug_identified", "unknown")
        except Exception as e:
            action = {"bug_identified": "unknown", "config_changes": {}, "explanation": "failed"}
            action_str = "unknown"

        fix_result = step_env({
            "action_type": "submit_fix",
            "bug_identified": action.get("bug_identified", "unknown"),
            "config_changes": action.get("config_changes", {}),
            "explanation": action.get("explanation", ""),
        })

        reward = fix_result.get("reward", 0.0) or 0.0
        done = fix_result.get("done", True)
        rewards.append(reward)
        steps_taken = 1
        score = float(reward)
        score = max(0.001, min(0.999, score))
        success = score >= 0.5

        log_step(step=1, action=action_str, reward=reward, done=done, error=None)

    except Exception as e:
        log_step(step=1, action="error", reward=0.0, done=True, error=str(e))
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    scores = {}
    for task_id in ["easy", "medium", "hard", "very_hard", "expert_1", "expert_2"]:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"Task {task_id} failed: {e}", flush=True)
            scores[task_id] = 0.0

    avg = sum(scores.values()) / len(scores)
    print(f"\nFINAL AVG SCORE: {avg:.2f}", flush=True)


if __name__ == "__main__":
    main()