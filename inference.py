"""
Inference Script for ML Experiment Debugger Environment
MANDATORY env variables:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier
    HF_TOKEN       Your Hugging Face / API key
"""
import time
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
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def reset_env(task_id: str) -> dict:
    for attempt in range(3):
        try:
            response = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(5)


def step_env(action: dict) -> dict:
    response = requests.post(f"{BASE_URL}/step", json={"action": action}, timeout=30)
    response.raise_for_status()
    return response.json()


def ask_agent(observation: dict, task_id: str) -> dict:
    training_log = "\n".join(observation.get("training_log", [])[:8])
    current_config = json.dumps(observation.get("current_config", {}))
    message = observation.get("message", "")

    system_prompt = 'You are an ML debugging expert. Respond with JSON only: {"action_type": "diagnose", "response": "your detailed text analysis identifying the bug and fix"}'

    user_prompt = f"""Task: {task_id}
Message: {message}
Training log:
{training_log}
Config: {current_config}
Diagnose the bug and suggest a fix. Your response field must be a plain text string."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    import re
    raw = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', raw)

    parsed = json.loads(raw)

    # Ensure response is always a plain string
    if isinstance(parsed.get("response"), dict):
        parsed["response"] = " ".join(str(v) for v in parsed["response"].values())
    elif not isinstance(parsed.get("response"), str):
        parsed["response"] = str(parsed.get("response", ""))

    return parsed


def run_task(task_id: str) -> float:
    rewards = []
    steps_taken = 0
    score = 0.01
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
            "action_type": action.get("action_type", "diagnose"),
            "response": action.get("response", ""),
            "bug_identified": action.get("bug_identified", ""),
            "config_changes": action.get("config_changes", {}),
            "explanation": action.get("explanation", ""),
        })

        reward = fix_result.get("reward", 0.0) or 0.0
        reward = max(0.01, min(0.99, float(reward)))
        done = fix_result.get("done", True)
        rewards.append(reward)
        steps_taken = 1
        score = reward
        success = score >= 0.5

        log_step(step=1, action=action_str, reward=reward, done=done, error=None)

    except Exception as e:
        rewards.append(0.01)
        steps_taken = 1
        score = 0.01
        success = False
        log_step(step=1, action="error", reward=0.01, done=True, error=str(e))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def wait_for_server(max_wait: int = 120) -> bool:
    print(f"Waiting for server at {BASE_URL}...", flush=True)
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"Server ready after {time.time()-start:.1f}s", flush=True)
                return True
        except:
            pass
        time.sleep(3)
    print("ERROR: Server did not start in time", flush=True)
    return False


def main():
    wait_for_server()
    scores = {}
    for task_id in ["easy", "medium", "hard", "very_hard", "expert_1", "expert_2"]:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"Task {task_id} failed: {e}", flush=True)
            scores[task_id] = 0.01

    avg = sum(scores.values()) / len(scores)
    print(f"\nFINAL AVG SCORE: {avg:.2f}", flush=True)


if __name__ == "__main__":
    main()