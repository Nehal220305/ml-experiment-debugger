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
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

parser = argparse.ArgumentParser(description="ML Experiment Debugger inference")
parser.add_argument("--host", type=str, default="http://localhost:7860")
args = parser.parse_args()
BASE_URL = args.host.rstrip("/")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


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

    system_prompt = """You are an expert ML engineer debugging broken training experiments.
Available bugs — use EXACTLY these strings:
- learning_rate_too_high
- data_leakage
- label_noise
- wrong_loss_function

Respond with a valid JSON object only:
{
  "bug_identified": "<bug name>",
  "config_changes": {<key-value fixes>},
  "explanation": "<reasoning>"
}
No markdown, no backticks, no extra text."""

    user_prompt = f"""Task: {task_id}
Message: {message}
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
    print(f"\n{'='*50}")
    print(f"Task: {task_id.upper()}")
    print(f"{'='*50}")

    result = reset_env(task_id)
    observation = result.get("observation", {})
    print(f"Message: {observation.get('message', '')}")

    try:
        action = ask_agent(observation, task_id)
        print(f"Bug identified: {action.get('bug_identified')}")
        print(f"Config changes: {action.get('config_changes')}")
    except Exception as e:
        print(f"Agent failed: {e}")
        action = {"bug_identified": "unknown", "config_changes": {}, "explanation": "failed"}

    fix_result = step_env({
        "action_type": "submit_fix",
        "bug_identified": action.get("bug_identified", "unknown"),
        "config_changes": action.get("config_changes", {}),
        "explanation": action.get("explanation", ""),
    })

    reward = fix_result.get("reward", 0.0) or 0.0
    print(f"Score: {reward:.2f}")
    return float(reward)


def main():
    print("ML Experiment Debugger — Inference")
    print(f"Server: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")

    scores = {}
    for task_id in ["easy", "medium", "hard", "very_hard"]:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            scores[task_id] = 0.0

    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:<8} {score:.2f}  {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'avg':<8} {avg:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
