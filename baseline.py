"""
Baseline inference script for the ML Experiment Debugger Environment.
Uses Groq API to run an LLM agent against all 3 tasks.

Usage:
    export GROQ_API_KEY=your_key_here
    python baseline.py --host http://localhost:7860
"""

import os
import json
import argparse
import requests
from groq import Groq

parser = argparse.ArgumentParser(description="ML Experiment Debugger baseline")
parser.add_argument(
    "--host",
    type=str,
    default="http://localhost:7860",
    help="Base URL of the running environment server",
)
args = parser.parse_args()
BASE_URL = args.host.rstrip("/")

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def reset_env(task_id: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def step_env(action: dict) -> dict:
    response = requests.post(
        f"{BASE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def ask_agent(observation: dict, task_id: str) -> dict:
    training_log = "\n".join(observation.get("training_log", []))
    current_config = json.dumps(observation.get("current_config", {}), indent=2)
    message = observation.get("message", "")
    hint = observation.get("hint", "")

    system_prompt = """You are an expert ML engineer debugging broken training experiments.
You will receive a training log and config showing a broken ML experiment.
Your job is to identify the bug and fix the config.

Available bugs — use EXACTLY these strings:
- learning_rate_too_high: learning rate causes NaN loss
- data_leakage: validation set is same as training set
- label_noise: training labels are corrupted
- wrong_loss_function: wrong loss function for classification
- vanishing_gradients: deep network with sigmoid causes gradients to vanish
- missing_normalization: unnormalized inputs cause unstable training

You must respond with a valid JSON object with these exact fields:
{
  "bug_identified": "<one of the three bug names above>",
  "config_changes": {<key-value fixes to apply to the config>},
  "explanation": "<your reasoning>"
}

Only respond with the JSON object. No markdown, no backticks, no extra text."""

    user_prompt = f"""Task: {task_id}
Message: {message}
Hint: {hint if hint else 'None'}

Training log:
{training_log}

Current config:
{current_config}

Diagnose the bug and provide the fix as JSON."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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
    raw = raw.strip()

    return json.loads(raw)


def run_task(task_id: str) -> float:
    print(f"\n{'='*50}")
    print(f"Task: {task_id.upper()}")
    print(f"{'='*50}")

    result = reset_env(task_id)
    observation = result.get("observation", {})

    print(f"Message: {observation.get('message', '')}")
    print(f"Training log (first 3 steps):")
    for line in observation.get("training_log", [])[:3]:
        print(f"  {line}")
    print("  ...")

    try:
        action = ask_agent(observation, task_id)
        print(f"\nAgent action:")
        print(f"  Bug identified: {action.get('bug_identified')}")
        print(f"  Config changes: {action.get('config_changes')}")
        print(f"  Explanation: {action.get('explanation', '')[:120]}...")
    except Exception as e:
        print(f"Agent failed to parse response: {e}")
        action = {
            "bug_identified": "unknown",
            "config_changes": {},
            "explanation": "Failed to parse",
        }

    fix_result = step_env({
        "action_type": "submit_fix",
        "bug_identified": action.get("bug_identified", "unknown"),
        "config_changes": action.get("config_changes", {}),
        "explanation": action.get("explanation", ""),
    })

    final_reward = fix_result.get("reward", 0.0) or 0.0
    fix_obs = fix_result.get("observation", {})
    print(f"Fix result: {fix_obs.get('message', '')}")
    print(f"Final score: {final_reward:.2f}")

    return float(final_reward)


def main():
    print("ML Experiment Debugger — Baseline Evaluation")
    print(f"Server: {BASE_URL}")
    print(f"Model: llama-3.3-70b-versatile (Groq)")

    scores = {}
    for task_id in ["easy", "medium", "hard", "very_hard", "expert_1", "expert_2"]:
        try:
            score = run_task(task_id)
            scores[task_id] = score
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            scores[task_id] = 0.0

    print(f"\n{'='*50}")
    print("BASELINE RESULTS")
    print(f"{'='*50}")
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:<8} {score:.2f}  {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'avg':<8} {avg:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()