# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Baseline inference script for the ML Experiment Debugger Environment.
Uses the OpenAI API to run an LLM agent against all 3 tasks.
Reads OPENAI_API_KEY from environment variables.

Usage:
    export OPENAI_API_KEY=your_key_here
    python baseline.py --host http://localhost:7860
"""

import os
import json
import argparse
import requests
from openai import OpenAI

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="ML Experiment Debugger baseline")
parser.add_argument(
    "--host",
    type=str,
    default="http://localhost:7860",
    help="Base URL of the running environment server",
)
args = parser.parse_args()
BASE_URL = args.host.rstrip("/")

# ── OpenAI client ─────────────────────────────────────────────────────────────

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── Helper functions ──────────────────────────────────────────────────────────

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
        json=action,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def ask_agent(observation: dict, task_id: str) -> dict:
    """
    Ask the LLM to diagnose and fix the broken ML config.
    Returns a parsed action dict.
    """
    training_log = "\n".join(observation.get("training_log", []))
    current_config = json.dumps(observation.get("current_config", {}), indent=2)
    message = observation.get("message", "")
    hint = observation.get("hint", "")

    system_prompt = """You are an expert ML engineer debugging broken training experiments.
You will receive a training log and config showing a broken ML experiment.
Your job is to identify the bug and fix the config.

Available bugs:
- learning_rate_too_high: learning rate causes NaN loss
- data_leakage: validation set is same as training set
- label_noise: training labels are corrupted

You must respond with a valid JSON object with these fields:
{
  "action_type": "submit_fix",
  "bug_identified": "<bug name>",
  "config_changes": {<key-value fixes>},
  "explanation": "<your reasoning>"
}

Only respond with the JSON object, nothing else."""

    user_prompt = f"""Task: {task_id}
Message: {message}
Hint: {hint if hint else 'None'}

Training log:
{training_log}

Current config:
{current_config}

Diagnose the bug and provide the fix as JSON."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code blocks if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ── Run one task ──────────────────────────────────────────────────────────────

def run_task(task_id: str) -> float:
    print(f"\n{'='*50}")
    print(f"Task: {task_id.upper()}")
    print(f"{'='*50}")

    # Reset environment
    result = reset_env(task_id)
    observation = result.get("observation", {})

    print(f"Message: {observation.get('message', '')}")
    print(f"Training log (first 3 steps):")
    for line in observation.get("training_log", [])[:3]:
        print(f"  {line}")
    print("  ...")

    # Step 1: Agent identifies and fixes in one shot
    try:
        action = ask_agent(observation, task_id)
        print(f"\nAgent action:")
        print(f"  Bug identified: {action.get('bug_identified')}")
        print(f"  Config changes: {action.get('config_changes')}")
        print(f"  Explanation: {action.get('explanation', '')[:100]}...")
    except Exception as e:
        print(f"Agent failed to parse response: {e}")
        action = {
            "action_type": "submit_fix",
            "bug_identified": "unknown",
            "config_changes": {},
            "explanation": "Failed to parse",
        }

    # Step 2: First identify the bug
    identify_result = step_env({
        "action_type": "identify_bug",
        "bug_identified": action.get("bug_identified", "unknown"),
        "explanation": action.get("explanation", ""),
    })
    identify_obs = identify_result.get("observation", {})
    print(f"\nIdentify result: {identify_obs.get('message', '')}")

    # Step 3: Submit the fix
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("ML Experiment Debugger — Baseline Evaluation")
    print(f"Server: {BASE_URL}")
    print(f"Model: gpt-4o-mini")

    scores = {}
    for task_id in ["easy", "medium", "hard"]:
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