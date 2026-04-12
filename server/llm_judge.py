"""
LLM-as-judge grader for ML Experiment Debugger.
Uses HF router to evaluate agent free-text responses.
Falls back to keyword grader if LLM judge fails.
"""

import os
import json
import re
from typing import Tuple
from openai import OpenAI

HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Use HF router if token available, else Groq
if HF_TOKEN:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
    JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
elif GROQ_API_KEY:
    from groq import Groq as GroqClient
    groq_client = GroqClient(api_key=GROQ_API_KEY)
    client = None
    JUDGE_MODEL = "llama-3.3-70b-versatile"
else:
    client = None
    JUDGE_MODEL = None


JUDGE_SYSTEM_PROMPT = """You are an expert ML engineer evaluating an AI agent's diagnosis of a broken ML training experiment.

You will be given:
1. The task description
2. The actual bug in the experiment
3. The agent's free-text response

Score the agent's response on these criteria:
- Bug identification (0-40 points): Did the agent correctly identify the bug?
- Fix suggestion (0-40 points): Did the agent suggest a valid, specific fix?
- Explanation quality (0-20 points): Did the agent explain WHY the bug causes the observed symptoms?

Respond with ONLY a JSON object:
{
  "bug_score": <0-40>,
  "fix_score": <0-40>,
  "explanation_score": <0-20>,
  "total": <0-100>,
  "feedback": "<one sentence summary>"
}

Be strict — only give full bug_score if the agent correctly named the specific bug.
Only give full fix_score if the agent suggested a concrete, actionable fix with specific values."""


TASK_CONTEXT = {
    "easy": {
        "bug": "learning_rate_too_high",
        "description": "Learning rate is too high causing gradient explosion and NaN loss",
        "correct_fix": "Reduce learning rate to 0.01 or lower",
    },
    "medium": {
        "bug": "data_leakage",
        "description": "Validation set is same as training set causing misleadingly perfect accuracy",
        "correct_fix": "Fix the train/val split so validation data is held out",
    },
    "hard": {
        "bug": "label_noise + bad_learning_rate",
        "description": "25-40% of labels are corrupted AND learning rate is too low",
        "correct_fix": "Remove label noise (set to 0) AND increase learning rate to 0.01",
    },
    "very_hard": {
        "bug": "wrong_loss_function",
        "description": "MSE loss used for binary classification instead of cross-entropy",
        "correct_fix": "Change loss function to binary cross-entropy",
    },
    "expert_1": {
        "bug": "vanishing_gradients",
        "description": "Deep sigmoid network causing vanishing gradients — loss stalls",
        "correct_fix": "Change activation from sigmoid to ReLU, or use batch normalization",
    },
    "expert_2": {
        "bug": "missing_normalization",
        "description": "Input features have very large values (>1000) causing unstable training",
        "correct_fix": "Normalize input features using StandardScaler before training",
    },
}

DIFFICULTY_CAPS = {
    "easy": 0.95,
    "medium": 0.85,
    "hard": 0.80,
    "very_hard": 0.75,
    "expert_1": 0.60,
    "expert_2": 0.55,
}


def judge_with_llm(task_id: str, response: str) -> Tuple[float, str]:
    """Use LLM to judge the agent's response."""
    context = TASK_CONTEXT.get(task_id, {})
    cap = DIFFICULTY_CAPS.get(task_id, 0.80)

    user_prompt = f"""Task: {task_id}
Actual bug: {context.get('bug', 'unknown')}
Bug description: {context.get('description', '')}
Correct fix: {context.get('correct_fix', '')}

Agent's response:
{response}

Score this response."""

    try:
        if client:
            # HF router or OpenAI-compatible
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = completion.choices[0].message.content.strip()
        elif groq_client:
            # Groq fallback
            completion = groq_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = completion.choices[0].message.content.strip()
        else:
            return None, None

        # Parse JSON response
        raw = re.sub(r'```json|```', '', raw).strip()
        result = json.loads(raw)

        total = result.get("total", 0) / 100.0
        feedback = result.get("feedback", "")

        bug_score = result.get("bug_score", 0)
        fix_score = result.get("fix_score", 0)
        exp_score = result.get("explanation_score", 0)

        feedback_parts = []
        if bug_score >= 30:
            feedback_parts.append("✓ Correctly identified bug")
        elif bug_score >= 15:
            feedback_parts.append("~ Partially identified bug")
        else:
            feedback_parts.append("✗ Did not identify bug")

        if fix_score >= 30:
            feedback_parts.append("✓ Suggested valid fix")
        elif fix_score >= 15:
            feedback_parts.append("~ Partial fix suggestion")
        else:
            feedback_parts.append("✗ No valid fix")

        if exp_score >= 15:
            feedback_parts.append("✓ Good explanation")

        feedback_parts.append(f"[LLM-judge: {feedback}]")

        # Apply difficulty cap
        score = min(total, cap)
        score = max(0.01, min(0.99, round(score, 2)))

        return score, " | ".join(feedback_parts)

    except Exception as e:
        error_str = str(e)
        if "rate_limit" in error_str or "429" in error_str:
            # Rate limited — fall back to keyword grader silently
            return None, None
        return None, f"LLM judge failed: {error_str}"


def grade_with_llm_judge(task_id: str, response: str, bug: str) -> Tuple[float, str]:
    """
    Grade using LLM judge with keyword fallback.
    Returns (score, feedback).
    """
    # Try LLM judge first
    score, feedback = judge_with_llm(task_id, response)

    if score is not None:
        return score, feedback

    # Fall back to keyword grader
    from server.graders import grade_response
    return grade_response(task_id, response, bug)