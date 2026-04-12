"""
Pure-Python grader logic for ML Experiment Debugger.
Zero external dependencies — instantly importable for testing.
Implements position-aware scoring, negation filtering, and difficulty caps.
"""

from typing import List, Tuple
import random

EASY_SCENARIOS = [
    {
        "id": "easy_lr",
        "description": "Training loss explodes to NaN within the first few steps. The model cannot learn.",
        "bug": "learning_rate_too_high",
        "config_override": {"learning_rate": random.choice([50.0, 100.0, 500.0, 1000.0]), "optimizer": "sgd"},
    },
    {
        "id": "easy_lr_2",
        "description": "Loss spikes to infinity after step 2. Gradient norms are extremely large.",
        "bug": "learning_rate_too_high",
        "config_override": {"learning_rate": random.choice([10.0, 20.0, 50.0]), "optimizer": "sgd"},
    },
    {
        "id": "easy_lr_3",
        "description": "Model weights diverge immediately. Training is completely unstable.",
        "bug": "learning_rate_too_high",
        "config_override": {"learning_rate": random.choice([100.0, 200.0]), "optimizer": "adam"},
    },
]

MEDIUM_SCENARIOS = [
    {
        "id": "medium_leakage_1",
        "description": "Val accuracy is suspiciously perfect but test accuracy is terrible.",
        "bug": "data_leakage",
        "config_override": {"fix_train_val_split": False},
    },
    {
        "id": "medium_leakage_2",
        "description": "Validation loss is identical to training loss at every step. Something is wrong with the data split.",
        "bug": "data_leakage",
        "config_override": {"fix_train_val_split": False, "learning_rate": 0.001},
    },
    {
        "id": "medium_leakage_3",
        "description": "Model achieves 99% validation accuracy on step 1 — impossibly fast convergence.",
        "bug": "data_leakage",
        "config_override": {"fix_train_val_split": False, "optimizer": "sgd"},
    },
]

HARD_SCENARIOS = [
    {
        "id": "hard_noise_1",
        "description": "Model trains without errors. Loss looks stable. But real-world performance is catastrophically poor. Two things are wrong — find both.",
        "bug": "label_noise",
        "config_override": {"label_noise_pct": random.choice([0.25, 0.30, 0.35, 0.40]), "learning_rate": random.choice([0.00001, 0.0001, 0.00005])},
    },
    {
        "id": "hard_noise_2",
        "description": "Training appears healthy but deployment accuracy is near random. The data pipeline may be compromised.",
        "bug": "label_noise",
        "config_override": {"label_noise_pct": random.choice([0.30, 0.35]), "learning_rate": 0.00005},
    },
    {
        "id": "hard_noise_3",
        "description": "Loss converges but model fails completely on unseen data. Two configuration issues detected.",
        "bug": "label_noise",
        "config_override": {"label_noise_pct": random.choice([0.25, 0.40]), "learning_rate": 0.00001},
    },
]


def safe_reward(raw: float) -> float:
    """Clamp reward strictly between 0.01 and 0.99."""
    return round(min(max(float(raw), 0.01), 0.99), 2)


def negation_filtered_hits(keywords: List[str], response_lower: str) -> int:
    """Count keyword hits with negation filtering."""
    hits = 0
    for kw in keywords:
        if kw in response_lower:
            if f"not {kw}" not in response_lower and f"no {kw}" not in response_lower:
                hits += 1
    return hits


def grade_easy(response: str, bug: str) -> Tuple[float, str]:
    """
    Easy task grader — single clear bug.
    Max score: 0.95
    """
    r = response.lower()
    score = 0.0
    feedback = []

    bug_keywords = {
        "learning_rate_too_high": [
            "learning rate", "lr", "exploding gradient", "nan",
            "too high", "gradient explosion", "step size", "rate is"
        ],
        "data_leakage": [
            "data leakage", "leakage", "validation set", "train val split",
            "same data", "identical accuracy", "overfitting", "data leak"
        ],
        "label_noise": [
            "label noise", "noisy label", "corrupted label", "label corruption",
            "wrong label", "mislabeled", "flipped label", "noise"
        ],
        "wrong_loss_function": [
            "loss function", "mse", "mean squared error", "cross entropy",
            "bce", "binary cross", "classification loss", "wrong loss",
            "classification task", "for classification"
        ],
        "vanishing_gradients": [
            "vanishing gradient", "sigmoid", "gradient vanish", "deep network",
            "gradient flow", "activation function", "relu", "barely moves",
            "stalls", "dying"
        ],
        "missing_normalization": [
            "normalization", "normalize", "feature scale", "standardize",
            "scaler", "large values", "input scale", "feature normalization",
            "lack of normalization", "lack of feature"
        ],
    }

    fix_keywords = {
        "learning_rate_too_high": [
            "reduce", "lower", "decrease", "0.01", "0.001",
            "smaller", "less", "gradient clipping"
        ],
        "data_leakage": [
            "separate", "split", "held-out", "train test split",
            "fix split", "independent validation", "fix_train_val_split"
        ],
        "label_noise": [
            "clean", "remove noise", "fix labels", "label_noise_pct",
            "correct labels", "0.0", "reduce noise"
        ],
        "wrong_loss_function": [
            "cross entropy", "bce", "binary cross", "log loss",
            "cross-entropy", "change the loss", "cross-entropy loss",
            "to cross", "use cross", "replace", "switch"
        ],
        "vanishing_gradients": [
            "relu", "batch norm", "residual", "skip connection",
            "change activation", "adam", "gradient clipping"
        ],
        "missing_normalization": [
            "normalize", "standardize", "standard scaler", "min max",
            "feature scaling", "preprocess", "normalization",
            "standardscaler", "minmaxscaler", "add normalization",
            "normalization layer", "scale the input", "zero mean"
        ],
    }

    # Bug identification (0.4)
    keywords = bug_keywords.get(bug, [])
    hits = negation_filtered_hits(keywords, r)
    if hits >= 3:
        score += 0.4
        feedback.append("✓ Correctly identified the bug")
    elif hits >= 2:
        score += 0.2
        feedback.append("~ Partially identified the bug")
    else:
        feedback.append("✗ Did not identify the bug")

    # Fix suggestion (0.4)
    fix_kws = fix_keywords.get(bug, [])
    fix_hits = negation_filtered_hits(fix_kws, r)
    if fix_hits >= 1:
        score += 0.4
        feedback.append("✓ Suggested a valid fix")
    else:
        feedback.append("✗ Did not suggest a valid fix")

    # Explanation quality (0.2)
    causal_terms = ["because", "due to", "since", "causes", "resulting",
                    "as a result", "leads to", "indicates", "therefore"]
    has_causal = any(term in r for term in causal_terms)
    if has_causal and len(response) > 100:
        score += 0.1
        feedback.append("✓ Provided causal explanation")
    if len(response) > 200:
        score += 0.1
        feedback.append("✓ Thorough analysis")

    return safe_reward(min(score, 0.95)), " | ".join(feedback)


def grade_medium(response: str, bug: str) -> Tuple[float, str]:
    """
    Medium task grader — requires identifying bug AND explaining symptoms.
    Max score: 0.85
    """
    r = response.lower()
    score = 0.0
    feedback = []

    # Same bug identification but requires causal explanation too
    bug_score, bug_feedback = grade_easy(response, bug)

    # Base score from easy grader but capped lower
    base = min(bug_score, 0.6)
    score = base

    # Additional: must explain WHY the bug causes the observed symptoms
    symptom_terms = ["symptom", "observed", "because", "training log shows",
                     "evident from", "indicates", "log shows", "we can see"]
    symptom_hits = sum(1 for t in symptom_terms if t in r)
    if symptom_hits >= 2:
        score = min(score + 0.15, 0.85)
        feedback.append("✓ Explained observed symptoms")
    elif symptom_hits >= 1:
        score = min(score + 0.05, 0.85)

    # Must mention specific values from the log
    specific_values = ["lr=", "learning_rate", "0.0001", "0.00001",
                       "label_noise_pct", "loss=mse", "mse_loss",
                       "grad_norm", "sigmoid", "normalize"]
    value_hits = sum(1 for v in specific_values if v in r)
    if value_hits >= 2:
        score = min(score + 0.1, 0.85)
        feedback.append("✓ Referenced specific values from log")

    feedback.insert(0, bug_feedback.split(" | ")[0])
    return safe_reward(min(score, 0.85)), " | ".join(feedback)


def grade_hard(response: str, bug: str) -> Tuple[float, str]:
    """
    Hard task grader — requires identifying BOTH bugs.
    Position-aware: penalizes if wrong bug mentioned first.
    Max score: 0.80
    """
    r = response.lower()
    score = 0.0
    feedback = []

    lines = [line for line in r.split("\n") if line.strip()]
    if not lines:
        return safe_reward(0.01), "✗ Empty response"

    third = max(1, len(lines) // 3)
    first_part = " ".join(lines[:third])
    rest_part = " ".join(lines[third:])

    # Hard task has TWO bugs: label_noise + bad learning_rate
    # Primary bug should be label_noise (mentioned first)
    primary_keywords = [
        "label noise", "noisy label", "corrupted", "mislabeled",
        "label corruption", "noise", "flipped"
    ]
    secondary_keywords = [
        "learning rate", "lr", "too low", "too small", "0.0001",
        "slow convergence", "underfitting"
    ]

    # Primary bug in first third (0.35)
    primary_hits = negation_filtered_hits(primary_keywords, first_part)
    if primary_hits >= 1:
        score += 0.35
        feedback.append("✓ Identified primary bug (label noise) first")
    else:
        primary_total = negation_filtered_hits(primary_keywords, r)
        if primary_total >= 1:
            score += 0.15
            feedback.append("~ Identified label noise but not first")
        else:
            feedback.append("✗ Missed label noise bug")

    # Secondary bug anywhere (0.25)
    secondary_hits = negation_filtered_hits(secondary_keywords, r)
    if secondary_hits >= 1:
        score += 0.25
        feedback.append("✓ Identified secondary bug (bad learning rate)")
    else:
        feedback.append("✗ Missed learning rate bug")

    # Fix suggestions for both (0.3)
    fix_primary = ["label_noise_pct", "clean", "fix labels", "remove noise", "0.0"]
    fix_secondary = ["increase", "higher lr", "0.01", "learning_rate", "larger"]
    fix_p_hits = negation_filtered_hits(fix_primary, r)
    fix_s_hits = negation_filtered_hits(fix_secondary, r)

    if fix_p_hits >= 1 and fix_s_hits >= 1:
        score += 0.3
        feedback.append("✓ Suggested fixes for both bugs")
    elif fix_p_hits >= 1 or fix_s_hits >= 1:
        score += 0.15
        feedback.append("~ Suggested fix for one bug")

    # Exclusivity penalty — if agent dumps everything in first line
    if primary_hits >= 1 and secondary_hits >= 1 and len(lines) < 3:
        score -= 0.1
        feedback.append("⚠ Response too brief")

    return safe_reward(min(score, 0.80)), " | ".join(feedback)


def grade_very_hard(response: str, bug: str) -> Tuple[float, str]:
    """
    Very hard — wrong loss function, agent must explain WHY MSE fails for classification.
    Max score: 0.85
    """
    r = response.lower()
    score = 0.0
    feedback = []

    # Must identify wrong loss function
    loss_keywords = ["loss function", "mse", "mean squared error",
                     "wrong loss", "classification loss", "for classification",
                     "classification task", "not suitable", "incorrect loss"]
    hits = negation_filtered_hits(loss_keywords, r)

    if hits >= 3:
        score += 0.35
        feedback.append("✓ Correctly identified wrong loss function")
    elif hits >= 2:
        score += 0.20
        feedback.append("~ Partially identified wrong loss function")
    else:
        feedback.append("✗ Did not identify wrong loss function")

    # Must explain WHY MSE fails for classification
    why_keywords = ["class boundary", "probability", "not suitable for classification",
                    "regression", "class", "binary", "sigmoid output",
                    "optimizes wrong objective", "doesn't optimize"]
    why_hits = sum(1 for kw in why_keywords if kw in r)
    if why_hits >= 1:
        score += 0.25
        feedback.append("✓ Explained why MSE fails for classification")

    # Fix suggestion
    fix_keywords = ["cross entropy", "bce", "binary cross", "cross-entropy",
                    "cross-entropy loss", "log loss", "change loss", "use cross"]
    fix_hits = negation_filtered_hits(fix_keywords, r)
    if fix_hits >= 1:
        score += 0.25
        feedback.append("✓ Suggested correct loss function")

    return safe_reward(min(score, 0.75)), " | ".join(feedback)


def grade_expert_1(response: str, bug: str) -> Tuple[float, str]:
    """
    Expert 1 — vanishing gradients in deep sigmoid network.
    Max score: 0.80
    """
    r = response.lower()
    score = 0.0
    feedback = []

    # Must identify vanishing gradients
    vg_keywords = ["vanishing gradient", "sigmoid", "gradient vanish",
                   "gradient norm", "barely moves", "stalls", "deep network",
                   "gradient flow", "backpropagation"]
    hits = negation_filtered_hits(vg_keywords, r)

    if hits >= 3:
        score += 0.35
        feedback.append("✓ Correctly identified vanishing gradients")
    elif hits >= 2:
        score += 0.20
        feedback.append("~ Partially identified vanishing gradients")
    else:
        feedback.append("✗ Did not identify vanishing gradients")

    # Must explain mechanism
    mechanism_keywords = ["saturation", "saturate", "derivative", "chain rule",
                          "multiplied", "exponentially small", "deep layers",
                          "activation", "gradient becomes"]
    mech_hits = sum(1 for kw in mechanism_keywords if kw in r)
    if mech_hits >= 1:
        score += 0.20
        feedback.append("✓ Explained vanishing gradient mechanism")

    # Fix suggestion
    fix_keywords = ["relu", "batch norm", "residual", "skip connection",
                    "change activation", "adam", "leaky relu",
                    "gradient clipping", "xavier", "he initialization"]
    fix_hits = negation_filtered_hits(fix_keywords, r)
    if fix_hits >= 2:
        score += 0.25
        feedback.append("✓ Suggested multiple valid fixes")
    elif fix_hits >= 1:
        score += 0.15
        feedback.append("~ Suggested one fix")

    return safe_reward(min(score, 0.60)), " | ".join(feedback)


def grade_expert_2(response: str, bug: str) -> Tuple[float, str]:
    """
    Expert 2 — missing normalization causing oscillating loss.
    Max score: 0.80
    """
    r = response.lower()
    score = 0.0
    feedback = []

    # Must identify missing normalization
    norm_keywords = ["normalization", "normalize", "feature scale",
                     "standardize", "large values", "input scale",
                     "feature normalization", "lack of normalization",
                     "unnormalized", "raw feature"]
    hits = negation_filtered_hits(norm_keywords, r)

    if hits >= 3:
        score += 0.35
        feedback.append("✓ Correctly identified missing normalization")
    elif hits >= 2:
        score += 0.20
        feedback.append("~ Partially identified normalization issue")
    else:
        feedback.append("✗ Did not identify normalization issue")

    # Must explain WHY large values cause oscillation
    why_keywords = ["gradient", "oscillat", "large gradient", "unstable",
                    "scale", "magnitude", "weight update", "explod"]
    why_hits = sum(1 for kw in why_keywords if kw in r)
    if why_hits >= 2:
        score += 0.20
        feedback.append("✓ Explained why large values cause oscillation")
    elif why_hits >= 1:
        score += 0.10

    # Fix suggestion
    fix_keywords = ["standardscaler", "minmaxscaler", "standard scaler",
                    "normalize", "standardize", "zero mean", "unit variance",
                    "normalization layer", "preprocess", "scale the"]
    fix_hits = negation_filtered_hits(fix_keywords, r)
    if fix_hits >= 1:
        score += 0.25
        feedback.append("✓ Suggested normalization fix")

    return safe_reward(min(score, 0.55)), " | ".join(feedback)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    "very_hard": grade_very_hard,
    "expert_1": grade_expert_1,
    "expert_2": grade_expert_2,
}


def grade_response(task_id: str, response: str, bug: str) -> Tuple[float, str]:
    """Main entry point for grading."""
    grader = GRADERS.get(task_id, grade_easy)
    return grader(response, bug)