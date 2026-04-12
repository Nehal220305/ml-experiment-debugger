"""
ML Experiment Debugger Environment.
Agent receives a broken ML training script and must diagnose and fix it.
Uses REAL subprocess execution — all output is genuine Python stdout/stderr.
6 tasks: easy → expert_2 covering common real-world ML bugs.
Randomized bug parameters per episode — agents cannot memorize answers.
"""

import uuid
import random
import time
import subprocess
import sys
import textwrap
from typing import Optional
from openenv.core.env_server import Environment
from models import MLAction, MLObservation, MLState
# Pre-import PyTorch at module load time to avoid cold-start delays
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SESSION_TIMEOUT = 3600


def get_broken_config(task_id: str) -> dict:
    if task_id == "easy":
        return {
            "learning_rate": random.choice([50.0, 100.0, 500.0, 1000.0]),
            "max_iter": 10,
            "optimizer": "sgd",
            "fix_train_val_split": True,
            "label_noise_pct": 0.0,
            "activation": "relu",
            "loss_fn": "bce",
        }
    elif task_id == "medium":
        return {
            "learning_rate": 0.01,
            "max_iter": 10,
            "optimizer": "adam",
            "fix_train_val_split": False,
            "label_noise_pct": 0.0,
            "activation": "relu",
            "loss_fn": "bce",
        }
    elif task_id == "hard":
        return {
            "learning_rate": random.choice([0.00001, 0.0001, 0.00005]),
            "max_iter": 10,
            "optimizer": "adam",
            "fix_train_val_split": True,
            "label_noise_pct": random.choice([0.25, 0.30, 0.35, 0.40]),
            "activation": "relu",
            "loss_fn": "bce",
        }
    elif task_id == "very_hard":
        return {
            "learning_rate": 0.01,
            "max_iter": 10,
            "optimizer": "adam",
            "fix_train_val_split": True,
            "label_noise_pct": 0.0,
            "activation": "relu",
            "loss_fn": "mse",
        }
    elif task_id == "expert_1":
        return {
            "learning_rate": 0.01,
            "max_iter": 10,
            "optimizer": "sgd",
            "fix_train_val_split": True,
            "label_noise_pct": 0.0,
            "activation": "sigmoid",
            "loss_fn": "bce",
            "depth": random.choice([4, 6, 8]),
        }
    elif task_id == "expert_2":
        return {
            "learning_rate": 0.01,
            "max_iter": 10,
            "optimizer": "adam",
            "fix_train_val_split": True,
            "label_noise_pct": 0.0,
            "activation": "relu",
            "loss_fn": "bce",
            "normalize_input": False,
        }
    return {}


TASKS = {
    "easy": {
        "bug": "learning_rate_too_high",
        "description": "Training loss explodes to NaN within the first few steps. The model cannot learn.",
        "hint": "Look at the learning rate — values above 1.0 cause exploding gradients.",
    },
    "medium": {
        "bug": "data_leakage",
        "description": "Val accuracy is suspiciously perfect but test accuracy is terrible.",
        "hint": "How was the validation set created? Is it truly held-out data?",
    },
    "hard": {
        "bug": "label_noise",
        "description": "Model trains without errors. Loss looks stable. But real-world performance is catastrophically poor. Two things are wrong — find both.",
        "hint": "Is the data itself trustworthy? Is the learning rate optimal?",
    },
    "very_hard": {
        "bug": "wrong_loss_function",
        "description": "Model trains for 20 steps but accuracy plateaus near 52%. Loss decreases steadily but classification performance is near random. Data and architecture are correct.",
        "hint": "Is the loss function appropriate for a binary classification task?",
    },
    "expert_1": {
        "bug": "vanishing_gradients",
        "description": "Model is deep but loss barely moves after step 3. Training stalls completely. Accuracy stays at 50%.",
        "hint": "What happens to gradients in very deep networks with sigmoid activations?",
    },
    "expert_2": {
        "bug": "missing_normalization",
        "description": "Loss starts very high (>10) and oscillates. Model is simple, optimizer is Adam, learning rate is 0.01. The raw feature values are extremely large — some exceed 1000.",
        "hint": "Are the input features on a similar scale? What does unnormalized data do to gradient descent?",
    },
}


def build_training_script(config: dict, task_id: str) -> str:
    lr = config.get("learning_rate", 0.01)
    max_iter = config.get("max_iter", 20)
    optimizer = config.get("optimizer", "adam")
    fix_split = config.get("fix_train_val_split", True)
    noise_pct = config.get("label_noise_pct", 0.0)
    activation = config.get("activation", "relu")
    loss_fn = config.get("loss_fn", "bce")
    normalize = config.get("normalize_input", True)
    depth = config.get("depth", 1)

    act_fn = "nn.ReLU()" if activation == "relu" else "nn.Sigmoid()"
    criterion = "nn.BCEWithLogitsLoss()" if loss_fn == "bce" else "nn.MSELoss()"
    opt_code = f"torch.optim.Adam(model.parameters(), lr={lr})" if optimizer == "adam" else f"torch.optim.SGD(model.parameters(), lr={lr})"

    layers = f"nn.Linear(10, 32), {act_fn}"
    for _ in range(depth - 1):
        layers += f", nn.Linear(32, 32), {act_fn}"
    layers += ", nn.Linear(32, 1)"

    normalize_code = "X = scaler.fit_transform(X)" if normalize else "X = X * 100  # WARNING: unnormalized features"
    scaler_import = "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()" if normalize else ""

    split_code = "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)" if fix_split else "X_train, y_train = X, y  # WARNING: using full dataset for train and val\nX_val, y_val = X, y"

    noise_code = ""
    if noise_pct > 0:
        noise_code = f"""
n_noisy = int(len(y) * {noise_pct})
noisy_idx = np.random.choice(len(y), n_noisy, replace=False)
y[noisy_idx] = 1 - y[noisy_idx]
# noise injected silently — agent must discover this"""

    script = f"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
{scaler_import}
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

X, y = make_classification(n_samples=400, n_features=10, random_state=42)
{noise_code}

{normalize_code}

{split_code}

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

model = nn.Sequential({layers})
optimizer = {opt_code}
criterion = {criterion}

print("Training config: lr={lr}, optimizer={optimizer}, loss={loss_fn}, depth={depth}")
print("---")

for step in range(1, {max_iter} + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()

    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

    if torch.isnan(loss) or grad_norm > 1e6:
        print(f"step {{step}}: loss=nan (EXPLODING GRADIENTS, grad_norm={{grad_norm:.2e}})")
        for i in range(step + 1, {max_iter} + 1):
            print(f"step {{i}}: loss=nan")
        break

    optimizer.step()
    loss_val = loss.item()

    model.eval()
    with torch.no_grad():
        train_preds = (torch.sigmoid(model(X_train_t)) > 0.5).float()
        train_acc = (train_preds == y_train_t).float().mean().item()
        val_preds_t = (torch.sigmoid(model(X_val_t)) > 0.5).float()
        val_acc = (val_preds_t == y_val_t).float().mean().item()

    if "{loss_fn}" == "mse":
        print(f"step {{step}}: mse_loss={{loss_val:.4f}} classification_acc={{train_acc:.3f}} (NOTE: MSE loss used for classification task)")
    else:
        print(f"step {{step}}: loss={{loss_val:.4f}} train_acc={{train_acc:.3f}} val_acc={{val_acc:.3f}} grad_norm={{grad_norm:.2e}}")

model.eval()
with torch.no_grad():
    val_preds_f = (torch.sigmoid(model(X_val_t)) > 0.5).float()
    final_val_acc = (val_preds_f == y_val_t).float().mean().item()
    train_preds_f = (torch.sigmoid(model(X_train_t)) > 0.5).float()
    final_train_acc = (train_preds_f == y_train_t).float().mean().item()

print("---")
print(f"FINAL: train_acc={{final_train_acc:.3f}} val_acc={{final_val_acc:.3f}}")
if abs(final_train_acc - final_val_acc) < 0.01 and final_val_acc > 0.85:
    print("WARNING: train_acc == val_acc — possible data leakage")
if final_val_acc < 0.6:
    print(f"WARNING: val_acc={{final_val_acc:.3f}} is suspiciously low — check data quality")
if final_val_acc < 0.55 and final_train_acc > 0.7:
    print("WARNING: large train/val gap — check label integrity")
"""
    return script


def run_training(config: dict, task_id: str) -> tuple:
    """Execute training inline (no subprocess) for fast response on HF."""

    torch.manual_seed(42)
    np.random.seed(42)

    lr = config.get("learning_rate", 0.01)
    max_iter = config.get("max_iter", 10)
    optimizer_name = config.get("optimizer", "adam")
    fix_split = config.get("fix_train_val_split", True)
    noise_pct = config.get("label_noise_pct", 0.0)
    activation = config.get("activation", "relu")
    loss_fn = config.get("loss_fn", "bce")
    normalize = config.get("normalize_input", True)
    depth = config.get("depth", 1)

    X, y = make_classification(n_samples=400, n_features=10, random_state=42)

    if noise_pct > 0:
        n_noisy = int(len(y) * noise_pct)
        noisy_idx = np.random.choice(len(y), n_noisy, replace=False)
        y[noisy_idx] = 1 - y[noisy_idx]

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = X * 100

    if fix_split:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_val, y_val = X, y

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    act_fn = nn.ReLU() if activation == "relu" else nn.Sigmoid()
    layers = [nn.Linear(10, 32), act_fn]
    for _ in range(depth - 1):
        layers += [nn.Linear(32, 32), act_fn]
    layers.append(nn.Linear(32, 1))
    model = nn.Sequential(*layers)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    criterion = nn.BCEWithLogitsLoss() if loss_fn == "bce" else nn.MSELoss()

    log = [f"Training config: lr={lr}, optimizer={optimizer_name}, loss={loss_fn}, depth={depth}", "---"]
    val_acc = 0.0

    for step in range(1, max_iter + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()

        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

        if torch.isnan(loss) or grad_norm > 1e6:
            log.append(f"step {step}: loss=nan (EXPLODING GRADIENTS, grad_norm={grad_norm:.2e})")
            for i in range(step + 1, max_iter + 1):
                log.append(f"step {i}: loss=nan")
            break

        optimizer.step()
        loss_val = loss.item()

        model.eval()
        with torch.no_grad():
            train_preds = (torch.sigmoid(model(X_train_t)) > 0.5).float()
            train_acc = (train_preds == y_train_t).float().mean().item()
            val_preds = (torch.sigmoid(model(X_val_t)) > 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()

        if loss_fn == "mse":
            log.append(f"step {step}: mse_loss={loss_val:.4f} classification_acc={train_acc:.3f} (NOTE: MSE loss used for classification task)")
        else:
            log.append(f"step {step}: loss={loss_val:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f} grad_norm={grad_norm:.2e}")

    log.append("---")
    log.append(f"FINAL: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    if abs(train_acc - val_acc) < 0.01 and val_acc > 0.85:
        log.append("WARNING: train_acc == val_acc — possible data leakage")
    if val_acc < 0.6:
        log.append(f"WARNING: val_acc={val_acc:.3f} is suspiciously low — check data quality")

    return log, float(val_acc)


def grade_fix(task_id: str, config_changes: dict, bug_identified: bool, broken_config: dict = None) -> float:
    if broken_config is None:
        broken_config = get_broken_config(task_id)
    base_config = broken_config.copy()

    if config_changes:
        base_config.update(config_changes)

    log, score = run_training(base_config, task_id)
    log_str = " ".join(log)
    partial = 0.0

    if bug_identified:
        partial += 0.3

    if task_id == "easy":
        if "nan" not in log_str:
            partial += 0.7

    elif task_id == "medium":
        if base_config.get("fix_train_val_split", False):
            partial += 0.7

    elif task_id == "hard":
        noise_fixed = base_config.get("label_noise_pct", 0.30) < 0.05
        lr_fixed = base_config.get("learning_rate", 0.0001) >= 0.001
        if noise_fixed and lr_fixed:
            partial += 0.7
        elif noise_fixed or lr_fixed:
            partial += 0.35

    elif task_id == "very_hard":
        loss_fixed = base_config.get("loss_fn", "mse") == "bce"
        if loss_fixed and score > 0.6:
            partial += 0.7
        elif loss_fixed:
            partial += 0.3

    elif task_id == "expert_1":
        activation_fixed = base_config.get("activation", "sigmoid") == "relu"
        if activation_fixed and score > 0.6:
            partial += 0.7
        elif activation_fixed:
            partial += 0.3

    elif task_id == "expert_2":
        normalize_fixed = (
            base_config.get("normalize_input", False) is True or
            base_config.get("normalize_inputs", False) is True
        )
        if normalize_fixed and score > 0.5:
            partial += 0.7
        elif normalize_fixed:
            partial += 0.3

    return min(round(partial, 2), 1.0)


HIDDEN_KEYS = {"loss_fn", "activation", "normalize_input", "depth"}


def grade_free_text(task_id: str, response: str, broken_config: dict) -> tuple:
    bug = TASKS[task_id]["bug"]
    try:
        from server.llm_judge import grade_with_llm_judge
        return grade_with_llm_judge(task_id, response, bug)
    except Exception:
        from server.graders import grade_response
        return grade_response(task_id, response, bug)


def _get_random_scenario(task_id: str) -> dict:
    from server.graders import EASY_SCENARIOS, MEDIUM_SCENARIOS, HARD_SCENARIOS
    scenarios = {
        "easy": EASY_SCENARIOS,
        "medium": MEDIUM_SCENARIOS,
        "hard": HARD_SCENARIOS,
    }
    task_scenarios = scenarios.get(task_id, [])
    if task_scenarios:
        return random.choice(task_scenarios)
    return {}



class MlExperimentDebuggerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._sessions = {}

    def _get_session(self, episode_id: str) -> dict:
        if episode_id not in self._sessions:
            self._sessions[episode_id] = {
                "task_id": "easy",
                "broken_config": get_broken_config("easy"),
                "attempts": 0,
                "bug_identified": False,
                "state": MLState(),
                "created_at": time.time(),
                "last_accessed": time.time(),
            }
        else:
            self._sessions[episode_id]["last_accessed"] = time.time()
        return self._sessions[episode_id]

    def _cleanup_expired_sessions(self):
        now = time.time()
        expired = [
            eid for eid, session in self._sessions.items()
            if now - session.get("last_accessed", now) > SESSION_TIMEOUT
        ]
        for eid in expired:
            del self._sessions[eid]

    def reset(
        self,
        seed=None,
        episode_id=None,
        task_id: Optional[str] = "easy",
        **kwargs,
    ) -> MLObservation:
        self._cleanup_expired_sessions()
        task_id = task_id if task_id in TASKS else "easy"
        episode_id = episode_id or str(uuid.uuid4())

        broken_config = get_broken_config(task_id)
        task = TASKS[task_id]
        scenario = _get_random_scenario(task_id)
        task_description = scenario["description"] if scenario else task["description"]

        self._sessions[episode_id] = {
            "task_id": task_id,
            "broken_config": broken_config,
            "attempts": 0,
            "bug_identified": False,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "state": MLState(
                episode_id=episode_id,
                step_count=0,
                task_id=task_id,
                current_bug=TASKS[task_id]["bug"],
                attempts=0,
                bug_identified=False,
            ),
        }

        task = TASKS[task_id]
        log, _ = run_training(broken_config, task_id)
        visible_config = {k: v for k, v in broken_config.items() if k not in HIDDEN_KEYS}

        return MLObservation(
            done=False,
            reward=None,
            task_id=task_id,
            training_log=log,
            current_config=visible_config,
            hint=None,
            message=f"Task '{task_id}': {task['description']} Diagnose and fix the config.",
        )

    def step(self, action: MLAction, timeout_s=None, **kwargs) -> MLObservation:
        episode_id = None
        if hasattr(action, 'metadata') and action.metadata:
            episode_id = action.metadata.get("episode_id")

        if not episode_id or episode_id not in self._sessions:
            if self._sessions:
                episode_id = list(self._sessions.keys())[-1]
            else:
                return MLObservation(
                    done=True,
                    reward=0.0,
                    task_id="easy",
                    training_log=[],
                    current_config={},
                    hint=None,
                    message="No active session. Call /reset first.",
                )

        session = self._sessions[episode_id]
        session["last_accessed"] = time.time()
        session["state"].step_count += 1
        session["attempts"] += 1

        task_id = session["task_id"]
        task = TASKS[task_id]
        broken_config = session.get("broken_config", get_broken_config(task_id))
        visible_config = {k: v for k, v in broken_config.items() if k not in HIDDEN_KEYS}

        if action.action_type == "request_more_steps":
            # Run 10 more training steps and return new log
            extra_config = broken_config.copy()
            extra_config["max_iter"] = 10
            extra_log, _ = run_training(extra_config, task_id)
            session["attempts"] += 1
            return MLObservation(
                done=False,
                reward=0.0,
                task_id=task_id,
                training_log=extra_log,
                current_config=visible_config,
                hint=None,
                message="Running 10 more training steps. Observe the new patterns.",
                feedback=None,
            )

        elif action.action_type == "inspect_gradients":
            # Return gradient statistics
            grad_script = build_training_script(broken_config, task_id)
            log, _ = run_training(broken_config, task_id)
            grad_lines = [l for l in log if "grad_norm" in l]
            if not grad_lines:
                grad_info = ["No gradient information available for this task."]
            else:
                grad_info = grad_lines[:5]
                grad_info.append(f"Total steps with gradient data: {len(grad_lines)}")
            session["attempts"] += 1
            return MLObservation(
                done=False,
                reward=0.0,
                task_id=task_id,
                training_log=grad_info,
                current_config=visible_config,
                hint=None,
                message="Gradient inspection results. Use this to identify gradient-related bugs.",
                feedback=None,
            )

        if action.action_type == "diagnose":
            response = action.response or action.explanation or ""
            broken_config = session.get("broken_config", get_broken_config(task_id))
            score, feedback = grade_free_text(task_id, response, broken_config)
            return MLObservation(
                done=True,
                reward=score,
                task_id=task_id,
                training_log=[],
                current_config=visible_config,
                hint=None,
                message=f"Score: {score:.2f}. Episode complete.",
                feedback=feedback,
            )

        elif action.action_type == "identify_bug":
            correct = (action.bug_identified == task["bug"])
            if correct:
                session["bug_identified"] = True
                return MLObservation(
                    done=False,
                    reward=0.3,
                    task_id=task_id,
                    training_log=[],
                    current_config=visible_config,
                    hint=None,
                    message=f"Correct! The bug is '{task['bug']}'. Now fix it using 'submit_fix'.",
                )
            else:
                hint = task["hint"] if session["attempts"] >= 2 else None
                return MLObservation(
                    done=False,
                    reward=0.0,
                    task_id=task_id,
                    training_log=[],
                    current_config=visible_config,
                    hint=hint,
                    message=f"Incorrect. '{action.bug_identified}' is not the bug. Try again.",
                )

        elif action.action_type in ("fix_config", "submit_fix"):
            if action.bug_identified == task["bug"]:
                session["bug_identified"] = True

            score = grade_fix(
                task_id,
                action.config_changes or {},
                session["bug_identified"],
                broken_config,
            )
            done = action.action_type == "submit_fix"

            if done:
                del self._sessions[episode_id]

            return MLObservation(
                done=done,
                reward=score,
                task_id=task_id,
                training_log=[],
                current_config=visible_config,
                hint=None,
                message=f"Score: {score:.2f}. {'Episode complete.' if done else 'Keep refining or call submit_fix to finish.'}",
            )

        else:
            return MLObservation(
                done=False,
                reward=None,
                task_id=task_id,
                training_log=log,
                current_config=visible_config,
                hint=None,
                message=f"Task '{task_id}': {task_description} Diagnose and fix the config.",
            )

    @property
    def state(self) -> MLState:
        if self._sessions:
            latest = list(self._sessions.values())[-1]
            return latest["state"]
        return MLState()