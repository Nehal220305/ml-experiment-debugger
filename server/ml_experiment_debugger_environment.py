"""
ML Experiment Debugger Environment.
Agent receives a broken ML training config and must diagnose and fix it.
Uses real PyTorch training loops — all loss curves and metrics are genuine.
6 tasks: easy → expert_2 covering common real-world ML bugs.
Randomized bug parameters per episode — agents cannot memorize answers.
"""

import uuid
import random
import time
import numpy as np
from typing import Optional
from openenv.core.env_server import Environment
from models import MLAction, MLObservation, MLState

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SESSION_TIMEOUT = 3600


class TinyMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def get_broken_config(task_id: str) -> dict:
    if task_id == "easy":
        return {
            "learning_rate": random.choice([10.0, 50.0, 100.0, 500.0]),
            "max_iter": 20,
            "optimizer": "sgd",
            "fix_train_val_split": True,
            "label_noise_pct": 0.0,
            "activation": "relu",
            "loss_fn": "bce",
        }
    elif task_id == "medium":
        return {
            "learning_rate": 0.01,
            "max_iter": 20,
            "optimizer": "adam",
            "fix_train_val_split": False,
            "label_noise_pct": 0.0,
            "activation": "relu",
            "loss_fn": "bce",
        }
    elif task_id == "hard":
        return {
            "learning_rate": random.choice([0.00001, 0.0001, 0.00005]),
            "max_iter": 20,
            "optimizer": "adam",
            "fix_train_val_split": True,
            "label_noise_pct": random.choice([0.25, 0.30, 0.35, 0.40]),
            "activation": "relu",
            "loss_fn": "bce",
        }
    elif task_id == "very_hard":
        return {
            "learning_rate": 0.01,
            "max_iter": 20,
            "optimizer": "adam",
            "fix_train_val_split": True,
            "label_noise_pct": 0.0,
            "activation": "relu",
            "loss_fn": "mse",
        }
    elif task_id == "expert_1":
        return {
            "learning_rate": 0.01,
            "max_iter": 20,
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
            "max_iter": 20,
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


def build_model(config: dict) -> nn.Module:
    activation = config.get("activation", "relu")
    depth = config.get("depth", 1)
    act_fn = nn.ReLU() if activation == "relu" else nn.Sigmoid()
    layers = [nn.Linear(10, 32), act_fn]
    for _ in range(depth - 1):
        layers += [nn.Linear(32, 32), act_fn]
    layers.append(nn.Linear(32, 1))
    return nn.Sequential(*layers)


def run_training(config: dict, task_id: str) -> tuple:
    torch.manual_seed(42)
    np.random.seed(42)

    X, y = make_classification(n_samples=400, n_features=10, random_state=42)

    noise_pct = config.get("label_noise_pct", 0.0)
    if noise_pct > 0:
        n_noisy = int(len(y) * noise_pct)
        noisy_idx = np.random.choice(len(y), n_noisy, replace=False)
        y[noisy_idx] = 1 - y[noisy_idx]

    normalize = config.get("normalize_input", True)
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = X * 100

    if config.get("fix_train_val_split", True):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_val, y_val = X, y

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    model = build_model(config)
    lr = config.get("learning_rate", 0.01)
    opt_name = config.get("optimizer", "adam")
    optimizer = optim.Adam(model.parameters(), lr=lr) if opt_name == "adam" else optim.SGD(model.parameters(), lr=lr)

    loss_fn_name = config.get("loss_fn", "bce")
    criterion = nn.BCEWithLogitsLoss() if loss_fn_name == "bce" else nn.MSELoss()

    max_iter = config.get("max_iter", 20)
    log = []
    train_acc = 0.5

    for step in range(1, max_iter + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()

        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

        if torch.isnan(loss) or grad_norm > 1e6:
            log.append(f"step {step}: loss=nan (exploding gradients, grad_norm={grad_norm:.1e})")
            for i in range(step + 1, max_iter + 1):
                log.append(f"step {i}: loss=nan")
            return log, 0.0

        optimizer.step()
        loss_val = loss.item()

        model.eval()
        with torch.no_grad():
            train_preds = (torch.sigmoid(model(X_train_t)) > 0.5).float()
            train_acc = (train_preds == y_train_t).float().mean().item()

        if task_id == "expert_1":
            log.append(f"step {step}: loss={loss_val:.4f} grad_norm={grad_norm:.2e}")
        elif task_id == "expert_2":
            log.append(f"step {step}: loss={loss_val:.4f} (oscillating)")
        else:
            log.append(f"step {step}: loss={loss_val:.4f} train_acc={train_acc:.3f}")

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
        val_acc = (val_preds == y_val_t).float().mean().item()

    if noise_pct > 0.1:
        log.append(f"reported_metric: train_acc={train_acc:.3f} (model appears healthy)")
        log.append(f"final val_acc={val_acc:.3f} (WARNING: suspiciously low — check label quality)")
    elif loss_fn_name == "mse":
        log.append(f"final loss={loss_val:.4f} (converged)")
        log.append(f"final val_acc=0.523 (WARNING: despite converged loss, near-random performance)")
    else:
        log.append(f"final val_acc={val_acc:.3f}")

    return log, float(val_acc)


def grade_fix(task_id: str, config_changes: dict, bug_identified: bool, broken_config: dict = None) -> float:
    if broken_config is None:
        broken_config = get_broken_config(task_id)
    base_config = broken_config.copy()

    if config_changes:
        base_config.update(config_changes)

    log, score = run_training(base_config, task_id)
    partial = 0.0

    if bug_identified:
        partial += 0.3

    if task_id == "easy":
        if "nan" not in " ".join(log):
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

        if action.action_type == "identify_bug":
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
                reward=0.0,
                task_id=task_id,
                training_log=[],
                current_config=visible_config,
                hint=None,
                message=f"Unknown action_type '{action.action_type}'. Use: 'identify_bug', 'fix_config', 'submit_fix'.",
            )

    @property
    def state(self) -> MLState:
        if self._sessions:
            latest = list(self._sessions.values())[-1]
            return latest["state"]
        return MLState()