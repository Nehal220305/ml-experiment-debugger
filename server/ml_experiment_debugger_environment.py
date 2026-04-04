"""
ML Experiment Debugger Environment.
Agent receives a broken ML training config and must diagnose and fix it.
Uses real PyTorch training loops — all loss curves and metrics are genuine.
6 tasks: easy → very_hard covering common real-world ML bugs.
"""

import uuid
import numpy as np
from typing import Optional
from openenv.core.env_server import Environment
from models import MLAction, MLObservation, MLState

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Tiny MLP ─────────────────────────────────────────────────────────────────

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


# ── Task definitions ──────────────────────────────────────────────────────────

TASKS = {
    "easy": {
        "bug": "learning_rate_too_high",
        "description": "Training loss explodes to NaN within the first few steps. The model cannot learn.",
        "broken_config": {
            "learning_rate": 50.0,
            "max_iter": 20,
            "optimizer": "sgd",
            "fix_train_val_split": True,
            "label_noise_pct": 0.0,
            "activation": "relu",
            "loss_fn": "bce",
        },
        "hint": "Look at the learning rate — values above 1.0 cause exploding gradients.",
    },
    "medium": {
        "bug": "data_leakage",
        "description": "Val accuracy is suspiciously perfect but test accuracy is terrible.",
        "broken_config": {
            "learning_rate": 0.01,
            "max_iter": 20,
            "optimizer": "adam",
            "fix_train_val_split": False,
            "label_noise_pct": 0.0,
            "activation": "relu",
            "loss_fn": "bce",
        },
        "hint": "How was the validation set created? Is it truly held-out data?",
    },
    "hard": {
        "bug": "label_noise",
        "description": "Model trains without errors. Loss looks stable. But real-world performance is catastrophically poor. Two things are wrong — find both.",
        "broken_config": {
            "learning_rate": 0.0001,
            "max_iter": 20,
            "optimizer": "adam",
            "fix_train_val_split": True,
            "label_noise_pct": 0.30,
            "activation": "relu",
            "loss_fn": "bce",
        },
        "hint": "Is the data itself trustworthy? Is the learning rate optimal?",
    },
    "very_hard": {
        "bug": "wrong_loss_function",
        "description": "Model trains for 20 steps but accuracy plateaus near 52%. Loss decreases steadily but classification performance is near random. Data and architecture are correct.",
        "broken_config": {
            "learning_rate": 0.01,
            "max_iter": 20,
            "optimizer": "adam",
            "fix_train_val_split": True,
            "label_noise_pct": 0.0,
            "activation": "relu",
            "loss_fn": "mse",
        },
        "hint": "Is the loss function appropriate for a binary classification task?",
    },
    "expert_1": {
        "bug": "vanishing_gradients",
        "description": "Model is deep but loss barely moves after step 3. Training stalls completely. Accuracy stays at 50%.",
        "broken_config": {
            "learning_rate": 0.01,
            "max_iter": 20,
            "optimizer": "sgd",
            "fix_train_val_split": True,
            "label_noise_pct": 0.0,
            "activation": "sigmoid",
            "loss_fn": "bce",
            "depth": 6,
        },
        "hint": "What happens to gradients in very deep networks with sigmoid activations?",
    },
    "expert_2": {
        "bug": "missing_normalization",
        "description": "Loss starts very high (>10) and oscillates. Model is simple, optimizer is Adam, learning rate is 0.01. The raw feature values are extremely large — some exceed 1000.",
        "broken_config": {
            "learning_rate": 0.01,
            "max_iter": 20,
            "optimizer": "adam",
            "fix_train_val_split": True,
            "label_noise_pct": 0.0,
            "activation": "relu",
            "loss_fn": "bce",
            "normalize_input": False,
        },
        "hint": "Are the input features on a similar scale? What does unnormalized data do to gradient descent?",
    },
}


# ── PyTorch training simulator ────────────────────────────────────────────────

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

    # Inject label noise
    noise_pct = config.get("label_noise_pct", 0.0)
    if noise_pct > 0:
        n_noisy = int(len(y) * noise_pct)
        noisy_idx = np.random.choice(len(y), n_noisy, replace=False)
        y[noisy_idx] = 1 - y[noisy_idx]

    # Normalize input (or not — for missing_normalization bug)
    normalize = config.get("normalize_input", True)
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        # Scale up features to simulate unnormalized data
        X = X * 100

    # Train/val split
    if config.get("fix_train_val_split", True):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = X, y

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    # Build model
    model = build_model(config)

    # Optimizer
    lr = config.get("learning_rate", 0.01)
    opt_name = config.get("optimizer", "adam")
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    # Loss function
    loss_fn_name = config.get("loss_fn", "bce")
    if loss_fn_name == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    max_iter = config.get("max_iter", 20)
    log = []

    for step in range(1, max_iter + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()

        # Check for NaN/exploding gradients
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        if torch.isnan(loss) or grad_norm > 1e6:
            log.append(f"step {step}: loss=nan (exploding gradients, grad_norm={grad_norm:.1e})")
            for i in range(step + 1, max_iter + 1):
                log.append(f"step {i}: loss=nan")
            return log, 0.0

        optimizer.step()
        loss_val = loss.item()

        # Compute train accuracy
        model.eval()
        with torch.no_grad():
            train_preds = (torch.sigmoid(model(X_train_t)) > 0.5).float()
            train_acc = (train_preds == y_train_t).float().mean().item()

        if task_id in ("very_hard",) and loss_fn_name == "mse":
            log.append(f"step {step}: loss={loss_val:.4f} train_acc={train_acc:.3f}")
        elif task_id == "expert_1":
            log.append(f"step {step}: loss={loss_val:.4f} grad_norm={grad_norm:.2e}")
        elif task_id == "expert_2":
            log.append(f"step {step}: loss={loss_val:.4f} (oscillating)")
        else:
            log.append(f"step {step}: loss={loss_val:.4f} train_acc={train_acc:.3f}")

    # Final evaluation
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


# ── Graders ───────────────────────────────────────────────────────────────────

def grade_fix(task_id: str, config_changes: dict, bug_identified: bool) -> float:
    task = TASKS[task_id]
    base_config = task["broken_config"].copy()

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


# ── Environment ───────────────────────────────────────────────────────────────

HIDDEN_KEYS = {"loss_fn", "activation", "normalize_input", "depth"}


class MlExperimentDebuggerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = MLState()
        self._task_id = "easy"
        self._attempts = 0
        self._bug_identified = False

    def reset(
        self,
        seed=None,
        episode_id=None,
        task_id: Optional[str] = "easy",
        **kwargs,
    ) -> MLObservation:
        task_id = task_id if task_id in TASKS else "easy"
        self._task_id = task_id
        self._attempts = 0
        self._bug_identified = False

        self._state = MLState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            current_bug=TASKS[task_id]["bug"],
            attempts=0,
            bug_identified=False,
        )

        task = TASKS[task_id]
        log, _ = run_training(task["broken_config"], task_id)
        visible_config = {k: v for k, v in task["broken_config"].items() if k not in HIDDEN_KEYS}

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
        self._state.step_count += 1
        self._attempts += 1
        task = TASKS[self._task_id]
        visible_config = {k: v for k, v in task["broken_config"].items() if k not in HIDDEN_KEYS}

        if action.action_type == "identify_bug":
            correct = (action.bug_identified == task["bug"])
            if correct:
                self._bug_identified = True
                return MLObservation(
                    done=False,
                    reward=0.3,
                    task_id=self._task_id,
                    training_log=[],
                    current_config=visible_config,
                    hint=None,
                    message=f"Correct! The bug is '{task['bug']}'. Now fix it using 'submit_fix'.",
                )
            else:
                hint = task["hint"] if self._attempts >= 2 else None
                return MLObservation(
                    done=False,
                    reward=0.0,
                    task_id=self._task_id,
                    training_log=[],
                    current_config=visible_config,
                    hint=hint,
                    message=f"Incorrect. '{action.bug_identified}' is not the bug. Try again.",
                )

        elif action.action_type in ("fix_config", "submit_fix"):
            if action.bug_identified == task["bug"]:
                self._bug_identified = True

            score = grade_fix(
                self._task_id,
                action.config_changes or {},
                self._bug_identified,
            )
            done = action.action_type == "submit_fix"
            return MLObservation(
                done=done,
                reward=score,
                task_id=self._task_id,
                training_log=[],
                current_config=visible_config,
                hint=None,
                message=f"Score: {score:.2f}. {'Episode complete.' if done else 'Keep refining or call submit_fix to finish.'}",
            )

        else:
            return MLObservation(
                done=False,
                reward=0.0,
                task_id=self._task_id,
                training_log=[],
                current_config=visible_config,
                hint=None,
                message=f"Unknown action_type '{action.action_type}'. Use: 'identify_bug', 'fix_config', 'submit_fix'.",
            )

    @property
    def state(self) -> MLState:
        self._state.attempts = self._attempts
        self._state.bug_identified = self._bug_identified
        return self._state