"""
ML Experiment Debugger Environment.
Agent receives a broken ML training config and must diagnose and fix it.
4 tasks: easy (exploding loss), medium (data leakage), hard (label noise), very_hard (wrong loss function).
"""

import uuid
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Optional
from openenv.core.env_server import Environment
from models import MLAction, MLObservation, MLState


TASKS = {
    "easy": {
        "bug": "learning_rate_too_high",
        "description": "Training loss explodes to NaN within the first few steps. The model cannot learn.",
        "broken_config": {
            "learning_rate": 50.0,
            "max_iter": 20,
            "penalty": "l2",
            "fix_train_val_split": True,
            "label_noise_pct": 0.0,
        },
        "hint": "Look at the learning rate — values above 1.0 cause exploding gradients in SGD.",
    },
    "medium": {
        "bug": "data_leakage",
        "description": "Val accuracy is suspiciously perfect but test accuracy is terrible.",
        "broken_config": {
            "learning_rate": 0.01,
            "max_iter": 20,
            "penalty": "l2",
            "fix_train_val_split": False,
            "label_noise_pct": 0.0,
        },
        "hint": "How was the validation set created? Is it truly held-out data?",
    },
    "hard": {
        "bug": "label_noise",
        "description": "Model trains without errors. Loss looks stable. But real-world performance is catastrophically poor. Two things are wrong — find both.",
        "broken_config": {
            "learning_rate": 0.0001,
            "max_iter": 20,
            "penalty": "l2",
            "fix_train_val_split": True,
            "label_noise_pct": 0.30,
        },
        "hint": "Is the data itself trustworthy? Is the learning rate optimal?",
    },
    "very_hard": {
        "bug": "wrong_loss_function",
        "description": "Model trains for 20 steps but accuracy plateaus near 52%. Loss decreases steadily but classification performance is near random. Data and architecture are correct.",
        "broken_config": {
            "learning_rate": 0.01,
            "max_iter": 20,
            "penalty": "l2",
            "fix_train_val_split": True,
            "label_noise_pct": 0.0,
            "loss": "squared_error",
        },
        "hint": "Is the loss function appropriate for a binary classification task?",
    },
}


def run_training(config: dict, task_id: str) -> tuple:
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)

    noise_pct = config.get("label_noise_pct", 0.0)
    if noise_pct > 0:
        n_noisy = int(len(y) * noise_pct)
        noisy_idx = np.random.choice(len(y), n_noisy, replace=False)
        y[noisy_idx] = 1 - y[noisy_idx]

    if config.get("fix_train_val_split", False):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = X, y

    lr = config.get("learning_rate", 0.01)
    max_iter = config.get("max_iter", 20)
    penalty = config.get("penalty", "l2")
    loss_fn = config.get("loss", "log_loss")

    if lr > 10.0:
        log = []
        log.append("step 1: loss=2.847")
        log.append("step 2: loss=941.23 (unstable)")
        log.append("step 3: loss=nan")
        for i in range(4, max_iter + 1):
            log.append(f"step {i}: loss=nan")
        return log, 0.0

    log = []
    clf = SGDClassifier(
        loss=loss_fn,
        learning_rate="constant",
        eta0=lr,
        penalty=penalty,
        max_iter=1,
        warm_start=True,
        random_state=42,
    )

    train_acc = 0.5
    for step in range(1, max_iter + 1):
        try:
            clf.fit(X_train, y_train)
            train_preds = clf.predict(X_train)
            train_acc = accuracy_score(y_train, train_preds)

            if np.any(np.isnan(clf.coef_)) or np.any(np.abs(clf.coef_) > 1e6):
                log.append(f"step {step}: loss=nan (weights exploded)")
                return log, 0.0

            if loss_fn == "squared_error":
                fake_loss = max(0.05, 0.45 - step * 0.018)
                log.append(f"step {step}: loss={fake_loss:.3f} (decreasing)")
            else:
                log.append(f"step {step}: train_acc={train_acc:.3f}")
        except Exception as e:
            log.append(f"step {step}: ERROR — {str(e)}")
            return log, 0.0

    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)

    if noise_pct > 0.1:
        log.append(f"reported_metric: train_acc={train_acc:.3f} (model appears healthy)")
        log.append(f"final val_acc={val_acc:.3f} (WARNING: suspiciously low — check label quality)")
    elif loss_fn == "squared_error":
        log.append(f"final train_loss=0.05 (loss converged)")
        log.append(f"final val_acc=0.523")
        log.append("WARNING: despite converged loss, model barely beats random on held-out data")
    else:
        log.append(f"final val_acc={val_acc:.3f}")

    return log, float(val_acc)


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
        if "nan" not in " ".join(log) and score > 0.6:
            partial += 0.7
        elif "nan" not in " ".join(log):
            partial += 0.3

    elif task_id == "medium":
        if base_config.get("fix_train_val_split", False):
            partial += 0.7

    elif task_id == "hard":
        noise_fixed = base_config.get("label_noise_pct", 0.30) < 0.05
        split_fixed = base_config.get("fix_train_val_split", True)
        if noise_fixed and split_fixed:
            partial += 0.7
        elif noise_fixed:
            partial += 0.35

    elif task_id == "very_hard":
        loss_fixed = base_config.get("loss", "squared_error") == "log_loss"
        if loss_fixed:
            partial += 0.7

    return min(round(partial, 2), 1.0)


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

        visible_config = {k: v for k, v in task["broken_config"].items() if k != "loss"}

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

        if action.action_type == "identify_bug":
            correct = (action.bug_identified == task["bug"])
            if correct:
                self._bug_identified = True
                return MLObservation(
                    done=False,
                    reward=0.3,
                    task_id=self._task_id,
                    training_log=[],
                    current_config={k: v for k, v in task["broken_config"].items() if k != "loss"},
                    hint=None,
                    message=f"Correct! The bug is '{task['bug']}'. Now fix it using 'fix_config' or 'submit_fix'.",
                )
            else:
                hint = task["hint"] if self._attempts >= 2 else None
                return MLObservation(
                    done=False,
                    reward=0.0,
                    task_id=self._task_id,
                    training_log=[],
                    current_config={k: v for k, v in task["broken_config"].items() if k != "loss"},
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
                current_config={k: v for k, v in task["broken_config"].items() if k != "loss"},
                hint=None,
                message=f"Score: {score:.2f}. {'Episode complete.' if done else 'Keep refining or call submit_fix to finish.'}",
            )

        else:
            return MLObservation(
                done=False,
                reward=0.0,
                task_id=self._task_id,
                training_log=[],
                current_config={k: v for k, v in task["broken_config"].items() if k != "loss"},
                hint=None,
                message=f"Unknown action_type '{action.action_type}'. Use: 'identify_bug', 'fix_config', 'submit_fix'.",
            )

    @property
    def state(self) -> MLState:
        self._state.attempts = self._attempts
        self._state.bug_identified = self._bug_identified
        return self._state