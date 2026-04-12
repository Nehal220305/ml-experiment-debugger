"""
Data models for the ML Experiment Debugger Environment.
An agent receives a broken ML training config and must diagnose and fix it.
"""

from typing import Optional, Dict, Any, List
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class MLAction(Action):
    """What the agent sends — free-text diagnosis OR structured fix."""

    action_type: str = Field(
        ...,
        description="One of: 'diagnose' (free-text), 'submit_fix' (structured)"
    )
    response: Optional[str] = Field(
        default=None,
        description="Free-text diagnosis and fix explanation. Used with action_type='diagnose'"
    )
    bug_identified: Optional[str] = Field(
        default=None,
        description="Bug name e.g. 'learning_rate_too_high', 'data_leakage', 'label_noise', 'wrong_loss_function', 'vanishing_gradients', 'missing_normalization'"
    )
    config_changes: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Key-value config fixes e.g. {'learning_rate': 0.001}"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Agent's reasoning — used for partial credit"
    )


class MLObservation(Observation):
    """What the environment sends back to the agent after each action."""

    task_id: str = Field(
        default="easy",
        description="Active task: easy, medium, hard, very_hard, expert_1, expert_2"
    )
    training_log: List[str] = Field(
        default_factory=list,
        description="Real Python training output showing loss, accuracy, gradient norms"
    )
    current_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="The broken config the agent must fix"
    )
    hint: Optional[str] = Field(
        default=None,
        description="Hint shown after wrong attempts"
    )
    message: str = Field(
        default="",
        description="Feedback message after each action"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Detailed scoring feedback from the grader"
    )


class MLState(State):
    """Internal episode state — not sent to agent directly."""

    task_id: str = Field(default="easy", description="Current task")
    current_bug: str = Field(default="", description="The actual bug injected")
    attempts: int = Field(default=0, description="Number of fix attempts")
    bug_identified: bool = Field(default=False, description="Whether bug was correctly named")