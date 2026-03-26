# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data models for the ML Experiment Debugger Environment.
An agent receives a broken ML training config and must diagnose and fix it.
"""

from typing import Optional, Dict, Any, List
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class MLAction(Action):
    """What the agent sends — identify the bug and/or propose a fix."""

    action_type: str = Field(
        ...,
        description="One of: 'identify_bug', 'fix_config', 'submit_fix'"
    )
    bug_identified: Optional[str] = Field(
        default=None,
        description="Bug name e.g. 'learning_rate_too_high', 'data_leakage', 'label_noise'"
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

    # done and reward are already in Observation base class
    task_id: str = Field(
        default="easy",
        description="Active task: 'easy', 'medium', or 'hard'"
    )
    training_log: List[str] = Field(
        default_factory=list,
        description="Training output e.g. ['step 1: loss=2.31', 'step 2: loss=nan']"
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


class MLState(State):
    """Internal episode state — not sent to agent directly."""

    # episode_id and step_count already in State base class
    task_id: str = Field(default="easy", description="Current task")
    current_bug: str = Field(default="", description="The actual bug injected")
    attempts: int = Field(default=0, description="Number of fix attempts")
    bug_identified: bool = Field(default=False, description="Whether bug was correctly named")