# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Client for the ML Experiment Debugger Environment.
Translates between typed models and the WebSocket wire format.
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import MLAction, MLObservation, MLState


class MlExperimentDebuggerEnv(EnvClient[MLAction, MLObservation, MLState]):

    def _step_payload(self, action: MLAction) -> dict:
        """Convert MLAction → dict to send over WebSocket."""
        payload = {"action_type": action.action_type}
        if action.bug_identified is not None:
            payload["bug_identified"] = action.bug_identified
        if action.config_changes is not None:
            payload["config_changes"] = action.config_changes
        if action.explanation is not None:
            payload["explanation"] = action.explanation
        return payload

    def _parse_result(self, payload: dict) -> StepResult:
        """Convert raw WebSocket response → StepResult with typed MLObservation."""
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=MLObservation(
                done=payload.get("done", False),
                reward=payload.get("reward", None),
                task_id=obs_data.get("task_id", "easy"),
                training_log=obs_data.get("training_log", []),
                current_config=obs_data.get("current_config", {}),
                hint=obs_data.get("hint", None),
                message=obs_data.get("message", ""),
            ),
            reward=payload.get("reward", None),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> MLState:
        """Convert raw WebSocket state response → typed MLState."""
        return MLState(
            episode_id=payload.get("episode_id", None),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "easy"),
            current_bug=payload.get("current_bug", ""),
            attempts=payload.get("attempts", 0),
            bug_identified=payload.get("bug_identified", False),
        )