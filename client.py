# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""My Real World Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import (
        MyRealWorldAction,
        MyRealWorldObservation,
        SupportTriageReward,
        TicketSnapshot,
    )
except ImportError:
    from models import (
        MyRealWorldAction,
        MyRealWorldObservation,
        SupportTriageReward,
        TicketSnapshot,
    )


class MyRealWorldEnv(
    EnvClient[MyRealWorldAction, MyRealWorldObservation, State]
):
    """
    Client for the My Real World Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MyRealWorldEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(MyRealWorldAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MyRealWorldEnv.from_docker_image("my_real_world_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MyRealWorldAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MyRealWorldAction) -> Dict:
        """
        Convert MyRealWorldAction to JSON payload for step message.

        Args:
            action: MyRealWorldAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "ticket_id": action.ticket_id,
            "value": action.value,
            "note": action.note,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MyRealWorldObservation]:
        """
        Parse server response into StepResult[MyRealWorldObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MyRealWorldObservation
        """
        obs_data = payload.get("observation", {})
        tickets = [
            TicketSnapshot.model_validate(ticket)
            for ticket in obs_data.get("tickets", [])
        ]
        observation = MyRealWorldObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            objective=obs_data.get("objective", ""),
            success_criteria=obs_data.get("success_criteria", []),
            tickets=tickets,
            focused_ticket_id=obs_data.get("focused_ticket_id"),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 0),
            progress_score=obs_data.get("progress_score", 0.0),
            task_score=obs_data.get("task_score", 0.0),
            last_action_feedback=obs_data.get("last_action_feedback", ""),
            reward_breakdown=SupportTriageReward.model_validate(
                obs_data.get("reward_breakdown", {})
            ),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
