from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        MyRealWorldAction,
        MyRealWorldObservation,
        SupportTriageReward,
        TicketSnapshot,
    )
    from .tasks import TASKS, TASKS_BY_ID, TaskDefinition
except ImportError:
    from models import (
        MyRealWorldAction,
        MyRealWorldObservation,
        SupportTriageReward,
        TicketSnapshot,
    )
    from server.tasks import TASKS, TASKS_BY_ID, TaskDefinition


VALID_PRIORITIES = {"low", "normal", "high", "urgent"}
VALID_QUEUES = {
    "billing",
    "account_support",
    "technical_incident",
    "logistics",
    "security",
    "trust_safety",
}
VALID_RESPONSE_TEMPLATES = {
    "refund_ack",
    "password_reset",
    "shipping_delay",
    "invoice_copy",
    "deescalation_warning",
    "incident_ack",
}
VALID_RESOLUTION_CODES = {"resolved", "spam", "duplicate", "abuse"}


def _strict_unit_interval(value: float) -> float:
    """Clamp score to the strict open interval (0, 1) for evaluator compatibility."""
    return min(0.99, max(0.01, float(value)))


class MyRealWorldEnvironment(Environment):
    """Customer support triage simulation with deterministic tasks and graders."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _global_task_cursor: int = 0

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._task: TaskDefinition = TASKS[0]
        self._tickets: Dict[str, Dict[str, object]] = {}
        self._focused_ticket_id: Optional[str] = None
        self._done: bool = False
        self._last_feedback: str = ""
        self._consecutive_noops = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> MyRealWorldObservation:
        requested_task_id = (
            kwargs.get("task_id")
            or kwargs.get("task")
            or kwargs.get("task_name")
        )
        self._task = self._select_task(
            requested_task_id=requested_task_id,
            seed=seed,
            episode_id=episode_id,
        )
        self._reset_count += 1
        self._done = False
        self._consecutive_noops = 0
        self._focused_ticket_id = None
        self._last_feedback = f"Task '{self._task.title}' loaded."
        self._tickets = {
            t["ticket_id"]: deepcopy(t) for t in self._task.initial_tickets
        }
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return self._build_observation(
            reward=SupportTriageReward(total=0.0),
            feedback="Episode reset.",
        )

    def _select_task(
        self,
        requested_task_id: Optional[str],
        seed: Optional[int],
        episode_id: Optional[str],
    ) -> TaskDefinition:
        if requested_task_id:
            normalized = str(requested_task_id).strip()
            if normalized in TASKS_BY_ID:
                return TASKS_BY_ID[normalized]

        if seed is not None:
            return TASKS[int(seed) % len(TASKS)]

        if episode_id:
            index = sum(ord(ch) for ch in str(episode_id)) % len(TASKS)
            return TASKS[index]

        index = MyRealWorldEnvironment._global_task_cursor % len(TASKS)
        MyRealWorldEnvironment._global_task_cursor += 1
        return TASKS[index]

    def step(
        self,
        action: MyRealWorldAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> MyRealWorldObservation:
        del timeout_s, kwargs

        # HTTP clients may call /step before /reset; initialize a clean task state.
        if not self._tickets:
            self._tickets = {
                t["ticket_id"]: deepcopy(t) for t in self._task.initial_tickets
            }
            self._done = False
            self._focused_ticket_id = None

        if self._done:
            return self._build_observation(
                reward=SupportTriageReward(
                    total=0.0,
                    safety=0.0,
                ),
                feedback="Episode already completed. Reset to start a new task.",
            )

        self._state.step_count += 1
        before_score = self._task.grader(self._tickets)

        feedback, changed = self._apply_action(action)

        after_score = self._task.grader(self._tickets)
        reward = self._compute_reward(action, before_score, after_score, changed)

        if after_score >= 0.999:
            self._done = True
            completion_bonus = 0.25
            reward.completion_bonus = completion_bonus
            reward.total += completion_bonus
            feedback = f"{feedback} Task objective achieved."
        elif self._state.step_count >= self._task.max_steps:
            self._done = True
            feedback = f"{feedback} Step budget exhausted."

        self._last_feedback = feedback
        return self._build_observation(reward=reward, feedback=feedback)

    @property
    def state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            done=self._done,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
        )

    def _build_observation(
        self,
        reward: SupportTriageReward,
        feedback: str,
    ) -> MyRealWorldObservation:
        task_score = self._task.grader(self._tickets)
        reported_score = _strict_unit_interval(task_score)
        tickets = [
            TicketSnapshot(
                ticket_id=str(t["ticket_id"]),
                customer_tier=str(t["customer_tier"]),
                category=str(t["category"]),
                urgency=int(t["urgency"]),
                sla_hours=int(t["sla_hours"]),
                summary=str(t["summary"]),
                status=str(t["status"]),
                assigned_queue=t.get("assigned_queue"),
                priority=t.get("priority"),
                tags=list(t.get("tags") or []),
                response_template=t.get("response_template"),
                resolution_code=t.get("resolution_code"),
                escalated=bool(t.get("escalated", False)),
            )
            for t in self._tickets.values()
        ]
        tickets.sort(key=lambda x: x.ticket_id)

        return MyRealWorldObservation(
            done=self._done,
            reward=round(reward.total, 4),
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            objective=self._task.objective,
            success_criteria=self._task.success_criteria,
            tickets=tickets,
            focused_ticket_id=self._focused_ticket_id,
            step_count=self._state.step_count,
            max_steps=self._task.max_steps,
            progress_score=round(reported_score, 4),
            task_score=round(reported_score, 4),
            last_action_feedback=feedback,
            reward_breakdown=reward,
        )

    def _apply_action(self, action: MyRealWorldAction) -> Tuple[str, bool]:
        action_type = action.action_type

        if action_type == "noop":
            self._consecutive_noops += 1
            return "No-op action taken.", False

        self._consecutive_noops = 0

        if action_type == "view_ticket":
            if not action.ticket_id or action.ticket_id not in self._tickets:
                return "Invalid ticket for view_ticket.", False
            self._focused_ticket_id = action.ticket_id
            return f"Focused ticket set to {action.ticket_id}.", True

        if not action.ticket_id or action.ticket_id not in self._tickets:
            return "Action requires a valid ticket_id.", False

        ticket = self._tickets[action.ticket_id]

        if action_type == "set_priority":
            value = (action.value or "").lower()
            if value not in VALID_PRIORITIES:
                return "Invalid priority value.", False
            changed = ticket.get("priority") != value
            ticket["priority"] = value
            if changed and ticket.get("status") == "open":
                ticket["status"] = "pending"
            return f"Priority set to {value} for {action.ticket_id}.", changed

        if action_type == "set_queue":
            value = (action.value or "").lower()
            if value not in VALID_QUEUES:
                return "Invalid queue value.", False
            changed = ticket.get("assigned_queue") != value
            ticket["assigned_queue"] = value
            return f"Queue set to {value} for {action.ticket_id}.", changed

        if action_type == "tag_ticket":
            value = (action.value or "").strip().lower()
            if not value:
                return "tag_ticket requires a non-empty value.", False
            tags: List[str] = list(ticket.get("tags") or [])
            if value in tags:
                return f"Tag {value} already present.", False
            tags.append(value)
            ticket["tags"] = tags
            return f"Tag {value} added to {action.ticket_id}.", True

        if action_type == "send_response":
            value = (action.value or "").lower()
            if value not in VALID_RESPONSE_TEMPLATES:
                return "Invalid response template.", False
            changed = ticket.get("response_template") != value
            ticket["response_template"] = value
            if ticket.get("status") == "open":
                ticket["status"] = "pending"
            return f"Response template {value} applied to {action.ticket_id}.", changed

        if action_type == "escalate_ticket":
            changed = not bool(ticket.get("escalated", False))
            ticket["escalated"] = True
            ticket["status"] = "escalated"
            if action.note:
                tags = list(ticket.get("tags") or [])
                note_tag = f"reason:{action.note.strip().lower().replace(' ', '_')}"
                if note_tag not in tags:
                    tags.append(note_tag)
                    ticket["tags"] = tags
            return f"Ticket {action.ticket_id} escalated.", changed

        if action_type == "close_ticket":
            value = (action.value or "").lower()
            if value not in VALID_RESOLUTION_CODES:
                return "Invalid resolution code.", False
            changed = ticket.get("status") != "closed" or ticket.get("resolution_code") != value
            ticket["status"] = "closed"
            ticket["resolution_code"] = value
            return f"Ticket {action.ticket_id} closed with {value}.", changed

        return "Unknown action type.", False

    def _compute_reward(
        self,
        action: MyRealWorldAction,
        before_score: float,
        after_score: float,
        changed: bool,
    ) -> SupportTriageReward:
        reward = SupportTriageReward(
            progress=round((after_score - before_score) * 0.7, 4),
            efficiency=-0.005,
            safety=0.0,
            completion_bonus=0.0,
        )

        if not changed:
            reward.efficiency -= 0.03

        if action.action_type == "noop" and self._consecutive_noops > 1:
            reward.efficiency -= min(0.12, 0.03 * self._consecutive_noops)

        if action.action_type == "close_ticket" and action.ticket_id in self._tickets:
            ticket = self._tickets[action.ticket_id]
            unsafe_close = (
                int(ticket.get("urgency", 1)) >= 4
                and not bool(ticket.get("escalated", False))
                and not ticket.get("response_template")
            )
            if unsafe_close:
                reward.safety -= 0.25

        if action.action_type == "set_priority" and action.value:
            if action.value.lower() == "urgent" and action.ticket_id in self._tickets:
                ticket = self._tickets[action.ticket_id]
                if int(ticket.get("urgency", 1)) <= 2:
                    reward.safety -= 0.05

        raw_total = reward.progress + reward.efficiency + reward.safety + reward.completion_bonus
        reward.total = round(max(0.0, min(1.0, raw_total)), 4)
        return reward
