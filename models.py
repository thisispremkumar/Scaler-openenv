from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation


class SupportTriageReward(BaseModel):
    """Structured reward breakdown for transparency and debugging."""

    total: float = 0.0
    progress: float = 0.0
    efficiency: float = 0.0
    safety: float = 0.0
    completion_bonus: float = 0.0


class TicketSnapshot(BaseModel):
    """Observable state for a single customer support ticket."""

    ticket_id: str
    customer_tier: Literal["vip", "business", "standard"]
    category: Literal["billing", "technical", "shipping", "account", "abuse"]
    urgency: int = Field(ge=1, le=5)
    sla_hours: int = Field(ge=1, le=168)
    summary: str
    status: Literal["open", "pending", "escalated", "closed"] = "open"
    assigned_queue: Optional[str] = None
    priority: Optional[Literal["low", "normal", "high", "urgent"]] = None
    tags: List[str] = Field(default_factory=list)
    response_template: Optional[str] = None
    resolution_code: Optional[str] = None
    escalated: bool = False


class SupportTriageObservation(Observation):
    """Environment observation returned after reset() and step()."""

    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    success_criteria: List[str]
    tickets: List[TicketSnapshot]
    focused_ticket_id: Optional[str] = None
    step_count: int = 0
    max_steps: int = 0
    progress_score: float = Field(default=0.01, ge=0.01, le=0.99)
    task_score: float = Field(default=0.01, ge=0.01, le=0.99)
    last_action_feedback: str = ""
    reward_breakdown: SupportTriageReward = Field(default_factory=SupportTriageReward)


class SupportTriageAction(Action):
    """Agent action for support operations."""

    action_type: Literal[
        "view_ticket",
        "set_priority",
        "set_queue",
        "tag_ticket",
        "send_response",
        "escalate_ticket",
        "close_ticket",
        "noop",
    ]
    ticket_id: Optional[str] = None
    value: Optional[str] = None
    note: Optional[str] = None


# Backward-compatible aliases to keep the template imports working.
MyRealWorldAction = SupportTriageAction
MyRealWorldObservation = SupportTriageObservation
MyRealWorldReward = SupportTriageReward
