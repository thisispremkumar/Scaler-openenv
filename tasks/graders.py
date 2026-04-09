"""Standalone deterministic graders for validator discovery.

All task scores are clamped strictly into the open interval (0, 1).
"""

from __future__ import annotations

from typing import Dict


SCORE_FLOOR = 0.01
SCORE_CEIL = 0.99


def _clamp_open_interval(raw_score: float) -> float:
    return round(min(max(float(raw_score), SCORE_FLOOR), SCORE_CEIL), 3)


def grade_ticket_state(ticket: Dict[str, object], expected: Dict[str, object], weight: float) -> float:
    if all(ticket.get(key) == value for key, value in expected.items()):
        return weight
    return 0.0


def grade_task_state(task_id: str, tickets: Dict[str, Dict[str, object]]) -> float:
    raw_score = 0.5

    if task_id == "support-easy-refund":
        t1 = tickets.get("E-1001", {})
        raw_score = 0.0
        raw_score += grade_ticket_state(t1, {"priority": "urgent"}, 0.35)
        raw_score += grade_ticket_state(t1, {"assigned_queue": "billing"}, 0.25)
        raw_score += grade_ticket_state(t1, {"response_template": "refund_ack"}, 0.25)
        raw_score += grade_ticket_state(t1, {"status": "pending"}, 0.15)

    elif task_id == "support-medium-mixed-triage":
        t1 = tickets.get("M-2101", {})
        t2 = tickets.get("M-2102", {})
        t3 = tickets.get("M-2103", {})
        raw_score = 0.0
        raw_score += grade_ticket_state(t2, {"priority": "urgent"}, 0.20)
        raw_score += grade_ticket_state(t2, {"assigned_queue": "technical_incident"}, 0.20)
        raw_score += grade_ticket_state(t2, {"escalated": True}, 0.20)
        raw_score += grade_ticket_state(t1, {"response_template": "password_reset"}, 0.15)
        raw_score += grade_ticket_state(t1, {"priority": "normal"}, 0.10)
        raw_score += grade_ticket_state(t3, {"status": "closed", "resolution_code": "spam"}, 0.15)

    elif task_id == "support-hard-queue-management":
        t1 = tickets.get("H-3001", {})
        t2 = tickets.get("H-3002", {})
        t3 = tickets.get("H-3003", {})
        t5 = tickets.get("H-3005", {})
        raw_score = 0.0
        raw_score += grade_ticket_state(t1, {"priority": "urgent"}, 0.15)
        raw_score += grade_ticket_state(t1, {"assigned_queue": "security"}, 0.15)
        raw_score += grade_ticket_state(t1, {"escalated": True}, 0.20)
        raw_score += grade_ticket_state(t2, {"priority": "high"}, 0.10)
        raw_score += grade_ticket_state(t2, {"assigned_queue": "logistics"}, 0.10)
        raw_score += grade_ticket_state(t2, {"response_template": "shipping_delay"}, 0.10)
        raw_score += grade_ticket_state(t3, {"assigned_queue": "billing"}, 0.05)
        raw_score += grade_ticket_state(t3, {"response_template": "invoice_copy"}, 0.05)
        raw_score += grade_ticket_state(t5, {"status": "closed", "resolution_code": "duplicate"}, 0.10)

    return _clamp_open_interval(raw_score)
