from __future__ import annotations

from typing import Dict


SCORE_FLOOR = 0.01
SCORE_CEIL = 0.99

TicketState = Dict[str, object]
TicketMap = Dict[str, TicketState]


def _clamp_open_interval(raw_score: float) -> float:
    # OpenEnv requires task scores to be strictly inside (0, 1).
    return round(min(max(float(raw_score), SCORE_FLOOR), SCORE_CEIL), 3)


def grade_task_state(task_id: str, tickets: TicketMap) -> float:
    """Score a task state for a known task id and current ticket map."""
    if task_id == "support-easy-refund":
        raw = _grade_support_easy_refund(tickets)
    elif task_id == "support-medium-mixed-triage":
        raw = _grade_support_medium_mixed_triage(tickets)
    elif task_id == "support-hard-queue-management":
        raw = _grade_support_hard_queue_management(tickets)
    else:
        raw = 0.5

    return _clamp_open_interval(raw)


def _ticket_score(ticket: TicketState, expected: Dict[str, object], weight: float) -> float:
    if all(ticket.get(k) == v for k, v in expected.items()):
        return weight
    return 0.0


def _grade_support_easy_refund(tickets: TicketMap) -> float:
    t1 = tickets.get("E-1001", {})
    score = 0.0
    score += _ticket_score(t1, {"priority": "urgent"}, 0.35)
    score += _ticket_score(t1, {"assigned_queue": "billing"}, 0.25)
    score += _ticket_score(t1, {"response_template": "refund_ack"}, 0.25)
    score += _ticket_score(t1, {"status": "pending"}, 0.15)
    return score


def _grade_support_medium_mixed_triage(tickets: TicketMap) -> float:
    t1 = tickets.get("M-2101", {})
    t2 = tickets.get("M-2102", {})
    t3 = tickets.get("M-2103", {})

    score = 0.0
    score += _ticket_score(t2, {"priority": "urgent"}, 0.2)
    score += _ticket_score(t2, {"assigned_queue": "technical_incident"}, 0.2)
    score += _ticket_score(t2, {"escalated": True}, 0.2)
    score += _ticket_score(t1, {"response_template": "password_reset"}, 0.15)
    score += _ticket_score(t1, {"priority": "normal"}, 0.1)
    score += _ticket_score(t3, {"status": "closed", "resolution_code": "spam"}, 0.15)
    return score


def _grade_support_hard_queue_management(tickets: TicketMap) -> float:
    a1 = tickets.get("H-3001", {})
    a2 = tickets.get("H-3002", {})
    a3 = tickets.get("H-3003", {})
    a5 = tickets.get("H-3005", {})

    score = 0.0
    score += _ticket_score(a1, {"priority": "urgent"}, 0.15)
    score += _ticket_score(a1, {"assigned_queue": "security"}, 0.15)
    score += _ticket_score(a1, {"escalated": True}, 0.2)
    score += _ticket_score(a2, {"priority": "high"}, 0.1)
    score += _ticket_score(a2, {"assigned_queue": "logistics"}, 0.1)
    score += _ticket_score(a2, {"response_template": "shipping_delay"}, 0.1)
    score += _ticket_score(a3, {"assigned_queue": "billing"}, 0.05)
    score += _ticket_score(a3, {"response_template": "invoice_copy"}, 0.05)
    score += _ticket_score(a5, {"status": "closed", "resolution_code": "duplicate"}, 0.1)
    return score
