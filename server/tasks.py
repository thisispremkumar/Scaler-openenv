from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal


Difficulty = Literal["easy", "medium", "hard"]
TicketState = Dict[str, object]


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    difficulty: Difficulty
    title: str
    objective: str
    success_criteria: List[str]
    max_steps: int
    initial_tickets: List[TicketState]
    grader: Callable[[Dict[str, TicketState]], float]


def _ticket_score(ticket: TicketState, expected: Dict[str, object], weight: float) -> float:
    if all(ticket.get(k) == v for k, v in expected.items()):
        return weight
    return 0.0


def grade_easy(tickets: Dict[str, TicketState]) -> float:
    t1 = tickets["E-1001"]
    score = 0.0
    score += _ticket_score(t1, {"priority": "urgent"}, 0.35)
    score += _ticket_score(t1, {"assigned_queue": "billing"}, 0.25)
    score += _ticket_score(t1, {"response_template": "refund_ack"}, 0.25)
    score += _ticket_score(t1, {"status": "pending"}, 0.15)
    return min(1.0, score)


def grade_medium(tickets: Dict[str, TicketState]) -> float:
    t1 = tickets["M-2101"]
    t2 = tickets["M-2102"]
    t3 = tickets["M-2103"]

    score = 0.0
    score += _ticket_score(t2, {"priority": "urgent"}, 0.2)
    score += _ticket_score(t2, {"assigned_queue": "technical_incident"}, 0.2)
    score += _ticket_score(t2, {"escalated": True}, 0.2)
    score += _ticket_score(t1, {"response_template": "password_reset"}, 0.15)
    score += _ticket_score(t1, {"priority": "normal"}, 0.1)
    score += _ticket_score(t3, {"status": "closed", "resolution_code": "spam"}, 0.15)
    return min(1.0, score)


def grade_hard(tickets: Dict[str, TicketState]) -> float:
    a1 = tickets["H-3001"]
    a2 = tickets["H-3002"]
    a3 = tickets["H-3003"]
    a5 = tickets["H-3005"]

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
    return min(1.0, score)


TASKS: List[TaskDefinition] = [
    TaskDefinition(
        task_id="support-easy-refund",
        difficulty="easy",
        title="VIP Refund Request",
        objective=(
            "Handle a single VIP billing ticket by setting urgent priority, routing it to billing, "
            "and sending an appropriate refund acknowledgement response."
        ),
        success_criteria=[
            "Set priority to urgent",
            "Assign queue to billing",
            "Send response template refund_ack",
            "Leave status in pending for back-office processing",
        ],
        max_steps=8,
        initial_tickets=[
            {
                "ticket_id": "E-1001",
                "customer_tier": "vip",
                "category": "billing",
                "urgency": 5,
                "sla_hours": 2,
                "summary": "Customer was double-charged for annual subscription and requests immediate refund.",
                "status": "open",
                "assigned_queue": None,
                "priority": None,
                "tags": [],
                "response_template": None,
                "resolution_code": None,
                "escalated": False,
            }
        ],
        grader=grade_easy,
    ),
    TaskDefinition(
        task_id="support-medium-mixed-triage",
        difficulty="medium",
        title="Mixed Inbox During Incident",
        objective=(
            "Triage three incoming tickets: escalate active outage, handle account issue, and close spam."
        ),
        success_criteria=[
            "Outage ticket set to urgent",
            "Outage ticket moved to technical_incident and escalated",
            "Password-reset ticket gets password_reset response at normal priority",
            "Spam ticket is closed with resolution spam",
        ],
        max_steps=12,
        initial_tickets=[
            {
                "ticket_id": "M-2101",
                "customer_tier": "standard",
                "category": "account",
                "urgency": 3,
                "sla_hours": 24,
                "summary": "Customer cannot access account after password reset email expired.",
                "status": "open",
                "assigned_queue": None,
                "priority": None,
                "tags": [],
                "response_template": None,
                "resolution_code": None,
                "escalated": False,
            },
            {
                "ticket_id": "M-2102",
                "customer_tier": "business",
                "category": "technical",
                "urgency": 5,
                "sla_hours": 1,
                "summary": "API returns 500s across production for all write requests since 09:10 UTC.",
                "status": "open",
                "assigned_queue": None,
                "priority": None,
                "tags": [],
                "response_template": None,
                "resolution_code": None,
                "escalated": False,
            },
            {
                "ticket_id": "M-2103",
                "customer_tier": "standard",
                "category": "abuse",
                "urgency": 1,
                "sla_hours": 72,
                "summary": "Repeated promotional message with no actionable support request.",
                "status": "open",
                "assigned_queue": None,
                "priority": None,
                "tags": [],
                "response_template": None,
                "resolution_code": None,
                "escalated": False,
            },
        ],
        grader=grade_medium,
    ),
    TaskDefinition(
        task_id="support-hard-queue-management",
        difficulty="hard",
        title="High-Pressure Multi-Ticket Triage",
        objective=(
            "Coordinate five tickets under SLA pressure, with security escalation and duplicate closure while "
            "keeping routing and responses accurate."
        ),
        success_criteria=[
            "Account takeover ticket escalated to security at urgent priority",
            "Shipping incident routed to logistics with shipping_delay response",
            "Invoice request routed to billing with invoice_copy response",
            "Duplicate ticket closed with resolution duplicate",
        ],
        max_steps=15,
        initial_tickets=[
            {
                "ticket_id": "H-3001",
                "customer_tier": "vip",
                "category": "account",
                "urgency": 5,
                "sla_hours": 1,
                "summary": "Possible account takeover: unauthorized login from new country and payment method changed.",
                "status": "open",
                "assigned_queue": None,
                "priority": None,
                "tags": [],
                "response_template": None,
                "resolution_code": None,
                "escalated": False,
            },
            {
                "ticket_id": "H-3002",
                "customer_tier": "business",
                "category": "shipping",
                "urgency": 4,
                "sla_hours": 4,
                "summary": "Bulk shipment delayed 5 days and customer needs revised ETA.",
                "status": "open",
                "assigned_queue": None,
                "priority": None,
                "tags": [],
                "response_template": None,
                "resolution_code": None,
                "escalated": False,
            },
            {
                "ticket_id": "H-3003",
                "customer_tier": "standard",
                "category": "billing",
                "urgency": 2,
                "sla_hours": 48,
                "summary": "Customer needs a PDF copy of last invoice for reimbursement.",
                "status": "open",
                "assigned_queue": None,
                "priority": None,
                "tags": [],
                "response_template": None,
                "resolution_code": None,
                "escalated": False,
            },
            {
                "ticket_id": "H-3004",
                "customer_tier": "standard",
                "category": "abuse",
                "urgency": 1,
                "sla_hours": 72,
                "summary": "Contains profanity but still asks for order tracking help.",
                "status": "open",
                "assigned_queue": None,
                "priority": None,
                "tags": [],
                "response_template": None,
                "resolution_code": None,
                "escalated": False,
            },
            {
                "ticket_id": "H-3005",
                "customer_tier": "standard",
                "category": "shipping",
                "urgency": 2,
                "sla_hours": 36,
                "summary": "Duplicate of ticket H-3002 reporting the same delayed shipment.",
                "status": "open",
                "assigned_queue": None,
                "priority": None,
                "tags": [],
                "response_template": None,
                "resolution_code": None,
                "escalated": False,
            },
        ],
        grader=grade_hard,
    ),
]
