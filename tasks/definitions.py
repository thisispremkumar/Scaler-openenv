"""Validator-friendly task definitions exposed separately from the server runtime."""

TASKS = {
    "support-easy-refund": {
        "description": (
            "Handle a single VIP billing refund request by setting urgent priority, "
            "routing to billing, sending refund_ack, and leaving the ticket pending."
        ),
        "steps": 8,
        "difficulty": "easy",
        "ideal_outcomes": {
            "E-1001": {
                "priority": "urgent",
                "assigned_queue": "billing",
                "response_template": "refund_ack",
                "status": "pending",
            }
        },
    },
    "support-medium-mixed-triage": {
        "description": (
            "Triage an outage, an account issue, and spam by escalating the outage, "
            "sending the password reset response, and closing the spam ticket safely."
        ),
        "steps": 12,
        "difficulty": "medium",
        "ideal_outcomes": {
            "M-2101": {
                "response_template": "password_reset",
                "priority": "normal",
            },
            "M-2102": {
                "priority": "urgent",
                "assigned_queue": "technical_incident",
                "escalated": True,
            },
            "M-2103": {
                "status": "closed",
                "resolution_code": "spam",
            },
        },
    },
    "support-hard-queue-management": {
        "description": (
            "Coordinate five tickets under SLA pressure, including security escalation, "
            "shipping delay handling, invoice routing, and duplicate closure."
        ),
        "steps": 15,
        "difficulty": "hard",
        "ideal_outcomes": {
            "H-3001": {
                "priority": "urgent",
                "assigned_queue": "security",
                "escalated": True,
            },
            "H-3002": {
                "priority": "high",
                "assigned_queue": "logistics",
                "response_template": "shipping_delay",
            },
            "H-3003": {
                "assigned_queue": "billing",
                "response_template": "invoice_copy",
            },
            "H-3005": {
                "status": "closed",
                "resolution_code": "duplicate",
            },
        },
    },
}

TASK_NAMES = list(TASKS.keys())
