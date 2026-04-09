"""Standalone task metadata and grader exports for validator discovery."""

from .definitions import TASKS, TASK_NAMES
from .graders import grade_task_state, grade_ticket_state

__all__ = [
    "TASKS",
    "TASK_NAMES",
    "grade_task_state",
    "grade_ticket_state",
]
