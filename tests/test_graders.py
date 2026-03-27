from __future__ import annotations

from copy import deepcopy

from server.tasks import TASKS


def _to_map(task):
    return {t["ticket_id"]: deepcopy(t) for t in task.initial_tickets}


def test_tasks_cover_expected_difficulty_range() -> None:
    assert len(TASKS) >= 3
    difficulties = {task.difficulty for task in TASKS}
    assert {"easy", "medium", "hard"}.issubset(difficulties)


def test_graders_are_bounded_and_deterministic() -> None:
    for task in TASKS:
        ticket_map = _to_map(task)
        score_a = task.grader(ticket_map)
        score_b = task.grader(ticket_map)
        assert 0.0 <= score_a <= 1.0
        assert score_a == score_b


def test_graders_give_non_perfect_initial_scores() -> None:
    for task in TASKS:
        score = task.grader(_to_map(task))
        assert score < 1.0
