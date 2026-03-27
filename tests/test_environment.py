from __future__ import annotations

from models import SupportTriageAction
from server.my_real_world_env_environment import MyRealWorldEnvironment


def test_reset_rotates_tasks_in_order() -> None:
    env = MyRealWorldEnvironment()
    first = env.reset().task_id
    second = env.reset().task_id
    third = env.reset().task_id
    assert len({first, second, third}) == 3


def test_noop_penalty_accumulates() -> None:
    env = MyRealWorldEnvironment()
    env.reset()
    first = env.step(SupportTriageAction(action_type="noop"))
    second = env.step(SupportTriageAction(action_type="noop"))
    assert isinstance(first.reward, float)
    assert isinstance(second.reward, float)
    assert second.reward < first.reward


def test_easy_task_can_be_completed_programmatically() -> None:
    env = MyRealWorldEnvironment()
    obs = env.reset()
    assert obs.task_id == "support-easy-refund"

    obs = env.step(
        SupportTriageAction(
            action_type="set_priority",
            ticket_id="E-1001",
            value="urgent",
        )
    )
    assert obs.done is False

    obs = env.step(
        SupportTriageAction(
            action_type="set_queue",
            ticket_id="E-1001",
            value="billing",
        )
    )
    assert obs.done is False

    obs = env.step(
        SupportTriageAction(
            action_type="send_response",
            ticket_id="E-1001",
            value="refund_ack",
        )
    )
    assert obs.done is True
    assert obs.task_score == 1.0


def test_episode_ends_after_max_steps() -> None:
    env = MyRealWorldEnvironment()
    obs = env.reset()

    done = False
    for _ in range(obs.max_steps):
        obs = env.step(SupportTriageAction(action_type="noop"))
        done = obs.done

    assert done is True
