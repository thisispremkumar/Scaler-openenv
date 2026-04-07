from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, List, Optional

from openai import OpenAI

try:
    from my_real_world_env import MyRealWorldEnv, SupportTriageAction
except Exception:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from client import MyRealWorldEnv
    from models import SupportTriageAction


IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "")
TASK_NAME = os.getenv("MY_REAL_WORLD_ENV_TASK", "support-triage")
BENCHMARK = os.getenv("MY_REAL_WORLD_ENV_BENCHMARK", "my_real_world_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a customer-support triage agent.
    Return exactly one JSON object with keys action_type, ticket_id, value, note.
    Allowed action_type: view_ticket, set_priority, set_queue, tag_ticket,
    send_response, escalate_ticket, close_ticket, noop.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _format_action(action: SupportTriageAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"), ensure_ascii=False)


def _build_prompt(obs: Any, step: int, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task objective:
        {obs.objective}

        Task details:
        {obs.model_dump_json(indent=2)}

        Step: {step}
        Previous actions:
        {history_block}

        Return one JSON object with action_type, ticket_id, value, note.
        """
    ).strip()


def _resolve_model_name(client: OpenAI) -> str:
    model_name = MODEL_NAME.strip()
    if model_name:
        return model_name

    try:
        models = client.models.list()
        data = getattr(models, "data", None) or []
        if data:
            first = data[0]
            model_id = getattr(first, "id", "")
            if model_id:
                return str(model_id)
    except Exception:
        pass

    return "gpt-4o-mini"


def _choose_action(client: OpenAI, model_name: str, obs: Any, step: int, history: List[str]) -> SupportTriageAction:
    try:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(obs, step, history)},
            ],
        )
        content = (completion.choices[0].message.content or "{}").strip()
        payload = json.loads(content)
        return SupportTriageAction.model_validate(payload)
    except Exception:
        return SupportTriageAction(action_type="noop")


async def _open_environment(env_base_url: str):
    if IMAGE_NAME:
        return await MyRealWorldEnv.from_docker_image(IMAGE_NAME)
    return MyRealWorldEnv(base_url=env_base_url)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    model_name = _resolve_model_name(client)
    env_base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")

    env = None
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []
    task_name = TASK_NAME
    started = False

    try:
        env = await _open_environment(env_base_url)
        result = await env.reset()
        obs = result.observation
        task_name = obs.task_id or TASK_NAME
        log_start(task=task_name, env=BENCHMARK, model=model_name)
        started = True

        last_error: Optional[str] = None
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = _choose_action(client, model_name, obs, step, history)
            action_text = _format_action(action)

            try:
                result = await env.step(action)
                obs = result.observation
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                last_error = None
            except Exception as exc:
                reward = 0.0
                done = False
                last_error = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_text, reward=reward, done=done, error=last_error)
            history.append(f"Step {step}: {action_text} -> reward {reward:+.2f}")

            if done:
                break

        score = float(getattr(obs, "task_score", 0.0)) if 'obs' in locals() else 0.0
        score = min(max(score, 0.0), 1.0)
        success = bool(score >= SUCCESS_SCORE_THRESHOLD)
    except Exception:
        if not started:
            log_start(task=task_name, env=BENCHMARK, model=model_name)
            started = True
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
