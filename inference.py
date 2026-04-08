"""
inference.py -- Support Triage AI Agent (Hackathon Submission)

Runs one full episode against the support triage environment using an LLM
to decide triage actions. Prints structured logs consumed by the evaluator.

Required environment variables:
    HF_TOKEN or API_KEY -- Hugging Face / API key

Required-with-default environment variables:
    API_BASE_URL -- OpenAI-compatible endpoint
    MODEL_NAME   -- Preferred model identifier

Optional environment variables:
    ENV_BASE_URL -- Environment server URL (default: http://localhost:8000)
    TASK_NAME    -- Task label for [START]

STDOUT FORMAT (machine-parsed):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
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


ENV_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY")
TOKEN = API_KEY or HF_TOKEN

TASK_NAME = os.getenv("TASK_NAME", os.getenv("MY_REAL_WORLD_ENV_TASK", "support-triage"))
BENCHMARK = os.getenv("MY_REAL_WORLD_ENV_BENCHMARK", "my_real_world_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

VALID_ACTIONS = {
    "view_ticket",
    "set_priority",
    "set_queue",
    "tag_ticket",
    "send_response",
    "escalate_ticket",
    "close_ticket",
    "noop",
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a customer-support triage agent.
    Return exactly one JSON object with keys action_type, ticket_id, value, note.
    Allowed action_type: view_ticket, set_priority, set_queue, tag_ticket,
    send_response, escalate_ticket, close_ticket, noop.
    """
).strip()


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def get_llm_client() -> OpenAI:
    if not TOKEN:
        raise EnvironmentError("HF_TOKEN or API_KEY environment variable is required")
    return OpenAI(base_url=API_BASE_URL, api_key=TOKEN)


def _build_prompt(obs: Any, step: int, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task objective:
        {obs.objective}

        Observation JSON:
        {obs.model_dump_json(indent=2)}

        Step: {step}
        Previous actions:
        {history_block}

        Return exactly one JSON object with action_type, ticket_id, value, note.
        """
    ).strip()


def _resolve_model_candidates(client: OpenAI) -> List[str]:
    candidates: List[str] = []
    preferred = MODEL_NAME.strip()
    if preferred:
        candidates.append(preferred)

    try:
        models = client.models.list()
        for item in getattr(models, "data", []) or []:
            model_id = getattr(item, "id", "")
            if model_id and model_id not in candidates:
                candidates.append(str(model_id))
                if len(candidates) >= 8:
                    break
    except Exception:
        pass

    for fallback in ["google/gemma-4-31B-it", "gpt-4.1-mini", "gpt-4o-mini"]:
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _resolve_working_model(client: OpenAI) -> str:
    last_error: Exception | None = None
    for model_name in _resolve_model_candidates(client):
        for _ in range(2):
            try:
                client.chat.completions.create(
                    model=model_name,
                    temperature=0,
                    max_tokens=1,
                    messages=[
                        {"role": "system", "content": "Reply with one word."},
                        {"role": "user", "content": "ping"},
                    ],
                )
                return model_name
            except Exception as exc:
                last_error = exc
    raise RuntimeError(f"Failed to make proxy API call: {last_error}")


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
        payload = json.loads((completion.choices[0].message.content or "{}").strip())
        action = SupportTriageAction.model_validate(payload)
        if action.action_type not in VALID_ACTIONS:
            return SupportTriageAction(action_type="noop")
        return action
    except Exception:
        return SupportTriageAction(action_type="noop")


def _action_to_log(action: SupportTriageAction) -> str:
    # Keep action field compact and machine-friendly.
    return action.action_type


def compute_score(final_task_score: float, rewards: List[float]) -> float:
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        blended = (final_task_score + mean_reward) / 2.0
        return min(max(round(blended, 4), 0.0), 1.0)
    return min(max(round(final_task_score, 4), 0.0), 1.0)


def run_episode(task_name: str) -> None:
    client = get_llm_client()
    model_name = _resolve_working_model(client)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []
    emitted_step = False

    log_start(task=task_name, env_name=BENCHMARK, model=model_name)

    try:
        with MyRealWorldEnv(base_url=ENV_URL).sync() as env:
            result = env.reset()
            obs = result.observation

            while not result.done and steps_taken < MAX_STEPS:
                step = steps_taken + 1
                action = _choose_action(client, model_name, obs, step, history)
                action_str = _action_to_log(action)

                error: Optional[str] = None
                try:
                    result = env.step(action)
                    obs = result.observation
                    reward = float(result.reward or 0.0)
                    done = bool(result.done)
                except Exception as exc:
                    reward = 0.0
                    done = True
                    error = str(exc)

                rewards.append(reward)
                steps_taken = step
                emitted_step = True
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                history.append(f"step={step} action={action_str} reward={reward:.2f}")

                if done:
                    break

            task_score = float(getattr(obs, "task_score", 0.0)) if "obs" in locals() else 0.0
            score = compute_score(task_score, rewards)
            success = bool(score >= SUCCESS_SCORE_THRESHOLD)
    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)
        if not emitted_step:
            steps_taken = 1
            rewards.append(0.0)
            log_step(step=1, action="noop", reward=0.0, done=True, error=str(exc))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Support triage inference")
    parser.add_argument("--task", type=str, default=TASK_NAME)
    args = parser.parse_args()
    run_episode(task_name=args.task)
