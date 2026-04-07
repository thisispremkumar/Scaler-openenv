from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

IMPORT_ERROR: Exception | None = None
try:
    from .client import MyRealWorldEnv
    from .models import SupportTriageAction, SupportTriageObservation
except Exception:
    try:
        # Support direct execution (python inference.py) where package context is absent.
        current_dir = Path(__file__).resolve().parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from client import MyRealWorldEnv
        from models import SupportTriageAction, SupportTriageObservation
    except Exception as exc:
        MyRealWorldEnv = None
        SupportTriageAction = None
        SupportTriageObservation = None
        IMPORT_ERROR = exc


SYSTEM_PROMPT = (
    "You are a customer-support triage agent. "
    "Return exactly one JSON object with keys action_type, ticket_id, value, note. "
    "Allowed action_type: view_ticket, set_priority, set_queue, tag_ticket, "
    "send_response, escalate_ticket, close_ticket, noop."
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def choose_action(
    client: Any,
    model_name: str,
    obs: Any,
    seed: int,
) -> Any:
    prompt = (
        "Current support triage observation as JSON:\n"
        f"{obs.model_dump_json(indent=2)}\n\n"
        "Return exactly one valid action JSON object."
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0,
            seed=seed,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:
        print(f"model_request_error={exc}", flush=True)
        return SupportTriageAction(action_type="noop")

    content = completion.choices[0].message.content or "{}"
    try:
        payload = json.loads(content)
        return SupportTriageAction.model_validate(payload)
    except Exception:
        return SupportTriageAction(action_type="noop")


def run_inference() -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("Missing required dependency: openai")
    if (
        MyRealWorldEnv is None
        or SupportTriageAction is None
        or SupportTriageObservation is None
    ):
        raise RuntimeError(f"Missing required local modules/dependencies: {IMPORT_ERROR}")

    api_base_url = _require_env("API_BASE_URL")
    api_key = _require_env("API_KEY")
    model_name = _require_env("MODEL_NAME")

    env_base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
    seed = int(os.getenv("INFERENCE_SEED", "7"))

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    scores: List[float] = []
    started = time.time()

    with MyRealWorldEnv(base_url=env_base_url).sync() as env:
        for i in range(3):
            result = env.reset()
            obs = result.observation

            print(f"[START] task={obs.task_id}", flush=True)

            while not result.done and obs.step_count < obs.max_steps:
                action = choose_action(client=client, model_name=model_name, obs=obs, seed=seed)
                result = env.step(action)
                obs = result.observation
                print(
                    f"[STEP] step={obs.step_count} reward={float(result.reward):.6f}",
                    flush=True,
                )

            score = float(obs.task_score)
            scores.append(score)
            print(
                f"[END] task={obs.task_id} score={score:.3f} "
                f"steps={obs.step_count}/{obs.max_steps}",
                flush=True,
            )
            print(
                f"task={i + 1} id={obs.task_id} difficulty={obs.difficulty} "
                f"score={score:.3f} steps={obs.step_count}/{obs.max_steps}",
                flush=True,
            )

    elapsed_s = time.time() - started
    average = sum(scores) / len(scores)

    print("\nInference summary", flush=True)
    print(f"scores={','.join(f'{x:.3f}' for x in scores)}", flush=True)
    print(f"average={average:.3f}", flush=True)
    print(f"runtime_seconds={elapsed_s:.2f}", flush=True)

    return {
        "scores": scores,
        "average": average,
        "runtime_seconds": elapsed_s,
        "env_base_url": env_base_url,
        "model_name": model_name,
    }


def main() -> None:
    run_inference()


if __name__ == "__main__":
    main()
