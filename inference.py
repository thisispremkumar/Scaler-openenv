from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from openai import OpenAI

from .client import MyRealWorldEnv
from .models import SupportTriageAction, SupportTriageObservation


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
    client: OpenAI,
    model_name: str,
    obs: SupportTriageObservation,
    seed: int,
) -> SupportTriageAction:
    prompt = (
        "Current support triage observation as JSON:\n"
        f"{obs.model_dump_json(indent=2)}\n\n"
        "Return exactly one valid action JSON object."
    )

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

    content = completion.choices[0].message.content or "{}"
    try:
        payload = json.loads(content)
        return SupportTriageAction.model_validate(payload)
    except Exception:
        return SupportTriageAction(action_type="noop")


def run_inference() -> Dict[str, Any]:
    api_base_url = _require_env("API_BASE_URL")
    model_name = _require_env("MODEL_NAME")
    hf_token = _require_env("HF_TOKEN")

    env_base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
    seed = int(os.getenv("INFERENCE_SEED", "7"))

    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    scores: List[float] = []
    started = time.time()

    with MyRealWorldEnv(base_url=env_base_url) as env:
        for i in range(3):
            result = env.reset()
            obs = result.observation

            while not result.done and obs.step_count < obs.max_steps:
                action = choose_action(client=client, model_name=model_name, obs=obs, seed=seed)
                result = env.step(action)
                obs = result.observation

            score = float(obs.task_score)
            scores.append(score)
            print(
                f"task={i + 1} id={obs.task_id} difficulty={obs.difficulty} "
                f"score={score:.3f} steps={obs.step_count}/{obs.max_steps}"
            )

    elapsed_s = time.time() - started
    average = sum(scores) / len(scores)

    print("\nInference summary")
    print(f"scores={','.join(f'{x:.3f}' for x in scores)}")
    print(f"average={average:.3f}")
    print(f"runtime_seconds={elapsed_s:.2f}")

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
