from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from .client import MyRealWorldEnv
from .models import SupportTriageAction, SupportTriageObservation


SYSTEM_PROMPT = """
You are a customer-support triage agent.
Return exactly one JSON object with keys: action_type, ticket_id, value, note.
Rules:
- Use only action_type in: view_ticket, set_priority, set_queue, tag_ticket, send_response, escalate_ticket, close_ticket, noop
- Never invent ticket IDs.
- Prefer conservative, SLA-aware triage.
- Close tickets only when clearly spam/duplicate.
""".strip()


def choose_action(
    client: OpenAI,
    model: str,
    seed: int,
    obs: SupportTriageObservation,
) -> SupportTriageAction:
    user_prompt = (
        "Current observation JSON:\n"
        f"{obs.model_dump_json(indent=2)}\n\n"
        "Respond with one next action as JSON only."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=seed,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(content)
        return SupportTriageAction.model_validate(payload)
    except Exception:
        return SupportTriageAction(action_type="noop")


def run_baseline(
    base_url: str,
    model: str,
    seed: int,
    openai_base_url: str | None = None,
) -> List[float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if openai_base_url:
        client_kwargs["base_url"] = openai_base_url
    llm_client = OpenAI(**client_kwargs)

    scores: List[float] = []
    with MyRealWorldEnv(base_url=base_url) as env:
        for task_index in range(3):
            result = env.reset()
            obs = result.observation
            while not result.done and obs.step_count < obs.max_steps:
                action = choose_action(llm_client, model, seed, obs)
                result = env.step(action)
                obs = result.observation

            scores.append(float(obs.task_score))
            print(
                f"task={task_index + 1} id={obs.task_id} difficulty={obs.difficulty} "
                f"score={obs.task_score:.3f} steps={obs.step_count}/{obs.max_steps}"
            )

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI baseline on support triage environment.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL"))
    args = parser.parse_args()

    scores = run_baseline(
        base_url=args.base_url,
        model=args.model,
        seed=args.seed,
        openai_base_url=args.openai_base_url,
    )
    average = sum(scores) / len(scores)

    print("\nBaseline summary")
    print(f"scores={','.join(f'{score:.3f}' for score in scores)}")
    print(f"average={average:.3f}")


if __name__ == "__main__":
    main()
