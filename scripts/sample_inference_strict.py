from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import List

from openai import OpenAI
import requests

try:
    from my_real_world_env import MyRealWorldEnv, SupportTriageAction
except Exception:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from client import MyRealWorldEnv
    from models import SupportTriageAction


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
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


def choose_action(client: OpenAI, model_name: str, objective: str, step: int) -> str:
    prompt = textwrap.dedent(
        f"""
        You are solving a customer support triage task.
        Objective: {objective}
        Current step: {step}
        Return exactly one action_type from:
        view_ticket, set_priority, set_queue, tag_ticket, send_response, escalate_ticket, close_ticket, noop
        """
    ).strip()
    completion = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": "Return only a valid action_type token."},
            {"role": "user", "content": prompt},
        ],
    )
    action_type = (completion.choices[0].message.content or "").strip()
    allowed = {
        "view_ticket",
        "set_priority",
        "set_queue",
        "tag_ticket",
        "send_response",
        "escalate_ticket",
        "close_ticket",
        "noop",
    }
    if action_type not in allowed:
        raise RuntimeError(f"Model returned invalid action_type: {action_type!r}")
    return action_type


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def _start_container(image_name: str) -> tuple[str, str]:
    host_port = _find_free_port()
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "-p",
            f"{host_port}:7860",
            image_name,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    container_id = result.stdout.strip()
    base_url = f"http://localhost:{host_port}"
    return container_id, base_url


def _wait_ready(base_url: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Container at {base_url} did not become ready within {timeout_s}s")


async def main() -> None:
    api_base_url = _require_env("API_BASE_URL")
    model_name = _require_env("MODEL_NAME")
    hf_token = _require_env("HF_TOKEN")
    image_name = _require_env("LOCAL_IMAGE_NAME")

    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    rewards: List[float] = []
    score = 0.0
    steps_taken = 0
    success = False

    container_id, base_url = _start_container(image_name)
    _wait_ready(base_url)
    try:
        env_client = MyRealWorldEnv(base_url=base_url)
        with env_client.sync() as env:
            result = env.reset()
            obs = result.observation

            log_start(task=obs.task_id, env="my_real_world_env", model=model_name)

            while not result.done and obs.step_count < obs.max_steps:
                action_type = choose_action(client, model_name, obs.objective, obs.step_count + 1)
                action = SupportTriageAction(action_type=action_type)
                result = env.step(action)
                obs = result.observation
                reward = float(result.reward or 0.0)
                rewards.append(reward)
                steps_taken = obs.step_count
                log_step(
                    step=obs.step_count,
                    action=action_type,
                    reward=reward,
                    done=bool(result.done),
                    error=None,
                )

            score = float(obs.task_score)
            success = bool(result.done and score >= 0.1)
    finally:
        subprocess.run(["docker", "stop", container_id], capture_output=True)
        subprocess.run(["docker", "rm", container_id], capture_output=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
