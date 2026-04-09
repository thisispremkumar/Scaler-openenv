# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the customer-support triage environment.

This module creates an HTTP server that exposes MyRealWorldEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from typing import Any, Dict, List
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    # dotenv not installed, skip loading
    pass

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import MyRealWorldAction, MyRealWorldObservation
    from .my_real_world_env_environment import MyRealWorldEnvironment
    from .tasks import TASKS, TaskDefinition
except (ModuleNotFoundError, ImportError):
    from models import MyRealWorldAction, MyRealWorldObservation
    from server.my_real_world_env_environment import MyRealWorldEnvironment
    from server.tasks import TASKS, TaskDefinition



# Create the app with web interface and README integration
app = create_app(
    MyRealWorldEnvironment,
    MyRealWorldAction,
    MyRealWorldObservation,
    env_name="support_triage_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

TASK_SCORE_RANGE = [0.01, 0.99]
TASK_PASS_SCORE = 0.10


def _task_to_payload(task: TaskDefinition) -> Dict[str, Any]:
    return {
        "id": task.task_id,
        "task_id": task.task_id,
        "name": task.title,
        "description": task.objective,
        "objective": task.objective,
        "difficulty": task.difficulty,
        "grader": "programmatic",
        "grader_type": "programmatic",
        "grader_config": {},
        "has_grader": True,
        "score_range": TASK_SCORE_RANGE,
        "pass_score": TASK_PASS_SCORE,
    }


@app.get("/tasks")
async def get_tasks() -> List[Dict[str, Any]]:
    return [_task_to_payload(task) for task in TASKS]


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m my_real_world_env.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn my_real_world_env.server.app:app --workers 4
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
