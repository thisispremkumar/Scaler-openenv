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

import os
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
except (ModuleNotFoundError, ImportError):
    from models import MyRealWorldAction, MyRealWorldObservation
    from server.my_real_world_env_environment import MyRealWorldEnvironment



# Create the app with web interface and README integration
app = create_app(
    MyRealWorldEnvironment,
    MyRealWorldAction,
    MyRealWorldObservation,
    env_name="support_triage_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


@app.get("/tasks")
async def get_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "id": "cargo_food",
            "task_id": "cargo_food",
            "name": "cargo_food",
            "description": "Complete bilateral food-compliance screening for an agricultural shipment.",
            "objective": "Extract shipment basics, ask for missing details only when needed, and select the exact food import/export compliance package.",
            "difficulty": "easy",
            "grader": "programmatic",
            "grader_type": "programmatic",
            "grader_config": {},
            "has_grader": True,
            "score_range": [0.01, 0.99],
            "pass_score": 0.70,
        },
        {
            "id": "cargo_electronics",
            "task_id": "cargo_electronics",
            "name": "cargo_electronics",
            "description": "Resolve export-control obligations for a dual-use electronics shipment.",
            "objective": "Identify the correct origin/destination details and choose the matching electronics laws, regulators, and documents.",
            "difficulty": "medium",
            "grader": "programmatic",
            "grader_type": "programmatic",
            "grader_config": {},
            "has_grader": True,
            "score_range": [0.01, 0.99],
            "pass_score": 0.78,
        },
        {
            "id": "cargo_pharma",
            "task_id": "cargo_pharma",
            "name": "cargo_pharma",
            "description": "Validate pharmaceutical API compliance across origin and destination jurisdictions.",
            "objective": "Handle stricter pharma extraction and select the exact controlled-substance paperwork without hallucinating extra laws.",
            "difficulty": "hard",
            "grader": "programmatic",
            "grader_type": "programmatic",
            "grader_config": {},
            "has_grader": True,
            "score_range": [0.01, 0.99],
            "pass_score": 0.85,
        },
    ]


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
