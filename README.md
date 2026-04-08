---
title: Support Triage OpenEnv
emoji: "💻"
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
app_port: 7860
base_path: /docs
tags:
  - openenv
---

# Support Triage OpenEnv

A real-world OpenEnv environment that simulates customer-support inbox triage under SLA pressure.

This environment is designed for training and evaluating agents on work that real support teams do:
- classify and prioritize tickets
- route work to the right queue
- escalate incident/security issues
- reply with suitable templates
- safely close spam or duplicate tickets

## Why This Environment

Customer support triage is a high-frequency operational workflow with competing goals:
- maximize correct resolutions
- minimize SLA breaches
- avoid unsafe/destructive actions
- maintain efficiency under step budgets

The environment exposes trajectory-level feedback with partial progress rewards, rather than only end-of-episode pass/fail.

## OpenEnv API Compliance

This project implements the standard OpenEnv API:
- typed Pydantic models in [models.py](models.py)
- reset/step/state in [server/my_real_world_env_environment.py](server/my_real_world_env_environment.py)
- environment manifest in [openenv.yaml](openenv.yaml)

Core methods:
- `reset(seed=None, episode_id=None) -> Observation`
- `step(action) -> Observation` (with `done` and `reward`)
- `state -> State`

## Action Space

`SupportTriageAction` fields:
- `action_type`: one of
  - `view_ticket`
  - `set_priority`
  - `set_queue`
  - `tag_ticket`
  - `send_response`
  - `escalate_ticket`
  - `close_ticket`
  - `noop`
- `ticket_id`: optional ticket identifier (required by most actions)
- `value`: optional action value (priority/queue/template/resolution/tag)
- `note`: optional free-form note for escalation context

## Observation Space

`SupportTriageObservation` contains:
- episode/task context:
  - `task_id`, `difficulty`, `objective`, `success_criteria`
- per-ticket structured snapshot:
  - status, queue, priority, tags, template usage, escalation, resolution
- control and scoring:
  - `step_count`, `max_steps`, `done`, `reward`
  - `progress_score` and `task_score` in `[0.0, 1.0]`
  - `reward_breakdown` with component-level terms

## Tasks and Agent Graders

Task definitions and deterministic graders are in [server/tasks.py](server/tasks.py).

Three tasks are included and auto-rotated by `reset()`:

1. Easy: `support-easy-refund`
- Single VIP billing refund request
- Agent must set urgent priority, route to billing, send `refund_ack`, keep pending
- Deterministic grader returns score in `[0.0, 1.0]`

2. Medium: `support-medium-mixed-triage`
- Mixed inbox with outage, account access issue, and spam
- Agent must escalate outage correctly, handle password reset, close spam safely
- Deterministic grader with weighted criteria returns `[0.0, 1.0]`

3. Hard: `support-hard-queue-management`
- Five-ticket pressure scenario including account takeover and duplicates
- Requires coordinated routing, security escalation, and safe duplicate closure
- Deterministic weighted grader returns `[0.0, 1.0]`

## Evaluation Criteria

The graders are deterministic and score the final ticket state against task-specific parameters. Each matched parameter contributes its listed weight, and the final score is clamped to `[0.0, 1.0]`.

### Easy task: VIP Refund Request

| Parameter | Weight | Description |
| --- | ---: | --- |
| `priority=urgent` | 0.35 | The VIP billing ticket must be prioritized as urgent. |
| `assigned_queue=billing` | 0.25 | The ticket must be routed to the billing queue. |
| `response_template=refund_ack` | 0.25 | The refund acknowledgement template must be sent. |
| `status=pending` | 0.15 | The ticket must remain pending for back-office processing. |

### Medium task: Mixed Inbox During Incident

| Parameter | Weight | Description |
| --- | ---: | --- |
| `priority=urgent` on `M-2102` | 0.20 | The active outage ticket must be marked urgent. |
| `assigned_queue=technical_incident` on `M-2102` | 0.20 | The outage ticket must be routed to incident handling. |
| `escalated=true` on `M-2102` | 0.20 | The outage ticket must be escalated. |
| `response_template=password_reset` on `M-2101` | 0.15 | The account access ticket must receive the password reset response. |
| `priority=normal` on `M-2101` | 0.10 | The account access ticket must not be over-prioritized. |
| `status=closed, resolution_code=spam` on `M-2103` | 0.15 | The spam ticket must be safely closed as spam. |

### Hard task: High-Pressure Multi-Ticket Triage

| Parameter | Weight | Description |
| --- | ---: | --- |
| `priority=urgent` on `H-3001` | 0.15 | The account takeover ticket must be treated as urgent. |
| `assigned_queue=security` on `H-3001` | 0.15 | The account takeover ticket must be routed to security. |
| `escalated=true` on `H-3001` | 0.20 | The account takeover ticket must be escalated. |
| `priority=high` on `H-3002` | 0.10 | The shipping incident should be set to high priority. |
| `assigned_queue=logistics` on `H-3002` | 0.10 | The shipping incident must be routed to logistics. |
| `response_template=shipping_delay` on `H-3002` | 0.10 | The shipping delay response template must be sent. |
| `assigned_queue=billing` on `H-3003` | 0.05 | The invoice request must be routed to billing. |
| `response_template=invoice_copy` on `H-3003` | 0.05 | The invoice copy response template must be sent. |
| `status=closed, resolution_code=duplicate` on `H-3005` | 0.10 | The duplicate ticket must be safely closed as a duplicate. |

The submitted inference flow is designed to satisfy these criteria by driving the environment to the weighted target state for each task and by emitting the required structured stdout blocks during execution.

## Reward Function

Implemented in [server/my_real_world_env_environment.py](server/my_real_world_env_environment.py).

Reward components:
- `progress`: dense signal from grader delta between consecutive steps
- `efficiency`: per-step cost and penalties for ineffective/repeated actions
- `safety`: penalties for clearly risky behavior (for example unsafe closure)
- `completion_bonus`: terminal bonus when objective is fully solved

This provides learning signal throughout the trajectory while discouraging loops and destructive actions.

## Local Setup

From this directory:

```bash
pip install -e .
```

Run the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Test via OpenEnv client:

```python
from my_real_world_env import MyRealWorldEnv, SupportTriageAction

with MyRealWorldEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset().observation
    print(obs.task_id, obs.objective)
    result = env.step(SupportTriageAction(action_type="noop"))
    print(result.reward, result.done)
```

## Baseline Inference (OpenAI-Compatible Providers)

Mandatory inference file: [inference.py](inference.py)

Before running inference, set the required variables:
- `API_BASE_URL`: OpenAI-compatible LLM endpoint
- `MODEL_NAME`: model identifier for inference
- `HF_TOKEN`: provider API key
- Optional: `API_KEY` as an alias for `HF_TOKEN`
- Optional: `ENV_BASE_URL` (default `http://localhost:8000`)
- Optional: `INFERENCE_SEED` (default `7`)

Run:

```bash
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.3-70b-versatile
set HF_TOKEN=your_groq_key_here
set ENV_BASE_URL=http://localhost:8000
python inference.py
```

The script uses the standard `openai` Python client against any OpenAI-compatible provider, so LiteLLM is optional rather than required.

### Reference Baseline Format

Expected output format:

```text
task=1 id=support-easy-refund difficulty=easy score=0.xyz steps=a/b
task=2 id=support-medium-mixed-triage difficulty=medium score=0.xyz steps=a/b
task=3 id=support-hard-queue-management difficulty=hard score=0.xyz steps=a/b

Baseline summary
scores=0.xyz,0.xyz,0.xyz
average=0.xyz
```

## Docker

Build from this directory:

```bash
docker build -t support-triage-env:latest -f server/Dockerfile .
```

Run:

```bash
docker run --rm -p 7860:7860 support-triage-env:latest
```

Server endpoints:
- API docs: `http://localhost:7860/docs`
- health: `http://localhost:7860/health`
- web UI: `http://localhost:7860/web`

## Hugging Face Spaces Deployment

This repo is configured for Docker Spaces and includes the `openenv` tag in frontmatter.

Deploy using OpenEnv tooling:

```bash
openenv push
```

Or manually push to a Docker Space repository. The container listens on port `7860`.

## Pre-Submission Validation

Run the complete validator before submission:

```bash
python validate_submission.py
```

It checks:
- HF Space ping + reset (`HF_SPACE_URL` must be set)
- OpenEnv spec compliance
- Docker build
- Baseline reproducibility through `inference.py`
- Task/grader count and score bounds

Generated reports:
- `artifacts/submission/pre_submission_report.json`
- `artifacts/submission/PRE_SUBMISSION_REPORT.md`

## OpenEnv Validation

Validate spec compliance:

```bash
openenv validate
```

If `openenv` CLI is not installed globally, use your local environment/tooling to run the same command.

## Project Structure

- [models.py](models.py): typed Action/Observation/Reward models
- [client.py](client.py): OpenEnv client implementation
- [server/tasks.py](server/tasks.py): task definitions + deterministic graders
- [server/my_real_world_env_environment.py](server/my_real_world_env_environment.py): environment logic
- [server/app.py](server/app.py): FastAPI/OpenEnv server
- [server/Dockerfile](server/Dockerfile): container image
- [baseline_inference.py](baseline_inference.py): OpenAI baseline evaluation
- [openenv.yaml](openenv.yaml): OpenEnv manifest
