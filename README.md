---
title: Scaler Openenv
emoji: 💻
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
license: apache-2.0
---

# Real-World Environment - Customer Support Triage

An OpenEnv-compatible environment that simulates a customer support triage system with deterministic graders and multi-component reward shaping.

## Features

- **3 Tasks**: Easy (refund), Medium (mixed inbox), Hard (multi-ticket coordination)
- **Deterministic Graders**: Reproducible task completion scoring (0.0-1.0)
- **Multi-Component Rewards**: Progress, efficiency, safety, completion bonus
- **OpenEnv Spec Compliant**: Full HTTP API with reset/step/state endpoints
- **Docker Ready**: Auto-deployed to HuggingFace Spaces

## Environment Variables

Add these as HuggingFace Spaces secrets:

- `API_BASE_URL`: `https://router.huggingface.co/v1`
- `MODEL_NAME`: `nvidia/llama-3.1-nemotron-70b-instruct`
- `HF_TOKEN`: Your HuggingFace API token (get from https://huggingface.co/settings/tokens)

## API Endpoints

- `GET /health` - Health check
- `POST /reset` - Reset environment
- `POST /step` - Execute action
- `GET /state` - Get current state

## Development

See documentation files in the repository for detailed setup and deployment instructions.
