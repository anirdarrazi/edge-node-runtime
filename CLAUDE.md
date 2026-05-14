# Edge Node Runtime Notes

Keep this file in sync with `AGENTS.md`.

## Position

`edge-node-runtime` is the Docker runtime for AUTONOMOUSc edge nodes. It runs an inference engine plus the node agent, polls the control plane for assignments, reports health, and emits runtime receipts for audit evidence.

## Supported Setup

The supported owner/runtime setup is Docker only.

- No installer wizard.
- No browser Quick Start.
- No repo-local app launcher.
- No generated `./data/service/runtime.env` path.
- Runtime configuration comes from `.env` or explicit Docker environment variables.

Normal run command:

```bash
docker run --rm \
  --gpus all \
  --env-file .env \
  -p 8000:8000 \
  -p 8011:8011 \
  -v autonomousc-edge:/var/lib/autonomousc \
  anirdarrazi/autonomousc-ai-edge-runtime:latest
```

The inference API is on `:8000`. Startup status is on `:8011/startup-status`.

## Credentials

Preferred configuration is `NODE_ID` plus `NODE_KEY` in `.env`.

For a one-time interactive claim, use the Docker bootstrap command with the same persistent volume:

```bash
docker run --rm -it \
  --env-file .env \
  -v autonomousc-edge:/var/lib/autonomousc \
  --entrypoint node-agent-bootstrap \
  anirdarrazi/autonomousc-ai-edge-runtime:latest
```

`OPERATOR_TOKEN` is a legacy headless enrollment fallback.

## Important Files

- `Dockerfile.service`: public Docker runtime image. It starts `node-agent-single-container`.
- `Dockerfile.single`: compatibility single-container image.
- `docker-compose.yml`: development stack using `.env`.
- `.env.example`: Docker runtime environment template.
- `src/node_agent/main.py`: node-agent worker loop and credential handling.
- `src/node_agent/single_container.py`: in-container vLLM plus node-agent supervisor.
- `src/node_agent/runtime_bundle/`: bundled compose/runtime metadata.

## Development

Build the public runtime image:

```bash
bash build-manager-image.sh
```

Run tests:

```bash
python -m pytest
```

Keep changes focused on the Docker runtime path unless the user explicitly asks to revive installer/service UI behavior.
