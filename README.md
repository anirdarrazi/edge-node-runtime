# Edge Node Runtime

Python runtime for AUTONOMOUSc edge nodes.

Contents:

- `src/node_agent`: node enrollment, polling, assignment execution, and reporting
- `Dockerfile`: container image build for the node agent
- `docker-compose.yml`: local appliance runtime with `vllm` and `vector`
- `.env.example`: environment template for local/runtime configuration

Bootstrap and run:

```bash
docker compose run --rm node-agent-bootstrap
docker compose up -d node-agent vllm vector
```

Notes:

- `node-agent-bootstrap` opens the first-run terminal claim flow and waits for browser approval in `marketplace-console`.
- `node-agent` runs headless after credentials have been stored in the shared `credentials` volume.
- `OPERATOR_TOKEN` is now only a legacy fallback for development or controlled migrations.

Run tests with:

```bash
python -m pytest
```
