# Edge Node Runtime

Python runtime for AUTONOMOUSc edge nodes.

Contents:

- `src/node_agent`: node enrollment, polling, assignment execution, and reporting
- `Dockerfile`: container image build for the node agent
- `docker-compose.yml`: local appliance runtime with `vllm` and `vector`
- `.env.example`: environment template for local/runtime configuration

Bootstrap and run:

```bash
bash install.sh
```

Windows PowerShell:

```powershell
.\install.ps1
```

Already installed and just want to start the runtime again:

```bash
bash start.sh
```

Windows PowerShell:

```powershell
.\start.ps1
```

Notes:

- `install.sh` creates `.env` from `.env.example` if needed, starts `vllm`, launches the first-run terminal claim flow, and then starts the long-running services.
- `install.ps1` and `start.ps1` provide the same flow for Windows-first setups.
- `node-agent-bootstrap` opens the terminal claim flow and waits for browser approval in `marketplace-console`.
- `node-agent` runs headless after credentials have been stored in the shared `credentials` volume.
- `OPERATOR_TOKEN` is now only a legacy fallback for development or controlled migrations.
- `.env.example` defaults to the production control plane at `https://edge.autonomousc.com`. Override it only for local Worker development.

Run tests with:

```bash
python -m pytest
```
