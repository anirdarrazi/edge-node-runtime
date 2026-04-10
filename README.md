# Edge Node Runtime

Python runtime for AUTONOMOUSc edge nodes.

Contents:

- `src/node_agent`: node enrollment, polling, assignment execution, and reporting
- `Dockerfile`: container image build for the node agent
- `Dockerfile.service`: owner-facing runtime manager image
- `docker-compose.yml`: local appliance runtime with `vllm` and `vector`
- `.env.example`: environment template for local/runtime configuration
- `build-manager-image.ps1` / `build-manager-image.sh`: build the runtime manager image

## Owner Install Target

The owner-facing distribution path is the runtime manager Docker container. It orchestrates the runtime stack through the host Docker socket while the repo-local scripts remain available for development.

### Runtime manager container

Build and publish the runtime manager image:

```bash
bash build-manager-image.sh
```

Or on Windows while preparing a release:

```powershell
.\build-manager-image.ps1
```

Once the image is published, a node owner can launch it with:

```bash
docker run -d \
  --name autonomousc-node-runtime-manager \
  -p 8765:8765 \
  --add-host=host.docker.internal:host-gateway \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v autonomousc-edge-manager:/var/lib/autonomousc/manager \
  anirdarrazi/autonomousc-ai-edge-runtime@sha256:4662922dd7912bbd928f0703e27472829cacc0a858732a2d48caa167a96561db
```

Then open `http://127.0.0.1:8765` and complete the guided setup.

Notes:

- The manager container bundles the runtime UI and owner compose assets.
- The same published image is also used for the bundled `node-agent` service inside the runtime bundle.
- The actual runtime services still run as sibling containers on the host Docker engine.
- The manager container talks to host-published services through `host.docker.internal`, which is why the `--add-host` flag is required on Linux.

## Repo-local install

Bootstrap and run:

```bash
bash install.sh
```

Windows PowerShell:

```powershell
.\install.ps1
```

Windows double-click launcher:

`Install AUTONOMOUSc Edge Node.cmd`

The install scripts now launch the local node service in your browser. The service handles first-time setup and ongoing operations from one place. It:

- checks Docker and GPU prerequisites
- writes `.env` for you
- starts `vllm`
- creates the node claim and waits for browser approval
- stores credentials locally in `./data/credentials`
- starts `node-agent` and `vector` after the node is approved
- gives you local start, stop, restart, update, and diagnostics controls without needing direct Docker commands

Already installed and just want to start the runtime again:

```bash
bash start.sh
```

Windows PowerShell:

```powershell
.\start.ps1
```

Windows double-click launcher:

`Open AUTONOMOUSc Edge Node.cmd`

Stop the background service:

```bash
bash stop.sh
```

Windows PowerShell:

```powershell
.\stop.ps1
```

Windows double-click launcher:

`Stop AUTONOMOUSc Edge Node.cmd`

Notes:

- `install.sh`, `install.ps1`, `start.sh`, and `start.ps1` create a local service virtual environment and open the background-service UI.
- On Windows, the friendly `.cmd` launchers bypass PowerShell execution-policy friction so owners can just double-click into setup or reopen the app later.
- The Windows scripts now require Python 3.11 or newer explicitly and guide owners toward reinstalling cleanly if the local environment is incomplete.
- The local UI runs at `http://127.0.0.1:8765` by default and stays available while the background service is running.
- Automatic updates now follow the signed runtime release manifest and only pull digest-pinned images.
- Repo-local installs still use the checked-out `docker-compose.yml`, which builds `node-agent` from source for development.
- Manager-container installs use the bundled runtime assets and the published digest-pinned `anirdarrazi/autonomousc-ai-edge-runtime` image instead of rebuilding from source.
- Diagnostics bundles are written to `./data/diagnostics`.
- `node-agent-bootstrap` is still available as a legacy fallback for direct terminal claim flows.
- `node-agent` runs headless after credentials have been stored in `./data/credentials`.
- `OPERATOR_TOKEN` is now only a legacy fallback for development or controlled migrations.
- `ATTESTATION_PROVIDER=simulated` is fine for local bring-up, but restricted work now requires hardware-backed attestation metadata before the control plane will schedule it.
- `.env.example` defaults to the production control plane at `https://edge.autonomousc.com`. Override it only for local Worker development.

Run tests with:

```bash
python -m pytest
```
