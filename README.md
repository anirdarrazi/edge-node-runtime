# Edge Node Runtime

Python runtime for AUTONOMOUSc edge nodes.

Contents:

- `src/node_agent`: node enrollment, polling, assignment execution, and reporting
- `Dockerfile`: container image build for the node agent
- `docker-compose.yml`: local appliance runtime with `vllm` and `vector`
- `.env.example`: environment template for local/runtime configuration

Run tests with:

```bash
python -m pytest
```
