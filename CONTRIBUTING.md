# Contributing to edge-node-runtime

## Expected repository layout

- `src/` - Runtime source (`node_agent` package).
- `scripts/` - Python/Node utility scripts for runtime operations.
- `runtime_bundle/` - Runtime manifest artifacts.
- `data/` - Local runtime data directory (gitignored by policy).
- `runtime scripts` (`app.sh`, `start.sh`, `stop.sh`, `repair.sh`, `install.sh`) used for local lifecycle actions.
- `.github/workflows/` - Agent-doc sync and docs workflows.
- `pyproject.toml`, `Dockerfile`, `docker-compose.yml`.
- `.gitattributes`, `.gitignore`.

## Ignored outputs and ephemeral files

Keep these ignored and cleanup-targeted:

- Python/runtime cache and build outputs: `.build-venv/`, `.installer-venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `dist/`, `build/`.
- Runtime and test artifacts: `runtime-state/`, `state/`, `data/`, `diagnostics/`, `live-drill-artifacts/`, `test artifacts/`, `test-artifacts/`, `test-output/`.
- Logs and stress outputs: `*.log`, `*.out`, `*.err`, `*.err.log`, `*.err.txt`, `*.tmp*`, `*.json.err*`, `*.json.out*`, `*stress*.*`, `codex-*.json`.
- Environment-like files are intentionally ignored only where policy marks them as non-repo state.

## Cleanup command

There is no npm wrapper in this repository. Run:

```bash
node ./scripts/clean-repo.mjs
```

This runs the shared cleanup implementation (`scripts/clean-repo.mjs`) scoped to runtime-specific ignore policy.

## Contributor documentation and AGENTS/CLAUDE sync

- Keep `AGENTS.md` and `CLAUDE.md` synchronized for every docs change.
- Only document commands that actually exist in runtime workflows or scripts.
- Apply AGENTS/CLAUDE updates using [AGENTS/CLAUDE sync instructions](../docs/AGENTS_CLAUDE_SYNC.md).
- Run `python scripts/verify_agent_claude_sync.py --strict --expected-no-npm-run-only`.
- Run `node ./scripts/verify-docs-hygiene.mjs` before review.

## Cross-repo release audit

Before release, complete:

- [ ] `node ./scripts/clean-repo.mjs` in `edge-node-runtime`.
- [ ] `python scripts/verify_agent_claude_sync.py --strict --expected-no-npm-run-only`.
- [ ] `node ./scripts/verify-docs-hygiene.mjs` in all three repos.
- [ ] Close the corresponding checklist in [docs/CROSS_REPO_PRE_RELEASE_AUDIT.md](../docs/CROSS_REPO_PRE_RELEASE_AUDIT.md).
