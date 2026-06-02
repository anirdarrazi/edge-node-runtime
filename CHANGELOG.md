# Changelog

## Unreleased

### Docs and Contract

- Added a workflow command contract check (`verify-workflow-commands`) that validates all workflow `npm run` references are guarded against `package.json` drift.
- Added an explicit rule that `npm run check:ci` references are only allowed when a `check:ci` script is explicitly defined.
- Added a docs maintenance checklist and local development command policy section for docs contributors.
- Documented command-policy migration notes in this file to make docs contract changes explicit.
- Added local `node ./scripts/verify-docs-hygiene.mjs --dry-run` docs validation mode for friendly status output.
- Added docs-contract CI checks for tracked-text CRLF enforcement and oversized tracked artifacts (`verify-tracked-line-endings`, `verify-tracked-artifact-size`).
