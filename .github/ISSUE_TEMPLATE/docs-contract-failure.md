---
name: Docs contract failure
about: Report a docs-contract violation before merging
title: "[docs-contract] "
labels: docs-contract
assignees: []
---

## Affected repository
- [ ] edge-control
- [ ] marketplace-console
- [ ] edge-node-runtime

## Failing check
- [ ] verify:agent-doc-sync
- [ ] verify:docs
- [ ] verify:workflow-commands
- [ ] verify:clean-room
- [ ] verify:tracked-line-endings
- [ ] verify:tracked-artifact-size
- [ ] other (describe below)

## Failure category (triage)
- [ ] Docs-policy drift (command references, checklist docs)
- [ ] AGENTS/CLAUDE mismatch
- [ ] Ignore pattern gap (`.gitignore`)
- [ ] Line-ending policy mismatch (`.gitattributes`)
- [ ] CI contract regression
- [ ] Toolchain / lockfile check mismatch
- [ ] Release or cleanup workflow miss

## Failing command output

Paste the exact command and the first 80 relevant lines of output:

```text

```

## Repro steps

1.
2.
3.

## Suggested fix

Describe expected change and where it should be applied (AGENTS.md, CLAUDE.md, workflow, script, etc.):

## Optional attachments

- Workflow URL:
- PR/commit URL:
- Local log artifact path:
