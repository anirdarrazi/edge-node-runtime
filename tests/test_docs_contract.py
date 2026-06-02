from pathlib import Path
import re

DOC_FILES = ("AGENTS.md", "CLAUDE.md")
COMMAND_RE = re.compile(r"\bnpm\s+run\s+([A-Za-z0-9][A-Za-z0-9:_-]*)")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(.*)$")
CODE_FENCE_RE = re.compile(r"^\s*([`~]{3,})(?:\s.*)?$")


def _collect_npm_run_refs(doc_path: Path):
    refs = []
    in_code_block = False
    code_fence = None
    list_section_started = False
    list_context = None

    lines = doc_path.read_text(encoding="utf-8").splitlines()

    for line_number, line in enumerate(lines, start=1):
        list_match = LIST_ITEM_RE.match(line)
        fence_match = CODE_FENCE_RE.match(line)

        if list_match:
            list_section_started = True
            list_context = list_match.group(1).strip()
        elif not in_code_block and not line.strip():
            list_section_started = False

        if fence_match:
            fence = fence_match.group(1)
            if in_code_block and fence[0] == code_fence[0] and len(fence) >= len(code_fence):
                in_code_block = False
                code_fence = None
                continue

            if not in_code_block:
                in_code_block = True
                code_fence = fence
                continue

        if not in_code_block and not list_section_started:
            continue

        for match in COMMAND_RE.finditer(line):
            refs.append(
                {
                    "line": line_number,
                    "command": match.group(1),
                    "raw": match.group(0).strip(),
                    "context": (
                        f"code block" if in_code_block else f"list item: {list_context}"
                    ),
                }
            )

    return refs


def test_docs_contract_runtime_expected_no_npm_run_refs():
    repo_root = Path(__file__).resolve().parents[1]
    findings = []

    for doc in DOC_FILES:
        path = repo_root / doc
        refs = _collect_npm_run_refs(path)

        for ref in refs:
            findings.append(
                f"repo=edge-node-runtime doc={doc} line={ref['line']} command={ref['command']} "
                f"raw={ref['raw']} context={ref['context']}"
            )

    assert not findings, (
        "[docs-contract:no-npm-run] edge-node-runtime: expected no npm run refs in AGENTS.md and CLAUDE.md.\n"
        "Found:\n" + "\n".join(findings)
    )
