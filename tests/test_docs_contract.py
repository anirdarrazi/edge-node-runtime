from pathlib import Path
import re

DOC_FILES = ("AGENTS.md", "CLAUDE.md")
COMMAND_RE = re.compile(r"\bnpm\s+run\s+([A-Za-z0-9][A-Za-z0-9:_-]*)")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(.*)$")
CODE_FENCE_RE = re.compile(r"^\s*([`~]{3,})(?:.*)?$")
OWNER_INSTALL_EXAMPLES = (
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    Path("..") / "docs" / "NODE_OWNER_GUIDE.md",
    Path("..") / "docs" / "PRODUCT.md",
)
OWNER_SETUP_PORT_BIND_RE = re.compile(r"^\s*-p\s+['\"]?127\.0\.0\.1:8765:8765(?:/\\w+)?['\"]?\s*\\?\s*$")
OWNER_SETUP_FORBIDDEN_PORT_BIND_RE = re.compile(r"^\s*-p\s+['\"]?.*:8000:8000['\"]?\s*\\?\s*$")
OWNER_SETUP_PORT_BIND_LINE_RE = re.compile(r"^\s*-p\s+['\"]?([^'\"\s\\]+)['\"]?\s*\\?\s*$")
OWNER_INSTALL_PUBLISH_RE = re.compile(r"^\s*-p\s+['\"]?([^'\"\s\\]+)['\"]?\s*\\?\s*$")
OWNER_MESSAGE_RULE = "Run locally. Open 127.0.0.1:8765 only."
OWNER_RULE_DOCS = (
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    Path("..") / "docs" / "PRODUCT.md",
    Path("..") / "docs" / "NODE_OWNER_GUIDE.md",
)


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


def _collect_owner_install_bindings(doc_path: Path):
    for line_number, line in enumerate(doc_path.read_text(encoding="utf-8").splitlines(), start=1):
        match = OWNER_SETUP_PORT_BIND_LINE_RE.match(line)
        if match:
            binding = match.group(1)
            if OWNER_SETUP_FORBIDDEN_PORT_BIND_RE.match(line):
                yield {
                    "line": line_number,
                    "raw": line.strip(),
                    "type": "forbidden",
                }
            elif OWNER_SETUP_PORT_BIND_RE.match(line):
                yield {
                    "line": line_number,
                    "raw": line.strip(),
                    "type": "required",
                }
            elif binding.count(":") >= 2 and re.fullmatch(r"[^:]+:\d+:\d+(?:/\\w+)?", binding):
                yield {
                    "line": line_number,
                    "raw": line.strip(),
                    "type": "binding",
                }


def _parse_owner_install_port_binding(binding: str):
    try:
        host_ports = binding.split("/", 1)[0]
        parts = host_ports.split(":")
        if len(parts) < 3:
            return None

        host = ":".join(parts[:-2])
        host_port = int(parts[-2])
        container_port = int(parts[-1])
    except ValueError:
        return None

    return host, host_port, container_port


def _assert_owner_install_publish_contract():
    repo_root = Path(__file__).resolve().parents[1]
    findings = []

    for doc in OWNER_INSTALL_EXAMPLES:
        path = repo_root / doc
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = OWNER_INSTALL_PUBLISH_RE.match(line)
            if not match:
                continue

            parsed = _parse_owner_install_port_binding(match.group(1))
            if not parsed:
                continue

            host, host_port, container_port = parsed
            if host_port == 8000:
                findings.append(
                    f"owner-install-contract:file={doc} line={line_number} raw={line.strip()} "
                    "contains forbidden -p mapping exposing host port 8000."
                )
            if (
                host_port == 8765
                and container_port == 8765
                and host != "127.0.0.1"
            ):
                findings.append(
                    f"owner-install-contract:file={doc} line={line_number} raw={line.strip()} "
                    "contains non-loopback host for 8765 publish."
                )

    assert not findings, (
        "[docs-contract:owner-publish] enforce owner install publish bindings.\n"
        "Found:\n" + "\n".join(findings)
    )


def _assert_owner_setup_install_contract():
    repo_root = Path(__file__).resolve().parents[1]
    findings = []

    for doc in OWNER_INSTALL_EXAMPLES:
        path = repo_root / doc
        required_found = False
        any_binding_seen = False

        for record in _collect_owner_install_bindings(path):
            if record["type"] == "forbidden":
                findings.append(
                    f"owner-install-contract:file={doc} line={record['line']} raw={record['raw']} "
                    "contains forbidden host port mapping on 8000 for a local owner install example."
                )
            if record["type"] == "required":
                required_found = True
            if record["type"] == "binding":
                any_binding_seen = True

        if any_binding_seen and not required_found:
            findings.append(f"owner-install-contract:file={doc} does not contain required local bind -p 127.0.0.1:8765:8765.")

    assert not findings, (
        "[docs-contract:owner-install] enforce owner setup install port contract in README/AGENTS/CLAUDE.\n"
        "Found:\n" + "\n".join(findings)
    )


def _assert_owner_quick_start_rule_contract():
    repo_root = Path(__file__).resolve().parents[1]
    missing = []
    for doc in OWNER_RULE_DOCS:
        path = repo_root / doc
        content = path.read_text(encoding="utf-8")
        if OWNER_MESSAGE_RULE not in content:
            missing.append(doc.as_posix())

    assert not missing, (
        "[docs-contract:owner-rule] owner Quick Start ownership path rule missing from:\n"
        + "\n".join(f"repo=edge-node-runtime doc={doc}" for doc in missing)
    )


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


def test_owner_install_examples_bindings_contract():
    _assert_owner_setup_install_contract()


def test_public_owner_install_snippets_enforce_8765_loopback_only():
    _assert_owner_install_publish_contract()


def test_owner_quick_start_rule_contract():
    _assert_owner_quick_start_rule_contract()
