#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import re
import sys


ALLOWED_PLACEHOLDER_RE = re.compile(
    r"^(\{\{\s*.+?\s*\}\}|\[\[.+?\]\]|<!--\s*(?:placeholder|architecture):\s*.+?\s*-->)$",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n").replace("\ufeff", "")


def normalize_line(line: str, strict: bool) -> str:
    normalized = line.rstrip()
    if strict and ALLOWED_PLACEHOLDER_RE.match(normalized):
        return "__DOC_PLACEHOLDER__"
    return normalized


def first_diff(left_text: str, right_text: str, strict: bool) -> int:
    left_lines = normalize(left_text).split("\n")
    right_lines = normalize(right_text).split("\n")
    max_lines = max(len(left_lines), len(right_lines))

    for i in range(max_lines):
        if i >= len(left_lines):
            return i + 1
        if i >= len(right_lines):
            return i + 1

        left_line = normalize_line(left_lines[i], strict)
        right_line = normalize_line(right_lines[i], strict)
        if left_line != right_line:
            return i + 1

    return 0


def parse_args() -> tuple[bool, bool]:
    parser = ArgumentParser()
    parser.add_argument("--expected-no-npm-run-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    return args.expected_no_npm_run_only, args.strict


root = Path(__file__).resolve().parent.parent
pairs = [("AGENTS.md", "CLAUDE.md")]
command_pattern = re.compile(r"\bnpm\s+run\s+([A-Za-z0-9][A-Za-z0-9:_-]*)")
expected_no_npm_run_mode, strict_mode = parse_args()
legacy_expected_no_npm_run_mode = "--expected-no-npm-run-only" in sys.argv
expected_no_npm_run_mode = expected_no_npm_run_mode or legacy_expected_no_npm_run_mode

drifted = False
npm_run_refs = []

for left_name, right_name in pairs:
    left = (root / left_name).read_text(encoding="utf-8")
    right = (root / right_name).read_text(encoding="utf-8")
    left_norm = normalize(left)
    right_norm = normalize(right)

    if strict_mode:
        line = first_diff(left_norm, right_norm, strict=strict_mode)
        if line:
            drifted = True
            left_lines = left_norm.split("\n")
            right_lines = right_norm.split("\n")
            print(f"[doc-sync] Drift detected: {left_name} and {right_name}", flush=True)
            print(f"First diff at line {line}", flush=True)
            print(f"  {left_name}: {left_lines[line - 1] if line - 1 < len(left_lines) else '<EOF>'}")
            print(
                f"  {right_name}: {right_lines[line - 1] if line - 1 < len(right_lines) else '<EOF>'}",
                flush=True,
            )
    elif left_norm != right_norm:
        drifted = True
        line = first_diff(left_norm, right_norm, strict=False)
        left_lines = left_norm.split("\n")
        right_lines = right_norm.split("\n")
        print(f"[doc-sync] Drift detected: {left_name} and {right_name}", flush=True)
        print(f"First diff at line {line}", flush=True)
        print(f"  {left_name}: {left_lines[line - 1] if line - 1 < len(left_lines) else '<EOF>'}")
        print(
            f"  {right_name}: {right_lines[line - 1] if line - 1 < len(right_lines) else '<EOF>'}",
            flush=True,
        )

    for index, line in enumerate(left_norm.split('\n'), start=1):
        for command in command_pattern.findall(line):
            npm_run_refs.append((left_name, index, command))

    for index, line in enumerate(right_norm.split('\n'), start=1):
        for command in command_pattern.findall(line):
            npm_run_refs.append((right_name, index, command))

if drifted and not expected_no_npm_run_mode:
    raise SystemExit("[doc-sync] AGENTS.md and CLAUDE.md must be byte-for-byte identical in each project.")

if npm_run_refs:
    print(
        "[docs-contract:no-npm-run] edge-node-runtime is non-npm; AGENTS.md/CLAUDE.md must not contain npm run references.",
        flush=True,
    )
    for file_name, line_number, command in npm_run_refs:
        print(
            f"  repo=edge-node-runtime doc={file_name} line={line_number} command={command}",
            flush=True,
        )
    raise SystemExit("[doc-sync] Remove npm run references from docs for non-node projects.")

if expected_no_npm_run_mode:
    print("[docs-contract:no-npm-run] No npm run references found in AGENTS.md or CLAUDE.md.")
elif not drifted:
    print("[doc-sync] AGENTS.md and CLAUDE.md are in sync and contain no npm run references.")
