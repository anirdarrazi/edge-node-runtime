#!/usr/bin/env python3

from pathlib import Path
import re


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def first_diff(left_text: str, right_text: str) -> int:
    left_lines = normalize(left_text).split("\n")
    right_lines = normalize(right_text).split("\n")
    max_lines = max(len(left_lines), len(right_lines))

    for i in range(max_lines):
        if i >= len(left_lines):
            return i + 1
        if i >= len(right_lines):
            return i + 1
        if left_lines[i] != right_lines[i]:
            return i + 1

    return 0


root = Path(__file__).resolve().parent.parent
pairs = [("AGENTS.md", "CLAUDE.md")]
command_pattern = re.compile(r"\bnpm\s+run\s+([A-Za-z0-9][A-Za-z0-9:_-]*)")

drifted = False
npm_run_refs = []

for left_name, right_name in pairs:
    left = (root / left_name).read_text(encoding="utf-8")
    right = (root / right_name).read_text(encoding="utf-8")
    left_norm = normalize(left)
    right_norm = normalize(right)

    if left_norm != right_norm:
        drifted = True
        line = first_diff(left_norm, right_norm)
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

if drifted:
    raise SystemExit("[doc-sync] AGENTS.md and CLAUDE.md must be byte-for-byte identical in each project.")

if npm_run_refs:
    print(
        "[doc-sync] edge-node-runtime is non-npm; AGENTS.md/CLAUDE.md must not contain npm run references.",
        flush=True,
    )
    for file_name, line_number, command in npm_run_refs:
        print(f"  {file_name}:{line_number} -> npm run {command}", flush=True)
    raise SystemExit("[doc-sync] Remove npm run references from docs for non-node projects.")

print("[doc-sync] AGENTS.md and CLAUDE.md are in sync and contain no npm run references.")
