from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from node_agent.release_manifest import bundled_release_dir, load_release_manifest  # noqa: E402


HF_API_BASE = "https://huggingface.co/api/models"


@dataclass(frozen=True)
class ArtifactDefinition:
    model: str
    operation: str
    runtime_engine: str
    model_files: tuple[str, ...]
    tokenizer_files: tuple[str, ...]


ARTIFACT_DEFINITIONS = (
    ArtifactDefinition(
        model="meta-llama/Llama-3.1-8B-Instruct",
        operation="responses",
        runtime_engine="vllm",
        model_files=(
            "config.json",
            "generation_config.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "model.safetensors.index.json",
        ),
        tokenizer_files=(
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    ),
    ArtifactDefinition(
        model="BAAI/bge-large-en-v1.5",
        operation="embeddings",
        runtime_engine="vllm",
        model_files=(
            "1_Pooling/config.json",
            "config.json",
            "config_sentence_transformers.json",
            "model.safetensors",
            "modules.json",
            "sentence_bert_config.json",
        ),
        tokenizer_files=(
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ),
    ),
    ArtifactDefinition(
        model="google/gemma-4-E4B-it",
        operation="responses",
        runtime_engine="vllm",
        model_files=(
            "config.json",
            "generation_config.json",
            "model.safetensors",
        ),
        tokenizer_files=(
            "chat_template.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    ),
)


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url)
    token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def huggingface_model_info(model: str) -> dict[str, Any]:
    return fetch_json(f"{HF_API_BASE}/{model}")


def huggingface_tree(model: str, revision: str) -> dict[str, dict[str, Any]]:
    payload = fetch_json(f"{HF_API_BASE}/{model}/tree/{revision}?recursive=1")
    tree: dict[str, dict[str, Any]] = {}
    for entry in payload:
        if isinstance(entry, dict) and entry.get("type") == "file" and isinstance(entry.get("path"), str):
            tree[str(entry["path"])] = entry
    return tree


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def manifest_digest(value: Any) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(stable_json(value).encode('utf-8')).hexdigest()}"


def resolved_file_digest(model: str, revision: str, path: str, tree_entry: dict[str, Any]) -> str:
    lfs = tree_entry.get("lfs")
    if isinstance(lfs, dict) and isinstance(lfs.get("oid"), str) and lfs["oid"]:
        return f"sha256:{lfs['oid']}"
    oid = tree_entry.get("oid")
    if not isinstance(oid, str) or not oid:
        raise RuntimeError(f"{model}@{revision} file {path} is missing an immutable object id.")
    return f"gitsha1:{oid}"


def resolved_file_size(tree_entry: dict[str, Any]) -> int:
    lfs = tree_entry.get("lfs")
    if isinstance(lfs, dict) and isinstance(lfs.get("size"), int):
        return int(lfs["size"])
    size = tree_entry.get("size")
    if isinstance(size, int):
        return size
    raise RuntimeError("Tree entry is missing a usable file size.")


def artifact_manifest(
    definition: ArtifactDefinition,
    *,
    revision: str,
    tree: dict[str, dict[str, Any]],
    artifact_kind: str,
    files: tuple[str, ...],
) -> dict[str, Any]:
    manifest_files: list[dict[str, Any]] = []
    for path in files:
        tree_entry = tree.get(path)
        if tree_entry is None:
            raise RuntimeError(f"{definition.model}@{revision} is missing required file {path}.")
        manifest_files.append(
            {
                "path": path,
                "digest": resolved_file_digest(definition.model, revision, path, tree_entry),
                "size_bytes": resolved_file_size(tree_entry),
            }
        )
    manifest_files.sort(key=lambda item: str(item["path"]))
    return {
        "artifact_kind": artifact_kind,
        "operation": definition.operation,
        "repository": definition.model,
        "revision": revision,
        "runtime_engine": definition.runtime_engine,
        "files": manifest_files,
    }


def build_manifest() -> dict[str, Any]:
    release_manifest = load_release_manifest()
    artifacts: list[dict[str, Any]] = []
    for definition in ARTIFACT_DEFINITIONS:
        info = huggingface_model_info(definition.model)
        revision = info.get("sha")
        if not isinstance(revision, str) or not revision:
            raise RuntimeError(f"{definition.model} did not return a stable snapshot sha.")
        tree = huggingface_tree(definition.model, revision)
        model_manifest = artifact_manifest(
            definition,
            revision=revision,
            tree=tree,
            artifact_kind="huggingface_model_snapshot",
            files=definition.model_files,
        )
        tokenizer_manifest = artifact_manifest(
            definition,
            revision=revision,
            tree=tree,
            artifact_kind="huggingface_tokenizer_snapshot",
            files=definition.tokenizer_files,
        )
        artifacts.append(
            {
                "model": definition.model,
                "operation": definition.operation,
                "source": "huggingface",
                "repository": definition.model,
                "revision": revision,
                "model_manifest_digest": manifest_digest(model_manifest),
                "tokenizer_digest": manifest_digest(tokenizer_manifest),
                "model_manifest": model_manifest,
                "tokenizer_manifest": tokenizer_manifest,
            }
        )
    artifacts.sort(key=lambda item: (str(item["operation"]), str(item["model"])))
    return {
        "version": release_manifest.version,
        "generated_at": release_manifest.published_at,
        "artifacts": artifacts,
    }


def render_manifest(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate bundled model artifact digests from upstream model snapshots.")
    parser.add_argument("--output", default=str(bundled_release_dir() / "model-artifacts.json"))
    parser.add_argument("--check", action="store_true", help="Fail if the bundled manifest does not match the generated output.")
    args = parser.parse_args(argv)

    rendered = render_manifest(build_manifest())
    output_path = Path(args.output)

    if args.check:
        current = output_path.read_text(encoding="utf-8")
        if current != rendered:
            raise SystemExit("Bundled model-artifacts.json is stale. Regenerate it before release.")
        return 0

    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
