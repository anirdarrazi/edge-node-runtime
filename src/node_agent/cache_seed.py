from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence


def parse_preload_models(value: str | None) -> tuple[str, ...]:
    seen: set[str] = set()
    models: list[str] = []
    for line in str(value or "").replace("\r", "\n").split("\n"):
        for part in line.split(","):
            model = part.strip()
            if not model or model in seen:
                continue
            seen.add(model)
            models.append(model)
    return tuple(models)


def preload_hf_cache(
    models: Sequence[str],
    cache_dir: Path,
    *,
    log: Callable[[str], None] | None = None,
    snapshot_download: Callable[..., object] | None = None,
) -> tuple[str, ...]:
    logger = log or print
    resolved_models = tuple(dict.fromkeys(str(model).strip() for model in models if str(model).strip()))
    if not resolved_models:
        logger("Skipping Hugging Face cache preload because no starter models were configured.")
        return ()

    resolved_cache_dir = cache_dir.expanduser()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    downloader = snapshot_download
    if downloader is None:
        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download
        except ImportError as error:  # pragma: no cover
            raise RuntimeError(
                "huggingface_hub is required to preload starter-model cache layers during image build."
            ) from error
        downloader = hf_snapshot_download

    for model in resolved_models:
        logger(f"Preloading Hugging Face cache for {model} into {resolved_cache_dir} ...")
        downloader(repo_id=model, cache_dir=str(resolved_cache_dir))

    logger(f"Preloaded {len(resolved_models)} starter model(s) into {resolved_cache_dir}.")
    return resolved_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preload Hugging Face cache entries into an image layer.")
    parser.add_argument("--cache-dir", required=True, help="Target Hugging Face cache directory.")
    parser.add_argument(
        "--models",
        default="",
        help="Comma- or newline-separated Hugging Face model ids to preload. Leave blank to skip.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    preload_hf_cache(parse_preload_models(args.models), Path(args.cache_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
