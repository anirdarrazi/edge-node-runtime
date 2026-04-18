from __future__ import annotations

from typing import Any


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def _safe_positive_int(value: Any) -> int | None:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return max(1, parsed) if parsed > 0 else None


def _safe_nonnegative_int(value: Any) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _configured_models(supported_models: str | None) -> set[str]:
    return {model.strip() for model in str(supported_models or "").split(",") if model.strip()}


def recommended_embeddings_concurrency_limit(memory_gb: float | None, *, embedding_only: bool) -> int:
    if memory_gb is None or memory_gb < 12:
        return 1
    if memory_gb < 16:
        return 3 if embedding_only else 2
    if memory_gb < 24:
        return 4 if embedding_only else 3
    if memory_gb < 48:
        return 6 if embedding_only else 4
    return 8 if embedding_only else 6


def resolved_embeddings_concurrency_limit(
    *,
    supported_models: str | None,
    operations: list[str] | tuple[str, ...],
    gpu_memory_gb: float | None,
    max_concurrent_assignments: int,
    override: Any = None,
) -> int | None:
    if "embeddings" not in operations:
        return None
    models = _configured_models(supported_models)
    if DEFAULT_EMBEDDING_MODEL not in models:
        return None

    configured_override = _safe_positive_int(override)
    if configured_override is not None:
        return max(max_concurrent_assignments, configured_override)

    embedding_only = list(operations) == ["embeddings"] or models == {DEFAULT_EMBEDDING_MODEL}
    recommended = recommended_embeddings_concurrency_limit(gpu_memory_gb, embedding_only=embedding_only)
    return max(max_concurrent_assignments, recommended)


def resolved_embeddings_microbatch_assignment_limit(
    *,
    supported_models: str | None,
    operations: list[str] | tuple[str, ...],
    gpu_memory_gb: float | None,
    max_concurrent_assignments: int,
    pull_bundle_size: int,
    override: Any = None,
) -> int | None:
    if "embeddings" not in operations:
        return None
    models = _configured_models(supported_models)
    if DEFAULT_EMBEDDING_MODEL not in models:
        return None

    configured_override = _safe_positive_int(override)
    if configured_override is not None:
        return max(max_concurrent_assignments, min(64, configured_override))

    embedding_only = list(operations) == ["embeddings"] or models == {DEFAULT_EMBEDDING_MODEL}
    concurrency = recommended_embeddings_concurrency_limit(gpu_memory_gb, embedding_only=embedding_only)
    recommended_bundle = max(max_concurrent_assignments, concurrency * 4)
    configured_bundle = _safe_positive_int(pull_bundle_size)
    return min(64, max(recommended_bundle, configured_bundle or 1))


def max_worker_assignments_from_capabilities(capabilities: dict[str, Any]) -> int:
    base_limit = _safe_positive_int(capabilities.get("max_concurrent_assignments")) or 1
    embedding_limit = _safe_nonnegative_int(capabilities.get("max_concurrent_assignments_embeddings"))
    microbatch_limit = _safe_nonnegative_int(capabilities.get("max_microbatch_assignments_embeddings"))
    pull_bundle_limit = _safe_nonnegative_int(capabilities.get("max_pull_bundle_assignments"))
    return max(1, base_limit, embedding_limit, microbatch_limit, pull_bundle_limit)
