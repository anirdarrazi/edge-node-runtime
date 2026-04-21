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


def resolved_local_queue_assignment_limit(
    *,
    supported_models: str | None,
    operations: list[str] | tuple[str, ...],
    gpu_memory_gb: float | None,
    max_concurrent_assignments: int,
    pull_bundle_size: int,
    max_microbatch_assignments: Any = None,
    override: Any = None,
) -> int:
    configured_override = _safe_positive_int(override)
    if configured_override is not None:
        return max(max_concurrent_assignments, min(64, configured_override))

    configured_bundle = _safe_positive_int(pull_bundle_size) or 1
    embedding_microbatch_limit = resolved_embeddings_microbatch_assignment_limit(
        supported_models=supported_models,
        operations=operations,
        gpu_memory_gb=gpu_memory_gb,
        max_concurrent_assignments=max_concurrent_assignments,
        pull_bundle_size=pull_bundle_size,
        override=max_microbatch_assignments,
    )
    if embedding_microbatch_limit is not None:
        recommended = max(
            configured_bundle,
            max_concurrent_assignments * 4,
            embedding_microbatch_limit + (max(1, max_concurrent_assignments) * 4),
        )
        return min(64, max(max_concurrent_assignments, recommended))

    recommended = max(max_concurrent_assignments * 4, configured_bundle)
    return min(32, max(max_concurrent_assignments, recommended))


def max_worker_assignments_from_capabilities(capabilities: dict[str, Any]) -> int:
    if capabilities.get("heat_governor_paused") is True:
        return 0
    if "max_concurrent_assignments" in capabilities and _safe_nonnegative_int(capabilities.get("max_concurrent_assignments")) == 0:
        return 0
    base_limit = _safe_positive_int(capabilities.get("max_concurrent_assignments")) or 1
    embedding_limit = _safe_nonnegative_int(capabilities.get("max_concurrent_assignments_embeddings"))
    return max(1, base_limit, embedding_limit)


def max_microbatch_assignments_from_capabilities(capabilities: dict[str, Any]) -> int:
    if capabilities.get("heat_governor_paused") is True:
        return 0
    return max(1, _safe_nonnegative_int(capabilities.get("max_microbatch_assignments_embeddings")) or 1)


def max_local_queue_assignments_from_capabilities(capabilities: dict[str, Any]) -> int:
    if capabilities.get("heat_governor_paused") is True:
        return 0
    explicit_limit = _safe_nonnegative_int(capabilities.get("max_local_queue_assignments"))
    if explicit_limit > 0:
        return explicit_limit
    if "max_concurrent_assignments" in capabilities and _safe_nonnegative_int(capabilities.get("max_concurrent_assignments")) == 0:
        return 0
    base_limit = _safe_positive_int(capabilities.get("max_concurrent_assignments")) or 1
    embedding_limit = _safe_nonnegative_int(capabilities.get("max_concurrent_assignments_embeddings"))
    microbatch_limit = _safe_nonnegative_int(capabilities.get("max_microbatch_assignments_embeddings"))
    pull_bundle_limit = _safe_nonnegative_int(capabilities.get("max_pull_bundle_assignments"))
    return max(1, base_limit, embedding_limit, microbatch_limit, pull_bundle_limit)
