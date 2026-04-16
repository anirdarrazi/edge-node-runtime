from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Any

from .gguf_artifacts import GgufArtifactContract
from .runtime_quality import resolved_runtime_quality
from .runtime_profiles import LLAMA_CPP_INFERENCE_ENGINE
from .runtime_tuple import RuntimeTuple

RUNTIME_EVIDENCE_SIGNATURE_ALGORITHM = "hmac-sha256"
RUNTIME_EVIDENCE_BUNDLE_VERSION = "runtime_evidence.v1"


def _sorted_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sorted_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_json(item) for item in value]
    return value


def _sha256_json(value: Any) -> str:
    payload = json.dumps(_sorted_json(value), separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _signature_payload(*, digest: str, signed_at: str, signature_algorithm: str) -> str:
    payload = {
        "digest": digest,
        "signed_at": signed_at,
        "signature_algorithm": signature_algorithm,
    }
    return json.dumps(_sorted_json(payload), separators=(",", ":"), ensure_ascii=True)


def _startup_args(settings: object) -> dict[str, Any]:
    common = {
        "current_model": getattr(settings, "current_model", None),
        "max_context_tokens": getattr(settings, "max_context_tokens", None),
        "max_batch_tokens": getattr(settings, "max_batch_tokens", None),
        "max_concurrent_assignments": getattr(settings, "max_concurrent_assignments", None),
        "inference_base_url": getattr(settings, "resolved_inference_base_url", None),
    }
    if getattr(settings, "resolved_inference_engine", None) == LLAMA_CPP_INFERENCE_ENGINE:
        return {
            **common,
            "hf_repo": getattr(settings, "llama_cpp_hf_repo", None),
            "hf_file": getattr(settings, "llama_cpp_hf_file", None),
            "alias": getattr(settings, "llama_cpp_alias", None),
            "embedding": getattr(settings, "llama_cpp_embedding", None),
            "pooling": getattr(settings, "llama_cpp_pooling", None),
        }
    return {
        **common,
        "served_model": getattr(settings, "current_model", None),
    }


def _driver_metadata(settings: object) -> dict[str, Any]:
    return {
        "runtime_backend": getattr(settings, "runtime_backend", None),
        "attestation_provider": getattr(settings, "attestation_provider", None),
        "gpu_name": getattr(settings, "gpu_name", None),
        "gpu_memory_gb": getattr(settings, "gpu_memory_gb", None),
        "driver_version": os.getenv("NVIDIA_DRIVER_VERSION") or os.getenv("DRIVER_VERSION") or None,
        "cuda_version": os.getenv("CUDA_VERSION") or None,
        "nvidia_visible_devices": os.getenv("NVIDIA_VISIBLE_DEVICES") or None,
    }


def resolved_runtime_evidence_bundle(
    settings: object,
    runtime_tuple: RuntimeTuple,
    gguf_artifact: GgufArtifactContract | None,
) -> dict[str, Any]:
    profile = getattr(settings, "resolved_runtime_profile", None)
    runtime_quality = resolved_runtime_quality(settings, gguf_artifact)
    bundle: dict[str, Any] = {
        "runtime_evidence_bundle_version": RUNTIME_EVIDENCE_BUNDLE_VERSION,
        "runtime_profile": getattr(profile, "id", None),
        "inference_engine": getattr(settings, "resolved_inference_engine", None),
        "deployment_target": getattr(settings, "resolved_deployment_target", None),
        "capacity_class": getattr(settings, "resolved_capacity_class", None),
        "routing_lane": getattr(settings, "resolved_routing_lane", None),
        "model_format": getattr(profile, "model_format", None),
        "quality_class": runtime_quality["quality_class"],
        "exactness_class": runtime_quality["exactness_class"],
        "artifact_manifest_type": getattr(profile, "artifact_manifest_type", None),
        "runtime_image": getattr(settings, "resolved_runtime_image", None),
        "runtime_image_digest": runtime_tuple.runtime_image_digest,
        "current_model": getattr(settings, "current_model", None),
        "model_manifest_digest": runtime_tuple.model_manifest_digest,
        "gguf_file_digest": runtime_tuple.gguf_file_digest,
        "quantization_type": runtime_tuple.quantization_type,
        "tokenizer_digest": runtime_tuple.tokenizer_digest,
        "chat_template_digest": runtime_tuple.chat_template_digest,
        "effective_context_tokens": runtime_tuple.effective_context_tokens,
        "runtime_tuple_digest": runtime_tuple.runtime_tuple_digest,
        "startup_args": _startup_args(settings),
        "driver_metadata": _driver_metadata(settings),
    }
    if gguf_artifact is not None:
        bundle["gguf_artifact"] = gguf_artifact.payload()
    return bundle


def sign_runtime_evidence(
    node_key: str,
    *,
    digest: str,
    signed_at: str,
    signature_algorithm: str = RUNTIME_EVIDENCE_SIGNATURE_ALGORITHM,
) -> str:
    payload = _signature_payload(
        digest=digest,
        signed_at=signed_at,
        signature_algorithm=signature_algorithm,
    ).encode("utf-8")
    signature = hmac.new(node_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"sha256:{signature}"


def resolved_signed_runtime_evidence(
    settings: object,
    runtime_tuple: RuntimeTuple,
    gguf_artifact: GgufArtifactContract | None,
) -> dict[str, Any] | None:
    node_key = getattr(settings, "node_key", None)
    if not isinstance(node_key, str) or not node_key:
        return None
    bundle = resolved_runtime_evidence_bundle(settings, runtime_tuple, gguf_artifact)
    digest = _sha256_json(bundle)
    signed_at = datetime.now(timezone.utc).isoformat()
    return {
        "digest": digest,
        "signature": sign_runtime_evidence(node_key, digest=digest, signed_at=signed_at),
        "signature_algorithm": RUNTIME_EVIDENCE_SIGNATURE_ALGORITHM,
        "signed_at": signed_at,
        "bundle": bundle,
    }


__all__ = [
    "RUNTIME_EVIDENCE_BUNDLE_VERSION",
    "RUNTIME_EVIDENCE_SIGNATURE_ALGORITHM",
    "resolved_runtime_evidence_bundle",
    "resolved_signed_runtime_evidence",
    "sign_runtime_evidence",
]
