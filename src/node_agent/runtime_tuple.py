from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .gguf_artifacts import resolved_gguf_artifact_contract
from .model_artifacts import (
    resolved_chat_template_digest,
    resolved_model_manifest_digest,
    resolved_tokenizer_digest,
)
from .runtime_profiles import DEFAULT_EMBEDDING_MODEL, llama_cpp_model_source

KNOWN_EFFECTIVE_CONTEXT_TOKENS: dict[tuple[str, str], int] = {
    (DEFAULT_EMBEDDING_MODEL, "embeddings"): 512,
}


@dataclass(frozen=True)
class RuntimeTuple:
    runtime_image_digest: str | None
    model_manifest_digest: str | None
    tokenizer_digest: str | None
    chat_template_digest: str | None
    gguf_file_digest: str | None
    quantization_type: str | None
    effective_context_tokens: int | None
    runtime_tuple_digest: str

    def payload(self) -> dict[str, Any]:
        return {
            "runtime_image_digest": self.runtime_image_digest,
            "model_manifest_digest": self.model_manifest_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "chat_template_digest": self.chat_template_digest,
            "gguf_file_digest": self.gguf_file_digest,
            "quantization_type": self.quantization_type,
            "effective_context_tokens": self.effective_context_tokens,
            "runtime_tuple_digest": self.runtime_tuple_digest,
        }


def _sorted_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sorted_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_json(item) for item in value]
    return value


def _sha256_json(value: Any) -> str:
    payload = json.dumps(_sorted_json(value), separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _known_effective_context_tokens(model: str | None, operation: str | None) -> int | None:
    if not isinstance(model, str) or not model or not isinstance(operation, str) or not operation:
        return None
    value = KNOWN_EFFECTIVE_CONTEXT_TOKENS.get((model, operation))
    return value if isinstance(value, int) and value > 0 else None


def _effective_context_tokens(
    settings: object,
    model: str | None,
    operation: str | None,
    gguf_expected_context: int | None,
) -> int | None:
    max_context = getattr(settings, "max_context_tokens", None)
    local_max = max_context if isinstance(max_context, int) and max_context > 0 else None
    known_context = _known_effective_context_tokens(model, operation)
    expected_context = gguf_expected_context if gguf_expected_context is not None else known_context
    if local_max is not None and expected_context is not None:
        return min(local_max, expected_context)
    return local_max or expected_context


def _default_operation_for_settings(settings: object) -> str:
    profile = getattr(settings, "resolved_runtime_profile", None)
    supported_apis = getattr(profile, "supported_apis", ())
    if isinstance(supported_apis, tuple) and len(supported_apis) == 1:
        value = supported_apis[0]
        if isinstance(value, str) and value:
            return value
    current_model = getattr(settings, "current_model", None)
    if not isinstance(current_model, str) or not current_model:
        current_model = getattr(settings, "vllm_model", None)
    source = llama_cpp_model_source(current_model if isinstance(current_model, str) else None)
    if source is not None and source.embedding_enabled:
        return "embeddings"
    if current_model == DEFAULT_EMBEDDING_MODEL:
        return "embeddings"
    return "responses"


def resolved_runtime_tuple(settings: object, model: str | None, operation: str | None) -> RuntimeTuple:
    gguf_artifact = resolved_gguf_artifact_contract(settings, model, operation)
    runtime_image_digest = getattr(settings, "docker_image", None)
    if not isinstance(runtime_image_digest, str) or not runtime_image_digest:
        runtime_image_digest = None
    model_manifest_digest = resolved_model_manifest_digest(settings, model, operation)
    tokenizer_digest = resolved_tokenizer_digest(settings, model, operation)
    chat_template_digest = resolved_chat_template_digest(settings, model, operation)
    gguf_file_digest = gguf_artifact.file_digest if gguf_artifact is not None else None
    quantization_type = gguf_artifact.quantization_type if gguf_artifact is not None else None
    effective_context_tokens = _effective_context_tokens(
        settings,
        model,
        operation,
        gguf_artifact.expected_context_tokens if gguf_artifact is not None else None,
    )
    if chat_template_digest is None and gguf_artifact is not None and gguf_artifact.capabilities.chat:
        chat_template_digest = gguf_artifact.file_digest
    fields = {
        "runtime_image_digest": runtime_image_digest,
        "model_manifest_digest": model_manifest_digest,
        "tokenizer_digest": tokenizer_digest,
        "chat_template_digest": chat_template_digest,
        "gguf_file_digest": gguf_file_digest,
        "quantization_type": quantization_type,
        "effective_context_tokens": effective_context_tokens,
    }
    return RuntimeTuple(
        runtime_image_digest=runtime_image_digest,
        model_manifest_digest=model_manifest_digest,
        tokenizer_digest=tokenizer_digest,
        chat_template_digest=chat_template_digest,
        gguf_file_digest=gguf_file_digest,
        quantization_type=quantization_type,
        effective_context_tokens=effective_context_tokens,
        runtime_tuple_digest=_sha256_json(fields),
    )


def resolved_default_runtime_tuple(settings: object) -> RuntimeTuple:
    current_model = getattr(settings, "current_model", None)
    if not isinstance(current_model, str) or not current_model:
        current_model = getattr(settings, "vllm_model", None)
    model = current_model if isinstance(current_model, str) and current_model else None
    return resolved_runtime_tuple(settings, model, _default_operation_for_settings(settings))
