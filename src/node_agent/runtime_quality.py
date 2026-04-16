from __future__ import annotations

QUALITY_CLASS_EXACT_AUDITED = "exact_audited"
QUALITY_CLASS_QUANTIZED_HIGH_FIDELITY = "quantized_high_fidelity"
QUALITY_CLASS_QUANTIZED_BALANCED = "quantized_balanced"
QUALITY_CLASS_QUANTIZED_ECONOMY = "quantized_economy"
QUALITY_CLASS_QUANTIZED_UNKNOWN = "quantized_unknown"

EXACTNESS_CLASS_EXACT_AUDITED = "exact_audited"
EXACTNESS_CLASS_QUANTIZED_VERIFIED = "quantized_verified"
EXACTNESS_CLASS_QUANTIZED_BEST_EFFORT = "quantized_best_effort"

QUALITY_PRICE_MULTIPLIERS: dict[str, float] = {
    QUALITY_CLASS_EXACT_AUDITED: 1.0,
    QUALITY_CLASS_QUANTIZED_HIGH_FIDELITY: 1.15,
    QUALITY_CLASS_QUANTIZED_BALANCED: 1.0,
    QUALITY_CLASS_QUANTIZED_ECONOMY: 0.85,
    QUALITY_CLASS_QUANTIZED_UNKNOWN: 0.9,
}


def quality_class_for_model_format(model_format: str | None, quantization_type: str | None) -> str:
    if model_format != "gguf":
        return QUALITY_CLASS_EXACT_AUDITED

    normalized = (quantization_type or "").strip().upper()
    if not normalized:
        return QUALITY_CLASS_QUANTIZED_UNKNOWN
    if normalized.startswith(("Q8", "IQ8")) or normalized in {"F16", "BF16"}:
        return QUALITY_CLASS_QUANTIZED_HIGH_FIDELITY
    if normalized.startswith(("Q6", "Q5", "IQ5")):
        return QUALITY_CLASS_QUANTIZED_BALANCED
    if normalized.startswith(("Q4", "IQ4", "Q3", "IQ3", "Q2", "IQ2")):
        return QUALITY_CLASS_QUANTIZED_ECONOMY
    return QUALITY_CLASS_QUANTIZED_UNKNOWN


def default_exactness_class_for_quality_class(quality_class: str) -> str:
    if quality_class == QUALITY_CLASS_EXACT_AUDITED:
        return EXACTNESS_CLASS_EXACT_AUDITED
    return EXACTNESS_CLASS_QUANTIZED_BEST_EFFORT


def quality_price_multiplier_for_quality_class(quality_class: str) -> float:
    return QUALITY_PRICE_MULTIPLIERS.get(quality_class, QUALITY_PRICE_MULTIPLIERS[QUALITY_CLASS_QUANTIZED_UNKNOWN])


def resolved_runtime_quality(settings: object, gguf_artifact: object | None) -> dict[str, str | float]:
    model_format = getattr(getattr(settings, "resolved_runtime_profile", None), "model_format", None)
    quantization_type = getattr(gguf_artifact, "quantization_type", None)
    quality_class = quality_class_for_model_format(model_format, quantization_type)
    return {
        "quality_class": quality_class,
        "exactness_class": default_exactness_class_for_quality_class(quality_class),
        "quality_price_multiplier": quality_price_multiplier_for_quality_class(quality_class),
    }


__all__ = [
    "EXACTNESS_CLASS_EXACT_AUDITED",
    "EXACTNESS_CLASS_QUANTIZED_BEST_EFFORT",
    "EXACTNESS_CLASS_QUANTIZED_VERIFIED",
    "QUALITY_CLASS_EXACT_AUDITED",
    "QUALITY_CLASS_QUANTIZED_BALANCED",
    "QUALITY_CLASS_QUANTIZED_ECONOMY",
    "QUALITY_CLASS_QUANTIZED_HIGH_FIDELITY",
    "QUALITY_CLASS_QUANTIZED_UNKNOWN",
    "default_exactness_class_for_quality_class",
    "quality_class_for_model_format",
    "quality_price_multiplier_for_quality_class",
    "resolved_runtime_quality",
]
