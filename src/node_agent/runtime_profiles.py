from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .release_manifest import DEFAULT_VLLM_IMAGE
from .runtime_backend import MANAGER_RUNTIME_BACKEND, SINGLE_CONTAINER_RUNTIME_BACKEND


AUTO_RUNTIME_PROFILE = "auto"
HOME_LLAMA_CPP_GGUF_PROFILE = "home_llama_cpp_gguf"
VAST_VLLM_SAFETENSORS_PROFILE = "vast_vllm_safetensors"
PARTNER_VLLM_TRUSTED_PROFILE = "partner_vllm_trusted"
HOME_EMBEDDINGS_LLAMA_CPP_PROFILE = "home_embeddings_llama_cpp"

AUTO_INFERENCE_ENGINE = "auto"
VLLM_INFERENCE_ENGINE = "vllm"
LLAMA_CPP_INFERENCE_ENGINE = "llama_cpp"

AUTO_DEPLOYMENT_TARGET = "auto"
HOME_EDGE_DEPLOYMENT_TARGET = "home_edge"
VAST_AI_DEPLOYMENT_TARGET = "vast_ai"
GENERIC_DEPLOYMENT_TARGET = "generic"

MODEL_FORMAT_GGUF = "gguf"
MODEL_FORMAT_SAFETENSORS = "safetensors"

ARTIFACT_MANIFEST_GGUF = "gguf_hf_file"
ARTIFACT_MANIFEST_AUDITED_SAFETENSORS = "audited_safetensors"

TRUST_POLICY_COMMUNITY_BEST_EFFORT = "community_best_effort"
TRUST_POLICY_ELASTIC_UNTRUSTED = "elastic_untrusted"
TRUST_POLICY_TRUSTED_ONLY = "trusted_only"

PRICING_TIER_HOME_HEAT_OFFSET = "home_heat_offset"
PRICING_TIER_ELASTIC_MARKET = "elastic_market"
PRICING_TIER_PARTNER_TRUSTED = "partner_trusted"

CAPACITY_CLASS_HOME_HEAT = "home_heat"
CAPACITY_CLASS_ELASTIC_BURST = "elastic_burst"
CAPACITY_CLASS_TRUSTED_PARTNER = "trusted_partner"
CAPACITY_CLASS_GENERIC = "generic"

TRUSTED_ELIGIBILITY_PROFILE_TRUSTED = "profile_trusted"
TRUSTED_ELIGIBILITY_RUNTIME_AND_MODEL_DIGEST_MATCH = "runtime_and_model_digest_match"
TRUSTED_ELIGIBILITY_NONE = "none"

ROUTING_LANE_COMMUNITY_QUANTIZED_HOME = "community_quantized_home"
ROUTING_LANE_ELASTIC_EXACT_VAST = "elastic_exact_vast"
ROUTING_LANE_TRUSTED_EXACT_PARTNER = "trusted_exact_partner"

BURST_PHASE_PROVISION = "provision"
BURST_PHASE_WARM_MODEL = "warm_model"
BURST_PHASE_REGISTER_TEMPORARY_NODE = "register_temporary_node"
BURST_PHASE_ACCEPT_BURST_WORK = "accept_burst_work"
BURST_PHASE_DRAIN = "drain"
BURST_PHASE_TERMINATE = "terminate"
VAST_BURST_LIFECYCLE = (
    BURST_PHASE_PROVISION,
    BURST_PHASE_WARM_MODEL,
    BURST_PHASE_REGISTER_TEMPORARY_NODE,
    BURST_PHASE_ACCEPT_BURST_WORK,
    BURST_PHASE_DRAIN,
    BURST_PHASE_TERMINATE,
)
DEFAULT_VAST_BURST_COST_CEILING_USD = 0.25

SUPPORTED_API_RESPONSES = "responses"
SUPPORTED_API_EMBEDDINGS = "embeddings"

EMBEDDING_POOLING_CLS = "cls"
DEFAULT_RESPONSE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_LLAMA_CPP_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"


@dataclass(frozen=True)
class LlamaCppModelSource:
    model: str
    hf_repo: str
    hf_file: str
    alias: str
    embedding_enabled: bool = False
    pooling: str | None = None


@dataclass(frozen=True)
class RuntimeProfile:
    id: str
    label: str
    inference_engine: str
    deployment_target: str
    model_format: str
    image: str
    readiness_path: str
    supported_apis: tuple[str, ...]
    trust_policy: str
    pricing_tier: str
    artifact_manifest_type: str
    capacity_class: str
    routing_lane: str
    max_privacy_tier: str
    exact_model_guarantee: bool
    quantized_output_disclosure_required: bool
    trusted_eligibility: str
    burst_lifecycle: tuple[str, ...]
    burst_cost_ceiling_usd: float | None
    default_model: str
    supported_models: tuple[str, ...]

    @property
    def reports_audited_manifests(self) -> bool:
        return self.artifact_manifest_type == ARTIFACT_MANIFEST_AUDITED_SAFETENSORS

    @property
    def supports_trusted_assignments(self) -> bool:
        return self.trusted_eligibility == TRUSTED_ELIGIBILITY_PROFILE_TRUSTED

    @property
    def routing_lane_label(self) -> str:
        return routing_lane_label(self.routing_lane)

    @property
    def routing_lane_detail(self) -> str:
        return routing_lane_detail(self.routing_lane)

    @property
    def routing_lane_policy_summary(self) -> str:
        return routing_lane_policy_summary(self.routing_lane)

    @property
    def routing_lane_allowed_privacy_tiers(self) -> tuple[str, ...]:
        return routing_lane_policy(self.routing_lane).allowed_privacy_tiers

    @property
    def routing_lane_allowed_result_guarantees(self) -> tuple[str, ...]:
        return routing_lane_policy(self.routing_lane).allowed_result_guarantees

    @property
    def routing_lane_allowed_trust_requirements(self) -> tuple[str, ...]:
        return routing_lane_policy(self.routing_lane).allowed_trust_requirements

    def payload(self) -> dict[str, Any]:
        payload = {
            "runtime_profile": self.id,
            "runtime_profile_label": self.label,
            "inference_engine": self.inference_engine,
            "deployment_target": self.deployment_target,
            "model_format": self.model_format,
            "runtime_image": self.image,
            "readiness_path": self.readiness_path,
            "supported_apis": list(self.supported_apis),
            "trust_policy": self.trust_policy,
            "pricing_tier": self.pricing_tier,
            "artifact_manifest_type": self.artifact_manifest_type,
            "capacity_class": self.capacity_class,
            "routing_lane": self.routing_lane,
            "routing_lane_label": self.routing_lane_label,
            "routing_lane_detail": self.routing_lane_detail,
            "routing_lane_policy_summary": self.routing_lane_policy_summary,
            "routing_lane_allowed_privacy_tiers": list(self.routing_lane_allowed_privacy_tiers),
            "routing_lane_allowed_result_guarantees": list(self.routing_lane_allowed_result_guarantees),
            "routing_lane_allowed_trust_requirements": list(self.routing_lane_allowed_trust_requirements),
            "max_privacy_tier": self.max_privacy_tier,
            "exact_model_guarantee": self.exact_model_guarantee,
            "quantized_output_disclosure_required": self.quantized_output_disclosure_required,
            "trusted_eligibility": self.trusted_eligibility,
            "burst_lifecycle": list(self.burst_lifecycle),
        }
        if self.burst_cost_ceiling_usd is not None:
            payload["burst_cost_ceiling_usd"] = self.burst_cost_ceiling_usd
        return payload


LLAMA_CPP_MODEL_SOURCES: dict[str, LlamaCppModelSource] = {
    DEFAULT_RESPONSE_MODEL: LlamaCppModelSource(
        model=DEFAULT_RESPONSE_MODEL,
        hf_repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        hf_file="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        alias=DEFAULT_RESPONSE_MODEL,
    ),
    DEFAULT_EMBEDDING_MODEL: LlamaCppModelSource(
        model=DEFAULT_EMBEDDING_MODEL,
        hf_repo="CompendiumLabs/bge-large-en-v1.5-gguf",
        hf_file="bge-large-en-v1.5-q8_0.gguf",
        alias=DEFAULT_EMBEDDING_MODEL,
        embedding_enabled=True,
        pooling=EMBEDDING_POOLING_CLS,
    ),
}


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    HOME_LLAMA_CPP_GGUF_PROFILE: RuntimeProfile(
        id=HOME_LLAMA_CPP_GGUF_PROFILE,
        label="Home llama.cpp GGUF",
        inference_engine=LLAMA_CPP_INFERENCE_ENGINE,
        deployment_target=HOME_EDGE_DEPLOYMENT_TARGET,
        model_format=MODEL_FORMAT_GGUF,
        image=DEFAULT_LLAMA_CPP_IMAGE,
        readiness_path="/health",
        supported_apis=(SUPPORTED_API_RESPONSES,),
        trust_policy=TRUST_POLICY_COMMUNITY_BEST_EFFORT,
        pricing_tier=PRICING_TIER_HOME_HEAT_OFFSET,
        artifact_manifest_type=ARTIFACT_MANIFEST_GGUF,
        capacity_class=CAPACITY_CLASS_HOME_HEAT,
        routing_lane=ROUTING_LANE_COMMUNITY_QUANTIZED_HOME,
        max_privacy_tier="standard",
        exact_model_guarantee=False,
        quantized_output_disclosure_required=True,
        trusted_eligibility=TRUSTED_ELIGIBILITY_NONE,
        burst_lifecycle=(),
        burst_cost_ceiling_usd=None,
        default_model=DEFAULT_RESPONSE_MODEL,
        supported_models=(DEFAULT_RESPONSE_MODEL, DEFAULT_EMBEDDING_MODEL),
    ),
    VAST_VLLM_SAFETENSORS_PROFILE: RuntimeProfile(
        id=VAST_VLLM_SAFETENSORS_PROFILE,
        label="Vast.ai vLLM safetensors",
        inference_engine=VLLM_INFERENCE_ENGINE,
        deployment_target=VAST_AI_DEPLOYMENT_TARGET,
        model_format=MODEL_FORMAT_SAFETENSORS,
        image=DEFAULT_VLLM_IMAGE,
        readiness_path="/v1/models",
        supported_apis=(SUPPORTED_API_RESPONSES, SUPPORTED_API_EMBEDDINGS),
        trust_policy=TRUST_POLICY_ELASTIC_UNTRUSTED,
        pricing_tier=PRICING_TIER_ELASTIC_MARKET,
        artifact_manifest_type=ARTIFACT_MANIFEST_AUDITED_SAFETENSORS,
        capacity_class=CAPACITY_CLASS_ELASTIC_BURST,
        routing_lane=ROUTING_LANE_ELASTIC_EXACT_VAST,
        max_privacy_tier="restricted",
        exact_model_guarantee=True,
        quantized_output_disclosure_required=False,
        trusted_eligibility=TRUSTED_ELIGIBILITY_RUNTIME_AND_MODEL_DIGEST_MATCH,
        burst_lifecycle=VAST_BURST_LIFECYCLE,
        burst_cost_ceiling_usd=DEFAULT_VAST_BURST_COST_CEILING_USD,
        default_model=DEFAULT_RESPONSE_MODEL,
        supported_models=(DEFAULT_RESPONSE_MODEL, DEFAULT_EMBEDDING_MODEL),
    ),
    PARTNER_VLLM_TRUSTED_PROFILE: RuntimeProfile(
        id=PARTNER_VLLM_TRUSTED_PROFILE,
        label="Partner vLLM trusted",
        inference_engine=VLLM_INFERENCE_ENGINE,
        deployment_target=GENERIC_DEPLOYMENT_TARGET,
        model_format=MODEL_FORMAT_SAFETENSORS,
        image=DEFAULT_VLLM_IMAGE,
        readiness_path="/v1/models",
        supported_apis=(SUPPORTED_API_RESPONSES, SUPPORTED_API_EMBEDDINGS),
        trust_policy=TRUST_POLICY_TRUSTED_ONLY,
        pricing_tier=PRICING_TIER_PARTNER_TRUSTED,
        artifact_manifest_type=ARTIFACT_MANIFEST_AUDITED_SAFETENSORS,
        capacity_class=CAPACITY_CLASS_TRUSTED_PARTNER,
        routing_lane=ROUTING_LANE_TRUSTED_EXACT_PARTNER,
        max_privacy_tier="restricted",
        exact_model_guarantee=True,
        quantized_output_disclosure_required=False,
        trusted_eligibility=TRUSTED_ELIGIBILITY_PROFILE_TRUSTED,
        burst_lifecycle=(),
        burst_cost_ceiling_usd=None,
        default_model=DEFAULT_RESPONSE_MODEL,
        supported_models=(DEFAULT_RESPONSE_MODEL, DEFAULT_EMBEDDING_MODEL),
    ),
    HOME_EMBEDDINGS_LLAMA_CPP_PROFILE: RuntimeProfile(
        id=HOME_EMBEDDINGS_LLAMA_CPP_PROFILE,
        label="Home embeddings llama.cpp",
        inference_engine=LLAMA_CPP_INFERENCE_ENGINE,
        deployment_target=HOME_EDGE_DEPLOYMENT_TARGET,
        model_format=MODEL_FORMAT_GGUF,
        image=DEFAULT_LLAMA_CPP_IMAGE,
        readiness_path="/health",
        supported_apis=(SUPPORTED_API_EMBEDDINGS,),
        trust_policy=TRUST_POLICY_COMMUNITY_BEST_EFFORT,
        pricing_tier=PRICING_TIER_HOME_HEAT_OFFSET,
        artifact_manifest_type=ARTIFACT_MANIFEST_GGUF,
        capacity_class=CAPACITY_CLASS_HOME_HEAT,
        routing_lane=ROUTING_LANE_COMMUNITY_QUANTIZED_HOME,
        max_privacy_tier="standard",
        exact_model_guarantee=False,
        quantized_output_disclosure_required=True,
        trusted_eligibility=TRUSTED_ELIGIBILITY_NONE,
        burst_lifecycle=(),
        burst_cost_ceiling_usd=None,
        default_model=DEFAULT_EMBEDDING_MODEL,
        supported_models=(DEFAULT_EMBEDDING_MODEL,),
    ),
}


@dataclass(frozen=True)
class RoutingLanePolicy:
    label: str
    narrative: str
    allowed_privacy_tiers: tuple[str, ...]
    allowed_result_guarantees: tuple[str, ...]
    allowed_trust_requirements: tuple[str, ...]


ROUTING_LANE_POLICIES: dict[str, RoutingLanePolicy] = {
    ROUTING_LANE_COMMUNITY_QUANTIZED_HOME: RoutingLanePolicy(
        label="Community quantized home",
        narrative="Routes home llama.cpp GGUF capacity into the community lane only.",
        allowed_privacy_tiers=("standard",),
        allowed_result_guarantees=("community_best_effort",),
        allowed_trust_requirements=("untrusted_allowed",),
    ),
    ROUTING_LANE_ELASTIC_EXACT_VAST: RoutingLanePolicy(
        label="Elastic exact Vast.ai",
        narrative="Routes temporary Vast.ai vLLM burst capacity for elastic exact-model work.",
        allowed_privacy_tiers=("standard", "confidential", "restricted"),
        allowed_result_guarantees=("community_best_effort", "exact_model_audited"),
        allowed_trust_requirements=("untrusted_allowed", "trusted_only"),
    ),
    ROUTING_LANE_TRUSTED_EXACT_PARTNER: RoutingLanePolicy(
        label="Trusted exact partner",
        narrative="Routes verified partner vLLM capacity for premium and audited exact-model work.",
        allowed_privacy_tiers=("standard", "confidential", "restricted"),
        allowed_result_guarantees=("community_best_effort", "exact_model_audited"),
        allowed_trust_requirements=("untrusted_allowed", "trusted_only"),
    ),
}


def _normalize_known(value: str | None, *, known: set[str], fallback: str) -> str:
    normalized = (value or fallback).strip().lower().replace("-", "_")
    return normalized if normalized in known else fallback


def normalize_runtime_profile_id(value: str | None) -> str:
    return _normalize_known(
        value,
        known=set(RUNTIME_PROFILES),
        fallback=AUTO_RUNTIME_PROFILE,
    )


def normalize_deployment_target(value: str | None) -> str:
    return _normalize_known(
        value,
        known={HOME_EDGE_DEPLOYMENT_TARGET, VAST_AI_DEPLOYMENT_TARGET, GENERIC_DEPLOYMENT_TARGET},
        fallback=AUTO_DEPLOYMENT_TARGET,
    )


def normalize_inference_engine(value: str | None) -> str:
    return _normalize_known(
        value,
        known={VLLM_INFERENCE_ENGINE, LLAMA_CPP_INFERENCE_ENGINE},
        fallback=AUTO_INFERENCE_ENGINE,
    )


def runtime_profile_by_id(profile_id: str | None) -> RuntimeProfile | None:
    normalized = normalize_runtime_profile_id(profile_id)
    if normalized == AUTO_RUNTIME_PROFILE:
        return None
    return RUNTIME_PROFILES[normalized]


def deployment_target_for_runtime_backend(runtime_backend: str) -> str:
    if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
        return VAST_AI_DEPLOYMENT_TARGET
    if runtime_backend == MANAGER_RUNTIME_BACKEND:
        return HOME_EDGE_DEPLOYMENT_TARGET
    return GENERIC_DEPLOYMENT_TARGET


def _resolve_deployment_target_legacy(configured: str | None, runtime_backend: str) -> str:
    normalized = normalize_deployment_target(configured)
    if normalized != AUTO_DEPLOYMENT_TARGET:
        return normalized
    return deployment_target_for_runtime_backend(runtime_backend)


def _resolve_inference_engine_legacy(
    configured_engine: str | None,
    *,
    deployment_target: str,
    runtime_backend: str,
) -> str:
    normalized = normalize_inference_engine(configured_engine)
    if normalized != AUTO_INFERENCE_ENGINE:
        return normalized
    if deployment_target == VAST_AI_DEPLOYMENT_TARGET:
        return VLLM_INFERENCE_ENGINE
    if deployment_target == HOME_EDGE_DEPLOYMENT_TARGET:
        return LLAMA_CPP_INFERENCE_ENGINE
    if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
        return VLLM_INFERENCE_ENGINE
    return LLAMA_CPP_INFERENCE_ENGINE


def _profile_id_for_legacy_runtime(engine: str, target: str, model: str | None) -> str:
    if engine == LLAMA_CPP_INFERENCE_ENGINE:
        source = llama_cpp_model_source(model)
        if source is not None and source.embedding_enabled:
            return HOME_EMBEDDINGS_LLAMA_CPP_PROFILE
        return HOME_LLAMA_CPP_GGUF_PROFILE
    if target == VAST_AI_DEPLOYMENT_TARGET:
        return VAST_VLLM_SAFETENSORS_PROFILE
    return PARTNER_VLLM_TRUSTED_PROFILE


def resolve_runtime_profile(
    configured_profile: str | None,
    *,
    configured_engine: str | None,
    configured_deployment_target: str | None,
    runtime_backend: str,
    model: str | None = None,
) -> RuntimeProfile:
    profile_id = normalize_runtime_profile_id(configured_profile)
    if profile_id != AUTO_RUNTIME_PROFILE:
        return RUNTIME_PROFILES[profile_id]
    target = _resolve_deployment_target_legacy(configured_deployment_target, runtime_backend)
    engine = _resolve_inference_engine_legacy(
        configured_engine,
        deployment_target=target,
        runtime_backend=runtime_backend,
    )
    return RUNTIME_PROFILES[_profile_id_for_legacy_runtime(engine, target, model)]


def resolve_deployment_target(configured: str | None, runtime_backend: str) -> str:
    return _resolve_deployment_target_legacy(configured, runtime_backend)


def resolve_inference_engine(
    configured_engine: str | None,
    *,
    deployment_target: str,
    runtime_backend: str,
) -> str:
    return _resolve_inference_engine_legacy(
        configured_engine,
        deployment_target=deployment_target,
        runtime_backend=runtime_backend,
    )


def inference_engine_label(engine: str) -> str:
    if engine == LLAMA_CPP_INFERENCE_ENGINE:
        return "llama.cpp"
    if engine == VLLM_INFERENCE_ENGINE:
        return "vLLM"
    return "Automatic"


def deployment_target_label(target: str) -> str:
    if target == HOME_EDGE_DEPLOYMENT_TARGET:
        return "Home edge"
    if target == VAST_AI_DEPLOYMENT_TARGET:
        return "Vast.ai"
    if target == GENERIC_DEPLOYMENT_TARGET:
        return "Generic"
    return "Automatic"


def routing_lane_policy(lane: str) -> RoutingLanePolicy:
    return ROUTING_LANE_POLICIES.get(
        lane,
        RoutingLanePolicy(
            label="Automatic",
            narrative="Routing lane workload policy is selected automatically from the active runtime profile.",
            allowed_privacy_tiers=(),
            allowed_result_guarantees=(),
            allowed_trust_requirements=(),
        ),
    )


def routing_lane_label(lane: str) -> str:
    return routing_lane_policy(lane).label


def _format_policy_list(values: tuple[str, ...]) -> str:
    if not values:
        return "none"
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _format_privacy_tier(tier: str) -> str:
    return tier


def _format_result_guarantee(guarantee: str) -> str:
    if guarantee == "exact_model_audited":
        return "exact audited results"
    return "community best-effort results"


def _format_trust_requirement(requirement: str) -> str:
    if requirement == "trusted_only":
        return "trusted-only jobs"
    return "untrusted-allowed jobs"


def routing_lane_detail(lane: str) -> str:
    return routing_lane_policy_summary(lane)


def routing_lane_policy_summary(lane: str) -> str:
    policy = routing_lane_policy(lane)
    if not policy.allowed_privacy_tiers:
        return policy.narrative
    return (
        f"{policy.narrative} Allowed privacy tiers: "
        f"{_format_policy_list(tuple(_format_privacy_tier(tier) for tier in policy.allowed_privacy_tiers))}. "
        f"Allowed result guarantees: "
        f"{_format_policy_list(tuple(_format_result_guarantee(value) for value in policy.allowed_result_guarantees))}. "
        f"Allowed trust requirements: "
        f"{_format_policy_list(tuple(_format_trust_requirement(value) for value in policy.allowed_trust_requirements))}."
    )


def default_runtime_profile(runtime_backend: str) -> RuntimeProfile:
    return resolve_runtime_profile(
        None,
        configured_engine=None,
        configured_deployment_target=None,
        runtime_backend=runtime_backend,
    )


def default_inference_base_url(_engine_or_profile: str) -> str:
    # The compose service still uses the "vllm" name for compatibility, but it also exposes
    # a neutral "inference-runtime" alias so internal URLs do not imply a specific engine.
    return "http://inference-runtime:8000"


def readiness_probe_path(engine_or_profile: str) -> str:
    profile = runtime_profile_by_id(engine_or_profile)
    if profile is not None:
        return profile.readiness_path
    if engine_or_profile == LLAMA_CPP_INFERENCE_ENGINE:
        return "/health"
    return "/v1/models"


def runtime_supports_audited_manifests(engine_or_profile: str) -> bool:
    profile = runtime_profile_by_id(engine_or_profile)
    if profile is not None:
        return profile.reports_audited_manifests
    return engine_or_profile == VLLM_INFERENCE_ENGINE


def runtime_profile_supports_trusted_assignments(profile_id: str | None) -> bool:
    profile = runtime_profile_by_id(profile_id)
    return bool(profile and profile.supports_trusted_assignments)


def llama_cpp_model_source(model: str | None) -> LlamaCppModelSource | None:
    if not isinstance(model, str) or not model.strip():
        return None
    return LLAMA_CPP_MODEL_SOURCES.get(model.strip())


__all__ = [
    "ARTIFACT_MANIFEST_AUDITED_SAFETENSORS",
    "ARTIFACT_MANIFEST_GGUF",
    "AUTO_DEPLOYMENT_TARGET",
    "AUTO_INFERENCE_ENGINE",
    "AUTO_RUNTIME_PROFILE",
    "BURST_PHASE_ACCEPT_BURST_WORK",
    "BURST_PHASE_DRAIN",
    "BURST_PHASE_PROVISION",
    "BURST_PHASE_REGISTER_TEMPORARY_NODE",
    "BURST_PHASE_TERMINATE",
    "BURST_PHASE_WARM_MODEL",
    "CAPACITY_CLASS_ELASTIC_BURST",
    "CAPACITY_CLASS_GENERIC",
    "CAPACITY_CLASS_HOME_HEAT",
    "CAPACITY_CLASS_TRUSTED_PARTNER",
    "DEFAULT_VAST_BURST_COST_CEILING_USD",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_LLAMA_CPP_IMAGE",
    "DEFAULT_RESPONSE_MODEL",
    "EMBEDDING_POOLING_CLS",
    "GENERIC_DEPLOYMENT_TARGET",
    "HOME_EDGE_DEPLOYMENT_TARGET",
    "HOME_EMBEDDINGS_LLAMA_CPP_PROFILE",
    "HOME_LLAMA_CPP_GGUF_PROFILE",
    "LLAMA_CPP_INFERENCE_ENGINE",
    "LLAMA_CPP_MODEL_SOURCES",
    "LlamaCppModelSource",
    "MODEL_FORMAT_GGUF",
    "MODEL_FORMAT_SAFETENSORS",
    "PARTNER_VLLM_TRUSTED_PROFILE",
    "PRICING_TIER_ELASTIC_MARKET",
    "PRICING_TIER_HOME_HEAT_OFFSET",
    "PRICING_TIER_PARTNER_TRUSTED",
    "ROUTING_LANE_COMMUNITY_QUANTIZED_HOME",
    "ROUTING_LANE_ELASTIC_EXACT_VAST",
    "ROUTING_LANE_TRUSTED_EXACT_PARTNER",
    "ROUTING_LANE_POLICIES",
    "RUNTIME_PROFILES",
    "RoutingLanePolicy",
    "RuntimeProfile",
    "SUPPORTED_API_EMBEDDINGS",
    "SUPPORTED_API_RESPONSES",
    "TRUST_POLICY_COMMUNITY_BEST_EFFORT",
    "TRUST_POLICY_ELASTIC_UNTRUSTED",
    "TRUST_POLICY_TRUSTED_ONLY",
    "TRUSTED_ELIGIBILITY_NONE",
    "TRUSTED_ELIGIBILITY_PROFILE_TRUSTED",
    "TRUSTED_ELIGIBILITY_RUNTIME_AND_MODEL_DIGEST_MATCH",
    "VAST_AI_DEPLOYMENT_TARGET",
    "VAST_BURST_LIFECYCLE",
    "VAST_VLLM_SAFETENSORS_PROFILE",
    "VLLM_INFERENCE_ENGINE",
    "default_inference_base_url",
    "default_runtime_profile",
    "deployment_target_for_runtime_backend",
    "deployment_target_label",
    "inference_engine_label",
    "llama_cpp_model_source",
    "normalize_deployment_target",
    "normalize_inference_engine",
    "normalize_runtime_profile_id",
    "readiness_probe_path",
    "resolve_deployment_target",
    "resolve_inference_engine",
    "resolve_runtime_profile",
    "routing_lane_detail",
    "routing_lane_label",
    "routing_lane_policy",
    "routing_lane_policy_summary",
    "runtime_profile_by_id",
    "runtime_profile_supports_trusted_assignments",
    "runtime_supports_audited_manifests",
]
