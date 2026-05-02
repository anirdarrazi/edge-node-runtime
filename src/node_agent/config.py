from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .heat_governor import (
    DEFAULT_GPU_TEMP_LIMIT_C,
    DEFAULT_HEAT_GOVERNOR_MODE,
    normalize_heat_governor_mode,
    normalize_local_clock,
    normalize_owner_objective,
)
from .inference_engine import (
    AUTO_DEPLOYMENT_TARGET,
    AUTO_INFERENCE_ENGINE,
    AUTO_RUNTIME_PROFILE,
    DEFAULT_RESPONSE_MODEL,
    EMBEDDING_POOLING_CLS,
    default_inference_base_url,
    llama_cpp_model_source,
    normalize_runtime_profile_id,
    resolve_runtime_profile,
)
from .release_manifest import DEFAULT_NODE_AGENT_IMAGE
from .runtime_backend import detect_runtime_backend


class NodeAgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    edge_control_url: str = Field(default="http://edge-control:8787")
    edge_control_fallback_urls: str | None = None
    artifact_mirror_base_urls: str | None = None
    operator_token: str | None = None
    node_label: str = "AUTONOMOUSc Edge Node"
    node_region: str = "eu-se-1"
    trust_tier: str = "restricted"
    restricted_capable: bool = True
    node_id: str | None = None
    node_key: str | None = None
    node_session_id: str | None = None
    boot_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    credentials_path: str = "/var/lib/autonomousc/credentials/node-credentials.json"
    attestation_state_path: str = "/var/lib/autonomousc/credentials/attestation-state.json"
    recovery_note_path: str = "/var/lib/autonomousc/credentials/recovery-note.txt"
    autopilot_state_path: str = "/var/lib/autonomousc/scratch/autopilot-state.json"
    heat_governor_state_path: str = "/var/lib/autonomousc/scratch/heat-governor-state.json"
    control_plane_state_path: str = "/var/lib/autonomousc/scratch/control-plane-state.json"
    fault_injection_state_path: str = "/var/lib/autonomousc/scratch/fault-injection-state.json"
    runtime_profile: str = AUTO_RUNTIME_PROFILE
    deployment_target: str = AUTO_DEPLOYMENT_TARGET
    inference_engine: str = AUTO_INFERENCE_ENGINE
    inference_base_url: str | None = None
    runtime_image: str | None = None
    vllm_image: str | None = None
    vllm_base_url: str | None = None
    vllm_startup_timeout_seconds: int = 600
    vllm_extra_args: str = ""
    vllm_memory_profiler_estimate_cudagraphs: bool = False
    vllm_model: str = DEFAULT_RESPONSE_MODEL
    owner_target_model: str | None = None
    owner_target_supported_models: str | None = None
    llama_cpp_hf_repo: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    llama_cpp_hf_file: str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    llama_cpp_alias: str = "meta-llama/Llama-3.1-8B-Instruct"
    llama_cpp_embedding: bool = False
    llama_cpp_pooling: str | None = EMBEDDING_POOLING_CLS
    capacity_class: str | None = None
    temporary_node: bool = False
    burst_provider: Literal["vast_ai"] | None = None
    burst_lease_id: str | None = None
    burst_lease_phase: Literal[
        "provision",
        "warm_model",
        "register_temporary_node",
        "accept_burst_work",
        "drain",
        "terminate",
    ] | None = None
    burst_cost_ceiling_usd: float | None = None
    gpu_name: str = "Generic GPU"
    gpu_memory_gb: float = 24.0
    max_context_tokens: int = 32768
    max_batch_tokens: int = 50000
    max_concurrent_assignments: int = 2
    max_concurrent_assignments_cap: int | None = None
    max_concurrent_assignments_embeddings: int | None = None
    max_microbatch_assignments_embeddings: int | None = None
    max_local_queue_assignments: int | None = None
    pull_bundle_size: int = 16
    model_cache_budget_gb: float | None = None
    model_cache_reserve_free_gb: float | None = None
    offline_install_bundle_dir: str | None = None
    heat_governor_mode: str = DEFAULT_HEAT_GOVERNOR_MODE
    owner_objective: Literal["balanced", "earnings_only", "heat_first"] = "balanced"
    target_gpu_utilization_pct: int = 100
    min_gpu_memory_headroom_pct: float = 20.0
    allow_high_gpu_memory_pressure: bool = False
    thermal_headroom: float = 0.8
    heat_demand: Literal["none", "low", "medium", "high"] = "none"
    room_temp_c: float | None = None
    target_temp_c: float | None = None
    outside_temp_c: float | None = None
    quiet_hours_start_local: str | None = None
    quiet_hours_end_local: str | None = None
    gpu_temp_c: float | None = None
    gpu_temp_limit_c: float = DEFAULT_GPU_TEMP_LIMIT_C
    power_watts: float | None = None
    estimated_heat_output_watts: float | None = None
    gpu_power_limit_enabled: bool = True
    max_power_cap_watts: int | None = None
    energy_price_kwh: float | None = None
    supported_models: str = "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    poll_interval_seconds: int = 10
    control_plane_grace_seconds: int = 180
    control_plane_retry_floor_seconds: int = 3
    control_plane_retry_cap_seconds: int = 30
    agent_version: str = "0.1.0"
    docker_image: str = DEFAULT_NODE_AGENT_IMAGE
    model_manifest_digest: str | None = None
    tokenizer_digest: str | None = None
    attestation_provider: Literal["simulated", "hardware"] = "simulated"
    restricted_attestation_max_age_seconds: int = 3600

    @field_validator(
        "operator_token",
        "node_id",
        "node_key",
        "node_session_id",
        "edge_control_fallback_urls",
        "artifact_mirror_base_urls",
        "inference_base_url",
        "runtime_image",
        "vllm_image",
        "vllm_base_url",
        "owner_target_model",
        "owner_target_supported_models",
        "llama_cpp_pooling",
        "capacity_class",
        "burst_provider",
        "burst_lease_id",
        "burst_lease_phase",
        "burst_cost_ceiling_usd",
        "max_concurrent_assignments_embeddings",
        "max_microbatch_assignments_embeddings",
        "max_local_queue_assignments",
        "model_cache_budget_gb",
        "model_cache_reserve_free_gb",
        "offline_install_bundle_dir",
        "room_temp_c",
        "target_temp_c",
        "outside_temp_c",
        "quiet_hours_start_local",
        "quiet_hours_end_local",
        "gpu_temp_c",
        "gpu_temp_limit_c",
        "power_watts",
        "estimated_heat_output_watts",
        "max_power_cap_watts",
        "energy_price_kwh",
        "model_manifest_digest",
        "tokenizer_digest",
        mode="before",
    )
    @classmethod
    def _blank_optional_env_as_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("target_gpu_utilization_pct", mode="before")
    @classmethod
    def _clamp_target_gpu_utilization_pct(cls, value: Any) -> int:
        if value is None or (isinstance(value, str) and not value.strip()):
            return 100
        try:
            parsed = int(float(str(value).strip()))
        except (TypeError, ValueError):
            return 100
        return min(100, max(30, parsed))

    @field_validator("boot_id", mode="before")
    @classmethod
    def _default_blank_boot_id(cls, value: Any) -> str:
        if value is None or (isinstance(value, str) and not value.strip()):
            return uuid.uuid4().hex
        return str(value).strip()

    @field_validator("heat_governor_mode", mode="before")
    @classmethod
    def _normalize_heat_governor_mode(cls, value: Any) -> str:
        return normalize_heat_governor_mode(value)

    @field_validator("owner_objective", mode="before")
    @classmethod
    def _normalize_owner_objective(cls, value: Any) -> str:
        return normalize_owner_objective(value)

    @field_validator("quiet_hours_start_local", "quiet_hours_end_local", mode="before")
    @classmethod
    def _normalize_local_clock(cls, value: Any) -> str | None:
        return normalize_local_clock(value)

    @field_validator("gpu_temp_limit_c", mode="before")
    @classmethod
    def _clamp_gpu_temp_limit_c(cls, value: Any) -> float:
        if value is None or (isinstance(value, str) and not value.strip()):
            return DEFAULT_GPU_TEMP_LIMIT_C
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            return DEFAULT_GPU_TEMP_LIMIT_C
        return min(95.0, max(65.0, parsed))

    @field_validator("min_gpu_memory_headroom_pct", mode="before")
    @classmethod
    def _clamp_min_gpu_memory_headroom_pct(cls, value: Any) -> float:
        if value is None or (isinstance(value, str) and not str(value).strip()):
            return 20.0
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            return 20.0
        return min(60.0, max(5.0, parsed))

    @field_validator("model_cache_budget_gb", "model_cache_reserve_free_gb", mode="before")
    @classmethod
    def _positive_optional_float(cls, value: Any) -> float | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @field_validator("max_power_cap_watts", mode="before")
    @classmethod
    def _positive_optional_int(cls, value: Any) -> int | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            parsed = int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @property
    def runtime_backend(self) -> str:
        return detect_runtime_backend()

    @property
    def resolved_runtime_profile(self):
        return resolve_runtime_profile(
            self.runtime_profile,
            configured_engine=self.inference_engine,
            configured_deployment_target=self.deployment_target,
            runtime_backend=self.runtime_backend,
            model=self.vllm_model,
        )

    @property
    def resolved_runtime_profile_id(self) -> str:
        return self.resolved_runtime_profile.id

    @property
    def resolved_deployment_target(self) -> str:
        return self.resolved_runtime_profile.deployment_target

    @property
    def resolved_inference_engine(self) -> str:
        return self.resolved_runtime_profile.inference_engine

    @property
    def resolved_capacity_class(self) -> str:
        configured = (self.capacity_class or "").strip()
        return configured or self.resolved_runtime_profile.capacity_class

    @property
    def resolved_routing_lane(self) -> str:
        return self.resolved_runtime_profile.routing_lane

    @property
    def resolved_routing_lane_label(self) -> str:
        return self.resolved_runtime_profile.routing_lane_label

    @property
    def resolved_routing_lane_detail(self) -> str:
        return self.resolved_runtime_profile.routing_lane_detail

    @property
    def resolved_routing_lane_policy_summary(self) -> str:
        return self.resolved_runtime_profile.routing_lane_policy_summary

    @property
    def resolved_routing_lane_allowed_privacy_tiers(self) -> tuple[str, ...]:
        return self.resolved_runtime_profile.routing_lane_allowed_privacy_tiers

    @property
    def resolved_routing_lane_allowed_result_guarantees(self) -> tuple[str, ...]:
        return self.resolved_runtime_profile.routing_lane_allowed_result_guarantees

    @property
    def resolved_routing_lane_allowed_trust_requirements(self) -> tuple[str, ...]:
        return self.resolved_runtime_profile.routing_lane_allowed_trust_requirements

    @property
    def resolved_max_privacy_tier(self) -> str:
        return self.resolved_runtime_profile.max_privacy_tier

    @property
    def resolved_exact_model_guarantee(self) -> bool:
        return self.resolved_runtime_profile.exact_model_guarantee

    @property
    def resolved_quantized_output_disclosure_required(self) -> bool:
        return self.resolved_runtime_profile.quantized_output_disclosure_required

    @property
    def resolved_inference_base_url(self) -> str:
        configured = (self.inference_base_url or "").strip()
        if configured:
            return configured
        legacy = (self.vllm_base_url or "").strip()
        if legacy:
            return legacy
        return default_inference_base_url(self.resolved_inference_engine)

    @property
    def resolved_runtime_image(self) -> str:
        configured = (self.runtime_image or "").strip()
        if configured:
            return configured
        legacy = (self.vllm_image or "").strip()
        if legacy:
            return legacy
        return self.resolved_runtime_profile.image

    @property
    def current_model(self) -> str:
        profile = self.resolved_runtime_profile
        model = (self.vllm_model or "").strip()
        if normalize_runtime_profile_id(self.runtime_profile) != AUTO_RUNTIME_PROFILE and model == DEFAULT_RESPONSE_MODEL:
            return profile.default_model
        return model or profile.default_model

    @property
    def current_model_source(self):
        return llama_cpp_model_source(self.current_model)

    @property
    def reports_audited_manifests(self) -> bool:
        return self.resolved_runtime_profile.reports_audited_manifests

    @property
    def supports_trusted_assignments(self) -> bool:
        return self.resolved_runtime_profile.supports_trusted_assignments

    @property
    def resolved_node_session_id(self) -> str:
        configured = (self.node_session_id or "").strip()
        return configured or self.boot_id


class AssignmentEnvelope(BaseModel):
    assignment_id: str
    execution_id: str
    node_session_id: str | None = None
    assignment_nonce: str
    operation: str
    model: str
    privacy_tier: str
    node_trust_requirement: str
    result_guarantee: str
    allowed_regions: list[str]
    required_vram_gb: float
    required_context_tokens: int
    token_budget: dict
    item_count: int
    expected_runtime_image_digest: str | None = None
    expected_model_manifest_digest: str | None = None
    expected_tokenizer_digest: str | None = None
    expected_chat_template_digest: str | None = None
    expected_gguf_file_digest: str | None = None
    expected_quantization_type: str | None = None
    expected_effective_context_tokens: int | None = None
    expected_runtime_tuple_digest: str | None = None
    microbatch_key: str | None = None
    input_artifact_url: str
    input_artifact_mirror_urls: list[str] = Field(default_factory=list)
    input_artifact_expires_at: str | None = None
    input_artifact_sha256: str
    input_artifact_encryption: dict


class NodeClaimSession(BaseModel):
    claim_id: str
    claim_code: str
    approval_url: str
    poll_token: str
    expires_at: str
    poll_interval_seconds: int


class NodeClaimPollResult(BaseModel):
    status: Literal["pending", "approved", "consumed", "expired"]
    expires_at: str
    node_id: str | None = None
    node_key: str | None = None
