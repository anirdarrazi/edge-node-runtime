from node_agent.runtime_backend import MANAGER_RUNTIME_BACKEND, SINGLE_CONTAINER_RUNTIME_BACKEND
from node_agent.runtime_profiles import (
    ARTIFACT_MANIFEST_AUDITED_SAFETENSORS,
    ARTIFACT_MANIFEST_GGUF,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GEMMA_4_E4B_MODEL,
    DEFAULT_PUBLIC_SMOKE_TEST_API_PATH,
    HOME_EMBEDDINGS_LLAMA_CPP_PROFILE,
    HOME_LLAMA_CPP_GGUF_PROFILE,
    PARTNER_VLLM_TRUSTED_PROFILE,
    RTX_5060_TI_16GB_GEMMA4_E4B_PROFILE,
    VAST_VLLM_SAFETENSORS_PROFILE,
    default_public_smoke_test_model,
    resolve_runtime_profile,
)


def test_home_llama_profile_declares_llama_cpp_gguf_policy() -> None:
    profile = resolve_runtime_profile(
        HOME_LLAMA_CPP_GGUF_PROFILE,
        configured_engine="vllm",
        configured_deployment_target="vast_ai",
        runtime_backend=SINGLE_CONTAINER_RUNTIME_BACKEND,
    )

    assert profile.id == HOME_LLAMA_CPP_GGUF_PROFILE
    assert profile.inference_engine == "llama_cpp"
    assert profile.model_format == "gguf"
    assert profile.readiness_path == "/health"
    assert profile.smoke_test_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert profile.smoke_test_api_path == "/v1/responses"
    assert profile.artifact_manifest_type == ARTIFACT_MANIFEST_GGUF
    assert profile.trust_policy == "community_best_effort"
    assert profile.routing_lane == "community_quantized_home"
    assert profile.routing_lane_label == "Community quantized home"
    assert profile.routing_lane_allowed_privacy_tiers == ("standard",)
    assert profile.routing_lane_allowed_result_guarantees == ("community_best_effort",)
    assert profile.routing_lane_allowed_trust_requirements == ("untrusted_allowed",)
    assert profile.max_privacy_tier == "standard"
    assert profile.exact_model_guarantee is False
    assert profile.quantized_output_disclosure_required is True
    assert "Allowed privacy tiers: standard." in profile.routing_lane_policy_summary
    assert "Allowed result guarantees: community best-effort results." in profile.routing_lane_policy_summary


def test_auto_home_embedding_model_resolves_embedding_profile() -> None:
    profile = resolve_runtime_profile(
        "auto",
        configured_engine="llama_cpp",
        configured_deployment_target="home_edge",
        runtime_backend=MANAGER_RUNTIME_BACKEND,
        model=DEFAULT_EMBEDDING_MODEL,
    )

    assert profile.id == HOME_EMBEDDINGS_LLAMA_CPP_PROFILE
    assert profile.supported_apis == ("embeddings",)
    assert profile.smoke_test_model == default_public_smoke_test_model()
    assert profile.smoke_test_api_path == DEFAULT_PUBLIC_SMOKE_TEST_API_PATH


def test_vast_and_partner_profiles_split_elastic_from_trusted_vllm() -> None:
    vast = resolve_runtime_profile(
        "auto",
        configured_engine=None,
        configured_deployment_target=None,
        runtime_backend=SINGLE_CONTAINER_RUNTIME_BACKEND,
    )
    partner = resolve_runtime_profile(
        PARTNER_VLLM_TRUSTED_PROFILE,
        configured_engine=None,
        configured_deployment_target=None,
        runtime_backend=MANAGER_RUNTIME_BACKEND,
    )

    assert vast.id == VAST_VLLM_SAFETENSORS_PROFILE
    assert vast.artifact_manifest_type == ARTIFACT_MANIFEST_AUDITED_SAFETENSORS
    assert vast.trust_policy == "elastic_untrusted"
    assert vast.capacity_class == "elastic_burst"
    assert vast.routing_lane == "elastic_exact_vast"
    assert vast.trusted_eligibility == "runtime_and_model_digest_match"
    assert vast.burst_lifecycle == (
        "provision",
        "warm_model",
        "register_temporary_node",
        "accept_burst_work",
        "drain",
        "terminate",
    )
    assert vast.burst_cost_ceiling_usd == 0.25
    assert vast.vast_launch is not None
    assert vast.vast_launch.runtype == "args"
    assert vast.vast_launch.min_disk_gb == 80
    assert vast.vast_launch.preferred_smoke_test_model == DEFAULT_EMBEDDING_MODEL
    assert vast.vast_launch.smoke_test_api_path == "/v1/embeddings"
    assert vast.vast_launch.safe_price_ceiling_usd == 0.25
    assert vast.smoke_test_model == default_public_smoke_test_model()
    assert vast.smoke_test_api_path == DEFAULT_PUBLIC_SMOKE_TEST_API_PATH
    assert vast.payload()["routing_lane_label"] == "Elastic exact Vast.ai"
    assert vast.payload()["routing_lane_allowed_privacy_tiers"] == ["standard", "confidential", "restricted"]
    assert vast.payload()["routing_lane_allowed_result_guarantees"] == [
        "community_best_effort",
        "exact_model_audited",
    ]
    assert vast.payload()["routing_lane_allowed_trust_requirements"] == [
        "untrusted_allowed",
        "trusted_only",
    ]
    assert vast.payload()["max_privacy_tier"] == "restricted"
    assert vast.payload()["exact_model_guarantee"] is True
    assert vast.payload()["quantized_output_disclosure_required"] is False
    assert vast.payload()["smoke_test_model"] == DEFAULT_EMBEDDING_MODEL
    assert vast.payload()["smoke_test_api_path"] == "/v1/embeddings"
    assert vast.payload()["burst_cost_ceiling_usd"] == 0.25
    assert vast.payload()["vast_launch"] == {
        "runtype": "args",
        "min_disk_gb": 80,
        "preferred_smoke_test_model": DEFAULT_EMBEDDING_MODEL,
        "smoke_test_api_path": "/v1/embeddings",
        "safe_price_ceiling_usd": 0.25,
    }
    assert partner.supports_trusted_assignments is True
    assert partner.routing_lane == "trusted_exact_partner"


def test_rtx_5060_ti_gemma_profile_targets_vast_responses() -> None:
    profile = resolve_runtime_profile(
        RTX_5060_TI_16GB_GEMMA4_E4B_PROFILE,
        configured_engine=None,
        configured_deployment_target=None,
        runtime_backend=SINGLE_CONTAINER_RUNTIME_BACKEND,
    )

    assert profile.id == RTX_5060_TI_16GB_GEMMA4_E4B_PROFILE
    assert profile.inference_engine == "vllm"
    assert profile.deployment_target == "vast_ai"
    assert profile.model_format == "safetensors"
    assert profile.default_model == DEFAULT_GEMMA_4_E4B_MODEL
    assert profile.supported_models == (DEFAULT_GEMMA_4_E4B_MODEL,)
    assert profile.supported_apis == ("responses",)
    assert profile.smoke_test_model == DEFAULT_GEMMA_4_E4B_MODEL
    assert profile.smoke_test_api_path == "/v1/responses"
    assert profile.vast_launch is not None
    assert profile.vast_launch.min_disk_gb == 80
    assert profile.vast_launch.preferred_smoke_test_model == DEFAULT_GEMMA_4_E4B_MODEL
    assert profile.vast_launch.smoke_test_api_path == "/v1/responses"


def test_auto_vast_gemma_model_resolves_5060_ti_profile() -> None:
    profile = resolve_runtime_profile(
        "auto",
        configured_engine="vllm",
        configured_deployment_target="vast_ai",
        runtime_backend=SINGLE_CONTAINER_RUNTIME_BACKEND,
        model=DEFAULT_GEMMA_4_E4B_MODEL,
    )

    assert profile.id == RTX_5060_TI_16GB_GEMMA4_E4B_PROFILE
