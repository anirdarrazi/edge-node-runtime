from node_agent.gguf_artifacts import (
    find_gguf_artifact,
    load_gguf_artifacts_manifest,
    resolved_gguf_artifact_contract,
)
from node_agent.model_artifacts import (
    find_model_artifact,
    load_model_artifacts_manifest,
    resolved_chat_template_digest,
    resolved_model_manifest_digest,
    resolved_tokenizer_digest,
)


class StubSettings:
    runtime_profile = "partner_vllm_trusted"
    resolved_runtime_profile_id = "partner_vllm_trusted"
    resolved_inference_engine = "vllm"
    vllm_model = "meta-llama/Llama-3.1-8B-Instruct"
    model_manifest_digest = None
    tokenizer_digest = None


class GgufStubSettings:
    runtime_profile = "home_llama_cpp_gguf"
    resolved_runtime_profile_id = "home_llama_cpp_gguf"
    resolved_inference_engine = "llama_cpp"
    current_model = "meta-llama/Llama-3.1-8B-Instruct"
    vllm_model = "meta-llama/Llama-3.1-8B-Instruct"
    model_manifest_digest = None
    tokenizer_digest = None


def test_bundled_model_artifacts_manifest_matches_the_signed_release_version() -> None:
    manifest = load_model_artifacts_manifest()

    assert manifest.version == "2026.04.10.1"
    assert len(manifest.artifacts) >= 2


def test_bundled_gguf_artifacts_manifest_is_separate_from_safetensors_manifests() -> None:
    manifest = load_gguf_artifacts_manifest()

    assert manifest.version == "2026.04.10.1"
    assert len(manifest.artifacts) >= 2
    assert all(artifact.artifact_manifest_type == "gguf_hf_file" for artifact in manifest.artifacts)
    assert all(artifact.runtime_engine == "llama_cpp" for artifact in manifest.artifacts)


def test_model_artifact_lookup_returns_expected_release_descriptors() -> None:
    artifact = find_model_artifact("meta-llama/Llama-3.1-8B-Instruct", "responses")

    assert artifact is not None
    assert artifact.revision == "0e9e39f249a16976918f6564b8830bc894c89659"
    assert artifact.model_manifest_digest.startswith("sha256:")
    assert artifact.tokenizer_digest.startswith("sha256:")
    assert any(file.path == "model.safetensors.index.json" for file in artifact.model_manifest.files)
    assert any(file.path == "tokenizer.json" for file in artifact.tokenizer_manifest.files)


def test_gemma_e4b_model_artifact_is_bundled_for_vllm_responses() -> None:
    artifact = find_model_artifact("google/gemma-4-E4B-it", "responses", runtime_engine="vllm")

    assert artifact is not None
    assert artifact.revision == "c53e9d33178b12afbad4a48334d21e19b8c29761"
    assert artifact.model_manifest_digest.startswith("sha256:")
    assert artifact.tokenizer_digest.startswith("sha256:")
    assert any(file.path == "model.safetensors" for file in artifact.model_manifest.files)
    assert any(file.path == "chat_template.jinja" for file in artifact.tokenizer_manifest.files)


def test_runtime_metadata_resolution_prefers_bundled_release_manifest() -> None:
    settings = StubSettings()
    embeddings_artifact = find_model_artifact("BAAI/bge-large-en-v1.5", "embeddings")

    assert embeddings_artifact is not None
    assert (
        resolved_model_manifest_digest(settings, "BAAI/bge-large-en-v1.5", "embeddings")
        == embeddings_artifact.model_manifest_digest
    )
    assert (
        resolved_tokenizer_digest(settings, "BAAI/bge-large-en-v1.5", "embeddings")
        == embeddings_artifact.tokenizer_digest
    )


def test_chat_template_digest_is_derived_for_responses_only() -> None:
    settings = StubSettings()

    chat_digest = resolved_chat_template_digest(settings, "meta-llama/Llama-3.1-8B-Instruct", "responses")

    assert isinstance(chat_digest, str)
    assert chat_digest.startswith("sha256:")
    assert resolved_chat_template_digest(settings, "BAAI/bge-large-en-v1.5", "embeddings") is None


def test_gguf_artifact_contract_declares_quantized_file_metadata() -> None:
    artifact = find_gguf_artifact("meta-llama/Llama-3.1-8B-Instruct", "responses")

    assert artifact is not None
    assert artifact.repository == "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    assert artifact.filename == "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    assert artifact.quantization_type == "Q4_K_M"
    assert artifact.file_digest == "sha256:7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c"
    assert artifact.expected_context_tokens == 32768
    assert artifact.capabilities.chat is True
    assert artifact.capabilities.embeddings is False


def test_gguf_artifact_resolution_only_applies_to_llama_cpp_profiles() -> None:
    assert resolved_gguf_artifact_contract(StubSettings(), "meta-llama/Llama-3.1-8B-Instruct", "responses") is None

    artifact = resolved_gguf_artifact_contract(
        GgufStubSettings(),
        "meta-llama/Llama-3.1-8B-Instruct",
        "responses",
    )

    assert artifact is not None
    assert artifact.artifact_manifest_type == "gguf_hf_file"
