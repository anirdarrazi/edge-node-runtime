from node_agent.model_artifacts import (
    find_model_artifact,
    load_model_artifacts_manifest,
    resolved_model_manifest_digest,
    resolved_tokenizer_digest,
)


class StubSettings:
    vllm_model = "meta-llama/Llama-3.1-8B-Instruct"
    model_manifest_digest = None
    tokenizer_digest = None


def test_bundled_model_artifacts_manifest_matches_the_signed_release_version() -> None:
    manifest = load_model_artifacts_manifest()

    assert manifest.version == "2026.04.10.1"
    assert len(manifest.artifacts) >= 2


def test_model_artifact_lookup_returns_expected_release_descriptors() -> None:
    artifact = find_model_artifact("meta-llama/Llama-3.1-8B-Instruct", "responses")

    assert artifact is not None
    assert artifact.revision == "0e9e39f249a16976918f6564b8830bc894c89659"
    assert artifact.model_manifest_digest.startswith("sha256:")
    assert artifact.tokenizer_digest.startswith("sha256:")
    assert any(file.path == "model.safetensors.index.json" for file in artifact.model_manifest.files)
    assert any(file.path == "tokenizer.json" for file in artifact.tokenizer_manifest.files)


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
