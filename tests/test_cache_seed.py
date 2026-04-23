from pathlib import Path

from node_agent.cache_seed import parse_preload_models, preload_hf_cache


def test_parse_preload_models_splits_commas_newlines_and_deduplicates() -> None:
    assert parse_preload_models("BAAI/bge-large-en-v1.5,\nQwen/Qwen-Image, BAAI/bge-large-en-v1.5") == (
        "BAAI/bge-large-en-v1.5",
        "Qwen/Qwen-Image",
    )


def test_preload_hf_cache_skips_when_no_models_are_configured(tmp_path: Path) -> None:
    logs: list[str] = []
    calls: list[dict[str, str]] = []

    result = preload_hf_cache(
        (),
        tmp_path / "hf-cache",
        log=logs.append,
        snapshot_download=lambda **kwargs: calls.append(dict(kwargs)),
    )

    assert result == ()
    assert calls == []
    assert logs == ["Skipping Hugging Face cache preload because no starter models were configured."]


def test_preload_hf_cache_downloads_each_model_once(tmp_path: Path) -> None:
    logs: list[str] = []
    calls: list[dict[str, str]] = []

    result = preload_hf_cache(
        ("BAAI/bge-large-en-v1.5", "Qwen/Qwen-Image"),
        tmp_path / "hf-cache",
        log=logs.append,
        snapshot_download=lambda **kwargs: calls.append(dict(kwargs)),
    )

    assert result == ("BAAI/bge-large-en-v1.5", "Qwen/Qwen-Image")
    assert calls == [
        {"repo_id": "BAAI/bge-large-en-v1.5", "cache_dir": str(tmp_path / "hf-cache")},
        {"repo_id": "Qwen/Qwen-Image", "cache_dir": str(tmp_path / "hf-cache")},
    ]
    assert logs[-1] == f"Preloaded 2 starter model(s) into {tmp_path / 'hf-cache'}."
