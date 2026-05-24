from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from node_agent import live_vast_batchrouter_drill as drill


def minimal_config(tmp_path: Path) -> drill.DrillConfig:
    return drill.DrillConfig(
        vast_api_key="vast-secret",
        batchrouter_api_key="br-secret",
        operator_token="autc_live_secret",
        edge_control_cwd=tmp_path,
        artifact_dir=tmp_path / "artifacts",
    )


def test_sql_quote_escapes_single_quotes() -> None:
    assert drill.sql_quote("batch_'_1") == "'batch_''_1'"


def test_command_for_platform_uses_npx_cmd_on_windows() -> None:
    command = drill.command_for_platform(["npx", "wrangler", "--version"], platform_name="nt")
    assert command == ["npx.cmd", "wrangler", "--version"]


def test_command_for_platform_uses_direct_command_on_posix() -> None:
    assert drill.command_for_platform(["npx", "wrangler"], platform_name="posix") == ["npx", "wrangler"]


def test_extract_d1_rows_sums_rows_read() -> None:
    rows, rows_read = drill.extract_d1_rows(
        [
            {"results": [{"node_id": "node_a"}], "meta": {"rows_read": 2}},
            {"results": [{"node_id": "node_b"}], "meta": {"rows_read": 3}},
        ]
    )
    assert rows == [{"node_id": "node_a"}, {"node_id": "node_b"}]
    assert rows_read == 5


def test_parse_json_output_skips_wrangler_banner() -> None:
    payload = drill.parse_json_output("banner\n[{\"results\":[{\"ok\":true}],\"meta\":{\"rows_read\":1}}]")
    assert payload[0]["results"][0]["ok"] is True


def test_build_batch_manifest_pins_autonomousc_provider(tmp_path: Path) -> None:
    config = minimal_config(tmp_path)
    manifest = drill.build_batch_manifest(config, run_id="run_123")
    assert manifest["provider_preferences"] == {
        "only": ["autonomousc"],
        "allow_fallbacks": False,
        "data_collection": "deny",
        "zdr": True,
    }
    assert len(manifest["items"]) == 500
    assert {item["model"] for item in manifest["items"]} == {"gemma-4-e4b-it"}
    assert manifest["items"][0]["input"]["max_output_tokens"] == 64
    assert manifest["max_price"] == "0.0500"


def test_build_node_enroll_payload_uses_static_drill_node_shape(tmp_path: Path) -> None:
    config = minimal_config(tmp_path)
    payload = drill.build_node_enroll_payload(config, label="node-label")
    assert payload["label"] == "node-label"
    assert payload["capabilities"]["batchrouter_capacity_tier"] == "edge"
    assert payload["capabilities"]["max_batch_items"] == 250
    assert payload["runtime"]["deployment_target"] == "vast_ai"
    assert payload["runtime"]["docker_image"].endswith(":single-cuda-latest")


def test_machine_ids_from_offer_report_reads_summarized_machine_ids() -> None:
    assert drill.machine_ids_from_offer_report({"machine_ids": ["49732", "71705"]}) == (
        "49732",
        "71705",
    )
    assert drill.machine_ids_from_offer_report({"machine_id": "host-a", "machine_ids": ["host-b"]}) == (
        "host-a",
        "host-b",
    )


def test_choose_victim_prefers_accepted_assignment(tmp_path: Path) -> None:
    node_a = drill.DrillNode(node_id="node_a", node_key="key_a")
    node_b = drill.DrillNode(node_id="node_b", node_key="key_b")
    rows = [
        {"node_id": "node_a", "status": "assigned", "attempts": 1},
        {"node_id": "node_b", "status": "accepted", "attempts": 1},
    ]
    assert (
        drill.choose_victim_node(rows, [node_a, node_b], require_accepted_assignment=False).node_id
        == "node_b"
    )
    assert (
        drill.choose_victim_node(rows, [node_a, node_b], require_accepted_assignment=True).node_id
        == "node_b"
    )


def test_choose_victim_respects_accepted_requirement(tmp_path: Path) -> None:
    node = drill.DrillNode(node_id="node_a", node_key="key_a")
    rows = [{"node_id": "node_a", "status": "assigned", "attempts": 1}]
    assert drill.choose_victim_node(rows, [node], require_accepted_assignment=True) is None
    assert drill.choose_victim_node(rows, [node], require_accepted_assignment=False) is node


def test_wrangler_d1_query_uses_json_remote_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps([{"results": [{"node_id": "node_1"}], "meta": {"rows_read": 7}}]),
            stderr="",
        )

    monkeypatch.setattr(drill.subprocess, "run", fake_run)
    client = drill.WranglerD1Client(
        cwd=tmp_path,
        database="autonomousc-edge-network-executions-db",
        platform_name="posix",
    )
    result = client.assignment_summary("batch_123")

    assert result.rows == [{"node_id": "node_1"}]
    assert result.rows_read == 7
    command = calls[0]["command"]
    assert command[:5] == ["npx", "wrangler", "d1", "execute", "autonomousc-edge-network-executions-db"]
    assert "--remote" in command
    assert "--json" in command
    assert "batch_123" in command[-1]
    assert "\n" not in command[-1]
    assert command[-1].endswith(";")


def test_build_config_parses_offer_controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VAST_API_KEY", "vast-secret")
    monkeypatch.setenv("BATCHROUTER_API_KEY", "br-secret")
    monkeypatch.setenv("AUTONOMOUSC_OPERATOR_API_KEY", "autc_live_secret")
    args = drill.parse_args(
        [
            "--edge-control-cwd",
            str(tmp_path),
            "--preferred-offer-id",
            "123",
            "--preferred-offer-id",
            "456",
            "--exclude-offer-id",
            "999",
            "--exclude-offer-ids",
            "1001,1002",
            "--exclude-machine-id",
            "host-a",
            "--exclude-machine-ids",
            "host-b,host-c",
            "--launch-timeout-seconds",
            "180",
            "--launch-progress-grace-seconds",
            "90",
            "--startup-progress-stale-seconds",
            "300",
            "--startup-max-vllm-restarts",
            "1",
        ]
    )

    config = drill.build_config_from_args(args)

    assert config.preferred_offer_ids == (123, 456)
    assert config.exclude_offer_ids == (999, 1001, 1002)
    assert config.exclude_machine_ids == ("host-a", "host-b", "host-c")
    assert config.launch_timeout_seconds == 180
    assert config.launch_progress_grace_seconds == 90
    assert config.startup_progress_stale_seconds == 300
    assert config.startup_max_vllm_restarts == 1
