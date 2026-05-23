from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Mapping

from .vast_smoke import (
    DEFAULT_DISK_GB,
    DEFAULT_DURABLE_RUNTIME_PROFILE,
    DEFAULT_MIN_CUDA_MAX_GOOD,
    DEFAULT_MIN_INET_DOWN_MBPS,
    DEFAULT_MIN_RELIABILITY,
    DEFAULT_MIN_VRAM_GB,
    DEFAULT_OFFER_LIMIT,
    DEFAULT_VAST_LAUNCH_PROFILE,
    DEFAULT_VAST_SMOKE_MODEL,
    VastAPI,
    VastSmokeConfig,
    VastSmokeError,
    VAST_REPORTED_VRAM_TOLERANCE_GB,
    _config_value,
    _float_value,
    _int_value,
    affordable_offers,
    default_vast_smoke_config_path,
    first_nonempty,
    load_vast_smoke_config,
    offer_machine_id_values,
    offer_supports_minimum_cuda,
    parse_identity_list,
    parse_int_list,
    runtime_policy_rejection_diagnostic,
    summarize_offer,
)


def select_fleet_offers(
    offers: list[dict[str, Any]],
    *,
    nodes: int,
    allow_same_machine: bool = False,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_offer_ids: set[int] = set()
    used_machine_ids: set[str] = set()
    for offer in offers:
        offer_id = _int_value(offer, "id")
        if offer_id <= 0 or offer_id in used_offer_ids:
            continue
        machine_ids = offer_machine_id_values(offer)
        if not allow_same_machine and machine_ids and not machine_ids.isdisjoint(used_machine_ids):
            continue
        selected.append(offer)
        used_offer_ids.add(offer_id)
        used_machine_ids.update(machine_ids)
        if len(selected) >= nodes:
            return selected
    return selected


def launch_args_for_offer(offer: Mapping[str, Any]) -> list[str]:
    return ["--preferred-offer-id", str(_int_value(dict(offer), "id"))]


def candidate_machine_count(offers: list[dict[str, Any]]) -> int | None:
    machine_ids: set[str] = set()
    missing_identity = False
    for offer in offers:
        ids = offer_machine_id_values(offer)
        if ids:
            machine_ids.update(ids)
        else:
            missing_identity = True
    if missing_identity:
        return None
    return len(machine_ids)


def offer_matches_non_runtime_quality_floor(offer: Mapping[str, Any], config: VastSmokeConfig) -> bool:
    offer_dict = dict(offer)
    if _float_value(offer_dict, "dph_total", default=10**9) > float(config.max_price):
        return False
    if not offer_supports_minimum_cuda(offer_dict, config.min_cuda_max_good):
        return False
    if (
        config.min_vram_gb is not None
        and float(config.min_vram_gb) > 0
        and (_float_value(offer_dict, "gpu_ram") / 1024.0) + VAST_REPORTED_VRAM_TOLERANCE_GB < float(config.min_vram_gb)
    ):
        return False
    if config.disk_gb is not None and int(config.disk_gb) > 0:
        if "disk_space" in offer_dict and _float_value(offer_dict, "disk_space") < int(config.disk_gb):
            return False
    if config.min_reliability is not None and float(config.min_reliability) > 0:
        reliability = _float_value(offer_dict, "reliability") or _float_value(offer_dict, "reliability2")
        if reliability < float(config.min_reliability):
            return False
    if (
        config.min_inet_down_mbps is not None
        and float(config.min_inet_down_mbps) > 0
        and _float_value(offer_dict, "inet_down") < float(config.min_inet_down_mbps)
    ):
        return False
    return True


def fleet_partial_note(
    *,
    requested_nodes: int,
    selected_count: int,
    candidate_count: int,
    unique_candidate_machines: int | None,
    allow_same_machine: bool,
) -> str:
    if candidate_count <= 0:
        return (
            "No eligible full-profile Vast offers were available for the requested fleet size. "
            "Inspect market_diagnostics.rejection_summary before relaxing any safety floor."
        )
    if candidate_count < requested_nodes:
        return (
            f"Only {candidate_count} eligible full-profile Vast offer(s) were available for "
            f"{requested_nodes} requested node(s). Wait for more full GPU supply, raise the price ceiling, "
            "or choose a different runtime profile."
        )
    if not allow_same_machine and unique_candidate_machines is not None and unique_candidate_machines < requested_nodes:
        return (
            f"{candidate_count} eligible offer(s) were available, but only {unique_candidate_machines} distinct "
            f"machine(s) could satisfy {requested_nodes} requested node(s). Use --allow-same-machine only when "
            "same-host correlated failure risk is acceptable."
        )
    return (
        f"Only {selected_count} offer(s) could be selected for {requested_nodes} requested node(s). "
        "Inspect selected_offers and market_diagnostics before launching."
    )


def market_diagnostics(
    offers: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    config: VastSmokeConfig,
    requested_nodes: int,
    allow_same_machine: bool,
) -> dict[str, Any]:
    non_runtime_floor_matches = [
        offer for offer in offers if offer_matches_non_runtime_quality_floor(offer, config)
    ]
    unique_candidate_machines = candidate_machine_count(candidates)
    return {
        "searched_offer_count": len(offers),
        "non_runtime_quality_floor_count": len(non_runtime_floor_matches),
        "eligible_candidate_count": len(candidates),
        "selected_count": len(selected),
        "requested_nodes": requested_nodes,
        "allow_same_machine": allow_same_machine,
        "unique_candidate_machine_count": unique_candidate_machines,
        "rejection_summary": runtime_policy_rejection_diagnostic(
            non_runtime_floor_matches,
            model=config.model,
            runtime_profile=config.runtime_profile,
        ).strip()
        or None,
    }


def build_config_from_args(args: argparse.Namespace) -> VastSmokeConfig:
    config_values, _config_path = load_vast_smoke_config(str(args.config or ""))
    api_key = first_nonempty(
        str(args.api_key or ""),
        os.getenv("VAST_API_KEY"),
        _config_value(config_values, "api_key", "vast_api_key"),
    )
    if not api_key:
        default_path = default_vast_smoke_config_path()
        raise VastSmokeError(
            "A Vast.ai API key is required. Pass --api-key, set VAST_API_KEY, "
            f"or add api_key to a local config file such as {default_path}."
        )
    return VastSmokeConfig(
        api_key=api_key,
        model=str(args.model or DEFAULT_VAST_SMOKE_MODEL).strip() or DEFAULT_VAST_SMOKE_MODEL,
        max_price=float(args.max_price),
        disk_gb=max(1, int(args.disk_gb)),
        min_vram_gb=max(1.0, float(args.min_vram_gb)),
        min_cuda_max_good=float(args.min_cuda_max_good) if args.min_cuda_max_good is not None else None,
        min_reliability=max(0.0, min(1.0, float(args.min_reliability))),
        min_inet_down_mbps=max(0.0, float(args.min_inet_down_mbps)),
        offer_limit=max(1, int(args.offer_limit)),
        runtime_profile=str(args.runtime_profile or DEFAULT_DURABLE_RUNTIME_PROFILE).strip()
        or DEFAULT_DURABLE_RUNTIME_PROFILE,
        exclude_offer_ids=parse_int_list(
            list(getattr(args, "exclude_offer_id", []) or ()),
            csv_value=str(getattr(args, "exclude_offer_ids", "") or ""),
        ),
        exclude_machine_ids=parse_identity_list(
            list(getattr(args, "exclude_machine_id", []) or ()),
            csv_value=str(getattr(args, "exclude_machine_ids", "") or ""),
        ),
    )


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    config = build_config_from_args(args)
    api = VastAPI(config.api_key)
    offers = api.search_offers(config)
    candidates = affordable_offers(
        offers,
        max_price=config.max_price,
        min_cuda_max_good=config.min_cuda_max_good,
        min_vram_gb=config.min_vram_gb,
        disk_gb=config.disk_gb,
        min_reliability=config.min_reliability,
        min_inet_down_mbps=config.min_inet_down_mbps,
        model=config.model,
        runtime_profile=config.runtime_profile,
        exclude_offer_ids=config.exclude_offer_ids,
        exclude_machine_ids=config.exclude_machine_ids,
    )
    selected = select_fleet_offers(
        candidates,
        nodes=max(1, int(args.nodes)),
        allow_same_machine=bool(args.allow_same_machine),
    )
    requested_nodes = max(1, int(args.nodes))
    status = "ok" if len(selected) >= requested_nodes else "partial"
    diagnostics = market_diagnostics(
        offers,
        candidates,
        selected,
        config=config,
        requested_nodes=requested_nodes,
        allow_same_machine=bool(args.allow_same_machine),
    )
    return {
        "status": status,
        "requested": {
            "nodes": requested_nodes,
            "model": config.model,
            "max_price": round(float(config.max_price), 6),
            "disk_gb": config.disk_gb,
            "min_vram_gb": config.min_vram_gb,
            "min_cuda_max_good": config.min_cuda_max_good,
            "min_reliability": config.min_reliability,
            "min_inet_down_mbps": config.min_inet_down_mbps,
            "runtime_profile": config.runtime_profile,
            "offer_limit": config.offer_limit,
            "exclude_offer_ids": list(config.exclude_offer_ids),
            "exclude_machine_ids": list(config.exclude_machine_ids),
            "allow_same_machine": bool(args.allow_same_machine),
        },
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_offers": [
            {
                "node_index": index,
                "offer": summarize_offer(offer),
                "launch_args": launch_args_for_offer(offer),
            }
            for index, offer in enumerate(selected, start=1)
        ],
        "notes": (
            []
            if len(selected) >= requested_nodes
            else [
                fleet_partial_note(
                    requested_nodes=requested_nodes,
                    selected_count=len(selected),
                    candidate_count=len(candidates),
                    unique_candidate_machines=diagnostics["unique_candidate_machine_count"],
                    allow_same_machine=bool(args.allow_same_machine),
                )
            ]
        ),
        "market_diagnostics": diagnostics,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan distinct Vast.ai offers for an AUTONOMOUSc fleet smoke test.")
    parser.add_argument("--api-key", default="", help="Temporary Vast.ai API key. Prefer VAST_API_KEY.")
    parser.add_argument(
        "--config",
        default="",
        help=(
            "Optional local JSON config file for secrets such as api_key. "
            f"Defaults to {default_vast_smoke_config_path()} when that file already exists."
        ),
    )
    parser.add_argument("--nodes", type=int, default=1, help="Number of fleet nodes to plan.")
    parser.add_argument("--model", default=DEFAULT_VAST_SMOKE_MODEL, help="Model the fleet will serve.")
    parser.add_argument("--max-price", type=float, default=DEFAULT_VAST_LAUNCH_PROFILE.safe_price_ceiling_usd, help="Maximum hourly price in USD.")
    parser.add_argument("--disk-gb", type=int, default=DEFAULT_DISK_GB, help="Minimum disk size in GB.")
    parser.add_argument("--min-vram-gb", type=float, default=DEFAULT_MIN_VRAM_GB, help="Minimum GPU VRAM in GB.")
    parser.add_argument("--min-cuda-max-good", type=float, default=DEFAULT_MIN_CUDA_MAX_GOOD, help="Minimum cuda_max_good host capability.")
    parser.add_argument("--min-reliability", type=float, default=DEFAULT_MIN_RELIABILITY, help="Minimum host reliability score.")
    parser.add_argument("--min-inet-down-mbps", type=float, default=DEFAULT_MIN_INET_DOWN_MBPS, help="Minimum internet download speed in Mbps.")
    parser.add_argument("--runtime-profile", default=DEFAULT_DURABLE_RUNTIME_PROFILE, help="Runtime profile the planned nodes will advertise.")
    parser.add_argument("--offer-limit", type=int, default=DEFAULT_OFFER_LIMIT, help="Maximum number of Vast offers to inspect.")
    parser.add_argument(
        "--exclude-offer-id",
        action="append",
        type=int,
        default=[],
        help="Skip a Vast offer ID. May be passed more than once.",
    )
    parser.add_argument(
        "--exclude-offer-ids",
        default="",
        help="Comma-separated Vast offer IDs to skip.",
    )
    parser.add_argument(
        "--exclude-machine-id",
        action="append",
        default=[],
        help="Skip a Vast machine/host identity when the offer payload exposes one. May be passed more than once.",
    )
    parser.add_argument(
        "--exclude-machine-ids",
        default="",
        help="Comma-separated Vast machine/host identities to skip when the offer payload exposes them.",
    )
    parser.add_argument("--allow-same-machine", action="store_true", help="Allow multiple selected offers from the same machine identity.")
    parser.add_argument("--json-indent", type=int, default=2, help="JSON indentation level for the plan.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(args)
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error) or error.__class__.__name__}, indent=max(0, int(args.json_indent))))
        return 1
    print(json.dumps(plan, indent=max(0, int(args.json_indent)), ensure_ascii=False))
    return 0 if plan["status"] == "ok" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
