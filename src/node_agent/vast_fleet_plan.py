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
    _config_value,
    _int_value,
    affordable_offers,
    default_vast_smoke_config_path,
    first_nonempty,
    load_vast_smoke_config,
    offer_machine_id_values,
    parse_identity_list,
    parse_int_list,
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
                (
                    "Not enough distinct eligible Vast offers were available for the requested fleet size. "
                    "Rerun with --offer-limit higher, --max-price higher, or --allow-same-machine if sharing a host is intentional."
                )
            ]
        ),
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
