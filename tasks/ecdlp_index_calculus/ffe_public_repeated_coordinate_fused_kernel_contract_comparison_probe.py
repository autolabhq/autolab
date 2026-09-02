#!/usr/bin/env python3
"""Compare event-key and batch-lane fused-kernel contracts side by side.

The event-key contract is the conservative portable boundary: one first-pass
candidate-point operation per public event key.  The batch-lane contract is the
optimized boundary: first-pass candidate-point work is charged by validated
grouped-left-y native lanes.  This probe keeps those accounting models separate
and requires the batch-lane ABI guard before reporting an optimized below-rho
claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_EVENT_KEY_CONTRACT_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_contract_target67_672_744.json"
)
DEFAULT_LANE_CONTRACT_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_batch_lane_contract_target67_672_744.json"
)
DEFAULT_LANE_ABI_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_batch_lane_abi_target67_672_744.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_contract_comparison.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_ratio(ops: int, rho: int) -> float | None:
    return round(ops / rho, 8) if rho else None


def records_by_source(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("source_name") or ""): record
        for record in source.get("records") or []
        if isinstance(record, dict)
    }


def add_failure(failures: list[dict[str, Any]], code: str, detail: dict[str, Any] | None = None) -> None:
    failures.append({"code": code, **(detail or {})})


def claim_class(event_below: bool, lane_below: bool) -> str:
    if event_below and lane_below:
        return "event_key_and_batch_lane_below_rho"
    if lane_below:
        return "batch_lane_only_below_rho"
    if event_below:
        return "event_key_only_below_rho"
    return "not_below_rho"


def required_bool(checks: dict[str, Any], name: str) -> bool:
    return bool(checks.get(name))


def compare_record(
    source_name: str,
    event_record: dict[str, Any] | None,
    lane_record: dict[str, Any] | None,
    abi_record: dict[str, Any] | None,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    event_counters = event_record.get("work_counters") if isinstance(event_record, dict) else {}
    lane_counters = lane_record.get("lane_amortized_work_counters") if isinstance(lane_record, dict) else {}
    event_checks = event_record.get("checks") if isinstance(event_record, dict) else {}
    lane_checks = lane_record.get("checks") if isinstance(lane_record, dict) else {}
    abi_checks = abi_record.get("checks") if isinstance(abi_record, dict) else {}

    event_ok = bool(event_record and event_record.get("contract_status") == "fused_kernel_contract_verified")
    lane_ok = bool(
        lane_record and lane_record.get("batch_lane_contract_status") == "batch_lane_contract_verified"
    )
    abi_ok = bool(abi_record and abi_record.get("batch_lane_abi_status") == "batch_lane_abi_verified")

    event_ops = as_int(event_counters.get("fused_ops"))
    lane_ops = as_int(lane_counters.get("lane_fused_ops"))
    rho = as_int(event_counters.get("generic_rho_steps") or lane_counters.get("generic_rho_steps"))
    event_below = bool(event_counters.get("fused_below_rho"))
    lane_below = bool(lane_counters.get("lane_fused_below_rho"))
    saved_ops = max(0, event_ops - lane_ops)

    checks = {
        "event_key_contract_verified": event_ok,
        "batch_lane_contract_verified": lane_ok,
        "batch_lane_abi_verified": abi_ok,
        "event_key_ops_match_lane_original": event_ops == as_int(lane_counters.get("original_fused_ops")),
        "rho_steps_match": rho == as_int(lane_counters.get("generic_rho_steps")),
        "event_key_first_pass_ops_match": as_int(event_counters.get("shared_first_event_pass_ops"))
        == as_int(lane_counters.get("event_key_first_pass_ops")),
        "second_pass_ops_match": as_int(event_counters.get("second_event_pass_ops"))
        == as_int(lane_counters.get("second_event_pass_ops")),
        "lane_saved_ops_match": saved_ops == as_int(lane_counters.get("lane_total_saved_ops")),
        "source_contract_verified_by_lane": required_bool(lane_checks, "source_contract_verified"),
        "batch_lane_native_verified": required_bool(lane_checks, "batch_lane_verified"),
        "preserves_second_pass_fanout": required_bool(lane_checks, "preserves_second_pass_fanout")
        and required_bool(abi_checks, "preserves_second_pass_fanout"),
        "relation_replay_derived": required_bool(lane_checks, "relation_replay_derived")
        and required_bool(abi_checks, "relation_replay_derived"),
        "relation_replay_matches_secret": required_bool(lane_checks, "relation_replay_matches_secret")
        and required_bool(abi_checks, "relation_replay_matches_secret"),
        "lane_curve_guarded": required_bool(abi_checks, "all_lanes_on_curve"),
        "lane_subgroup_guarded": required_bool(abi_checks, "all_lanes_in_subgroup"),
        "lane_count_matches": required_bool(abi_checks, "batch_lane_count_matches"),
        "case_count_matches": required_bool(abi_checks, "case_count_matches"),
    }

    for name, passed in checks.items():
        if not passed:
            add_failure(failures, f"{name}_failed")

    conservative_claim_allowed = event_ok and event_below
    optimized_claim_allowed = (
        event_ok
        and lane_ok
        and abi_ok
        and lane_below
        and checks["preserves_second_pass_fanout"]
        and checks["relation_replay_derived"]
        and checks["relation_replay_matches_secret"]
        and checks["lane_curve_guarded"]
        and checks["lane_subgroup_guarded"]
    )

    return {
        "source_name": source_name,
        "public_group_key": (
            event_record.get("public_group_key")
            if isinstance(event_record, dict)
            else lane_record.get("public_group_key")
            if isinstance(lane_record, dict)
            else None
        ),
        "comparison_status": "contract_comparison_verified" if not failures else "contract_comparison_failed_check",
        "claim_class": claim_class(event_below, lane_below),
        "claim_permissions": {
            "conservative_event_key_below_rho_claim": conservative_claim_allowed,
            "optimized_batch_lane_below_rho_claim": optimized_claim_allowed,
            "optimized_claim_requires": [
                "batch_lane_contract_verified",
                "batch_lane_abi_verified",
                "lane_curve_guarded",
                "lane_subgroup_guarded",
                "preserved_second_pass_fanout",
                "relation_replay_matches_secret",
            ],
        },
        "work_counters": {
            "generic_rho_steps": rho,
            "event_key_fused_ops": event_ops,
            "event_key_fused_ops_over_rho": clean_ratio(event_ops, rho),
            "event_key_fused_below_rho": event_below,
            "event_key_first_pass_ops": as_int(lane_counters.get("event_key_first_pass_ops")),
            "batch_lane_fused_ops": lane_ops,
            "batch_lane_fused_ops_over_rho": clean_ratio(lane_ops, rho),
            "batch_lane_fused_below_rho": lane_below,
            "batch_lane_first_pass_ops": as_int(lane_counters.get("batch_lane_first_pass_ops")),
            "lane_first_pass_saved_ops": as_int(lane_counters.get("lane_first_pass_saved_ops")),
            "lane_total_saved_ops": saved_ops,
            "second_event_pass_ops": as_int(lane_counters.get("second_event_pass_ops")),
        },
        "checks": checks,
        "failures": failures,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [record for record in records if record.get("comparison_status") == "contract_comparison_verified"]
    event_below = [
        record for record in verified if bool((record.get("work_counters") or {}).get("event_key_fused_below_rho"))
    ]
    lane_below = [
        record for record in verified if bool((record.get("work_counters") or {}).get("batch_lane_fused_below_rho"))
    ]
    lane_only = [record for record in lane_below if record.get("claim_class") == "batch_lane_only_below_rho"]
    both = [record for record in lane_below if record.get("claim_class") == "event_key_and_batch_lane_below_rho"]
    neither = [record for record in verified if record.get("claim_class") == "not_below_rho"]
    optimized = [
        record
        for record in verified
        if bool((record.get("claim_permissions") or {}).get("optimized_batch_lane_below_rho_claim"))
    ]
    event_ratios = [
        (record.get("work_counters") or {}).get("event_key_fused_ops_over_rho")
        for record in event_below
        if (record.get("work_counters") or {}).get("event_key_fused_ops_over_rho") is not None
    ]
    lane_ratios = [
        (record.get("work_counters") or {}).get("batch_lane_fused_ops_over_rho")
        for record in lane_below
        if (record.get("work_counters") or {}).get("batch_lane_fused_ops_over_rho") is not None
    ]
    return {
        "record_count": len(records),
        "contract_comparison_verified_count": len(verified),
        "event_key_conservative_below_rho_count": len(event_below),
        "batch_lane_optimized_below_rho_count": len(lane_below),
        "batch_lane_abi_guarded_below_rho_count": len(optimized),
        "batch_lane_only_below_rho_count": len(lane_only),
        "event_key_and_batch_lane_below_rho_count": len(both),
        "not_below_rho_count": len(neither),
        "total_event_key_fused_ops": sum(
            as_int((record.get("work_counters") or {}).get("event_key_fused_ops")) for record in verified
        ),
        "total_batch_lane_fused_ops": sum(
            as_int((record.get("work_counters") or {}).get("batch_lane_fused_ops")) for record in verified
        ),
        "total_lane_saved_ops": sum(
            as_int((record.get("work_counters") or {}).get("lane_total_saved_ops")) for record in verified
        ),
        "mean_event_key_below_rho_ops_over_rho": (
            round(mean([float(value) for value in event_ratios]), 8) if event_ratios else None
        ),
        "mean_batch_lane_below_rho_ops_over_rho": (
            round(mean([float(value) for value in lane_ratios]), 8) if lane_ratios else None
        ),
        "lane_only_source_names": [str(record.get("source_name") or "") for record in lane_only],
        "interpretation": (
            "Event-key and batch-lane accounting are now compared without mixing "
            "claim boundaries.  Conservative below-rho claims require only the "
            "event-key contract; optimized below-rho claims additionally require "
            "the batch-lane contract and ABI/curve guard."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-key-contract-source", type=Path, default=DEFAULT_EVENT_KEY_CONTRACT_SOURCE)
    parser.add_argument("--lane-contract-source", type=Path, default=DEFAULT_LANE_CONTRACT_SOURCE)
    parser.add_argument("--lane-abi-source", type=Path, default=DEFAULT_LANE_ABI_SOURCE)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    event_source = load_json(args.event_key_contract_source)
    lane_source = load_json(args.lane_contract_source)
    abi_source = load_json(args.lane_abi_source)
    event_by_source = records_by_source(event_source)
    lane_by_source = records_by_source(lane_source)
    abi_by_source = records_by_source(abi_source)
    wanted = set(args.source_name or [])
    source_names = sorted(wanted or (set(event_by_source) | set(lane_by_source) | set(abi_by_source)))
    records = [
        compare_record(
            source_name,
            event_by_source.get(source_name),
            lane_by_source.get(source_name),
            abi_by_source.get(source_name),
        )
        for source_name in source_names
    ]

    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_kernel_contract_comparison_probe_v1",
        "method": "side_by_side_event_key_and_batch_lane_contract_accounting",
        "parameters": {
            "event_key_contract_source": str(args.event_key_contract_source),
            "lane_contract_source": str(args.lane_contract_source),
            "lane_abi_source": str(args.lane_abi_source),
            "source_names": sorted(wanted),
        },
        "summary": summarize(records),
        "records": sorted(
            records,
            key=lambda record: (
                record.get("comparison_status") != "contract_comparison_verified",
                str(record.get("claim_class") or ""),
                str(record.get("source_name") or ""),
            ),
        ),
        "accounting_boundaries": {
            "conservative_event_key_contract": (
                "One first-pass candidate-point operation is charged per public event key."
            ),
            "optimized_batch_lane_contract": (
                "First-pass candidate-point work is charged by validated grouped-left-y "
                "native lanes, and requires the batch-lane ABI/curve guard."
            ),
        },
        "non_claims": [
            "This is a comparison of existing verified contracts; it is not a new relation search.",
            "Lane-only below-rho records are optimized-implementation claims, not conservative event-key claims.",
            "No large-field or general ECDLP break is claimed by this artifact.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0 if output["summary"]["contract_comparison_verified_count"] == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
