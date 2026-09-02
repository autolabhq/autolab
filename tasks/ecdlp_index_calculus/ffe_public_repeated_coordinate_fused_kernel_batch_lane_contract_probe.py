#!/usr/bin/env python3
"""Verify lane-amortized contracts for fused repeated-coordinate kernels.

The portable fused-kernel contract charges one first-pass candidate-point
operation per public event key.  The native batch-lane kernel proves a lower
implementation boundary: within a shared-denominator group, all event keys with
the same left y-coordinate share the same candidate point.  This probe promotes
that native result back into the contract/cost layer.

It preserves the original second-pass row checks, accepted relation summaries,
and modular relation derivation.  Only the first-pass candidate-point charge is
replaced by the validated batch-lane count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_CONTRACT_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_contract_target67_672_744.json"
)
DEFAULT_BATCH_LANE_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_batch_lane_target67_672_744.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_batch_lane_contract.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_ratio(ops: int, rho: int) -> float | None:
    return round(ops / rho, 8) if rho else None


def batch_records_by_source(batch_source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("source_name") or ""): record
        for record in batch_source.get("records") or []
        if isinstance(record, dict)
    }


def lane_counter_record(contract_record: dict[str, Any], batch_record: dict[str, Any] | None) -> dict[str, Any]:
    source_name = str(contract_record.get("source_name") or "")
    original = contract_record.get("work_counters") or {}
    checks = contract_record.get("checks") or {}
    kernel_contract = contract_record.get("kernel_contract") or {}
    relation_replay = contract_record.get("relation_form_replay") or {}
    first_pass = kernel_contract.get("first_pass_candidate_point_groups") or []
    second_pass = kernel_contract.get("second_pass_row_checks") or []
    summaries = kernel_contract.get("accepted_relation_summaries") or []
    failures = []

    if batch_record is None:
        failures.append({"code": "missing_batch_lane_record", "source_name": source_name})
        lane_count = 0
        batch_case_count = 0
    else:
        lane_count = as_int(batch_record.get("lane_count"))
        batch_case_count = as_int(batch_record.get("case_count"))
        if batch_record.get("batch_lane_status") != "batch_lane_native_kernel_verified":
            failures.append(
                {
                    "code": "batch_lane_record_not_verified",
                    "batch_lane_status": batch_record.get("batch_lane_status"),
                }
            )

    event_key_first_pass_ops = sum(as_int(group.get("candidate_point_ops"), 0) for group in first_pass)
    first_pass_group_count = len(first_pass)
    first_pass_group_instance_count = sum(as_int(group.get("fanout"), 0) for group in first_pass)
    second_pass_check_count = len(second_pass)
    accepted_relation_count = sum(1 for check in second_pass if bool(check.get("accepted_relation")))
    rho = as_int(original.get("generic_rho_steps"))
    lane_first_pass_ops = lane_count
    lane_first_pass_saved_ops = max(0, event_key_first_pass_ops - lane_first_pass_ops)
    lane_fused_ops = (
        as_int(original.get("nonshared_base_ops"))
        + as_int(original.get("shared_leaf_ops"))
        + as_int(original.get("shared_hit_root_ops"))
        + lane_first_pass_ops
        + as_int(original.get("second_event_pass_ops"))
    )

    if batch_case_count != first_pass_group_count:
        failures.append(
            {
                "code": "batch_case_count_mismatch",
                "batch_case_count": batch_case_count,
                "first_pass_group_count": first_pass_group_count,
            }
        )
    if as_int(checks.get("first_pass_group_count"), -1) != first_pass_group_count:
        failures.append(
            {
                "code": "contract_first_pass_group_count_mismatch",
                "recorded": checks.get("first_pass_group_count"),
                "observed": first_pass_group_count,
            }
        )
    if as_int(checks.get("first_pass_group_instance_count"), -1) != second_pass_check_count:
        failures.append(
            {
                "code": "contract_fanout_mismatch",
                "first_pass_group_instance_count": checks.get("first_pass_group_instance_count"),
                "second_pass_check_count": second_pass_check_count,
            }
        )
    if first_pass_group_instance_count != second_pass_check_count:
        failures.append(
            {
                "code": "observed_fanout_mismatch",
                "first_pass_group_instance_count": first_pass_group_instance_count,
                "second_pass_check_count": second_pass_check_count,
            }
        )
    if accepted_relation_count != len(summaries):
        failures.append(
            {
                "code": "accepted_summary_count_mismatch",
                "accepted_relation_count": accepted_relation_count,
                "summary_count": len(summaries),
            }
        )
    if not bool(relation_replay.get("matches_recorded_secret")):
        failures.append({"code": "relation_replay_secret_mismatch"})
    if not bool(relation_replay.get("derived")):
        failures.append({"code": "relation_replay_not_derived"})
    if contract_record.get("contract_status") != "fused_kernel_contract_verified":
        failures.append(
            {"code": "source_contract_not_verified", "contract_status": contract_record.get("contract_status")}
        )

    return {
        "source_name": source_name,
        "public_group_key": contract_record.get("public_group_key"),
        "batch_lane_contract_status": (
            "batch_lane_contract_verified" if not failures else "batch_lane_contract_failed_check"
        ),
        "relation_form_replay": relation_replay,
        "lane_amortized_work_counters": {
            "nonshared_base_ops": as_int(original.get("nonshared_base_ops")),
            "shared_leaf_ops": as_int(original.get("shared_leaf_ops")),
            "shared_hit_root_ops": as_int(original.get("shared_hit_root_ops")),
            "event_key_first_pass_ops": event_key_first_pass_ops,
            "batch_lane_first_pass_ops": lane_first_pass_ops,
            "second_event_pass_ops": as_int(original.get("second_event_pass_ops")),
            "original_fused_ops": as_int(original.get("fused_ops")),
            "lane_fused_ops": lane_fused_ops,
            "generic_rho_steps": rho,
            "original_fused_ops_over_rho": original.get("fused_ops_over_rho"),
            "lane_fused_ops_over_rho": clean_ratio(lane_fused_ops, rho),
            "lane_fused_below_rho": bool(rho and lane_fused_ops < rho),
            "lane_first_pass_saved_ops": lane_first_pass_saved_ops,
            "lane_total_saved_ops": max(0, as_int(original.get("fused_ops")) - lane_fused_ops),
        },
        "checks": {
            "source_contract_verified": contract_record.get("contract_status") == "fused_kernel_contract_verified",
            "batch_lane_verified": bool(
                batch_record and batch_record.get("batch_lane_status") == "batch_lane_native_kernel_verified"
            ),
            "first_pass_group_count": first_pass_group_count,
            "batch_case_count": batch_case_count,
            "batch_lane_count": lane_count,
            "first_pass_group_instance_count": first_pass_group_instance_count,
            "second_pass_check_count": second_pass_check_count,
            "accepted_relation_count": accepted_relation_count,
            "accepted_summary_count": len(summaries),
            "relation_replay_derived": bool(relation_replay.get("derived")),
            "relation_replay_matches_secret": bool(relation_replay.get("matches_recorded_secret")),
            "preserves_second_pass_fanout": first_pass_group_instance_count == second_pass_check_count,
            "batch_cases_match_first_pass_groups": batch_case_count == first_pass_group_count,
            "summary_count_matches_accepted_checks": accepted_relation_count == len(summaries),
        },
        "lane_amortization_target": bool(
            rho and lane_fused_ops < rho and lane_first_pass_saved_ops > 0
        ),
        "failures": failures,
        "implementation_boundary": (
            "Lane-amortized contract: first-pass candidate-point work is charged "
            "by validated native batch lanes while second-pass event fanout and "
            "relation derivation remain unchanged."
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [record for record in records if record.get("batch_lane_contract_status") == "batch_lane_contract_verified"]
    derived = [record for record in records if bool((record.get("relation_form_replay") or {}).get("derived"))]
    matched_secret = [
        record for record in records if bool((record.get("relation_form_replay") or {}).get("matches_recorded_secret"))
    ]
    below = [
        record
        for record in verified
        if bool((record.get("lane_amortized_work_counters") or {}).get("lane_fused_below_rho"))
    ]
    lane_amortized = [record for record in below if bool(record.get("lane_amortization_target"))]
    ratios = [
        (record.get("lane_amortized_work_counters") or {}).get("lane_fused_ops_over_rho")
        for record in below
        if (record.get("lane_amortized_work_counters") or {}).get("lane_fused_ops_over_rho") is not None
    ]
    return {
        "record_count": len(records),
        "batch_lane_contract_verified_count": len(verified),
        "batch_lane_contract_derived_count": len(derived),
        "batch_lane_contract_secret_match_count": len(matched_secret),
        "batch_lane_contract_below_rho_count": len(below),
        "batch_lane_contract_lane_amortized_below_rho_count": len(lane_amortized),
        "total_original_fused_ops": sum(
            as_int((record.get("lane_amortized_work_counters") or {}).get("original_fused_ops"))
            for record in verified
        ),
        "total_lane_fused_ops": sum(
            as_int((record.get("lane_amortized_work_counters") or {}).get("lane_fused_ops"))
            for record in verified
        ),
        "total_lane_saved_ops": sum(
            as_int((record.get("lane_amortized_work_counters") or {}).get("lane_total_saved_ops"))
            for record in verified
        ),
        "mean_batch_lane_below_rho_ops_over_rho": (
            round(mean([float(value) for value in ratios]), 8) if ratios else None
        ),
        "interpretation": (
            "The portable contract now charges first-pass candidate-point work "
            "by native batch-lane count while preserving second-pass event checks "
            "and relation-derived secrets."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-source", type=Path, default=DEFAULT_CONTRACT_SOURCE)
    parser.add_argument("--batch-lane-source", type=Path, default=DEFAULT_BATCH_LANE_SOURCE)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    contract_source = load_json(args.contract_source)
    batch_source = load_json(args.batch_lane_source)
    batch_by_source = batch_records_by_source(batch_source)
    wanted = set(args.source_name or [])
    records = []
    for record in contract_source.get("records") or []:
        if not isinstance(record, dict):
            continue
        source_name = str(record.get("source_name") or "")
        if wanted and source_name not in wanted:
            continue
        records.append(lane_counter_record(record, batch_by_source.get(source_name)))

    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_kernel_batch_lane_contract_probe_v1",
        "method": "portable_contract_with_native_batch_lane_first_pass_cost",
        "parameters": {
            "contract_source": str(args.contract_source),
            "batch_lane_source": str(args.batch_lane_source),
            "source_names": sorted(wanted),
        },
        "summary": summarize(records),
        "records": sorted(
            records,
            key=lambda record: (
                record.get("batch_lane_contract_status") != "batch_lane_contract_verified",
                not bool((record.get("lane_amortized_work_counters") or {}).get("lane_fused_below_rho")),
                str(record.get("source_name") or ""),
            ),
        ),
        "non_claims": [
            "This is a lane-amortized contract over existing relation systems, not a new relation search.",
            "The event-key contract remains the conservative fallback until lane accounting is independently promoted.",
            "Second-pass relation predicates and modular linear algebra are unchanged from the portable contract.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
