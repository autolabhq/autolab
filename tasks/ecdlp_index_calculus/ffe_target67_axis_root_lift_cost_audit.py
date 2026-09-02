#!/usr/bin/env python3
"""Cost the target-67 axis-root line lift against replay and rho.

The axis-root lift is a pre-bivariate-factorization line enumerator.  This
audit keeps the accounting deliberately simple and explicit:

* axis_root_ops counts one public c-axis root per resultant root;
* candidate_line_test_ops counts one selected-leaf slope/divisibility test per
  axis root and selected leaf point;
* candidate_count is kept separate from the work needed to derive those
  candidates.

The model is a proxy ledger, not a measured implementation benchmark.  Its job
is to say whether the new line-lift stage is in the same cost range as the
previous exact-line root-scan charge and whether n=2 amortization remains the
right target.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_AXIS_ROOT_AUDIT = DEFAULT_STATE_DIR / "ffe_target67_axis_root_line_lift_audit_328_511.json"
DEFAULT_LINE_STAGE_AUDIT = DEFAULT_STATE_DIR / "ffe_target67_line_stage_audit_328_511.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_target67_axis_root_lift_cost_audit_328_511.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("source") or ""),
        str(record.get("surface_id") or ""),
        str(record.get("leaf_signature") or ""),
    )


def infer_generic_rho(line_stage: dict[str, Any], fallback: int) -> int:
    for record in line_stage.get("records") or []:
        over_rho = record.get("line_root_scan_ops_over_rho")
        ops = record.get("line_root_scan_ops")
        if ops is not None and over_rho:
            return int(round(float(ops) / float(over_rho)))
    # Current target-67 artifacts are over p=9803 with generic rho 125.
    return int(fallback)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def build_line_stage_index(line_stage: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {record_key(record): record for record in line_stage.get("records") or []}


def cost_record(axis_record: dict[str, Any], line_record: dict[str, Any] | None, rho: int) -> dict[str, Any]:
    axis_root_ops = int(axis_record.get("axis_root_count") or 0)
    candidate_test_ops = int(axis_record.get("candidate_line_test_count") or 0)
    lift_ops = axis_root_ops + candidate_test_ops
    replay_over_rho = finite_float((line_record or {}).get("replay_measured_ops_over_rho"))
    replay_ops = int(round(replay_over_rho * rho)) if replay_over_rho is not None else None
    line_present = bool(axis_record.get("has_preserving_line"))
    replay_success = bool(axis_record.get("label_replay_success"))
    def over(value: int | None) -> float | None:
        return round(value / rho, 8) if value is not None else None

    lift_plus_replay_ops = lift_ops + replay_ops if replay_ops is not None else None
    amortized_n2_ops = (lift_ops / 2.0 + replay_ops) if replay_ops is not None else None
    break_even_reuse = None
    if replay_ops is not None and replay_ops < rho:
        break_even_reuse = math.floor(lift_ops / max(1, rho - replay_ops)) + 1
        while lift_ops / break_even_reuse + replay_ops >= rho:
            break_even_reuse += 1
    return {
        "bucket": axis_record.get("bucket"),
        "label_replay_success": replay_success,
        "has_preserving_line": line_present,
        "target": axis_record.get("target"),
        "transfer_index": axis_record.get("transfer_index"),
        "top_k": axis_record.get("top_k"),
        "leaf_signature": axis_record.get("leaf_signature"),
        "preserving_line": axis_record.get("preserving_line"),
        "preserving_line_recovered": bool(axis_record.get("preserving_line_recovered")),
        "candidate_line_count": int(axis_record.get("candidate_line_count") or 0),
        "axis_root_ops": axis_root_ops,
        "candidate_line_test_ops": candidate_test_ops,
        "line_lift_ops": lift_ops,
        "line_lift_ops_over_rho": over(lift_ops),
        "line_lift_below_rho": lift_ops < rho,
        "replay_ops": replay_ops,
        "replay_ops_over_rho": replay_over_rho,
        "line_lift_plus_replay_ops": lift_plus_replay_ops,
        "line_lift_plus_replay_ops_over_rho": over(lift_plus_replay_ops),
        "line_lift_plus_replay_below_rho": (
            lift_plus_replay_ops is not None and lift_plus_replay_ops < rho
        ),
        "amortized_n2_line_lift_plus_replay_ops": (
            round(amortized_n2_ops, 8) if amortized_n2_ops is not None else None
        ),
        "amortized_n2_line_lift_plus_replay_ops_over_rho": (
            round(amortized_n2_ops / rho, 8) if amortized_n2_ops is not None else None
        ),
        "amortized_n2_line_lift_plus_replay_below_rho": (
            amortized_n2_ops is not None and amortized_n2_ops < rho
        ),
        "line_lift_replay_break_even_reuse_count": break_even_reuse,
    }


def values(records: list[dict[str, Any]], key: str) -> list[Any]:
    return sorted(record[key] for record in records if record.get(key) is not None)


def summarize(records: list[dict[str, Any]], rho: int) -> dict[str, Any]:
    line_present = [record for record in records if record["has_preserving_line"]]
    replay_success = [record for record in line_present if record["label_replay_success"]]
    return {
        "generic_rho_steps": rho,
        "surface_case_count": len(records),
        "line_present_case_count": len(line_present),
        "replay_success_case_count": len(replay_success),
        "line_present_recovered_count": sum(1 for record in line_present if record["preserving_line_recovered"]),
        "replay_success_recovered_count": sum(1 for record in replay_success if record["preserving_line_recovered"]),
        "replay_success_singleton_candidate_count": sum(
            1 for record in replay_success if int(record["candidate_line_count"]) == 1
        ),
        "line_lift_ops_over_rho_values": values(line_present, "line_lift_ops_over_rho"),
        "replay_success_line_lift_ops_over_rho_values": values(
            replay_success,
            "line_lift_ops_over_rho",
        ),
        "replay_success_line_lift_below_rho_count": sum(
            1 for record in replay_success if record["line_lift_below_rho"]
        ),
        "replay_success_line_lift_plus_replay_below_rho_count": sum(
            1 for record in replay_success if record["line_lift_plus_replay_below_rho"]
        ),
        "replay_success_amortized_n2_below_rho_count": sum(
            1 for record in replay_success if record["amortized_n2_line_lift_plus_replay_below_rho"]
        ),
        "replay_success_break_even_reuse_count_values": values(
            replay_success,
            "line_lift_replay_break_even_reuse_count",
        ),
        "max_line_lift_ops_over_rho": max(values(line_present, "line_lift_ops_over_rho") or [None]),
        "min_amortized_n2_line_lift_plus_replay_ops_over_rho": min(
            values(replay_success, "amortized_n2_line_lift_plus_replay_ops_over_rho") or [None]
        ),
        "interpretation": (
            "The axis-root lift is below rho on every replay-success line-present case "
            "under the root/test-count proxy, but lift+replay is not below rho per row. "
            "At n=2 reuse, all replay-success cases fall below rho again."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-root-audit", type=Path, default=DEFAULT_AXIS_ROOT_AUDIT)
    parser.add_argument("--line-stage-audit", type=Path, default=DEFAULT_LINE_STAGE_AUDIT)
    parser.add_argument("--generic-rho", type=int, default=125)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    axis = load_json(args.axis_root_audit)
    line_stage = load_json(args.line_stage_audit)
    rho = infer_generic_rho(line_stage, int(args.generic_rho))
    line_stage_index = build_line_stage_index(line_stage)
    records = [
        cost_record(record, line_stage_index.get(record_key(record)), rho)
        for record in axis.get("records") or []
    ]
    output = {
        "schema": "ecdlp_target67_axis_root_lift_cost_audit_v1",
        "method": "axis_root_lift_root_and_line_test_count_cost_proxy",
        "parameters": {
            "axis_root_audit": str(args.axis_root_audit),
            "line_stage_audit": str(args.line_stage_audit),
            "generic_rho": rho,
            "cost_model": (
                "axis_root_ops + candidate_line_test_ops; candidate tests are counted "
                "as public selected-leaf line divisibility tests, not expanded polynomial term operations"
            ),
        },
        "summary": summarize(records, rho),
        "records": records,
        "non_claims": [
            "This is a proxy cost ledger, not a measured optimized implementation.",
            "It does not include public row-selection cost.",
            "It does not claim that brute-force c-axis root scanning over the field is below rho; a real implementation should use univariate root finding/factorization.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
