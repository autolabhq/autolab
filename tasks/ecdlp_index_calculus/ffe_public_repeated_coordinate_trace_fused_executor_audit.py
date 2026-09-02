#!/usr/bin/env python3
"""Recompute shared-pass costs from materialized public hit traces.

`ffe_public_repeated_coordinate_hit_event_share_probe.py --charge-policy`
records a charged replay by subtracting named shared counters.  This audit
removes that dependency: it recomputes the same shared leaf/root/event-pass
cost from the public trace carried by each selected row.

The executor model is still an audit model, not a low-level optimized kernel:
it does not refactor the verifier scan itself.  But all savings are derived
from materialized public rows:

* selected leaf signatures are shared once per identical selected leaf set;
* hit-root work is charged on the union of selected hit-root values;
* one x-match pass is shared by duplicate `(leaf signature, scout, trial)`
  events across rows.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = WORKTREE_ROOT / "ecdlp_index_calculus_state" / "ffe_public_repeated_coordinate_trace_fused_executor_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_ratio(ops: int, rho: int) -> float | None:
    return round(ops / rho, 8) if rho else None


def signature(values: list[Any]) -> str:
    return "|".join(str(value) for value in values)


def row_leaf_signature(row: dict[str, Any]) -> str:
    if row.get("selected_leaf_signature") is not None:
        return str(row.get("selected_leaf_signature"))
    return signature(sorted(as_int(leaf) for leaf in row.get("selected_leaf_indices") or []))


def row_hit_root_values(row: dict[str, Any]) -> list[int]:
    values = row.get("selected_hit_root_values") or row.get("selected_hit_roots_values")
    if isinstance(values, list):
        return sorted({as_int(value) for value in values})
    return []


def row_hit_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    values = row.get("hit_event_summaries") or row.get("x_match_summaries")
    return [event for event in values if isinstance(event, dict)] if isinstance(values, list) else []


def event_key(leaf_signature: str, event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        leaf_signature,
        as_int(event.get("scout_pos"), -1),
        as_int(event.get("original_trial"), -1),
    )


def row_nonshared_base_ops(row: dict[str, Any]) -> int:
    return (
        as_int(row.get("preassociation_filter_ops"))
        - as_int(row.get("selected_leaf_count"))
        - as_int(row.get("selected_hit_roots"))
        - (2 * as_int(row.get("selected_hit_events")))
    )


def trace_complete(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if len(row_hit_root_values(row)) != as_int(row.get("selected_hit_roots")):
            return False
        if len(row_hit_events(row)) != as_int(row.get("selected_hit_events")):
            return False
    return True


def grouped_trace(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        leaf_sig = row_leaf_signature(row)
        group = groups.setdefault(
            leaf_sig,
            {
                "leaf_signature": leaf_sig,
                "row_keys": [],
                "leaf_count_charge": 0,
                "hit_root_values": set(),
                "event_keys": set(),
                "baseline_leaf_ops": 0,
                "baseline_hit_root_ops": 0,
                "baseline_event_pass_ops": 0,
            },
        )
        group["row_keys"].append(str(row.get("row_key") or ""))
        group["leaf_count_charge"] = max(group["leaf_count_charge"], as_int(row.get("selected_leaf_count")))
        group["hit_root_values"].update(row_hit_root_values(row))
        group["event_keys"].update(event_key(leaf_sig, event) for event in row_hit_events(row))
        group["baseline_leaf_ops"] += as_int(row.get("selected_leaf_count"))
        group["baseline_hit_root_ops"] += as_int(row.get("selected_hit_roots"))
        group["baseline_event_pass_ops"] += as_int(row.get("selected_hit_events"))
    return groups


def executor_record(record: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in record.get("selected_rows") or [] if isinstance(row, dict)]
    replay = record.get("replay") or {}
    charged = record.get("charged_replay") or {}
    rho = as_int(replay.get("generic_rho_steps"))
    baseline_ops = as_int(replay.get("ops"))
    complete = trace_complete(rows)
    groups = grouped_trace(rows)

    nonshared_base_ops = sum(row_nonshared_base_ops(row) for row in rows)
    trace_leaf_ops = sum(as_int(group["leaf_count_charge"]) for group in groups.values())
    trace_hit_root_ops = sum(len(group["hit_root_values"]) for group in groups.values())
    trace_first_event_pass_ops = sum(len(group["event_keys"]) for group in groups.values())
    trace_second_event_pass_ops = sum(as_int(row.get("selected_hit_events")) for row in rows)
    trace_ops = (
        nonshared_base_ops
        + trace_leaf_ops
        + trace_hit_root_ops
        + trace_first_event_pass_ops
        + trace_second_event_pass_ops
    )
    charged_ops = as_int(charged.get("charged_ops"), baseline_ops)

    group_records = []
    for group in sorted(groups.values(), key=lambda item: str(item["leaf_signature"])):
        group_records.append(
            {
                "leaf_signature": group["leaf_signature"],
                "row_keys": sorted(group["row_keys"]),
                "leaf_count_charge": as_int(group["leaf_count_charge"]),
                "hit_root_values": sorted(as_int(value) for value in group["hit_root_values"]),
                "event_key_count": len(group["event_keys"]),
                "baseline_leaf_ops": as_int(group["baseline_leaf_ops"]),
                "baseline_hit_root_ops": as_int(group["baseline_hit_root_ops"]),
                "baseline_event_pass_ops": as_int(group["baseline_event_pass_ops"]),
            }
        )

    return {
        "source_name": record.get("source_name"),
        "rule": record.get("rule"),
        "public_group_key": record.get("public_group_key"),
        "selected_row_keys": record.get("selected_row_keys"),
        "trace_complete": complete,
        "replay": {
            "public_key_verified": bool(replay.get("public_key_verified")),
            "derived_secret": replay.get("derived_secret"),
            "rank": as_int(replay.get("rank")),
            "relation_count": as_int(replay.get("relation_count")),
            "baseline_ops": baseline_ops,
            "baseline_ops_over_rho": replay.get("ops_over_rho"),
            "generic_rho_steps": rho,
        },
        "trace_fused_executor": {
            "model": "trace_fused_leaf_root_one_event_pass",
            "materialization_status": "trace_derived_work_counters_not_low_level_kernel",
            "nonshared_base_ops": nonshared_base_ops,
            "trace_leaf_ops": trace_leaf_ops,
            "trace_hit_root_ops": trace_hit_root_ops,
            "trace_first_event_pass_ops": trace_first_event_pass_ops,
            "trace_second_event_pass_ops": trace_second_event_pass_ops,
            "trace_ops": trace_ops,
            "trace_ops_over_rho": clean_ratio(trace_ops, rho),
            "trace_below_rho": bool(rho and trace_ops < rho),
            "trace_savings_vs_baseline_ops": baseline_ops - trace_ops,
            "matches_charged_ops": bool(trace_ops == charged_ops),
            "charged_ops": charged_ops,
            "charged_ops_over_rho": charged.get("charged_ops_over_rho"),
        },
        "groups": group_records,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [record for record in records if bool((record.get("replay") or {}).get("public_key_verified"))]
    complete = [record for record in verified if bool(record.get("trace_complete"))]
    below = [
        record
        for record in complete
        if bool((record.get("trace_fused_executor") or {}).get("trace_below_rho"))
    ]
    matched = [
        record
        for record in complete
        if bool((record.get("trace_fused_executor") or {}).get("matches_charged_ops"))
    ]
    shared_below = [
        record
        for record in below
        if as_int((record.get("trace_fused_executor") or {}).get("trace_savings_vs_baseline_ops")) > 0
    ]
    ratios = [
        (record.get("trace_fused_executor") or {}).get("trace_ops_over_rho")
        for record in below
        if (record.get("trace_fused_executor") or {}).get("trace_ops_over_rho") is not None
    ]
    return {
        "record_count": len(records),
        "verified_replay_count": len(verified),
        "trace_complete_verified_count": len(complete),
        "trace_fused_below_rho_count": len(below),
        "trace_fused_shared_savings_below_rho_count": len(shared_below),
        "trace_fused_matches_charged_ops_count": len(matched),
        "mean_trace_fused_below_rho_ops_over_rho": round(mean(ratios), 8) if ratios else None,
        "interpretation": (
            "Shared-pass cost is recomputed from selected leaf signatures, "
            "hit-root IDs, and all x-match traces.  The remaining non-claim is "
            "that these are trace-derived work counters, not yet a low-level "
            "fused algebraic kernel."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charged-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = load_json(args.charged_source)
    records = [
        executor_record(record)
        for record in source.get("records") or []
        if isinstance(record, dict)
    ]
    output = {
        "schema": "ecdlp_public_repeated_coordinate_trace_fused_executor_audit_v1",
        "method": "trace_derived_fused_leaf_root_one_event_pass_work_counters",
        "parameters": {
            "charged_source": str(args.charged_source),
        },
        "summary": summarize(records),
        "records": sorted(
            records,
            key=lambda record: (
                not bool((record.get("trace_fused_executor") or {}).get("trace_below_rho")),
                str(record.get("source_name") or ""),
            ),
        ),
        "non_claims": [
            "This audit recomputes shared-pass work from traces; it does not yet fuse the verifier scanner internals.",
            "Promotion still requires moving the same work counters into the algebraic replay path.",
            "Relation acceptance remains an audit label after the public x-match trace is built.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
