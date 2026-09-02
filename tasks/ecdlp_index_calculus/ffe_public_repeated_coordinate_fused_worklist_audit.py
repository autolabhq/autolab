#!/usr/bin/env python3
"""Build a public fused-worklist plan for repeated-coordinate hit traces.

The trace-fused executor audit recomputes shared costs from materialized traces.
This audit takes the next narrower step toward an executable kernel: it emits
the public worklist that a fused leaf/root/x-match pass would run, and checks
that accepted x-match instances can be linked back to relation summaries from
the decomposition artifacts.

It still does not replace the low-level verifier scanner.  The output is a
kernel implementation target and consistency check, not a promoted ECDLP
speedup claim.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = WORKTREE_ROOT / "ecdlp_index_calculus_state" / "ffe_public_repeated_coordinate_fused_worklist_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return path
    candidate = WORKTREE_ROOT / path
    return candidate if candidate.exists() else path


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


def relation_summary_key(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "relation_index": as_int(summary.get("relation_index")),
        "leaf_index": as_int(summary.get("leaf_index")),
        "candidate_pos": as_int(summary.get("candidate_pos")),
        "scheduled_trial": as_int(summary.get("scheduled_trial")),
        "original_trial": as_int(summary.get("original_trial")),
        "q_coeff": as_int(summary.get("q_coeff")),
        "rhs": as_int(summary.get("rhs")),
        "term_shape": summary.get("term_shape"),
        "factor_support": [as_int(value) for value in summary.get("factor_support") or []],
        "terms": [as_int(value) for value in summary.get("terms") or []],
    }


def record_recovery(record: dict[str, Any]) -> dict[str, Any] | None:
    source_path = resolve_path(record.get("source_path"))
    if source_path is None or not source_path.exists():
        return None
    artifact = load_json(source_path)
    public_group = record.get("public_group_key") or {}
    target = public_group.get("target")
    transfer = as_int(public_group.get("transfer_index"))
    coord_text = str(public_group.get("coordinate_key") or "")
    for recovery in artifact.get("recoveries") or []:
        if not isinstance(recovery, dict):
            continue
        coord = recovery.get("coordinate") or {}
        recovery_coord = f"{as_int(coord.get('b'))},{as_int(coord.get('c'))}"
        if (
            recovery.get("target") == target
            and as_int(recovery.get("transfer_index")) == transfer
            and recovery_coord == coord_text
        ):
            return recovery
    return None


def relation_summary_lookup(record: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    recovery = record_recovery(record)
    if recovery is None:
        return {}
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in recovery.get("row_costs") or []:
        if not isinstance(row, dict):
            continue
        row_key = str(row.get("row_key") or "")
        for summary in row.get("event_summaries") or []:
            if isinstance(summary, dict):
                lookup[(row_key, as_int(summary.get("relation_index"), -1))] = relation_summary_key(summary)
    return lookup


def trace_complete(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if len(row_hit_root_values(row)) != as_int(row.get("selected_hit_roots")):
            return False
        if len(row_hit_events(row)) != as_int(row.get("selected_hit_events")):
            return False
    return True


def build_event_instances(
    rows: list[dict[str, Any]],
    relation_summaries: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    instances = []
    missing_relation_summaries = []
    for row in rows:
        row_key = str(row.get("row_key") or "")
        leaf_signature = row_leaf_signature(row)
        for ordinal, event in enumerate(row_hit_events(row)):
            relation_index = event.get("relation_index")
            relation_summary = None
            if bool(event.get("accepted_relation")):
                relation_summary = relation_summaries.get((row_key, as_int(relation_index, -1)))
                if relation_summary is None:
                    missing_relation_summaries.append(
                        {
                            "row_key": row_key,
                            "relation_index": relation_index,
                            "leaf_signature": leaf_signature,
                            "scout_pos": as_int(event.get("scout_pos"), -1),
                            "original_trial": as_int(event.get("original_trial"), -1),
                        }
                    )
            instances.append(
                {
                    "row_key": row_key,
                    "leaf_signature": leaf_signature,
                    "event_ordinal": ordinal,
                    "event_key": list(event_key(leaf_signature, event)),
                    "leaf_index": as_int(event.get("leaf_index"), -1),
                    "scout_pos": as_int(event.get("scout_pos"), -1),
                    "candidate_pos": as_int(event.get("candidate_pos"), -1),
                    "scheduled_trial": as_int(event.get("scheduled_trial"), -1),
                    "original_trial": as_int(event.get("original_trial"), -1),
                    "accepted_relation": bool(event.get("accepted_relation")),
                    "relation_index": relation_index,
                    "relation_summary": relation_summary,
                }
            )
    return instances, missing_relation_summaries


def leaf_worklist(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row_leaf_signature(row)].append(row)
    worklist = []
    for leaf_signature, group in sorted(groups.items()):
        baseline_leaf_ops = sum(as_int(row.get("selected_leaf_count")) for row in group)
        shared_leaf_ops = max([as_int(row.get("selected_leaf_count")) for row in group] or [0])
        worklist.append(
            {
                "leaf_signature": leaf_signature,
                "row_keys": sorted(str(row.get("row_key") or "") for row in group),
                "baseline_leaf_ops": baseline_leaf_ops,
                "shared_leaf_ops": shared_leaf_ops,
                "leaf_ops_saved": baseline_leaf_ops - shared_leaf_ops,
            }
        )
    return worklist


def hit_root_worklist(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row_leaf_signature(row)].append(row)
    worklist = []
    for leaf_signature, group in sorted(groups.items()):
        root_to_rows: dict[int, set[str]] = defaultdict(set)
        baseline_hit_root_ops = 0
        for row in group:
            row_key = str(row.get("row_key") or "")
            roots = row_hit_root_values(row)
            baseline_hit_root_ops += len(roots)
            for root in roots:
                root_to_rows[root].add(row_key)
        shared_hit_root_ops = len(root_to_rows)
        worklist.append(
            {
                "leaf_signature": leaf_signature,
                "hit_roots": [
                    {"root": root, "row_keys": sorted(row_keys)}
                    for root, row_keys in sorted(root_to_rows.items())
                ],
                "baseline_hit_root_ops": baseline_hit_root_ops,
                "shared_hit_root_ops": shared_hit_root_ops,
                "hit_root_ops_saved": baseline_hit_root_ops - shared_hit_root_ops,
            }
        )
    return worklist


def event_worklist(instances: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        groups[tuple(instance["event_key"])].append(instance)

    conflicts = []
    worklist = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        candidate_positions = sorted({as_int(item.get("candidate_pos"), -1) for item in group})
        leaf_indices = sorted({as_int(item.get("leaf_index"), -1) for item in group})
        accepted_values = sorted({bool(item.get("accepted_relation")) for item in group})
        relation_summaries = [
            item.get("relation_summary")
            for item in group
            if isinstance(item.get("relation_summary"), dict)
        ]
        public_consistent = len(candidate_positions) == 1 and len(leaf_indices) == 1
        if len(group) > 1 and not public_consistent:
            conflicts.append(
                {
                    "event_key": list(key),
                    "candidate_pos_values": candidate_positions,
                    "leaf_index_values": leaf_indices,
                    "kind": "duplicate_event_public_field_mismatch",
                }
            )
        worklist.append(
            {
                "event_key": list(key),
                "row_instances": [
                    {
                        "row_key": item["row_key"],
                        "scheduled_trial": item["scheduled_trial"],
                        "candidate_pos": item["candidate_pos"],
                        "accepted_relation": item["accepted_relation"],
                        "relation_index": item["relation_index"],
                    }
                    for item in sorted(group, key=lambda item: (str(item["row_key"]), item["scheduled_trial"]))
                ],
                "fanout": len(group),
                "first_pass_reused": len(group) > 1,
                "first_pass_ops_saved": max(0, len(group) - 1),
                "candidate_pos_values": candidate_positions,
                "leaf_index_values": leaf_indices,
                "accepted_relation_values": accepted_values,
                "accepted_relation_count": sum(1 for item in group if bool(item.get("accepted_relation"))),
                "relation_summaries": relation_summaries,
                "relation_summary_count": len(relation_summaries),
                "public_fields_consistent": public_consistent,
            }
        )
    return worklist, conflicts


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in record.get("selected_rows") or [] if isinstance(row, dict)]
    replay = record.get("replay") or {}
    charged = record.get("charged_replay") or {}
    rho = as_int(replay.get("generic_rho_steps"))
    baseline_ops = as_int(replay.get("ops"))
    relation_summaries = relation_summary_lookup(record)
    instances, missing_relation_summaries = build_event_instances(rows, relation_summaries)
    event_plan, event_conflicts = event_worklist(instances)
    leaf_plan = leaf_worklist(rows)
    root_plan = hit_root_worklist(rows)

    nonshared_base_ops = sum(row_nonshared_base_ops(row) for row in rows)
    shared_leaf_ops = sum(as_int(item.get("shared_leaf_ops")) for item in leaf_plan)
    shared_hit_root_ops = sum(as_int(item.get("shared_hit_root_ops")) for item in root_plan)
    shared_first_pass_ops = len(event_plan)
    second_pass_ops = len(instances)
    fused_ops = nonshared_base_ops + shared_leaf_ops + shared_hit_root_ops + shared_first_pass_ops + second_pass_ops
    first_pass_saved_ops = sum(as_int(item.get("first_pass_ops_saved")) for item in event_plan)
    accepted_instance_count = sum(1 for item in instances if bool(item.get("accepted_relation")))
    linked_relation_summary_count = sum(
        1 for item in instances if bool(item.get("accepted_relation")) and isinstance(item.get("relation_summary"), dict)
    )
    complete = trace_complete(rows)
    ready = bool(complete and not event_conflicts and not missing_relation_summaries)
    below = bool(rho and fused_ops < rho)
    charged_ops = as_int(charged.get("charged_ops"), baseline_ops)

    return {
        "source_name": record.get("source_name"),
        "rule": record.get("rule"),
        "public_group_key": record.get("public_group_key"),
        "selected_row_keys": record.get("selected_row_keys"),
        "replay": {
            "public_key_verified": bool(replay.get("public_key_verified")),
            "derived_secret": replay.get("derived_secret"),
            "rank": as_int(replay.get("rank")),
            "relation_count": as_int(replay.get("relation_count")),
            "baseline_ops": baseline_ops,
            "baseline_ops_over_rho": replay.get("ops_over_rho"),
            "generic_rho_steps": rho,
        },
        "fused_worklist": {
            "status": "ready_for_kernel_implementation" if ready else "requires_trace_fix_before_kernel",
            "materialization_status": "public_worklist_not_low_level_kernel",
            "trace_complete": complete,
            "nonshared_base_ops": nonshared_base_ops,
            "shared_leaf_ops": shared_leaf_ops,
            "shared_hit_root_ops": shared_hit_root_ops,
            "shared_first_event_pass_ops": shared_first_pass_ops,
            "second_event_pass_ops": second_pass_ops,
            "fused_ops": fused_ops,
            "fused_ops_over_rho": clean_ratio(fused_ops, rho),
            "fused_below_rho": below,
            "matches_charged_ops": bool(fused_ops == charged_ops),
            "charged_ops": charged_ops,
            "charged_ops_over_rho": charged.get("charged_ops_over_rho"),
            "ops_saved_vs_baseline": baseline_ops - fused_ops,
            "first_pass_saved_ops": first_pass_saved_ops,
            "reused_first_pass_event_count": sum(1 for item in event_plan if bool(item.get("first_pass_reused"))),
            "event_instance_count": len(instances),
            "unique_first_pass_event_count": len(event_plan),
            "accepted_instance_count": accepted_instance_count,
            "linked_relation_summary_count": linked_relation_summary_count,
            "missing_relation_summary_count": len(missing_relation_summaries),
        },
        "leaf_worklist": leaf_plan,
        "hit_root_worklist": root_plan,
        "event_worklist": event_plan,
        "event_conflicts": event_conflicts,
        "missing_relation_summaries": missing_relation_summaries,
        "implementation_note": (
            "The first event pass is public and reusable across duplicate event "
            "keys; relation acceptance and linear-form emission remain row "
            "instances in the second pass."
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [record for record in records if bool((record.get("replay") or {}).get("public_key_verified"))]
    ready = [record for record in verified if (record.get("fused_worklist") or {}).get("status") == "ready_for_kernel_implementation"]
    below = [record for record in ready if bool((record.get("fused_worklist") or {}).get("fused_below_rho"))]
    shared_below = [
        record
        for record in below
        if as_int((record.get("fused_worklist") or {}).get("ops_saved_vs_baseline")) > 0
    ]
    event_reuse_below = [
        record
        for record in below
        if as_int((record.get("fused_worklist") or {}).get("first_pass_saved_ops")) > 0
    ]
    matched = [
        record
        for record in ready
        if bool((record.get("fused_worklist") or {}).get("matches_charged_ops"))
    ]
    conflicts = [record for record in records if record.get("event_conflicts")]
    missing = [record for record in records if record.get("missing_relation_summaries")]
    ratios = [
        (record.get("fused_worklist") or {}).get("fused_ops_over_rho")
        for record in below
        if (record.get("fused_worklist") or {}).get("fused_ops_over_rho") is not None
    ]
    return {
        "record_count": len(records),
        "verified_replay_count": len(verified),
        "ready_for_kernel_implementation_count": len(ready),
        "fused_worklist_below_rho_count": len(below),
        "fused_worklist_shared_savings_below_rho_count": len(shared_below),
        "fused_worklist_event_reuse_below_rho_count": len(event_reuse_below),
        "fused_worklist_matches_charged_ops_count": len(matched),
        "event_conflict_record_count": len(conflicts),
        "missing_relation_summary_record_count": len(missing),
        "mean_fused_worklist_below_rho_ops_over_rho": round(mean(ratios), 8) if ratios else None,
        "interpretation": (
            "The audit emits a public fused worklist and relation-summary links. "
            "It is a kernel implementation target; it does not replace the "
            "low-level FFE/summation-polynomial scanner."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charged-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = load_json(args.charged_source)
    records = [
        audit_record(record)
        for record in source.get("records") or []
        if isinstance(record, dict)
    ]
    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_worklist_audit_v1",
        "method": "public_leaf_root_xmatch_fused_worklist_with_relation_summary_links",
        "parameters": {
            "charged_source": str(args.charged_source),
        },
        "summary": summarize(records),
        "records": sorted(
            records,
            key=lambda record: (
                not bool((record.get("fused_worklist") or {}).get("fused_below_rho")),
                str(record.get("source_name") or ""),
            ),
        ),
        "non_claims": [
            "This artifact is an executable worklist specification, not the optimized algebraic kernel itself.",
            "Relation labels are used only to verify that accepted x-match instances link back to replay summaries.",
            "Promotion still requires the replay scanner to consume this public worklist directly and report actual work counters.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
