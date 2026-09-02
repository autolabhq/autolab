#!/usr/bin/env python3
"""Audit whether charged hit-event sharing is materialized by current artifacts.

The charged hit-event replay proves a useful public-row accounting target, but
it still subtracts shared work from row-level counters.  This audit separates
what is already present in the replay artifacts from what a real shared
FFE/summation-polynomial pass still has to expose:

* duplicate selected-leaf identities are present and can be shared from the
  artifact;
* selected hit-root counts are present, but root identities are not;
* selected hit-event counts are present, but non-relation x-match events are
  not summarized.

The output is intentionally conservative.  A charged replay is not considered
materialized unless every charged component has the public trace needed to
recompute the saving without post-hoc counters.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = WORKTREE_ROOT / "ecdlp_index_calculus_state" / "ffe_public_repeated_coordinate_shared_pass_materialization_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 8) if denominator else None


def signature(values: list[Any]) -> str:
    return "|".join(str(value) for value in values)


def row_leaf_signature(row: dict[str, Any]) -> str:
    if row.get("selected_leaf_signature") is not None:
        return str(row.get("selected_leaf_signature"))
    return signature(sorted(as_int(leaf) for leaf in row.get("selected_leaf_indices") or []))


def duplicate_share(values: list[int]) -> int:
    if len(values) < 2:
        return 0
    return sum(values) - max(values)


def leaf_share_from_rows(rows: list[dict[str, Any]]) -> int:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row_leaf_signature(row)].append(as_int(row.get("selected_leaf_count")))
    return sum(duplicate_share(values) for values in grouped.values())


def hit_root_share_from_counts(rows: list[dict[str, Any]]) -> int:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row_leaf_signature(row)].append(as_int(row.get("selected_hit_roots")))
    return sum(duplicate_share(values) for values in grouped.values())


def event_pass_share_from_counts(rows: list[dict[str, Any]]) -> int:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row_leaf_signature(row)].append(as_int(row.get("selected_hit_events")))
    return sum(duplicate_share(values) for values in grouped.values())


def has_hit_root_id_trace(row: dict[str, Any]) -> bool:
    roots = row.get("selected_hit_root_values") or row.get("selected_hit_roots_values")
    return isinstance(roots, list) and len(roots) == as_int(row.get("selected_hit_roots"))


def has_all_xmatch_trace(row: dict[str, Any]) -> bool:
    events = row.get("hit_event_summaries") or row.get("x_match_summaries")
    return isinstance(events, list) and len(events) == as_int(row.get("selected_hit_events"))


def relation_event_count(row: dict[str, Any]) -> int:
    if row.get("relation_event_count") is not None:
        return as_int(row.get("relation_event_count"))
    summaries = row.get("event_summaries")
    return len(summaries) if isinstance(summaries, list) else 0


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in record.get("selected_rows") or [] if isinstance(row, dict)]
    replay = record.get("replay") or {}
    charged = record.get("charged_replay") or {}
    charge_policy = str(charged.get("charge_policy") or "")
    total_hit_events = sum(as_int(row.get("selected_hit_events")) for row in rows)
    total_relation_events = sum(relation_event_count(row) for row in rows)
    non_relation_hit_events = max(0, total_hit_events - total_relation_events)

    charged_leaf_share = as_int(charged.get("shared_leaf_ops"))
    charged_hit_root_share = as_int(charged.get("shared_hit_root_ops"))
    charged_event_share = (
        as_int(charged.get("shared_one_event_pass_ops"))
        + as_int(charged.get("shared_second_event_pass_ops"))
    )
    artifact_leaf_share = leaf_share_from_rows(rows)
    artifact_hit_root_share = hit_root_share_from_counts(rows)
    artifact_event_share = event_pass_share_from_counts(rows)

    missing_hit_root_ids = bool(charged_hit_root_share and not all(has_hit_root_id_trace(row) for row in rows))
    missing_xmatch_trace = bool(charged_event_share and not all(has_all_xmatch_trace(row) for row in rows))
    relation_only_event_trace = bool(charged_event_share and non_relation_hit_events > 0 and missing_xmatch_trace)
    charged_below = bool(charged.get("charged_below_rho"))
    total_shared_ops = charged_leaf_share + charged_hit_root_share + charged_event_share
    materialized = bool(
        charged_below
        and total_shared_ops > 0
        and charged_leaf_share <= artifact_leaf_share
        and charged_hit_root_share <= artifact_hit_root_share
        and charged_event_share <= artifact_event_share
        and not missing_hit_root_ids
        and not missing_xmatch_trace
    )

    blockers = []
    if charged_leaf_share > artifact_leaf_share:
        blockers.append("leaf_share_exceeds_artifact_duplicate_leaf_counts")
    if charged_hit_root_share > artifact_hit_root_share:
        blockers.append("hit_root_share_exceeds_artifact_duplicate_count_model")
    if charged_event_share > artifact_event_share:
        blockers.append("event_share_exceeds_artifact_duplicate_count_model")
    if missing_hit_root_ids:
        blockers.append("missing_selected_hit_root_id_trace")
    if missing_xmatch_trace:
        blockers.append("missing_all_xmatch_event_trace")
    if relation_only_event_trace:
        blockers.append("stored_event_summaries_cover_relations_only")

    return {
        "source_name": record.get("source_name"),
        "rule": record.get("rule"),
        "charge_policy": charge_policy,
        "public_group_key": record.get("public_group_key"),
        "selected_row_keys": record.get("selected_row_keys"),
        "replay": {
            "public_key_verified": bool(replay.get("public_key_verified")),
            "derived_secret": replay.get("derived_secret"),
            "rank": as_int(replay.get("rank")),
            "relation_count": as_int(replay.get("relation_count")),
            "baseline_ops": as_int(replay.get("ops")),
            "baseline_ops_over_rho": replay.get("ops_over_rho"),
            "generic_rho_steps": as_int(replay.get("generic_rho_steps")),
        },
        "charged_replay": charged,
        "trace_coverage": {
            "selected_row_count": len(rows),
            "selected_leaf_signatures": sorted({row_leaf_signature(row) for row in rows}),
            "selected_hit_event_count": total_hit_events,
            "relation_event_summary_count": total_relation_events,
            "non_relation_hit_event_count": non_relation_hit_events,
            "relation_event_summary_coverage": clean_ratio(total_relation_events, total_hit_events),
            "all_rows_have_hit_root_id_trace": all(has_hit_root_id_trace(row) for row in rows),
            "all_rows_have_xmatch_trace": all(has_all_xmatch_trace(row) for row in rows),
        },
        "share_accounting": {
            "charged_leaf_share": charged_leaf_share,
            "artifact_duplicate_leaf_share": artifact_leaf_share,
            "charged_hit_root_share": charged_hit_root_share,
            "artifact_duplicate_hit_root_share_from_counts": artifact_hit_root_share,
            "charged_event_pass_share": charged_event_share,
            "artifact_duplicate_event_pass_share_from_counts": artifact_event_share,
        },
        "materialization_status": (
            "materialized_from_current_artifact" if materialized else "requires_additional_public_scan_trace"
        ),
        "materialized_charged_below_rho": materialized,
        "requires_shared_savings_for_charged_win": bool(charged_below and total_shared_ops > 0),
        "blockers": blockers,
        "kernel_work_item": {
            "needed_for_promotion": bool(charged_below and blockers),
            "capture_selected_hit_root_ids": missing_hit_root_ids,
            "capture_all_xmatch_events": missing_xmatch_trace,
            "preserve_relation_event_labels_for_audit_only": bool(charged_event_share),
        },
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    charged_below = [record for record in records if bool((record.get("charged_replay") or {}).get("charged_below_rho"))]
    charged_below_with_shared_savings = [
        record for record in charged_below if bool(record.get("requires_shared_savings_for_charged_win"))
    ]
    materialized = [record for record in records if bool(record.get("materialized_charged_below_rho"))]
    missing_roots = [record for record in charged_below if "missing_selected_hit_root_id_trace" in (record.get("blockers") or [])]
    missing_xmatches = [record for record in charged_below if "missing_all_xmatch_event_trace" in (record.get("blockers") or [])]
    relation_only = [record for record in charged_below if "stored_event_summaries_cover_relations_only" in (record.get("blockers") or [])]
    charged_ratios = [
        (record.get("charged_replay") or {}).get("charged_ops_over_rho")
        for record in charged_below
        if (record.get("charged_replay") or {}).get("charged_ops_over_rho") is not None
    ]
    if not missing_roots and not missing_xmatches and len(materialized) == len(charged_below_with_shared_savings):
        interpretation = (
            "The current artifacts materialize the public hit-root IDs and all "
            "x-match traces needed to audit the charged shared-pass rows. The "
            "remaining promotion step is to move the same sharing into the "
            "algebraic replay kernel instead of subtracting charged counters."
        )
    else:
        interpretation = (
            "The charged replay remains a valid accounting target, but current "
            "artifacts do not yet materialize the hit-root identities or all "
            "x-match event traces needed for a promoted shared-pass kernel."
        )
    return {
        "record_count": len(records),
        "charged_below_rho_count": len(charged_below),
        "charged_below_with_shared_savings_count": len(charged_below_with_shared_savings),
        "baseline_already_below_rho_count": len(charged_below) - len(charged_below_with_shared_savings),
        "materialized_shared_savings_below_rho_count": len(materialized),
        "charged_below_missing_hit_root_id_trace_count": len(missing_roots),
        "charged_below_missing_all_xmatch_trace_count": len(missing_xmatches),
        "charged_below_relation_only_event_trace_count": len(relation_only),
        "mean_charged_below_ops_over_rho": round(mean(charged_ratios), 8) if charged_ratios else None,
        "interpretation": interpretation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charged-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    charged_source = load_json(args.charged_source)
    records = [
        audit_record(record)
        for record in charged_source.get("records") or []
        if isinstance(record, dict) and isinstance(record.get("charged_replay"), dict)
    ]
    output = {
        "schema": "ecdlp_public_repeated_coordinate_shared_pass_materialization_audit_v1",
        "method": "charged_replay_public_trace_materialization_boundary",
        "parameters": {
            "charged_source": str(args.charged_source),
        },
        "summary": summarize(records),
        "records": sorted(
            records,
            key=lambda record: (
                not bool(record.get("materialized_charged_below_rho")),
                str(record.get("source_name") or ""),
                str(record.get("selected_row_keys") or []),
            ),
        ),
        "non_claims": [
            "This audit does not implement a shared FFE/summation-polynomial kernel.",
            "Count-derived hit-root and hit-event sharing is not promoted without public traces for the shared objects.",
            "Relation-event summaries remain verifier labels unless all x-match events are captured before relation acceptance.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
