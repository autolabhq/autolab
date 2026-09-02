#!/usr/bin/env python3
"""Emit and verify portable fused-kernel contracts for repeated coordinates.

The hot-path executor proves that one representative candidate point per
public event key is sufficient for row-specific relation emission.  This probe
turns that executor output into a lower-level kernel contract: first-pass
candidate-point outputs, second-pass row checks, accepted compact relation
summaries, independent modular derivation, and work counters recomputed from
the contract itself.

The contract is meant to be the handoff target for an optimized
FFE/summation-polynomial implementation.  It should be possible to replace the
Python row-context executor with a lower-level kernel and compare against this
artifact without re-reading scanner internals.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_fused_worklist_replay_probe as form_replay


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_contract.json"
DEFAULT_FRONTIER_TARGETS = form_replay.DEFAULT_FRONTIER_TARGETS


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any, default: int = 0) -> int:
    return form_replay.as_int(value, default)


def clean_ratio(ops: int, rho: int) -> float | None:
    return form_replay.clean_ratio(ops, rho)


def event_key_tuple(raw: Any) -> tuple[str, int, int]:
    values = list(raw or [])
    if len(values) != 3:
        return ("", -1, -1)
    return str(values[0]), as_int(values[1], -1), as_int(values[2], -1)


def event_key_list(key: tuple[str, int, int]) -> list[Any]:
    return [key[0], key[1], key[2]]


def relation_summary_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(summary.get("row_key") or ""),
        as_int(summary.get("relation_index"), -1),
        as_int(summary.get("candidate_pos"), -1),
        as_int(summary.get("scheduled_trial"), -1),
        as_int(summary.get("original_trial"), -1),
        as_int(summary.get("q_coeff"), -1),
        as_int(summary.get("rhs"), -1),
        tuple(as_int(value, -1) for value in summary.get("terms") or []),
    )


def relation_summaries_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    summaries = []
    for row in rows:
        row_key = str(row.get("row_key") or "")
        scan = row.get("scan") or {}
        for summary in scan.get("event_summaries") or []:
            if not isinstance(summary, dict):
                continue
            enriched = dict(summary)
            enriched["row_key"] = str(enriched.get("row_key") or row_key)
            key = relation_summary_key(enriched)
            if key in seen:
                continue
            seen.add(key)
            summaries.append(enriched)
    return summaries


def relation_summary_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup = {}
    for summary in relation_summaries_from_rows(rows):
        lookup[(str(summary.get("row_key") or ""), as_int(summary.get("relation_index"), -1))] = summary
    return lookup


def first_pass_contract(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for group in groups:
        key = event_key_tuple(group.get("event_key"))
        out.append(
            {
                "event_key": event_key_list(key),
                "leaf_signature": key[0],
                "scout_pos": key[1],
                "original_trial": key[2],
                "candidate_point": group.get("representative_candidate_point"),
                "candidate_point_ops": as_int(group.get("candidate_point_ops"), 1),
                "fanout": as_int(group.get("fanout")),
                "candidate_point_reused": bool(group.get("candidate_point_reused")),
                "candidate_point_reuse_saved_ops": as_int(group.get("candidate_point_reuse_saved_ops")),
                "accepted_relation_count": as_int(group.get("accepted_relation_count")),
                "representative_row_key": group.get("representative_row_key"),
                "representative_scheduled_trial": as_int(
                    group.get("representative_scheduled_trial"), -1
                ),
                "row_instances": group.get("row_instances") or [],
            }
        )
    return sorted(
        out,
        key=lambda item: (
            str(item["leaf_signature"]),
            as_int(item["scout_pos"]),
            as_int(item["original_trial"]),
        ),
    )


def second_pass_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relation_lookup = relation_summary_lookup(rows)
    checks = []
    for row in rows:
        row_key = str(row.get("row_key") or "")
        scan = row.get("scan") or {}
        for event in scan.get("hit_event_summaries") or []:
            if not isinstance(event, dict):
                continue
            relation_index = event.get("relation_index")
            accepted = bool(event.get("accepted_relation"))
            relation_summary = None
            if accepted:
                relation_summary = relation_lookup.get((row_key, as_int(relation_index, -1)))
            key = event_key_tuple(event.get("event_key"))
            checks.append(
                {
                    "row_key": row_key,
                    "event_key": event_key_list(key),
                    "leaf_signature": key[0],
                    "scout_pos": key[1],
                    "original_trial": key[2],
                    "scheduled_trial": as_int(event.get("scheduled_trial"), -1),
                    "candidate_pos": as_int(event.get("candidate_pos"), -1),
                    "accepted_relation": accepted,
                    "relation_index": relation_index,
                    "relation_summary": relation_summary,
                    "candidate_point_source": event.get("candidate_point_source"),
                    "representative_row_key": event.get("representative_row_key"),
                    "representative_scheduled_trial": as_int(
                        event.get("representative_scheduled_trial"), -1
                    ),
                }
            )
    return sorted(
        checks,
        key=lambda item: (
            str(item["row_key"]),
            str(item["leaf_signature"]),
            as_int(item["scout_pos"]),
            as_int(item["original_trial"]),
            as_int(item["scheduled_trial"]),
        ),
    )


def shared_leaf_ops(rows: list[dict[str, Any]]) -> tuple[int, int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        scan = row.get("scan") or {}
        signature = str(scan.get("selected_leaf_signature") or "")
        grouped[signature].append(as_int(scan.get("selected_leaf_count")))
    baseline = sum(sum(values) for values in grouped.values())
    shared = sum(max(values) for values in grouped.values())
    return baseline, shared


def shared_hit_root_ops(rows: list[dict[str, Any]]) -> tuple[int, int]:
    grouped: dict[str, set[int]] = defaultdict(set)
    baseline = 0
    for row in rows:
        scan = row.get("scan") or {}
        signature = str(scan.get("selected_leaf_signature") or "")
        roots = {as_int(value) for value in scan.get("selected_hit_root_values") or []}
        baseline += len(roots)
        grouped[signature].update(roots)
    return baseline, sum(len(values) for values in grouped.values())


def work_counters_from_contract(
    rows: list[dict[str, Any]],
    first_pass: list[dict[str, Any]],
    second_pass: list[dict[str, Any]],
    rho: int,
) -> dict[str, Any]:
    nonshared_base_ops = sum(as_int((row.get("scan") or {}).get("nonshared_base_ops")) for row in rows)
    baseline_leaf_ops, leaf_ops = shared_leaf_ops(rows)
    baseline_hit_root_ops, hit_root_ops = shared_hit_root_ops(rows)
    first_pass_ops = sum(as_int(group.get("candidate_point_ops"), 1) for group in first_pass)
    second_pass_ops = len(second_pass)
    fused_ops = nonshared_base_ops + leaf_ops + hit_root_ops + first_pass_ops + second_pass_ops
    return {
        "nonshared_base_ops": nonshared_base_ops,
        "baseline_leaf_ops": baseline_leaf_ops,
        "shared_leaf_ops": leaf_ops,
        "baseline_hit_root_ops": baseline_hit_root_ops,
        "shared_hit_root_ops": hit_root_ops,
        "shared_first_event_pass_ops": first_pass_ops,
        "second_event_pass_ops": second_pass_ops,
        "fused_ops": fused_ops,
        "generic_rho_steps": rho,
        "fused_ops_over_rho": clean_ratio(fused_ops, rho),
        "fused_below_rho": bool(rho and fused_ops < rho),
        "first_pass_saved_ops": max(0, second_pass_ops - first_pass_ops),
    }


def relation_replay_from_contract(
    public_group: dict[str, Any],
    summaries: list[dict[str, Any]],
    order_map: dict[str, int],
    recorded_secret: Any,
) -> dict[str, Any]:
    target_modulus, order_source = form_replay.target_order(public_group.get("target"), order_map)
    factor_count = form_replay.factor_variable_count(summaries)
    forms = [
        form_replay.summary_to_form(summary, factor_count, target_modulus)
        for summary in summaries
    ]
    derived = form_replay.derive_secret_from_forms(forms, target_modulus)
    return {
        "target_order": target_modulus,
        "target_order_source": order_source,
        "factor_variable_count": factor_count,
        "relation_summary_count": len(summaries),
        "form_count": len(forms),
        "rank": derived["rank"],
        "consistent": bool(derived["consistent"]),
        "derived": bool(derived["derived"]),
        "derived_secret": derived["derived_secret"],
        "recorded_secret": recorded_secret,
        "matches_recorded_secret": (
            derived["derived_secret"] is not None
            and recorded_secret is not None
            and as_int(derived["derived_secret"]) == as_int(recorded_secret)
        ),
    }


def contract_record(record: dict[str, Any], order_map: dict[str, int]) -> dict[str, Any]:
    replay = record.get("fused_hot_path_replay") or {}
    public_group = record.get("public_group_key") or {}
    rows = replay.get("rows") or []
    first_pass = first_pass_contract(replay.get("candidate_point_groups") or [])
    second_pass = second_pass_contract(rows)
    summaries = [
        check["relation_summary"]
        for check in second_pass
        if bool(check.get("accepted_relation")) and isinstance(check.get("relation_summary"), dict)
    ]
    relation_replay = relation_replay_from_contract(
        public_group,
        summaries,
        order_map,
        replay.get("derived_secret"),
    )
    rho = as_int(replay.get("generic_rho_steps"))
    counters = work_counters_from_contract(rows, first_pass, second_pass, rho)
    executor_counters = replay.get("work_counters") or {}
    accepted_count = sum(1 for check in second_pass if bool(check.get("accepted_relation")))
    missing_summaries = [
        {
            "row_key": check.get("row_key"),
            "event_key": check.get("event_key"),
            "relation_index": check.get("relation_index"),
        }
        for check in second_pass
        if bool(check.get("accepted_relation")) and not isinstance(check.get("relation_summary"), dict)
    ]
    group_instance_count = sum(as_int(group.get("fanout")) for group in first_pass)
    status_ok = bool(
        relation_replay["matches_recorded_secret"]
        and as_int(relation_replay["rank"]) == as_int(replay.get("rank"))
        and as_int(counters["fused_ops"]) == as_int(executor_counters.get("fused_ops"))
        and as_int(counters["shared_first_event_pass_ops"])
        == as_int(executor_counters.get("shared_first_event_pass_ops"))
        and as_int(counters["second_event_pass_ops"])
        == as_int(executor_counters.get("second_event_pass_ops"))
        and group_instance_count == len(second_pass)
        and accepted_count == as_int(replay.get("accepted_instance_count"))
        and not missing_summaries
    )
    return {
        "source_name": record.get("source_name"),
        "public_group_key": public_group,
        "selected_row_keys": record.get("selected_row_keys"),
        "contract_status": "fused_kernel_contract_verified" if status_ok else "fused_kernel_contract_failed_check",
        "kernel_contract": {
            "first_pass_candidate_point_groups": first_pass,
            "second_pass_row_checks": second_pass,
            "accepted_relation_summaries": summaries,
            "row_cost_basis": [
                {
                    "row_key": row.get("row_key"),
                    "selected_leaf_signature": (row.get("scan") or {}).get("selected_leaf_signature"),
                    "selected_hit_root_values": (row.get("scan") or {}).get("selected_hit_root_values") or [],
                    "nonshared_base_ops": as_int((row.get("scan") or {}).get("nonshared_base_ops")),
                }
                for row in rows
            ],
        },
        "relation_form_replay": relation_replay,
        "work_counters": counters,
        "checks": {
            "first_pass_group_count": len(first_pass),
            "first_pass_group_instance_count": group_instance_count,
            "second_pass_check_count": len(second_pass),
            "accepted_relation_count": accepted_count,
            "missing_relation_summary_count": len(missing_summaries),
            "missing_relation_summaries": missing_summaries,
            "matches_executor_fused_ops": bool(
                as_int(counters["fused_ops"]) == as_int(executor_counters.get("fused_ops"))
            ),
            "matches_executor_first_pass_ops": bool(
                as_int(counters["shared_first_event_pass_ops"])
                == as_int(executor_counters.get("shared_first_event_pass_ops"))
            ),
            "matches_executor_second_pass_ops": bool(
                as_int(counters["second_event_pass_ops"])
                == as_int(executor_counters.get("second_event_pass_ops"))
            ),
            "matches_executor_rank": bool(
                as_int(relation_replay["rank"]) == as_int(replay.get("rank"))
            ),
            "matches_executor_secret": bool(relation_replay["matches_recorded_secret"]),
        },
        "event_reuse_target": bool(
            counters.get("fused_below_rho") and as_int(counters.get("first_pass_saved_ops")) > 0
        ),
        "implementation_boundary": (
            "Portable first-pass/second-pass contract derived from the Python "
            "hot-path executor.  A lower-level FFE/summation-polynomial kernel "
            "should emit this same contract without depending on scanner internals."
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [record for record in records if record.get("contract_status") == "fused_kernel_contract_verified"]
    derived = [record for record in records if bool((record.get("relation_form_replay") or {}).get("derived"))]
    matched_secret = [
        record for record in records if bool((record.get("relation_form_replay") or {}).get("matches_recorded_secret"))
    ]
    matched_ops = [
        record for record in records if bool((record.get("checks") or {}).get("matches_executor_fused_ops"))
    ]
    below = [
        record
        for record in verified
        if bool((record.get("work_counters") or {}).get("fused_below_rho"))
    ]
    event_reuse = [record for record in below if bool(record.get("event_reuse_target"))]
    ratios = [
        (record.get("work_counters") or {}).get("fused_ops_over_rho")
        for record in below
        if (record.get("work_counters") or {}).get("fused_ops_over_rho") is not None
    ]
    return {
        "record_count": len(records),
        "contract_verified_count": len(verified),
        "contract_derived_count": len(derived),
        "contract_secret_match_count": len(matched_secret),
        "contract_fused_ops_match_count": len(matched_ops),
        "contract_below_rho_count": len(below),
        "contract_event_reuse_below_rho_count": len(event_reuse),
        "mean_contract_below_rho_ops_over_rho": (
            round(mean([float(value) for value in ratios]), 8) if ratios else None
        ),
        "interpretation": (
            "The hot-path executor has been lowered into a portable contract: "
            "first-pass candidate-point groups, second-pass row checks, compact "
            "relation summaries, independent derivation, and recomputed work counters."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-source", type=Path, required=True)
    parser.add_argument("--frontier-targets", type=Path, default=DEFAULT_FRONTIER_TARGETS)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = load_json(args.executor_source)
    order_map = form_replay.load_order_map(args.frontier_targets)
    wanted = set(args.source_name or [])
    records = []
    for record in source.get("records") or []:
        if not isinstance(record, dict):
            continue
        source_name = str(record.get("source_name") or "")
        if wanted and source_name not in wanted:
            continue
        records.append(contract_record(record, order_map))

    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_kernel_contract_probe_v1",
        "method": "portable_first_pass_second_pass_kernel_contract_from_hot_path_executor",
        "parameters": {
            "executor_source": str(args.executor_source),
            "frontier_targets": str(args.frontier_targets),
            "source_names": sorted(wanted),
        },
        "summary": summarize(records),
        "records": sorted(
            records,
            key=lambda record: (
                not bool(record.get("event_reuse_target")),
                not bool((record.get("work_counters") or {}).get("fused_below_rho")),
                str(record.get("source_name") or ""),
            ),
        ),
        "non_claims": [
            "This is a portable contract emitted from the Python executor, not the optimized FFE kernel itself.",
            "Accepted relation summaries are compact verifier outputs used to replay modular linear algebra.",
            "A promoted implementation must emit the same first-pass and second-pass contract from lower-level FFE/summation-polynomial code.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
