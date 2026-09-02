#!/usr/bin/env python3
"""Audit fused candidate-point reuse for repeated-coordinate worklists.

The fused worklist scanner already schedules public event keys and emits
relation forms through the verifier.  This probe tests the next lower boundary:
for duplicate public event keys, can the first-pass elliptic-curve candidate
point be computed once and reused across row-specific relation checks?

The script rematerializes row contexts, groups row instances by public event
key, audits candidate-point equality within each group, then emits relation
forms using one shared candidate point for point-consistent groups.  Extra
candidate-point recomputations used for the equality audit are reported but not
charged to the planned fused-kernel counters.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_fused_worklist_scanner_probe as scanner_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_candidate_point_probe.json"

replay_probe = scanner_probe.replay_probe


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any, default: int = 0) -> int:
    return scanner_probe.as_int(value, default)


def clean_ratio(ops: int, rho: int) -> float | None:
    return scanner_probe.clean_ratio(ops, rho)


def compact_point(point: Any) -> Any:
    if isinstance(point, tuple):
        return [compact_point(value) for value in point]
    if isinstance(point, list):
        return [compact_point(value) for value in point]
    if isinstance(point, dict):
        return {str(key): compact_point(value) for key, value in sorted(point.items())}
    try:
        return int(point)
    except (TypeError, ValueError):
        return point


def instance_key(instance: dict[str, Any]) -> tuple[str, tuple[str, int, int], int]:
    return scanner_probe.instance_key(instance)


def expected_instance_lookup(instances: list[dict[str, Any]]) -> dict[tuple[str, tuple[str, int, int], int], dict[str, Any]]:
    return {instance_key(instance): instance for instance in instances}


def prepare_row_state(
    row_key: str,
    leaves: set[int],
    allowed_keys: set[tuple[str, int, int]],
    context: dict[str, Any],
) -> dict[str, Any]:
    built = context["built"]
    local_args = context["local_args"]
    p = int(built["p"])
    row_leaf_signature = scanner_probe.leaf_signature(leaves)
    cover = replay_probe.filtered_leaf_gcd_association_with_trace(
        built["scouts"],
        context["components"],
        p,
        leaves,
    )
    scheduled_rows = replay_probe.direct_witness_probe.eval_cover_probe.schedule_rows(
        built["rows"],
        cover["row_hit_count"],
        int(built["scheduled_row_count"]),
    )
    ordered_scouts = replay_probe.direct_witness_probe.eval_cover_probe.order_scouts(
        built["scouts"],
        cover["hits_by_scout"],
        scheduled_rows,
        str(built["scout_order"]),
    )
    scout_to_leaf = replay_probe.direct_witness_probe.critical_leaf_probe.scout_leaf_map(
        context["components"]
    )
    scheduled_by_original = {int(row["original_trial"]): row for row in scheduled_rows}
    keyed_instances: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)

    for candidate_pos, scout in enumerate(ordered_scouts[: int(built["selected_limit"])], start=1):
        scout_pos = int(scout["scout_pos"])
        leaf_index = scout_to_leaf[scout_pos]
        hit_rows = [
            scheduled_by_original[trial]
            for trial in cover["hits_by_scout"].get(scout_pos, [])
            if trial in scheduled_by_original
        ]
        for row in hit_rows:
            event_key = (row_leaf_signature, scout_pos, int(row["original_trial"]))
            if event_key not in allowed_keys:
                continue
            keyed_instances[event_key].append(
                {
                    "row_key": row_key,
                    "event_key": scanner_probe.compact_event_key(event_key),
                    "leaf_index": leaf_index,
                    "scout_pos": scout_pos,
                    "candidate_pos": candidate_pos,
                    "scheduled_trial": int(row["trial"]),
                    "original_trial": int(row["original_trial"]),
                    "row": row,
                    "scout": scout,
                }
            )

    return {
        "row_key": row_key,
        "leaves": leaves,
        "built": built,
        "local_args": local_args,
        "cover": cover,
        "keyed_instances": keyed_instances,
        "forms": [],
        "relations": [],
        "relation_events": [],
        "hit_events": [],
        "seen_forms": set(),
    }


def emit_group_relations(
    verifier: Any,
    event_key: tuple[str, int, int],
    instances: list[dict[str, Any]],
    row_states: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    target_guided_probe = (
        replay_probe.direct_witness_probe.critical_leaf_probe.frontier_signed_target_guided_probe
    )
    local_points = []
    for instance in instances:
        state = row_states[instance["row_key"]]
        built = state["built"]
        point = verifier.add_points(
            instance["scout"]["left"]["point"],
            instance["scout"]["right"]["point"],
            built["ainvs"],
            int(built["p"]),
        )
        local_points.append(point)

    representative_point = local_points[0] if local_points else None
    point_consistent = all(point == representative_point for point in local_points)
    emitted: list[tuple[str, dict[str, Any]]] = []
    accepted_count = 0

    for index, instance in enumerate(instances):
        state = row_states[instance["row_key"]]
        built = state["built"]
        candidate_point = representative_point if point_consistent else local_points[index]
        before = len(state["forms"])
        hit_summary = {
            "event_key": scanner_probe.compact_event_key(event_key),
            "leaf_index": int(instance["leaf_index"]),
            "scout_pos": int(instance["scout_pos"]),
            "candidate_pos": int(instance["candidate_pos"]),
            "scheduled_trial": int(instance["scheduled_trial"]),
            "original_trial": int(instance["original_trial"]),
            "accepted_relation": False,
            "relation_index": None,
            "candidate_point_source": "shared_representative" if point_consistent else "row_local",
        }
        if target_guided_probe.add_relation_if_valid(
            verifier,
            built["challenge"],
            built["base"],
            built["public"],
            built["ainvs"],
            int(built["p"]),
            int(built["order"]),
            candidate_point,
            [int(value) for value in instance["scout"]["unsigned_indices"]],
            instance["row"],
            state["forms"],
            state["seen_forms"],
            state["relations"],
        ):
            relation_event = {
                "event_key": scanner_probe.compact_event_key(event_key),
                "leaf_index": int(instance["leaf_index"]),
                "scout_pos": int(instance["scout_pos"]),
                "candidate_pos": int(instance["candidate_pos"]),
                "scheduled_trial": int(instance["scheduled_trial"]),
                "original_trial": int(instance["original_trial"]),
                "form_index": before,
                "form": state["forms"][-1],
                "candidate_point_source": hit_summary["candidate_point_source"],
            }
            state["relation_events"].append(relation_event)
            emitted.append((instance["row_key"], relation_event))
            hit_summary["accepted_relation"] = True
            hit_summary["relation_index"] = len(state["relation_events"]) - 1
            accepted_count += 1
        state["hit_events"].append(hit_summary)

    unique_point_count = len({json.dumps(compact_point(point), sort_keys=True) for point in local_points})
    return (
        {
            "event_key": scanner_probe.compact_event_key(event_key),
            "fanout": len(instances),
            "point_consistent": point_consistent,
            "unique_candidate_point_count": unique_point_count,
            "planned_first_pass_ops": 1 if point_consistent else len(instances),
            "audit_candidate_point_ops": len(instances),
            "candidate_point_reused": bool(point_consistent and len(instances) > 1),
            "candidate_point_reuse_saved_ops": (
                max(0, len(instances) - 1) if point_consistent else 0
            ),
            "accepted_relation_count": accepted_count,
            "representative_candidate_point": compact_point(representative_point),
            "row_instances": [
                {
                    "row_key": instance["row_key"],
                    "scheduled_trial": int(instance["scheduled_trial"]),
                    "candidate_pos": int(instance["candidate_pos"]),
                }
                for instance in instances
            ],
        },
        emitted,
    )


def compact_row_scan(
    verifier: Any,
    row_key: str,
    state: dict[str, Any],
    event_limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    built = state["built"]
    local_args = state["local_args"]
    leaves = state["leaves"]
    cover = state["cover"]
    relation_events = state["relation_events"]
    derived = replay_probe.dependency_circuit_probe.public_derive_from_events(
        verifier,
        relation_events,
        int(built["order"]),
        built["base"],
        built["public"],
        built["ainvs"],
        int(built["p"]),
    )
    costs = replay_probe.direct_witness_probe.preassociation_filter_probe.prefilter_costs(
        built,
        len(leaves),
        len(cover["selected_hit_root_values"]),
        len(state["hit_events"]),
        int(local_args.row_factor),
        int(local_args.product_factor),
    )
    preassociation_ops = int(costs.get("preassociation_filter_ops") or 0)
    nonshared_base_ops = (
        preassociation_ops
        - len(leaves)
        - len(cover["selected_hit_root_values"])
        - (2 * len(state["hit_events"]))
    )
    scan = {
        "selected_leaf_indices": sorted(leaves),
        "selected_leaf_signature": scanner_probe.leaf_signature(leaves),
        "selected_leaf_count": len(leaves),
        "selected_hit_roots": len(cover["selected_hit_root_values"]),
        "selected_hit_root_values": cover["selected_hit_root_values"],
        "selected_hit_events": len(state["hit_events"]),
        "hit_event_summaries": state["hit_events"],
        "candidate_verifications": len(state["hit_events"]),
        "relation_count": len(relation_events),
        "rank": int(derived.get("rank") or 0),
        "row_public_key_verified": bool(derived.get("public_key_verified")),
        "derived_secret": derived.get("derived_secret"),
        "preassociation_filter_ops": preassociation_ops,
        "preassociation_filter_ops_over_rho": costs.get("preassociation_filter_ops_over_rho"),
        "nonshared_base_ops": nonshared_base_ops,
        "relation_events": relation_events,
        "event_summaries": [
            replay_probe.dependency_circuit_probe.event_summary(index, event, int(built["order"]))
            for index, event in enumerate(relation_events)
        ],
    }
    compact = replay_probe.compact_scan(row_key, leaves, scan, int(built["order"]), event_limit)
    compact["nonshared_base_ops"] = nonshared_base_ops
    return scan, compact


def run_candidate_point_probe(
    verifier: Any,
    row_leaves: dict[str, set[int]],
    allowed_by_row: dict[str, set[tuple[str, int, int]]],
    expected_instances: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    event_limit: int,
) -> dict[str, Any]:
    row_states = {
        row_key: prepare_row_state(row_key, row_leaves[row_key], allowed_by_row.get(row_key, set()), context)
        for row_key, context in contexts.items()
        if row_key in row_leaves
    }
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for state in row_states.values():
        for event_key, instances in state["keyed_instances"].items():
            groups[event_key].extend(instances)

    group_records = []
    row_events: list[tuple[str, dict[str, Any]]] = []
    for event_key in sorted(groups, key=scanner_probe.event_sort_key):
        group_record, emitted = emit_group_relations(verifier, event_key, groups[event_key], row_states)
        group_records.append(group_record)
        row_events.extend(emitted)

    rows = []
    row_scans = []
    challenge_seeds: set[str] = set()
    generic_rho_steps = 0
    baseline_ops = 0
    nonshared_base_ops = 0
    observed_instances = []
    for row_key in sorted(row_states):
        state = row_states[row_key]
        built = state["built"]
        local_args = state["local_args"]
        scan, compact = compact_row_scan(verifier, row_key, state, event_limit)
        challenge_seed = str(built.get("challenge_seed") or local_args.challenge_seed)
        challenge_seeds.add(challenge_seed)
        generic_rho_steps = max(generic_rho_steps, int(built.get("generic_rho_steps") or 0))
        baseline_ops += int(scan.get("preassociation_filter_ops") or 0)
        nonshared_base_ops += int(scan.get("nonshared_base_ops") or 0)
        row_scans.append((row_key, scan))
        rows.append(
            {
                "row_key": row_key,
                "challenge_seed": challenge_seed,
                "generic_rho_steps": int(built.get("generic_rho_steps") or 0),
                "scan": compact,
            }
        )
        for event in state["hit_events"]:
            observed_instances.append(
                {
                    "row_key": row_key,
                    "event_key": event["event_key"],
                    "scheduled_trial": int(event["scheduled_trial"]),
                    "candidate_pos": int(event["candidate_pos"]),
                    "accepted_relation": bool(event["accepted_relation"]),
                    "relation_index": event["relation_index"],
                }
            )

    derive = replay_probe.derive_from_events(verifier, row_events, contexts)
    baseline_leaf_ops, leaf_ops = scanner_probe.shared_leaf_ops(row_leaves)
    baseline_root_ops, hit_root_ops = scanner_probe.shared_hit_root_ops(row_scans)
    planned_first_pass_ops = sum(as_int(group["planned_first_pass_ops"]) for group in group_records)
    audit_candidate_point_ops = sum(as_int(group["audit_candidate_point_ops"]) for group in group_records)
    second_event_pass_ops = len(observed_instances)
    fused_ops = (
        nonshared_base_ops
        + leaf_ops
        + hit_root_ops
        + planned_first_pass_ops
        + second_event_pass_ops
    )
    expected_by_key = expected_instance_lookup(expected_instances)
    observed_by_key = expected_instance_lookup(observed_instances)
    missing_instances = [
        expected_by_key[key] for key in sorted(set(expected_by_key) - set(observed_by_key), key=str)
    ]
    unexpected_instances = [
        observed_by_key[key] for key in sorted(set(observed_by_key) - set(expected_by_key), key=str)
    ]
    acceptance_mismatches = []
    for key in sorted(set(expected_by_key) & set(observed_by_key), key=str):
        expected = expected_by_key[key]
        observed = observed_by_key[key]
        if bool(expected.get("accepted_relation")) != bool(observed.get("accepted_relation")):
            acceptance_mismatches.append(
                {
                    "row_key": expected["row_key"],
                    "event_key": expected["event_key"],
                    "scheduled_trial": expected["scheduled_trial"],
                    "expected_accepted_relation": bool(expected.get("accepted_relation")),
                    "observed_accepted_relation": bool(observed.get("accepted_relation")),
                }
            )

    point_conflicts = [group for group in group_records if not bool(group.get("point_consistent"))]
    reused_groups = [
        group for group in group_records if bool(group.get("candidate_point_reused"))
    ]
    return {
        "selected_row_count": len(row_leaves),
        "materialized_row_count": len(rows),
        "selected_leaf_count": sum(len(leaves) for leaves in row_leaves.values()),
        "challenge_seeds": sorted(challenge_seeds),
        "baseline_ops": baseline_ops,
        "generic_rho_steps": generic_rho_steps,
        "baseline_ops_over_rho": clean_ratio(baseline_ops, generic_rho_steps),
        "candidate_point_group_count": len(group_records),
        "candidate_point_consistent_group_count": len(group_records) - len(point_conflicts),
        "candidate_point_conflict_count": len(point_conflicts),
        "candidate_point_reused_group_count": len(reused_groups),
        "candidate_point_reuse_saved_ops": sum(
            as_int(group.get("candidate_point_reuse_saved_ops")) for group in group_records
        ),
        "audit_candidate_point_ops": audit_candidate_point_ops,
        "relation_event_instance_count": len(observed_instances),
        "accepted_instance_count": sum(
            1 for instance in observed_instances if bool(instance.get("accepted_relation"))
        ),
        "work_counters": {
            "nonshared_base_ops": nonshared_base_ops,
            "baseline_leaf_ops": baseline_leaf_ops,
            "shared_leaf_ops": leaf_ops,
            "baseline_hit_root_ops": baseline_root_ops,
            "shared_hit_root_ops": hit_root_ops,
            "shared_first_event_pass_ops": planned_first_pass_ops,
            "second_event_pass_ops": second_event_pass_ops,
            "fused_ops": fused_ops,
            "generic_rho_steps": generic_rho_steps,
            "fused_ops_over_rho": clean_ratio(fused_ops, generic_rho_steps),
            "fused_below_rho": bool(generic_rho_steps and fused_ops < generic_rho_steps),
            "first_pass_saved_ops": max(0, second_event_pass_ops - planned_first_pass_ops),
        },
        "worklist_observation": {
            "expected_instance_count": len(expected_instances),
            "observed_instance_count": len(observed_instances),
            "missing_instance_count": len(missing_instances),
            "unexpected_instance_count": len(unexpected_instances),
            "acceptance_mismatch_count": len(acceptance_mismatches),
            "missing_instances": missing_instances,
            "unexpected_instances": unexpected_instances,
            "acceptance_mismatches": acceptance_mismatches,
        },
        **derive,
        "candidate_point_groups": group_records,
        "candidate_point_conflicts": point_conflicts,
        "rows": rows,
        "audit_note": (
            "The script recomputes candidate points for all row instances to "
            "prove equality.  Planned fused counters charge one first-pass "
            "candidate-point op only for point-consistent event-key groups."
        ),
    }


def prototype_record(
    verifier: Any,
    records: list[dict[str, Any]],
    worklist_record: dict[str, Any],
    charged_record: dict[str, Any] | None,
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]],
    bundle_cache: dict[tuple[Any, ...], dict[str, Any]],
    event_summary_limit: int,
) -> dict[str, Any]:
    source_name = str(worklist_record.get("source_name") or "")
    if charged_record is None:
        return {"source_name": source_name, "prototype_status": "missing_charged_locator_record"}
    source_path = scanner_probe.resolve_path(charged_record.get("source_path"))
    if not source_path.exists():
        return {
            "source_name": source_name,
            "prototype_status": "missing_decomposition_source",
            "source_path": str(source_path),
        }
    candidate_artifact, params, _decomposition = scanner_probe.candidate_source_parameters(source_path)
    if not params:
        return {
            "source_name": source_name,
            "prototype_status": "missing_candidate_source_parameters",
            "source_path": str(source_path),
            "candidate_artifact": str(candidate_artifact) if candidate_artifact is not None else None,
        }

    row_leaves = scanner_probe.row_leaves_from_worklist(worklist_record)
    allowed_by_row, expected_instances = scanner_probe.expected_worklist_instances(worklist_record)
    public_group = worklist_record.get("public_group_key") or {}
    case = {
        "target": public_group.get("target"),
        "transfer_index": as_int(public_group.get("transfer_index")),
        "top_k": as_int(public_group.get("top_k")),
    }
    bundle = scanner_probe.build_input_bundle(params, event_summary_limit, bundle_cache)
    contexts, errors = replay_probe.materialize_contexts(
        verifier,
        records,
        bundle["config_source"],
        bundle["specs_by_target"],
        case,
        sorted(row_leaves),
        bundle["args"],
        context_cache,
    )
    prototype = run_candidate_point_probe(
        verifier,
        row_leaves,
        allowed_by_row,
        expected_instances,
        contexts,
        event_summary_limit,
    )
    counters = prototype.get("work_counters") or {}
    observation = prototype.get("worklist_observation") or {}
    recorded_replay = worklist_record.get("replay") or {}
    worklist_counters = worklist_record.get("fused_worklist") or {}
    matches_secret = (
        prototype.get("derived_secret") is not None
        and recorded_replay.get("derived_secret") is not None
        and as_int(prototype.get("derived_secret")) == as_int(recorded_replay.get("derived_secret"))
    )
    matches_fused_ops = as_int(counters.get("fused_ops")) == as_int(worklist_counters.get("fused_ops"))
    matches_unique_events = (
        as_int(counters.get("shared_first_event_pass_ops"))
        == as_int(worklist_counters.get("shared_first_event_pass_ops"))
    )
    matches_second_pass = (
        as_int(counters.get("second_event_pass_ops"))
        == as_int(worklist_counters.get("second_event_pass_ops"))
    )
    consumed = bool(
        not errors
        and as_int(prototype.get("candidate_point_conflict_count")) == 0
        and matches_secret
        and bool(prototype.get("public_key_verified"))
        and matches_fused_ops
        and matches_unique_events
        and matches_second_pass
        and as_int(observation.get("missing_instance_count")) == 0
        and as_int(observation.get("unexpected_instance_count")) == 0
        and as_int(observation.get("acceptance_mismatch_count")) == 0
    )
    return {
        "source_name": source_name,
        "rule": worklist_record.get("rule"),
        "source_path": str(source_path),
        "candidate_artifact": str(candidate_artifact) if candidate_artifact is not None else None,
        "public_group_key": public_group,
        "selected_row_keys": sorted(row_leaves),
        "prototype_status": (
            "candidate_point_fused_reuse_verified"
            if consumed
            else "candidate_point_fused_reuse_failed_check"
        ),
        "context_error_count": len(errors),
        "context_errors": errors,
        "candidate_point_reuse_replay": prototype,
        "matches": {
            "matches_recorded_secret": matches_secret,
            "recorded_replay_public_key_verified": bool(recorded_replay.get("public_key_verified")),
            "matches_worklist_fused_ops": matches_fused_ops,
            "matches_worklist_unique_first_pass_events": matches_unique_events,
            "matches_worklist_second_pass_instances": matches_second_pass,
            "matches_worklist_relation_count": (
                as_int(prototype.get("relation_count")) == as_int(recorded_replay.get("relation_count"))
            ),
            "matches_worklist_rank": (
                as_int(prototype.get("rank")) == as_int(recorded_replay.get("rank"))
            ),
        },
        "event_reuse_target": bool(
            counters.get("fused_below_rho") and as_int(counters.get("first_pass_saved_ops")) > 0
        ),
        "implementation_boundary": (
            "Audits candidate-point equality and emits relations with shared "
            "representative points for point-consistent event keys.  This is a "
            "fused first-pass prototype, not yet an optimized standalone FFE kernel."
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [
        record
        for record in records
        if record.get("prototype_status") == "candidate_point_fused_reuse_verified"
    ]
    derived = [
        record
        for record in records
        if bool((record.get("candidate_point_reuse_replay") or {}).get("derived"))
    ]
    matched_secret = [
        record for record in records if bool((record.get("matches") or {}).get("matches_recorded_secret"))
    ]
    matched_ops = [
        record for record in records if bool((record.get("matches") or {}).get("matches_worklist_fused_ops"))
    ]
    conflicts = [
        record
        for record in records
        if as_int((record.get("candidate_point_reuse_replay") or {}).get("candidate_point_conflict_count")) > 0
    ]
    below = [
        record
        for record in verified
        if bool(((record.get("candidate_point_reuse_replay") or {}).get("work_counters") or {}).get("fused_below_rho"))
    ]
    event_reuse = [record for record in below if bool(record.get("event_reuse_target"))]
    ratios = [
        ((record.get("candidate_point_reuse_replay") or {}).get("work_counters") or {}).get(
            "fused_ops_over_rho"
        )
        for record in below
        if ((record.get("candidate_point_reuse_replay") or {}).get("work_counters") or {}).get(
            "fused_ops_over_rho"
        )
        is not None
    ]
    return {
        "record_count": len(records),
        "candidate_point_reuse_verified_count": len(verified),
        "candidate_point_conflict_record_count": len(conflicts),
        "candidate_point_derived_count": len(derived),
        "candidate_point_secret_match_count": len(matched_secret),
        "candidate_point_fused_ops_match_count": len(matched_ops),
        "candidate_point_below_rho_count": len(below),
        "candidate_point_event_reuse_below_rho_count": len(event_reuse),
        "mean_candidate_point_below_rho_ops_over_rho": (
            round(mean([float(value) for value in ratios]), 8) if ratios else None
        ),
        "interpretation": (
            "Duplicate public event keys now have an audited candidate-point "
            "reuse boundary.  Remaining work is implementing the same plan as "
            "a lower-level optimized FFE/summation-polynomial pass."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist-source", type=Path, required=True)
    parser.add_argument("--charged-source", type=Path, required=True)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--event-summary-limit", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    worklist_source = load_json(args.worklist_source)
    charged_source = load_json(args.charged_source)
    charged_by_source = scanner_probe.charged_records_by_source(charged_source)
    wanted = set(args.source_name or [])
    verifier = replay_probe.relation_probe.load_verifier_module()
    records = verifier.load_records()
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    bundle_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    prototype_records = []
    for record in worklist_source.get("records") or []:
        if not isinstance(record, dict):
            continue
        source_name = str(record.get("source_name") or "")
        if wanted and source_name not in wanted:
            continue
        prototype_records.append(
            prototype_record(
                verifier,
                records,
                record,
                charged_by_source.get(source_name),
                context_cache,
                bundle_cache,
                int(args.event_summary_limit),
            )
        )

    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_candidate_point_probe_v1",
        "method": "audited_candidate_point_reuse_for_public_event_key_groups",
        "parameters": {
            "worklist_source": str(args.worklist_source),
            "charged_source": str(args.charged_source),
            "source_names": sorted(wanted),
            "event_summary_limit": int(args.event_summary_limit),
            "charged_source_usage": "locator_only_for_decomposition_source_paths",
        },
        "summary": summarize(prototype_records),
        "records": sorted(
            prototype_records,
            key=lambda record: (
                not bool(record.get("event_reuse_target")),
                not bool(((record.get("candidate_point_reuse_replay") or {}).get("work_counters") or {}).get("fused_below_rho")),
                str(record.get("source_name") or ""),
            ),
        ),
        "non_claims": [
            "This audits and prototypes fused first-pass candidate-point reuse; it is not a standalone optimized FFE kernel.",
            "The charged source is used only to locate decomposition artifacts; charged work counters are not consumed.",
            "Audit recomputation of candidate points is reported separately from planned fused-kernel work counters.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
