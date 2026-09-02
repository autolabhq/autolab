#!/usr/bin/env python3
"""Execute repeated-coordinate fused worklists with one candidate point per key.

The candidate-point audit proved that duplicate public event-key groups have
identical elliptic-curve candidate points.  This executor removes that audit
recomputation from the planned hot path: for each public event key it computes
one representative candidate point, reuses it for every row instance in the
group, emits row-specific relation forms, and checks the usual secret/counter
gates.

This is still a Python verifier-path executor rather than a hand-optimized FFE
kernel.  Its purpose is to prove the executable control flow and work counters
the lower-level summation-polynomial/FFE implementation must preserve.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_fused_candidate_point_probe as candidate_point_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_executor.json"

scanner_probe = candidate_point_probe.scanner_probe
replay_probe = candidate_point_probe.replay_probe


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any, default: int = 0) -> int:
    return scanner_probe.as_int(value, default)


def clean_ratio(ops: int, rho: int) -> float | None:
    return scanner_probe.clean_ratio(ops, rho)


def compact_point(point: Any) -> Any:
    return candidate_point_probe.compact_point(point)


def instance_key(instance: dict[str, Any]) -> tuple[str, tuple[str, int, int], int]:
    return scanner_probe.instance_key(instance)


def sorted_group_instances(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        instances,
        key=lambda item: (
            str(item.get("row_key") or ""),
            as_int(item.get("scheduled_trial"), -1),
            as_int(item.get("candidate_pos"), -1),
        ),
    )


def emit_group_relations_hot_path(
    verifier: Any,
    event_key: tuple[str, int, int],
    instances: list[dict[str, Any]],
    row_states: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    target_guided_probe = (
        replay_probe.direct_witness_probe.critical_leaf_probe.frontier_signed_target_guided_probe
    )
    ordered_instances = sorted_group_instances(instances)
    representative = ordered_instances[0]
    representative_state = row_states[representative["row_key"]]
    representative_built = representative_state["built"]
    representative_point = verifier.add_points(
        representative["scout"]["left"]["point"],
        representative["scout"]["right"]["point"],
        representative_built["ainvs"],
        int(representative_built["p"]),
    )
    emitted: list[tuple[str, dict[str, Any]]] = []
    accepted_count = 0

    for instance in ordered_instances:
        state = row_states[instance["row_key"]]
        built = state["built"]
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
            "candidate_point_source": "hot_path_representative",
            "representative_row_key": representative["row_key"],
            "representative_scheduled_trial": int(representative["scheduled_trial"]),
        }
        if target_guided_probe.add_relation_if_valid(
            verifier,
            built["challenge"],
            built["base"],
            built["public"],
            built["ainvs"],
            int(built["p"]),
            int(built["order"]),
            representative_point,
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
                "candidate_point_source": "hot_path_representative",
                "representative_row_key": representative["row_key"],
                "representative_scheduled_trial": int(representative["scheduled_trial"]),
            }
            state["relation_events"].append(relation_event)
            emitted.append((instance["row_key"], relation_event))
            hit_summary["accepted_relation"] = True
            hit_summary["relation_index"] = len(state["relation_events"]) - 1
            accepted_count += 1
        state["hit_events"].append(hit_summary)

    return (
        {
            "event_key": scanner_probe.compact_event_key(event_key),
            "fanout": len(ordered_instances),
            "candidate_point_ops": 1,
            "candidate_point_reused": len(ordered_instances) > 1,
            "candidate_point_reuse_saved_ops": max(0, len(ordered_instances) - 1),
            "accepted_relation_count": accepted_count,
            "representative_candidate_point": compact_point(representative_point),
            "representative_row_key": representative["row_key"],
            "representative_scheduled_trial": int(representative["scheduled_trial"]),
            "row_instances": [
                {
                    "row_key": instance["row_key"],
                    "scheduled_trial": int(instance["scheduled_trial"]),
                    "candidate_pos": int(instance["candidate_pos"]),
                }
                for instance in ordered_instances
            ],
        },
        emitted,
    )


def execute_fused_hot_path(
    verifier: Any,
    row_leaves: dict[str, set[int]],
    allowed_by_row: dict[str, set[tuple[str, int, int]]],
    expected_instances: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    event_limit: int,
) -> dict[str, Any]:
    row_states = {
        row_key: candidate_point_probe.prepare_row_state(
            row_key,
            row_leaves[row_key],
            allowed_by_row.get(row_key, set()),
            context,
        )
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
        group_record, emitted = emit_group_relations_hot_path(
            verifier,
            event_key,
            groups[event_key],
            row_states,
        )
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
        scan, compact = candidate_point_probe.compact_row_scan(
            verifier,
            row_key,
            state,
            event_limit,
        )
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
    first_pass_ops = sum(as_int(group["candidate_point_ops"]) for group in group_records)
    second_event_pass_ops = len(observed_instances)
    fused_ops = (
        nonshared_base_ops
        + leaf_ops
        + hit_root_ops
        + first_pass_ops
        + second_event_pass_ops
    )

    expected_by_key = {instance_key(instance): instance for instance in expected_instances}
    observed_by_key = {instance_key(instance): instance for instance in observed_instances}
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

    reused_groups = [group for group in group_records if bool(group.get("candidate_point_reused"))]
    return {
        "selected_row_count": len(row_leaves),
        "materialized_row_count": len(rows),
        "selected_leaf_count": sum(len(leaves) for leaves in row_leaves.values()),
        "challenge_seeds": sorted(challenge_seeds),
        "baseline_ops": baseline_ops,
        "generic_rho_steps": generic_rho_steps,
        "baseline_ops_over_rho": clean_ratio(baseline_ops, generic_rho_steps),
        "candidate_point_group_count": len(group_records),
        "candidate_point_ops": first_pass_ops,
        "candidate_point_reused_group_count": len(reused_groups),
        "candidate_point_reuse_saved_ops": sum(
            as_int(group.get("candidate_point_reuse_saved_ops")) for group in group_records
        ),
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
            "shared_first_event_pass_ops": first_pass_ops,
            "second_event_pass_ops": second_event_pass_ops,
            "fused_ops": fused_ops,
            "generic_rho_steps": generic_rho_steps,
            "fused_ops_over_rho": clean_ratio(fused_ops, generic_rho_steps),
            "fused_below_rho": bool(generic_rho_steps and fused_ops < generic_rho_steps),
            "first_pass_saved_ops": max(0, second_event_pass_ops - first_pass_ops),
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
        "rows": rows,
        "hot_path_note": (
            "Computes one representative candidate point per public event key "
            "and reuses it for all row-specific relation predicates.  Candidate "
            "point equality is not recomputed here; use the candidate-point "
            "audit artifact as the oracle for conflict checks."
        ),
    }


def audit_lookup(audit_source: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not audit_source:
        return {}
    return {
        str(record.get("source_name") or ""): record
        for record in audit_source.get("records") or []
        if isinstance(record, dict) and record.get("source_name") is not None
    }


def audit_record_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    replay = record.get("candidate_point_reuse_replay") or {}
    counters = replay.get("work_counters") or {}
    return {
        "prototype_status": record.get("prototype_status"),
        "candidate_point_conflict_count": as_int(replay.get("candidate_point_conflict_count")),
        "candidate_point_group_count": as_int(replay.get("candidate_point_group_count")),
        "candidate_point_reused_group_count": as_int(replay.get("candidate_point_reused_group_count")),
        "candidate_point_reuse_saved_ops": as_int(replay.get("candidate_point_reuse_saved_ops")),
        "derived_secret": replay.get("derived_secret"),
        "rank": as_int(replay.get("rank")),
        "fused_ops": as_int(counters.get("fused_ops")),
        "fused_ops_over_rho": counters.get("fused_ops_over_rho"),
    }


def executor_record(
    verifier: Any,
    records: list[dict[str, Any]],
    worklist_record: dict[str, Any],
    charged_record: dict[str, Any] | None,
    audit_record: dict[str, Any] | None,
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]],
    bundle_cache: dict[tuple[Any, ...], dict[str, Any]],
    event_summary_limit: int,
) -> dict[str, Any]:
    source_name = str(worklist_record.get("source_name") or "")
    if charged_record is None:
        return {"source_name": source_name, "executor_status": "missing_charged_locator_record"}
    source_path = scanner_probe.resolve_path(charged_record.get("source_path"))
    if not source_path.exists():
        return {
            "source_name": source_name,
            "executor_status": "missing_decomposition_source",
            "source_path": str(source_path),
        }
    candidate_artifact, params, _decomposition = scanner_probe.candidate_source_parameters(source_path)
    if not params:
        return {
            "source_name": source_name,
            "executor_status": "missing_candidate_source_parameters",
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
    execution = execute_fused_hot_path(
        verifier,
        row_leaves,
        allowed_by_row,
        expected_instances,
        contexts,
        event_summary_limit,
    )
    counters = execution.get("work_counters") or {}
    observation = execution.get("worklist_observation") or {}
    recorded_replay = worklist_record.get("replay") or {}
    worklist_counters = worklist_record.get("fused_worklist") or {}
    audit_summary = audit_record_summary(audit_record)
    matches_secret = (
        execution.get("derived_secret") is not None
        and recorded_replay.get("derived_secret") is not None
        and as_int(execution.get("derived_secret")) == as_int(recorded_replay.get("derived_secret"))
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
    matches_audit = None
    if audit_summary is not None:
        matches_audit = bool(
            as_int(execution.get("candidate_point_group_count")) == as_int(audit_summary.get("candidate_point_group_count"))
            and as_int(execution.get("candidate_point_reused_group_count")) == as_int(audit_summary.get("candidate_point_reused_group_count"))
            and as_int(execution.get("candidate_point_reuse_saved_ops")) == as_int(audit_summary.get("candidate_point_reuse_saved_ops"))
            and as_int(counters.get("fused_ops")) == as_int(audit_summary.get("fused_ops"))
            and execution.get("derived_secret") == audit_summary.get("derived_secret")
        )
    consumed = bool(
        not errors
        and matches_secret
        and bool(execution.get("public_key_verified"))
        and matches_fused_ops
        and matches_unique_events
        and matches_second_pass
        and as_int(observation.get("missing_instance_count")) == 0
        and as_int(observation.get("unexpected_instance_count")) == 0
        and as_int(observation.get("acceptance_mismatch_count")) == 0
        and (matches_audit is not False)
    )
    return {
        "source_name": source_name,
        "rule": worklist_record.get("rule"),
        "source_path": str(source_path),
        "candidate_artifact": str(candidate_artifact) if candidate_artifact is not None else None,
        "public_group_key": public_group,
        "selected_row_keys": sorted(row_leaves),
        "executor_status": (
            "fused_hot_path_candidate_point_executor_verified"
            if consumed
            else "fused_hot_path_candidate_point_executor_failed_check"
        ),
        "context_error_count": len(errors),
        "context_errors": errors,
        "fused_hot_path_replay": execution,
        "candidate_point_audit_oracle": audit_summary,
        "matches": {
            "matches_recorded_secret": matches_secret,
            "recorded_replay_public_key_verified": bool(recorded_replay.get("public_key_verified")),
            "matches_worklist_fused_ops": matches_fused_ops,
            "matches_worklist_unique_first_pass_events": matches_unique_events,
            "matches_worklist_second_pass_instances": matches_second_pass,
            "matches_worklist_relation_count": (
                as_int(execution.get("relation_count")) == as_int(recorded_replay.get("relation_count"))
            ),
            "matches_worklist_rank": (
                as_int(execution.get("rank")) == as_int(recorded_replay.get("rank"))
            ),
            "matches_candidate_point_audit_oracle": matches_audit,
        },
        "event_reuse_target": bool(
            counters.get("fused_below_rho") and as_int(counters.get("first_pass_saved_ops")) > 0
        ),
        "implementation_boundary": (
            "Executes one candidate-point computation per public event key and "
            "reuses it for row-specific relation predicates.  It still uses "
            "the Python row-context/verifier path rather than a standalone "
            "optimized FFE kernel."
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [
        record
        for record in records
        if record.get("executor_status") == "fused_hot_path_candidate_point_executor_verified"
    ]
    derived = [
        record
        for record in records
        if bool((record.get("fused_hot_path_replay") or {}).get("derived"))
    ]
    matched_secret = [
        record for record in records if bool((record.get("matches") or {}).get("matches_recorded_secret"))
    ]
    matched_ops = [
        record for record in records if bool((record.get("matches") or {}).get("matches_worklist_fused_ops"))
    ]
    audit_matched = [
        record
        for record in records
        if (record.get("matches") or {}).get("matches_candidate_point_audit_oracle") is True
    ]
    below = [
        record
        for record in verified
        if bool(((record.get("fused_hot_path_replay") or {}).get("work_counters") or {}).get("fused_below_rho"))
    ]
    event_reuse = [record for record in below if bool(record.get("event_reuse_target"))]
    ratios = [
        ((record.get("fused_hot_path_replay") or {}).get("work_counters") or {}).get(
            "fused_ops_over_rho"
        )
        for record in below
        if ((record.get("fused_hot_path_replay") or {}).get("work_counters") or {}).get(
            "fused_ops_over_rho"
        )
        is not None
    ]
    return {
        "record_count": len(records),
        "fused_hot_path_verified_count": len(verified),
        "fused_hot_path_derived_count": len(derived),
        "fused_hot_path_secret_match_count": len(matched_secret),
        "fused_hot_path_fused_ops_match_count": len(matched_ops),
        "fused_hot_path_audit_oracle_match_count": len(audit_matched),
        "fused_hot_path_below_rho_count": len(below),
        "fused_hot_path_event_reuse_below_rho_count": len(event_reuse),
        "mean_fused_hot_path_below_rho_ops_over_rho": (
            round(mean([float(value) for value in ratios]), 8) if ratios else None
        ),
        "interpretation": (
            "The hot path now computes one candidate point per public event key "
            "and reuses it for row-specific relation emission.  The remaining "
            "promotion is a lower-level optimized FFE/summation-polynomial implementation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist-source", type=Path, required=True)
    parser.add_argument("--charged-source", type=Path, required=True)
    parser.add_argument("--candidate-point-audit-source", type=Path)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--event-summary-limit", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    worklist_source = load_json(args.worklist_source)
    charged_source = load_json(args.charged_source)
    audit_source = (
        load_json(args.candidate_point_audit_source)
        if args.candidate_point_audit_source is not None
        else None
    )
    charged_by_source = scanner_probe.charged_records_by_source(charged_source)
    audit_by_source = audit_lookup(audit_source)
    wanted = set(args.source_name or [])
    verifier = replay_probe.relation_probe.load_verifier_module()
    records = verifier.load_records()
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    bundle_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    executor_records = []
    for record in worklist_source.get("records") or []:
        if not isinstance(record, dict):
            continue
        source_name = str(record.get("source_name") or "")
        if wanted and source_name not in wanted:
            continue
        executor_records.append(
            executor_record(
                verifier,
                records,
                record,
                charged_by_source.get(source_name),
                audit_by_source.get(source_name),
                context_cache,
                bundle_cache,
                int(args.event_summary_limit),
            )
        )

    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_kernel_executor_probe_v1",
        "method": "hot_path_one_candidate_point_per_public_event_key",
        "parameters": {
            "worklist_source": str(args.worklist_source),
            "charged_source": str(args.charged_source),
            "candidate_point_audit_source": (
                str(args.candidate_point_audit_source)
                if args.candidate_point_audit_source is not None
                else None
            ),
            "source_names": sorted(wanted),
            "event_summary_limit": int(args.event_summary_limit),
            "charged_source_usage": "locator_only_for_decomposition_source_paths",
            "audit_source_usage": "post_run_oracle_only_not_hot_path_input",
        },
        "summary": summarize(executor_records),
        "records": sorted(
            executor_records,
            key=lambda record: (
                not bool(record.get("event_reuse_target")),
                not bool(((record.get("fused_hot_path_replay") or {}).get("work_counters") or {}).get("fused_below_rho")),
                str(record.get("source_name") or ""),
            ),
        ),
        "non_claims": [
            "This executes the fused candidate-point hot path in Python; it is not a standalone optimized FFE kernel.",
            "The charged source is used only to locate decomposition artifacts; charged work counters are not consumed.",
            "The candidate-point audit source, when provided, is used only as a post-run oracle and not as a hot-path input.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
