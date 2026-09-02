#!/usr/bin/env python3
"""Consume fused repeated-coordinate worklists through the verifier scanner.

The worklist replay probe proved that accepted relation summaries are enough
to derive the secret from the public fused worklist.  This scanner probe moves
one layer lower: it rematerializes the row contexts, uses the public worklist
event keys to drive x-match candidate checks, emits relation forms through the
existing verifier predicate, and recomputes fused work counters from the actual
observed scanner events.

This is still not an optimized low-level FFE kernel.  It uses the existing row
context builder and verifier relation predicate.  The important boundary is
that relation forms are emitted by this run rather than read from accepted
relation summaries.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_cost_decomposition_probe as cost_decomp
import ffe_single_hit_root_relation_replay_probe as replay_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_worklist_scanner.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(raw: Any) -> Path:
    path = Path(str(raw))
    if path.exists():
        return path
    candidate = WORKTREE_ROOT / path
    if candidate.exists():
        return candidate
    return path


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_ratio(ops: int, rho: int) -> float | None:
    return round(ops / rho, 8) if rho else None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def leaf_signature(leaves: set[int] | list[int]) -> str:
    return "|".join(str(int(leaf)) for leaf in sorted(leaves))


def parse_leaf_signature(raw: Any) -> set[int]:
    text = str(raw or "")
    if not text:
        return set()
    return {as_int(part, -1) for part in text.split("|") if part != "" and as_int(part, -1) >= 0}


def normalized_event_key(raw: Any) -> tuple[str, int, int]:
    values = list(raw or [])
    if len(values) != 3:
        return ("", -1, -1)
    return (str(values[0]), as_int(values[1], -1), as_int(values[2], -1))


def compact_event_key(key: tuple[str, int, int]) -> list[Any]:
    return [key[0], key[1], key[2]]


def event_sort_key(key: tuple[str, int, int]) -> tuple[str, int, int]:
    return key


def row_leaves_from_worklist(record: dict[str, Any]) -> dict[str, set[int]]:
    selected_rows = {str(row) for row in record.get("selected_row_keys") or []}
    row_leaves: dict[str, set[int]] = {row_key: set() for row_key in selected_rows}
    for item in record.get("leaf_worklist") or []:
        if not isinstance(item, dict):
            continue
        leaves = parse_leaf_signature(item.get("leaf_signature"))
        for row_key in item.get("row_keys") or []:
            row_key = str(row_key)
            if selected_rows and row_key not in selected_rows:
                continue
            row_leaves.setdefault(row_key, set()).update(leaves)

    fallback_signatures = [
        parse_leaf_signature(value)
        for value in (record.get("public_group_key") or {}).get("leaf_signatures") or []
    ]
    fallback = fallback_signatures[0] if len(fallback_signatures) == 1 else set()
    for row_key in list(row_leaves):
        if not row_leaves[row_key] and fallback:
            row_leaves[row_key] = set(fallback)
    return {row_key: leaves for row_key, leaves in sorted(row_leaves.items()) if leaves}


def expected_worklist_instances(
    record: dict[str, Any],
) -> tuple[dict[str, set[tuple[str, int, int]]], list[dict[str, Any]]]:
    allowed_by_row: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    instances: list[dict[str, Any]] = []
    for event in record.get("event_worklist") or []:
        if not isinstance(event, dict):
            continue
        key = normalized_event_key(event.get("event_key"))
        for row_instance in event.get("row_instances") or []:
            if not isinstance(row_instance, dict):
                continue
            row_key = str(row_instance.get("row_key") or "")
            if not row_key:
                continue
            allowed_by_row[row_key].add(key)
            instances.append(
                {
                    "row_key": row_key,
                    "event_key": compact_event_key(key),
                    "scheduled_trial": as_int(row_instance.get("scheduled_trial"), -1),
                    "candidate_pos": as_int(row_instance.get("candidate_pos"), -1),
                    "accepted_relation": bool(row_instance.get("accepted_relation")),
                    "relation_index": row_instance.get("relation_index"),
                }
            )
    return dict(allowed_by_row), instances


def instance_key(instance: dict[str, Any]) -> tuple[str, tuple[str, int, int], int]:
    return (
        str(instance.get("row_key") or ""),
        normalized_event_key(instance.get("event_key")),
        as_int(instance.get("scheduled_trial"), -1),
    )


def candidate_source_parameters(source_path: Path) -> tuple[Path | None, dict[str, Any], dict[str, Any]]:
    decomposition = load_json(source_path)
    sources = (decomposition.get("parameters") or {}).get("candidate_sources") or []
    if not sources:
        return None, {}, decomposition
    first_source = sources[0]
    artifact = first_source.get("artifact") if isinstance(first_source, dict) else None
    if artifact is None:
        return None, {}, decomposition
    artifact_path = resolve_path(artifact)
    if not artifact_path.exists():
        return artifact_path, {}, decomposition
    source = load_json(artifact_path)
    return artifact_path, source.get("parameters") or {}, decomposition


def build_input_bundle(
    params: dict[str, Any],
    event_summary_limit: int,
    bundle_cache: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    bank_path = resolve_path(params["bank_source"])
    config_path = resolve_path(params["config_source"])
    direct_path = resolve_path(params["direct_source"])
    transfer_path = resolve_path(params["transfer_source"])
    transfer_source = load_json(transfer_path)
    transfer_params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    if not isinstance(transfer_params, dict):
        transfer_params = {}
    radius = as_int(params.get("radius") or transfer_params.get("radius"), 4)
    cache_key = (str(bank_path), str(config_path), str(direct_path), radius, event_summary_limit)
    if cache_key not in bundle_cache:
        bank_source = load_json(bank_path)
        config_source = load_json(config_path)
        direct_source = load_json(direct_path)
        bundle_cache[cache_key] = {
            "bank_path": str(bank_path),
            "config_path": str(config_path),
            "direct_path": str(direct_path),
            "transfer_path": str(transfer_path),
            "radius": radius,
            "config_source": config_source,
            "specs_by_target": replay_probe.build_specs_by_target(
                bank_source,
                direct_source,
                radius,
            ),
            "args": cost_decomp.replay_args(params, event_summary_limit),
        }
    return bundle_cache[cache_key]


def collect_relation_events_with_worklist_keys(
    verifier: Any,
    built: dict[str, Any],
    cover: dict[str, Any],
    scheduled_rows: list[dict[str, Any]],
    ordered_scouts: list[dict[str, Any]],
    selected_limit: int,
    scout_to_leaf: dict[int, int],
    row_leaf_signature: str,
    allowed_keys: set[tuple[str, int, int]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_guided_probe = (
        replay_probe.direct_witness_probe.critical_leaf_probe.frontier_signed_target_guided_probe
    )
    challenge = built["challenge"]
    base = built["base"]
    public = built["public"]
    ainvs = built["ainvs"]
    p = int(built["p"])
    order = int(built["order"])
    scheduled_by_original = {int(row["original_trial"]): row for row in scheduled_rows}
    forms: list[tuple[Any, int, list[int]]] = []
    relations: list[dict[str, Any]] = []
    relation_events: list[dict[str, Any]] = []
    hit_events: list[dict[str, Any]] = []
    seen_forms: set[tuple[Any, int]] = set()
    x_matches = 0
    candidate_verifications = 0
    scanned_candidates = 0
    observed_keys: set[tuple[str, int, int]] = set()

    for candidate_pos, scout in enumerate(ordered_scouts[:selected_limit], start=1):
        scanned_candidates = candidate_pos
        scout_pos = int(scout["scout_pos"])
        leaf_index = scout_to_leaf[scout_pos]
        hit_rows = [
            scheduled_by_original[trial]
            for trial in cover["hits_by_scout"].get(scout_pos, [])
            if trial in scheduled_by_original
        ]
        keyed_rows = []
        for row in hit_rows:
            key = (row_leaf_signature, scout_pos, int(row["original_trial"]))
            if key in allowed_keys:
                keyed_rows.append((key, row))
        if not keyed_rows:
            continue

        candidate_point = verifier.add_points(scout["left"]["point"], scout["right"]["point"], ainvs, p)
        for key, row in keyed_rows:
            observed_keys.add(key)
            x_matches += 1
            candidate_verifications += 1
            before = len(forms)
            hit_summary = {
                "event_key": compact_event_key(key),
                "leaf_index": leaf_index,
                "scout_pos": scout_pos,
                "candidate_pos": candidate_pos,
                "scheduled_trial": int(row["trial"]),
                "original_trial": int(row["original_trial"]),
                "accepted_relation": False,
                "relation_index": None,
            }
            if target_guided_probe.add_relation_if_valid(
                verifier,
                challenge,
                base,
                public,
                ainvs,
                p,
                order,
                candidate_point,
                [int(index) for index in scout["unsigned_indices"]],
                row,
                forms,
                seen_forms,
                relations,
            ):
                relation_events.append(
                    {
                        "event_key": compact_event_key(key),
                        "leaf_index": leaf_index,
                        "scout_pos": scout_pos,
                        "candidate_pos": candidate_pos,
                        "scheduled_trial": int(row["trial"]),
                        "original_trial": int(row["original_trial"]),
                        "form_index": before,
                        "form": forms[-1],
                    }
                )
                hit_summary["accepted_relation"] = True
                hit_summary["relation_index"] = len(relation_events) - 1
            hit_events.append(hit_summary)
            if len(relations) >= args.max_relations:
                break
        if len(relations) >= args.max_relations:
            break

    derived = (
        replay_probe.direct_witness_probe.critical_leaf_probe.frontier_signed_dual_sieve_probe.rank_probe(
            verifier,
            forms,
            order,
        )
    )
    return {
        "forms": forms,
        "relations": relations,
        "relation_events": relation_events,
        "hit_events": hit_events,
        "x_matches": x_matches,
        "candidate_verifications": candidate_verifications,
        "scanned_candidates": scanned_candidates,
        "observed_event_keys": sorted(observed_keys, key=event_sort_key),
        "accepted_unsigned_relations": len(relations),
        "unsigned_relation_rank": int(derived["mixed_wide_relation_rank"]),
        "unsigned_relations_derive_secret": bool(derived["mixed_wide_relations_derive_secret"]),
    }


def scan_selected_with_worklist_keys(
    verifier: Any,
    built: dict[str, Any],
    components: dict[str, Any],
    selected: set[int],
    allowed_keys: set[tuple[str, int, int]],
    local_args: argparse.Namespace,
) -> dict[str, Any]:
    p = int(built["p"])
    order = int(built["order"])
    row_leaf_signature = leaf_signature(selected)
    cover = replay_probe.filtered_leaf_gcd_association_with_trace(
        built["scouts"],
        components,
        p,
        selected,
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
    scout_to_leaf = replay_probe.direct_witness_probe.critical_leaf_probe.scout_leaf_map(components)
    scan_args = argparse.Namespace(max_relations=local_args.max_relations)
    relation_scan = collect_relation_events_with_worklist_keys(
        verifier,
        built,
        cover,
        scheduled_rows,
        ordered_scouts,
        int(built["selected_limit"]),
        scout_to_leaf,
        row_leaf_signature,
        allowed_keys,
        scan_args,
    )
    derived = replay_probe.dependency_circuit_probe.public_derive_from_events(
        verifier,
        relation_scan["relation_events"],
        order,
        built["base"],
        built["public"],
        built["ainvs"],
        p,
    )
    costs = replay_probe.direct_witness_probe.preassociation_filter_probe.prefilter_costs(
        built,
        len(selected),
        len(cover["selected_hit_root_values"]),
        int(relation_scan["x_matches"]),
        int(local_args.row_factor),
        int(local_args.product_factor),
    )
    preassociation_ops = int(costs.get("preassociation_filter_ops") or 0)
    nonshared_base_ops = (
        preassociation_ops
        - len(selected)
        - len(cover["selected_hit_root_values"])
        - (2 * int(relation_scan["x_matches"]))
    )
    observed = {tuple(key) for key in relation_scan["observed_event_keys"]}
    return {
        "selected_leaf_indices": sorted(selected),
        "selected_leaf_signature": row_leaf_signature,
        "selected_leaf_count": len(selected),
        "selected_hit_roots": len(cover["selected_hit_root_values"]),
        "selected_hit_root_values": cover["selected_hit_root_values"],
        "selected_hit_events": int(relation_scan["x_matches"]),
        "hit_event_summaries": relation_scan["hit_events"],
        "candidate_verifications": int(relation_scan["candidate_verifications"]),
        "relation_count": int(relation_scan["accepted_unsigned_relations"]),
        "rank": int(derived.get("rank") or 0),
        "row_public_key_verified": bool(derived.get("public_key_verified")),
        "derived_secret": derived.get("derived_secret"),
        "preassociation_filter_ops": preassociation_ops,
        "preassociation_filter_ops_over_rho": costs.get("preassociation_filter_ops_over_rho"),
        "nonshared_base_ops": nonshared_base_ops,
        "worklist_event_key_count": len(allowed_keys),
        "observed_worklist_event_key_count": len(observed),
        "missing_worklist_event_keys": [
            compact_event_key(key) for key in sorted(allowed_keys - observed, key=event_sort_key)
        ],
        "relation_events": relation_scan["relation_events"],
        "event_summaries": [
            replay_probe.dependency_circuit_probe.event_summary(index, event, order)
            for index, event in enumerate(relation_scan["relation_events"])
        ],
    }


def shared_leaf_ops(row_leaves: dict[str, set[int]]) -> tuple[int, int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for leaves in row_leaves.values():
        grouped[leaf_signature(leaves)].append(len(leaves))
    baseline = sum(sum(values) for values in grouped.values())
    shared = sum(max(values) for values in grouped.values())
    return baseline, shared


def shared_hit_root_ops(row_scans: list[tuple[str, dict[str, Any]]]) -> tuple[int, int]:
    baseline = 0
    grouped: dict[str, set[int]] = defaultdict(set)
    for _row_key, scan in row_scans:
        signature = str(scan.get("selected_leaf_signature") or "")
        roots = {as_int(root) for root in scan.get("selected_hit_root_values") or []}
        baseline += len(roots)
        grouped[signature].update(roots)
    return baseline, sum(len(roots) for roots in grouped.values())


def replay_worklist_selection(
    verifier: Any,
    row_leaves: dict[str, set[int]],
    allowed_by_row: dict[str, set[tuple[str, int, int]]],
    expected_instances: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    event_limit: int,
) -> dict[str, Any]:
    rows = []
    row_scans: list[tuple[str, dict[str, Any]]] = []
    row_events: list[tuple[str, dict[str, Any]]] = []
    observed_instances: list[dict[str, Any]] = []
    challenge_seeds: set[str] = set()
    generic_rho_steps = 0
    baseline_ops = 0
    nonshared_base_ops = 0
    unique_observed_keys: set[tuple[str, int, int]] = set()

    for row_key in sorted(row_leaves):
        context = contexts.get(row_key)
        if not context:
            continue
        leaves = {int(leaf) for leaf in row_leaves[row_key]}
        if not leaves:
            continue
        built = context["built"]
        local_args = context["local_args"]
        scan = scan_selected_with_worklist_keys(
            verifier,
            built,
            context["components"],
            leaves,
            allowed_by_row.get(row_key, set()),
            local_args,
        )
        order = int(built["order"])
        challenge_seed = str(built.get("challenge_seed") or local_args.challenge_seed)
        challenge_seeds.add(challenge_seed)
        generic_rho_steps = max(generic_rho_steps, int(built.get("generic_rho_steps") or 0))
        baseline_ops += int(scan.get("preassociation_filter_ops") or 0)
        nonshared_base_ops += int(scan.get("nonshared_base_ops") or 0)
        row_scans.append((row_key, scan))
        compact = replay_probe.compact_scan(row_key, leaves, scan, order, event_limit)
        compact["nonshared_base_ops"] = int(scan.get("nonshared_base_ops") or 0)
        rows.append(
            {
                "row_key": row_key,
                "challenge_seed": challenge_seed,
                "generic_rho_steps": int(built.get("generic_rho_steps") or 0),
                "scan": compact,
            }
        )
        for event in scan.get("relation_events") or []:
            row_events.append((row_key, event))
        for event in scan.get("hit_event_summaries") or []:
            key = normalized_event_key(event.get("event_key"))
            unique_observed_keys.add(key)
            observed_instances.append(
                {
                    "row_key": row_key,
                    "event_key": compact_event_key(key),
                    "scheduled_trial": as_int(event.get("scheduled_trial"), -1),
                    "candidate_pos": as_int(event.get("candidate_pos"), -1),
                    "accepted_relation": bool(event.get("accepted_relation")),
                    "relation_index": event.get("relation_index"),
                }
            )

    derive = replay_probe.derive_from_events(verifier, row_events, contexts)
    baseline_leaf_ops, leaf_ops = shared_leaf_ops(row_leaves)
    baseline_root_ops, hit_root_ops = shared_hit_root_ops(row_scans)
    first_event_pass_ops = len(unique_observed_keys)
    second_event_pass_ops = len(observed_instances)
    fused_ops = (
        nonshared_base_ops
        + leaf_ops
        + hit_root_ops
        + first_event_pass_ops
        + second_event_pass_ops
    )
    expected_by_key = {instance_key(instance): instance for instance in expected_instances}
    observed_by_key = {instance_key(instance): instance for instance in observed_instances}
    missing_instances = [
        expected_by_key[key]
        for key in sorted(set(expected_by_key) - set(observed_by_key), key=str)
    ]
    unexpected_instances = [
        observed_by_key[key]
        for key in sorted(set(observed_by_key) - set(expected_by_key), key=str)
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

    return {
        "selected_row_count": len([row for row in row_leaves.values() if row]),
        "materialized_row_count": len(rows),
        "selected_leaf_count": sum(len(leaves) for leaves in row_leaves.values()),
        "challenge_seeds": sorted(challenge_seeds),
        "baseline_ops": baseline_ops,
        "generic_rho_steps": generic_rho_steps,
        "baseline_ops_over_rho": clean_ratio(baseline_ops, generic_rho_steps),
        "baseline_below_rho": bool(generic_rho_steps and baseline_ops < generic_rho_steps),
        "relation_event_instance_count": len(observed_instances),
        "unique_event_key_count": first_event_pass_ops,
        "accepted_instance_count": sum(
            1 for instance in observed_instances if bool(instance.get("accepted_relation"))
        ),
        "work_counters": {
            "nonshared_base_ops": nonshared_base_ops,
            "baseline_leaf_ops": baseline_leaf_ops,
            "shared_leaf_ops": leaf_ops,
            "baseline_hit_root_ops": baseline_root_ops,
            "shared_hit_root_ops": hit_root_ops,
            "shared_first_event_pass_ops": first_event_pass_ops,
            "second_event_pass_ops": second_event_pass_ops,
            "fused_ops": fused_ops,
            "generic_rho_steps": generic_rho_steps,
            "fused_ops_over_rho": clean_ratio(fused_ops, generic_rho_steps),
            "fused_below_rho": bool(generic_rho_steps and fused_ops < generic_rho_steps),
            "first_pass_saved_ops": max(0, second_event_pass_ops - first_event_pass_ops),
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
        "rows": rows,
    }


def charged_records_by_source(charged_source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("source_name") or ""): record
        for record in charged_source.get("records") or []
        if isinstance(record, dict) and record.get("source_name") is not None
    }


def scanner_record(
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
        return {
            "source_name": source_name,
            "scanner_status": "missing_charged_locator_record",
            "implementation_boundary": (
                "The charged source is used only to locate decomposition artifacts; "
                "no charged work counters are consumed."
            ),
        }
    source_path = resolve_path(charged_record.get("source_path"))
    if not source_path.exists():
        return {
            "source_name": source_name,
            "scanner_status": "missing_decomposition_source",
            "source_path": str(source_path),
        }
    candidate_artifact, params, _decomposition = candidate_source_parameters(source_path)
    if not params:
        return {
            "source_name": source_name,
            "scanner_status": "missing_candidate_source_parameters",
            "source_path": str(source_path),
            "candidate_artifact": str(candidate_artifact) if candidate_artifact is not None else None,
        }

    row_leaves = row_leaves_from_worklist(worklist_record)
    allowed_by_row, expected_instances = expected_worklist_instances(worklist_record)
    public_group = worklist_record.get("public_group_key") or {}
    case = {
        "target": public_group.get("target"),
        "transfer_index": as_int(public_group.get("transfer_index")),
        "top_k": as_int(public_group.get("top_k")),
    }
    bundle = build_input_bundle(params, event_summary_limit, bundle_cache)
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
    scanner_replay = replay_worklist_selection(
        verifier,
        row_leaves,
        allowed_by_row,
        expected_instances,
        contexts,
        event_summary_limit,
    )
    counters = scanner_replay.get("work_counters") or {}
    observation = scanner_replay.get("worklist_observation") or {}
    worklist_counters = worklist_record.get("fused_worklist") or {}
    recorded_replay = worklist_record.get("replay") or {}
    matches_secret = (
        scanner_replay.get("derived_secret") is not None
        and recorded_replay.get("derived_secret") is not None
        and as_int(scanner_replay.get("derived_secret")) == as_int(recorded_replay.get("derived_secret"))
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
        and matches_secret
        and bool(scanner_replay.get("public_key_verified"))
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
        "scanner_status": (
            "worklist_keyed_scanner_consumed_and_relation_emitted"
            if consumed
            else "worklist_keyed_scanner_failed_check"
        ),
        "context_error_count": len(errors),
        "context_errors": errors,
        "relation_scanner_replay": scanner_replay,
        "matches": {
            "matches_recorded_secret": matches_secret,
            "recorded_replay_public_key_verified": bool(recorded_replay.get("public_key_verified")),
            "matches_worklist_fused_ops": matches_fused_ops,
            "matches_worklist_unique_first_pass_events": matches_unique_events,
            "matches_worklist_second_pass_instances": matches_second_pass,
            "matches_worklist_relation_count": (
                as_int(scanner_replay.get("relation_count"))
                == as_int(recorded_replay.get("relation_count"))
            ),
            "matches_worklist_rank": (
                as_int(scanner_replay.get("rank")) == as_int(recorded_replay.get("rank"))
            ),
        },
        "event_reuse_target": bool(
            (counters.get("fused_below_rho"))
            and as_int(counters.get("first_pass_saved_ops")) > 0
        ),
        "implementation_boundary": (
            "Consumes public worklist keys through the existing scanner and "
            "emits relation forms in this run.  The row context builder and "
            "verifier predicate are reused; an optimized fused FFE kernel is "
            "still pending."
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    consumed = [
        record
        for record in records
        if record.get("scanner_status") == "worklist_keyed_scanner_consumed_and_relation_emitted"
    ]
    derived = [
        record
        for record in records
        if bool((record.get("relation_scanner_replay") or {}).get("derived"))
    ]
    matched = [
        record
        for record in records
        if bool((record.get("matches") or {}).get("matches_recorded_secret"))
    ]
    below = [
        record
        for record in consumed
        if bool(((record.get("relation_scanner_replay") or {}).get("work_counters") or {}).get("fused_below_rho"))
    ]
    event_reuse = [record for record in below if bool(record.get("event_reuse_target"))]
    matched_ops = [
        record
        for record in records
        if bool((record.get("matches") or {}).get("matches_worklist_fused_ops"))
    ]
    ratios = [
        ((record.get("relation_scanner_replay") or {}).get("work_counters") or {}).get(
            "fused_ops_over_rho"
        )
        for record in below
        if ((record.get("relation_scanner_replay") or {}).get("work_counters") or {}).get(
            "fused_ops_over_rho"
        )
        is not None
    ]
    return {
        "record_count": len(records),
        "scanner_consumed_count": len(consumed),
        "scanner_derived_count": len(derived),
        "scanner_secret_match_count": len(matched),
        "scanner_fused_ops_match_count": len(matched_ops),
        "scanner_below_rho_count": len(below),
        "scanner_event_reuse_below_rho_count": len(event_reuse),
        "mean_scanner_below_rho_ops_over_rho": mean_or_none([float(value) for value in ratios]),
        "interpretation": (
            "The public fused worklist now drives scanner-layer x-match checks "
            "and relation-form emission.  This narrows the remaining work to "
            "an optimized fused FFE/summation-polynomial kernel."
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
    charged_by_source = charged_records_by_source(charged_source)
    wanted = set(args.source_name or [])
    verifier = replay_probe.relation_probe.load_verifier_module()
    records = verifier.load_records()
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    bundle_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    scanner_records = []
    for record in worklist_source.get("records") or []:
        if not isinstance(record, dict):
            continue
        source_name = str(record.get("source_name") or "")
        if wanted and source_name not in wanted:
            continue
        scanner_records.append(
            scanner_record(
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
        "schema": "ecdlp_public_repeated_coordinate_fused_worklist_scanner_probe_v1",
        "method": "consume_public_fused_worklist_through_scanner_relation_emission",
        "parameters": {
            "worklist_source": str(args.worklist_source),
            "charged_source": str(args.charged_source),
            "source_names": sorted(wanted),
            "event_summary_limit": int(args.event_summary_limit),
            "charged_source_usage": "locator_only_for_decomposition_source_paths",
        },
        "summary": summarize(scanner_records),
        "records": sorted(
            scanner_records,
            key=lambda record: (
                not bool(record.get("event_reuse_target")),
                not bool(((record.get("relation_scanner_replay") or {}).get("work_counters") or {}).get("fused_below_rho")),
                str(record.get("source_name") or ""),
            ),
        ),
        "non_claims": [
            "This is scanner-layer consumption of the public worklist, not a hand-optimized fused FFE kernel.",
            "The charged source is used only to locate decomposition artifacts; charged work counters are not consumed.",
            "The row context builder and verifier relation predicate are existing code paths, so promotion still requires implementing the fused low-level pass.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
