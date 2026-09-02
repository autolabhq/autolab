#!/usr/bin/env python3
"""Replay retained single-hit-root FFE relation cases through the verifier.

The 152-175 single-hit-root bridge joins below-rho FFE root hyperplanes to
compact low-term signature cases, but those bridge rows intentionally omit raw
relation events.  This probe reconstructs the original selected row/leaf sets
from the signature sources, filters them to retained single-hit-root surfaces,
and reruns the verifier-facing relation scanner.

The claim remains narrow: this proves that the retained surfaces can replay
public-key-verified relation systems under the same transfer challenge seeds.
It does not claim a generalized row/leaf selector or an end-to-end ECDLP
speedup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


CAMPAIGN_TASK_DIR = Path(
    os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
).resolve()
if str(CAMPAIGN_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_TASK_DIR))

import frontier_signed_eval_cover_dependency_circuit_probe as dependency_circuit_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_direct_witness_probe as direct_witness_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_compress_probe as compress_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_loto_leaf_trim_probe as leaf_trim_probe
import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_transfer_probe as salt_neighborhood_probe
import relation_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_LIVE_STATE_DIR = Path(
    os.environ.get("ECDLP_LIVE_STATE_DIR", "/Volumes/Volume/autolab/ecdlp_index_calculus_state")
)
DEFAULT_BRIDGE_SOURCE = DEFAULT_STATE_DIR / "ffe_single_hit_root_relation_bridge_152_175.json"
DEFAULT_BANK_SOURCE = DEFAULT_LIVE_STATE_DIR / leaf_trim_probe.DEFAULT_BANK_SOURCE.name
DEFAULT_CONFIG_SOURCE = DEFAULT_LIVE_STATE_DIR / leaf_trim_probe.DEFAULT_CONFIG_SOURCE.name
DEFAULT_DIRECT_SOURCE = DEFAULT_LIVE_STATE_DIR / leaf_trim_probe.DEFAULT_DIRECT_SOURCE.name
DEFAULT_TRANSFER_SOURCE = DEFAULT_LIVE_STATE_DIR / leaf_trim_probe.DEFAULT_TRANSFER_SOURCE.name
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_single_hit_root_relation_replay_152_175.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    row_leaf_key = tuple(
        (
            str(item.get("row_key")),
            tuple(int(leaf) for leaf in item.get("leaf_indices") or []),
        )
        for item in case.get("row_leaf_keys") or []
        if isinstance(item, dict)
    )
    return (
        str(case.get("target")),
        int(case.get("transfer_index") or 0),
        int(case.get("top_k") or 0),
        str(case.get("policy")),
        str(case.get("leaf_selector") or case.get("selector")),
        row_leaf_key,
    )


def case_key_string(case: dict[str, Any]) -> str:
    return "|".join(str(part) for part in case_key(case))


def form_key(event: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    coeffs, rhs, _terms = event["form"]
    return tuple(int(coeff) for coeff in coeffs), int(rhs)


def compact_event_summary(
    row_key: str,
    index: int,
    event: dict[str, Any],
    order: int,
) -> dict[str, Any]:
    summary = dependency_circuit_probe.event_summary(index, event, order)
    summary["row_key"] = row_key
    return summary


def compact_scan(
    row_key: str,
    leaves: set[int],
    scan: dict[str, Any],
    order: int,
    event_limit: int,
) -> dict[str, Any]:
    event_summaries = [
        compact_event_summary(row_key, index, event, order)
        for index, event in enumerate(scan.get("relation_events") or [])
    ]
    compact = {
        key: value
        for key, value in scan.items()
        if key not in {"relation_events", "event_summaries"}
    }
    compact["selected_leaf_indices"] = sorted(int(leaf) for leaf in leaves)
    compact["selected_leaf_count"] = len(leaves)
    compact["event_summary_count"] = len(event_summaries)
    compact["event_summaries"] = event_summaries[:event_limit]
    compact["event_summaries_truncated"] = len(event_summaries) > event_limit
    return compact


def retained_surface_ids(bridge: dict[str, Any]) -> set[str]:
    return {
        str(row.get("surface_id"))
        for row in bridge.get("retained_surfaces") or []
        if isinstance(row, dict) and row.get("surface_id") is not None
    }


def source_cases_from_bridge(bridge: dict[str, Any], retained_ids: set[str]) -> list[dict[str, Any]]:
    cases_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    bridge_profiles_by_case_key = {
        str(case.get("case_key")): (
            case.get("row_leaf_keys")
            or case.get("retained_row_leaf_keys")
            or case.get("marginal_lift_row_leaf_keys")
            or []
        )
        for case in bridge.get("retained_source_cases") or []
        if isinstance(case, dict) and case.get("case_key") is not None
    }
    selected_case_keys = {
        str(case.get("case_key"))
        for case in bridge.get("retained_source_cases") or []
        if isinstance(case, dict) and case.get("case_key") is not None
    }
    for window in (bridge.get("parameters") or {}).get("windows") or []:
        if not isinstance(window, dict):
            continue
        label = str(window.get("label"))
        signature_path = Path(str(window.get("signature_source")))
        signature = load_json(signature_path)
        for case in signature.get("positive_cases") or []:
            if not isinstance(case, dict):
                continue
            key_string = case_key_string(case)
            if selected_case_keys and key_string not in selected_case_keys:
                continue
            case_surface_ids = {str(surface) for surface in case.get("surface_ids") or []}
            if not case_surface_ids.intersection(retained_ids):
                continue
            enriched = dict(case)
            enriched["window"] = label
            enriched["signature_source"] = str(signature_path)
            enriched["case_key"] = key_string
            if bridge_profiles_by_case_key.get(key_string):
                enriched["bridge_retained_row_leaf_keys"] = bridge_profiles_by_case_key[key_string]
            cases_by_key[case_key(case)] = enriched
    return sorted(
        cases_by_key.values(),
        key=lambda row: (
            str(row.get("target")),
            int(row.get("transfer_index") or 0),
            str(row.get("case_key")),
        ),
    )


def row_leaf_groups(
    case: dict[str, Any],
    retained_ids: set[str] | None,
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    grouped: dict[str, set[int]] = defaultdict(set)
    kept_profiles: list[dict[str, Any]] = []
    for item in case.get("row_leaf_keys") or []:
        if not isinstance(item, dict):
            continue
        surface_id = str(item.get("surface_id") or "")
        if retained_ids is not None and surface_id not in retained_ids:
            continue
        key = str(item.get("row_key") or "")
        if not key:
            continue
        leaves = {int(leaf) for leaf in item.get("leaf_indices") or []}
        if not leaves:
            continue
        grouped[key].update(leaves)
        kept_profiles.append(
            {
                "row_key": key,
                "surface_id": surface_id or None,
                "leaf_indices": sorted(leaves),
                "salt": item.get("salt"),
            }
        )
    return dict(grouped), kept_profiles


def row_leaf_groups_from_profiles(
    profiles: list[dict[str, Any]],
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    grouped: dict[str, set[int]] = defaultdict(set)
    kept_profiles: list[dict[str, Any]] = []
    for item in profiles:
        if not isinstance(item, dict):
            continue
        key = str(item.get("row_key") or "")
        if not key:
            continue
        leaves = {int(leaf) for leaf in item.get("leaf_indices") or []}
        if not leaves:
            continue
        grouped[key].update(leaves)
        kept_profiles.append(
            {
                "row_key": key,
                "surface_id": item.get("surface_id"),
                "leaf_indices": sorted(leaves),
                "salt": item.get("salt"),
            }
        )
    return dict(grouped), kept_profiles


def build_specs_by_target(
    bank_source: dict[str, Any],
    direct_source: dict[str, Any],
    radius: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    bank_rows = {
        compress_probe.row_key(row): row
        for row in bank_source.get("bank_rows") or []
        if isinstance(row, dict) and compress_probe.row_key(row)
    }
    target_specs = salt_neighborhood_probe.witness_specs(direct_source, bank_rows, radius)
    return leaf_trim_probe.specs_by_target_and_key(target_specs)


def materialize_contexts(
    verifier: Any,
    records: list[dict[str, Any]],
    config_source: dict[str, Any],
    specs_by_target: dict[str, dict[str, dict[str, Any]]],
    case: dict[str, Any],
    row_keys: list[str],
    args: argparse.Namespace,
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    target = str(case.get("target"))
    transfer_index = int(case.get("transfer_index") or 0)
    top_k = int(case.get("top_k") or 0)
    specs = specs_by_target.get(target, {})
    contexts: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for row_key in row_keys:
        cache_key = (target, transfer_index, top_k, row_key)
        if cache_key not in context_cache:
            spec = specs.get(row_key)
            if spec is None:
                context_cache[cache_key] = {
                    "error": {
                        "target": target,
                        "transfer_index": transfer_index,
                        "top_k": top_k,
                        "row_key": row_key,
                        "error": "candidate spec not found",
                    }
                }
            else:
                row, built, components, local_args, feature_rows = leaf_trim_probe.scan_candidate_context(
                    verifier,
                    records,
                    config_source,
                    spec,
                    transfer_index,
                    top_k,
                    args,
                )
                if row.get("error"):
                    context_cache[cache_key] = {"error": row}
                else:
                    context_cache[cache_key] = {
                        "row": row,
                        "built": built,
                        "components": components,
                        "local_args": local_args,
                        "feature_rows": feature_rows,
                    }
        context = context_cache[cache_key]
        if context.get("error"):
            errors.append(context["error"])
        else:
            contexts[row_key] = context
    return contexts, errors


def filtered_leaf_gcd_association_with_trace(
    scouts: list[dict[str, Any]],
    components: dict[str, Any],
    p: int,
    selected_leaf_indices: set[int],
) -> dict[str, Any]:
    association_probe = direct_witness_probe.association_probe
    hit_poly = components["hit_poly"]
    hit_roots = components["hit_roots"]
    rows_by_x = components["rows_by_x"]
    row_order = components["row_order"]
    hits_by_scout = association_probe.empty_hits(scouts)
    selected_hit_roots: set[int] = set()
    for leaf_index, leaf in enumerate(components["leaves"]):
        if leaf_index not in selected_leaf_indices:
            continue
        gcd_poly = association_probe.polyops.poly_gcd(hit_poly, leaf["poly"], p)
        if max(0, len(gcd_poly) - 1) <= 0:
            continue
        roots = association_probe.roots_in_hit_set(gcd_poly, hit_roots, p)
        selected_hit_roots.update(int(root) for root in roots)
        association_probe.add_leaf_hits(hits_by_scout, leaf, roots, rows_by_x)
    return {
        "hits_by_scout": association_probe.finalize_hits(hits_by_scout, row_order),
        "row_hit_count": association_probe.row_hit_counts(hits_by_scout),
        "selected_hit_root_values": sorted(selected_hit_roots),
    }


def collect_relation_events_with_xmatch_trace(
    verifier: Any,
    built: dict[str, Any],
    cover: dict[str, Any],
    scheduled_rows: list[dict[str, Any]],
    ordered_scouts: list[dict[str, Any]],
    selected_limit: int,
    scout_to_leaf: dict[int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_guided_probe = direct_witness_probe.critical_leaf_probe.frontier_signed_target_guided_probe
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
    for candidate_pos, scout in enumerate(ordered_scouts[:selected_limit], start=1):
        scanned_candidates = candidate_pos
        scout_pos = int(scout["scout_pos"])
        leaf_index = scout_to_leaf[scout_pos]
        hit_rows = [
            scheduled_by_original[trial]
            for trial in cover["hits_by_scout"].get(scout_pos, [])
            if trial in scheduled_by_original
        ]
        candidate_point = verifier.add_points(scout["left"]["point"], scout["right"]["point"], ainvs, p)
        for row in hit_rows:
            x_matches += 1
            candidate_verifications += 1
            before = len(forms)
            hit_summary = {
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
    derived = direct_witness_probe.critical_leaf_probe.frontier_signed_dual_sieve_probe.rank_probe(
        verifier,
        forms,
        order,
    )
    return {
        "forms": forms,
        "relations": relations,
        "relation_events": relation_events,
        "hit_events": hit_events,
        "x_matches": x_matches,
        "candidate_verifications": candidate_verifications,
        "scanned_candidates": scanned_candidates,
        "accepted_unsigned_relations": len(relations),
        "unsigned_relation_rank": int(derived["mixed_wide_relation_rank"]),
        "unsigned_relations_derive_secret": bool(derived["mixed_wide_relations_derive_secret"]),
    }


def scan_selected_with_trace(
    verifier: Any,
    built: dict[str, Any],
    components: dict[str, Any],
    selected: set[int],
    local_args: argparse.Namespace,
) -> dict[str, Any]:
    p = int(built["p"])
    order = int(built["order"])
    cover = filtered_leaf_gcd_association_with_trace(
        built["scouts"],
        components,
        p,
        selected,
    )
    scheduled_rows = direct_witness_probe.eval_cover_probe.schedule_rows(
        built["rows"],
        cover["row_hit_count"],
        int(built["scheduled_row_count"]),
    )
    ordered_scouts = direct_witness_probe.eval_cover_probe.order_scouts(
        built["scouts"],
        cover["hits_by_scout"],
        scheduled_rows,
        str(built["scout_order"]),
    )
    scout_to_leaf = direct_witness_probe.critical_leaf_probe.scout_leaf_map(components)
    scan_args = argparse.Namespace(max_relations=local_args.max_relations)
    relation_scan = collect_relation_events_with_xmatch_trace(
        verifier,
        built,
        cover,
        scheduled_rows,
        ordered_scouts,
        int(built["selected_limit"]),
        scout_to_leaf,
        scan_args,
    )
    derived = dependency_circuit_probe.public_derive_from_events(
        verifier,
        relation_scan["relation_events"],
        order,
        built["base"],
        built["public"],
        built["ainvs"],
        p,
    )
    costs = direct_witness_probe.preassociation_filter_probe.prefilter_costs(
        built,
        len(selected),
        len(cover["selected_hit_root_values"]),
        int(relation_scan["x_matches"]),
        int(local_args.row_factor),
        int(local_args.product_factor),
    )
    return {
        "selected_leaf_indices": sorted(selected),
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
        "preassociation_filter_ops": int(costs.get("preassociation_filter_ops") or 0),
        "preassociation_filter_ops_over_rho": costs.get("preassociation_filter_ops_over_rho"),
        "relation_events": relation_scan["relation_events"],
        "event_summaries": [
            dependency_circuit_probe.event_summary(index, event, order)
            for index, event in enumerate(relation_scan["relation_events"])
        ],
    }


def derive_from_events(
    verifier: Any,
    row_events: list[tuple[str, dict[str, Any]]],
    contexts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not row_events:
        return {
            "relation_count": 0,
            "rank": 0,
            "public_key_verified": False,
            "derived": False,
            "derived_secret": None,
            "unique_form_count": 0,
            "duplicate_form_count": 0,
        }
    first_row_key = row_events[0][0]
    built = contexts[first_row_key]["built"]
    forms = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    duplicates = 0
    for _row_key, event in row_events:
        key = form_key(event)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        forms.append(event)
    derived = dependency_circuit_probe.public_derive_from_events(
        verifier,
        forms,
        int(built["order"]),
        built["base"],
        built["public"],
        built["ainvs"],
        int(built["p"]),
    )
    return {
        "relation_count": len(forms),
        "rank": int(derived.get("rank") or 0),
        "public_key_verified": bool(derived.get("public_key_verified")),
        "derived": bool(derived.get("derived")),
        "derived_secret": derived.get("derived_secret"),
        "unique_form_count": len(forms),
        "duplicate_form_count": duplicates,
    }


def replay_selection(
    verifier: Any,
    row_leaves: dict[str, set[int]],
    contexts: dict[str, dict[str, Any]],
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]],
    event_limit: int,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    rows = []
    row_events: list[tuple[str, dict[str, Any]]] = []
    challenge_seeds: set[str] = set()
    generic_rho_steps = 0
    total_ops = 0
    for row_key in sorted(row_leaves):
        context = contexts.get(row_key)
        if not context:
            continue
        leaves = {int(leaf) for leaf in row_leaves[row_key]}
        if not leaves:
            continue
        built = context["built"]
        local_args = context["local_args"]
        challenge_seed = str(built.get("challenge_seed") or local_args.challenge_seed)
        cache_key = (challenge_seed, row_key, tuple(sorted(leaves)))
        if cache_key not in scan_cache:
            scan_cache[cache_key] = scan_selected_with_trace(
                verifier,
                built,
                context["components"],
                leaves,
                local_args,
            )
        scan = scan_cache[cache_key]
        order = int(built["order"])
        challenge_seeds.add(challenge_seed)
        generic_rho_steps = max(generic_rho_steps, int(built.get("generic_rho_steps") or 0))
        total_ops += int(scan.get("preassociation_filter_ops") or 0)
        rows.append(
            {
                "row_key": row_key,
                "challenge_seed": built.get("challenge_seed") or local_args.challenge_seed,
                "generic_rho_steps": int(built.get("generic_rho_steps") or 0),
                "scan": compact_scan(row_key, leaves, scan, order, event_limit),
            }
        )
        for event in scan.get("relation_events") or []:
            row_events.append((row_key, event))

    derive = derive_from_events(verifier, row_events, contexts)
    selected_leaf_count = sum(len(leaves) for leaves in row_leaves.values())
    return (
        {
            "selected_row_count": len([row for row in row_leaves.values() if row]),
            "materialized_row_count": len(rows),
            "selected_leaf_count": selected_leaf_count,
            "challenge_seeds": sorted(challenge_seeds),
            "ops": total_ops,
            "generic_rho_steps": generic_rho_steps,
            "ops_over_rho": round_or_none(total_ops / generic_rho_steps if generic_rho_steps else None),
            "below_rho": bool(generic_rho_steps and total_ops < generic_rho_steps),
            **derive,
            "rows": rows,
        },
        row_events,
    )


def replay_case(
    verifier: Any,
    records: list[dict[str, Any]],
    config_source: dict[str, Any],
    specs_by_target: dict[str, dict[str, dict[str, Any]]],
    case: dict[str, Any],
    retained_ids: set[str],
    args: argparse.Namespace,
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]],
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if case.get("bridge_retained_row_leaf_keys"):
        retained_leaves, retained_profiles = row_leaf_groups_from_profiles(
            case.get("bridge_retained_row_leaf_keys") or []
        )
    else:
        retained_leaves, retained_profiles = row_leaf_groups(case, retained_ids)
    source_leaves, source_profiles = row_leaf_groups(case, None)
    all_row_keys = sorted(set(retained_leaves) | set(source_leaves))
    contexts, errors = materialize_contexts(
        verifier,
        records,
        config_source,
        specs_by_target,
        case,
        all_row_keys,
        args,
        context_cache,
    )
    retained_replay, retained_events = replay_selection(
        verifier,
        retained_leaves,
        contexts,
        scan_cache,
        args.event_summary_limit,
    )
    source_replay: dict[str, Any] | None = None
    source_events: list[tuple[str, dict[str, Any]]] = []
    if args.include_source_case_replay:
        source_replay, source_events = replay_selection(
            verifier,
            source_leaves,
            contexts,
            scan_cache,
            args.event_summary_limit,
        )
    signature_verified = bool(case.get("public_key_verified"))
    source_verified = (
        bool(source_replay.get("public_key_verified")) if source_replay is not None else None
    )
    result = {
        "case_key": str(case.get("case_key") or case_key_string(case)),
        "window": case.get("window"),
        "signature_source": case.get("signature_source"),
        "target": case.get("target"),
        "transfer_index": int(case.get("transfer_index") or 0),
        "top_k": int(case.get("top_k") or 0),
        "policy": case.get("policy"),
        "row_selector": case.get("row_selector"),
        "leaf_selector": case.get("leaf_selector") or case.get("selector"),
        "signature": {
            "public_key_verified": signature_verified,
            "relation_count": int(case.get("relation_count") or 0),
            "rank": int(case.get("rank") or 0),
            "ops_over_rho": case.get("ops_over_rho"),
            "below_rho": bool(case.get("below_rho")),
            "selected_row_count": int(case.get("selected_row_count") or 0),
            "selected_leaf_count": int(case.get("selected_leaf_count") or 0),
        },
        "retained_row_leaf_key_count": len(retained_profiles),
        "source_row_leaf_key_count": len(source_profiles),
        "retained_row_leaf_keys": retained_profiles,
        "context_error_count": len(errors),
        "context_errors": errors,
        "retained_only_replay": retained_replay,
        "source_case_replay": source_replay,
        "matches": {
            "retained_matches_signature_public_key": bool(
                retained_replay.get("public_key_verified") == signature_verified
            ),
            "retained_matches_signature_rank": bool(
                int(retained_replay.get("rank") or 0) == int(case.get("rank") or 0)
            ),
            "source_matches_signature_public_key": (
                None if source_verified is None else bool(source_verified == signature_verified)
            ),
            "source_matches_signature_rank": (
                None
                if source_replay is None
                else bool(int(source_replay.get("rank") or 0) == int(case.get("rank") or 0))
            ),
        },
    }
    return result, {
        "retained_only": retained_events,
        "source_case": source_events,
        "contexts": contexts,
    }


def group_relation_replays(
    verifier: Any,
    cases: list[dict[str, Any]],
    event_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in event_records:
        case = record["case"]
        events = record["events"]
        contexts = record["contexts"]
        if not events:
            continue
        first_row = events[0][0]
        built = contexts[first_row]["built"]
        target = str(case.get("target"))
        challenge_seed = str(built.get("challenge_seed"))
        key = (target, challenge_seed)
        if key not in grouped:
            grouped[key] = {
                "target": target,
                "challenge_seed": challenge_seed,
                "transfer_index": int(case.get("transfer_index") or 0),
                "cases": [],
                "row_events": [],
                "contexts": contexts,
            }
        grouped[key]["cases"].append(case)
        grouped[key]["row_events"].extend(events)
        grouped[key]["contexts"].update(contexts)

    out = []
    for (_target, _challenge_seed), group in sorted(grouped.items()):
        derive = derive_from_events(verifier, group["row_events"], group["contexts"])
        case_keys = sorted(str(case.get("case_key")) for case in group["cases"])
        out.append(
            {
                "target": group["target"],
                "challenge_seed": group["challenge_seed"],
                "transfer_index": group["transfer_index"],
                "case_count": len(case_keys),
                "case_keys": case_keys,
                "row_event_count": len(group["row_events"]),
                **derive,
            }
        )
    return out


def summarize_cases(cases: list[dict[str, Any]], groups: list[dict[str, Any]]) -> dict[str, Any]:
    retained_verified = [
        case
        for case in cases
        if (case.get("retained_only_replay") or {}).get("public_key_verified")
    ]
    retained_ratios = [
        float((case.get("retained_only_replay") or {}).get("ops_over_rho"))
        for case in cases
        if (case.get("retained_only_replay") or {}).get("ops_over_rho") is not None
    ]
    source_cases = [
        case
        for case in cases
        if isinstance(case.get("source_case_replay"), dict)
    ]
    source_verified = [
        case
        for case in source_cases
        if (case.get("source_case_replay") or {}).get("public_key_verified")
    ]
    retained_mismatches = [
        case
        for case in cases
        if not (case.get("matches") or {}).get("retained_matches_signature_public_key")
    ]
    source_mismatches = [
        case
        for case in source_cases
        if not (case.get("matches") or {}).get("source_matches_signature_public_key")
    ]
    context_errors = sum(int(case.get("context_error_count") or 0) for case in cases)
    return {
        "source_case_count": len(cases),
        "signature_public_key_verified_count": sum(
            1 for case in cases if (case.get("signature") or {}).get("public_key_verified")
        ),
        "retained_only_public_key_verified_count": len(retained_verified),
        "retained_only_replay_mismatch_count": len(retained_mismatches),
        "retained_only_context_error_count": context_errors,
        "retained_only_relation_count_sum": sum(
            int((case.get("retained_only_replay") or {}).get("relation_count") or 0)
            for case in cases
        ),
        "retained_only_max_rank": max(
            [int((case.get("retained_only_replay") or {}).get("rank") or 0) for case in cases]
            or [0]
        ),
        "retained_only_min_ops_over_rho": round(min(retained_ratios), 8) if retained_ratios else None,
        "retained_only_mean_ops_over_rho": mean_or_none(retained_ratios),
        "retained_only_max_ops_over_rho": round(max(retained_ratios), 8) if retained_ratios else None,
        "source_case_public_key_verified_count": len(source_verified) if source_cases else None,
        "source_case_replay_mismatch_count": len(source_mismatches) if source_cases else None,
        "challenge_group_count": len(groups),
        "challenge_group_public_key_verified_count": sum(
            1 for group in groups if group.get("public_key_verified")
        ),
        "challenge_group_relation_count_sum": sum(
            int(group.get("relation_count") or 0) for group in groups
        ),
        "challenge_group_max_rank": max(
            [int(group.get("rank") or 0) for group in groups] or [0]
        ),
        "bridge_status": (
            "retained_surface_relation_replay_verified_case_level"
            if cases and len(retained_verified) == len(cases) and not context_errors
            else "partial_retained_surface_relation_replay"
        ),
        "next_obligation": (
            "Replace signature-provided row/leaf keys with a public assembly "
            "selector and test whether the same retained-root mechanism scales "
            "to fresh targets without verifier-assisted leaf choice."
        ),
    }


def summarize_targets(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_target[str(case.get("target"))].append(case)
    out: dict[str, Any] = {}
    for target, rows in sorted(by_target.items()):
        transfers = sorted({int(row.get("transfer_index") or 0) for row in rows})
        ratios = [
            float((row.get("retained_only_replay") or {}).get("ops_over_rho"))
            for row in rows
            if (row.get("retained_only_replay") or {}).get("ops_over_rho") is not None
        ]
        out[target] = {
            "case_count": len(rows),
            "transfer_indices": transfers,
            "retained_only_public_key_verified_count": sum(
                1 for row in rows if (row.get("retained_only_replay") or {}).get("public_key_verified")
            ),
            "retained_only_relation_count_sum": sum(
                int((row.get("retained_only_replay") or {}).get("relation_count") or 0)
                for row in rows
            ),
            "retained_only_min_ops_over_rho": round(min(ratios), 8) if ratios else None,
            "retained_only_mean_ops_over_rho": mean_or_none(ratios),
            "retained_only_max_ops_over_rho": round(max(ratios), 8) if ratios else None,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-source", type=Path, default=DEFAULT_BRIDGE_SOURCE)
    parser.add_argument("--bank-source", type=Path, default=DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--row-pool", type=int, default=512)
    parser.add_argument("--row-count", type=int, default=128)
    parser.add_argument("--scout-limit", type=int, default=192)
    parser.add_argument("--scout-mode", default="s3_coeff_spread")
    parser.add_argument("--scout-order", default="eval_cover_hits_high")
    parser.add_argument("--selected-limit", type=int, default=64)
    parser.add_argument("--factor-base-size", type=int, default=16)
    parser.add_argument("--max-relations", type=int, default=96)
    parser.add_argument("--min-distinct-indices", type=int, default=4)
    parser.add_argument("--min-unsigned-distinct-indices", type=int, default=2)
    parser.add_argument(
        "--allow-combined-coefficients",
        dest="require_unit_coefficients",
        action="store_false",
    )
    parser.set_defaults(require_unit_coefficients=True)
    parser.add_argument("--row-factor", type=int, default=512)
    parser.add_argument("--product-factor", type=int, default=4096)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--event-summary-limit", type=int, default=12)
    parser.add_argument(
        "--include-source-case-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    bridge = load_json(args.bridge_source)
    bank_source = load_json(args.bank_source)
    config_source = load_json(args.config_source)
    direct_source = load_json(args.direct_source)
    transfer_source = load_json(args.transfer_source)
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    if not isinstance(params, dict):
        params = {}
    radius = int(args.radius if args.radius is not None else params.get("radius") or 4)
    retained_ids = retained_surface_ids(bridge)
    source_cases = source_cases_from_bridge(bridge, retained_ids)
    specs_by_target = build_specs_by_target(bank_source, direct_source, radius)
    verifier = relation_probe.load_verifier_module()
    records = verifier.load_records()

    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    replayed_cases: list[dict[str, Any]] = []
    retained_event_records = []
    for case in source_cases:
        replayed, event_bundle = replay_case(
            verifier,
            records,
            config_source,
            specs_by_target,
            case,
            retained_ids,
            args,
            context_cache,
            scan_cache,
        )
        replayed_cases.append(replayed)
        retained_event_records.append(
            {
                "case": replayed,
                "events": event_bundle["retained_only"],
                "contexts": event_bundle["contexts"],
            }
        )

    challenge_groups = group_relation_replays(verifier, replayed_cases, retained_event_records)
    output = {
        "schema": "ecdlp_ffe_single_hit_root_relation_replay_probe_v1",
        "method": "verifier_replay_of_retained_single_hit_root_row_leaf_relation_events",
        "parameters": {
            "bridge_source": str(args.bridge_source),
            "bank_source": str(args.bank_source),
            "config_source": str(args.config_source),
            "direct_source": str(args.direct_source),
            "transfer_source": str(args.transfer_source),
            "radius": radius,
            "row_pool": args.row_pool,
            "row_count": args.row_count,
            "scout_limit": args.scout_limit,
            "scout_mode": args.scout_mode,
            "scout_order": args.scout_order,
            "selected_limit": args.selected_limit,
            "factor_base_size": args.factor_base_size,
            "max_relations": args.max_relations,
            "min_distinct_indices": args.min_distinct_indices,
            "min_unsigned_distinct_indices": args.min_unsigned_distinct_indices,
            "require_unit_coefficients": args.require_unit_coefficients,
            "seed": args.seed,
            "retained_surface_count": len(retained_ids),
            "include_source_case_replay": args.include_source_case_replay,
            "event_summary_limit": args.event_summary_limit,
        },
        "summary": summarize_cases(replayed_cases, challenge_groups),
        "target_summaries": summarize_targets(replayed_cases),
        "challenge_groups": challenge_groups,
        "cases": replayed_cases,
        "non_claims": [
            "This is a replay of signature-provided row/leaf keys, not a fresh public row/leaf selector.",
            "Challenge groups are only combined within one transfer challenge seed; different transfer seeds are not mixed.",
            "A public-key-verified relation replay is evidence for the retained-root component, not a full ECDLP index-calculus break.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
