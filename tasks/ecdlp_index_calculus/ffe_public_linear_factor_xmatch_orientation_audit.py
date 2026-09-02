#!/usr/bin/env python3
"""Audit public x-match orientation after a linear-factor leaf gate.

The public linear factor gate gets the transfer-211 system to a verified
1.05109-rho near miss.  The remaining overhead is that each selected leaf still
emits two x-matches while only one becomes a valid relation.  This probe keeps
the same public factor gate, records every x-match candidate, and evaluates
simple public orientation rules before relation verification.

The audit reports both measured replay costs and explicitly marked charged
models.  A below-rho charged model is only a research lead unless the rule is
public, preserves the verifier-backed derivation, and its charge assumptions are
called out in the output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Any


CAMPAIGN_TASK_DIR = Path(
    os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
).resolve()
if str(CAMPAIGN_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_TASK_DIR))

WORKTREE_TASK_DIR = Path(__file__).resolve().parent
if str(WORKTREE_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(WORKTREE_TASK_DIR))

import frontier_signed_eval_cover_dependency_circuit_probe as dependency_circuit_probe
import frontier_signed_eval_cover_preassociation_filter_probe as preassociation_filter_probe
import frontier_signed_eval_cover_projection_probe as eval_cover_probe
import frontier_signed_eval_cover_critical_leaf_probe as critical_leaf_probe
import frontier_signed_target_guided_probe
import relation_probe

import ffe_public_linear_factor_gate_replay_probe as factor_gate_probe
import ffe_single_hit_root_relation_replay_probe as replay_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SIGNATURE_SOURCE = DEFAULT_STATE_DIR / "low_term_total3_total4_public_bounded_full_selector_208_215.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_linear_factor_xmatch_orientation_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def form_key(event: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    coeffs, rhs, _terms = event["form"]
    return tuple(int(coeff) for coeff in coeffs), int(rhs)


def term_shape(indices: list[int]) -> str:
    counts: dict[int, int] = defaultdict(int)
    for index in indices:
        counts[int(index)] += 1
    return "+".join(str(count) for count in sorted(counts.values(), reverse=True))


def support_key(match: dict[str, Any]) -> str:
    support = sorted({int(index) for index in match.get("unsigned_indices") or []})
    return "+".join(str(index) for index in support)


def compact_xmatch(match: dict[str, Any], order: int) -> dict[str, Any]:
    row = {
        key: match.get(key)
        for key in (
            "leaf_index",
            "scout_pos",
            "candidate_pos",
            "scheduled_trial",
            "original_trial",
            "valid_relation",
            "duplicate_form",
        )
    }
    row["unsigned_indices"] = [int(index) for index in match.get("unsigned_indices") or []]
    row["factor_support"] = sorted({int(index) for index in row["unsigned_indices"]})
    row["term_shape"] = term_shape(row["unsigned_indices"])
    if match.get("event"):
        row["event_summary"] = dependency_circuit_probe.event_summary(0, match["event"], order)
    return row


def collect_xmatches(
    verifier: Any,
    built: dict[str, Any],
    components: dict[str, Any],
    selected: set[int],
    local_args: argparse.Namespace,
) -> dict[str, Any]:
    p = int(built["p"])
    order = int(built["order"])
    cover = preassociation_filter_probe.filtered_leaf_gcd_association(
        built["scouts"],
        components,
        p,
        selected,
    )
    scheduled_rows = eval_cover_probe.schedule_rows(
        built["rows"],
        cover["row_hit_count"],
        int(built["scheduled_row_count"]),
    )
    ordered_scouts = eval_cover_probe.order_scouts(
        built["scouts"],
        cover["hits_by_scout"],
        scheduled_rows,
        str(built["scout_order"]),
    )
    scout_to_leaf = critical_leaf_probe.scout_leaf_map(components)
    scheduled_by_original = {int(row["original_trial"]): row for row in scheduled_rows}

    forms: list[tuple[Any, int, list[int]]] = []
    relations: list[dict[str, Any]] = []
    seen_forms: set[tuple[Any, int]] = set()
    xmatches: list[dict[str, Any]] = []
    for candidate_pos, scout in enumerate(ordered_scouts[: int(built["selected_limit"])], start=1):
        scout_pos = int(scout["scout_pos"])
        leaf_index = int(scout_to_leaf[scout_pos])
        if leaf_index not in selected:
            continue
        hit_rows = [
            scheduled_by_original[trial]
            for trial in cover["hits_by_scout"].get(scout_pos, [])
            if trial in scheduled_by_original
        ]
        if not hit_rows:
            continue
        candidate_point = verifier.add_points(scout["left"]["point"], scout["right"]["point"], built["ainvs"], p)
        for row in hit_rows:
            before = len(forms)
            before_keys = set(seen_forms)
            valid = frontier_signed_target_guided_probe.add_relation_if_valid(
                verifier,
                built["challenge"],
                built["base"],
                built["public"],
                built["ainvs"],
                p,
                order,
                candidate_point,
                [int(index) for index in scout["unsigned_indices"]],
                row,
                forms,
                seen_forms,
                relations,
            )
            duplicate = valid and len(forms) == before
            event = None
            if valid and len(forms) > before:
                event = {
                    "leaf_index": leaf_index,
                    "scout_pos": scout_pos,
                    "candidate_pos": candidate_pos,
                    "scheduled_trial": int(row["trial"]),
                    "original_trial": int(row["original_trial"]),
                    "form_index": before,
                    "form": forms[-1],
                }
            xmatches.append(
                {
                    "leaf_index": leaf_index,
                    "scout_pos": scout_pos,
                    "candidate_pos": candidate_pos,
                    "scheduled_trial": int(row["trial"]),
                    "original_trial": int(row["original_trial"]),
                    "unsigned_indices": [int(index) for index in scout["unsigned_indices"]],
                    "valid_relation": bool(valid and len(forms) > before),
                    "duplicate_form": bool(duplicate or before_keys == seen_forms and valid),
                    "event": event,
                }
            )
    events = [match["event"] for match in xmatches if match.get("event")]
    derived = dependency_circuit_probe.public_derive_from_events(
        verifier,
        events,
        order,
        built["base"],
        built["public"],
        built["ainvs"],
        p,
    )
    costs = preassociation_filter_probe.prefilter_costs(
        built,
        len(selected),
        int(cover["selected_hit_root_values"]),
        len(xmatches),
        int(local_args.row_factor),
        int(local_args.product_factor),
    )
    return {
        "selected_leaf_indices": sorted(selected),
        "selected_leaf_count": len(selected),
        "selected_hit_roots": int(cover["selected_hit_root_values"]),
        "x_match_count": len(xmatches),
        "relation_count": len(events),
        "rank": int(derived.get("rank") or 0),
        "public_key_verified": bool(derived.get("public_key_verified")),
        "derived_secret": derived.get("derived_secret"),
        "costs": costs,
        "xmatches": xmatches,
        "events": events,
        "order": order,
        "rho": int(built["generic_rho_steps"]),
    }


def public_rule_matches(rule: str, xmatches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if rule == "all":
        return list(xmatches)
    if rule == "scheduled_trial_min":
        if not xmatches:
            return []
        minimum = min(int(match["scheduled_trial"]) for match in xmatches)
        return [match for match in xmatches if int(match["scheduled_trial"]) == minimum]
    if rule == "candidate_pos_min":
        if not xmatches:
            return []
        minimum = min(int(match["candidate_pos"]) for match in xmatches)
        return [match for match in xmatches if int(match["candidate_pos"]) == minimum]
    if rule.startswith("term_shape:"):
        wanted = rule.split(":", 1)[1]
        return [
            match
            for match in xmatches
            if term_shape([int(index) for index in match.get("unsigned_indices") or []]) == wanted
        ]
    if rule == "salt_parity_even_35_odd_24":
        out = []
        for match in xmatches:
            salt = int(match.get("salt") or 0)
            support = sorted({int(index) for index in match.get("unsigned_indices") or []})
            wanted = [3, 5] if salt % 2 == 0 else [2, 4]
            if support == wanted:
                out.append(match)
        return out
    if rule.startswith("where:"):
        return [match for match in xmatches if dnf_rule_matches(rule, match, xmatches)]
    raise ValueError(f"unknown rule: {rule}")


def dnf_rule_matches(rule: str, match: dict[str, Any], row_xmatches: list[dict[str, Any]]) -> bool:
    expression = rule.split(":", 1)[1].strip()
    if not expression:
        return False
    return any(
        all(public_predicate_matches(part.strip(), match, row_xmatches) for part in clause.split("&") if part.strip())
        for clause in expression.split("|")
    )


def public_predicate_matches(predicate: str, match: dict[str, Any], row_xmatches: list[dict[str, Any]]) -> bool:
    if predicate == "candidate_eq_scheduled":
        return int(match.get("candidate_pos") or 0) == int(match.get("scheduled_trial") or 0)
    if predicate == "candidate_lt_scheduled":
        return int(match.get("candidate_pos") or 0) < int(match.get("scheduled_trial") or 0)
    if predicate == "candidate_gt_scheduled":
        return int(match.get("candidate_pos") or 0) > int(match.get("scheduled_trial") or 0)
    if predicate.startswith("candidate_pos="):
        return int(match.get("candidate_pos") or 0) == int(predicate.split("=", 1)[1])
    if predicate.startswith("scheduled_trial="):
        return int(match.get("scheduled_trial") or 0) == int(predicate.split("=", 1)[1])
    if predicate.startswith("support="):
        wanted = [int(part) for part in predicate.split("=", 1)[1].split("+") if part != ""]
        support = sorted({int(index) for index in match.get("unsigned_indices") or []})
        return support == wanted
    if predicate.startswith("term_shape="):
        wanted_shape = predicate.split("=", 1)[1]
        return term_shape([int(index) for index in match.get("unsigned_indices") or []]) == wanted_shape
    if predicate.startswith("salt_mod"):
        lhs, rhs = predicate.split("=", 1)
        modulus = int(lhs.removeprefix("salt_mod"))
        return int(match.get("salt") or 0) % modulus == int(rhs)
    if predicate.startswith("row_xmatch_count="):
        return len(row_xmatches) == int(predicate.split("=", 1)[1])
    if predicate.startswith("row_xmatch_count<="):
        return len(row_xmatches) <= int(predicate.split("<=", 1)[1])
    if predicate.startswith("row_xmatch_count>="):
        return len(row_xmatches) >= int(predicate.split(">=", 1)[1])
    raise ValueError(f"unknown where predicate: {predicate}")


def activation_features(
    case: dict[str, Any],
    rows: list[dict[str, Any]],
    kept_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    xmatches = [match for row in rows for match in row.get("xmatches") or []]
    support_counts: dict[str, int] = defaultdict(int)
    scheduled_counts: dict[int, int] = defaultdict(int)
    salt_mods: dict[int, list[int]] = {2: [], 3: [], 4: []}
    row_xmatch_counts: list[int] = []
    for row in rows:
        row_xmatches = row.get("xmatches") or []
        row_xmatch_counts.append(len(row_xmatches))
        salt = None
        row_key = str(row.get("row_key") or "")
        if ":salt" in row_key:
            try:
                salt = int(row_key.rsplit(":salt", 1)[1])
            except ValueError:
                salt = None
        if salt is not None:
            for modulus in salt_mods:
                salt_mods[modulus].append(salt % modulus)
        for match in row_xmatches:
            support_counts[support_key(match)] += 1
            scheduled_counts[int(match.get("scheduled_trial") or 0)] += 1
    return {
        "transfer_index": int(case.get("transfer_index") or 0),
        "top_k": int(case.get("top_k") or 0),
        "policy": case.get("policy"),
        "leaf_selector": case.get("leaf_selector") or case.get("selector"),
        "row_count": len(rows),
        "factor_zero_profile_count": len(kept_profiles),
        "xmatch_count": len(xmatches),
        "support05_count": int(support_counts.get("0+5") or 0),
        "support15_count": int(support_counts.get("1+5") or 0),
        "scheduled1_count": int(scheduled_counts.get(1) or 0),
        "row_xmatch_counts": sorted(row_xmatch_counts),
        "row_xmatch_count_min": min(row_xmatch_counts) if row_xmatch_counts else 0,
        "row_xmatch_count_max": max(row_xmatch_counts) if row_xmatch_counts else 0,
        "salt_mod2_pattern": sorted(salt_mods[2]),
        "salt_mod3_pattern": sorted(salt_mods[3]),
        "salt_mod4_pattern": sorted(salt_mods[4]),
        "salt_mod2_residue_counts": {str(residue): salt_mods[2].count(residue) for residue in range(2)},
        "salt_mod3_residue_counts": {str(residue): salt_mods[3].count(residue) for residue in range(3)},
        "salt_mod4_residue_counts": {str(residue): salt_mods[4].count(residue) for residue in range(4)},
    }


def activation_numeric_feature(features: dict[str, Any], key: str) -> int:
    if key.startswith("transfer_mod"):
        modulus = int(key.removeprefix("transfer_mod"))
        return int(features["transfer_index"]) % modulus
    if key.startswith("salt_mod") and "_residue" in key and key.endswith("_count"):
        prefix, residue_text = key.removesuffix("_count").split("_residue", 1)
        return int(features[f"{prefix}_residue_counts"].get(residue_text, 0))
    return int(features[key])


def activation_predicate_matches(predicate: str, features: dict[str, Any]) -> bool:
    if ">=" in predicate:
        key, value = predicate.split(">=", 1)
        return activation_numeric_feature(features, key) >= int(value)
    if "<=" in predicate:
        key, value = predicate.split("<=", 1)
        return activation_numeric_feature(features, key) <= int(value)
    key, value = predicate.split("=", 1)
    if key.startswith("transfer_mod"):
        return activation_numeric_feature(features, key) == int(value)
    if key.startswith("salt_mod") and "_residue" in key and key.endswith("_count"):
        return activation_numeric_feature(features, key) == int(value)
    if key in {"row_xmatch_counts", "salt_mod2_pattern", "salt_mod3_pattern", "salt_mod4_pattern"}:
        return ",".join(str(part) for part in features[key]) == value
    if key in {
        "transfer_index",
        "top_k",
        "row_count",
        "factor_zero_profile_count",
        "xmatch_count",
        "row_xmatch_count_min",
        "row_xmatch_count_max",
        "support05_count",
        "support15_count",
        "scheduled1_count",
    }:
        return int(features[key]) == int(value)
    return str(features.get(key)) == value


def activation_rule_matches(rule: str, features: dict[str, Any]) -> bool:
    if rule in {"", "all"}:
        return True
    if not rule.startswith("activate:"):
        raise ValueError(f"unknown activation rule: {rule}")
    expression = rule.split(":", 1)[1].strip()
    if not expression:
        return False
    return any(
        all(
            activation_predicate_matches(part.strip(), features)
            for part in clause.split("&")
            if part.strip()
        )
        for clause in expression.split("|")
    )


def derive_rule(
    verifier: Any,
    rows: list[dict[str, Any]],
    rule: str,
) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    events = []
    selected_xmatches = 0
    selected_compact = []
    contexts: dict[str, dict[str, Any]] = {}
    for row in rows:
        selected = public_rule_matches(rule, row["xmatches"])
        selected_xmatches += len(selected)
        selected_compact.extend(
            {
                "row_key": row["row_key"],
                **compact_xmatch(match, int(row["order"])),
            }
            for match in selected
        )
        events.extend(match["event"] for match in selected if match.get("event"))
        contexts.update(row["contexts"])
    if not events:
        return (
            {
                "relation_count": 0,
                "rank": 0,
                "public_key_verified": False,
                "derived_secret": None,
            },
            selected_xmatches,
            selected_compact,
        )
    first_row = rows[0]
    built = first_row["built"]
    derived = dependency_circuit_probe.public_derive_from_events(
        verifier,
        events,
        int(built["order"]),
        built["base"],
        built["public"],
        built["ainvs"],
        int(built["p"]),
    )
    return (
        {
            "relation_count": len(events),
            "rank": int(derived.get("rank") or 0),
            "public_key_verified": bool(derived.get("public_key_verified")),
            "derived_secret": derived.get("derived_secret"),
        },
        selected_xmatches,
        selected_compact,
    )


def charged_models(rows: list[dict[str, Any]], selected_xmatches: int, selected_relation_count: int) -> dict[str, Any]:
    if not rows:
        return {}
    rho = int(rows[0]["rho"])
    measured_oriented_ops = 0
    shared_leaf_savings = 0
    shared_hit_root_savings = 0
    grouped_leaf_keys: dict[tuple[int, tuple[int, ...]], int] = defaultdict(int)
    for row in rows:
        cost_inputs = row["cost_inputs"]
        selected_for_row = len(public_rule_matches(row["active_rule"], row["xmatches"]))
        oriented = (
            int(cost_inputs["core_ops"])
            + int(cost_inputs["selected_leaf_count"])
            + 2 * selected_for_row
        )
        measured_oriented_ops += oriented
        grouped_leaf_keys[(int(row["p"]), tuple(row["selected_leaf_indices"]))] += 1
    for (_p, _leaves), count in grouped_leaf_keys.items():
        if count > 1:
            shared_leaf_savings += count - 1
            shared_hit_root_savings += count - 1
    shared_leaf_ops = measured_oriented_ops - shared_leaf_savings
    shared_leaf_hit_root_ops = measured_oriented_ops - shared_leaf_savings - shared_hit_root_savings
    relation_only_event_ops = measured_oriented_ops - max(0, selected_xmatches - selected_relation_count)
    return {
        "rho": rho,
        "measured_oriented_ops": measured_oriented_ops,
        "measured_oriented_ops_over_rho": round(measured_oriented_ops / max(1, rho), 8),
        "shared_leaf_ops": shared_leaf_ops,
        "shared_leaf_ops_over_rho": round(shared_leaf_ops / max(1, rho), 8),
        "shared_leaf_hit_root_ops": shared_leaf_hit_root_ops,
        "shared_leaf_hit_root_ops_over_rho": round(shared_leaf_hit_root_ops / max(1, rho), 8),
        "relation_only_event_ops": relation_only_event_ops,
        "relation_only_event_ops_over_rho": round(relation_only_event_ops / max(1, rho), 8),
        "shared_leaf_savings": shared_leaf_savings,
        "shared_hit_root_savings": shared_hit_root_savings,
        "selected_xmatch_count": selected_xmatches,
        "selected_relation_count": selected_relation_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-source", type=Path, default=DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--bank-source", type=Path, default=factor_gate_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=factor_gate_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=factor_gate_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=factor_gate_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--factor", type=factor_gate_probe.parse_factor, required=True)
    parser.add_argument("--factor-p", type=int)
    parser.add_argument("--target")
    parser.add_argument("--transfer-index", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--policy")
    parser.add_argument("--leaf-selector")
    parser.add_argument("--radius", type=int)
    parser.add_argument("--rules", default="scheduled_trial_min,candidate_pos_min,term_shape:2+2,salt_parity_even_35_odd_24,all")
    parser.add_argument("--activation-rule", default="all")
    parser.add_argument("--event-summary-limit", type=int, default=12)
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
    parser.add_argument("--allow-combined-coefficients", dest="require_unit_coefficients", action="store_false")
    parser.set_defaults(require_unit_coefficients=True)
    parser.add_argument("--row-factor", type=int, default=512)
    parser.add_argument("--product-factor", type=int, default=4096)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    signature = load_json(args.signature_source)
    source_cases = [
        case
        for case in signature.get("positive_cases") or []
        if isinstance(case, dict) and factor_gate_probe.case_selected(case, args)
    ]
    bank_source = load_json(args.bank_source)
    config_source = load_json(args.config_source)
    direct_source = load_json(args.direct_source)
    transfer_source = load_json(args.transfer_source)
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    radius = int(args.radius if args.radius is not None else (params or {}).get("radius") or 4)
    specs_by_target = replay_probe.build_specs_by_target(bank_source, direct_source, radius)
    verifier = relation_probe.load_verifier_module()
    records = verifier.load_records()

    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    cases = []
    for case in source_cases:
        source_leaves, _source_profiles = replay_probe.row_leaf_groups(case, None)
        contexts, errors = replay_probe.materialize_contexts(
            verifier,
            records,
            config_source,
            specs_by_target,
            case,
            sorted(source_leaves),
            args,
            context_cache,
        )
        factor_leaves, kept_profiles, rejected_profiles = factor_gate_probe.gated_row_leaves(
            case,
            contexts,
            args.factor,
            args.factor_p,
        )
        rows = []
        for row_key in sorted(factor_leaves):
            if row_key not in contexts or not factor_leaves[row_key]:
                continue
            context = contexts[row_key]
            scan = collect_xmatches(
                verifier,
                context["built"],
                context["components"],
                set(factor_leaves[row_key]),
                context["local_args"],
            )
            row_salt = None
            if ":salt" in row_key:
                try:
                    row_salt = int(row_key.rsplit(":salt", 1)[1])
                except ValueError:
                    row_salt = None
            for match in scan["xmatches"]:
                match["row_key"] = row_key
                match["salt"] = row_salt
            rows.append(
                {
                    "row_key": row_key,
                    "selected_leaf_indices": sorted(factor_leaves[row_key]),
                    "p": int(context["built"]["p"]),
                    "rho": int(context["built"]["generic_rho_steps"]),
                    "built": context["built"],
                    "contexts": {row_key: context},
                    "cost_inputs": {
                        "core_ops": int(scan["costs"]["projected_core_ops_before_filtered_association"]),
                        "selected_leaf_count": int(scan["selected_leaf_count"]),
                        "selected_hit_roots": int(scan["selected_hit_roots"]),
                    },
                    "xmatches": scan["xmatches"],
                    "events": scan["events"],
                    "order": scan["order"],
                    "baseline_costs": scan["costs"],
                    "baseline_public_key_verified": bool(scan["public_key_verified"]),
                    "baseline_rank": int(scan["rank"]),
                }
            )
        activation = activation_features(case, rows, kept_profiles)
        activated = activation_rule_matches(str(args.activation_rule), activation)
        rules = [rule.strip() for rule in args.rules.split(",") if rule.strip()]
        rule_results = []
        if activated:
            for rule in rules:
                for row in rows:
                    row["active_rule"] = rule
                derive, selected_xmatches, selected_compact = derive_rule(verifier, rows, rule)
                charges = charged_models(rows, selected_xmatches, int(derive["relation_count"]))
                rule_results.append(
                    {
                        "rule": rule,
                        **derive,
                        "selected_xmatch_count": selected_xmatches,
                        "selected_xmatches": selected_compact[: args.event_summary_limit],
                        "selected_xmatches_truncated": len(selected_compact) > args.event_summary_limit,
                        "charged_models": charges,
                    }
                )
        cases.append(
            {
                "case_key": replay_probe.case_key_string(case),
                "target": case.get("target"),
                "transfer_index": int(case.get("transfer_index") or 0),
                "top_k": int(case.get("top_k") or 0),
                "policy": case.get("policy"),
                "leaf_selector": case.get("leaf_selector") or case.get("selector"),
                "source_public_key_verified": bool(case.get("public_key_verified")),
                "source_ops_over_rho": case.get("ops_over_rho"),
                "context_error_count": len(errors),
                "context_errors": errors,
                "activation_rule": args.activation_rule,
                "activation_rule_matched": activated,
                "activation_features": activation,
                "factor_zero_profiles": kept_profiles,
                "factor_rejected_profiles": rejected_profiles,
                "rows": [
                    {
                        "row_key": row["row_key"],
                        "selected_leaf_indices": row["selected_leaf_indices"],
                        "baseline_costs": row["baseline_costs"],
                        "baseline_rank": row["baseline_rank"],
                        "baseline_public_key_verified": row["baseline_public_key_verified"],
                        "xmatches": [compact_xmatch(match, int(row["order"])) for match in row["xmatches"]],
                    }
                    for row in rows
                ],
                "rule_results": rule_results,
            }
        )

    verified_rules = []
    for case in cases:
        for rule in case.get("rule_results") or []:
            if rule.get("public_key_verified"):
                verified_rules.append((case, rule))
    best_verified = sorted(
        verified_rules,
        key=lambda item: (
            float(((item[1].get("charged_models") or {}).get("measured_oriented_ops_over_rho")) or 10**18),
            str(item[0].get("case_key")),
            str(item[1].get("rule")),
        ),
    )[:8]
    verified_measured_below = [
        (case, rule)
        for case, rule in verified_rules
        if float((rule.get("charged_models") or {}).get("measured_oriented_ops_over_rho") or 10**18) < 1.0
    ]
    verified_shared_leaf_below = [
        (case, rule)
        for case, rule in verified_rules
        if float((rule.get("charged_models") or {}).get("shared_leaf_ops_over_rho") or 10**18) < 1.0
    ]
    verified_shared_leaf_hit_root_below = [
        (case, rule)
        for case, rule in verified_rules
        if float((rule.get("charged_models") or {}).get("shared_leaf_hit_root_ops_over_rho") or 10**18) < 1.0
    ]
    output = {
        "schema": "ecdlp_ffe_public_linear_factor_xmatch_orientation_audit_v1",
        "method": "public_linear_factor_gate_xmatch_orientation_audit",
        "parameters": {
            "signature_source": str(args.signature_source),
            "factor": {
                "b_coeff": int(args.factor[0]),
                "c_coeff": int(args.factor[1]),
                "constant": int(args.factor[2]),
                "p": args.factor_p,
            },
            "target": args.target,
            "transfer_index": args.transfer_index,
            "top_k": args.top_k,
            "policy": args.policy,
            "leaf_selector": args.leaf_selector,
            "rules": rules,
            "activation_rule": args.activation_rule,
            "radius": radius,
        },
        "summary": {
            "source_case_count": len(cases),
            "activated_case_count": sum(1 for case in cases if case.get("activation_rule_matched")),
            "verified_rule_count": len(verified_rules),
            "verified_measured_below_rho_rule_count": len(verified_measured_below),
            "verified_shared_leaf_below_rho_rule_count": len(verified_shared_leaf_below),
            "verified_shared_leaf_hit_root_below_rho_rule_count": len(verified_shared_leaf_hit_root_below),
            "min_verified_measured_ops_over_rho": round(
                min(
                    (
                        float((rule.get("charged_models") or {}).get("measured_oriented_ops_over_rho"))
                        for _case, rule in verified_rules
                        if (rule.get("charged_models") or {}).get("measured_oriented_ops_over_rho") is not None
                    ),
                    default=10**18,
                ),
                8,
            )
            if verified_rules
            else None,
            "min_verified_shared_leaf_ops_over_rho": round(
                min(
                    (
                        float((rule.get("charged_models") or {}).get("shared_leaf_ops_over_rho"))
                        for _case, rule in verified_rules
                        if (rule.get("charged_models") or {}).get("shared_leaf_ops_over_rho") is not None
                    ),
                    default=10**18,
                ),
                8,
            )
            if verified_rules
            else None,
            "min_verified_shared_leaf_hit_root_ops_over_rho": round(
                min(
                    (
                        float((rule.get("charged_models") or {}).get("shared_leaf_hit_root_ops_over_rho"))
                        for _case, rule in verified_rules
                        if (rule.get("charged_models") or {}).get("shared_leaf_hit_root_ops_over_rho") is not None
                    ),
                    default=10**18,
                ),
                8,
            )
            if verified_rules
            else None,
            "best_verified_rules": [
                {
                    "case_key": case.get("case_key"),
                    "rule": rule.get("rule"),
                    "rank": rule.get("rank"),
                    "relation_count": rule.get("relation_count"),
                    "derived_secret": rule.get("derived_secret"),
                    "selected_xmatch_count": rule.get("selected_xmatch_count"),
                    "charged_models": rule.get("charged_models"),
                }
                for case, rule in best_verified
            ],
            "interpretation": (
                "Rules are public filters over x-match metadata emitted after "
                "the public linear factor gate.  shared_leaf_hit_root_ops is a "
                "charged model that amortizes identical selected leaves and hit "
                "roots across rows; it is not a measured direct replay cost."
            ),
        },
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
