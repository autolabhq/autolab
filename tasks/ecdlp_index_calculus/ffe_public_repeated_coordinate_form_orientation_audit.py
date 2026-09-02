#!/usr/bin/env python3
"""Audit public form-orientation features for repeated-coordinate row rules.

Root-positive and row-pair rules can select relation-bearing rows, but held-out
rank-2 systems can still fail public-key verification.  This probe replays one
or more public row rules and records relation-form features before treating the
verifier result as a label.  It is a diagnostic for the next selector surface,
not a speedup proof.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_activation_rule_miner as activation_miner
import ffe_public_repeated_coordinate_root_positive_row_miner as row_miner
import ffe_public_repeated_coordinate_row_rule_replay_probe as row_replay
import ffe_single_hit_root_relation_replay_probe as replay_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_form_orientation_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_window(raw: str) -> tuple[str, Path]:
    return row_miner.parse_window(raw)


def parse_named_row_rule(raw: str) -> tuple[str, tuple[tuple[str, ...], ...]]:
    parts = raw.split("|", 1)
    if len(parts) == 2 and parts[0] and parts[1].startswith("row_activate:"):
        return parts[0], row_miner.parse_row_rule(parts[1])
    return f"rule{abs(hash(raw)) % 100000}", row_miner.parse_row_rule(raw)


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def centered(value: int, modulus: int) -> int:
    value %= modulus
    if value > modulus // 2:
        value -= modulus
    return value


def signature(items: list[Any]) -> str:
    return ",".join(str(item) for item in items)


def compact_event(
    row_key: str,
    event_index: int,
    event: dict[str, Any],
    contexts: dict[str, dict[str, Any]],
    row_salt: dict[str, int],
) -> dict[str, Any]:
    built = contexts[row_key]["built"]
    order = int(built["order"])
    coeffs, rhs, terms = event["form"]
    coeff_values = [int(coeff) for coeff in coeffs]
    nonzero_indices = [index for index, coeff in enumerate(coeff_values) if coeff % order]
    summary = replay_probe.dependency_circuit_probe.event_summary(event_index, event, order)
    candidate_pos = event.get("candidate_pos")
    scheduled_trial = event.get("scheduled_trial")
    return {
        "row_key": row_key,
        "salt": row_salt.get(row_key),
        "leaf_index": event.get("leaf_index"),
        "candidate_pos": candidate_pos,
        "scheduled_trial": scheduled_trial,
        "candidate_minus_scheduled": (
            int(candidate_pos) - int(scheduled_trial)
            if candidate_pos is not None and scheduled_trial is not None
            else None
        ),
        "scout_pos": event.get("scout_pos"),
        "form_index": event.get("form_index"),
        "rhs": int(rhs),
        "rhs_mod2": int(rhs) % 2,
        "rhs_mod4": int(rhs) % 4,
        "rhs_mod8": int(rhs) % 8,
        "rhs_mod16": int(rhs) % 16,
        "coeff_nonzero_count": len(nonzero_indices),
        "coeff_nonzero_indices": nonzero_indices,
        "coeff_support_signature": signature(nonzero_indices),
        "coeff_mod2_signature": signature([coeff % 2 for coeff in coeff_values]),
        "coeff_mod4_signature": signature([coeff % 4 for coeff in coeff_values]),
        "coeff_center_norm_sum": sum(abs(centered(coeff, order)) for coeff in coeff_values),
        "coeff_center_norm_max": max([abs(centered(coeff, order)) for coeff in coeff_values] or [0]),
        "term_shape": summary.get("term_shape"),
        "term_signature": summary.get("term_signature") or summary.get("terms"),
        "factor_support": summary.get("factor_support"),
        "terms": terms,
        "summary": summary,
    }


def selected_salt_map(selected_profiles: list[dict[str, Any]]) -> dict[str, int]:
    salts: dict[str, int] = {}
    for profile in selected_profiles:
        row_key = str(profile.get("row_key") or "")
        if row_key and profile.get("salt") is not None:
            salts[row_key] = int(profile["salt"])
    return salts


def system_features(case: dict[str, Any], replay: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    coordinate = case["coordinate"]
    row_event_counts = Counter(str(event["row_key"]) for event in events)
    salts = sorted({int(event["salt"]) for event in events if event.get("salt") is not None})
    candidate_positions = sorted(
        int(event["candidate_pos"]) for event in events if event.get("candidate_pos") is not None
    )
    candidate_deltas = sorted(
        int(event["candidate_minus_scheduled"])
        for event in events
        if event.get("candidate_minus_scheduled") is not None
    )
    rhs_mod16 = sorted(int(event["rhs_mod16"]) for event in events)
    support_signatures = sorted(str(event.get("coeff_support_signature") or "") for event in events)
    coeff_mod2 = sorted(str(event.get("coeff_mod2_signature") or "") for event in events)
    term_shapes = sorted(str(event.get("term_shape") or "") for event in events)
    return {
        "window": case.get("window"),
        "rule_name": case.get("rule_name"),
        "target": case.get("target"),
        "transfer_index": int(case.get("transfer_index") or 0),
        "transfer_mod2": int(case.get("transfer_index") or 0) % 2,
        "transfer_mod3": int(case.get("transfer_index") or 0) % 3,
        "transfer_mod4": int(case.get("transfer_index") or 0) % 4,
        "top_k": int(case.get("top_k") or 0),
        "policy_family": row_miner.policy_family(str(case.get("policy") or "")),
        "leaf_selector": case.get("leaf_selector"),
        "leaf_total": row_miner.selector_total(str(case.get("leaf_selector") or "")) or -1,
        "b_mod16": int(coordinate["b"]) % 16,
        "c_mod16": int(coordinate["c"]) % 16,
        "b_minus_c_mod16": (int(coordinate["b"]) - int(coordinate["c"])) % 16,
        "source_ops_millirhos": int(round(float(case.get("source_ops_over_rho") or 0.0) * 1000)),
        "selected_row_count": int(replay.get("selected_row_count") or 0),
        "materialized_row_count": int(replay.get("materialized_row_count") or 0),
        "relation_count": int(replay.get("relation_count") or 0),
        "rank": int(replay.get("rank") or 0),
        "unique_form_count": int(replay.get("unique_form_count") or 0),
        "duplicate_form_count": int(replay.get("duplicate_form_count") or 0),
        "ops_over_rho_bucket": int(round(float(replay.get("ops_over_rho") or 0.0) * 1000)),
        "event_count": len(events),
        "rows_with_events": len(row_event_counts),
        "max_events_per_row": max(row_event_counts.values() or [0]),
        "row_event_count_signature": signature(sorted(row_event_counts.values())),
        "event_salt_signature": signature(salts),
        "event_salt_span": max(salts) - min(salts) if salts else -1,
        "candidate_pos_signature": signature(candidate_positions),
        "candidate_pos_min": min(candidate_positions) if candidate_positions else -1,
        "candidate_pos_span": max(candidate_positions) - min(candidate_positions) if candidate_positions else -1,
        "candidate_delta_signature": signature(candidate_deltas),
        "rhs_mod16_signature": signature(rhs_mod16),
        "coeff_support_signature": "|".join(support_signatures),
        "coeff_mod2_signature": "|".join(coeff_mod2),
        "term_shape_signature": "|".join(term_shapes),
    }


def compact_replay_with_rows(replay: dict[str, Any]) -> dict[str, Any]:
    compact = row_replay.compact_replay(replay)
    compact["challenge_seeds"] = replay.get("challenge_seeds")
    return compact


def feature_atoms(features: dict[str, Any]) -> list[str]:
    excluded = {"window", "target", "rule_name"}
    atoms: list[str] = []
    for key, value in sorted(features.items()):
        if key in excluded or value in {None, ""}:
            continue
        if isinstance(value, (int, float, str)):
            atoms.append(f"{key}={value}")
    return atoms


def separator_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    relation_cases = [
        case
        for case in cases
        if int((case.get("row_rule_replay") or {}).get("relation_count") or 0) >= 2
    ]
    verified = [case for case in relation_cases if (case.get("row_rule_replay") or {}).get("public_key_verified")]
    unverified = [case for case in relation_cases if not (case.get("row_rule_replay") or {}).get("public_key_verified")]
    atom_counts: dict[str, Counter[str]] = {"verified": Counter(), "unverified": Counter()}
    for case in verified:
        atom_counts["verified"].update(feature_atoms(case.get("system_features") or {}))
    for case in unverified:
        atom_counts["unverified"].update(feature_atoms(case.get("system_features") or {}))
    positive_only = [
        {"atom": atom, "verified_count": count, "unverified_count": atom_counts["unverified"].get(atom, 0)}
        for atom, count in atom_counts["verified"].items()
        if count and not atom_counts["unverified"].get(atom)
    ]
    negative_only = [
        {"atom": atom, "unverified_count": count, "verified_count": atom_counts["verified"].get(atom, 0)}
        for atom, count in atom_counts["unverified"].items()
        if count and not atom_counts["verified"].get(atom)
    ]
    positive_only.sort(key=lambda item: (-int(item["verified_count"]), item["atom"]))
    negative_only.sort(key=lambda item: (-int(item["unverified_count"]), item["atom"]))
    return {
        "relation_case_count": len(relation_cases),
        "verified_relation_case_count": len(verified),
        "unverified_relation_case_count": len(unverified),
        "positive_only_atoms": positive_only[:24],
        "negative_only_atoms": negative_only[:24],
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [case for case in cases if int((case.get("row_rule_replay") or {}).get("selected_row_count") or 0) > 0]
    relation_cases = [
        case for case in selected if int((case.get("row_rule_replay") or {}).get("relation_count") or 0) >= 2
    ]
    verified = [
        case for case in relation_cases if bool((case.get("row_rule_replay") or {}).get("public_key_verified"))
    ]
    below = [case for case in verified if bool((case.get("row_rule_replay") or {}).get("below_rho"))]
    ratios = [
        float((case.get("row_rule_replay") or {}).get("ops_over_rho"))
        for case in relation_cases
        if (case.get("row_rule_replay") or {}).get("ops_over_rho") is not None
    ]
    return {
        "input_case_count": len(cases),
        "selected_case_count": len(selected),
        "relation_case_count": len(relation_cases),
        "verified_relation_case_count": len(verified),
        "verified_below_rho_count": len(below),
        "min_relation_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "mean_relation_ops_over_rho": mean_or_none(ratios),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=parse_window, action="append", required=True)
    parser.add_argument("--activation-rule", type=activation_miner.parse_activation_rule)
    parser.add_argument("--row-rule", type=parse_named_row_rule, action="append", required=True)
    parser.add_argument("--event-summary-limit", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    verifier = replay_probe.relation_probe.load_verifier_module()
    verifier_records = verifier.load_records()
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    context_errors: list[dict[str, Any]] = []

    for window_name, path in args.window:
        artifact = load_json(path)
        bank_source, config_source, direct_source, radius = row_miner.context_sources(artifact)
        specs_by_target = replay_probe.build_specs_by_target(bank_source, direct_source, radius)
        local_args = row_miner.replay_args(artifact.get("parameters") or {}, int(args.event_summary_limit))
        for candidate in artifact.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            activation_selected, activation_clauses = row_miner.candidate_activation_match(
                candidate,
                window_name,
                str(path),
                args.activation_rule,
            )
            case_base = {
                "window": window_name,
                "case_key": candidate.get("case_key"),
                "target": candidate.get("target"),
                "transfer_index": int(candidate.get("transfer_index") or 0),
                "top_k": int(candidate.get("top_k") or 0),
                "policy": candidate.get("policy"),
                "leaf_selector": candidate.get("leaf_selector"),
                "coordinate": {
                    "b": int(candidate.get("b") if candidate.get("b") is not None else (candidate.get("coordinate") or {}).get("b") or 0),
                    "c": int(candidate.get("c") if candidate.get("c") is not None else (candidate.get("coordinate") or {}).get("c") or 0),
                },
                "source_ops_over_rho": candidate.get("source_ops_over_rho"),
                "source_public_key_verified": bool(candidate.get("source_public_key_verified")),
                "activation_selected": bool(activation_selected),
                "activation_clauses": activation_clauses,
            }
            replay_case = {
                "target": candidate.get("target"),
                "transfer_index": int(candidate.get("transfer_index") or 0),
                "top_k": int(candidate.get("top_k") or 0),
            }
            for rule_name, row_rule in args.row_rule:
                row_leaves, selected_profiles = row_replay.selected_row_leaves(
                    candidate,
                    window_name,
                    str(path),
                    activation_selected,
                    activation_clauses,
                    row_rule,
                )
                if not row_leaves:
                    continue
                contexts, errors = replay_probe.materialize_contexts(
                    verifier,
                    verifier_records,
                    config_source,
                    specs_by_target,
                    replay_case,
                    sorted(row_leaves),
                    local_args,
                    context_cache,
                )
                context_errors.extend(errors)
                replay, row_events = replay_probe.replay_selection(
                    verifier,
                    row_leaves,
                    contexts,
                    scan_cache,
                    int(args.event_summary_limit),
                )
                if int(replay.get("relation_count") or 0) < 1:
                    continue
                row_salts = selected_salt_map(selected_profiles)
                events = [
                    compact_event(row_key, index, event, contexts, row_salts)
                    for index, (row_key, event) in enumerate(row_events)
                    if row_key in contexts
                ]
                case = {
                    **case_base,
                    "rule_name": rule_name,
                    "row_rule": row_miner.rule_text(row_rule),
                    "row_rule_selected_profiles": selected_profiles,
                    "row_rule_replay": compact_replay_with_rows(replay),
                    "events": events,
                }
                case["system_features"] = system_features(case, replay, events)
                cases.append(case)

    output = {
        "schema": "ecdlp_public_repeated_coordinate_form_orientation_audit_v1",
        "method": "relation_form_feature_audit_before_public_key_verification",
        "parameters": {
            "windows": [{"name": name, "artifact": str(path)} for name, path in args.window],
            "activation_rule": (
                activation_miner.rule_text(args.activation_rule) if args.activation_rule else None
            ),
            "row_rules": [
                {"name": name, "row_rule": row_miner.rule_text(rule)}
                for name, rule in args.row_rule
            ],
            "event_summary_limit": int(args.event_summary_limit),
        },
        "summary": summarize_cases(cases),
        "separator_summary": separator_summary(cases),
        "context_error_count": len(context_errors),
        "context_errors": context_errors[:32],
        "context_errors_truncated": len(context_errors) > 32,
        "cases": cases,
        "non_claims": [
            "Public-key verification is used only as an audit label here.",
            "Feature separators are diagnostic until replayed as frozen selection rules on fresh windows.",
            "Window and exact target identifiers are excluded from separator atoms, but remaining atoms may still be overfit.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": output["summary"], "separator_summary": output["separator_summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
