#!/usr/bin/env python3
"""Replay repeated-coordinate row rules with a frozen form-orientation guard.

The form-orientation audit found that relation systems with
``candidate_pos_span=0`` were exactly the key-consistent systems in the current
632-671 repeated-coordinate data.  Later pair-subset work also needs a nonvacuous
rank-lift form, such as ``candidate_pos_span=0&candidate_pos_count>=2``.  This
probe makes those guards operational for future windows: row rules still select
public row/leaf work, relation events are scanned, and the guard decides whether
the resulting event system is eligible for verifier-backed derivation accounting.

The guard is event-stage, not free row selection.  Costs include the selected
row scans needed to expose relation events.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_activation_rule_miner as activation_miner
import ffe_public_repeated_coordinate_form_orientation_audit as form_audit
import ffe_public_repeated_coordinate_root_positive_row_miner as row_miner
import ffe_public_repeated_coordinate_row_rule_replay_probe as row_replay
import ffe_single_hit_root_relation_replay_probe as replay_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_form_guard_replay_probe.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_window(raw: str) -> tuple[str, Path]:
    return row_miner.parse_window(raw)


def parse_named_row_rule(raw: str) -> tuple[str, tuple[tuple[str, ...], ...]]:
    return form_audit.parse_named_row_rule(raw)


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def compact_guarded_replay(replay: dict[str, Any], guard_passed: bool) -> dict[str, Any]:
    compact = row_replay.compact_replay(replay)
    compact["guard_passed"] = bool(guard_passed)
    compact["guarded_public_key_verified"] = bool(guard_passed and compact["public_key_verified"])
    compact["guarded_below_rho"] = bool(guard_passed and compact["public_key_verified"] and compact["below_rho"])
    compact["challenge_seeds"] = replay.get("challenge_seeds")
    return compact


def candidate_pos_values(row_events: list[tuple[str, dict[str, Any]]]) -> list[int]:
    values = []
    for _row_key, event in row_events:
        if event.get("candidate_pos") is not None:
            values.append(int(event["candidate_pos"]))
    return values


def guard_features(row_events: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    positions = sorted(candidate_pos_values(row_events))
    position_counts = Counter(positions)
    return {
        "candidate_pos_values": positions,
        "candidate_pos_count": len(positions),
        "candidate_pos_signature": ",".join(str(value) for value in positions),
        "candidate_pos_min": min(positions) if positions else None,
        "candidate_pos_max": max(positions) if positions else None,
        "candidate_pos_max_duplicate": max(position_counts.values() or [0]),
        "candidate_pos_span": (max(positions) - min(positions)) if positions else None,
        "candidate_pos_unique_count": len(set(positions)),
    }


def _parse_guard_value(raw: str) -> int | str:
    try:
        return int(raw)
    except ValueError:
        return raw


def _guard_atom_matches(features: dict[str, Any], atom: str) -> bool:
    for operator in (">=", "<=", "="):
        if operator not in atom:
            continue
        key, raw_expected = (part.strip() for part in atom.split(operator, 1))
        if not key:
            return False
        actual = features.get(key)
        expected = _parse_guard_value(raw_expected)
        if actual is None:
            return False
        if operator == "=":
            return actual == expected
        if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            return False
        if operator == ">=":
            return actual >= expected
        return actual <= expected
    raise ValueError(f"unsupported form guard atom: {atom}")


def _guard_clause_matches(features: dict[str, Any], clause: str) -> bool:
    clause = clause.strip()
    if not clause:
        return False
    if "&" in clause:
        return all(_guard_atom_matches(features, atom.strip()) for atom in clause.split("&") if atom.strip())
    if clause == "candidate_pos_span=0":
        return features.get("candidate_pos_span") == 0
    if clause == "candidate_pos_unique_count=1":
        return features.get("candidate_pos_unique_count") == 1
    if clause == "all":
        return True
    return _guard_atom_matches(features, clause)


def guard_matches(features: dict[str, Any], guard: str) -> bool:
    if "|" in guard:
        return any(_guard_clause_matches(features, clause) for clause in guard.split("|") if clause.strip())
    return _guard_clause_matches(features, guard)


def case_record(
    candidate: dict[str, Any],
    window_name: str,
    rule_name: str,
    row_rule: tuple[tuple[str, ...], ...],
    replay: dict[str, Any],
    selected_profiles: list[dict[str, Any]],
    activation_selected: bool,
    activation_clauses: list[str],
    form_guard: str,
    form_guard_features: dict[str, Any],
    guard_passed: bool,
) -> dict[str, Any]:
    coordinate = candidate.get("coordinate") or {}
    return {
        "window": window_name,
        "rule_name": rule_name,
        "row_rule": row_miner.rule_text(row_rule),
        "case_key": candidate.get("case_key"),
        "target": candidate.get("target"),
        "transfer_index": int(candidate.get("transfer_index") or 0),
        "top_k": int(candidate.get("top_k") or 0),
        "policy": candidate.get("policy"),
        "leaf_selector": candidate.get("leaf_selector"),
        "coordinate": {
            "b": int(candidate.get("b") if candidate.get("b") is not None else coordinate.get("b") or 0),
            "c": int(candidate.get("c") if candidate.get("c") is not None else coordinate.get("c") or 0),
        },
        "source_ops_over_rho": candidate.get("source_ops_over_rho"),
        "source_public_key_verified": bool(candidate.get("source_public_key_verified")),
        "activation_selected": bool(activation_selected),
        "activation_clauses": activation_clauses,
        "row_rule_selected_profiles": selected_profiles,
        "form_guard": form_guard,
        "form_guard_features": form_guard_features,
        "form_guard_passed": bool(guard_passed),
        "guarded_replay": compact_guarded_replay(replay, guard_passed),
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        case for case in cases if int((case.get("guarded_replay") or {}).get("selected_row_count") or 0) > 0
    ]
    relation = [
        case for case in selected if int((case.get("guarded_replay") or {}).get("relation_count") or 0) >= 2
    ]
    passed = [case for case in relation if case.get("form_guard_passed")]
    rejected = [case for case in relation if not case.get("form_guard_passed")]
    guarded_verified = [
        case for case in passed if bool((case.get("guarded_replay") or {}).get("guarded_public_key_verified"))
    ]
    guarded_below = [
        case for case in guarded_verified if bool((case.get("guarded_replay") or {}).get("guarded_below_rho"))
    ]
    rejected_verified = [
        case for case in rejected if bool((case.get("guarded_replay") or {}).get("public_key_verified"))
    ]
    passed_ratios = [
        float((case.get("guarded_replay") or {}).get("ops_over_rho"))
        for case in passed
        if (case.get("guarded_replay") or {}).get("ops_over_rho") is not None
    ]
    return {
        "input_case_count": len(cases),
        "selected_case_count": len(selected),
        "relation_case_count": len(relation),
        "guard_passed_relation_case_count": len(passed),
        "guard_rejected_relation_case_count": len(rejected),
        "guarded_verified_case_count": len(guarded_verified),
        "guarded_below_rho_count": len(guarded_below),
        "rejected_but_verified_case_count": len(rejected_verified),
        "min_guard_passed_ops_over_rho": round(min(passed_ratios), 8) if passed_ratios else None,
        "mean_guard_passed_ops_over_rho": mean_or_none(passed_ratios),
        "by_rule": [
            {
                "rule_name": rule_name,
                "relation_case_count": len(rows),
                "guard_passed_relation_case_count": sum(1 for row in rows if row.get("form_guard_passed")),
                "guarded_verified_case_count": sum(
                    1 for row in rows if bool((row.get("guarded_replay") or {}).get("guarded_public_key_verified"))
                ),
                "guarded_below_rho_count": sum(
                    1 for row in rows if bool((row.get("guarded_replay") or {}).get("guarded_below_rho"))
                ),
                "rejected_but_verified_case_count": sum(
                    1
                    for row in rows
                    if not row.get("form_guard_passed")
                    and bool((row.get("guarded_replay") or {}).get("public_key_verified"))
                ),
            }
            for rule_name, rows in sorted(group_by_rule(relation).items())
        ],
        "best_guarded_verified_cases": [
            {
                "rule_name": case.get("rule_name"),
                "window": case.get("window"),
                "target": case.get("target"),
                "transfer_index": case.get("transfer_index"),
                "coordinate": case.get("coordinate"),
                "top_k": case.get("top_k"),
                "policy": case.get("policy"),
                "leaf_selector": case.get("leaf_selector"),
                "form_guard_features": case.get("form_guard_features"),
                "guarded_replay": case.get("guarded_replay"),
            }
            for case in sorted(
                guarded_verified,
                key=lambda item: (
                    float((item.get("guarded_replay") or {}).get("ops_over_rho") or 10**9),
                    str(item.get("rule_name") or ""),
                    int(item.get("transfer_index") or 0),
                ),
            )[:12]
        ],
    }


def group_by_rule(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(case.get("rule_name") or ""), []).append(case)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=parse_window, action="append", required=True)
    parser.add_argument("--activation-rule", type=activation_miner.parse_activation_rule)
    parser.add_argument("--row-rule", type=parse_named_row_rule, action="append", required=True)
    parser.add_argument("--form-guard", default="candidate_pos_span=0")
    parser.add_argument("--event-summary-limit", type=int, default=16)
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
                features = guard_features(row_events)
                guard_passed = guard_matches(features, str(args.form_guard))
                cases.append(
                    case_record(
                        candidate,
                        window_name,
                        rule_name,
                        row_rule,
                        replay,
                        selected_profiles,
                        activation_selected,
                        activation_clauses,
                        str(args.form_guard),
                        features,
                        guard_passed,
                    )
                )

    output = {
        "schema": "ecdlp_public_repeated_coordinate_form_guard_replay_probe_v1",
        "method": "public_row_rule_replay_with_frozen_form_orientation_guard",
        "parameters": {
            "windows": [{"name": name, "artifact": str(path)} for name, path in args.window],
            "activation_rule": (
                activation_miner.rule_text(args.activation_rule) if args.activation_rule else None
            ),
            "row_rules": [
                {"name": name, "row_rule": row_miner.rule_text(rule)}
                for name, rule in args.row_rule
            ],
            "form_guard": str(args.form_guard),
            "event_summary_limit": int(args.event_summary_limit),
        },
        "summary": summarize(cases),
        "context_error_count": len(context_errors),
        "context_errors": context_errors[:32],
        "context_errors_truncated": len(context_errors) > 32,
        "cases": cases,
        "non_claims": [
            "The form guard is applied after relation-event scanning; row scan costs are still charged.",
            "This is a frozen replay wrapper, not fresh validation unless the input window was held out from guard discovery.",
            "Rejected cases may still have rank and relation count; rejection means the candidate-position alignment guard failed.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
