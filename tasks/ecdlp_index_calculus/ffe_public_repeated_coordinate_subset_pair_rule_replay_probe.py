#!/usr/bin/env python3
"""Replay repeated-coordinate row subsets selected by public pair rules."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_form_guard_replay_probe as form_guard
import ffe_public_repeated_coordinate_branch_selector_miner as branch_miner
import ffe_public_repeated_coordinate_root_positive_row_miner as row_miner
import ffe_public_repeated_coordinate_subset_pair_rule_miner as pair_miner
import ffe_single_hit_root_relation_replay_probe as replay_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_subset_pair_rule_replay_probe.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_window(raw: str) -> tuple[str, Path]:
    return row_miner.parse_window(raw)


def parse_named_pair_rule(raw: str) -> tuple[str, tuple[tuple[str, ...], ...]]:
    parts = raw.split("|", 1)
    if len(parts) == 2 and parts[0] and parts[1].startswith("pair_activate:"):
        return parts[0], pair_miner.parse_pair_rule(parts[1])
    return f"rule{abs(hash(raw)) % 100000}", pair_miner.parse_pair_rule(raw)


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def source_ops_millirhos(value: Any) -> int:
    rounded = round_or_none(value)
    return int(round(float(rounded) * 1000)) if rounded is not None else -1


def feature_record(values: dict[str, Any]) -> dict[str, list[str]]:
    features: dict[str, list[str]] = {}
    for key, value in values.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float, str)):
            features.setdefault(key, []).append(str(value))
    return features


def matched_candidate_clauses(
    candidate: dict[str, Any],
    candidate_clauses: list[tuple[str, ...]],
) -> list[str]:
    if not candidate_clauses:
        return ["all"]
    values = branch_miner.public_feature_values(candidate, "coordinate")
    record = {"features": feature_record(values)}
    return [
        branch_miner.clause_text(clause)
        for clause in candidate_clauses
        if branch_miner.clause_matches(clause, record)
    ]


def row_public_record(
    candidate: dict[str, Any],
    profile: dict[str, Any],
    window_name: str,
    source_path: str,
) -> dict[str, Any]:
    record = row_miner.row_public_features(candidate, profile, window_name, source_path, False, [])
    row_miner.add_mod_features(record)
    full_replay = candidate.get("exact_coordinate_replay") or {}
    record["full_ops_millirhos"] = source_ops_millirhos(full_replay.get("ops_over_rho"))
    record["full_relation_count"] = int(full_replay.get("relation_count") or 0)
    record["full_rank"] = int(full_replay.get("rank") or 0)
    record["row_leaf_signature"] = str(record.get("leaf_index"))
    return record


def candidate_pair_record(
    candidate: dict[str, Any],
    rows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    salts = sorted(int(row["salt"]) for row in rows)
    salt_indices = sorted(int(row["salt_index"]) for row in rows)
    salt_deltas_from_min = sorted(int(row["salt_delta_from_min"]) for row in rows)
    salt_deltas_to_max = sorted(int(row["salt_delta_to_max"]) for row in rows)
    first = rows[0]
    record = {
        "window": first["window"],
        "target": first["target"],
        "transfer_index": first["transfer_index"],
        "top_k": first["top_k"],
        "policy_family": first["policy_family"],
        "leaf_selector": first["leaf_selector"],
        "leaf_total": first["leaf_total"],
        "b_mod5": first["b_mod5"],
        "b_mod16": first["b_mod16"],
        "c_mod5": first["c_mod5"],
        "c_mod16": first["c_mod16"],
        "b_minus_c_mod16": first["b_minus_c_mod16"],
        "coordinate_key": first["coordinate_key"],
        "row_count": first["row_count"],
        "salt_count": first["salt_count"],
        "salt_span": first["salt_span"],
        "leaf_min": first["leaf_min"],
        "leaf_min_mod8": first["leaf_min_mod8"],
        "leaf_signature": first["leaf_signature"],
        "source_ops_millirhos": first["source_ops_millirhos"],
        "full_ops_millirhos": first["full_ops_millirhos"],
        "full_relation_count": first["full_relation_count"],
        "full_rank": first["full_rank"],
        "subset_size": len(rows),
        "pair_salt_span": max(salts) - min(salts) if salts else -1,
        "pair_salt_min_mod5": min(salts) % 5 if salts else -1,
        "pair_salt_max_mod5": max(salts) % 5 if salts else -1,
        "pair_salt_sum_mod5": sum(salts) % 5 if salts else -1,
        "pair_salt_sum_mod8": sum(salts) % 8 if salts else -1,
        "pair_salt_index_span": max(salt_indices) - min(salt_indices) if salt_indices else -1,
        "pair_salt_index_signature": ",".join(str(value) for value in salt_indices),
        "pair_salt_delta_from_min_signature": ",".join(str(value) for value in salt_deltas_from_min),
        "pair_salt_delta_to_max_signature": ",".join(str(value) for value in salt_deltas_to_max),
        "pair_salt_signature": ",".join(str(value) for value in salts),
        "pair_leaf_signature": "|".join(str(row.get("row_leaf_signature") or "") for row in rows),
        "row_keys": sorted(str(row["row_key"]) for row in rows),
    }
    pair_miner.add_mod_features(record)
    return record


def compact_replay(replay: dict[str, Any], guard_passed: bool) -> dict[str, Any]:
    compact = form_guard.compact_guarded_replay(replay, guard_passed)
    compact["row_keys"] = [
        str(row.get("row_key") or "")
        for row in replay.get("rows") or []
        if row.get("row_key")
    ]
    return compact


def case_record(
    candidate: dict[str, Any],
    window_name: str,
    rule_name: str,
    pair_rule: tuple[tuple[str, ...], ...],
    pair_features: dict[str, Any],
    candidate_clauses: list[str],
    selected_rows: tuple[dict[str, Any], ...],
    replay: dict[str, Any],
    form_guard_name: str,
    form_guard_features: dict[str, Any],
    guard_passed: bool,
) -> dict[str, Any]:
    coordinate = candidate.get("coordinate") or {}
    return {
        "window": window_name,
        "rule_name": rule_name,
        "pair_rule": pair_miner.rule_text(pair_rule),
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
        "pair_features": pair_features,
        "candidate_clauses": candidate_clauses,
        "selected_profiles": [
            {
                "row_key": row.get("row_key"),
                "salt": row.get("salt"),
                "leaf_index": row.get("leaf_index"),
            }
            for row in selected_rows
        ],
        "form_guard": form_guard_name,
        "form_guard_features": form_guard_features,
        "form_guard_passed": bool(guard_passed),
        "guarded_replay": compact_replay(replay, guard_passed),
    }


def group_by_rule(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(case.get("rule_name") or ""), []).append(case)
    return grouped


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        case for case in cases if int((case.get("guarded_replay") or {}).get("selected_row_count") or 0) > 0
    ]
    relation = [
        case for case in selected if int((case.get("guarded_replay") or {}).get("relation_count") or 0) >= 2
    ]
    passed = [case for case in relation if case.get("form_guard_passed")]
    guarded_verified = [
        case for case in passed if bool((case.get("guarded_replay") or {}).get("guarded_public_key_verified"))
    ]
    guarded_below = [
        case for case in guarded_verified if bool((case.get("guarded_replay") or {}).get("guarded_below_rho"))
    ]
    passed_ratios = [
        float((case.get("guarded_replay") or {}).get("ops_over_rho"))
        for case in passed
        if (case.get("guarded_replay") or {}).get("ops_over_rho") is not None
    ]
    return {
        "input_pair_case_count": len(cases),
        "selected_pair_case_count": len(selected),
        "relation_pair_case_count": len(relation),
        "guard_passed_relation_pair_case_count": len(passed),
        "guarded_verified_pair_case_count": len(guarded_verified),
        "guarded_below_rho_pair_case_count": len(guarded_below),
        "min_guard_passed_ops_over_rho": round(min(passed_ratios), 8) if passed_ratios else None,
        "mean_guard_passed_ops_over_rho": mean_or_none(passed_ratios),
        "by_rule": [
            {
                "rule_name": rule_name,
                "relation_pair_case_count": len(rows),
                "guard_passed_relation_pair_case_count": sum(1 for row in rows if row.get("form_guard_passed")),
                "guarded_verified_pair_case_count": sum(
                    1 for row in rows if bool((row.get("guarded_replay") or {}).get("guarded_public_key_verified"))
                ),
                "guarded_below_rho_pair_case_count": sum(
                    1 for row in rows if bool((row.get("guarded_replay") or {}).get("guarded_below_rho"))
                ),
            }
            for rule_name, rows in sorted(group_by_rule(relation).items())
        ],
        "best_guarded_verified_pair_cases": [
            {
                "rule_name": case.get("rule_name"),
                "window": case.get("window"),
                "target": case.get("target"),
                "transfer_index": case.get("transfer_index"),
                "coordinate": case.get("coordinate"),
                "top_k": case.get("top_k"),
                "policy": case.get("policy"),
                "leaf_selector": case.get("leaf_selector"),
                "pair_features": case.get("pair_features"),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=parse_window, action="append", required=True)
    parser.add_argument(
        "--candidate-clause",
        type=branch_miner.parse_clause,
        action="append",
        default=[],
        help="Public candidate-stage selector clause; repeat for OR.",
    )
    parser.add_argument("--pair-rule", type=parse_named_pair_rule, action="append", required=True)
    parser.add_argument("--form-guard", default="candidate_pos_span=0")
    parser.add_argument("--max-subset-size", type=int, default=2)
    parser.add_argument("--event-summary-limit", type=int, default=16)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    verifier = replay_probe.relation_probe.load_verifier_module()
    verifier_records = verifier.load_records()
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    context_errors: list[dict[str, Any]] = []
    candidate_count = 0
    candidate_matched_count = 0
    candidate_clause_counts: dict[str, int] = {}

    for window_name, path in args.window:
        artifact = load_json(path)
        bank_source, config_source, direct_source, radius = row_miner.context_sources(artifact)
        specs_by_target = replay_probe.build_specs_by_target(bank_source, direct_source, radius)
        local_args = row_miner.replay_args(artifact.get("parameters") or {}, int(args.event_summary_limit))
        for candidate in artifact.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            candidate_count += 1
            matched_clauses = matched_candidate_clauses(candidate, args.candidate_clause)
            if not matched_clauses:
                continue
            candidate_matched_count += 1
            for clause in matched_clauses:
                candidate_clause_counts[clause] = candidate_clause_counts.get(clause, 0) + 1
            profiles = [
                profile
                for profile in candidate.get("profiles") or []
                if isinstance(profile, dict) and profile.get("row_key")
            ]
            row_records = [
                row_public_record(candidate, profile, window_name, str(path))
                for profile in profiles
            ]
            row_profiles = list(zip(row_records, profiles))
            replay_case = {
                "target": candidate.get("target"),
                "transfer_index": int(candidate.get("transfer_index") or 0),
                "top_k": int(candidate.get("top_k") or 0),
            }
            for size in range(1, min(int(args.max_subset_size), len(row_profiles)) + 1):
                for subset in itertools.combinations(row_profiles, size):
                    selected_rows = tuple(item[0] for item in subset)
                    selected_profiles = tuple(item[1] for item in subset)
                    pair_features = candidate_pair_record(candidate, selected_rows)
                    for rule_name, pair_rule in args.pair_rule:
                        if not pair_miner.rule_matches(pair_rule, pair_features):
                            continue
                        row_leaves: dict[str, set[int]] = {}
                        for profile in selected_profiles:
                            row_key = str(profile.get("row_key") or "")
                            leaf_index = int(profile.get("leaf_index") or 0)
                            row_leaves.setdefault(row_key, set()).add(leaf_index)
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
                        features = form_guard.guard_features(row_events)
                        guard_passed = form_guard.guard_matches(features, str(args.form_guard))
                        cases.append(
                            case_record(
                                candidate,
                                window_name,
                                rule_name,
                                pair_rule,
                                pair_features,
                                matched_clauses,
                                selected_rows,
                                replay,
                                str(args.form_guard),
                                features,
                                guard_passed,
                            )
                        )

    output = {
        "schema": "ecdlp_public_repeated_coordinate_subset_pair_rule_replay_probe_v1",
        "method": "public_pair_rule_subset_replay_with_form_guard",
        "parameters": {
            "windows": [{"name": name, "artifact": str(path)} for name, path in args.window],
            "pair_rules": [
                {"name": name, "pair_rule": pair_miner.rule_text(rule)}
                for name, rule in args.pair_rule
            ],
            "candidate_clauses": [
                branch_miner.clause_text(clause) for clause in args.candidate_clause
            ],
            "form_guard": str(args.form_guard),
            "max_subset_size": int(args.max_subset_size),
            "event_summary_limit": int(args.event_summary_limit),
        },
        "candidate_filter_summary": {
            "input_candidate_count": candidate_count,
            "matched_candidate_count": candidate_matched_count,
            "candidate_clause_counts": candidate_clause_counts,
        },
        "summary": summarize(cases),
        "context_error_count": len(context_errors),
        "context_errors": context_errors[:32],
        "context_errors_truncated": len(context_errors) > 32,
        "cases": cases,
        "non_claims": [
            "Pair rules are public subset selectors, but each selected subset still pays row replay costs.",
            "A rule mined from decompositions is not a speedup claim until validated on a later window.",
            "The form guard is applied after relation-event scanning, as in the row-rule replay probe.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
