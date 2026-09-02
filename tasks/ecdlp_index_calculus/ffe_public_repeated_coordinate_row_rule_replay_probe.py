#!/usr/bin/env python3
"""Replay repeated-coordinate candidates after public row-rule pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_activation_rule_miner as activation_miner
import ffe_public_repeated_coordinate_root_positive_row_miner as row_miner
import ffe_single_hit_root_relation_replay_probe as replay_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_row_rule_replay_probe.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_window(raw: str) -> tuple[str, Path]:
    parts = raw.split("|", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("window must be name|artifact_path")
    return parts[0], Path(parts[1])


def round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def compact_replay(replay: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_row_count": replay.get("selected_row_count"),
        "materialized_row_count": replay.get("materialized_row_count"),
        "selected_leaf_count": replay.get("selected_leaf_count"),
        "relation_count": replay.get("relation_count"),
        "rank": replay.get("rank"),
        "unique_form_count": replay.get("unique_form_count"),
        "duplicate_form_count": replay.get("duplicate_form_count"),
        "public_key_verified": bool(replay.get("public_key_verified")),
        "derived_secret": replay.get("derived_secret"),
        "ops": replay.get("ops"),
        "generic_rho_steps": replay.get("generic_rho_steps"),
        "ops_over_rho": replay.get("ops_over_rho"),
        "below_rho": bool(replay.get("below_rho")),
    }


def selected_row_leaves(
    candidate: dict[str, Any],
    window_name: str,
    source_path: str,
    activation_selected: bool,
    activation_clauses: list[str],
    row_rule: tuple[tuple[str, ...], ...],
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    row_leaves: dict[str, set[int]] = {}
    selected_profiles: list[dict[str, Any]] = []
    for profile in candidate.get("profiles") or []:
        if not isinstance(profile, dict):
            continue
        features = row_miner.row_public_features(
            candidate,
            profile,
            window_name,
            source_path,
            activation_selected,
            activation_clauses,
        )
        row_miner.add_mod_features(features)
        if not row_miner.dnf_matches(row_rule, features):
            continue
        row_key = str(profile.get("row_key") or "")
        leaf_index = int(profile.get("leaf_index") or 0)
        if not row_key:
            continue
        row_leaves.setdefault(row_key, set()).add(leaf_index)
        selected_profiles.append(
            {
                "row_key": row_key,
                "salt": profile.get("salt"),
                "surface_id": profile.get("surface_id"),
                "leaf_indices": [leaf_index],
                "public_features": {
                    key: features.get(key)
                    for key in (
                        "transfer_index",
                        "top_k",
                        "policy_family",
                        "leaf_selector",
                        "b_mod5",
                        "b_mod16",
                        "b_minus_c_mod16",
                        "salt",
                        "salt_mod2",
                        "salt_mod4",
                        "salt_delta_from_min",
                        "salt_delta_to_max",
                        "leaf_min_mod8",
                    )
                },
            }
        )
    return row_leaves, selected_profiles


def candidate_record(
    candidate: dict[str, Any],
    window_name: str,
    replay: dict[str, Any],
    selected_profiles: list[dict[str, Any]],
    activation_selected: bool,
    activation_clauses: list[str],
) -> dict[str, Any]:
    coordinate = candidate.get("coordinate") or {}
    return {
        "window": window_name,
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
        "row_rule_replay": compact_replay(replay),
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [case for case in cases if int((case.get("row_rule_replay") or {}).get("selected_row_count") or 0) > 0]
    verified = [
        case for case in selected if bool((case.get("row_rule_replay") or {}).get("public_key_verified"))
    ]
    below = [case for case in verified if bool((case.get("row_rule_replay") or {}).get("below_rho"))]
    ratios = [
        float((case.get("row_rule_replay") or {}).get("ops_over_rho"))
        for case in selected
        if (case.get("row_rule_replay") or {}).get("ops_over_rho") is not None
    ]
    verified_ratios = [
        float((case.get("row_rule_replay") or {}).get("ops_over_rho"))
        for case in verified
        if (case.get("row_rule_replay") or {}).get("ops_over_rho") is not None
    ]
    return {
        "input_case_count": len(cases),
        "selected_case_count": len(selected),
        "verified_case_count": len(verified),
        "verified_below_rho_count": len(below),
        "min_selected_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "mean_selected_ops_over_rho": mean_or_none(ratios),
        "min_verified_ops_over_rho": round(min(verified_ratios), 8) if verified_ratios else None,
        "best_verified_cases": [
            {
                "window": case.get("window"),
                "target": case.get("target"),
                "transfer_index": case.get("transfer_index"),
                "coordinate": case.get("coordinate"),
                "top_k": case.get("top_k"),
                "policy": case.get("policy"),
                "leaf_selector": case.get("leaf_selector"),
                "source_ops_over_rho": case.get("source_ops_over_rho"),
                "row_rule_replay": case.get("row_rule_replay"),
                "row_rule_selected_profiles": case.get("row_rule_selected_profiles"),
            }
            for case in sorted(
                verified,
                key=lambda item: (
                    float((item.get("row_rule_replay") or {}).get("ops_over_rho") or 10**9),
                    int(item.get("transfer_index") or 0),
                    str(item.get("case_key") or ""),
                ),
            )[:12]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=parse_window, action="append", required=True)
    parser.add_argument("--activation-rule", type=activation_miner.parse_activation_rule)
    parser.add_argument("--activated-only", action="store_true")
    parser.add_argument("--row-rule", type=row_miner.parse_row_rule, required=True)
    parser.add_argument("--event-summary-limit", type=int, default=4)
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
            if args.activated_only and not activation_selected:
                continue
            row_leaves, selected_profiles = selected_row_leaves(
                candidate,
                window_name,
                str(path),
                activation_selected,
                activation_clauses,
                args.row_rule,
            )
            case = {
                "target": candidate.get("target"),
                "transfer_index": int(candidate.get("transfer_index") or 0),
                "top_k": int(candidate.get("top_k") or 0),
            }
            contexts, errors = replay_probe.materialize_contexts(
                verifier,
                verifier_records,
                config_source,
                specs_by_target,
                case,
                sorted(row_leaves),
                local_args,
                context_cache,
            )
            context_errors.extend(errors)
            replay, _events = replay_probe.replay_selection(
                verifier,
                row_leaves,
                contexts,
                scan_cache,
                int(args.event_summary_limit),
            )
            cases.append(
                candidate_record(
                    candidate,
                    window_name,
                    replay,
                    selected_profiles,
                    activation_selected,
                    activation_clauses,
                )
            )

    output = {
        "schema": "ecdlp_public_repeated_coordinate_row_rule_replay_probe_v1",
        "method": "public_row_rule_pruned_coordinate_replay",
        "parameters": {
            "windows": [{"name": name, "artifact": str(path)} for name, path in args.window],
            "activation_rule": (
                activation_miner.rule_text(args.activation_rule) if args.activation_rule else None
            ),
            "activated_only": bool(args.activated_only),
            "row_rule": row_miner.rule_text(args.row_rule),
            "event_summary_limit": int(args.event_summary_limit),
        },
        "summary": summarize(cases),
        "context_error_count": len(context_errors),
        "context_errors": context_errors[:32],
        "context_errors_truncated": len(context_errors) > 32,
        "cases": cases,
        "non_claims": [
            "The row rule is public, but it was mined from selected_hit_roots labels and needs fresh-window validation.",
            "A row-rule replay is not a complete speedup if candidate coordinate activation itself was not frozen for the tested window.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
