#!/usr/bin/env python3
"""Audit candidate-position alignment in repeated-coordinate pair replays.

Strict pair guards promoted systems whose accepted relation events land on the
same candidate position.  This probe replays relation-bearing pair cases from
frozen pair-replay artifacts, records event-level form features, and treats
public-key verification as a label for separator diagnostics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_public_repeated_coordinate_form_guard_replay_probe as form_guard
import ffe_public_repeated_coordinate_root_positive_row_miner as row_miner
import ffe_public_repeated_coordinate_subset_pair_rule_replay_probe as pair_replay
import ffe_single_hit_root_relation_replay_probe as replay_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = (
    DEFAULT_STATE_DIR
    / "ffe_public_repeated_coordinate_candidate_position_alignment_audit.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else WORKTREE_ROOT / path


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


def compact_replay(replay: dict[str, Any], guard_passed: bool) -> dict[str, Any]:
    compact = pair_replay.compact_replay(replay, guard_passed)
    compact["challenge_seeds"] = replay.get("challenge_seeds")
    return compact


def selected_row_leaves(
    selected_profiles: list[dict[str, Any]],
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    row_leaves: dict[str, set[int]] = {}
    normalized: list[dict[str, Any]] = []
    for profile in selected_profiles:
        if not isinstance(profile, dict):
            continue
        row_key = str(profile.get("row_key") or "")
        if not row_key:
            continue
        if profile.get("leaf_index") is not None:
            leaves = [int(profile["leaf_index"])]
        else:
            leaves = [int(leaf) for leaf in profile.get("leaf_indices") or []]
        if not leaves:
            continue
        row_leaves.setdefault(row_key, set()).update(leaves)
        normalized.append(
            {
                "row_key": row_key,
                "salt": profile.get("salt"),
                "leaf_indices": sorted(set(leaves)),
            }
        )
    return row_leaves, normalized


def selected_salt_map(selected_profiles: list[dict[str, Any]]) -> dict[str, int]:
    salts: dict[str, int] = {}
    for profile in selected_profiles:
        row_key = str(profile.get("row_key") or "")
        if row_key and profile.get("salt") is not None:
            salts[row_key] = int(profile["salt"])
    return salts


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
        "event_index": event_index,
        "row_key": row_key,
        "salt": row_salt.get(row_key),
        "leaf_index": event.get("leaf_index"),
        "candidate_pos": candidate_pos,
        "scheduled_trial": scheduled_trial,
        "original_trial": event.get("original_trial"),
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


def add_mod_feature(features: dict[str, Any], key: str, value: Any, moduli: tuple[int, ...]) -> None:
    if value is None:
        return
    try:
        number = int(value)
    except (TypeError, ValueError):
        return
    features[key] = number
    for modulus in moduli:
        features[f"{key}_mod{modulus}"] = number % modulus


def system_features(case: dict[str, Any], replay: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    coordinate = case.get("coordinate") or {}
    pair_features = case.get("pair_features") or {}
    row_event_counts = Counter(str(event["row_key"]) for event in events)
    salts = sorted({int(event["salt"]) for event in events if event.get("salt") is not None})
    leaves = sorted({int(event["leaf_index"]) for event in events if event.get("leaf_index") is not None})
    candidate_positions = sorted(
        int(event["candidate_pos"]) for event in events if event.get("candidate_pos") is not None
    )
    scheduled_trials = sorted(
        int(event["scheduled_trial"]) for event in events if event.get("scheduled_trial") is not None
    )
    original_trials = sorted(
        int(event["original_trial"]) for event in events if event.get("original_trial") is not None
    )
    candidate_deltas = sorted(
        int(event["candidate_minus_scheduled"])
        for event in events
        if event.get("candidate_minus_scheduled") is not None
    )
    rhs_mod16 = sorted(int(event["rhs_mod16"]) for event in events)
    support_signatures = sorted(str(event.get("coeff_support_signature") or "") for event in events)
    coeff_mod2 = sorted(str(event.get("coeff_mod2_signature") or "") for event in events)
    coeff_mod4 = sorted(str(event.get("coeff_mod4_signature") or "") for event in events)
    term_shapes = sorted(str(event.get("term_shape") or "") for event in events)
    features: dict[str, Any] = {
        "window": case.get("window"),
        "pair_rule_name": case.get("rule_name"),
        "target": case.get("target"),
        "top_k": int(case.get("top_k") or 0),
        "policy_family": row_miner.policy_family(str(case.get("policy") or "")),
        "leaf_selector": case.get("leaf_selector"),
        "leaf_total": row_miner.selector_total(str(case.get("leaf_selector") or "")) or -1,
        "selected_row_count": int(replay.get("selected_row_count") or 0),
        "materialized_row_count": int(replay.get("materialized_row_count") or 0),
        "selected_leaf_count": int(replay.get("selected_leaf_count") or 0),
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
        "event_leaf_signature": signature(leaves),
        "candidate_pos_signature": signature(candidate_positions),
        "candidate_pos_count": len(candidate_positions),
        "candidate_pos_unique_count": len(set(candidate_positions)),
        "candidate_pos_min": min(candidate_positions) if candidate_positions else -1,
        "candidate_pos_max": max(candidate_positions) if candidate_positions else -1,
        "candidate_pos_span": max(candidate_positions) - min(candidate_positions) if candidate_positions else -1,
        "candidate_pos_aligned": int(bool(candidate_positions) and len(set(candidate_positions)) == 1),
        "candidate_pos_span0_count_ge2": int(
            len(candidate_positions) >= 2
            and max(candidate_positions, default=-1) - min(candidate_positions, default=-1) == 0
        ),
        "scheduled_trial_signature": signature(scheduled_trials),
        "original_trial_signature": signature(original_trials),
        "candidate_delta_signature": signature(candidate_deltas),
        "rhs_mod16_signature": signature(rhs_mod16),
        "coeff_support_signature": "|".join(support_signatures),
        "coeff_mod2_signature": "|".join(coeff_mod2),
        "coeff_mod4_signature": "|".join(coeff_mod4),
        "term_shape_signature": "|".join(term_shapes),
    }
    add_mod_feature(features, "transfer_index", case.get("transfer_index"), (2, 3, 4, 5, 8, 16))
    add_mod_feature(features, "b", coordinate.get("b"), (5, 8, 16))
    add_mod_feature(features, "c", coordinate.get("c"), (5, 8, 16))
    if coordinate.get("b") is not None and coordinate.get("c") is not None:
        features["b_minus_c_mod16"] = (int(coordinate["b"]) - int(coordinate["c"])) % 16
    for key in (
        "subset_size",
        "row_count",
        "salt_count",
        "salt_span",
        "leaf_min",
        "leaf_total",
        "pair_salt_span",
        "pair_salt_index_span",
        "pair_salt_min_mod5",
        "pair_salt_max_mod5",
        "pair_salt_sum_mod5",
        "pair_salt_sum_mod8",
        "full_ops_millirhos",
        "source_ops_millirhos",
    ):
        if key in pair_features:
            features[key] = pair_features.get(key)
    for key in (
        "leaf_signature",
        "pair_leaf_signature",
        "pair_salt_signature",
        "pair_salt_index_signature",
        "pair_salt_delta_from_min_signature",
        "pair_salt_delta_to_max_signature",
    ):
        if pair_features.get(key) is not None:
            features[key] = pair_features.get(key)
    return features


def feature_atoms(features: dict[str, Any], *, form_only: bool = False) -> list[str]:
    excluded = {
        "window",
        "target",
        "pair_rule_name",
        "transfer_index",
        "b",
        "c",
        "relation_count",
        "rank",
        "ops_over_rho_bucket",
    }
    form_prefixes = (
        "candidate_",
        "scheduled_",
        "original_",
        "rhs_",
        "coeff_",
        "term_",
        "row_event_",
        "event_",
    )
    atoms: list[str] = []
    for key, value in sorted(features.items()):
        if key in excluded or value in {None, ""}:
            continue
        if form_only and not key.startswith(form_prefixes):
            continue
        if isinstance(value, bool):
            atoms.append(f"{key}={int(value)}")
        elif isinstance(value, (int, float, str)):
            atoms.append(f"{key}={value}")
    return atoms


def atom_separators(
    verified: list[dict[str, Any]],
    unverified: list[dict[str, Any]],
    *,
    form_only: bool = False,
) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = {"verified": Counter(), "unverified": Counter()}
    for case in verified:
        counts["verified"].update(feature_atoms(case.get("system_features") or {}, form_only=form_only))
    for case in unverified:
        counts["unverified"].update(feature_atoms(case.get("system_features") or {}, form_only=form_only))
    positive_only = [
        {"atom": atom, "verified_count": count, "unverified_count": counts["unverified"].get(atom, 0)}
        for atom, count in counts["verified"].items()
        if count and not counts["unverified"].get(atom)
    ]
    negative_only = [
        {"atom": atom, "unverified_count": count, "verified_count": counts["verified"].get(atom, 0)}
        for atom, count in counts["unverified"].items()
        if count and not counts["verified"].get(atom)
    ]
    positive_only.sort(key=lambda item: (-int(item["verified_count"]), item["atom"]))
    negative_only.sort(key=lambda item: (-int(item["unverified_count"]), item["atom"]))
    return {
        "positive_only_atoms": positive_only[:32],
        "negative_only_atoms": negative_only[:32],
    }


def separator_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    relation_cases = [
        case for case in cases if int((case.get("pair_case_replay") or {}).get("relation_count") or 0) >= 2
    ]
    verified = [case for case in relation_cases if (case.get("pair_case_replay") or {}).get("public_key_verified")]
    unverified = [
        case for case in relation_cases if not (case.get("pair_case_replay") or {}).get("public_key_verified")
    ]
    return {
        "relation_case_count": len(relation_cases),
        "verified_relation_case_count": len(verified),
        "unverified_relation_case_count": len(unverified),
        "all_feature_separators": atom_separators(verified, unverified),
        "form_feature_separators": atom_separators(verified, unverified, form_only=True),
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    relation_cases = [
        case for case in cases if int((case.get("pair_case_replay") or {}).get("relation_count") or 0) >= 2
    ]
    verified = [case for case in relation_cases if (case.get("pair_case_replay") or {}).get("public_key_verified")]
    unverified = [
        case for case in relation_cases if not (case.get("pair_case_replay") or {}).get("public_key_verified")
    ]
    below = [case for case in verified if bool((case.get("pair_case_replay") or {}).get("below_rho"))]
    verified_ratios = [
        float((case.get("pair_case_replay") or {}).get("ops_over_rho"))
        for case in verified
        if (case.get("pair_case_replay") or {}).get("ops_over_rho") is not None
    ]
    signature_counts: dict[str, Counter[str]] = {"verified": Counter(), "unverified": Counter()}
    for case in verified:
        signature_counts["verified"].update([str((case.get("system_features") or {}).get("candidate_pos_signature"))])
    for case in unverified:
        signature_counts["unverified"].update([str((case.get("system_features") or {}).get("candidate_pos_signature"))])
    return {
        "input_case_count": len(cases),
        "relation_case_count": len(relation_cases),
        "verified_relation_case_count": len(verified),
        "unverified_relation_case_count": len(unverified),
        "verified_below_rho_count": len(below),
        "min_verified_ops_over_rho": round(min(verified_ratios), 8) if verified_ratios else None,
        "mean_verified_ops_over_rho": mean_or_none(verified_ratios),
        "candidate_pos_signature_counts": {
            label: dict(sorted(counter.items()))
            for label, counter in signature_counts.items()
        },
        "best_verified_relation_cases": [
            {
                "pair_replay_source": case.get("pair_replay_source"),
                "window": case.get("window"),
                "target": case.get("target"),
                "transfer_index": case.get("transfer_index"),
                "coordinate": case.get("coordinate"),
                "top_k": case.get("top_k"),
                "policy": case.get("policy"),
                "leaf_selector": case.get("leaf_selector"),
                "selected_profiles": case.get("selected_profiles"),
                "system_features": case.get("system_features"),
                "pair_case_replay": case.get("pair_case_replay"),
            }
            for case in sorted(
                verified,
                key=lambda item: (
                    float((item.get("pair_case_replay") or {}).get("ops_over_rho") or 10**9),
                    int(item.get("transfer_index") or 0),
                    str(item.get("case_key") or ""),
                ),
            )[:12]
        ],
    }


def replay_matches(original: dict[str, Any], replay: dict[str, Any]) -> dict[str, bool]:
    return {
        "relation_count": int(original.get("relation_count") or 0) == int(replay.get("relation_count") or 0),
        "rank": int(original.get("rank") or 0) == int(replay.get("rank") or 0),
        "public_key_verified": bool(original.get("public_key_verified")) == bool(replay.get("public_key_verified")),
        "derived_secret": original.get("derived_secret") == replay.get("derived_secret"),
        "ops_over_rho": round_or_none(original.get("ops_over_rho")) == round_or_none(replay.get("ops_over_rho")),
    }


def window_sources(pair_artifact: dict[str, Any]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for window in (pair_artifact.get("parameters") or {}).get("windows") or []:
        if not isinstance(window, dict):
            continue
        name = str(window.get("name") or "")
        artifact = str(window.get("artifact") or "")
        if name and artifact:
            sources[name] = resolve_path(Path(artifact))
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-replay", type=Path, action="append", required=True)
    parser.add_argument("--min-relation-count", type=int, default=2)
    parser.add_argument("--event-summary-limit", type=int, default=16)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    verifier = replay_probe.relation_probe.load_verifier_module()
    verifier_records = verifier.load_records()
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    scan_cache: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    material_cache: dict[Path, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    context_errors: list[dict[str, Any]] = []
    pair_sources: list[dict[str, Any]] = []
    skipped_cases: list[dict[str, Any]] = []

    for raw_pair_path in args.pair_replay:
        pair_path = resolve_path(raw_pair_path)
        pair_artifact = load_json(pair_path)
        pair_sources.append({"artifact": str(raw_pair_path), "schema": pair_artifact.get("schema")})
        sources_by_window = window_sources(pair_artifact)
        for pair_case in pair_artifact.get("cases") or []:
            if not isinstance(pair_case, dict):
                continue
            original_replay = pair_case.get("guarded_replay") or {}
            if int(original_replay.get("relation_count") or 0) < int(args.min_relation_count):
                continue
            window_name = str(pair_case.get("window") or "")
            source_path = sources_by_window.get(window_name)
            if source_path is None:
                skipped_cases.append(
                    {
                        "pair_replay_source": str(raw_pair_path),
                        "case_key": pair_case.get("case_key"),
                        "window": window_name,
                        "reason": "window source not found",
                    }
                )
                continue
            if source_path not in material_cache:
                coordinate_gate = load_json(source_path)
                bank_source, config_source, direct_source, radius = row_miner.context_sources(coordinate_gate)
                material_cache[source_path] = {
                    "artifact": coordinate_gate,
                    "config_source": config_source,
                    "specs_by_target": replay_probe.build_specs_by_target(bank_source, direct_source, radius),
                    "local_args": row_miner.replay_args(
                        coordinate_gate.get("parameters") or {},
                        int(args.event_summary_limit),
                    ),
                }
            material = material_cache[source_path]
            row_leaves, normalized_profiles = selected_row_leaves(pair_case.get("selected_profiles") or [])
            replay_case = {
                "target": pair_case.get("target"),
                "transfer_index": int(pair_case.get("transfer_index") or 0),
                "top_k": int(pair_case.get("top_k") or 0),
            }
            contexts, errors = replay_probe.materialize_contexts(
                verifier,
                verifier_records,
                material["config_source"],
                material["specs_by_target"],
                replay_case,
                sorted(row_leaves),
                material["local_args"],
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
            row_salts = selected_salt_map(pair_case.get("selected_profiles") or [])
            events = [
                compact_event(row_key, index, event, contexts, row_salts)
                for index, (row_key, event) in enumerate(row_events)
                if row_key in contexts
            ]
            guard = str(pair_case.get("form_guard") or (pair_artifact.get("parameters") or {}).get("form_guard") or "all")
            guard_features = form_guard.guard_features(row_events)
            guard_passed = form_guard.guard_matches(guard_features, guard)
            compact = compact_replay(replay, guard_passed)
            case = {
                "pair_replay_source": str(raw_pair_path),
                "window_source": str(source_path.relative_to(WORKTREE_ROOT) if source_path.is_relative_to(WORKTREE_ROOT) else source_path),
                "window": window_name,
                "rule_name": pair_case.get("rule_name"),
                "pair_rule": pair_case.get("pair_rule"),
                "case_key": pair_case.get("case_key"),
                "target": pair_case.get("target"),
                "transfer_index": int(pair_case.get("transfer_index") or 0),
                "top_k": int(pair_case.get("top_k") or 0),
                "policy": pair_case.get("policy"),
                "leaf_selector": pair_case.get("leaf_selector"),
                "coordinate": pair_case.get("coordinate"),
                "pair_features": pair_case.get("pair_features"),
                "selected_profiles": normalized_profiles,
                "form_guard": guard,
                "source_form_guard_features": pair_case.get("form_guard_features"),
                "replayed_form_guard_features": guard_features,
                "form_guard_features_match": pair_case.get("form_guard_features") == guard_features,
                "source_pair_case_replay": original_replay,
                "pair_case_replay": compact,
                "matches_source_pair_case_replay": replay_matches(original_replay, compact),
                "events": events,
            }
            case["system_features"] = system_features(case, replay, events)
            cases.append(case)

    output = {
        "schema": "ecdlp_public_repeated_coordinate_candidate_position_alignment_audit_v1",
        "method": "pair_relation_event_candidate_position_alignment_audit",
        "parameters": {
            "pair_replays": pair_sources,
            "min_relation_count": int(args.min_relation_count),
            "event_summary_limit": int(args.event_summary_limit),
        },
        "summary": summarize_cases(cases),
        "separator_summary": separator_summary(cases),
        "context_error_count": len(context_errors),
        "context_errors": context_errors[:32],
        "context_errors_truncated": len(context_errors) > 32,
        "skipped_case_count": len(skipped_cases),
        "skipped_cases": skipped_cases[:32],
        "skipped_cases_truncated": len(skipped_cases) > 32,
        "cases": cases,
        "non_claims": [
            "This audit uses public-key verification only as a diagnostic label.",
            "Candidate-position alignment is an event-stage guard; it is not free row selection.",
            "Separator atoms are descriptive for these windows until frozen and replayed on later windows.",
        ],
    }
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": output["summary"], "separator_summary": output["separator_summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
