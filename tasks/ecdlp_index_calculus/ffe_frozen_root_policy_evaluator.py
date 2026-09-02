#!/usr/bin/env python3
"""Evaluate a frozen root-hyperplane policy on held-out FFE surfaces.

This script closes one leakage gap in the first-fall/root-hyperplane line:
choose a single static public root-ordering policy from preregistered
pre-factor-gated calibration windows, then score that same policy on
preregistered held-out windows.

Learned policies from ``ffe_first_fall_root_hyperplane_selector_probe.py`` are
not selected here.  The existing selector outputs train learned policies within
each evaluated bank; static public policies are the cleanest way to freeze a
rule without reusing held-out factor outcomes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import ffe_first_fall_root_hyperplane_selector_probe as root_selector


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_TRAIN_PAIRS = (
    (
        "fresh_72_79",
        DEFAULT_STATE_DIR / "ffe_first_fall_root_hyperplane_selector_all_public_leaf_plus_fresh_72_79.json",
        DEFAULT_STATE_DIR / "ffe_preregistered_gate_manifest_all_public_leaf_plus_fresh_72_79.json",
    ),
    (
        "verified_over_rho_80_87",
        DEFAULT_STATE_DIR / "ffe_first_fall_root_hyperplane_selector_total3_total4_verified_over_rho_80_87.json",
        DEFAULT_STATE_DIR / "ffe_preregistered_gate_manifest_total3_total4_verified_over_rho_80_87.json",
    ),
)
DEFAULT_HOLDOUT_PAIRS = (
    (
        "total3_total4_88_95",
        DEFAULT_STATE_DIR / "ffe_first_fall_root_hyperplane_selector_total3_total4_preregistered_88_95.json",
        DEFAULT_STATE_DIR / "ffe_preregistered_gate_manifest_total3_total4_fixed_selector_88_95.json",
    ),
    (
        "total2_88_95",
        DEFAULT_STATE_DIR / "ffe_first_fall_root_hyperplane_selector_total2_preregistered_88_95.json",
        DEFAULT_STATE_DIR / "ffe_preregistered_gate_manifest_total2_fixed_selector_88_95.json",
    ),
)
DEFAULT_TRAIN_SURFACE_PAIRS = (
    (
        "fresh_72_79",
        DEFAULT_STATE_DIR / "ffe_first_fall_linear_factor_audit_all_public_leaf_plus_fresh_72_79.json",
        DEFAULT_STATE_DIR / "ffe_sage_factor_all_public_leaf_plus_fresh_72_79.json",
        DEFAULT_STATE_DIR / "ffe_preregistered_gate_manifest_all_public_leaf_plus_fresh_72_79.json",
    ),
    (
        "verified_over_rho_80_87",
        DEFAULT_STATE_DIR / "ffe_first_fall_linear_factor_audit_total3_total4_verified_over_rho_80_87.json",
        DEFAULT_STATE_DIR / "ffe_sage_factor_total3_total4_verified_over_rho_80_87.json",
        DEFAULT_STATE_DIR / "ffe_preregistered_gate_manifest_total3_total4_verified_over_rho_80_87.json",
    ),
)
DEFAULT_HOLDOUT_SURFACE_PAIRS = (
    (
        "total3_total4_88_95",
        DEFAULT_STATE_DIR / "ffe_first_fall_linear_factor_audit_total3_total4_preregistered_88_95.json",
        DEFAULT_STATE_DIR / "ffe_sage_factor_total3_total4_preregistered_88_95.json",
        DEFAULT_STATE_DIR / "ffe_preregistered_gate_manifest_total3_total4_fixed_selector_88_95.json",
    ),
    (
        "total2_88_95",
        DEFAULT_STATE_DIR / "ffe_first_fall_linear_factor_audit_total2_preregistered_88_95.json",
        DEFAULT_STATE_DIR / "ffe_sage_factor_total2_preregistered_88_95.json",
        DEFAULT_STATE_DIR / "ffe_preregistered_gate_manifest_total2_fixed_selector_88_95.json",
    ),
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_frozen_root_policy_train_le87_test_88_95.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_pair(raw: str) -> tuple[str, Path, Path]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "pair must be label:selector_json:gate_manifest_json"
        )
    label, selector, gate = parts
    if not label:
        raise argparse.ArgumentTypeError("pair label must be nonempty")
    return label, Path(selector), Path(gate)


def parse_surface_pair(raw: str) -> tuple[str, Path, Path, Path]:
    parts = raw.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "surface pair must be label:audit_json:sage_factor_json:gate_manifest_json"
        )
    label, audit, sage, gate = parts
    if not label:
        raise argparse.ArgumentTypeError("surface pair label must be nonempty")
    return label, Path(audit), Path(sage), Path(gate)


def gate_surface_ids(gate_source: dict[str, Any]) -> set[str]:
    return {
        str(surface.get("surface_id"))
        for surface in gate_source.get("surfaces") or []
        if isinstance(surface, dict) and surface.get("pre_factor_gate_selected")
    }


def filter_policy_rows(
    selector_source: dict[str, Any],
    selected_surface_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    policy_rows = selector_source.get("policy_rows") or {}
    if not isinstance(policy_rows, dict):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for policy, rows in policy_rows.items():
        if not isinstance(rows, list):
            continue
        filtered = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("surface_id")) in selected_surface_ids
        ]
        out[str(policy)] = filtered
    return out


def summarize_policy_rows(policy_rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {
        policy: root_selector.summarize_rows(rows)
        for policy, rows in sorted(policy_rows.items())
    }


def best_static_policy(policy_summaries: dict[str, dict[str, Any]]) -> str | None:
    static_summaries = {
        policy: summary
        for policy, summary in policy_summaries.items()
        if policy in root_selector.STATIC_POLICIES
        and int(summary.get("surface_count") or 0) > 0
    }
    return root_selector.best_policy_name(static_summaries)


def row_target_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("target")) for row in rows))


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "surface_count",
        "selected_pair_surface_count",
        "public_zero_capable_surface_count",
        "public_zero_recovered_count",
        "chosen_preserving_count",
        "chosen_false_positive_count",
        "below_rho_count",
        "direct_below_rho_count",
        "all_surfaces_preserving",
        "all_surfaces_false_positive_free",
        "all_surfaces_below_rho",
        "all_surfaces_direct_below_rho",
        "all_surfaces_below_rho_either_route",
        "min_total_ops_over_rho",
        "mean_total_ops_over_rho",
        "max_total_ops_over_rho",
        "min_direct_total_ops_over_rho",
        "mean_direct_total_ops_over_rho",
        "max_direct_total_ops_over_rho",
        "mean_evaluated_root_count",
        "mean_selector_eval_ops",
        "mean_train_row_count",
        "vacuous_preserving_surface_count",
    )
    return {key: summary.get(key) for key in keys if key in summary}


def source_report(
    label: str,
    selector_path: Path,
    gate_path: Path,
    selected_policy: str | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    selector_source = load_json(selector_path)
    gate_source = load_json(gate_path)
    selected_ids = gate_surface_ids(gate_source)
    policy_rows = filter_policy_rows(selector_source, selected_ids)
    policy_summaries = summarize_policy_rows(policy_rows)
    static_best = best_static_policy(policy_summaries)
    selected_rows = policy_rows.get(selected_policy or "", []) if selected_policy else []
    all_rows = next(iter(policy_rows.values()), [])
    report = {
        "label": label,
        "selector_source": str(selector_path),
        "gate_source": str(gate_path),
        "selector_schema": selector_source.get("schema"),
        "gate_schema": gate_source.get("schema"),
        "gate_surface_count": len(selected_ids),
        "selector_surface_count": len(all_rows),
        "target_counts": row_target_counts(all_rows),
        "best_static_policy": static_best,
        "best_static_policy_summary": compact_summary(policy_summaries.get(static_best or "", {})),
    }
    if selected_policy:
        report["selected_policy"] = selected_policy
        report["selected_policy_summary"] = compact_summary(
            root_selector.summarize_rows(selected_rows)
        )
        report["selected_policy_target_counts"] = row_target_counts(selected_rows)
    return report, policy_rows, policy_summaries


def aggregate_policy_rows(
    sources: list[dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        for policy, rows in source.items():
            out[policy].extend(rows)
    return dict(out)


def static_policy_table(policy_summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        policy: compact_summary(summary)
        for policy, summary in sorted(policy_summaries.items())
        if policy in root_selector.STATIC_POLICIES
        and int(summary.get("surface_count") or 0) > 0
    }


def gated_surface_rows(
    audit_path: Path,
    sage_factor_path: Path,
    gate_path: Path,
) -> list[dict[str, Any]]:
    selected_ids = gate_surface_ids(load_json(gate_path))
    surface_rows = root_selector.build_surface_rows(
        load_json(audit_path),
        load_json(sage_factor_path),
    )
    return [
        row for row in surface_rows if str(row.get("surface_id")) in selected_ids
    ]


def aggregate_surface_rows(
    pairs: list[tuple[str, Path, Path, Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    reports = []
    for label, audit_path, sage_factor_path, gate_path in pairs:
        rows = gated_surface_rows(audit_path, sage_factor_path, gate_path)
        all_rows.extend(rows)
        reports.append(
            {
                "label": label,
                "audit_source": str(audit_path),
                "sage_factor_source": str(sage_factor_path),
                "gate_source": str(gate_path),
                "surface_count": len(rows),
                "target_counts": row_target_counts(rows),
            }
        )
    return all_rows, reports


def order_candidates_with_external_train(
    policy: str,
    row: dict[str, Any],
    external_train_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    candidates = list(row.get("candidates") or [])
    if policy in root_selector.STATIC_POLICIES:
        ordered, _train_count = root_selector.order_candidates(
            policy,
            row,
            [row],
            "transfer_row",
        )
        return ordered, 0
    stats = root_selector.train_stats(external_train_rows)
    if policy in getattr(root_selector, "HYBRID_POLICIES", ()):
        return (
            root_selector.hybrid_order_candidates(
                policy,
                row,
                candidates,
                external_train_rows,
                stats,
            ),
            len(external_train_rows),
        )
    return (
        sorted(
            candidates,
            key=lambda candidate: root_selector.learned_sort_key(
                policy,
                row,
                candidate,
                external_train_rows,
                stats,
            ),
        ),
        len(external_train_rows),
    )


def audit_policy_row_with_external_train(
    policy: str,
    row: dict[str, Any],
    external_train_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered, train_row_count = order_candidates_with_external_train(
        policy,
        row,
        external_train_rows,
    )
    selector_ops = 0
    evaluated = 0
    chosen: dict[str, Any] | None = None
    for candidate in ordered:
        evaluated += 1
        selector_ops += int(candidate.get("selector_eval_ops") or 0)
        if candidate.get("public_zero"):
            chosen = candidate
            break
    if chosen is None:
        selector_ops = sum(int(candidate.get("selector_eval_ops") or 0) for candidate in ordered)
    root_scan_ops = int(chosen.get("factor_root_scan_ops") or 0) if chosen else None
    total_ops = selector_ops + root_scan_ops if root_scan_ops is not None else None
    direct_ops = root_selector.direct_root_recovery_ops(chosen)
    direct_total_ops = selector_ops + direct_ops if direct_ops is not None else None
    rho = int(row.get("generic_rho_steps") or 0)
    chosen_preserves = bool(chosen and chosen.get("preserves_selected_root_pairs"))
    return {
        "policy": policy,
        "policy_type": "static_public_order" if policy in root_selector.STATIC_POLICIES else "external_train_zero_prior",
        "surface_id": row.get("surface_id"),
        "target": row.get("target"),
        "row_key": row.get("row_key"),
        "salt": row.get("salt"),
        "transfer_index": row.get("transfer_index"),
        "p": row.get("p"),
        "candidate_count": int(row.get("candidate_count") or 0),
        "public_zero_root_count": int(row.get("public_zero_root_count") or 0),
        "preserving_root_count": int(row.get("preserving_root_count") or 0),
        "original_selected_root_pair_count": int(
            row.get("original_selected_root_pair_count") or 0
        ),
        "train_row_count": train_row_count,
        "evaluated_root_count": evaluated,
        "selector_eval_ops": selector_ops,
        "root_scan_ops": root_scan_ops,
        "total_ops": total_ops,
        "generic_rho_steps": rho,
        "total_ops_over_rho": round(total_ops / rho, 8)
        if total_ops is not None and rho
        else None,
        "direct_root_recovery_ops": direct_ops,
        "direct_total_ops": direct_total_ops,
        "direct_total_ops_over_rho": round(direct_total_ops / rho, 8)
        if direct_total_ops is not None and rho
        else None,
        "direct_ops_saved_vs_scan": (total_ops - direct_total_ops)
        if total_ops is not None and direct_total_ops is not None
        else None,
        "public_zero_recovered": chosen is not None,
        "chosen_preserves_selected_root_pairs": chosen_preserves,
        "chosen_false_positive": bool(chosen and not chosen_preserves),
        "below_rho": bool(chosen_preserves and total_ops is not None and rho and total_ops < rho),
        "direct_below_rho": bool(
            chosen_preserves
            and direct_total_ops is not None
            and rho
            and direct_total_ops < rho
        ),
        "chosen_candidate": root_selector.compact_candidate(chosen),
        "top_root_sample": [
            {
                "factor_index": int(candidate.get("factor_index") or 0),
                "root": int(candidate.get("root") or 0),
                "public_zero": bool(candidate.get("public_zero")),
                "preserves_selected_root_pairs": bool(
                    candidate.get("preserves_selected_root_pairs")
                ),
            }
            for candidate in ordered[:8]
        ],
    }


def external_train_policy_rows(
    policies: tuple[str, ...],
    holdout_rows: list[dict[str, Any]],
    external_train_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        policy: [
            audit_policy_row_with_external_train(policy, row, external_train_rows)
            for row in holdout_rows
        ]
        for policy in policies
    }


def best_policy_by_mean_cost(policy_summaries: dict[str, dict[str, Any]]) -> str | None:
    candidates = {
        policy: summary
        for policy, summary in policy_summaries.items()
        if int(summary.get("surface_count") or 0) > 0
        and int(summary.get("chosen_false_positive_count") or 0) == 0
        and int(summary.get("chosen_preserving_count") or 0)
        == int(summary.get("surface_count") or 0)
    }
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda policy: (
            float(candidates[policy].get("mean_total_ops_over_rho") or 10**18),
            float(candidates[policy].get("max_total_ops_over_rho") or 10**18),
            -int(candidates[policy].get("below_rho_count") or 0),
            policy,
        ),
    )


def selection_sensitivity_report(
    train_summaries: dict[str, dict[str, Any]],
    external_holdout_summaries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    static_train = {
        policy: summary
        for policy, summary in train_summaries.items()
        if policy in root_selector.STATIC_POLICIES
    }
    rules = {
        "count_first_all_policy": root_selector.best_policy_name(train_summaries),
        "count_first_static_policy": best_static_policy(train_summaries),
        "mean_cost_all_policy": best_policy_by_mean_cost(train_summaries),
        "mean_cost_static_policy": best_policy_by_mean_cost(static_train),
    }
    return {
        rule_name: {
            "selected_policy": policy,
            "train_summary": compact_summary(train_summaries.get(policy or "", {})),
            "external_train_holdout_summary": compact_summary(
                external_holdout_summaries.get(policy or "", {})
            ),
        }
        for rule_name, policy in rules.items()
    }


def validate_preregistered_policies(policies: list[str]) -> tuple[str, ...]:
    unknown = [policy for policy in policies if policy not in root_selector.POLICIES]
    if unknown:
        known = ", ".join(root_selector.POLICIES)
        raise ValueError(
            f"unknown preregistered policy/policies: {', '.join(unknown)}; "
            f"known policies: {known}"
        )
    return tuple(dict.fromkeys(policies))


def preregistered_policy_summaries(
    policies: tuple[str, ...],
    external_holdout_summaries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        policy: compact_summary(external_holdout_summaries.get(policy, {}))
        for policy in policies
    }


def evaluate(
    train_pairs: list[tuple[str, Path, Path]],
    holdout_pairs: list[tuple[str, Path, Path]],
    train_surface_pairs: list[tuple[str, Path, Path, Path]],
    holdout_surface_pairs: list[tuple[str, Path, Path, Path]],
    preregistered_policies: tuple[str, ...] = (),
) -> dict[str, Any]:
    train_reports = []
    train_policy_row_sources = []
    train_policy_summary_sources = []
    for label, selector_path, gate_path in train_pairs:
        report, policy_rows, policy_summaries = source_report(label, selector_path, gate_path)
        train_reports.append(report)
        train_policy_row_sources.append(policy_rows)
        train_policy_summary_sources.append(policy_summaries)

    combined_train_rows = aggregate_policy_rows(train_policy_row_sources)
    combined_train_summaries = summarize_policy_rows(combined_train_rows)
    frozen_policy = best_static_policy(combined_train_summaries)
    frozen_train_rows = combined_train_rows.get(frozen_policy or "", [])
    frozen_train_summary = root_selector.summarize_rows(frozen_train_rows)

    holdout_reports = []
    holdout_policy_row_sources = []
    for label, selector_path, gate_path in holdout_pairs:
        report, policy_rows, _policy_summaries = source_report(
            label,
            selector_path,
            gate_path,
            frozen_policy,
        )
        holdout_reports.append(report)
        holdout_policy_row_sources.append(policy_rows)

    combined_holdout_rows = aggregate_policy_rows(holdout_policy_row_sources)
    combined_holdout_summaries = summarize_policy_rows(combined_holdout_rows)
    frozen_holdout_rows = combined_holdout_rows.get(frozen_policy or "", [])
    frozen_holdout_summary = root_selector.summarize_rows(frozen_holdout_rows)
    posthoc_holdout_best = best_static_policy(combined_holdout_summaries)
    train_surface_rows, train_surface_reports = aggregate_surface_rows(train_surface_pairs)
    holdout_surface_rows, holdout_surface_reports = aggregate_surface_rows(holdout_surface_pairs)
    external_holdout_rows = external_train_policy_rows(
        root_selector.POLICIES,
        holdout_surface_rows,
        train_surface_rows,
    )
    external_holdout_summaries = summarize_policy_rows(external_holdout_rows)
    selection_sensitivity = selection_sensitivity_report(
        combined_train_summaries,
        external_holdout_summaries,
    )
    preregistered_summaries = preregistered_policy_summaries(
        preregistered_policies,
        external_holdout_summaries,
    )

    return {
        "schema": "ecdlp_ffe_frozen_root_policy_evaluator_v1",
        "method": "train_le87_static_public_root_policy_test_holdout",
        "parameters": {
            "train_pairs": [
                {"label": label, "selector_source": str(selector), "gate_source": str(gate)}
                for label, selector, gate in train_pairs
            ],
            "holdout_pairs": [
                {"label": label, "selector_source": str(selector), "gate_source": str(gate)}
                for label, selector, gate in holdout_pairs
            ],
            "train_surface_pairs": [
                {
                    "label": label,
                    "audit_source": str(audit),
                    "sage_factor_source": str(sage),
                    "gate_source": str(gate),
                }
                for label, audit, sage, gate in train_surface_pairs
            ],
            "holdout_surface_pairs": [
                {
                    "label": label,
                    "audit_source": str(audit),
                    "sage_factor_source": str(sage),
                    "gate_source": str(gate),
                }
                for label, audit, sage, gate in holdout_surface_pairs
            ],
            "eligible_policy_family": list(root_selector.STATIC_POLICIES),
            "policy_selection_rule": (
                "Filter each selector bank by the preregistered pre-factor gate, "
                "aggregate train rows, then use the root-selector best_policy "
                "ordering over static public policies only."
            ),
            "excluded_policy_family": list(root_selector.LEARNED_POLICIES),
            "preregistered_policies": list(preregistered_policies),
            "exclusion_reason": (
                "Existing learned-policy selector rows train within each evaluated "
                "bank. This evaluator avoids held-out leakage by selecting only "
                "static public ordering policies."
            ),
        },
        "summary": {
            "frozen_policy": frozen_policy,
            "train_surface_count": int(frozen_train_summary.get("surface_count") or 0),
            "train_target_counts": row_target_counts(frozen_train_rows),
            "train_summary": compact_summary(frozen_train_summary),
            "holdout_surface_count": int(frozen_holdout_summary.get("surface_count") or 0),
            "holdout_target_counts": row_target_counts(frozen_holdout_rows),
            "holdout_summary": compact_summary(frozen_holdout_summary),
            "holdout_posthoc_best_static_policy": posthoc_holdout_best,
            "holdout_posthoc_best_static_summary": compact_summary(
                combined_holdout_summaries.get(posthoc_holdout_best or "", {})
            ),
            "frozen_policy_equals_holdout_posthoc_best_static": frozen_policy == posthoc_holdout_best,
            "interpretation": (
                "This is a frozen-policy audit, not a new factorization run. The "
                "pre-factor gate is chosen from manifest fields, the root policy "
                "is selected from <=87 static public-policy rows, and named "
                "holdout rows are used only for held-out scoring."
            ),
            "external_train_policy_note": (
                "The sensitivity block recomputes held-out learned policies using "
                "only <=87 gated surface rows as the external train set. Selection "
                "rules in that block are candidate next preregistrations unless "
                "they were fixed before a new unseen window."
            ),
            "preregistered_policy_summaries": preregistered_summaries,
        },
        "train_sources": train_reports,
        "holdout_sources": holdout_reports,
        "train_surface_sources": train_surface_reports,
        "holdout_surface_sources": holdout_surface_reports,
        "combined_train_static_policy_summaries": static_policy_table(combined_train_summaries),
        "combined_holdout_static_policy_summaries": static_policy_table(combined_holdout_summaries),
        "combined_train_all_policy_summaries": {
            policy: compact_summary(summary)
            for policy, summary in sorted(combined_train_summaries.items())
        },
        "external_train_holdout_policy_summaries": {
            policy: compact_summary(summary)
            for policy, summary in sorted(external_holdout_summaries.items())
        },
        "preregistered_policy_rows": {
            policy: external_holdout_rows.get(policy, [])
            for policy in preregistered_policies
        },
        "selection_sensitivity": selection_sensitivity,
        "frozen_policy_train_rows": frozen_train_rows,
        "frozen_policy_holdout_rows": frozen_holdout_rows,
        "external_train_holdout_rows": external_holdout_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-pair",
        type=parse_pair,
        action="append",
        default=[],
        help="Calibration pair as label:selector_json:gate_manifest_json.",
    )
    parser.add_argument(
        "--holdout-pair",
        type=parse_pair,
        action="append",
        default=[],
        help="Held-out pair as label:selector_json:gate_manifest_json.",
    )
    parser.add_argument(
        "--train-surface-pair",
        type=parse_surface_pair,
        action="append",
        default=[],
        help="Calibration surface pair as label:audit_json:sage_factor_json:gate_manifest_json.",
    )
    parser.add_argument(
        "--holdout-surface-pair",
        type=parse_surface_pair,
        action="append",
        default=[],
        help="Held-out surface pair as label:audit_json:sage_factor_json:gate_manifest_json.",
    )
    parser.add_argument(
        "--preregistered-policy",
        action="append",
        default=[],
        help=(
            "Policy fixed before an unseen holdout run; may be repeated. "
            "The evaluator reports its external-train holdout summary and rows."
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    train_pairs = args.train_pair or list(DEFAULT_TRAIN_PAIRS)
    holdout_pairs = args.holdout_pair or list(DEFAULT_HOLDOUT_PAIRS)
    train_surface_pairs = args.train_surface_pair or list(DEFAULT_TRAIN_SURFACE_PAIRS)
    holdout_surface_pairs = args.holdout_surface_pair or list(DEFAULT_HOLDOUT_SURFACE_PAIRS)
    preregistered_policies = validate_preregistered_policies(args.preregistered_policy)
    output = evaluate(
        train_pairs,
        holdout_pairs,
        train_surface_pairs,
        holdout_surface_pairs,
        preregistered_policies,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
