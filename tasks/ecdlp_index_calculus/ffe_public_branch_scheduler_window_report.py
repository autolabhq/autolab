#!/usr/bin/env python3
"""Summarize a frozen branch-scheduler validation window.

The scheduler output already records selected cases, but the campaign handoff
needs a stable bucket report: did the frozen target branch get an activation
opportunity, did old branches false-positive, and what was the best public
target near-miss when no verifier-backed row was selected?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_branch_scheduler_window_report.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def public_sort_key(case: dict[str, Any]) -> tuple[Any, ...]:
    try:
        ops = float(case.get("ops_over_rho"))
    except (TypeError, ValueError):
        ops = 10**9
    return (
        ops,
        int(case.get("selected_row_count") or 0),
        int(case.get("selected_leaf_count") or 0),
        int(case.get("top_k") or 0),
        str(case.get("policy") or ""),
        str(case.get("leaf_selector") or ""),
        tuple(
            (
                str(item.get("row_key") or ""),
                tuple(int(leaf) for leaf in item.get("leaf_indices") or []),
            )
            for item in case.get("row_leaf_keys") or []
            if isinstance(item, dict)
        ),
    )


def compact_public_case(case: dict[str, Any], source: Path) -> dict[str, Any]:
    return {
        "source": str(source),
        "target": case.get("target"),
        "transfer_index": int(case.get("transfer_index") or 0),
        "top_k": int(case.get("top_k") or 0),
        "policy": case.get("policy"),
        "row_selector": case.get("row_selector"),
        "leaf_selector": case.get("leaf_selector"),
        "ops_over_rho": case.get("ops_over_rho"),
        "below_rho": bool(case.get("below_rho")),
        "public_key_verified": bool(case.get("public_key_verified")),
        "rank": int(case.get("rank") or 0),
        "relation_count": int(case.get("relation_count") or 0),
        "selected_row_count": int(case.get("selected_row_count") or 0),
        "selected_leaf_count": int(case.get("selected_leaf_count") or 0),
        "unique_leaf_indices": [int(leaf) for leaf in case.get("unique_leaf_indices") or []],
        "row_leaf_keys": case.get("row_leaf_keys") or [],
    }


def collect_public_cases(public_sources: list[Path], target: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cases: list[dict[str, Any]] = []
    source_case_counts: dict[str, int] = {}
    for source in public_sources:
        data = load_json(source)
        source_cases = [case for case in data.get("positive_cases") or [] if isinstance(case, dict)]
        source_case_counts[str(source)] = len(source_cases)
        for case in source_cases:
            if str(case.get("target")) == target:
                cases.append(compact_public_case(case, source))
    return sorted(cases, key=public_sort_key), source_case_counts


def compact_scheduler_case(case: dict[str, Any]) -> dict[str, Any]:
    selection = case.get("public_selection") or {}
    labels = case.get("verifier_labels") or {}
    return {
        "branch": selection.get("branch"),
        "target": selection.get("target"),
        "transfer_index": int(selection.get("transfer_index") or 0),
        "top_k": int(selection.get("top_k") or 0),
        "policy": selection.get("policy"),
        "row_selector": selection.get("row_selector"),
        "leaf_selector": selection.get("leaf_selector"),
        "ops_over_rho": selection.get("ops_over_rho"),
        "selected_row_count": int(selection.get("selected_row_count") or 0),
        "selected_leaf_count": int(selection.get("selected_leaf_count") or 0),
        "unique_leaf_indices": [int(leaf) for leaf in selection.get("unique_leaf_indices") or []],
        "row_leaf_keys": selection.get("row_leaf_keys") or [],
        "below_rho": bool(labels.get("below_rho")),
        "public_key_verified": bool(labels.get("public_key_verified")),
        "rank": int(labels.get("rank") or 0),
        "relation_count": int(labels.get("relation_count") or 0),
    }


def summarize_stress_source(source: Path) -> dict[str, Any]:
    data = load_json(source)
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    return {
        "source": str(source),
        "best_policy": (summary.get("best_policy_summary") or {}).get("policy"),
        "best_policy_summary": summary.get("best_policy_summary"),
        "policy_summaries": summary.get("policy_summaries") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-source", type=Path, action="append", default=[])
    parser.add_argument("--public-source", type=Path, action="append", required=True)
    parser.add_argument("--scheduler-source", type=Path, required=True)
    parser.add_argument("--target", default="67.a1@9803")
    parser.add_argument("--target-branch-prefix", default="67_")
    parser.add_argument("--near-miss-limit", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    scheduler = load_json(args.scheduler_source)
    scheduler_cases = [
        compact_scheduler_case(case)
        for case in scheduler.get("cases") or []
        if isinstance(case, dict)
    ]
    target_scheduler_cases = [
        case for case in scheduler_cases if str(case.get("target")) == str(args.target)
    ]
    target_branch_cases = [
        case
        for case in scheduler_cases
        if str(case.get("branch") or "").startswith(str(args.target_branch_prefix))
    ]
    scheduler_false_positives = [
        case
        for case in scheduler_cases
        if not (bool(case.get("public_key_verified")) and bool(case.get("below_rho")))
    ]
    verified_scheduler_cases = [
        case
        for case in scheduler_cases
        if bool(case.get("public_key_verified")) and bool(case.get("below_rho"))
    ]
    target_public_cases, source_case_counts = collect_public_cases(args.public_source, str(args.target))
    target_cap1_cases = [
        case
        for case in target_public_cases
        if str(case.get("policy") or "") == "fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0"
    ]
    activation_opportunity_count = sum(
        1
        for case in target_branch_cases
        if bool(case.get("public_key_verified")) and bool(case.get("below_rho"))
    )
    if activation_opportunity_count:
        interpretation = (
            "The frozen target branch selected verifier-backed rows; exact line "
            "confirmation and activation replay are the next required checks."
        )
    elif target_branch_cases:
        interpretation = (
            "The frozen target branch selected rows, but none were verifier-backed "
            "below-rho recoveries; keep the activation rule frozen and inspect "
            "line/row separation before adding signatures."
        )
    else:
        interpretation = (
            "The frozen target branch had no activation opportunity in this window. "
            "This is a validation failure for branch reach, not evidence against "
            "the frozen x-match activation rule."
        )
    output = {
        "schema": "ecdlp_ffe_public_branch_scheduler_window_report_v1",
        "method": "bucket_frozen_scheduler_window_before_exact_line_replay",
        "parameters": {
            "stress_sources": [str(source) for source in args.stress_source],
            "public_sources": [str(source) for source in args.public_source],
            "scheduler_source": str(args.scheduler_source),
            "target": args.target,
            "target_branch_prefix": args.target_branch_prefix,
            "near_miss_limit": args.near_miss_limit,
        },
        "summary": {
            "source_case_counts": source_case_counts,
            "scheduler_selected_case_count": len(scheduler_cases),
            "scheduler_verified_below_rho_count": len(verified_scheduler_cases),
            "scheduler_false_positive_count": len(scheduler_false_positives),
            "target_scheduler_case_count": len(target_scheduler_cases),
            "target_branch_case_count": len(target_branch_cases),
            "target_branch_verified_below_rho_count": activation_opportunity_count,
            "target_public_candidate_count": len(target_public_cases),
            "target_cap1_public_candidate_count": len(target_cap1_cases),
            "target_cap1_verified_label_count": sum(
                bool(case.get("public_key_verified")) for case in target_cap1_cases
            ),
            "target_cap1_below_rho_count": sum(bool(case.get("below_rho")) for case in target_cap1_cases),
            "activation_opportunity_count": activation_opportunity_count,
            "interpretation": interpretation,
            "next_obligation": (
                "Do not widen the frozen activation rule from this window. "
                "Either validate branch reach on another future window or move "
                "to public line prediction / amortized exact-line confirmation."
            ),
        },
        "stress_summaries": [summarize_stress_source(source) for source in args.stress_source],
        "scheduler_cases": scheduler_cases,
        "scheduler_false_positive_cases": scheduler_false_positives,
        "target_public_near_misses": target_public_cases[: int(args.near_miss_limit)],
        "target_cap1_near_misses": sorted(target_cap1_cases, key=public_sort_key)[
            : int(args.near_miss_limit)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
