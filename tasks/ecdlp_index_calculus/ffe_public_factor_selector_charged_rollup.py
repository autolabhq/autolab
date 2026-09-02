#!/usr/bin/env python3
"""Charge public FFE factor selectors against Sage-backed rho counts.

The quadratic-root selector probe can recover preserving public factors even
when its input signatures do not include full surface records.  In that case
the probe cannot fill its own charged quadratic accounting fields.  This audit
joins the selector rows back to the Sage factorization surface records and
charges:

    selector evaluations + selected factor root scan

against the generic Pollard-rho step count recorded for the same surface.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SAGE_FACTOR_SOURCE = DEFAULT_STATE_DIR / "ffe_sage_factor_all_public_leaf_sources.json"
DEFAULT_PUBLIC_SELECTOR_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_factor_quadratic_root_all_public_leaf_sources.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "ffe_public_factor_quadratic_root_all_public_leaf_sources_charged_rollup.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def surface_rho_steps(sage_source: dict[str, Any]) -> dict[str, int]:
    rho_steps: dict[str, int] = {}
    for surface in sage_source.get("surfaces") or []:
        if not isinstance(surface, dict):
            continue
        candidate = surface.get("best_preserving_candidate") or {}
        surface_id = str(surface.get("surface_id") or "")
        generic_rho_steps = int(candidate.get("generic_rho_steps") or 0)
        if surface_id and generic_rho_steps > 0:
            rho_steps[surface_id] = generic_rho_steps
    return rho_steps


def charged_rows(policy: str, rows: list[dict[str, Any]], rho_steps: dict[str, int]) -> list[dict[str, Any]]:
    charged = []
    for row in rows:
        surface_id = str(row.get("surface_id") or "")
        candidate = row.get("chosen_candidate") or {}
        generic_rho_steps = int(rho_steps.get(surface_id) or 0)
        selector_eval_ops = int(row.get("selector_eval_ops") or 0)
        root_scan_ops = int(candidate.get("factor_root_scan_ops") or 0)
        total_ops = selector_eval_ops + root_scan_ops
        total_ops_over_rho = total_ops / generic_rho_steps if generic_rho_steps else None
        charged.append(
            {
                "surface_id": surface_id,
                "target": row.get("target"),
                "row_key": row.get("row_key"),
                "transfer_index": int(row.get("transfer_index") or 0),
                "policy": policy,
                "candidate_name": candidate.get("candidate_name"),
                "factor_index": candidate.get("factor_index"),
                "selector_eval_ops": selector_eval_ops,
                "root_scan_ops": root_scan_ops,
                "total_ops": total_ops,
                "generic_rho_steps": generic_rho_steps,
                "total_ops_over_rho": round(total_ops_over_rho, 8)
                if total_ops_over_rho is not None
                else None,
                "root_scan_ops_over_rho": candidate.get("factor_root_scan_ops_over_rho"),
                "public_zero_recovered": bool(row.get("public_zero_recovered")),
                "preserves_selected_root_pairs": bool(
                    row.get("chosen_preserves_selected_root_pairs_source")
                ),
                "false_positive_source": bool(row.get("chosen_false_positive_source")),
                "surface_record_present": bool(row.get("surface_record_present")),
            }
        )
    return charged


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_rows(policy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(row["total_ops_over_rho"])
        for row in rows
        if row.get("total_ops_over_rho") is not None
    ]
    target_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        target = str(row.get("target"))
        target_counts[target]["surface_count"] += 1
        if row.get("total_ops_over_rho") is not None and row["total_ops_over_rho"] < 1:
            target_counts[target]["below_rho_count"] += 1
        if row.get("preserves_selected_root_pairs"):
            target_counts[target]["preserve_count"] += 1
        if row.get("false_positive_source"):
            target_counts[target]["false_positive_count"] += 1
    return {
        "policy": policy,
        "surface_count": len(rows),
        "charged_surface_count": len(ratios),
        "total_ops_below_rho_count": sum(ratio < 1 for ratio in ratios),
        "preserve_count": sum(bool(row.get("preserves_selected_root_pairs")) for row in rows),
        "false_positive_count": sum(bool(row.get("false_positive_source")) for row in rows),
        "min_total_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "mean_total_ops_over_rho": round(mean(ratios), 8) if ratios else None,
        "max_total_ops_over_rho": round(max(ratios), 8) if ratios else None,
        "mean_selector_eval_ops": round(
            mean([float(row["selector_eval_ops"]) for row in rows]), 8
        )
        if rows
        else None,
        "target_summaries": {
            target: dict(counter) for target, counter in sorted(target_counts.items())
        },
    }


def best_policy_key(item: tuple[str, dict[str, Any]]) -> tuple[int, int, int, float, float, str]:
    policy, summary = item
    surface_count = int(summary["surface_count"])
    preserve_count = int(summary["preserve_count"])
    false_positive_count = int(summary["false_positive_count"])
    below_rho_count = int(summary["total_ops_below_rho_count"])
    max_ratio = float(summary["max_total_ops_over_rho"] or 10**9)
    mean_ratio = float(summary["mean_total_ops_over_rho"] or 10**9)
    all_preserve = int(preserve_count == surface_count)
    no_false_positive = int(false_positive_count == 0)
    return (below_rho_count, all_preserve, no_false_positive, -max_ratio, -mean_ratio, policy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage-factor-source", type=Path, default=DEFAULT_SAGE_FACTOR_SOURCE)
    parser.add_argument(
        "--public-selector-source",
        type=Path,
        default=DEFAULT_PUBLIC_SELECTOR_SOURCE,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sage_source = load_json(args.sage_factor_source)
    selector_source = load_json(args.public_selector_source)
    rho_steps = surface_rho_steps(sage_source)
    policy_rows = selector_source.get("policy_rows") or {}
    charged_by_policy = {
        str(policy): charged_rows(str(policy), rows, rho_steps)
        for policy, rows in policy_rows.items()
        if isinstance(rows, list)
    }
    policy_summaries = {
        policy: summarize_rows(policy, rows)
        for policy, rows in sorted(charged_by_policy.items())
    }
    if not policy_summaries:
        raise SystemExit("public selector source has no policy rows")
    best_policy, best_summary = max(policy_summaries.items(), key=best_policy_key)
    best_rows = sorted(
        charged_by_policy[best_policy],
        key=lambda row: (
            float(row["total_ops_over_rho"])
            if row.get("total_ops_over_rho") is not None
            else -1.0
        ),
        reverse=True,
    )
    output = {
        "schema": "ecdlp_public_factor_selector_charged_cost_rollup_v1",
        "method": "selector_eval_plus_root_scan_charged_against_sage_generic_rho_steps",
        "parameters": {
            "sage_factor_source": str(args.sage_factor_source),
            "public_selector_source": str(args.public_selector_source),
        },
        "summary": {
            "surface_count": int(best_summary["surface_count"]),
            "policy_count": len(policy_summaries),
            "best_policy_by_charged_total": best_policy,
            "best_policy_summary": best_summary,
            "all_best_policy_surfaces_below_rho": best_summary[
                "total_ops_below_rho_count"
            ]
            == best_summary["surface_count"],
            "all_best_policy_surfaces_preserve": best_summary["preserve_count"]
            == best_summary["surface_count"],
            "best_policy_false_positive_free": best_summary["false_positive_count"] == 0,
            "interpretation": (
                "A surface is counted below rho only when public selector "
                "evaluations plus the selected factor root scan are lower than "
                "the Sage-recorded generic Pollard-rho step count. This is a "
                "charged follow-up to the public selector probe for sources that "
                "lack full surface records."
            ),
        },
        "policy_summaries": policy_summaries,
        "best_policy_rows": best_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
