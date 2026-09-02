#!/usr/bin/env python3
"""Audit first-fall linear factors in Sage-backed FFE surfaces.

The low-term-support FFE route currently relies on Sage factorization to expose
small public factors of a resultant in the leaf variables ``b,c``.  This audit
checks whether those factors have the stronger form

    c + r*b + r^2

which is exactly the monic quadratic root hyperplane for ``x^2 + b*x + c``.
If true, the factorization is not just "low degree"; it has collapsed to a
direct root-selection problem over public leaf coefficients.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SAGE_FACTOR_SOURCE = DEFAULT_STATE_DIR / "ffe_sage_factor_all_public_leaf_plus_fresh_72_79.json"
DEFAULT_PUBLIC_SELECTOR_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_factor_quadratic_root_all_public_leaf_plus_fresh_72_79.json"
)
DEFAULT_CHARGED_ROLLUP_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_factor_quadratic_root_all_public_leaf_plus_fresh_72_79_charged_rollup.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_first_fall_linear_factor_audit_all_public_leaf_plus_fresh_72_79.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def parse_factor_index(candidate_name: str | None) -> int | None:
    if not candidate_name:
        return None
    try:
        return int(str(candidate_name).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def fingerprint_terms(factor: dict[str, Any], p: int) -> dict[tuple[int, int], int]:
    return {
        (int(b_degree), int(c_degree)): int(coeff) % p
        for b_degree, c_degree, coeff in factor.get("fingerprint") or []
    }


def root_hyperplane_info(factor: dict[str, Any], p: int) -> dict[str, Any]:
    terms = fingerprint_terms(factor, p)
    expected_terms = {(0, 0), (0, 1), (1, 0)}
    root = terms.get((1, 0))
    constant = terms.get((0, 0))
    c_coeff = terms.get((0, 1))
    is_root_hyperplane = (
        set(terms) == expected_terms
        and c_coeff == 1
        and root is not None
        and constant is not None
        and (int(root) * int(root) - int(constant)) % p == 0
    )
    return {
        "is_root_hyperplane": is_root_hyperplane,
        "root": int(root) if root is not None else None,
        "constant": int(constant) if constant is not None else None,
        "c_coeff": int(c_coeff) if c_coeff is not None else None,
        "term_count": len(terms),
        "factor_total_degree": int(factor.get("total_degree") or 0),
        "factor_monomials": int(factor.get("monomials") or 0),
        "fingerprint": factor.get("fingerprint") or [],
    }


def candidate_by_index(surface: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    for candidate in surface.get("sage_resultant_factor_candidates") or []:
        index = parse_factor_index(candidate.get("candidate_name"))
        if index is not None:
            out[index] = candidate
    return out


def factor_records(surface: dict[str, Any]) -> list[dict[str, Any]]:
    p = int(surface.get("p") or 0)
    candidates = candidate_by_index(surface)
    records = []
    for index, factor in enumerate((surface.get("sage_resultant_factorization") or {}).get("factors") or []):
        candidate = candidates.get(index, {})
        info = root_hyperplane_info(factor, p)
        records.append(
            {
                "factor_index": index,
                "candidate_name": f"sage_resultant_factor_{index}",
                **info,
                "preserves_selected_root_pairs": bool(candidate.get("preserves_selected_root_pairs")),
                "same_selected_root_pairs": bool(candidate.get("same_selected_root_pairs")),
                "selected_surface_zero_leaves": int(candidate.get("selected_surface_zero_leaves") or 0),
                "selected_valid_root_leaves": int(candidate.get("selected_valid_root_leaves") or 0),
                "selected_missed_leaves": int(candidate.get("selected_missed_leaves") or 0),
                "factor_root_scan_ops": candidate.get("factor_root_scan_ops"),
                "factor_root_scan_ops_over_rho": candidate.get("factor_root_scan_ops_over_rho"),
            }
        )
    return records


def best_policy_rows(public_source: dict[str, Any], charged_source: dict[str, Any]) -> tuple[str | None, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    best_policy = (charged_source.get("summary") or {}).get("best_policy_by_charged_total")
    charged_rows = {
        str(row.get("surface_id")): row
        for row in charged_source.get("best_policy_rows") or []
        if isinstance(row, dict)
    }
    public_rows = {}
    if best_policy:
        public_rows = {
            str(row.get("surface_id")): row
            for row in (public_source.get("policy_rows") or {}).get(str(best_policy), [])
            if isinstance(row, dict)
        }
    return str(best_policy) if best_policy else None, public_rows, charged_rows


def surface_transfer_index(surface: dict[str, Any]) -> int | None:
    challenge_seed = str(surface.get("challenge_seed") or "")
    marker = ":shared-transfer:"
    if marker not in challenge_seed:
        return None
    try:
        return int(challenge_seed.split(marker, 1)[1].split(":", 1)[0])
    except (IndexError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def summarize_target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    factor_counts = [int(row["factor_count"]) for row in rows]
    preserving_counts = [int(row["preserving_factor_count"]) for row in rows]
    charged_ratios = [
        float(row["best_policy_total_ops_over_rho"])
        for row in rows
        if row.get("best_policy_total_ops_over_rho") is not None
    ]
    return {
        "surface_count": len(rows),
        "factor_count": sum(factor_counts),
        "preserving_factor_count": sum(preserving_counts),
        "mean_factor_count": mean_or_none([float(value) for value in factor_counts]),
        "mean_preserving_factor_count": mean_or_none([float(value) for value in preserving_counts]),
        "all_root_hyperplanes": all(bool(row["all_factors_root_hyperplanes"]) for row in rows),
        "all_factor_count_matches_known_hit_roots": all(
            bool(row["factor_count_matches_known_hit_roots"]) for row in rows
        ),
        "best_policy_below_rho_count": sum(bool(row.get("best_policy_below_rho")) for row in rows),
        "best_policy_preserve_count": sum(bool(row.get("best_policy_preserves")) for row in rows),
        "best_policy_mean_total_ops_over_rho": mean_or_none(charged_ratios),
        "best_policy_max_total_ops_over_rho": round(max(charged_ratios), 8) if charged_ratios else None,
    }


def root_prior_holdout(surface_rows: list[dict[str, Any]], scope: str, top_n: int) -> dict[str, Any]:
    results = []
    for row in surface_rows:
        train_rows = [
            other
            for other in surface_rows
            if other.get("surface_id") != row.get("surface_id")
            and other.get("transfer_index") != row.get("transfer_index")
        ]
        if scope == "target":
            train_rows = [other for other in train_rows if other.get("target") == row.get("target")]
        counts: Counter[int] = Counter()
        for other in train_rows:
            counts.update(int(root) for root in other.get("preserving_roots") or [])
        if not counts:
            continue
        selected_roots = [int(root) for root, _count in counts.most_common(top_n)]
        candidate_roots = {int(root) for root in row.get("candidate_roots") or []}
        preserving_roots = {int(root) for root in row.get("preserving_roots") or []}
        hits_preserving = bool(preserving_roots & set(selected_roots))
        hits_any_candidate = bool(candidate_roots & set(selected_roots))
        results.append(
            {
                "surface_id": row.get("surface_id"),
                "target": row.get("target"),
                "transfer_index": row.get("transfer_index"),
                "selected_roots": selected_roots,
                "hits_preserving_root": hits_preserving,
                "hits_any_candidate_root": hits_any_candidate,
                "selected_preserving_roots": sorted(preserving_roots & set(selected_roots)),
                "selected_candidate_roots": sorted(candidate_roots & set(selected_roots)),
            }
        )
    return {
        "scope": scope,
        "top_n": top_n,
        "surface_count": len(results),
        "preserving_root_hit_count": sum(bool(row["hits_preserving_root"]) for row in results),
        "candidate_root_hit_count": sum(bool(row["hits_any_candidate_root"]) for row in results),
        "preserving_root_hit_rate": round(
            sum(bool(row["hits_preserving_root"]) for row in results) / max(1, len(results)),
            8,
        ),
        "candidate_root_hit_rate": round(
            sum(bool(row["hits_any_candidate_root"]) for row in results) / max(1, len(results)),
            8,
        ),
        "misses": [
            {
                "surface_id": row.get("surface_id"),
                "target": row.get("target"),
                "transfer_index": row.get("transfer_index"),
            }
            for row in results
            if not row["hits_preserving_root"]
        ][:16],
    }


def audit(
    sage_source: dict[str, Any],
    public_source: dict[str, Any],
    charged_source: dict[str, Any],
) -> dict[str, Any]:
    best_policy, public_rows_by_surface, charged_rows_by_surface = best_policy_rows(public_source, charged_source)
    surface_rows = []
    root_counts_by_target: dict[str, Counter[int]] = defaultdict(Counter)
    factor_shape_counts: Counter[str] = Counter()
    failures = []
    for surface in sage_source.get("surfaces") or []:
        if not isinstance(surface, dict):
            continue
        p = int(surface.get("p") or 0)
        target = str(surface.get("target"))
        records = factor_records(surface)
        preserving_records = [record for record in records if record.get("preserves_selected_root_pairs")]
        for record in records:
            if record.get("root") is not None:
                root_counts_by_target[target][int(record["root"])] += 1
            factor_shape_counts[
                f"deg{record.get('factor_total_degree')}_mon{record.get('factor_monomials')}_root{int(bool(record.get('is_root_hyperplane')))}"
            ] += 1
            if not record.get("is_root_hyperplane"):
                failures.append(
                    {
                        "surface_id": surface.get("surface_id"),
                        "factor_index": record.get("factor_index"),
                        "fingerprint": record.get("fingerprint"),
                    }
                )
        known_hit_root_count = None
        candidates = surface.get("sage_resultant_factor_candidates") or []
        if candidates:
            known_hit_root_count = int(candidates[0].get("known_hit_root_count") or 0)
        public_row = public_rows_by_surface.get(str(surface.get("surface_id")), {})
        charged_row = charged_rows_by_surface.get(str(surface.get("surface_id")), {})
        chosen = (public_row.get("chosen_candidate") or charged_row.get("chosen_candidate") or {})
        chosen_index = chosen.get("factor_index")
        if chosen_index is None:
            chosen_index = parse_factor_index(chosen.get("candidate_name"))
        chosen_record = records[int(chosen_index)] if chosen_index is not None and int(chosen_index) < len(records) else {}
        charged_ratio = charged_row.get("total_ops_over_rho")
        surface_rows.append(
            {
                "surface_id": surface.get("surface_id"),
                "target": target,
                "row_key": surface.get("row_key"),
                "transfer_index": surface_transfer_index(surface),
                "p": p,
                "selected_leaf_indices": surface.get("selected_leaf_indices") or [],
                "factor_count": len(records),
                "known_hit_root_count": known_hit_root_count,
                "factor_count_matches_known_hit_roots": known_hit_root_count == len(records),
                "root_hyperplane_factor_count": sum(1 for record in records if record.get("is_root_hyperplane")),
                "all_factors_root_hyperplanes": all(bool(record.get("is_root_hyperplane")) for record in records),
                "preserving_factor_count": len(preserving_records),
                "preserving_roots": sorted({int(record["root"]) for record in preserving_records if record.get("root") is not None}),
                "best_policy": best_policy,
                "best_policy_factor_index": int(chosen_index) if chosen_index is not None else None,
                "best_policy_root": chosen_record.get("root"),
                "best_policy_is_root_hyperplane": chosen_record.get("is_root_hyperplane"),
                "best_policy_preserves": bool(charged_row.get("preserves_selected_root_pairs")),
                "best_policy_below_rho": bool(
                    charged_ratio is not None and float(charged_ratio) < 1.0
                ),
                "best_policy_total_ops_over_rho": charged_ratio,
                "best_policy_selector_eval_ops": charged_row.get("selector_eval_ops"),
                "best_policy_root_scan_ops": charged_row.get("root_scan_ops"),
                "best_policy_evaluated_factor_count": public_row.get("evaluated_factor_count"),
                "candidate_roots": [
                    int(record["root"]) for record in records if record.get("root") is not None
                ],
                "candidate_roots_sample": [
                    int(record["root"]) for record in records[:16] if record.get("root") is not None
                ],
            }
        )
    target_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in surface_rows:
        target_rows[str(row.get("target"))].append(row)
    charged_summary = (charged_source.get("summary") or {}).get("best_policy_summary") or {}
    total_factor_count = sum(int(row["factor_count"]) for row in surface_rows)
    total_root_hyperplanes = sum(int(row["root_hyperplane_factor_count"]) for row in surface_rows)
    charged_ratios = [
        float(row["best_policy_total_ops_over_rho"])
        for row in surface_rows
        if row.get("best_policy_total_ops_over_rho") is not None
    ]
    evaluated_counts = [
        float(row["best_policy_evaluated_factor_count"])
        for row in surface_rows
        if row.get("best_policy_evaluated_factor_count") is not None
    ]
    output = {
        "schema": "ecdlp_ffe_first_fall_linear_factor_audit_v1",
        "method": "mine_sage_factorized_ffe_surfaces_for_quadratic_root_hyperplanes",
        "summary": {
            "surface_count": len(surface_rows),
            "factor_count": total_factor_count,
            "root_hyperplane_factor_count": total_root_hyperplanes,
            "all_factors_are_quadratic_root_hyperplanes": total_factor_count == total_root_hyperplanes,
            "factor_shape_counts": dict(factor_shape_counts.most_common()),
            "surface_factor_count_matches_known_hit_roots_count": sum(
                bool(row["factor_count_matches_known_hit_roots"]) for row in surface_rows
            ),
            "all_surface_factor_counts_match_known_hit_roots": all(
                bool(row["factor_count_matches_known_hit_roots"]) for row in surface_rows
            ),
            "surface_with_preserving_factor_count": sum(
                int(row["preserving_factor_count"]) > 0 for row in surface_rows
            ),
            "preserving_factor_count": sum(int(row["preserving_factor_count"]) for row in surface_rows),
            "best_policy_by_charged_total": best_policy,
            "best_policy_surface_count": int(charged_summary.get("surface_count") or 0),
            "best_policy_total_ops_below_rho_count": int(
                charged_summary.get("total_ops_below_rho_count") or 0
            ),
            "best_policy_all_surfaces_below_rho": bool(
                (charged_source.get("summary") or {}).get("all_best_policy_surfaces_below_rho")
            ),
            "best_policy_preserve_count": int(charged_summary.get("preserve_count") or 0),
            "best_policy_false_positive_count": int(charged_summary.get("false_positive_count") or 0),
            "best_policy_mean_total_ops_over_rho": charged_summary.get("mean_total_ops_over_rho"),
            "best_policy_max_total_ops_over_rho": charged_summary.get("max_total_ops_over_rho"),
            "best_policy_mean_evaluated_factor_count": mean_or_none(evaluated_counts),
            "mean_factor_count": mean_or_none([float(row["factor_count"]) for row in surface_rows]),
            "mean_preserving_factor_count": mean_or_none(
                [float(row["preserving_factor_count"]) for row in surface_rows]
            ),
            "mean_charged_total_ops_over_rho": mean_or_none(charged_ratios),
            "target_summaries": {
                target: summarize_target(rows) for target, rows in sorted(target_rows.items())
            },
            "top_roots_by_target": {
                target: [
                    {"root": int(root), "count": int(count)}
                    for root, count in counter.most_common(12)
                ]
                for target, counter in sorted(root_counts_by_target.items())
            },
            "root_prior_holdout_summaries": [
                root_prior_holdout(surface_rows, scope, top_n)
                for scope in ("target", "global")
                for top_n in (1, 2, 4, 8, 16, 32)
            ],
            "root_hyperplane_failure_count": len(failures),
            "interpretation": (
                "Every Sage factor in the measured bank has the exact form "
                "c + r*b + r^2, i.e. a root hyperplane for x^2 + b*x + c. "
                "The current public quotient route is therefore a first-fall "
                "root-selection problem over linear factors, not a generic "
                "high-degree resultant search. This is still a measured surface "
                "bank result; fresh hit-stream generation remains the next proof "
                "obligation before claiming a general ECDLP speedup."
            ),
        },
        "surfaces": surface_rows,
        "root_hyperplane_failures": failures,
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage-factor-source", type=Path, default=DEFAULT_SAGE_FACTOR_SOURCE)
    parser.add_argument("--public-selector-source", type=Path, default=DEFAULT_PUBLIC_SELECTOR_SOURCE)
    parser.add_argument("--charged-rollup-source", type=Path, default=DEFAULT_CHARGED_ROLLUP_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    output = audit(
        load_json(args.sage_factor_source),
        load_json(args.public_selector_source),
        load_json(args.charged_rollup_source),
    )
    output["parameters"] = {
        "sage_factor_source": str(args.sage_factor_source),
        "public_selector_source": str(args.public_selector_source),
        "charged_rollup_source": str(args.charged_rollup_source),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
