#!/usr/bin/env python3
"""Score a pre-factor selected-hit-root first policy on FFE surfaces.

The preregistered gate manifest records selected public leaf roots before Sage
factorization. This probe tests whether those roots can serve as a public root
locator for the later linear factors ``c + r*b + r^2`` without using preserving
factor labels to order candidates.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import ffe_first_fall_root_hyperplane_selector_probe as root_selector


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_prefactor_hit_root_policy_probe.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def gate_rows(gate_source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(surface.get("surface_id")): surface
        for surface in gate_source.get("surfaces") or []
        if isinstance(surface, dict) and surface.get("pre_factor_gate_selected")
    }


POLICIES = (
    "prefactor_selected_hit_root_first_target_hash",
    "prefactor_unique_leaf_hit_root_first_target_hash",
)


def selected_hit_roots(gate_row: dict[str, Any], p: int) -> set[int]:
    return {
        int(root) % p
        for root in gate_row.get("pre_factor_selected_hit_roots") or []
        if root is not None
    }


def root_leaf_ambiguity(gate_row: dict[str, Any], p: int) -> dict[int, int]:
    leaf_roots: dict[int, set[int]] = defaultdict(set)
    for pair in gate_row.get("pre_factor_selected_hit_root_pairs") or []:
        if not isinstance(pair, dict) or pair.get("root") is None or pair.get("leaf_index") is None:
            continue
        leaf_roots[int(pair["leaf_index"])].add(int(pair["root"]) % p)
    ambiguity: dict[int, int] = {}
    for roots in leaf_roots.values():
        count = len(roots)
        for root in roots:
            ambiguity[root] = min(ambiguity.get(root, count), count)
    return ambiguity


def sort_key(
    policy: str,
    row: dict[str, Any],
    candidate: dict[str, Any],
    hit_roots: set[int],
    ambiguity: dict[int, int],
) -> tuple[Any, ...]:
    root = int(candidate.get("root") or 0)
    in_prefactor_hits = root in hit_roots
    if policy == "prefactor_unique_leaf_hit_root_first_target_hash":
        return (
            0 if in_prefactor_hits else 1,
            int(ambiguity.get(root, 10**9)),
            root_selector.base_sort_key("target_root_hash", row, candidate),
            int(candidate.get("factor_index") or 0),
        )
    return (
        0 if in_prefactor_hits else 1,
        root_selector.base_sort_key("target_root_hash", row, candidate),
        int(candidate.get("factor_index") or 0),
    )


def score_row(policy: str, label: str, row: dict[str, Any], gate_row: dict[str, Any]) -> dict[str, Any]:
    p = int(row.get("p") or 0)
    hit_roots = selected_hit_roots(gate_row, p)
    ambiguity = root_leaf_ambiguity(gate_row, p)
    ordered = sorted(
        row.get("candidates") or [],
        key=lambda candidate: sort_key(policy, row, candidate, hit_roots, ambiguity),
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
    direct_ops = root_selector.direct_root_recovery_ops(chosen)
    total_ops = selector_ops + root_scan_ops if root_scan_ops is not None else None
    direct_total_ops = selector_ops + direct_ops if direct_ops is not None else None
    rho = int(row.get("generic_rho_steps") or 0)
    chosen_preserves = bool(chosen and chosen.get("preserves_selected_root_pairs"))
    return {
        "label": label,
        "policy": policy,
        "surface_id": row.get("surface_id"),
        "target": row.get("target"),
        "row_key": row.get("row_key"),
        "salt": row.get("salt"),
        "transfer_index": row.get("transfer_index"),
        "p": p,
        "pre_factor_selected_hit_roots": sorted(hit_roots),
        "pre_factor_selected_hit_root_ambiguity": {
            str(root): ambiguity[root] for root in sorted(ambiguity)
        },
        "candidate_count": int(row.get("candidate_count") or 0),
        "prefactor_hit_candidate_count": sum(
            int(candidate.get("root") or 0) in hit_roots for candidate in row.get("candidates") or []
        ),
        "public_zero_root_count": int(row.get("public_zero_root_count") or 0),
        "preserving_root_count": int(row.get("preserving_root_count") or 0),
        "original_selected_root_pair_count": int(row.get("original_selected_root_pair_count") or 0),
        "evaluated_root_count": evaluated,
        "selector_eval_ops": selector_ops,
        "root_scan_ops": root_scan_ops,
        "total_ops": total_ops,
        "generic_rho_steps": rho,
        "total_ops_over_rho": round(total_ops / rho, 8) if total_ops is not None and rho else None,
        "direct_root_recovery_ops": direct_ops,
        "direct_total_ops": direct_total_ops,
        "direct_total_ops_over_rho": round(direct_total_ops / rho, 8)
        if direct_total_ops is not None and rho
        else None,
        "public_zero_recovered": chosen is not None,
        "chosen_preserves_selected_root_pairs": chosen_preserves,
        "chosen_false_positive": bool(chosen and not chosen_preserves),
        "below_rho": bool(chosen_preserves and total_ops is not None and rho and total_ops < rho),
        "direct_below_rho": bool(
            chosen_preserves and direct_total_ops is not None and rho and direct_total_ops < rho
        ),
        "chosen_candidate": root_selector.compact_candidate(chosen),
        "top_root_sample": [
            {
                "factor_index": int(candidate.get("factor_index") or 0),
                "root": int(candidate.get("root") or 0),
                "in_pre_factor_selected_hit_roots": int(candidate.get("root") or 0) in hit_roots,
                "pre_factor_selected_hit_root_ambiguity": ambiguity.get(
                    int(candidate.get("root") or 0)
                ),
                "public_zero": bool(candidate.get("public_zero")),
                "preserves_selected_root_pairs": bool(candidate.get("preserves_selected_root_pairs")),
            }
            for candidate in ordered[:8]
        ],
    }


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(row["total_ops_over_rho"])
        for row in rows
        if row.get("total_ops_over_rho") is not None
    ]
    direct_ratios = [
        float(row["direct_total_ops_over_rho"])
        for row in rows
        if row.get("direct_total_ops_over_rho") is not None
    ]
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    by_target: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for counter in (by_label[str(row.get("label"))], by_target[str(row.get("target"))]):
            counter["surface_count"] += 1
            if row.get("chosen_preserves_selected_root_pairs"):
                counter["chosen_preserving_count"] += 1
            if row.get("chosen_false_positive"):
                counter["chosen_false_positive_count"] += 1
            if row.get("below_rho"):
                counter["below_rho_count"] += 1
            if row.get("direct_below_rho"):
                counter["direct_below_rho_count"] += 1
    return {
        "surface_count": len(rows),
        "public_zero_recovered_count": sum(bool(row.get("public_zero_recovered")) for row in rows),
        "chosen_preserving_count": sum(bool(row.get("chosen_preserves_selected_root_pairs")) for row in rows),
        "chosen_false_positive_count": sum(bool(row.get("chosen_false_positive")) for row in rows),
        "below_rho_count": sum(bool(row.get("below_rho")) for row in rows),
        "direct_below_rho_count": sum(bool(row.get("direct_below_rho")) for row in rows),
        "all_surfaces_preserving": bool(rows)
        and all(bool(row.get("chosen_preserves_selected_root_pairs")) for row in rows),
        "all_surfaces_false_positive_free": all(not bool(row.get("chosen_false_positive")) for row in rows),
        "all_surfaces_below_rho": bool(rows) and all(bool(row.get("below_rho")) for row in rows),
        "all_surfaces_direct_below_rho": bool(rows)
        and all(bool(row.get("direct_below_rho")) for row in rows),
        "min_total_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "mean_total_ops_over_rho": mean_or_none(ratios),
        "max_total_ops_over_rho": round(max(ratios), 8) if ratios else None,
        "min_direct_total_ops_over_rho": round(min(direct_ratios), 8) if direct_ratios else None,
        "mean_direct_total_ops_over_rho": mean_or_none(direct_ratios),
        "max_direct_total_ops_over_rho": round(max(direct_ratios), 8) if direct_ratios else None,
        "mean_evaluated_root_count": mean_or_none(
            [float(row.get("evaluated_root_count") or 0) for row in rows]
        ),
        "mean_selector_eval_ops": mean_or_none(
            [float(row.get("selector_eval_ops") or 0) for row in rows]
        ),
        "label_summaries": {label: dict(counter) for label, counter in sorted(by_label.items())},
        "target_summaries": {target: dict(counter) for target, counter in sorted(by_target.items())},
        "interpretation": (
            "This diagnostic orders Sage root hyperplanes whose root is already "
            "present in the preregistered pre-factor selected-hit-root set before "
            "falling back to target_root_hash. It must be preregistered on a new "
            "window before being treated as a promoted policy."
        ),
    }


def evaluate_pair(
    policy: str,
    gate_mode: str,
    label: str,
    audit_path: Path,
    sage_path: Path,
    gate_path: Path,
) -> list[dict[str, Any]]:
    gates = gate_rows(load_json(gate_path))
    surface_rows = root_selector.build_surface_rows(load_json(audit_path), load_json(sage_path))
    rows = []
    for row in surface_rows:
        surface_id = str(row.get("surface_id"))
        gate_row = gates.get(surface_id)
        if gate_row is None:
            continue
        if gate_mode == "single_prefactor_hit_root":
            hit_roots = selected_hit_roots(gate_row, int(row.get("p") or 0))
            if len(hit_roots) != 1:
                continue
        rows.append(score_row(policy, label, row, gate_row))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface-pair",
        type=parse_surface_pair,
        action="append",
        required=True,
        help="Surface pair as label:audit_json:sage_factor_json:gate_manifest_json.",
    )
    parser.add_argument("--policy", choices=POLICIES, default=POLICIES[0])
    parser.add_argument(
        "--gate-mode",
        choices=("all_prefactor_gate", "single_prefactor_hit_root"),
        default="all_prefactor_gate",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = []
    for label, audit_path, sage_path, gate_path in args.surface_pair:
        rows.extend(evaluate_pair(args.policy, args.gate_mode, label, audit_path, sage_path, gate_path))
    output = {
        "schema": "ecdlp_ffe_prefactor_hit_root_policy_probe_v1",
        "method": args.policy,
        "parameters": {
            "policy": args.policy,
            "gate_mode": args.gate_mode,
            "surface_pairs": [
                {
                    "label": label,
                    "audit_source": str(audit),
                    "sage_factor_source": str(sage),
                    "gate_source": str(gate),
                }
                for label, audit, sage, gate in args.surface_pair
            ],
            "ordering": (
                [
                    "candidate root is in pre_factor_selected_hit_roots",
                    "minimum selected-leaf hit-root multiplicity for candidate root",
                    "target_root_hash",
                    "factor_index",
                ]
                if args.policy == "prefactor_unique_leaf_hit_root_first_target_hash"
                else [
                    "candidate root is in pre_factor_selected_hit_roots",
                    "target_root_hash",
                    "factor_index",
                ]
            ),
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
