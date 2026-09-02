#!/usr/bin/env python3
"""Select low-term total-k stress cases using public row/leaf fields only.

The strict signature aggregator is useful for measuring positives, but it
classifies cases with verifier labels.  This probe emits the same downstream
``positive_cases`` shape from raw stress output while filtering and ranking only
on public fields: target, transfer, policy, row selector, leaf selector,
row/leaf keys, selected counts, and public ops/rho.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_STRESS_SOURCE = (
    DEFAULT_STATE_DIR
    / "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total3_total4_fixed_selector_192_199_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total3_total4_public_stress_selector_192_199.json"
DEFAULT_SELECTOR_RE = re.compile(r"^mode_(?:cost_)?low_term_support_total[0-9]+$")
DEFAULT_SEED = "ecdlp-frontier-signed-dual-sieve-v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def parse_allowed_selectors(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def row_salt(row_key: str) -> int | None:
    try:
        return int(str(row_key).rsplit("salt", 1)[1])
    except (IndexError, ValueError):
        return None


def inferred_surface_id(target: str, transfer_index: int, row_key: str, seed: str) -> str:
    challenge_seed = f"{seed}:shared-transfer:{transfer_index}:{target}"
    return f"{target}|{row_key}|{challenge_seed}"


def selector_allowed(selector: str, allowed: set[str]) -> bool:
    return selector in allowed if allowed else bool(DEFAULT_SELECTOR_RE.fullmatch(selector))


def compact_case(
    source: Path,
    policy: str,
    row_selector: str | None,
    row: dict[str, Any],
    seed: str,
) -> dict[str, Any] | None:
    target = str(row.get("target") or "")
    transfer_index = int(row.get("transfer_index") or 0)
    row_leaf_keys = []
    leaf_indices = []
    row_salts = []
    surface_ids = []
    for item in row.get("row_leaf_keys") or []:
        if not isinstance(item, dict):
            continue
        row_key = str(item.get("row_key") or "")
        leaves = [int(leaf) for leaf in item.get("leaf_indices") or []]
        if not row_key or not leaves:
            continue
        salt = row_salt(row_key)
        surface_id = inferred_surface_id(target, transfer_index, row_key, seed)
        row_leaf_keys.append(
            {
                "row_key": row_key,
                "surface_id": surface_id,
                "salt": salt,
                "leaf_indices": leaves,
            }
        )
        surface_ids.append(surface_id)
        leaf_indices.extend(leaves)
        if salt is not None:
            row_salts.append(salt)
    if not row_leaf_keys:
        return None
    return {
        "source": str(source),
        "policy": str(policy),
        "row_selector": row_selector,
        "leaf_selector": row.get("selector"),
        "target": target,
        "transfer_index": transfer_index,
        "top_k": int(row.get("top_k") or 0),
        "ops_over_rho": row.get("ops_over_rho"),
        "below_rho": bool(row.get("below_rho")),
        "public_key_verified": bool(row.get("public_key_verified")),
        "selected_leaf_count": int(row.get("selected_leaf_count") or 0),
        "selected_row_count": int(row.get("selected_row_count") or 0),
        "relation_count": int(row.get("relation_count") or 0),
        "rank": int(row.get("rank") or 0),
        "row_salts": row_salts,
        "leaf_indices": leaf_indices,
        "unique_leaf_indices": sorted(set(leaf_indices)),
        "surface_ids": sorted(set(surface_ids)),
        "row_leaf_keys": row_leaf_keys,
        "selection_basis": {
            "selected_by": "public_stress_row_leaf_cost",
            "ops_over_rho": row.get("ops_over_rho"),
            "below_rho": bool(row.get("below_rho")),
            "selected_row_count": int(row.get("selected_row_count") or 0),
            "selected_leaf_count": int(row.get("selected_leaf_count") or 0),
        },
    }


def case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    leaf_signature = tuple(
        (
            str(item.get("row_key")),
            tuple(int(leaf) for leaf in item.get("leaf_indices") or []),
        )
        for item in case.get("row_leaf_keys") or []
        if isinstance(item, dict)
    )
    return (
        str(case.get("target")),
        int(case.get("transfer_index") or 0),
        int(case.get("top_k") or 0),
        str(case.get("policy")),
        str(case.get("leaf_selector")),
        leaf_signature,
    )


def public_sort_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(case.get("ops_over_rho") or 10**9),
        int(case.get("selected_row_count") or 0),
        int(case.get("selected_leaf_count") or 0),
        int(case.get("top_k") or 0),
        str(case.get("policy")),
        str(case.get("leaf_selector")),
        tuple(str(item.get("row_key")) for item in case.get("row_leaf_keys") or []),
        tuple(tuple(int(leaf) for leaf in item.get("leaf_indices") or []) for item in case.get("row_leaf_keys") or []),
    )


def collect_cases(
    stress_source: Path,
    stress: dict[str, Any],
    allowed_selectors: set[str],
    max_ops_over_rho: float,
    require_below_rho: bool,
    seed: str,
) -> list[dict[str, Any]]:
    cases_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for policy, result in (stress.get("policies") or {}).items():
        if not isinstance(result, dict):
            continue
        row_selector = ((result.get("row_selector") or {}).get("selector")) or None
        for row in result.get("stress_leaf_results") or []:
            if not isinstance(row, dict):
                continue
            selector = str(row.get("selector") or "")
            if not selector_allowed(selector, allowed_selectors):
                continue
            if require_below_rho and not bool(row.get("below_rho")):
                continue
            try:
                ops_over_rho = float(row.get("ops_over_rho"))
            except (TypeError, ValueError):
                continue
            if ops_over_rho >= max_ops_over_rho:
                continue
            case = compact_case(stress_source, str(policy), row_selector, row, seed)
            if case is None:
                continue
            key = case_key(case)
            old = cases_by_key.get(key)
            if old is None or public_sort_key(case) < public_sort_key(old):
                cases_by_key[key] = case
    return sorted(cases_by_key.values(), key=public_sort_key)


def public_selected_cases(
    cases: list[dict[str, Any]],
    max_cases_per_challenge: int,
) -> list[dict[str, Any]]:
    if max_cases_per_challenge <= 0:
        return cases
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[(str(case.get("target")), int(case.get("transfer_index") or 0))].append(case)
    selected = []
    for rows in grouped.values():
        selected.extend(sorted(rows, key=public_sort_key)[:max_cases_per_challenge])
    return sorted(selected, key=public_sort_key)


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def summarize(
    cases: list[dict[str, Any]],
    all_public_cases: list[dict[str, Any]],
    require_below_rho: bool,
) -> dict[str, Any]:
    ratios = [float(case["ops_over_rho"]) for case in cases if case.get("ops_over_rho") is not None]
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_policy: Counter[str] = Counter()
    by_row_selector: Counter[str] = Counter()
    by_leaf_selector: Counter[str] = Counter()
    for case in cases:
        by_target[str(case.get("target"))].append(case)
        by_policy[str(case.get("policy"))] += 1
        by_row_selector[str(case.get("row_selector"))] += 1
        by_leaf_selector[str(case.get("leaf_selector"))] += 1
    target_summaries = []
    for target, rows in sorted(by_target.items()):
        best = min(rows, key=public_sort_key)
        target_summaries.append(
            {
                "target": target,
                "case_count": len(rows),
                "transfer_indices": sorted({int(row.get("transfer_index") or 0) for row in rows}),
                "best_ops_over_rho": best.get("ops_over_rho"),
                "best_case": best,
            }
        )
    return {
        "selected_public_case_count": len(cases),
        "all_public_candidate_case_count": len(all_public_cases),
        (
            "all_public_below_rho_case_count"
            if require_below_rho
            else "all_public_bounded_case_count"
        ): len(all_public_cases),
        "selected_signature_verified_label_count": sum(
            bool(case.get("public_key_verified")) for case in cases
        ),
        "selected_min_ops_over_rho": round(min(ratios), 8) if ratios else None,
        "selected_mean_ops_over_rho": mean_or_none(ratios),
        "selected_max_ops_over_rho": round(max(ratios), 8) if ratios else None,
        "target_count": len(by_target),
        "target_summaries": target_summaries,
        "policy_counts": dict(by_policy.most_common()),
        "row_selector_counts": dict(by_row_selector.most_common()),
        "leaf_selector_counts": dict(by_leaf_selector.most_common()),
        "positive_source": (
            "public_below_rho_stress_rows"
            if require_below_rho
            else "public_bounded_stress_rows"
        ),
        "positive_source_case_count": len(cases),
        "positive_source_surface_ids": sorted(
            {
                str(surface_id)
                for case in cases
                for surface_id in case.get("surface_ids") or []
            }
        ),
        "interpretation": (
            "Cases are selected from raw stress rows by public cost "
            "and deterministic public tie-breaks. Verifier fields are labels "
            "only and are not part of the selection rule."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-source", type=Path, default=DEFAULT_STRESS_SOURCE)
    parser.add_argument("--allowed-leaf-selectors")
    parser.add_argument("--max-ops-over-rho", type=float, default=1.0)
    parser.add_argument("--max-cases-per-challenge", type=int, default=0)
    parser.add_argument(
        "--allow-over-rho",
        action="store_true",
        help="Permit public candidates above rho as long as ops_over_rho is below the threshold.",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    stress = load_json(args.stress_source)
    allowed = parse_allowed_selectors(args.allowed_leaf_selectors)
    all_cases = collect_cases(
        args.stress_source,
        stress,
        allowed,
        float(args.max_ops_over_rho),
        not bool(args.allow_over_rho),
        str(args.seed),
    )
    selected_cases = public_selected_cases(all_cases, int(args.max_cases_per_challenge))
    output = {
        "schema": "ecdlp_low_term_totalk_public_stress_selector_probe_v1",
        "method": "public_below_rho_total_k_row_leaf_stress_selector",
        "parameters": {
            "stress_source": str(args.stress_source),
            "allowed_leaf_selectors": sorted(allowed),
            "default_selector_regex": DEFAULT_SELECTOR_RE.pattern if not allowed else None,
            "max_ops_over_rho": args.max_ops_over_rho,
            "max_cases_per_challenge": args.max_cases_per_challenge,
            "require_below_rho": not bool(args.allow_over_rho),
            "seed": args.seed,
            "selection_rule": [
                (
                    "require raw stress below_rho true"
                    if not args.allow_over_rho
                    else "allow raw stress below_rho false when bounded by max_ops_over_rho"
                ),
                "require ops_over_rho below threshold",
                "rank by ops_over_rho, selected row count, selected leaf count, top_k, policy, leaf selector",
            ],
            "forbidden_selection_fields": [
                "public_key_verified",
                "relation_count",
                "rank",
                "source_verified",
                "row_selector_public_key_verified",
            ],
        },
        "positive_cases": selected_cases,
        "diagnostics": {
            "all_public_below_rho_cases_sample": all_cases[:16],
        },
        "summary": summarize(selected_cases, all_cases, not bool(args.allow_over_rho)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
