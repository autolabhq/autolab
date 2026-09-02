#!/usr/bin/env python3
"""Score a frozen public target-67 target-cap1 family branch.

Exact target-67 branches have been too narrow: fresh windows can recover the
target below rho while shifting leaf tuples and salts.  This probe freezes a
broader public family rule over already materialized public-bounded selector
cases.  It may use relation-system shape such as rank and relation count, but
it does not use the verifier's public_key_verified label to choose cases.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_target67_family_branch_predictor.json"
DEFAULT_SELECTOR_RE = re.compile(r"^mode_(?:cost_)?low_term_support_total[34]$")
DEFAULT_TARGET = "67.a1@9803"
DEFAULT_POLICY = "fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0"
DEFAULT_ROW_SELECTOR = "target_cap1_ow1_hw3_lw0_sw0_cw0_aw0"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def leaf_tuple(case: dict[str, Any]) -> tuple[int, ...]:
    leaves = case.get("unique_leaf_indices")
    if leaves is None:
        leaves = case.get("leaf_indices")
    return tuple(sorted({int(leaf) for leaf in leaves or []}))


def row_profile_key(case: dict[str, Any]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (
            str(item.get("row_key") or ""),
            tuple(sorted({int(leaf) for leaf in item.get("leaf_indices") or []})),
        )
        for item in case.get("row_leaf_keys") or []
        if isinstance(item, dict)
    )


def public_system_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(case.get("target")),
        int(case.get("transfer_index") or 0),
        int(case.get("top_k") or 0),
        str(case.get("policy")),
        str(case.get("leaf_selector") or ""),
        row_profile_key(case),
    )


def public_sort_key(case: dict[str, Any]) -> tuple[Any, ...]:
    try:
        ops = float(case.get("ops_over_rho"))
    except (TypeError, ValueError):
        ops = 10**9
    leaves = leaf_tuple(case)
    return (
        ops,
        int(case.get("selected_row_count") or 0),
        int(case.get("selected_leaf_count") or 0),
        int(case.get("top_k") or 0),
        leaves,
        str(case.get("leaf_selector") or ""),
    )


def transfer_rank_sort_key(case: dict[str, Any]) -> tuple[Any, ...]:
    try:
        ops = float(case.get("ops_over_rho"))
    except (TypeError, ValueError):
        ops = 10**9
    leaves = leaf_tuple(case)
    return (
        -int(case.get("rank") or 0),
        -int(case.get("relation_count") or 0),
        ops,
        int(case.get("top_k") or 0),
        leaves,
        str(case.get("leaf_selector") or ""),
    )


def public_transfer_key(case: dict[str, Any], source: Path) -> tuple[Any, ...]:
    return (
        str(source),
        str(case.get("target")),
        int(case.get("transfer_index") or 0),
        str(case.get("policy")),
        str(case.get("row_selector") or ""),
    )


def low_term_total(case: dict[str, Any]) -> int | None:
    match = re.search(r"total(\d+)$", str(case.get("leaf_selector") or ""))
    return int(match.group(1)) if match else None


def leaf_span(leaves: tuple[int, ...]) -> int:
    return max(leaves) - min(leaves) if leaves else 0


def public_family_rule(case: dict[str, Any], args: argparse.Namespace) -> bool:
    if str(case.get("target")) != args.target:
        return False
    if str(case.get("policy")) != args.policy:
        return False
    if str(case.get("row_selector") or "") != args.row_selector:
        return False
    if int(case.get("selected_row_count") or 0) != args.selected_row_count:
        return False
    if not DEFAULT_SELECTOR_RE.fullmatch(str(case.get("leaf_selector") or "")):
        return False
    try:
        ops = float(case.get("ops_over_rho"))
    except (TypeError, ValueError):
        return False
    if ops > float(args.max_ops_over_rho):
        return False
    if args.require_below_rho and not bool(case.get("below_rho")):
        return False
    if int(case.get("rank") or 0) < int(args.min_rank):
        return False
    if int(case.get("relation_count") or 0) < int(args.min_relation_count):
        return False

    leaves = leaf_tuple(case)
    if args.require_leaf_zero and 0 not in leaves:
        return False

    if args.family_rule == "broad_low_term":
        return True
    if args.family_rule == "leaf0_total4_or_span10":
        total = low_term_total(case)
        return (total == 4 and len(leaves) >= 4) or leaf_span(leaves) >= 10
    raise ValueError(f"unknown family rule: {args.family_rule}")


def compact_case(case: dict[str, Any], source: Path) -> dict[str, Any]:
    leaves = leaf_tuple(case)
    public_fields = {
        "source": str(source),
        "branch": "target67_targetcap1_lowterm_family",
        "target": case.get("target"),
        "transfer_index": int(case.get("transfer_index") or 0),
        "top_k": int(case.get("top_k") or 0),
        "policy": case.get("policy"),
        "row_selector": case.get("row_selector"),
        "leaf_selector": case.get("leaf_selector"),
        "ops_over_rho": case.get("ops_over_rho"),
        "selected_row_count": int(case.get("selected_row_count") or 0),
        "selected_leaf_count": int(case.get("selected_leaf_count") or 0),
        "rank": int(case.get("rank") or 0),
        "relation_count": int(case.get("relation_count") or 0),
        "row_salts": [int(salt) for salt in case.get("row_salts") or []],
        "unique_leaf_indices": list(leaves),
        "leaf_span": leaf_span(leaves),
        "low_term_total": low_term_total(case),
        "row_leaf_keys": case.get("row_leaf_keys") or [],
        "surface_ids": case.get("surface_ids") or [],
    }
    verifier_labels = {
        "below_rho": bool(case.get("below_rho")),
        "public_key_verified": bool(case.get("public_key_verified")),
    }
    return {
        "public_system_key": repr(public_system_key(case)),
        "public_selection": public_fields,
        "verifier_labels": verifier_labels,
    }


def summarize(selected: list[dict[str, Any]], input_case_count: int, source_case_counts: dict[str, int]) -> dict[str, Any]:
    verified_below = [
        case
        for case in selected
        if bool(case["verifier_labels"].get("public_key_verified"))
        and bool(case["verifier_labels"].get("below_rho"))
    ]
    best_verified = None
    if verified_below:
        best_verified = min(
            verified_below,
            key=lambda case: (
                float(case["public_selection"].get("ops_over_rho") or 10**9),
                int(case["public_selection"].get("transfer_index") or 0),
                tuple(case["public_selection"].get("unique_leaf_indices") or []),
            ),
        )
    source_selected = Counter(case["public_selection"]["source"] for case in selected)
    transfer_counts = Counter(str(case["public_selection"]["transfer_index"]) for case in selected)
    selector_counts = Counter(case["public_selection"]["leaf_selector"] for case in selected)
    return {
        "input_case_count": input_case_count,
        "source_case_counts": source_case_counts,
        "selected_case_count": len(selected),
        "selected_verified_label_count": sum(
            1 for case in selected if bool(case["verifier_labels"].get("public_key_verified"))
        ),
        "selected_verified_below_rho_count": len(verified_below),
        "selected_false_positive_count": len(selected) - len(verified_below),
        "selected_by_source": dict(sorted(source_selected.items())),
        "selected_transfer_counts": dict(sorted(transfer_counts.items(), key=lambda item: int(item[0]))),
        "selected_leaf_selector_counts": dict(sorted(selector_counts.items())),
        "best_verified_below_rho_case": best_verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--row-selector", default=DEFAULT_ROW_SELECTOR)
    parser.add_argument(
        "--family-rule",
        choices=["leaf0_total4_or_span10", "broad_low_term"],
        default="leaf0_total4_or_span10",
    )
    parser.add_argument("--max-ops-over-rho", type=float, default=1.0)
    parser.add_argument("--min-rank", type=int, default=2)
    parser.add_argument("--min-relation-count", type=int, default=2)
    parser.add_argument("--selected-row-count", type=int, default=1)
    parser.add_argument("--require-below-rho", action="store_true", default=True)
    parser.add_argument("--allow-over-rho", dest="require_below_rho", action="store_false")
    parser.add_argument("--require-leaf-zero", action="store_true", default=True)
    parser.add_argument("--allow-without-leaf-zero", dest="require_leaf_zero", action="store_false")
    parser.add_argument("--dedupe-systems", action="store_true")
    parser.add_argument(
        "--select-best-per-transfer",
        action="store_true",
        help="after public family filtering, keep the highest-rank/relation candidate per source transfer",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    selected_raw: list[tuple[Path, dict[str, Any]]] = []
    input_case_count = 0
    source_case_counts: dict[str, int] = {}
    for source in args.source:
        data = load_json(source)
        cases = [case for case in data.get("positive_cases") or [] if isinstance(case, dict)]
        source_case_counts[str(source)] = len(cases)
        input_case_count += len(cases)
        selected_raw.extend((source, case) for case in cases if public_family_rule(case, args))

    if args.select_best_per_transfer:
        by_transfer: dict[tuple[Any, ...], tuple[Path, dict[str, Any]]] = {}
        for source, case in selected_raw:
            key = public_transfer_key(case, source)
            current = by_transfer.get(key)
            if current is None or transfer_rank_sort_key(case) < transfer_rank_sort_key(current[1]):
                by_transfer[key] = (source, case)
        selected_raw = sorted(by_transfer.values(), key=lambda item: public_sort_key(item[1]))

    if args.dedupe_systems:
        by_system: dict[tuple[Any, ...], tuple[Path, dict[str, Any]]] = {}
        for source, case in selected_raw:
            key = public_system_key(case)
            current = by_system.get(key)
            if current is None or public_sort_key(case) < public_sort_key(current[1]):
                by_system[key] = (source, case)
        selected_raw = sorted(by_system.values(), key=lambda item: public_sort_key(item[1]))
    else:
        selected_raw.sort(key=lambda item: public_sort_key(item[1]))

    selected = [compact_case(case, source) for source, case in selected_raw]
    result = {
        "schema": "ecdlp_public_target67_family_branch_predictor_v1",
        "method": "frozen_public_targetcap1_lowterm_family_selector",
        "parameters": {
            "sources": [str(source) for source in args.source],
            "target": args.target,
            "policy": args.policy,
            "row_selector": args.row_selector,
            "family_rule": args.family_rule,
            "max_ops_over_rho": args.max_ops_over_rho,
            "min_rank": args.min_rank,
            "min_relation_count": args.min_relation_count,
            "selected_row_count": args.selected_row_count,
            "require_below_rho": bool(args.require_below_rho),
            "require_leaf_zero": bool(args.require_leaf_zero),
            "dedupe_systems": bool(args.dedupe_systems),
            "select_best_per_transfer": bool(args.select_best_per_transfer),
            "label_use": (
                "public_key_verified is reported after selection and is not used; "
                "rank/relation_count are treated as public relation-system shape"
            ),
        },
        "summary": summarize(selected, input_case_count, source_case_counts),
        "cases": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
