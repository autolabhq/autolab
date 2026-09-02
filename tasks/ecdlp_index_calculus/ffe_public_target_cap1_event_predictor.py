#!/usr/bin/env python3
"""Freeze and score a public target-cap1 low-term event predictor.

This probe is intentionally narrower than the full public-bounded selector.
It reads already materialized public-bounded cases and selects candidate
systems using only public fields: target, row policy, leaf selector, top-k,
row/leaf profile, and public cost.  Verifier labels are summarized after the
selection and must not be used to choose rows in a held-out window.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_target_cap1_event_predictor.json"
DEFAULT_SELECTOR_RE = re.compile(r"^mode_(?:cost_)?low_term_support_total[34]$")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def parse_signature(raw: str) -> tuple[int, ...]:
    leaves = tuple(sorted({int(item) for item in raw.split(",") if item.strip()}))
    if not leaves:
        raise argparse.ArgumentTypeError("signature cannot be empty")
    return leaves


def parse_int_set(raw: str | None) -> set[int]:
    if not raw:
        return set()
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


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
        row_profile_key(case),
    )


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
        str(case.get("leaf_selector")),
        row_profile_key(case),
    )


def case_matches(case: dict[str, Any], args: argparse.Namespace, signatures: set[tuple[int, ...]], top_ks: set[int]) -> bool:
    if str(case.get("target")) != args.target:
        return False
    if str(case.get("policy")) != args.policy:
        return False
    row_selector = str(case.get("row_selector") or "")
    if args.row_selector and row_selector != args.row_selector:
        return False
    selector = str(case.get("leaf_selector") or "")
    if args.leaf_selector and selector != args.leaf_selector:
        return False
    if not args.leaf_selector and not DEFAULT_SELECTOR_RE.fullmatch(selector):
        return False
    if args.require_single_row and int(case.get("selected_row_count") or 0) != 1:
        return False
    if top_ks and int(case.get("top_k") or 0) not in top_ks:
        return False
    return leaf_tuple(case) in signatures


def compact_case(case: dict[str, Any], source: Path) -> dict[str, Any]:
    public_fields = {
        "source": str(source),
        "target": case.get("target"),
        "transfer_index": int(case.get("transfer_index") or 0),
        "top_k": int(case.get("top_k") or 0),
        "policy": case.get("policy"),
        "row_selector": case.get("row_selector"),
        "leaf_selector": case.get("leaf_selector"),
        "ops_over_rho": case.get("ops_over_rho"),
        "selected_row_count": int(case.get("selected_row_count") or 0),
        "selected_leaf_count": int(case.get("selected_leaf_count") or 0),
        "row_salts": [int(salt) for salt in case.get("row_salts") or []],
        "unique_leaf_indices": list(leaf_tuple(case)),
        "row_leaf_keys": case.get("row_leaf_keys") or [],
        "surface_ids": case.get("surface_ids") or [],
    }
    verifier_labels = {
        "below_rho": bool(case.get("below_rho")),
        "public_key_verified": bool(case.get("public_key_verified")),
        "rank": int(case.get("rank") or 0),
        "relation_count": int(case.get("relation_count") or 0),
    }
    return {
        "public_system_key": repr(public_system_key(case)),
        "public_selection": public_fields,
        "verifier_labels": verifier_labels,
    }


def summarize(selected: list[dict[str, Any]], input_case_count: int, source_case_counts: dict[str, int]) -> dict[str, Any]:
    selected_labels = [case["verifier_labels"] for case in selected]
    verified = [case for case in selected if case["verifier_labels"]["public_key_verified"]]
    verified_below = [
        case
        for case in selected
        if case["verifier_labels"]["public_key_verified"] and case["verifier_labels"]["below_rho"]
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
    signatures = Counter(
        ",".join(str(leaf) for leaf in case["public_selection"]["unique_leaf_indices"]) for case in selected
    )
    transfer_counts = Counter(str(case["public_selection"]["transfer_index"]) for case in selected)
    source_selected = Counter(case["public_selection"]["source"] for case in selected)
    return {
        "input_case_count": input_case_count,
        "source_case_counts": source_case_counts,
        "selected_case_count": len(selected),
        "selected_verified_label_count": sum(1 for label in selected_labels if label["public_key_verified"]),
        "selected_verified_below_rho_count": len(verified_below),
        "selected_rank_positive_count": sum(1 for label in selected_labels if int(label["rank"]) > 0),
        "selected_relation_positive_count": sum(1 for label in selected_labels if int(label["relation_count"]) > 0),
        "selected_signature_counts": dict(sorted(signatures.items())),
        "selected_transfer_counts": dict(sorted(transfer_counts.items(), key=lambda item: int(item[0]))),
        "selected_by_source": dict(sorted(source_selected.items())),
        "best_verified_below_rho_case": best_verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--target", default="22050.cf1@11731")
    parser.add_argument("--policy", default="fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0")
    parser.add_argument("--row-selector", default="target_cap1_ow1_hw3_lw0_sw0_cw0_aw0")
    parser.add_argument("--leaf-selector")
    parser.add_argument("--signature", action="append", type=parse_signature, required=True)
    parser.add_argument("--top-k-set")
    parser.add_argument("--require-single-row", action="store_true")
    parser.add_argument("--dedupe-systems", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    signatures = set(args.signature)
    top_ks = parse_int_set(args.top_k_set)

    selected_raw: list[tuple[Path, dict[str, Any]]] = []
    input_case_count = 0
    source_case_counts: dict[str, int] = {}
    for source in args.source:
        data = load_json(source)
        cases = [case for case in data.get("positive_cases") or [] if isinstance(case, dict)]
        source_case_counts[str(source)] = len(cases)
        input_case_count += len(cases)
        for case in cases:
            if case_matches(case, args, signatures, top_ks):
                selected_raw.append((source, case))

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
        "schema": "ecdlp_public_target_cap1_event_predictor_v1",
        "method": "frozen_public_signature_selector",
        "parameters": {
            "sources": [str(source) for source in args.source],
            "target": args.target,
            "policy": args.policy,
            "row_selector": args.row_selector,
            "leaf_selector": args.leaf_selector,
            "signatures": [list(signature) for signature in sorted(signatures)],
            "top_k_set": sorted(top_ks),
            "require_single_row": bool(args.require_single_row),
            "dedupe_systems": bool(args.dedupe_systems),
            "label_use": "verifier labels are reported after public selection and are not used by the selector",
        },
        "summary": summarize(selected, input_case_count, source_case_counts),
        "cases": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
