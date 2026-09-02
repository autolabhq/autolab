#!/usr/bin/env python3
"""Score frozen public branch choices across target-cap1 windows.

The single-target predictor is useful once a branch is known, but the current
campaign boundary is branch selection itself.  This probe reads public-bounded
stress cases and selects rows with frozen branch definitions using only public
fields: target, policy, row selector, top-k, low-term signature, leaf selector,
selected counts, salts, and public cost.  Verifier labels are summarized only
after selection.
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
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_branch_scheduler.json"
DEFAULT_SELECTOR_RE = re.compile(r"^mode_(?:cost_)?low_term_support_total[34]$")
DEFAULT_POLICY = "fixed_target_cap1_ow1_hw3_lw0_sw0_cw0_aw0"
DEFAULT_ROW_SELECTOR = "target_cap1_ow1_hw3_lw0_sw0_cw0_aw0"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def parse_int_set(raw: str | None) -> set[int]:
    if not raw:
        return set()
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def parse_signature(raw: str) -> tuple[str, tuple[int, ...]]:
    mode = "exact"
    value = raw
    if ":" in raw:
        maybe_mode, maybe_value = raw.split(":", 1)
        if maybe_mode in {"exact", "contains"}:
            mode = maybe_mode
            value = maybe_value
    leaves = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not leaves:
        raise argparse.ArgumentTypeError("signature cannot be empty")
    return mode, leaves


def parse_branch(raw: str) -> dict[str, Any]:
    """Parse name|target|top_k_set|signature[|leaf_selector][|policy][|row_selector][|salt_set]."""
    parts = [part.strip() for part in raw.split("|")]
    if len(parts) < 4:
        raise argparse.ArgumentTypeError(
            "branch must be name|target|top_k_set|signature[|leaf_selector][|policy][|row_selector][|salt_set]"
        )
    signature_mode, signature = parse_signature(parts[3])
    return {
        "name": parts[0],
        "target": parts[1],
        "top_k_set": sorted(parse_int_set(parts[2])),
        "signature_mode": signature_mode,
        "signature": list(signature),
        "leaf_selector": parts[4] if len(parts) > 4 and parts[4] else None,
        "policy": parts[5] if len(parts) > 5 and parts[5] else DEFAULT_POLICY,
        "row_selector": parts[6] if len(parts) > 6 and parts[6] else DEFAULT_ROW_SELECTOR,
        "salt_set": sorted(parse_int_set(parts[7])) if len(parts) > 7 and parts[7] else [],
    }


def parse_suppression(raw: str) -> tuple[str, str]:
    parts = [part.strip() for part in raw.split("|")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(
            "suppression must be suppressed_branch|activator_branch"
        )
    return parts[0], parts[1]


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


def public_system_key(branch: dict[str, Any], case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(branch["name"]),
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
    return (
        ops,
        int(case.get("selected_row_count") or 0),
        int(case.get("selected_leaf_count") or 0),
        int(case.get("top_k") or 0),
        str(case.get("leaf_selector") or ""),
        row_profile_key(case),
    )


def case_matches_branch(case: dict[str, Any], branch: dict[str, Any], require_single_row: bool) -> bool:
    if str(case.get("target")) != branch["target"]:
        return False
    if str(case.get("policy")) != branch["policy"]:
        return False
    if str(case.get("row_selector") or "") != branch["row_selector"]:
        return False
    if require_single_row and int(case.get("selected_row_count") or 0) != 1:
        return False
    selector = str(case.get("leaf_selector") or "")
    if branch.get("leaf_selector"):
        if selector != branch["leaf_selector"]:
            return False
    elif not DEFAULT_SELECTOR_RE.fullmatch(selector):
        return False
    if int(case.get("top_k") or 0) not in set(branch["top_k_set"]):
        return False
    salt_set = {int(salt) for salt in branch.get("salt_set") or []}
    if salt_set:
        row_salts = {int(salt) for salt in case.get("row_salts") or []}
        if not row_salts or not row_salts.issubset(salt_set):
            return False
    signature = tuple(int(leaf) for leaf in branch["signature"])
    leaves = leaf_tuple(case)
    if str(branch.get("signature_mode") or "exact") == "contains":
        return set(signature).issubset(leaves)
    return leaves == signature


def compact_case(case: dict[str, Any], source: Path, branch: dict[str, Any]) -> dict[str, Any]:
    public_fields = {
        "source": str(source),
        "branch": branch["name"],
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
        "public_system_key": repr(public_system_key(branch, case)),
        "public_selection": public_fields,
        "verifier_labels": verifier_labels,
    }


def summarize(
    selected: list[dict[str, Any]],
    input_case_count: int,
    source_case_counts: dict[str, int],
    branches: list[dict[str, Any]],
) -> dict[str, Any]:
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
                str(case["public_selection"].get("branch")),
            ),
        )
    branch_counts = Counter(case["public_selection"]["branch"] for case in selected)
    branch_verified = Counter(
        case["public_selection"]["branch"]
        for case in selected
        if case["verifier_labels"]["public_key_verified"]
    )
    branch_verified_below = Counter(case["public_selection"]["branch"] for case in verified_below)
    source_selected = Counter(case["public_selection"]["source"] for case in selected)
    transfer_counts = Counter(str(case["public_selection"]["transfer_index"]) for case in selected)
    return {
        "input_case_count": input_case_count,
        "source_case_counts": source_case_counts,
        "branch_count": len(branches),
        "branch_names": [str(branch["name"]) for branch in branches],
        "selected_case_count": len(selected),
        "selected_verified_label_count": sum(
            1 for case in selected if case["verifier_labels"]["public_key_verified"]
        ),
        "selected_verified_below_rho_count": len(verified_below),
        "selected_rank_positive_count": sum(
            1 for case in selected if int(case["verifier_labels"]["rank"]) > 0
        ),
        "selected_relation_positive_count": sum(
            1 for case in selected if int(case["verifier_labels"]["relation_count"]) > 0
        ),
        "selected_by_branch": dict(sorted(branch_counts.items())),
        "verified_by_branch": dict(sorted(branch_verified.items())),
        "verified_below_rho_by_branch": dict(sorted(branch_verified_below.items())),
        "selected_by_source": dict(sorted(source_selected.items())),
        "selected_transfer_counts": dict(sorted(transfer_counts.items(), key=lambda item: int(item[0]))),
        "abstained": len(selected) == 0,
        "best_verified_below_rho_case": best_verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--branch", type=parse_branch, action="append", required=True)
    parser.add_argument(
        "--suppress-branch-when-active",
        type=parse_suppression,
        action="append",
        default=[],
        help=(
            "public branch precedence rule: suppress_branch|activator_branch. "
            "If the activator branch has any public match in a source, the "
            "suppressed branch is ignored for that source."
        ),
    )
    parser.add_argument("--require-single-row", action="store_true")
    parser.add_argument("--dedupe-systems", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    branch_names = {str(branch["name"]) for branch in args.branch}
    unknown_suppression_names = sorted(
        {
            name
            for pair in args.suppress_branch_when_active
            for name in pair
            if name not in branch_names
        }
    )
    if unknown_suppression_names:
        raise SystemExit(
            "suppression references unknown branches: "
            + ", ".join(unknown_suppression_names)
        )

    selected_raw: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    input_case_count = 0
    source_case_counts: dict[str, int] = {}
    suppressed_by_source: dict[str, Counter[str]] = {}
    for source in args.source:
        data = load_json(source)
        cases = [case for case in data.get("positive_cases") or [] if isinstance(case, dict)]
        source_case_counts[str(source)] = len(cases)
        input_case_count += len(cases)
        source_matches: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
        for case in cases:
            for branch in args.branch:
                if case_matches_branch(case, branch, bool(args.require_single_row)):
                    source_matches.append((source, case, branch))
        active_branches = {str(branch["name"]) for _source, _case, branch in source_matches}
        suppressed_branches = {
            suppressed
            for suppressed, activator in args.suppress_branch_when_active
            if activator in active_branches
        }
        if suppressed_branches:
            counter: Counter[str] = Counter()
            kept_matches: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
            for item in source_matches:
                branch_name = str(item[2]["name"])
                if branch_name in suppressed_branches:
                    counter[branch_name] += 1
                else:
                    kept_matches.append(item)
            suppressed_by_source[str(source)] = counter
            source_matches = kept_matches
        selected_raw.extend(source_matches)

    if args.dedupe_systems:
        by_system: dict[tuple[Any, ...], tuple[Path, dict[str, Any], dict[str, Any]]] = {}
        for source, case, branch in selected_raw:
            key = public_system_key(branch, case)
            current = by_system.get(key)
            if current is None or public_sort_key(case) < public_sort_key(current[1]):
                by_system[key] = (source, case, branch)
        selected_raw = sorted(by_system.values(), key=lambda item: (str(item[2]["name"]), public_sort_key(item[1])))
    else:
        selected_raw.sort(key=lambda item: (str(item[2]["name"]), public_sort_key(item[1])))

    selected = [compact_case(case, source, branch) for source, case, branch in selected_raw]
    result = {
        "schema": "ecdlp_public_branch_scheduler_v1",
        "method": "frozen_public_branch_selector",
        "parameters": {
            "sources": [str(source) for source in args.source],
            "branches": args.branch,
            "suppress_branch_when_active": [
                {"suppressed_branch": suppressed, "activator_branch": activator}
                for suppressed, activator in args.suppress_branch_when_active
            ],
            "require_single_row": bool(args.require_single_row),
            "dedupe_systems": bool(args.dedupe_systems),
            "label_use": "verifier labels are reported after public branch selection and are not used by the scheduler",
        },
        "summary": summarize(selected, input_case_count, source_case_counts, args.branch),
        "suppressed_by_source": {
            source: dict(sorted(counter.items()))
            for source, counter in sorted(suppressed_by_source.items())
        },
        "cases": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
