#!/usr/bin/env python3
"""Build a direct public fourth-leaf trigger bridge from total3/total4 cases.

This probe is the fresh-window counterpart to the retrospective trigger audit:
it does not need FFE root-policy anchors.  It pairs public total3 cases with
same-row public total4 lifts and retains only the marginal rows whose leaf set
adds the declared fourth-leaf trigger.

Verifier labels are copied for reporting only.  They are not used for pairing,
filtering, ranking, or bridge emission.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SIGNATURE_SOURCE = DEFAULT_STATE_DIR / "low_term_total3_total4_public_bounded_full_selector_208_215.json"
DEFAULT_OUT = (
    DEFAULT_STATE_DIR
    / "ffe_public_fourth_leaf_trigger_bridge_total3_total4_public_bounded_full_base90_add34_208_215.json"
)
DEFAULT_SEED = "ecdlp-frontier-signed-dual-sieve-v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def int_list(values: Any) -> list[int]:
    out = []
    for value in values or []:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def row_schedule_key(row_key: str) -> str:
    if ":salt" in str(row_key):
        return str(row_key).rsplit(":salt", 1)[0]
    return str(row_key)


def row_salt(row_key: str) -> int | None:
    try:
        return int(str(row_key).rsplit("salt", 1)[1])
    except (IndexError, ValueError):
        return None


def signature_surface_id(case: dict[str, Any], row_key: str, seed: str) -> str:
    target = str(case.get("target"))
    transfer_index = int(case.get("transfer_index") or 0)
    challenge_seed = f"{seed}:shared-transfer:{transfer_index}:{target}"
    return f"{target}|{row_key}|{challenge_seed}"


def normalized_row_leaf_keys(case: dict[str, Any], seed: str) -> list[dict[str, Any]]:
    out = []
    for item in case.get("row_leaf_keys") or []:
        if not isinstance(item, dict):
            continue
        row_key = str(item.get("row_key") or "")
        leaves = int_list(item.get("leaf_indices"))
        if not row_key or not leaves:
            continue
        surface_id = str(item.get("surface_id") or signature_surface_id(case, row_key, seed))
        out.append(
            {
                "row_key": row_key,
                "row_schedule_key": row_schedule_key(row_key),
                "surface_id": surface_id,
                "salt": item.get("salt") if item.get("salt") is not None else row_salt(row_key),
                "leaf_indices": sorted(leaves),
            }
        )
    return out


def row_key_tuple(case: dict[str, Any], seed: str) -> tuple[str, ...]:
    return tuple(str(item.get("row_key")) for item in normalized_row_leaf_keys(case, seed))


def case_key(case: dict[str, Any], seed: str) -> tuple[Any, ...]:
    row_leaf_signature = tuple(
        (
            str(item.get("row_key")),
            tuple(int(leaf) for leaf in item.get("leaf_indices") or []),
        )
        for item in normalized_row_leaf_keys(case, seed)
    )
    return (
        str(case.get("target")),
        int(case.get("transfer_index") or 0),
        int(case.get("top_k") or 0),
        str(case.get("policy")),
        str(case.get("leaf_selector") or case.get("selector")),
        row_leaf_signature,
    )


def case_key_string(case: dict[str, Any], seed: str) -> str:
    return "|".join(str(part) for part in case_key(case, seed))


def base_key(case: dict[str, Any], seed: str) -> tuple[Any, ...]:
    return (
        str(case.get("target")),
        int(case.get("transfer_index") or 0),
        int(case.get("top_k") or 0),
        str(case.get("policy")),
        row_key_tuple(case, seed),
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
        str(case.get("leaf_selector")),
    )


def best_total3_bases(cases: list[dict[str, Any]], seed: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    bases: dict[tuple[Any, ...], dict[str, Any]] = {}
    for case in cases:
        if "total3" not in str(case.get("leaf_selector") or case.get("selector") or ""):
            continue
        key = base_key(case, seed)
        old = bases.get(key)
        if old is None or public_sort_key(case) < public_sort_key(old):
            bases[key] = case
    return bases


def trigger_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target": args.trigger_target,
        "base_leaf_indices": sorted(int(leaf) for leaf in args.trigger_base_leaf),
        "added_leaf_indices": sorted(int(leaf) for leaf in args.trigger_added_leaf),
        "top_k": args.trigger_top_k,
        "policy": args.trigger_policy,
        "leaf_selector": args.trigger_leaf_selector,
    }


def matches_trigger(case: dict[str, Any], marginal: dict[str, Any], trigger: dict[str, Any]) -> bool:
    if trigger.get("target") and str(case.get("target")) != str(trigger["target"]):
        return False
    if trigger.get("top_k") is not None and int(case.get("top_k") or 0) != int(trigger["top_k"]):
        return False
    if trigger.get("policy") and str(case.get("policy")) != str(trigger["policy"]):
        return False
    if trigger.get("leaf_selector") and str(case.get("leaf_selector")) != str(trigger["leaf_selector"]):
        return False
    if trigger.get("base_leaf_indices"):
        if int_list(marginal.get("base_leaf_indices")) != list(trigger["base_leaf_indices"]):
            return False
    if trigger.get("added_leaf_indices"):
        if int_list(marginal.get("added_leaf_indices")) != list(trigger["added_leaf_indices"]):
            return False
    return True


def marginal_rows(total4_case: dict[str, Any], total3_case: dict[str, Any], seed: str) -> list[dict[str, Any]]:
    base_by_row = {
        str(item.get("row_key")): set(int_list(item.get("leaf_indices")))
        for item in normalized_row_leaf_keys(total3_case, seed)
    }
    out = []
    for item in normalized_row_leaf_keys(total4_case, seed):
        row_key = str(item.get("row_key"))
        leaves = set(int_list(item.get("leaf_indices")))
        base_leaves = base_by_row.get(row_key, set())
        added = sorted(leaves - base_leaves)
        if not added:
            continue
        marginal = dict(item)
        marginal["base_leaf_indices"] = sorted(base_leaves)
        marginal["added_leaf_indices"] = added
        out.append(marginal)
    return out


def collect_pairs(
    signature: dict[str, Any],
    trigger: dict[str, Any],
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = [case for case in signature.get("positive_cases") or [] if isinstance(case, dict)]
    bases = best_total3_bases(cases, seed)
    all_marginals = []
    selected_by_case: dict[str, dict[str, Any]] = {}
    for case in cases:
        selector = str(case.get("leaf_selector") or case.get("selector") or "")
        if "total4" not in selector:
            continue
        base = bases.get(base_key(case, seed))
        if base is None:
            continue
        matched_rows = []
        for marginal in marginal_rows(case, base, seed):
            feature = {
                "case_key": case_key_string(case, seed),
                "target": str(case.get("target")),
                "transfer_index": int(case.get("transfer_index") or 0),
                "top_k": int(case.get("top_k") or 0),
                "policy": case.get("policy"),
                "leaf_selector": case.get("leaf_selector"),
                "row_key": marginal.get("row_key"),
                "surface_id": marginal.get("surface_id"),
                "base_leaf_indices": marginal.get("base_leaf_indices"),
                "added_leaf_indices": marginal.get("added_leaf_indices"),
                "base_added_key": (
                    f"{','.join(str(leaf) for leaf in marginal.get('base_leaf_indices') or [])}"
                    "->"
                    f"{','.join(str(leaf) for leaf in marginal.get('added_leaf_indices') or [])}"
                ),
                "signature_public_key_verified": bool(case.get("public_key_verified")),
                "signature_relation_count": int(case.get("relation_count") or 0),
                "signature_rank": int(case.get("rank") or 0),
                "source_ops_over_rho": case.get("ops_over_rho"),
            }
            all_marginals.append(feature)
            if matches_trigger(case, marginal, trigger):
                matched_rows.append(marginal)
        if not matched_rows:
            continue
        key = case_key_string(case, seed)
        selected = dict(case)
        selected["case_key"] = key
        selected["total4_lift_base_case_key"] = case_key_string(base, seed)
        selected["retention_mode"] = "public_fourth_leaf_trigger_marginal_surfaces_only"
        selected["row_leaf_keys"] = matched_rows
        selected["marginal_lift_row_leaf_keys"] = matched_rows
        selected["surface_ids"] = sorted({str(row.get("surface_id")) for row in matched_rows})
        selected["marginal_lift_surface_ids"] = selected["surface_ids"]
        selected["trigger"] = trigger
        selected_by_case[key] = selected
    return sorted(selected_by_case.values(), key=lambda row: str(row.get("case_key"))), all_marginals


def retained_surfaces(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = {}
    for case in cases:
        for item in case.get("row_leaf_keys") or []:
            surface_id = str(item.get("surface_id"))
            rows[surface_id] = {
                "surface_id": surface_id,
                "target": case.get("target"),
                "transfer_index": int(case.get("transfer_index") or 0),
                "row_key": item.get("row_key"),
                "salt": item.get("salt"),
                "source": "public_fourth_leaf_trigger_marginal_surface",
            }
    return [rows[key] for key in sorted(rows)]


def summarize(cases: list[dict[str, Any]], all_marginals: list[dict[str, Any]]) -> dict[str, Any]:
    feature_counts = Counter(str(row.get("base_added_key")) for row in all_marginals)
    selected_transfers = sorted({f"{case.get('target')}|{case.get('transfer_index')}" for case in cases})
    return {
        "bridge_status": "public_fourth_leaf_trigger_bridge_ready" if cases else "no_trigger_matches",
        "all_marginal_row_count": len(all_marginals),
        "all_marginal_base_added_counts": dict(feature_counts.most_common()),
        "selected_case_count": len(cases),
        "selected_row_count": sum(len(case.get("row_leaf_keys") or []) for case in cases),
        "selected_surface_count": len({surface.get("surface_id") for surface in retained_surfaces(cases)}),
        "selected_challenge_count": len(selected_transfers),
        "selected_transfers": selected_transfers,
        "selected_signature_verified_label_count": sum(bool(case.get("public_key_verified")) for case in cases),
        "next_obligation": (
            "Replay the trigger bridge if selected_case_count is nonzero; otherwise treat the frozen trigger as absent on this window."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-source", type=Path, default=DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--trigger-target")
    parser.add_argument("--trigger-base-leaf", type=int, action="append", default=[])
    parser.add_argument("--trigger-added-leaf", type=int, action="append", default=[])
    parser.add_argument("--trigger-top-k", type=int)
    parser.add_argument("--trigger-policy")
    parser.add_argument("--trigger-leaf-selector")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signature = load_json(args.signature_source)
    trigger = trigger_args(args)
    selected, all_marginals = collect_pairs(signature, trigger, str(args.seed))
    output = {
        "schema": "ffe_public_fourth_leaf_trigger_bridge/v2",
        "method": "direct_public_total3_total4_fourth_leaf_trigger_bridge",
        "parameters": {
            "signature_source": str(args.signature_source),
            "seed": args.seed,
            "trigger": trigger,
            "windows": [
                {
                    "label": args.out.stem,
                    "signature_source": str(args.signature_source),
                }
            ],
            "forbidden_selection_fields": [
                "public_key_verified",
                "relation_count",
                "rank",
                "source_verified",
            ],
        },
        "summary": summarize(selected, all_marginals),
        "retained_surfaces": retained_surfaces(selected),
        "retained_source_cases": selected,
        "marginal_feature_sample": all_marginals[:64],
        "non_claims": [
            "This bridge tests whether a declared public fourth-leaf lift appears and replays on a fresh source.",
            "Verifier labels are copied only for reporting and are not used for trigger selection.",
            "A replay success is still a component result, not a complete ECDLP index-calculus algorithm.",
        ],
    }
    write_json(args.out, output)
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
