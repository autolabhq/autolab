#!/usr/bin/env python3
"""Build exact-materialization queues for the full-remainder candidate rule.

The enriched sparsity miner found a pre-materialization shell:

    bank_best_filter_top_k=12 & profile_top_k=12

The remaining predicate, original_selected_root_pair_count=0, is only known
after exact profile materialization.  This script harvests selector cases that
match the public shell and emits concrete --profile arguments for
ffe_sage_factor_exact_profile_subset_probe.py.
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
DEFAULT_SELECTOR_GLOB = "low_term_total3_total4_public_bounded_full_selector_*.json"
DEFAULT_BANK_SOURCE = Path(
    "/Volumes/Volume/autolab/ecdlp_index_calculus_state/"
    "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_full_remainder_topk12_root0_candidate_queue.json"
DEFAULT_EXCLUDED_TRANSFERS = (420, 618)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_row_salt(row_key: str) -> int | None:
    match = re.search(r":salt(\d+)$", row_key)
    return int(match.group(1)) if match else None


def leaf_tuple(values: Any) -> tuple[int, ...]:
    leaves: list[int] = []
    for value in values or []:
        leaf = as_int(value)
        if leaf is not None:
            leaves.append(leaf)
    return tuple(sorted(set(leaves)))


def load_bank_rows(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    rows = data.get("bank_rows") or []
    return {
        str(row.get("row_key")): row
        for row in rows
        if isinstance(row, dict) and row.get("row_key") is not None
    }


def collect_selector_paths(
    state_dir: Path,
    selector_artifacts: list[Path],
    selector_globs: list[str],
) -> list[Path]:
    paths: list[Path] = []
    paths.extend(selector_artifacts)
    if not paths:
        for pattern in selector_globs:
            paths.extend(sorted(state_dir.glob(pattern)))
    return sorted(dict.fromkeys(paths))


def profile_key(
    target: str,
    transfer_index: int,
    top_k: int,
    policy: str,
    leaf_selector: str,
    row_key: str,
    leaves: tuple[int, ...],
) -> tuple[Any, ...]:
    return (target, transfer_index, top_k, policy, leaf_selector, row_key, leaves)


def materialization_key(
    target: str,
    transfer_index: int,
    row_key: str,
    leaves: tuple[int, ...],
) -> tuple[Any, ...]:
    return (target, transfer_index, row_key, leaves)


def profile_arg_from_key(key: tuple[Any, ...]) -> str:
    target, transfer_index, top_k, policy, leaf_selector, row_key, leaves = key
    leaf_text = ",".join(str(leaf) for leaf in leaves)
    return f"{target}|{transfer_index}|{top_k}|{policy}|{leaf_selector}|{row_key}|{leaf_text}"


def miner_materialization_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
    target = record.get("target")
    transfer_index = as_int(record.get("transfer_index"))
    row_key = record.get("row_key")
    leaves = leaf_tuple(record.get("selected_leaf_indices"))
    if (
        target is None
        or transfer_index is None
        or row_key is None
        or not leaves
    ):
        return None
    return materialization_key(
        str(target),
        transfer_index,
        str(row_key),
        leaves,
    )


def exact_surface_materialization_key(surface: dict[str, Any]) -> tuple[Any, ...] | None:
    exact = surface.get("exact_profile") or {}
    target = exact.get("target") or surface.get("target")
    transfer_index = as_int(exact.get("transfer_index") or surface.get("transfer_index"))
    row_key = exact.get("row_key") or surface.get("row_key")
    leaves = leaf_tuple(exact.get("leaf_indices") or surface.get("selected_leaf_indices"))
    if target is None or transfer_index is None or row_key is None or not leaves:
        return None
    return materialization_key(str(target), transfer_index, str(row_key), leaves)


def load_existing_miner_keys(paths: list[Path]) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for path in paths:
        data = load_json(path)
        for record in data.get("records") or []:
            if not isinstance(record, dict):
                continue
            key = miner_materialization_key(record)
            if key is not None:
                keys.add(key)
    return keys


def load_existing_exact_keys(paths: list[Path]) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for path in paths:
        data = load_json(path)
        for surface in data.get("surfaces") or []:
            if not isinstance(surface, dict):
                continue
            key = exact_surface_materialization_key(surface)
            if key is not None:
                keys.add(key)
    return keys


def compact_bank_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "bank_best_filter_top_k": as_int(row.get("best_filter_top_k")),
        "bank_best_filter_mode": row.get("best_filter_mode"),
        "bank_best_filter_ops_over_rho": as_float(row.get("best_filter_ops_over_rho")),
        "bank_source_selectors": row.get("source_selectors") or [],
        "bank_source_windows": row.get("source_windows") or [],
    }


def add_candidate_source(
    entry: dict[str, Any],
    source: Path,
    case: dict[str, Any],
    profile_arg: str,
) -> None:
    source_text = str(source)
    if source_text not in entry["source_artifacts"]:
        entry["source_artifacts"].append(source_text)
    if profile_arg not in entry["equivalent_profiles"]:
        entry["equivalent_profiles"].append(profile_arg)
    entry["source_case_count"] += 1
    ops = as_float(case.get("ops_over_rho"))
    if ops is not None:
        current = entry.get("selector_ops_over_rho_min")
        entry["selector_ops_over_rho_min"] = ops if current is None else min(current, ops)
    entry["selector_below_rho_label_any"] = bool(
        entry.get("selector_below_rho_label_any") or case.get("below_rho")
    )
    entry["public_key_verified_label_any"] = bool(
        entry.get("public_key_verified_label_any") or case.get("public_key_verified")
    )


def build_candidates(args: argparse.Namespace) -> dict[str, Any]:
    bank_rows = load_bank_rows(args.bank_source)
    bank_top_rows = {
        row_key: row
        for row_key, row in bank_rows.items()
        if as_int(row.get("best_filter_top_k")) == args.bank_top_k
    }
    selector_paths = collect_selector_paths(
        args.state_dir,
        args.selector_artifact,
        args.selector_glob,
    )
    existing_miner_keys = load_existing_miner_keys(args.existing_miner_artifact)
    existing_exact_keys = load_existing_exact_keys(args.existing_exact_artifact)
    existing_keys = existing_miner_keys | existing_exact_keys
    target_filter = set(args.target)
    row_key_filter = set(args.row_key)
    excluded_transfers = set(args.exclude_transfer)

    skip_reasons: Counter[str] = Counter()
    candidates_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    raw_case_count = 0
    raw_row_profile_count = 0

    for path in selector_paths:
        data = load_json(path)
        for case in data.get("positive_cases") or []:
            if not isinstance(case, dict):
                continue
            raw_case_count += 1
            target = case.get("target")
            transfer_index = as_int(case.get("transfer_index"))
            top_k = as_int(case.get("top_k"))
            policy = case.get("policy")
            leaf_selector = case.get("leaf_selector") or case.get("selector")
            if target is None or transfer_index is None or top_k is None or policy is None or leaf_selector is None:
                skip_reasons["missing_case_identity"] += 1
                continue
            target_text = str(target)
            if target_filter and target_text not in target_filter:
                skip_reasons["target_filter"] += 1
                continue
            if top_k != args.profile_top_k:
                skip_reasons["profile_top_k_mismatch"] += 1
                continue
            if transfer_index in excluded_transfers:
                skip_reasons["excluded_transfer"] += 1
                continue
            for item in case.get("row_leaf_keys") or []:
                if not isinstance(item, dict):
                    continue
                raw_row_profile_count += 1
                row_key = str(item.get("row_key") or "")
                if row_key_filter and row_key not in row_key_filter:
                    skip_reasons["row_key_filter"] += 1
                    continue
                bank_row = bank_top_rows.get(row_key)
                if not bank_row:
                    skip_reasons["bank_top_k_mismatch"] += 1
                    continue
                leaves = leaf_tuple(item.get("leaf_indices"))
                if not leaves:
                    skip_reasons["empty_leaf_indices"] += 1
                    continue
                spec_key = profile_key(
                    target_text,
                    transfer_index,
                    top_k,
                    str(policy),
                    str(leaf_selector),
                    row_key,
                    leaves,
                )
                exact_key = materialization_key(target_text, transfer_index, row_key, leaves)
                candidate_key = exact_key if args.dedupe_surface_profiles else spec_key
                profile_arg = profile_arg_from_key(spec_key)
                already_materialized = exact_key in existing_keys
                if already_materialized and not args.include_existing:
                    skip_reasons["already_materialized"] += 1
                    continue
                if candidate_key not in candidates_by_key:
                    candidates_by_key[candidate_key] = {
                        "profile": profile_arg,
                        "profile_from_signature": f"{target_text}|{transfer_index}|{row_key}",
                        "materialization_key": {
                            "target": target_text,
                            "transfer_index": transfer_index,
                            "row_key": row_key,
                            "leaf_indices": list(leaves),
                        },
                        "target": target_text,
                        "transfer_index": transfer_index,
                        "top_k": top_k,
                        "policy": str(policy),
                        "row_selector": case.get("row_selector"),
                        "leaf_selector": str(leaf_selector),
                        "row_key": row_key,
                        "row_salt": parse_row_salt(row_key),
                        "leaf_indices": list(leaves),
                        "surface_id": item.get("surface_id"),
                        "surface_ids": case.get("surface_ids") or [],
                        "selector_rank_label": case.get("rank"),
                        "selector_relation_count_label": case.get("relation_count"),
                        "selector_ops_over_rho_min": as_float(case.get("ops_over_rho")),
                        "selector_below_rho_label_any": bool(case.get("below_rho")),
                        "public_key_verified_label_any": bool(case.get("public_key_verified")),
                        "source_artifacts": [],
                        "source_case_count": 0,
                        "equivalent_profiles": [],
                        "known_materialized_profile": already_materialized,
                        "required_post_materialization_predicate": "original_selected_root_pair_count=0",
                        "materialization_status": (
                            "already_materialized" if already_materialized else "fresh_candidate"
                        ),
                        **compact_bank_row(bank_row),
                    }
                add_candidate_source(candidates_by_key[candidate_key], path, case, profile_arg)

    candidates = sorted(candidates_by_key.values(), key=candidate_sort_key)
    candidate_count_before_limit = len(candidates)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    by_row_key: dict[str, int] = dict(Counter(entry["row_key"] for entry in candidates))
    by_transfer: dict[str, int] = dict(
        sorted(Counter(str(entry["transfer_index"]) for entry in candidates).items())
    )
    by_target: dict[str, int] = dict(Counter(entry["target"] for entry in candidates))
    by_status: dict[str, int] = dict(Counter(entry["materialization_status"] for entry in candidates))
    by_source: defaultdict[str, int] = defaultdict(int)
    for entry in candidates:
        for source in entry["source_artifacts"]:
            by_source[source] += 1

    return {
        "schema": "ecdlp_full_remainder_candidate_rule_queue_v1",
        "method": "public_shell_queue_for_exact_full_remainder_root0_materialization",
        "candidate_rule": {
            "public_shell": [
                f"bank_best_filter_top_k={args.bank_top_k}",
                f"profile_top_k={args.profile_top_k}",
            ],
            "requires_exact_materialization": ["original_selected_root_pair_count=0"],
            "excluded_transfers": sorted(excluded_transfers),
        },
        "inputs": {
            "bank_source": str(args.bank_source),
            "selector_artifacts": [str(path) for path in selector_paths],
            "existing_miner_artifacts": [str(path) for path in args.existing_miner_artifact],
            "existing_exact_artifacts": [str(path) for path in args.existing_exact_artifact],
            "dedupe_surface_profiles": args.dedupe_surface_profiles,
        },
        "summary": {
            "bank_top_k_row_count": len(bank_top_rows),
            "bank_top_k_rows": sorted(bank_top_rows),
            "selector_artifact_count": len(selector_paths),
            "raw_selector_case_count": raw_case_count,
            "raw_row_profile_count": raw_row_profile_count,
            "existing_miner_profile_key_count": len(existing_miner_keys),
            "existing_exact_profile_key_count": len(existing_exact_keys),
            "existing_profile_key_count": len(existing_keys),
            "candidate_count_before_limit": candidate_count_before_limit,
            "candidate_count": len(candidates),
            "skip_reasons": dict(sorted(skip_reasons.items())),
            "by_target": by_target,
            "by_row_key": by_row_key,
            "by_transfer": by_transfer,
            "by_materialization_status": by_status,
            "by_source_artifact": dict(sorted(by_source.items())),
        },
        "command_template": {
            "script": "tasks/ecdlp_index_calculus/ffe_sage_factor_exact_profile_subset_probe.py",
            "profile_argument_format": "target|transfer_index|top_k|policy|leaf_selector|row_key|leaf1,leaf2",
        },
        "candidates": candidates,
    }


def candidate_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    ops = entry.get("selector_ops_over_rho_min")
    bank_ops = entry.get("bank_best_filter_ops_over_rho")
    return (
        1 if entry.get("known_materialized_profile") else 0,
        1 if entry.get("row_key") == "22050.cf1@11731:uniform:256:salt165" else 0,
        bank_ops if bank_ops is not None else 10**9,
        ops if ops is not None else 10**9,
        str(entry.get("target")),
        str(entry.get("row_key")),
        int(entry.get("transfer_index") or 0),
        str(entry.get("leaf_selector")),
        entry.get("leaf_indices") or [],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--bank-source", type=Path, default=DEFAULT_BANK_SOURCE)
    parser.add_argument("--selector-artifact", type=Path, action="append", default=[])
    parser.add_argument("--selector-glob", action="append", default=[DEFAULT_SELECTOR_GLOB])
    parser.add_argument("--existing-miner-artifact", type=Path, action="append", default=[])
    parser.add_argument("--existing-exact-artifact", type=Path, action="append", default=[])
    parser.add_argument("--bank-top-k", type=int, default=12)
    parser.add_argument("--profile-top-k", type=int, default=12)
    parser.add_argument("--exclude-transfer", type=int, action="append", default=list(DEFAULT_EXCLUDED_TRANSFERS))
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--row-key", action="append", default=[])
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument(
        "--keep-equivalent-source-profiles",
        dest="dedupe_surface_profiles",
        action="store_false",
        help="Keep separate queue entries for profiles that materialize the same target/transfer/row/leaves surface.",
    )
    parser.set_defaults(dedupe_surface_profiles=True)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    output = build_candidates(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "candidate_count": output["summary"]["candidate_count"],
                "candidate_count_before_limit": output["summary"]["candidate_count_before_limit"],
                "by_row_key": output["summary"]["by_row_key"],
                "skip_reasons": output["summary"]["skip_reasons"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
