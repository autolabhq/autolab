#!/usr/bin/env python3
"""Scan public selector artifacts for unmaterialized FFE branch analogues.

The full-remainder wins currently sit in small, local pockets.  This scanner
keeps the next exact probes honest by ranking only public/pre-Sage features from
selector artifacts, then excluding row/leaf profiles that already have Sage
exact-profile output.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_STATE_DIR = Path("ecdlp_index_calculus_state")
DEFAULT_SELECTOR_GLOB = "low_term_total3_total4_public_bounded_full_selector_*.json"
DEFAULT_EXACT_GLOB = "ffe_sage_factor_exact_profiles*.json"
DEFAULT_BANK_SOURCE = Path(
    "/Volumes/Volume/autolab/ecdlp_index_calculus_state/"
    "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_probe.json"
)


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


def parse_transfer_index(surface_id: str) -> int | None:
    match = re.search(r"shared-transfer:(\d+):", surface_id)
    return int(match.group(1)) if match else None


def leaf_signature(leaves: list[Any]) -> str:
    return ",".join(str(int(leaf)) for leaf in sorted({int(leaf) for leaf in leaves}))


def expand_paths(raw_patterns: list[str], state_dir: Path) -> list[Path]:
    paths: set[Path] = set()
    for raw in raw_patterns:
        pattern = raw
        if not any(char in raw for char in "*?[]"):
            paths.add(Path(raw))
            continue
        if not Path(raw).is_absolute():
            pattern = str(state_dir / raw)
        for match in glob.glob(pattern):
            paths.add(Path(match))
    return sorted(path for path in paths if path.exists())


def load_bank_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = load_json(path)
    return {
        str(row.get("row_key")): row
        for row in data.get("bank_rows") or []
        if isinstance(row, dict) and row.get("row_key") is not None
    }


def exact_profile_ids(paths: list[Path]) -> set[str]:
    profile_ids: set[str] = set()
    for path in paths:
        data = load_json(path)
        for surface in data.get("surfaces") or []:
            if not isinstance(surface, dict):
                continue
            surface_profile_id = surface.get("surface_profile_id")
            if surface_profile_id:
                profile_ids.add(str(surface_profile_id))
                continue
            surface_id = surface.get("surface_id")
            leaves = surface.get("selected_leaf_indices") or []
            if surface_id and leaves:
                profile_ids.add(f"{surface_id}#leaves={leaf_signature(leaves)}")
    return profile_ids


def bank_source_window_labels(bank_row: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(window.get("window_label"))
            for window in bank_row.get("source_windows") or []
            if isinstance(window, dict) and window.get("window_label") is not None
        }
    )


def bank_source_selectors(bank_row: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(window.get("source_selector"))
            for window in bank_row.get("source_windows") or []
            if isinstance(window, dict) and window.get("source_selector") is not None
        }
    )


def candidate_from_case(
    selector_path: Path,
    case_index: int,
    case: dict[str, Any],
    row_item: dict[str, Any],
    bank_rows: dict[str, dict[str, Any]],
    evaluated_profiles: set[str],
    modulus: int,
) -> dict[str, Any] | None:
    target = str(case.get("target") or "")
    row_key = str(row_item.get("row_key") or "")
    surface_id = str(row_item.get("surface_id") or "")
    transfer = as_int(case.get("transfer_index"))
    if transfer is None:
        transfer = parse_transfer_index(surface_id)
    if not target or not row_key or not surface_id or transfer is None:
        return None
    leaves = sorted({int(leaf) for leaf in row_item.get("leaf_indices") or []})
    if not leaves:
        return None
    row_salt = as_int(row_item.get("salt"))
    if row_salt is None:
        row_salt = parse_row_salt(row_key)
    selected_profile_id = f"{surface_id}#leaves={leaf_signature(leaves)}"
    bank_row = bank_rows.get(row_key) or {}
    return {
        "selector_path": str(selector_path),
        "selector_case_index": case_index,
        "target": target,
        "transfer_index": transfer,
        "transfer_mod": transfer % modulus,
        "row_key": row_key,
        "row_salt": row_salt,
        "row_salt_transfer_mod": f"{row_salt}|{transfer % modulus}" if row_salt is not None else None,
        "surface_id": surface_id,
        "surface_profile_id": selected_profile_id,
        "leaf_indices": leaves,
        "leaf_signature": leaf_signature(leaves),
        "leaf_count": len(leaves),
        "top_k": as_int(case.get("top_k")),
        "policy": case.get("policy"),
        "row_selector": case.get("row_selector"),
        "leaf_selector": case.get("leaf_selector") or case.get("selector"),
        "selector_ops_over_rho": as_float(case.get("ops_over_rho")),
        "selector_below_rho": bool(case.get("below_rho")),
        "public_key_verified_label": bool(case.get("public_key_verified")),
        "relation_count_label": as_int(case.get("relation_count")),
        "rank_label": as_int(case.get("rank")),
        "selected_row_count": as_int(case.get("selected_row_count")),
        "selected_leaf_count": as_int(case.get("selected_leaf_count")),
        "bank_row_present": bool(bank_row),
        "bank_best_filter_mode": bank_row.get("best_filter_mode"),
        "bank_best_filter_top_k": as_int(bank_row.get("best_filter_top_k")),
        "bank_best_filter_ops_over_rho": as_float(bank_row.get("best_filter_ops_over_rho")),
        "bank_selected_relation_count": as_int(bank_row.get("selected_relation_count")),
        "bank_surface_hit_row_count": as_int(bank_row.get("surface_hit_row_count")),
        "bank_source_window_labels": bank_source_window_labels(bank_row),
        "bank_source_selectors": bank_source_selectors(bank_row),
        "already_exact_profile": selected_profile_id in evaluated_profiles,
        "profile_from_signature": f"{target}|{transfer}|{row_key}",
    }


def score_candidate(candidate: dict[str, Any], args: argparse.Namespace) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    ref_transfer_mod = args.reference_transfer % args.modulus
    transfer_mod = candidate.get("transfer_mod")
    row_salt = candidate.get("row_salt")
    leaves = set(candidate.get("leaf_indices") or [])

    if transfer_mod == ref_transfer_mod:
        score += 6
        reasons.append(f"transfer_mod{args.modulus}={ref_transfer_mod}")
    if row_salt == args.reference_row_salt:
        score += 4
        reasons.append(f"row_salt={args.reference_row_salt}")
    if row_salt == args.reference_row_salt and transfer_mod == ref_transfer_mod:
        score += 8
        reasons.append("literal_row_salt_transfer_branch")
    if args.reference_leaf in leaves:
        score += 4
        reasons.append(f"contains_leaf_{args.reference_leaf}")
    if leaves == {args.reference_leaf}:
        score += 3
        reasons.append("single_reference_leaf")
    if leaves == {args.reference_companion_leaf, args.reference_leaf}:
        score += 2
        reasons.append("reference_companion_pair")
    if candidate.get("top_k") == args.reference_top_k:
        score += 3
        reasons.append(f"top_k={args.reference_top_k}")
    if candidate.get("selector_below_rho"):
        score += 3
        reasons.append("public_selector_below_rho")
    ops = candidate.get("selector_ops_over_rho")
    if ops is not None and ops <= 1.0:
        score += 2
        reasons.append("public_selector_ops<=rho")
    if candidate.get("public_key_verified_label"):
        score += 2
        reasons.append("verifier_label_positive")
    if candidate.get("bank_best_filter_mode") == args.reference_bank_mode:
        score += 5
        reasons.append(f"bank_mode={args.reference_bank_mode}")
    if candidate.get("bank_best_filter_top_k") == args.reference_top_k:
        score += 2
        reasons.append(f"bank_top_k={args.reference_top_k}")
    bank_ops = candidate.get("bank_best_filter_ops_over_rho")
    if bank_ops is not None and bank_ops <= 0.5:
        score += 1
        reasons.append("bank_filter_ops<=0.5rho")
    if candidate.get("already_exact_profile"):
        score -= 6
        reasons.append("already_exact_profile")
    return score, reasons


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "score",
        "score_reasons",
        "target",
        "transfer_index",
        "transfer_mod",
        "row_key",
        "row_salt",
        "row_salt_transfer_mod",
        "leaf_signature",
        "top_k",
        "policy",
        "row_selector",
        "leaf_selector",
        "selector_ops_over_rho",
        "selector_below_rho",
        "public_key_verified_label",
        "relation_count_label",
        "rank_label",
        "bank_row_present",
        "bank_best_filter_mode",
        "bank_best_filter_top_k",
        "bank_best_filter_ops_over_rho",
        "bank_source_window_labels",
        "already_exact_profile",
        "profile_from_signature",
        "surface_profile_id",
        "selector_path",
    ]
    return {key: candidate.get(key) for key in keep}


def target_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_target[str(candidate["target"])].append(candidate)
    summaries = []
    for target, rows in sorted(by_target.items()):
        top = max(rows, key=lambda row: row["score"])
        summaries.append(
            {
                "target": target,
                "candidate_count": len(rows),
                "literal_branch_atom_count": sum(
                    1 for row in rows if row.get("row_salt_transfer_mod") == "165|10"
                ),
                "transfer_mod_match_count": sum(1 for row in rows if row.get("transfer_mod") == 10),
                "bank_low_term_span_count": sum(
                    1 for row in rows if row.get("bank_best_filter_mode") == "low_term_span"
                ),
                "best_candidate": compact_candidate(top),
            }
        )
    return summaries


def exact_probe_groups(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate["selector_path"], candidate["profile_from_signature"])
        if key in seen:
            continue
        seen.add(key)
        grouped[candidate["selector_path"]].append(candidate["profile_from_signature"])
        if sum(len(values) for values in grouped.values()) >= limit:
            break
    groups = []
    for selector_path, selectors in grouped.items():
        groups.append(
            {
                "signature_source": selector_path,
                "profile_from_signature": selectors,
            }
        )
    return groups


def top_candidates_by_target(
    candidates: list[dict[str, Any]],
    limit_per_target: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        bucket = grouped[str(candidate["target"])]
        if len(bucket) < limit_per_target:
            bucket.append(compact_candidate(candidate))
    return dict(sorted(grouped.items()))


def exact_probe_groups_by_target(
    candidates: list[dict[str, Any]],
    limit_per_target: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped_candidates[str(candidate["target"])].append(candidate)
    return {
        target: exact_probe_groups(rows, limit_per_target)
        for target, rows in sorted(grouped_candidates.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--selector-glob", action="append", default=[DEFAULT_SELECTOR_GLOB])
    parser.add_argument("--exact-glob", action="append", default=[DEFAULT_EXACT_GLOB])
    parser.add_argument("--bank-source", type=Path, default=DEFAULT_BANK_SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-candidates", type=int, default=40)
    parser.add_argument("--top-candidates-per-target", type=int, default=12)
    parser.add_argument("--exact-group-limit", type=int, default=12)
    parser.add_argument("--exact-group-limit-per-target", type=int, default=6)
    parser.add_argument("--include-already-exact", action="store_true")
    parser.add_argument("--reference-transfer", type=int, default=618)
    parser.add_argument("--reference-row-salt", type=int, default=165)
    parser.add_argument("--reference-leaf", type=int, default=90)
    parser.add_argument("--reference-companion-leaf", type=int, default=8)
    parser.add_argument("--reference-top-k", type=int, default=12)
    parser.add_argument("--reference-bank-mode", default="low_term_span")
    parser.add_argument("--modulus", type=int, default=32)
    args = parser.parse_args()

    selector_paths = expand_paths(args.selector_glob, args.state_dir)
    exact_paths = expand_paths(args.exact_glob, args.state_dir)
    bank_rows = load_bank_rows(args.bank_source)
    evaluated_profiles = exact_profile_ids(exact_paths)

    candidates: list[dict[str, Any]] = []
    selector_case_count = 0
    for selector_path in selector_paths:
        data = load_json(selector_path)
        for index, case in enumerate(data.get("positive_cases") or []):
            if not isinstance(case, dict):
                continue
            selector_case_count += 1
            for row_item in case.get("row_leaf_keys") or []:
                if not isinstance(row_item, dict):
                    continue
                candidate = candidate_from_case(
                    selector_path,
                    index,
                    case,
                    row_item,
                    bank_rows,
                    evaluated_profiles,
                    args.modulus,
                )
                if candidate is None:
                    continue
                score, reasons = score_candidate(candidate, args)
                candidate["score"] = score
                candidate["score_reasons"] = reasons
                if candidate["already_exact_profile"] and not args.include_already_exact:
                    continue
                candidates.append(candidate)

    candidates.sort(
        key=lambda row: (
            -int(row["score"]),
            str(row["target"]),
            int(row["transfer_index"]),
            str(row["row_key"]),
            str(row["leaf_signature"]),
        )
    )
    score_counts = Counter(str(candidate["score"]) for candidate in candidates)
    target_counts = Counter(str(candidate["target"]) for candidate in candidates)
    top_candidates = candidates[: args.top_candidates]
    output = {
        "schema": "ecdlp_ffe_cross_target_branch_analogue_scanner_v1",
        "method": "public_selector_branch_scoring_with_sage_exact_profile_exclusion",
        "parameters": {
            "state_dir": str(args.state_dir),
            "selector_globs": args.selector_glob,
            "exact_globs": args.exact_glob,
            "bank_source": str(args.bank_source) if args.bank_source else None,
            "include_already_exact": args.include_already_exact,
            "reference": {
                "transfer": args.reference_transfer,
                "transfer_modulus": args.modulus,
                "transfer_mod": args.reference_transfer % args.modulus,
                "row_salt": args.reference_row_salt,
                "leaf": args.reference_leaf,
                "companion_leaf": args.reference_companion_leaf,
                "top_k": args.reference_top_k,
                "bank_mode": args.reference_bank_mode,
            },
        },
        "summary": {
            "selector_file_count": len(selector_paths),
            "selector_case_count": selector_case_count,
            "exact_profile_file_count": len(exact_paths),
            "evaluated_profile_count": len(evaluated_profiles),
            "bank_row_count": len(bank_rows),
            "candidate_count": len(candidates),
            "target_counts": dict(sorted(target_counts.items())),
            "score_counts": dict(sorted(score_counts.items(), key=lambda item: int(item[0]), reverse=True)),
            "literal_row_salt_transfer_mod_count": sum(
                1 for candidate in candidates if candidate.get("row_salt_transfer_mod") == "165|10"
            ),
            "transfer_mod_match_count": sum(
                1 for candidate in candidates if candidate.get("transfer_mod") == args.reference_transfer % args.modulus
            ),
            "bank_low_term_span_count": sum(
                1 for candidate in candidates if candidate.get("bank_best_filter_mode") == args.reference_bank_mode
            ),
        },
        "target_summaries": target_summary(candidates),
        "top_candidates": [compact_candidate(candidate) for candidate in top_candidates],
        "top_candidates_by_target": top_candidates_by_target(
            candidates,
            args.top_candidates_per_target,
        ),
        "next_exact_probe_groups": exact_probe_groups(candidates, args.exact_group_limit),
        "next_exact_probe_groups_by_target": exact_probe_groups_by_target(
            candidates,
            args.exact_group_limit_per_target,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
