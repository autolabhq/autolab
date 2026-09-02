#!/usr/bin/env python3
"""Audit frozen ``salt165`` full-remainder holdout coverage.

This does not perform Sage factorization.  It scans public-bounded selector
artifacts for the frozen selector cell and joins any exact-profile Sage attempt
artifacts so the next replay step is based on visible coverage and failure
evidence.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SELECTOR_GLOB = "low_term_total3_total4_public_bounded_full_selector_*_*.json"
DEFAULT_EXACT_GLOB = "ffe_sage_factor_exact_profiles_22050_salt165_*transfer*.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def leaf_signature(leaves: list[Any]) -> str:
    return ",".join(str(int(leaf)) for leaf in sorted({int(leaf) for leaf in leaves}))


def row_salt(row_key: str) -> int | None:
    match = re.search(r":salt(\d+)$", row_key)
    return int(match.group(1)) if match else None


def window_from_path(path: Path) -> str:
    match = re.search(r"(\d+_\d+)", path.name)
    return match.group(1) if match else path.stem


def selector_profile_records(path: Path, target: str, row_key: str, modulus: int, residue: int, min_transfer: int) -> list[dict[str, Any]]:
    data = load_json(path)
    records: list[dict[str, Any]] = []
    for case in data.get("positive_cases") or []:
        if not isinstance(case, dict):
            continue
        if str(case.get("target")) != target:
            continue
        transfer = int(case.get("transfer_index") or -1)
        if transfer < min_transfer or transfer % modulus != residue:
            continue
        for item in case.get("row_leaf_keys") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("row_key") or "") != row_key:
                continue
            records.append(
                {
                    "source": str(path),
                    "window": window_from_path(path),
                    "target": target,
                    "transfer_index": transfer,
                    "transfer_modulus": modulus,
                    "transfer_residue": residue,
                    "top_k": int(case.get("top_k") or 0),
                    "policy": case.get("policy"),
                    "leaf_selector": case.get("leaf_selector") or case.get("selector"),
                    "row_key": row_key,
                    "row_salt": row_salt(row_key),
                    "leaf_indices": [int(leaf) for leaf in item.get("leaf_indices") or []],
                    "leaf_signature": leaf_signature(item.get("leaf_indices") or []),
                    "ops_over_rho_label": case.get("ops_over_rho"),
                    "public_key_verified_label": case.get("public_key_verified"),
                    "surface_id": item.get("surface_id"),
                }
            )
    return records


def same_target_neighbor_records(path: Path, target: str, wanted_salt: int, modulus: int, residue: int, min_transfer: int) -> list[dict[str, Any]]:
    data = load_json(path)
    records: list[dict[str, Any]] = []
    for case in data.get("positive_cases") or []:
        if not isinstance(case, dict) or str(case.get("target")) != target:
            continue
        transfer = int(case.get("transfer_index") or -1)
        if transfer < min_transfer or transfer % modulus != residue:
            continue
        for item in case.get("row_leaf_keys") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("row_key") or "")
            salt = row_salt(key)
            if salt is None or salt == wanted_salt:
                continue
            records.append(
                {
                    "source": str(path),
                    "window": window_from_path(path),
                    "transfer_index": transfer,
                    "row_key": key,
                    "row_salt": salt,
                    "leaf_signature": leaf_signature(item.get("leaf_indices") or []),
                    "ops_over_rho_label": case.get("ops_over_rho"),
                    "public_key_verified_label": case.get("public_key_verified"),
                }
            )
    return records


def exact_attempt_record(path: Path) -> dict[str, Any]:
    data = load_json(path)
    summary = data.get("summary") or {}
    surfaces = data.get("surfaces") or []
    profiles = (data.get("parameters") or {}).get("profiles") or []
    materialization_errors = data.get("materialization_errors") or []
    error_kinds = Counter()
    for error in materialization_errors:
        for fallback in error.get("fallback_errors") or []:
            error_kinds[str(fallback.get("error") or "unknown")] += 1
    transfers = sorted(
        {
            int(profile.get("transfer_index"))
            for profile in profiles
            if isinstance(profile, dict) and profile.get("transfer_index") is not None
        }
    )
    row_keys = sorted(
        {
            str(profile.get("row_key"))
            for profile in profiles
            if isinstance(profile, dict) and profile.get("row_key") is not None
        }
    )
    best = []
    for surface in surfaces:
        candidate = surface.get("best_preserving_candidate") or {}
        best.append(
            {
                "target": surface.get("target"),
                "row_key": surface.get("row_key"),
                "surface_id": surface.get("surface_id"),
                "candidate_name": candidate.get("candidate_name"),
                "full_remainder_monomials": candidate.get("full_remainder_monomials"),
                "remainder_ffe_ops_over_rho": candidate.get("remainder_ffe_ops_over_rho"),
                "remainder_ffe_beats_rho": candidate.get("remainder_ffe_beats_rho"),
                "known_hit_root_count": candidate.get("known_hit_root_count"),
            }
        )
    return {
        "source": str(path),
        "transfers": transfers,
        "profile_row_keys": row_keys,
        "requested_profile_count": summary.get("requested_profile_count"),
        "materialized_exact_profile_count": summary.get("materialized_exact_profile_count"),
        "materialization_error_count": summary.get("materialization_error_count"),
        "surface_count": summary.get("surface_count"),
        "min_preserving_sage_factor_remainder_ffe_ops_over_rho": summary.get(
            "min_preserving_sage_factor_remainder_ffe_ops_over_rho"
        ),
        "preserving_sage_factor_remainder_below_rho_count": summary.get(
            "preserving_sage_factor_remainder_below_rho_count"
        ),
        "error_kinds": dict(sorted(error_kinds.items())),
        "best_preserving_candidates": best,
    }


def compact_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(key)) for record in records).items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--selector-glob", default=DEFAULT_SELECTOR_GLOB)
    parser.add_argument("--exact-glob", default=DEFAULT_EXACT_GLOB)
    parser.add_argument("--target", default="22050.cf1@11731")
    parser.add_argument("--row-key", default="22050.cf1@11731:uniform:256:salt165")
    parser.add_argument("--transfer-modulus", type=int, default=6)
    parser.add_argument("--transfer-residue", type=int, default=0)
    parser.add_argument("--min-transfer", type=int, default=376)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    selector_paths = sorted(Path(path) for path in glob.glob(str(args.state_dir / args.selector_glob)))
    exact_paths = sorted(Path(path) for path in glob.glob(str(args.state_dir / args.exact_glob)))
    wanted_salt = row_salt(args.row_key)
    selector_records: list[dict[str, Any]] = []
    neighbor_records: list[dict[str, Any]] = []
    for path in selector_paths:
        selector_records.extend(
            selector_profile_records(
                path,
                args.target,
                args.row_key,
                args.transfer_modulus,
                args.transfer_residue,
                args.min_transfer,
            )
        )
        if wanted_salt is not None:
            neighbor_records.extend(
                same_target_neighbor_records(
                    path,
                    args.target,
                    wanted_salt,
                    args.transfer_modulus,
                    args.transfer_residue,
                    args.min_transfer,
                )
            )

    by_transfer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selector_records:
        by_transfer[str(record["transfer_index"])].append(record)
    exact_attempts = [exact_attempt_record(path) for path in exact_paths]
    materialized = [
        attempt for attempt in exact_attempts if int(attempt.get("materialized_exact_profile_count") or 0) > 0
    ]
    exact_materialized_profile_count = sum(
        int(attempt.get("materialized_exact_profile_count") or 0) for attempt in exact_attempts
    )
    exact_remainder_below_rho_count = sum(
        int(attempt.get("preserving_sage_factor_remainder_below_rho_count") or 0)
        for attempt in exact_attempts
    )
    min_remainder = min(
        (
            float(attempt["min_preserving_sage_factor_remainder_ffe_ops_over_rho"])
            for attempt in exact_attempts
            if attempt.get("min_preserving_sage_factor_remainder_ffe_ops_over_rho") is not None
        ),
        default=None,
    )
    if materialized:
        interpretation = (
            "The frozen public selector appears in later selector artifacts, and mounted campaign "
            "sources allow exact-profile Sage materialization. Successful holdout materialization "
            "now shifts the question from coverage to whether below-rho remainder wins survive "
            "selector and neighbor controls."
        )
    else:
        interpretation = (
            "The frozen public selector appears in later selector artifacts, but current exact-profile "
            "attempts cannot materialize because the replay witness spec lacks the selected candidate."
        )
    output = {
        "schema": "ecdlp_full_remainder_salt165_holdout_coverage_audit_v1",
        "method": "frozen_selector_public_coverage_joined_to_exact_profile_attempts",
        "selector": {
            "target": args.target,
            "row_key": args.row_key,
            "row_salt": wanted_salt,
            "transfer_modulus": args.transfer_modulus,
            "transfer_residue": args.transfer_residue,
            "min_transfer": args.min_transfer,
        },
        "summary": {
            "selector_artifact_count": len(selector_paths),
            "selector_profile_count": len(selector_records),
            "selected_transfer_count": len(by_transfer),
            "selected_transfers": [int(transfer) for transfer in sorted(by_transfer, key=int)],
            "leaf_signature_counts": compact_counts(selector_records, "leaf_signature"),
            "policy_counts": compact_counts(selector_records, "policy"),
            "public_verified_label_count": sum(
                1 for record in selector_records if record.get("public_key_verified_label") is True
            ),
            "neighbor_control_profile_count": len(neighbor_records),
            "neighbor_control_distinct_salt_count": len({record["row_salt"] for record in neighbor_records}),
            "exact_attempt_count": len(exact_attempts),
            "exact_attempt_materialized_count": len(materialized),
            "exact_attempt_materialized_profile_count": exact_materialized_profile_count,
            "exact_attempt_remainder_below_rho_count": exact_remainder_below_rho_count,
            "exact_attempt_min_remainder_ops_over_rho": min_remainder,
            "exact_attempt_error_count": sum(
                int(attempt.get("materialization_error_count") or 0) for attempt in exact_attempts
            ),
            "interpretation": interpretation,
        },
        "selector_profiles_by_transfer": {
            transfer: records for transfer, records in sorted(by_transfer.items(), key=lambda item: int(item[0]))
        },
        "neighbor_controls_sample": sorted(
            neighbor_records,
            key=lambda record: (
                int(record["transfer_index"]),
                int(record["row_salt"]),
                str(record["leaf_signature"]),
            ),
        )[:80],
        "exact_attempts": exact_attempts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
