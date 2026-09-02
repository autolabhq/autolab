#!/usr/bin/env python3
"""Join public activation/orientation evidence to exact FFE line factors.

This audit is intentionally post-validation and mechanism-facing.  It does not
mine a new rule.  It checks whether a frozen public support-rule recovery is
explained by a specific public linear factor appearing in exact Sage profiles.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LINE = (270, 1, 2514)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_line(raw: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("line must be b_coeff,c_coeff,constant")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def line_fingerprint(line: tuple[int, int, int]) -> list[list[int]]:
    b_coeff, c_coeff, constant = line
    terms = []
    if constant:
        terms.append([0, 0, constant])
    if c_coeff:
        terms.append([0, 1, c_coeff])
    if b_coeff:
        terms.append([1, 0, b_coeff])
    return sorted(terms)


def row_salt(row_key: str) -> int | None:
    if ":salt" not in row_key:
        return None
    try:
        return int(row_key.rsplit(":salt", 1)[1])
    except ValueError:
        return None


def window_label(path: Path) -> str | None:
    match = re.search(r"(\d{3})[_-](\d{3})", path.name)
    if not match:
        return None
    return f"{int(match.group(1))}-{int(match.group(2))}"


def support_key(match: dict[str, Any]) -> str:
    support = sorted({int(index) for index in match.get("factor_support") or match.get("unsigned_indices") or []})
    return "+".join(str(index) for index in support)


def best_rule_result(case: dict[str, Any]) -> dict[str, Any] | None:
    rule_results = case.get("rule_results") or []
    if not rule_results:
        return None
    verified = [
        rule
        for rule in rule_results
        if rule.get("public_key_verified")
        and (rule.get("charged_models") or {}).get("measured_oriented_ops_over_rho") is not None
    ]
    if verified:
        return min(
            verified,
            key=lambda rule: float((rule.get("charged_models") or {}).get("measured_oriented_ops_over_rho")),
        )
    return rule_results[0]


def selected_support_rows(rule: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not rule:
        return []
    rows = []
    for match in rule.get("selected_xmatches") or []:
        if (
            support_key(match) == "0+5"
            and int(match.get("scheduled_trial") or 0) == 1
        ):
            rows.append(
                {
                    "row_key": match.get("row_key"),
                    "salt": row_salt(str(match.get("row_key") or "")),
                    "valid_relation": bool(match.get("valid_relation")),
                    "candidate_pos": match.get("candidate_pos"),
                    "event_summary": match.get("event_summary"),
                }
            )
    return rows


def load_orientation_cases(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    transfer_summary: dict[int, dict[str, Any]] = {}
    for path in paths:
        artifact = load_json(path)
        summary = artifact.get("summary") or {}
        for case in artifact.get("cases") or []:
            transfer = int(case.get("transfer_index") or 0)
            rows = case.get("rows") or []
            row_keys = [str(row.get("row_key")) for row in rows if row.get("row_key")]
            salts = sorted(salt for salt in (row_salt(row_key) for row_key in row_keys) if salt is not None)
            rule = best_rule_result(case)
            support_rows = selected_support_rows(rule)
            charged = (rule or {}).get("charged_models") or {}
            measured = charged.get("measured_oriented_ops_over_rho")
            record = {
                "source": str(path),
                "window": window_label(path),
                "case_key": case.get("case_key"),
                "target": case.get("target"),
                "transfer_index": transfer,
                "policy": case.get("policy"),
                "leaf_selector": case.get("leaf_selector"),
                "row_keys": row_keys,
                "row_salts": salts,
                "salt_mod3_pattern": sorted(salt % 3 for salt in salts),
                "activation_rule_matched": bool(case.get("activation_rule_matched")),
                "rule": (rule or {}).get("rule"),
                "public_key_verified": bool((rule or {}).get("public_key_verified")),
                "derived_secret": (rule or {}).get("derived_secret"),
                "rank": (rule or {}).get("rank"),
                "relation_count": (rule or {}).get("relation_count"),
                "measured_ops_over_rho": measured,
                "measured_below_rho": measured is not None and float(measured) < 1.0,
                "selected_support_rows": support_rows,
                "valid_support_row_keys": sorted(
                    str(row["row_key"]) for row in support_rows if row.get("valid_relation")
                ),
                "valid_support_salts": sorted(
                    int(row["salt"]) for row in support_rows if row.get("valid_relation") and row.get("salt") is not None
                ),
            }
            cases.append(record)
        for key in ("source_case_count", "activated_case_count", "verified_rule_count"):
            transfer_summary.setdefault(-1, {})[f"{path.name}:{key}"] = summary.get(key)
    return cases, transfer_summary


def factor_for_candidate(surface: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any] | None:
    try:
        index = int(str(candidate.get("candidate_name")).rsplit("_", 1)[1])
    except (TypeError, ValueError, IndexError):
        return None
    factors = ((surface.get("sage_resultant_factorization") or {}).get("factors") or [])
    if index < 0 or index >= len(factors):
        return None
    return factors[index]


def load_exact_surfaces(paths: list[Path], wanted_fingerprint: list[list[int]]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        artifact = load_json(path)
        for surface in artifact.get("surfaces") or []:
            exact_profile = surface.get("exact_profile") or {}
            row_key = str(surface.get("row_key") or exact_profile.get("row_key") or "")
            transfer = int(exact_profile.get("transfer_index") or 0)
            line_candidates = []
            for candidate in surface.get("preserving_candidates") or []:
                factor = factor_for_candidate(surface, candidate)
                if not factor or factor.get("fingerprint") != wanted_fingerprint:
                    continue
                line_candidates.append(
                    {
                        "candidate_name": candidate.get("candidate_name"),
                        "factor": factor,
                        "selected_root_pair_count": int(candidate.get("selected_root_pair_count") or 0),
                        "selected_linear_root_recoveries": int(candidate.get("selected_linear_root_recoveries") or 0),
                        "selected_valid_root_leaves": int(candidate.get("selected_valid_root_leaves") or 0),
                        "selected_missed_leaves": int(candidate.get("selected_missed_leaves") or 0),
                        "factor_root_scan_ops_over_rho": candidate.get("factor_root_scan_ops_over_rho"),
                        "surface_ffe_ops_over_rho": candidate.get("surface_ffe_ops_over_rho"),
                        "remainder_ffe_ops_over_rho": candidate.get("remainder_ffe_ops_over_rho"),
                    }
                )
            records.append(
                {
                    "source": str(path),
                    "window": window_label(path),
                    "target": surface.get("target"),
                    "transfer_index": transfer,
                    "row_key": row_key,
                    "salt": row_salt(row_key),
                    "selected_leaf_indices": surface.get("selected_leaf_indices"),
                    "has_wanted_line_preserving_candidate": bool(line_candidates),
                    "wanted_line_candidates": line_candidates,
                    "wanted_line_selected_root_pair_count": sum(
                        int(candidate.get("selected_root_pair_count") or 0) for candidate in line_candidates
                    ),
                    "wanted_line_linear_root_recoveries": sum(
                        int(candidate.get("selected_linear_root_recoveries") or 0) for candidate in line_candidates
                    ),
                    "best_preserving_candidate": surface.get("best_preserving_candidate"),
                    "full_remainder_ffe_ops_over_rho": surface.get("full_remainder_ffe_ops_over_rho"),
                }
            )
    return records


def unique_support_systems(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for case in cases:
        if not case.get("public_key_verified"):
            continue
        key = (
            case.get("transfer_index"),
            tuple(case.get("row_keys") or []),
            tuple(case.get("valid_support_row_keys") or []),
            case.get("derived_secret"),
        )
        current = grouped.get(key)
        if current is None or str(case.get("leaf_selector")) < str(current.get("leaf_selector")):
            grouped[key] = case
    return sorted(
        grouped.values(),
        key=lambda case: (
            int(case.get("transfer_index") or 0),
            str(case.get("policy")),
            str(case.get("leaf_selector")),
        ),
    )


def summarize_transfer(
    transfer: int,
    cases: list[dict[str, Any]],
    exact_surfaces: list[dict[str, Any]],
) -> dict[str, Any]:
    transfer_cases = [case for case in cases if int(case.get("transfer_index") or 0) == transfer]
    transfer_exact = [surface for surface in exact_surfaces if int(surface.get("transfer_index") or 0) == transfer]
    activated = [case for case in transfer_cases if case.get("activation_rule_matched")]
    verified = [case for case in transfer_cases if case.get("public_key_verified")]
    measured_below = [case for case in verified if case.get("measured_below_rho")]
    line_rows = [
        surface
        for surface in transfer_exact
        if surface.get("has_wanted_line_preserving_candidate")
    ]
    line_selected_rows = [
        surface
        for surface in line_rows
        if int(surface.get("wanted_line_selected_root_pair_count") or 0) > 0
    ]
    representative = measured_below[0] if measured_below else (activated[0] if activated else (transfer_cases[0] if transfer_cases else None))
    support_row_keys = set(representative.get("valid_support_row_keys") or []) if representative else set()
    line_selected_row_keys = {str(surface.get("row_key")) for surface in line_selected_rows}
    return {
        "transfer_index": transfer,
        "window": next((case.get("window") for case in transfer_cases if case.get("window")), None)
        or next((surface.get("window") for surface in transfer_exact if surface.get("window")), None),
        "orientation_case_count": len(transfer_cases),
        "activated_case_count": len(activated),
        "verified_case_count": len(verified),
        "measured_below_rho_case_count": len(measured_below),
        "best_measured_ops_over_rho": min(
            (float(case["measured_ops_over_rho"]) for case in verified if case.get("measured_ops_over_rho") is not None),
            default=None,
        ),
        "derived_secrets": sorted({case.get("derived_secret") for case in verified if case.get("derived_secret") is not None}),
        "representative_row_salts": representative.get("row_salts") if representative else [],
        "representative_salt_mod3_pattern": representative.get("salt_mod3_pattern") if representative else [],
        "valid_support_salts": sorted({row_salt(row_key) for row_key in support_row_keys if row_salt(row_key) is not None}),
        "exact_profile_count": len(transfer_exact),
        "wanted_line_preserving_row_salts": sorted(
            salt for salt in (surface.get("salt") for surface in line_rows) if salt is not None
        ),
        "wanted_line_selected_root_row_salts": sorted(
            salt for salt in (surface.get("salt") for surface in line_selected_rows) if salt is not None
        ),
        "wanted_line_selected_root_row_count": len(line_selected_rows),
        "support_rows_equal_line_selected_rows": bool(support_row_keys)
        and support_row_keys == line_selected_row_keys,
        "mechanism_match": bool(measured_below)
        and len(support_row_keys) == 2
        and support_row_keys == line_selected_row_keys,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orientation-artifact", type=Path, action="append", required=True)
    parser.add_argument("--exact-profile-artifact", type=Path, action="append", required=True)
    parser.add_argument("--line", type=parse_line, default=DEFAULT_LINE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    wanted_fingerprint = line_fingerprint(args.line)
    orientation_cases, _orientation_artifact_summary = load_orientation_cases(args.orientation_artifact)
    exact_surfaces = load_exact_surfaces(args.exact_profile_artifact, wanted_fingerprint)
    support_systems = unique_support_systems(orientation_cases)

    exact_transfers = {int(surface.get("transfer_index") or 0) for surface in exact_surfaces}
    orientation_transfers = {int(case.get("transfer_index") or 0) for case in orientation_cases}
    transfer_summaries = [
        summarize_transfer(transfer, orientation_cases, exact_surfaces)
        for transfer in sorted(exact_transfers | orientation_transfers)
    ]
    mechanism_systems = []
    for system in support_systems:
        line_selected_rows = {
            str(surface.get("row_key"))
            for surface in exact_surfaces
            if int(surface.get("transfer_index") or 0) == int(system.get("transfer_index") or 0)
            and int(surface.get("wanted_line_selected_root_pair_count") or 0) > 0
        }
        support_rows = set(system.get("valid_support_row_keys") or [])
        mechanism_systems.append(
            {
                "transfer_index": system.get("transfer_index"),
                "derived_secret": system.get("derived_secret"),
                "measured_ops_over_rho": system.get("measured_ops_over_rho"),
                "policy": system.get("policy"),
                "leaf_selector": system.get("leaf_selector"),
                "row_salts": system.get("row_salts"),
                "salt_mod3_pattern": system.get("salt_mod3_pattern"),
                "valid_support_salts": system.get("valid_support_salts"),
                "line_selected_root_salts": sorted(
                    salt for salt in (row_salt(row_key) for row_key in line_selected_rows) if salt is not None
                ),
                "support_rows_equal_line_selected_rows": support_rows == line_selected_rows,
                "mechanism_match": len(support_rows) == 2 and support_rows == line_selected_rows,
            }
        )

    output = {
        "schema": "ecdlp_leaf79_public_factor_line_mechanism_audit_v1",
        "method": "join_public_activation_orientation_replay_to_exact_sage_line_factors",
        "parameters": {
            "line": {
                "b_coeff": args.line[0],
                "c_coeff": args.line[1],
                "constant": args.line[2],
                "fingerprint": wanted_fingerprint,
                "text": f"{args.line[0]}*b + {args.line[1]}*c + {args.line[2]}",
            },
            "orientation_artifacts": [str(path) for path in args.orientation_artifact],
            "exact_profile_artifacts": [str(path) for path in args.exact_profile_artifact],
        },
        "summary": {
            "orientation_case_count": len(orientation_cases),
            "orientation_activated_case_count": sum(1 for case in orientation_cases if case.get("activation_rule_matched")),
            "orientation_verified_case_count": sum(1 for case in orientation_cases if case.get("public_key_verified")),
            "orientation_verified_measured_below_rho_case_count": sum(
                1
                for case in orientation_cases
                if case.get("public_key_verified") and case.get("measured_below_rho")
            ),
            "unique_verified_support_system_count": len(support_systems),
            "unique_mechanism_matched_support_system_count": sum(
                1 for system in mechanism_systems if system.get("mechanism_match")
            ),
            "exact_surface_count": len(exact_surfaces),
            "exact_wanted_line_preserving_surface_count": sum(
                1 for surface in exact_surfaces if surface.get("has_wanted_line_preserving_candidate")
            ),
            "exact_wanted_line_selected_root_surface_count": sum(
                1 for surface in exact_surfaces if int(surface.get("wanted_line_selected_root_pair_count") or 0) > 0
            ),
            "transfers_with_mechanism_match": [
                summary["transfer_index"] for summary in transfer_summaries if summary.get("mechanism_match")
            ],
            "transfers_with_no_activation": [
                summary["transfer_index"]
                for summary in transfer_summaries
                if summary.get("orientation_case_count") and not summary.get("activated_case_count")
            ],
            "interpretation": (
                "A mechanism match means the public support-rule relation rows "
                "are exactly the rows where the requested exact Sage line has a "
                "selected root pair.  It supports the factor-line explanation, "
                "but it is not by itself a new frozen validation rule."
            ),
        },
        "transfer_summaries": transfer_summaries,
        "mechanism_support_systems": mechanism_systems,
        "exact_surfaces": exact_surfaces,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
