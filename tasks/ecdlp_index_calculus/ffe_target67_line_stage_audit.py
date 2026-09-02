#!/usr/bin/env python3
"""Audit the target-67 FFE line stage across exact profiles and replays.

This is a boundary audit, not a new selector.  It keeps three costs separate:
public row selection, exact degree-1 line confirmation, and public orientation
replay.  The goal is to make clear which target-67 rows are relation-backed,
which have a preserving FFE line, which replays recover the secret, and where
an additive line-confirmation charge still loses to rho.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_target67_line_stage_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def parse_factor_index(candidate_name: str | None) -> int | None:
    if not candidate_name:
        return None
    try:
        return int(str(candidate_name).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def row_salt(row_key: str) -> int | None:
    if ":salt" not in row_key:
        return None
    try:
        return int(row_key.rsplit(":salt", 1)[1])
    except ValueError:
        return None


def leaf_signature(leaves: list[int]) -> str:
    return ",".join(str(int(leaf)) for leaf in sorted({int(leaf) for leaf in leaves}))


def profile_key_from_parts(
    target: str,
    transfer_index: int,
    top_k: int,
    policy: str,
    leaf_selector: str,
    row_key: str,
    leaves: list[int],
) -> tuple[Any, ...]:
    return (
        target,
        int(transfer_index),
        int(top_k),
        policy,
        leaf_selector,
        row_key,
        tuple(sorted({int(leaf) for leaf in leaves})),
    )


def profile_key_from_exact(exact_profile: dict[str, Any]) -> tuple[Any, ...]:
    return profile_key_from_parts(
        str(exact_profile.get("target") or ""),
        int(exact_profile.get("transfer_index") or 0),
        int(exact_profile.get("top_k") or 0),
        str(exact_profile.get("policy") or ""),
        str(exact_profile.get("leaf_selector") or ""),
        str(exact_profile.get("row_key") or ""),
        [int(leaf) for leaf in exact_profile.get("leaf_indices") or []],
    )


def factor_by_index(surface: dict[str, Any], index: int | None) -> dict[str, Any] | None:
    if index is None:
        return None
    factors = ((surface.get("sage_resultant_factorization") or {}).get("factors") or [])
    if index < 0 or index >= len(factors):
        return None
    return factors[index]


def line_from_fingerprint(fingerprint: list[list[int]], p: int) -> dict[str, Any] | None:
    terms = {
        (int(b_degree), int(c_degree)): int(coeff) % int(p)
        for b_degree, c_degree, coeff in fingerprint
    }
    if set(terms) != {(0, 0), (0, 1), (1, 0)}:
        return None
    return {
        "b_coeff": int(terms[(1, 0)]),
        "c_coeff": int(terms[(0, 1)]),
        "constant": int(terms[(0, 0)]),
        "p": int(p),
        "fingerprint": sorted(fingerprint),
        "line": f"{int(terms[(1, 0)])}*b + {int(terms[(0, 1)])}*c + {int(terms[(0, 0)])}",
    }


def replay_line_key(params: dict[str, Any]) -> tuple[int, int, int, int] | None:
    factor = params.get("factor") or {}
    try:
        return (
            int(factor["b_coeff"]),
            int(factor["c_coeff"]),
            int(factor["constant"]),
            int(factor["p"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def exact_case_labels(exact_source: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    labels = {}
    for case in exact_source.get("cases") or []:
        if not isinstance(case, dict):
            continue
        exact_profile = case.get("exact_profile") or {}
        if not exact_profile:
            continue
        labels[profile_key_from_exact(exact_profile)] = {
            "source_ops_over_rho": case.get("source_ops_over_rho"),
            "public_key_verified": bool(case.get("public_key_verified")),
            "rank": int(case.get("union_rank") or 0),
            "relation_count": int(case.get("union_relation_count") or 0),
            "reconstructed_error_count": len(case.get("reconstructed_errors") or []),
        }
    return labels


def best_verified_rule(replay: dict[str, Any]) -> dict[str, Any] | None:
    verified = [
        row
        for row in (replay.get("summary") or {}).get("best_verified_rules") or []
        if isinstance(row, dict)
    ]
    if not verified:
        return None
    return min(
        verified,
        key=lambda row: float((row.get("charged_models") or {}).get("measured_oriented_ops_over_rho") or 10**9),
    )


def replay_records(paths: list[Path], target: str) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        replay = load_json(path)
        params = replay.get("parameters") or {}
        if str(params.get("target") or "") != target:
            continue
        summary = replay.get("summary") or {}
        line_key = replay_line_key(params)
        best_rule = best_verified_rule(replay)
        best_charged = (best_rule or {}).get("charged_models") or {}
        case_rows = []
        for case in replay.get("cases") or []:
            if not isinstance(case, dict):
                continue
            case_rows.append(
                {
                    "case_key": case.get("case_key"),
                    "source_ops_over_rho": case.get("source_ops_over_rho"),
                    "source_public_key_verified": bool(case.get("source_public_key_verified")),
                    "activation_rule_matched": bool(case.get("activation_rule_matched")),
                    "activation_features": case.get("activation_features") or {},
                    "factor_zero_leaf_indices": [
                        int(row.get("leaf_index"))
                        for row in case.get("factor_zero_profiles") or []
                        if row.get("leaf_index") is not None
                    ],
                }
            )
        records.append(
            {
                "source": str(path),
                "target": target,
                "transfer_index": int(params.get("transfer_index") or 0),
                "top_k": int(params.get("top_k") or 0),
                "policy": params.get("policy"),
                "leaf_selector": params.get("leaf_selector"),
                "factor": params.get("factor") or {},
                "line_key": line_key,
                "activation_rule": params.get("activation_rule"),
                "activated_case_count": int(summary.get("activated_case_count") or 0),
                "verified_rule_count": int(summary.get("verified_rule_count") or 0),
                "verified_measured_below_rho_rule_count": int(
                    summary.get("verified_measured_below_rho_rule_count") or 0
                ),
                "min_verified_measured_ops_over_rho": summary.get("min_verified_measured_ops_over_rho"),
                "best_rule": (best_rule or {}).get("rule"),
                "derived_secret": (best_rule or {}).get("derived_secret"),
                "best_rule_selected_xmatch_count": (best_rule or {}).get("selected_xmatch_count"),
                "best_rule_measured_ops_over_rho": best_charged.get("measured_oriented_ops_over_rho"),
                "cases": case_rows,
            }
        )
    return records


def replay_index(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            int(record.get("transfer_index") or 0),
            int(record.get("top_k") or 0),
            str(record.get("policy") or ""),
            str(record.get("leaf_selector") or ""),
            record.get("line_key"),
        )
        out[key].append(record)
    return out


def break_even_reuse_count(fixed_ops: Any, variable_ops: Any) -> int | None:
    """Smallest reuse count n with fixed_ops / n + variable_ops < 1."""
    if fixed_ops is None or variable_ops is None:
        return None
    fixed = float(fixed_ops)
    variable = float(variable_ops)
    if variable >= 1.0:
        return None
    for count in range(1, 1001):
        if fixed / count + variable < 1.0:
            return count
    return None


def best_preserving_line(surface: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    candidate = surface.get("best_preserving_candidate")
    if not isinstance(candidate, dict):
        return None, None
    index = parse_factor_index(candidate.get("candidate_name"))
    factor = factor_by_index(surface, index)
    if not factor:
        return candidate, None
    return candidate, line_from_fingerprint(factor.get("fingerprint") or [], int(surface.get("p") or 0))


def exact_records(paths: list[Path], target: str, replay_by_key: dict[tuple[Any, ...], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        exact_source = load_json(path)
        labels = exact_case_labels(exact_source)
        for surface in exact_source.get("surfaces") or []:
            if not isinstance(surface, dict) or str(surface.get("target") or "") != target:
                continue
            exact_profile = surface.get("exact_profile") or {}
            leaves = [int(leaf) for leaf in exact_profile.get("leaf_indices") or surface.get("selected_leaf_indices") or []]
            key = profile_key_from_exact(exact_profile)
            label = labels.get(key, {})
            candidate, line = best_preserving_line(surface)
            factor_index = parse_factor_index((candidate or {}).get("candidate_name"))
            line_key = None
            if line:
                line_key = (
                    int(line["b_coeff"]),
                    int(line["c_coeff"]),
                    int(line["constant"]),
                    int(line["p"]),
                )
            replay_key = (
                int(exact_profile.get("transfer_index") or 0),
                int(exact_profile.get("top_k") or 0),
                str(exact_profile.get("policy") or ""),
                str(exact_profile.get("leaf_selector") or ""),
                line_key,
            )
            replays = replay_by_key.get(replay_key, [])
            best_replay = None
            if replays:
                best_replay = min(
                    replays,
                    key=lambda row: float(row.get("best_rule_measured_ops_over_rho") or 10**9),
                )
            row_ops = label.get("source_ops_over_rho")
            line_ops = (candidate or {}).get("factor_root_scan_ops_over_rho")
            replay_ops = (best_replay or {}).get("best_rule_measured_ops_over_rho")
            line_plus_replay = (
                round(float(line_ops) + float(replay_ops), 8)
                if line_ops is not None and replay_ops is not None
                else None
            )
            row_line_replay = (
                round(float(row_ops) + float(line_ops) + float(replay_ops), 8)
                if row_ops is not None and line_ops is not None and replay_ops is not None
                else None
            )
            line_replay_break_even = break_even_reuse_count(line_ops, replay_ops)
            row_replay_ops = (
                round(float(row_ops) + float(replay_ops), 8)
                if row_ops is not None and replay_ops is not None
                else None
            )
            row_line_replay_break_even = (
                break_even_reuse_count(line_ops, row_replay_ops)
                if row_replay_ops is not None
                else None
            )
            if not candidate:
                bucket = "no_preserving_line"
            elif best_replay and int(best_replay.get("verified_rule_count") or 0) > 0:
                bucket = "line_present_replay_success"
            elif best_replay:
                bucket = "line_present_replay_failure"
            else:
                bucket = "line_present_no_replay"
            records.append(
                {
                    "source": str(path),
                    "surface_id": surface.get("surface_id"),
                    "target": target,
                    "transfer_index": int(exact_profile.get("transfer_index") or 0),
                    "top_k": int(exact_profile.get("top_k") or 0),
                    "policy": exact_profile.get("policy"),
                    "leaf_selector": exact_profile.get("leaf_selector"),
                    "row_key": exact_profile.get("row_key") or surface.get("row_key"),
                    "salt": row_salt(str(exact_profile.get("row_key") or surface.get("row_key") or "")),
                    "leaf_indices": leaves,
                    "leaf_signature": leaf_signature(leaves),
                    "source_ops_over_rho": row_ops,
                    "row_public_key_verified": bool(label.get("public_key_verified")),
                    "row_rank": int(label.get("rank") or 0),
                    "row_relation_count": int(label.get("relation_count") or 0),
                    "factor_count": int((surface.get("sage_resultant_factorization") or {}).get("factor_count") or 0),
                    "preserving_candidate_count": int(surface.get("preserving_candidate_count") or 0),
                    "has_preserving_line": candidate is not None,
                    "preserving_factor_index": factor_index,
                    "preserving_line": line,
                    "line_surface_ffe_ops_over_rho": (candidate or {}).get("surface_ffe_ops_over_rho"),
                    "line_root_scan_ops_over_rho": line_ops,
                    "line_root_scan_beats_rho": bool((candidate or {}).get("factor_root_scan_beats_rho")),
                    "line_full_remainder_ops_over_rho": (candidate or {}).get("remainder_ffe_ops_over_rho"),
                    "replay_source": (best_replay or {}).get("source"),
                    "replay_activation_rule": (best_replay or {}).get("activation_rule"),
                    "replay_activated_case_count": (best_replay or {}).get("activated_case_count"),
                    "replay_verified_rule_count": (best_replay or {}).get("verified_rule_count"),
                    "replay_best_rule": (best_replay or {}).get("best_rule"),
                    "replay_secret": (best_replay or {}).get("derived_secret"),
                    "replay_best_rule_selected_xmatch_count": (best_replay or {}).get("best_rule_selected_xmatch_count"),
                    "replay_measured_ops_over_rho": replay_ops,
                    "line_plus_replay_ops_over_rho": line_plus_replay,
                    "row_line_replay_ops_over_rho": row_line_replay,
                    "line_replay_break_even_reuse_count": line_replay_break_even,
                    "row_replay_ops_over_rho": row_replay_ops,
                    "row_line_replay_break_even_reuse_count": row_line_replay_break_even,
                    "bucket": bucket,
                }
            )
    return sorted(
        records,
        key=lambda row: (
            int(row.get("transfer_index") or 0),
            str(row.get("leaf_signature") or ""),
            str(row.get("source") or ""),
        ),
    )


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 8) if values else None


def min_or_none(values: list[float]) -> float | None:
    return round(min(values), 8) if values else None


def max_or_none(values: list[float]) -> float | None:
    return round(max(values), 8) if values else None


def summarize(records: list[dict[str, Any]], replays: list[dict[str, Any]], window_reports: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts = Counter(str(row.get("bucket")) for row in records)
    line_rows = [row for row in records if row.get("has_preserving_line")]
    replay_success = [row for row in records if row.get("bucket") == "line_present_replay_success"]
    replay_failure = [row for row in records if row.get("bucket") == "line_present_replay_failure"]
    line_root_ratios = [
        float(row["line_root_scan_ops_over_rho"])
        for row in line_rows
        if row.get("line_root_scan_ops_over_rho") is not None
    ]
    line_plus_replay = [
        float(row["line_plus_replay_ops_over_rho"])
        for row in replay_success
        if row.get("line_plus_replay_ops_over_rho") is not None
    ]
    row_line_replay = [
        float(row["row_line_replay_ops_over_rho"])
        for row in replay_success
        if row.get("row_line_replay_ops_over_rho") is not None
    ]
    line_replay_reuse_counts = [
        int(row["line_replay_break_even_reuse_count"])
        for row in replay_success
        if row.get("line_replay_break_even_reuse_count") is not None
    ]
    row_line_replay_reuse_counts = [
        int(row["row_line_replay_break_even_reuse_count"])
        for row in replay_success
        if row.get("row_line_replay_break_even_reuse_count") is not None
    ]
    factor_indices = [
        int(row["preserving_factor_index"])
        for row in line_rows
        if row.get("preserving_factor_index") is not None
    ]
    public_branch_failures = sum(
        int(((report.get("summary") or {}).get("activation_opportunity_count") or 0) == 0)
        for report in window_reports
    )
    return {
        "target_surface_count": len(records),
        "row_verified_count": sum(bool(row.get("row_public_key_verified")) for row in records),
        "preserving_line_count": len(line_rows),
        "no_preserving_line_count": int(bucket_counts.get("no_preserving_line", 0)),
        "replay_success_count": int(bucket_counts.get("line_present_replay_success", 0)),
        "replay_failure_count": int(bucket_counts.get("line_present_replay_failure", 0)),
        "line_present_no_replay_count": int(bucket_counts.get("line_present_no_replay", 0)),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "replay_artifact_count": len(replays),
        "replay_artifact_success_count": sum(int(row.get("verified_rule_count") or 0) > 0 for row in replays),
        "line_root_scan_below_rho_count": sum(ratio < 1.0 for ratio in line_root_ratios),
        "min_line_root_scan_ops_over_rho": min_or_none(line_root_ratios),
        "mean_line_root_scan_ops_over_rho": mean_or_none(line_root_ratios),
        "max_line_root_scan_ops_over_rho": max_or_none(line_root_ratios),
        "preserving_factor_index_min": min(factor_indices) if factor_indices else None,
        "preserving_factor_index_max": max(factor_indices) if factor_indices else None,
        "preserving_factor_index_values": sorted(set(factor_indices)),
        "line_plus_replay_below_rho_count": sum(ratio < 1.0 for ratio in line_plus_replay),
        "min_line_plus_replay_ops_over_rho": min_or_none(line_plus_replay),
        "min_line_replay_break_even_reuse_count": min(line_replay_reuse_counts)
        if line_replay_reuse_counts
        else None,
        "max_line_replay_break_even_reuse_count": max(line_replay_reuse_counts)
        if line_replay_reuse_counts
        else None,
        "line_replay_break_even_reuse_count_values": sorted(set(line_replay_reuse_counts)),
        "row_line_replay_below_rho_count": sum(ratio < 1.0 for ratio in row_line_replay),
        "min_row_line_replay_ops_over_rho": min_or_none(row_line_replay),
        "row_line_replay_break_even_reuse_count_values": sorted(set(row_line_replay_reuse_counts)),
        "window_report_count": len(window_reports),
        "branch_reach_failure_window_count": public_branch_failures,
        "interpretation": (
            "Target-67 has repeated stagewise below-rho pieces: selected rows, "
            "preserving degree-1 lines, and orientation replays.  However, the "
            "naive additive line-confirmation plus replay charge does not beat "
            "rho on the replay-success rows in this audit, so the remaining "
            "algorithmic gap is public line prediction or amortized confirmation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source", type=Path, action="append", required=True)
    parser.add_argument("--replay-source", type=Path, action="append", default=[])
    parser.add_argument("--window-report", type=Path, action="append", default=[])
    parser.add_argument("--target", default="67.a1@9803")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    replays = replay_records(args.replay_source, str(args.target))
    records = exact_records(args.exact_source, str(args.target), replay_index(replays))
    window_reports = [load_json(path) for path in args.window_report]
    output = {
        "schema": "ecdlp_target67_line_stage_audit_v1",
        "method": "join_target67_exact_ffe_line_profiles_to_public_orientation_replays",
        "parameters": {
            "target": args.target,
            "exact_sources": [str(path) for path in args.exact_source],
            "replay_sources": [str(path) for path in args.replay_source],
            "window_reports": [str(path) for path in args.window_report],
        },
        "summary": summarize(records, replays, window_reports),
        "records": records,
        "replay_records": replays,
        "window_reports": window_reports,
        "non_claims": [
            "This audit does not mine or validate a new selector.",
            "line_plus_replay_ops_over_rho and row_line_replay_ops_over_rho are naive additive charges for boundary accounting; prior replay artifacts remain stagewise measurements.",
            "A below-rho replay still needs public line prediction or amortized exact-line confirmation to become an end-to-end index-calculus speedup.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
