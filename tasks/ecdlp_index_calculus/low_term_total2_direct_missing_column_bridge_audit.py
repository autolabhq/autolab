#!/usr/bin/env python3
"""Audit direct-source bridges against missing branch-bank columns.

The branch-family duplicate-hit bank currently saturates a small set of factor
supports.  Direct-source certificates can expose broader selected supports, but
fresh validation showed that selected missing columns often collapse away in
the accepted relation forms.  This audit makes that distinction explicit.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATE_DIR = Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state")
SCHEMA = "ecdlp.low_term_total2_direct_missing_column_bridge_audit.v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def float_value(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int_set(raw: str) -> set[int]:
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def parse_paths(raw: str) -> list[Path]:
    paths: list[Path] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        matches = sorted(Path(path) for path in glob.glob(item))
        paths.extend(matches or [Path(item)])
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def artifact_label(path: Path, state_dir: Path) -> str:
    return str(Path(state_dir.name) / path.name)


def range_from_name(path: Path) -> tuple[int, int] | None:
    pairs = re.findall(r"(?<!\d)(\d{4})_(\d{4})(?!\d)", path.name)
    if not pairs:
        return None
    start, end = pairs[-1]
    return int(start), int(end)


def range_label(item: tuple[int, int] | None) -> str | None:
    if not item:
        return None
    return f"{item[0]}_{item[1]}"


def support_from_coeffs(coeffs: list[Any], order: int) -> tuple[int, ...]:
    return tuple(
        index
        for index, coeff in enumerate(coeffs[1:])
        if order and int_value(coeff) % order != 0
    )


def relation_support_key(support: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int_value(item) for item in support))


def load_branch_frontier(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    uncovered = set(int_value(item) for item in (payload.get("summary") or {}).get("branch_bank_uncovered_factor_columns") or [])
    branch_frontier = (
        ((payload.get("combined_branch_bank") or {}).get("target_eliminated_rank") or {})
        .get("unique_relation_frontier")
        or {}
    )
    support_rows = branch_frontier.get("support_rows") or []
    saturated_supports = {
        relation_support_key(row.get("factor_support") or [])
        for row in support_rows
        if row.get("factor_support")
    }
    work_order_columns: set[int] = set()
    for order in payload.get("next_work_orders") or []:
        for column in order.get("priority_columns") or []:
            work_order_columns.add(int_value(column))
    return {
        "artifact": str(path),
        "branch_bank_uncovered_factor_columns": sorted(uncovered),
        "branch_bank_saturated_supports": [list(support) for support in sorted(saturated_supports)],
        "work_order_columns": sorted(work_order_columns),
    }


def score_sort_key(path: Path) -> tuple[int, int, str]:
    item_range = range_from_name(path) or (0, 0)
    return item_range[1], item_range[0], path.name


def collect_rank_scores(paths: list[Path], state_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    scores: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(paths, key=score_sort_key):
        payload = load_json(path)
        for score in payload.get("scores") or []:
            candidate = score.get("candidate") or {}
            artifact = Path(str(candidate.get("artifact") or "")).name
            offset = int_value(candidate.get("artifact_offset"), -1)
            if not artifact or offset < 0:
                continue
            scores[(artifact, offset)] = {
                "artifact": artifact_label(path, state_dir),
                "claim_status": score.get("claim_status"),
                "deficiency_delta": int_value(score.get("deficiency_delta")),
                "rank_after": int_value(score.get("rank_after")),
                "rank_before": int_value(score.get("rank_before")),
                "rank_gain": int_value(score.get("rank_gain")),
                "unique_factor_relation_gain": int_value(score.get("unique_factor_relation_gain")),
            }
    return scores


def cert_rows(
    paths: list[Path],
    rank_scores: dict[tuple[str, int], dict[str, Any]],
    state_dir: Path,
    branch_uncovered: set[int],
    priority_columns: set[int],
    saturated_supports: set[tuple[int, ...]],
    start_min: int,
    end_max: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=score_sort_key):
        item_range = range_from_name(path)
        if not item_range:
            continue
        if item_range[1] < start_min or (end_max is not None and item_range[0] > end_max):
            continue
        payload = load_json(path)
        for offset, cert in enumerate(payload.get("certificates") or []):
            selected = cert.get("selected") or {}
            order = int_value(cert.get("order"))
            form_supports = sorted(
                {
                    support_from_coeffs(form.get("coeffs") or [], order)
                    for form in cert.get("forms") or []
                    if isinstance(form, dict)
                }
            )
            accepted_columns = {column for support in form_supports for column in support}
            selected_support = set(int_value(item) for item in cert.get("selected_term_support") or [])
            selected_missing = sorted(selected_support & branch_uncovered)
            accepted_missing = sorted(accepted_columns & branch_uncovered)
            accepted_missing_not_selected = sorted(set(accepted_missing) - set(selected_missing))
            selected_missing_not_accepted = sorted(set(selected_missing) - set(accepted_missing))
            selected_priority = sorted(selected_support & priority_columns)
            accepted_priority = sorted(accepted_columns & priority_columns)
            saturated_forms = [
                support for support in form_supports if relation_support_key(support) in saturated_supports
            ]
            score = rank_scores.get((path.name, offset)) or {}
            rank_gain = int_value(score.get("rank_gain"))
            if rank_gain > 0 and accepted_missing:
                classification = "ACCEPTED_MISSING_COLUMN_RANK_GAIN"
            elif rank_gain > 0:
                classification = "RANK_GAIN_WITHOUT_ACCEPTED_MISSING_COLUMN"
            elif accepted_missing:
                classification = "ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN"
            elif selected_missing and len(saturated_forms) == len(form_supports):
                classification = "SELECTED_MISSING_COLLAPSED_TO_SATURATED"
            elif selected_missing:
                classification = "SELECTED_MISSING_NO_ACCEPTED_MISSING_COLUMN"
            else:
                classification = "NO_MISSING_COLUMN_SIGNAL"
            rows.append(
                {
                    "accepted_missing_columns": accepted_missing,
                    "accepted_missing_not_selected_columns": accepted_missing_not_selected,
                    "accepted_priority_columns": accepted_priority,
                    "artifact": artifact_label(path, state_dir),
                    "artifact_offset": offset,
                    "classification": classification,
                    "direct_ops_over_rho": float_value(selected.get("direct_ops_over_rho")),
                    "form_supports": [list(support) for support in form_supports],
                    "forms_count": int_value(cert.get("forms_count")),
                    "public_key_verified": bool(cert.get("public_key_verified")),
                    "range": range_label(item_range),
                    "rank": int_value(cert.get("rank")),
                    "rank_score": score,
                    "row_keys": selected.get("row_keys") or [],
                    "saturated_form_supports": [list(support) for support in saturated_forms],
                    "selector": selected.get("selector"),
                    "selected_missing_columns": selected_missing,
                    "selected_missing_not_accepted_columns": selected_missing_not_accepted,
                    "selected_priority_columns": selected_priority,
                    "selected_term_support": sorted(selected_support),
                    "target": selected.get("target"),
                    "top_k": int_value(selected.get("top_k")),
                    "transfer_index": int_value(selected.get("transfer_index")),
                    "window": cert.get("window") or selected.get("window"),
                }
            )
    return rows


def collect_support_reports(
    paths: list[Path],
    direct_transfers: set[int],
    state_dir: Path,
    branch_uncovered: set[int],
    priority_columns: set[int],
    start_min: int,
    end_max: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=score_sort_key):
        item_range = range_from_name(path)
        if not item_range:
            continue
        if item_range[1] < start_min or (end_max is not None and item_range[0] > end_max):
            continue
        payload = load_json(path)
        for report in payload.get("case_reports") or []:
            selected_support = set(int_value(item) for item in report.get("selected_term_support") or [])
            transfer_index = int_value(report.get("transfer_index"))
            rows.append(
                {
                    "artifact": artifact_label(path, state_dir),
                    "direct_certificate_exported": transfer_index in direct_transfers,
                    "direct_public_key_verified": bool(report.get("direct_public_key_verified")),
                    "public_product_gate_selected": bool(report.get("public_product_gate_selected")),
                    "range": range_label(item_range),
                    "selected_missing_columns": sorted(selected_support & branch_uncovered),
                    "selected_priority_columns": sorted(selected_support & priority_columns),
                    "selected_term_support": sorted(selected_support),
                    "shared_product_public_key_verified": bool(report.get("shared_product_public_key_verified")),
                    "top_k": int_value(report.get("top_k")),
                    "transfer_index": transfer_index,
                }
            )
    return rows


def summarize_certificates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classifications = Counter(row["classification"] for row in rows)
    support_counts = Counter(
        tuple(support)
        for row in rows
        for support in row.get("form_supports") or []
    )
    direct_costs = [
        value
        for value in (float_value(row.get("direct_ops_over_rho")) for row in rows)
        if value is not None
    ]
    return {
        "accepted_missing_column_certificate_count": sum(1 for row in rows if row["accepted_missing_columns"]),
        "accepted_missing_not_selected_certificate_count": sum(
            1 for row in rows if row["accepted_missing_not_selected_columns"]
        ),
        "accepted_priority_column_certificate_count": sum(1 for row in rows if row["accepted_priority_columns"]),
        "certificate_count": len(rows),
        "classification_counts": dict(sorted(classifications.items())),
        "direct_below_rho_count": sum(1 for value in direct_costs if value < 1.0),
        "form_support_counts": [
            {"count": count, "support": list(support)}
            for support, count in support_counts.most_common()
        ],
        "passing_certificate_count": sum(1 for row in rows if row.get("public_key_verified")),
        "rank_gain_certificate_count": sum(1 for row in rows if int_value((row.get("rank_score") or {}).get("rank_gain")) > 0),
        "selected_missing_column_certificate_count": sum(1 for row in rows if row["selected_missing_columns"]),
        "selected_priority_column_certificate_count": sum(1 for row in rows if row["selected_priority_columns"]),
    }


def summarize_support_reports(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_missing = [row for row in rows if row.get("selected_missing_columns")]
    return {
        "case_report_count": len(rows),
        "direct_certificate_exported_count": sum(1 for row in rows if row.get("direct_certificate_exported")),
        "selected_missing_column_case_count": len(selected_missing),
        "selected_missing_without_direct_certificate_count": sum(
            1 for row in selected_missing if not row.get("direct_certificate_exported")
        ),
        "selected_priority_column_case_count": sum(1 for row in rows if row.get("selected_priority_columns")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-frontier", required=True, type=Path)
    parser.add_argument("--certificates", required=True)
    parser.add_argument("--rank-scorers", required=True)
    parser.add_argument("--support-scouts", default="")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR, type=Path)
    parser.add_argument("--priority-columns", default="")
    parser.add_argument("--start-min", default=5984, type=int)
    parser.add_argument("--end-max", default=None, type=int)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    branch = load_branch_frontier(args.branch_frontier)
    branch_uncovered = set(branch["branch_bank_uncovered_factor_columns"])
    priority_columns = parse_int_set(args.priority_columns) if args.priority_columns else set(branch["work_order_columns"])
    saturated_supports = {
        relation_support_key(support) for support in branch["branch_bank_saturated_supports"]
    }
    certificate_paths = parse_paths(args.certificates)
    scorer_paths = parse_paths(args.rank_scorers)
    support_paths = parse_paths(args.support_scouts) if args.support_scouts else []
    rank_scores = collect_rank_scores(scorer_paths, args.state_dir)
    certificates = cert_rows(
        certificate_paths,
        rank_scores,
        args.state_dir,
        branch_uncovered,
        priority_columns,
        saturated_supports,
        args.start_min,
        args.end_max,
    )
    direct_transfers = {int_value(row.get("transfer_index")) for row in certificates}
    support_reports = collect_support_reports(
        support_paths,
        direct_transfers,
        args.state_dir,
        branch_uncovered,
        priority_columns,
        args.start_min,
        args.end_max,
    )
    cert_summary = summarize_certificates(certificates)
    support_summary = summarize_support_reports(support_reports)
    claim_status = (
        "DIRECT_BRIDGE_ACCEPTED_MISSING_COLUMN_RANK_GAIN_FOUND"
        if cert_summary["rank_gain_certificate_count"] and cert_summary["accepted_missing_column_certificate_count"]
        else "DIRECT_BRIDGE_SELECTED_MISSING_COLUMNS_COLLAPSE_OR_DEPEND"
        if cert_summary["selected_missing_column_certificate_count"]
        else "DIRECT_BRIDGE_NO_MISSING_COLUMN_SIGNAL"
    )
    payload = {
        "artifacts": {
            "branch_frontier": str(args.branch_frontier),
            "certificates": [str(path) for path in certificate_paths],
            "rank_scorers": [str(path) for path in scorer_paths],
            "support_scouts": [str(path) for path in support_paths],
        },
        "branch_frontier": branch,
        "certificates": certificates,
        "claim_status": claim_status,
        "created_at": now_iso(),
        "honesty_boundary": [
            "TOY-EVIDENCE: controlled toy-prime ECDLP harness only.",
            "MODEL-BOUND: this reads mounted direct-certificate and rank-scorer artifacts without recomputing relation algebra.",
            "Selected support is not counted as rank progress unless accepted relation forms and rank-scorer evidence agree.",
            "This is a selector diagnostic, not target descent or a deployed-curve speedup claim.",
        ],
        "parameters": {
            "end_max": args.end_max,
            "priority_columns": sorted(priority_columns),
            "start_min": args.start_min,
        },
        "schema": SCHEMA,
        "summary": {
            **cert_summary,
            "claim_status": claim_status,
            "support_report_summary": support_summary,
        },
        "support_reports": support_reports,
    }
    write_json(args.out, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(json.dumps({"claim_status": claim_status}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
