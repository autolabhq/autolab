#!/usr/bin/env python3
"""Mine public features that predict accepted-form missing-column bridges.

The direct missing-column audit separates selected support from accepted
relation-form support.  This miner uses those audited certificates as labels,
then applies public/source-side tokens to fresh support-scout rows that do not
yet have exported direct certificates.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_accepted_form_public_feature_miner.v1"


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


def support_key(raw: Any) -> str:
    values = sorted(int_value(item) for item in (raw or []))
    return ",".join(str(item) for item in values)


def row_key_key(raw: Any) -> str:
    return "|".join(sorted(str(item) for item in (raw or [])))


def salts_from_row_keys(row_keys: Any) -> list[int]:
    salts = []
    for row_key in row_keys or []:
        match = re.search(r"salt(\d+)", str(row_key))
        if match:
            salts.append(int(match.group(1)))
    return sorted(salts)


def cost_bucket(value: Any) -> str:
    cost = float_value(value)
    if cost is None:
        return "missing"
    if cost < 0.70:
        return "lt_0.70"
    if cost < 0.75:
        return "0.70_0.75"
    if cost < 0.80:
        return "0.75_0.80"
    if cost < 0.90:
        return "0.80_0.90"
    return "ge_0.90"


def report_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_value(row.get("transfer_index")),
        row.get("selector"),
        int_value(row.get("top_k")),
        row_key_key(row.get("row_keys")),
        support_key(row.get("selected_term_support")),
    )


def load_support_reports(paths: list[Path], start_min: int, end_max: int | None) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        item_range = range_from_name(path)
        if not item_range:
            continue
        if item_range[1] < start_min or (end_max is not None and item_range[0] > end_max):
            continue
        payload = load_json(path)
        for offset, report in enumerate(payload.get("case_reports") or []):
            reports.append(
                {
                    **report,
                    "_artifact": str(path),
                    "_artifact_offset": offset,
                    "_range": range_label(item_range),
                }
            )
    return reports


def load_labeled_rows(bridge_audit: Path, support_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = load_json(bridge_audit)
    reports_by_identity = {report_identity(report): report for report in support_reports}
    rows = []
    for cert in payload.get("certificates") or []:
        support_report = reports_by_identity.get(report_identity(cert))
        if support_report is None:
            continue
        label = (
            "positive_rank_gain"
            if cert.get("classification") == "ACCEPTED_MISSING_COLUMN_RANK_GAIN"
            else "accepted_missing_dependent"
            if cert.get("classification") == "ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN"
            else "saturated_collapse"
            if cert.get("classification") == "SELECTED_MISSING_COLLAPSED_TO_SATURATED"
            else "other_nonpositive"
        )
        rows.append(
            {
                **cert,
                "_label": label,
                "_matched_support_report": True,
                "_support_report": support_report,
            }
        )
    return rows


def public_tokens(row: dict[str, Any], include_exact_row: bool = True) -> list[dict[str, str]]:
    report = row.get("_support_report") or row
    selected_support = sorted(int_value(item) for item in (row.get("selected_term_support") or report.get("selected_term_support") or []))
    selected_set = set(selected_support)
    selector = str(row.get("selector") or report.get("selector"))
    top_k = int_value(row.get("top_k") or report.get("top_k"))
    row_keys = row.get("row_keys") or report.get("row_keys") or []
    salts = salts_from_row_keys(row_keys)
    tokens: list[dict[str, str]] = []

    def add(token: str, family: str = "public") -> None:
        tokens.append({"family": family, "token": token})

    support = support_key(selected_support)
    add(f"selector={selector}")
    add(f"top_k={top_k}")
    add(f"selector_topk={selector}|{top_k}")
    add(f"selected_support={support}")
    add(f"topk_support={top_k}|{support}")
    add(f"selector_topk_support={selector}|{top_k}|{support}")
    add(f"support_size={len(selected_support)}")
    add(f"cost_bucket={cost_bucket(report.get('direct_ops_over_rho') or row.get('direct_ops_over_rho'))}")
    add(f"public_product_gate={bool(report.get('public_product_gate_selected'))}")
    add(f"priority_hit_15={15 in set(int_value(item) for item in (report.get('priority_hits') or []))}")
    for column in selected_support:
        add(f"selected_has={column}")
    for left in selected_support:
        for right in selected_support:
            if left < right and (left in {0, 13, 15} or right in {0, 13, 15}):
                add(f"selected_pair={left}:{right}")
    if 0 in selected_set and 15 in selected_set:
        add("selected_has_0_and_15")
    if 11 in selected_set and 15 in selected_set:
        add("selected_has_11_and_15")
    if 13 in selected_set:
        add("selected_has_13")
    if salts:
        add(f"salt_count={len(salts)}")
        add(f"salt_gap={salts[-1] - salts[0] if len(salts) >= 2 else 0}")
        add(f"salt_sum_mod4={sum(salts) % 4}")
        add(f"salt_min_mod4={salts[0] % 4}")
        add(f"salt_max_mod4={salts[-1] % 4}")
        if include_exact_row:
            add(f"exact_salts={','.join(str(salt) for salt in salts)}", "exact_row")
    if include_exact_row:
        add(f"exact_row_keys={row_key_key(row_keys)}", "exact_row")
    return tokens


def token_stats(labeled_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_token: dict[str, dict[str, Any]] = {}
    total_positive = sum(1 for row in labeled_rows if row["_label"] == "positive_rank_gain")
    for row in labeled_rows:
        seen = set()
        for token in public_tokens(row):
            key = token["token"]
            if key in seen:
                continue
            seen.add(key)
            item = by_token.setdefault(
                key,
                {
                    "collapse_count": 0,
                    "dependent_count": 0,
                    "examples": [],
                    "family": token["family"],
                    "nonpositive_count": 0,
                    "positive_count": 0,
                    "token": key,
                },
            )
            if row["_label"] == "positive_rank_gain":
                item["positive_count"] += 1
            else:
                item["nonpositive_count"] += 1
                if row["_label"] == "saturated_collapse":
                    item["collapse_count"] += 1
                if row["_label"] == "accepted_missing_dependent":
                    item["dependent_count"] += 1
            item["examples"].append(
                {
                    "classification": row.get("classification"),
                    "label": row["_label"],
                    "range": row.get("range"),
                    "transfer_index": row.get("transfer_index"),
                }
            )
    rows = []
    for item in by_token.values():
        positive = int_value(item["positive_count"])
        nonpositive = int_value(item["nonpositive_count"])
        rows.append(
            {
                **item,
                "precision_on_labeled": round(positive / (positive + nonpositive), 8)
                if positive + nonpositive
                else None,
                "recall_on_labeled": round(positive / total_positive, 8) if total_positive else None,
            }
        )
    rows.sort(
        key=lambda item: (
            -int_value(item["positive_count"]),
            int_value(item["nonpositive_count"]),
            item["family"] == "exact_row",
            str(item["token"]),
        )
    )
    return rows


def selected_rules(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strict = [
        row
        for row in stats
        if int_value(row.get("positive_count")) > 0
        and int_value(row.get("nonpositive_count")) == 0
        and row.get("family") != "exact_row"
    ]
    exact = [
        row
        for row in stats
        if int_value(row.get("positive_count")) > 0
        and int_value(row.get("nonpositive_count")) == 0
        and row.get("family") == "exact_row"
    ]
    return strict[:80] + exact[:20]


def apply_rules(
    support_reports: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    limit: int,
    promoted_only: bool = False,
) -> list[dict[str, Any]]:
    rule_by_token = {str(rule["token"]): rule for rule in rules if rule.get("family") != "exact_row"}
    candidates = []
    for report in support_reports:
        token_set = {token["token"] for token in public_tokens(report, include_exact_row=False)}
        matches = [rule_by_token[token] for token in sorted(token_set & set(rule_by_token))]
        if not matches:
            continue
        score = sum(int_value(rule.get("positive_count")) for rule in matches)
        strongest = sorted(
            matches,
            key=lambda item: (
                -int_value(item.get("positive_count")),
                str(item.get("token")),
            ),
        )[:8]
        match_tokens = {str(rule.get("token")) for rule in matches}
        if any(token.startswith("selector_topk_support=") for token in match_tokens):
            promotion_tier = "selector_topk_support"
        elif any(token.startswith("topk_support=") for token in match_tokens):
            promotion_tier = "topk_support"
        elif any(token.startswith("selector_topk=") for token in match_tokens):
            promotion_tier = "selector_topk"
        else:
            promotion_tier = "broad_atomic"
        if promoted_only and promotion_tier not in {"selector_topk_support", "topk_support"}:
            continue
        candidates.append(
            {
                "direct_ops_over_rho": float_value(report.get("direct_ops_over_rho")),
                "direct_public_key_verified_posthoc": bool(report.get("direct_public_key_verified")),
                "matched_rule_count": len(matches),
                "matched_rules": [
                    {
                        "positive_count": rule.get("positive_count"),
                        "recall_on_labeled": rule.get("recall_on_labeled"),
                        "token": rule.get("token"),
                    }
                    for rule in strongest
                ],
                "public_product_gate_selected": bool(report.get("public_product_gate_selected")),
                "promotion_tier": promotion_tier,
                "range": report.get("_range"),
                "row_keys": report.get("row_keys") or [],
                "score": score,
                "selected_term_support": sorted(int_value(item) for item in (report.get("selected_term_support") or [])),
                "selector": report.get("selector"),
                "shared_product_public_key_verified_posthoc": bool(report.get("shared_product_public_key_verified")),
                "top_k": int_value(report.get("top_k")),
                "transfer_index": int_value(report.get("transfer_index")),
            }
        )
    candidates.sort(
        key=lambda item: (
            -int_value(item.get("score")),
            -int_value(item.get("matched_rule_count")),
            float_value(item.get("direct_ops_over_rho")) or 999.0,
            int_value(item.get("transfer_index")),
            str(item.get("selector")),
            int_value(item.get("top_k")),
        )
    )
    return candidates[:limit]


def summarize_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_count": len(candidates),
        "direct_verified_posthoc_count": sum(1 for item in candidates if item.get("direct_public_key_verified_posthoc")),
        "ranges": dict(Counter(str(item.get("range")) for item in candidates)),
        "shared_verified_posthoc_count": sum(
            1 for item in candidates if item.get("shared_product_public_key_verified_posthoc")
        ),
        "tiers": dict(Counter(str(item.get("promotion_tier")) for item in candidates)),
        "transfers": sorted({int_value(item.get("transfer_index")) for item in candidates}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-audit", required=True, type=Path)
    parser.add_argument("--support-scouts", required=True)
    parser.add_argument("--calibration-start", type=int, default=5984)
    parser.add_argument("--calibration-end", type=int, default=6023)
    parser.add_argument("--holdout-start", type=int, default=6024)
    parser.add_argument("--holdout-end", type=int, default=6047)
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    support_paths = parse_paths(args.support_scouts)
    calibration_reports = load_support_reports(support_paths, args.calibration_start, args.calibration_end)
    holdout_reports = load_support_reports(support_paths, args.holdout_start, args.holdout_end)
    labeled_rows = load_labeled_rows(args.bridge_audit, calibration_reports)
    stats = token_stats(labeled_rows)
    rules = selected_rules(stats)
    holdout_candidates = apply_rules(holdout_reports, rules, args.candidate_limit)
    promoted_holdout_candidates = apply_rules(
        holdout_reports,
        rules,
        args.candidate_limit,
        promoted_only=True,
    )
    strict_public_rule_count = sum(1 for rule in rules if rule.get("family") != "exact_row")
    payload = {
        "artifacts": {
            "bridge_audit": str(args.bridge_audit),
            "support_scouts": [str(path) for path in support_paths],
        },
        "claim_status": (
            "PUBLIC_ACCEPTED_FORM_FEATURE_RULES_HAVE_HOLDOUT_CANDIDATES"
            if holdout_candidates and strict_public_rule_count
            else "PUBLIC_ACCEPTED_FORM_FEATURE_RULES_FOUND_NO_HOLDOUT_CANDIDATES"
            if strict_public_rule_count
            else "NO_STRICT_PUBLIC_ACCEPTED_FORM_FEATURE_RULES"
        ),
        "created_at": now_iso(),
        "holdout_candidates": holdout_candidates,
        "honesty_boundary": [
            "TOY-EVIDENCE: controlled toy-prime ECDLP harness only.",
            "MODEL-BOUND: labels come from direct bridge audit certificates and mounted rank-scorer outputs.",
            "The miner does not use direct_public_key_verified or shared_product_public_key_verified to rank candidates; those are reported only as posthoc fields.",
            "Strict rules with exact row-key tokens are reported separately and not used for holdout promotion.",
            "This is an accepted-form selector diagnostic, not target descent or a deployed-curve speedup claim.",
        ],
        "labeled_rows": [
            {
                "classification": row.get("classification"),
                "direct_ops_over_rho": row.get("direct_ops_over_rho"),
                "label": row.get("_label"),
                "matched_support_report": row.get("_matched_support_report"),
                "range": row.get("range"),
                "row_keys": row.get("row_keys"),
                "selected_term_support": row.get("selected_term_support"),
                "selector": row.get("selector"),
                "top_k": row.get("top_k"),
                "transfer_index": row.get("transfer_index"),
            }
            for row in labeled_rows
        ],
        "parameters": {
            "calibration_end": args.calibration_end,
            "calibration_start": args.calibration_start,
            "candidate_limit": args.candidate_limit,
            "holdout_end": args.holdout_end,
            "holdout_start": args.holdout_start,
        },
        "schema": SCHEMA,
        "promoted_holdout_candidates": promoted_holdout_candidates,
        "strict_rules": rules,
        "summary": {
            "calibration_label_counts": dict(Counter(row["_label"] for row in labeled_rows)),
            "holdout": summarize_candidates(holdout_candidates),
            "holdout_report_count": len(holdout_reports),
            "promoted_holdout": summarize_candidates(promoted_holdout_candidates),
            "strict_exact_row_rule_count": sum(1 for rule in rules if rule.get("family") == "exact_row"),
            "strict_public_rule_count": strict_public_rule_count,
            "token_count": len(stats),
        },
        "token_stats": stats[:200],
    }
    write_json(args.out, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(json.dumps({"claim_status": payload["claim_status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
