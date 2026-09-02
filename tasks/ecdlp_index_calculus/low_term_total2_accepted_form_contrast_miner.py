#!/usr/bin/env python3
"""Mine public contrasts for accepted-form rank-gain certificates.

The bridge audit can now distinguish several outcomes:

* selected priority-column rows that collapse or add no rank,
* repeated accepted-form rank gains such as support [10,14],
* the singleton accepted priority-column hit on support [11,15].

This script keeps those labels honest.  It mines only public row/support tokens
for future work-order selection, while reporting direct accepted-form tokens
separately as posthoc diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_accepted_form_contrast_miner.v1"


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


def support_tuple(raw: Any) -> tuple[int, ...]:
    return tuple(sorted(int_value(item) for item in (raw or [])))


def support_key(raw: Any) -> str:
    return ",".join(str(item) for item in support_tuple(raw))


def row_key_key(raw: Any) -> str:
    return "|".join(sorted(str(item) for item in (raw or [])))


def salts_from_row_keys(row_keys: Any) -> list[int]:
    salts = []
    for row_key in row_keys or []:
        match = re.search(r"salt(\d+)", str(row_key))
        if match:
            salts.append(int(match.group(1)))
    return sorted(salts)


def salt_gap_bucket(gap: int) -> str:
    if gap <= 2:
        return "le2"
    if gap <= 5:
        return "3_5"
    if gap <= 8:
        return "6_8"
    if gap <= 12:
        return "9_12"
    return "ge13"


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


def rank_gain(row: dict[str, Any]) -> int:
    return int_value((row.get("rank_score") or {}).get("rank_gain"))


def unique_gain(row: dict[str, Any]) -> int:
    return int_value((row.get("rank_score") or {}).get("unique_factor_relation_gain"))


def identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_value(row.get("transfer_index")),
        str(row.get("selector")),
        int_value(row.get("top_k")),
        row_key_key(row.get("row_keys")),
        support_key(row.get("selected_term_support")),
    )


def in_range(row: dict[str, Any], start: int | None, end: int | None) -> bool:
    transfer = int_value(row.get("transfer_index"))
    return (start is None or transfer >= start) and (end is None or transfer <= end)


def public_tokens(row: dict[str, Any]) -> set[str]:
    selected_support = support_tuple(row.get("selected_term_support"))
    selected_set = set(selected_support)
    selector = str(row.get("selector"))
    top_k = int_value(row.get("top_k"))
    support = ",".join(str(item) for item in selected_support)
    salts = salts_from_row_keys(row.get("row_keys") or [])
    selected_priority = sorted(int_value(item) for item in (row.get("selected_priority_columns") or []))

    tokens = {
        f"selector={selector}",
        f"top_k={top_k}",
        f"selector_topk={selector}|{top_k}",
        f"selected_support={support}",
        f"topk_support={top_k}|{support}",
        f"selector_topk_support={selector}|{top_k}|{support}",
        f"support_size={len(selected_support)}",
        f"cost_bucket={cost_bucket(row.get('direct_ops_over_rho'))}",
        f"selected_priority_count={len(selected_priority)}",
    }
    for column in selected_support:
        tokens.add(f"selected_has={column}")
    for column in selected_priority:
        tokens.add(f"selected_priority={column}")
    watched = {0, 1, 5, 10, 11, 13, 14, 15}
    for left in selected_support:
        for right in selected_support:
            if left < right and (left in watched or right in watched):
                tokens.add(f"selected_pair={left}:{right}")
    for left, right in [(0, 15), (10, 14), (11, 15), (13, 15), (10, 13), (1, 5)]:
        if left in selected_set and right in selected_set:
            tokens.add(f"selected_has_{left}_and_{right}")
    if 13 in selected_set:
        tokens.add("selected_has_13")
    if 15 in selected_set:
        tokens.add("selected_has_15")
    if salts:
        gap = salts[-1] - salts[0] if len(salts) >= 2 else 0
        tokens.update(
            {
                f"salt_count={len(salts)}",
                f"salt_gap={gap}",
                f"salt_gap_bucket={salt_gap_bucket(gap)}",
                f"salt_min_mod4={salts[0] % 4}",
                f"salt_max_mod4={salts[-1] % 4}",
                f"salt_sum_mod4={sum(salts) % 4}",
                f"salt_pair_mod4={','.join(str(salt % 4) for salt in salts)}",
                f"salt_min_bucket={(salts[0] // 4) * 4}",
                f"salt_max_bucket={(salts[-1] // 4) * 4}",
            }
        )
        if len(salts) >= 2:
            tokens.add(f"salt_adjacent={gap == 1}")
    return tokens


def direct_tokens(row: dict[str, Any]) -> set[str]:
    tokens = {f"classification={row.get('classification')}"}
    for support in row.get("form_supports") or []:
        key = support_key(support)
        tokens.add(f"form_support={key}")
    for column in row.get("accepted_missing_columns") or []:
        tokens.add(f"accepted_missing={int_value(column)}")
    for column in row.get("accepted_priority_columns") or []:
        tokens.add(f"accepted_priority={int_value(column)}")
    tokens.add(f"rank_gain={rank_gain(row)}")
    tokens.add(f"unique_gain_bucket={min(unique_gain(row), 5)}")
    return tokens


def has_form(row: dict[str, Any], support: tuple[int, ...]) -> bool:
    return any(support_tuple(item) == support for item in (row.get("form_supports") or []))


def has_accepted_priority(row: dict[str, Any], priorities: set[int]) -> bool:
    accepted = {int_value(item) for item in (row.get("accepted_priority_columns") or [])}
    return bool(accepted & priorities)


def has_selected_priority(row: dict[str, Any], priorities: set[int]) -> bool:
    selected = {int_value(item) for item in (row.get("selected_priority_columns") or [])}
    return bool(selected & priorities)


def row_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_missing_columns": sorted(int_value(item) for item in (row.get("accepted_missing_columns") or [])),
        "accepted_priority_columns": sorted(int_value(item) for item in (row.get("accepted_priority_columns") or [])),
        "classification": row.get("classification"),
        "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
        "form_supports": [list(support_tuple(item)) for item in (row.get("form_supports") or [])],
        "rank_gain": rank_gain(row),
        "row_keys": row.get("row_keys") or [],
        "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
        "selector": row.get("selector"),
        "top_k": int_value(row.get("top_k")),
        "transfer_index": int_value(row.get("transfer_index")),
        "unique_factor_relation_gain": unique_gain(row),
    }


def token_stats(
    positive_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
    token_fn,
    limit: int,
) -> list[dict[str, Any]]:
    pos_counts: Counter[str] = Counter()
    neg_counts: Counter[str] = Counter()
    for row in positive_rows:
        pos_counts.update(token_fn(row))
    for row in negative_rows:
        neg_counts.update(token_fn(row))
    rows = []
    total_pos = len(positive_rows)
    total_neg = len(negative_rows)
    for token in sorted(set(pos_counts) | set(neg_counts)):
        pos = pos_counts[token]
        neg = neg_counts[token]
        precision = pos / (pos + neg) if pos + neg else 0.0
        recall = pos / total_pos if total_pos else 0.0
        neg_rate = neg / total_neg if total_neg else 0.0
        # Favors high-positive, low-negative tokens without pretending this is a
        # calibrated probability model.
        score = (precision * recall) + math.log2((pos + 0.5) / (neg + 0.5))
        rows.append(
            {
                "negative_count": neg,
                "negative_rate": round(neg_rate, 8),
                "positive_count": pos,
                "precision": round(precision, 8),
                "recall": round(recall, 8),
                "score": round(score, 8),
                "token": token,
            }
        )
    rows.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["precision"]),
            -int(item["positive_count"]),
            int(item["negative_count"]),
            str(item["token"]),
        )
    )
    return rows[:limit]


def score_open_candidates(
    rows: list[dict[str, Any]],
    contrast_tokens: list[dict[str, Any]],
    start: int | None,
    end: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    weights = {
        str(item["token"]): float(item["score"])
        for item in contrast_tokens
        if int_value(item.get("positive_count")) > 0 and float(item.get("score") or 0.0) > 0.0
    }
    scored = []
    for row in rows:
        if not in_range(row, start, end):
            continue
        if row.get("direct_audit_matched"):
            continue
        tokens = public_tokens(row)
        matched = sorted(token for token in tokens if token in weights)
        if not matched:
            continue
        score = sum(weights[token] for token in matched)
        scored.append(
            {
                "contrast_score": round(score, 8),
                "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
                "matched_public_tokens": matched[:30],
                "row_keys": row.get("row_keys") or [],
                "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
                "selector": row.get("selector"),
                "top_k": int_value(row.get("top_k")),
                "transfer_index": int_value(row.get("transfer_index")),
            }
        )
    scored.sort(
        key=lambda item: (
            -float(item["contrast_score"]),
            float(item["direct_ops_over_rho"]) if item["direct_ops_over_rho"] is not None else 999.0,
            int(item["transfer_index"]),
            str(item["selector"]),
        )
    )
    return scored[:limit]


def summarize_forms(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[int, ...], dict[str, Any]] = defaultdict(
        lambda: {
            "accepted_missing_rank_gain_count": 0,
            "rank_gain_total": 0,
            "transfers": set(),
            "unique_factor_relation_gain_total": 0,
        }
    )
    for row in rows:
        if rank_gain(row) <= 0:
            continue
        for support in row.get("form_supports") or []:
            key = support_tuple(support)
            item = counts[key]
            if row.get("classification") == "ACCEPTED_MISSING_COLUMN_RANK_GAIN":
                item["accepted_missing_rank_gain_count"] += 1
            item["rank_gain_total"] += rank_gain(row)
            item["unique_factor_relation_gain_total"] += unique_gain(row)
            item["transfers"].add(int_value(row.get("transfer_index")))
    summaries = []
    for support, item in counts.items():
        summaries.append(
            {
                "accepted_missing_rank_gain_count": item["accepted_missing_rank_gain_count"],
                "rank_gain_total": item["rank_gain_total"],
                "support": list(support),
                "transfers": sorted(item["transfers"]),
                "unique_factor_relation_gain_total": item["unique_factor_relation_gain_total"],
            }
        )
    summaries.sort(
        key=lambda item: (
            -int(item["rank_gain_total"]),
            -int(item["accepted_missing_rank_gain_count"]),
            item["support"],
        )
    )
    return summaries


def make_contrast(
    name: str,
    positive_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
    open_rows: list[dict[str, Any]],
    open_start: int | None,
    open_end: int | None,
    top_tokens: int,
    open_limit: int,
) -> dict[str, Any]:
    public = token_stats(positive_rows, negative_rows, public_tokens, top_tokens)
    direct = token_stats(positive_rows, negative_rows, direct_tokens, top_tokens)
    return {
        "direct_diagnostic_tokens": direct,
        "name": name,
        "negative_count": len(negative_rows),
        "negative_transfers": sorted({int_value(row.get("transfer_index")) for row in negative_rows}),
        "open_candidate_scores": score_open_candidates(open_rows, public, open_start, open_end, open_limit),
        "positive_count": len(positive_rows),
        "positive_rows": [row_summary(row) for row in positive_rows],
        "positive_transfers": sorted({int_value(row.get("transfer_index")) for row in positive_rows}),
        "public_contrast_tokens": public,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-audit", required=True, type=Path)
    parser.add_argument("--promoted-audit", type=Path)
    parser.add_argument("--priority-columns", default="15")
    parser.add_argument("--audit-start", type=int)
    parser.add_argument("--audit-end", type=int)
    parser.add_argument("--open-start", type=int)
    parser.add_argument("--open-end", type=int)
    parser.add_argument("--top-tokens", type=int, default=25)
    parser.add_argument("--open-candidate-limit", type=int, default=20)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    priorities = {int(item.strip()) for item in args.priority_columns.split(",") if item.strip()}
    bridge = load_json(args.bridge_audit)
    certificates = [
        row
        for row in (bridge.get("certificates") or [])
        if in_range(row, args.audit_start, args.audit_end)
    ]
    open_rows: list[dict[str, Any]] = []
    if args.promoted_audit:
        promoted = load_json(args.promoted_audit)
        open_rows = promoted.get("audited_candidates") or []

    no_rank_selected_priority = [
        row
        for row in certificates
        if has_selected_priority(row, priorities) and rank_gain(row) <= 0
    ]
    all_no_rank = [row for row in certificates if rank_gain(row) <= 0]
    accepted_missing_rank_gain = [
        row
        for row in certificates
        if row.get("classification") == "ACCEPTED_MISSING_COLUMN_RANK_GAIN" and rank_gain(row) > 0
    ]
    accepted_priority_rank_gain = [
        row
        for row in accepted_missing_rank_gain
        if has_accepted_priority(row, priorities)
    ]
    accepted_10_14_rank_gain = [
        row
        for row in accepted_missing_rank_gain
        if has_form(row, (10, 14))
    ]

    contrasts = [
        make_contrast(
            "accepted_priority_rank_gain_vs_selected_priority_no_rank",
            accepted_priority_rank_gain,
            no_rank_selected_priority,
            open_rows,
            args.open_start,
            args.open_end,
            args.top_tokens,
            args.open_candidate_limit,
        ),
        make_contrast(
            "accepted_10_14_rank_gain_vs_selected_priority_no_rank",
            accepted_10_14_rank_gain,
            no_rank_selected_priority,
            open_rows,
            args.open_start,
            args.open_end,
            args.top_tokens,
            args.open_candidate_limit,
        ),
        make_contrast(
            "all_accepted_missing_rank_gain_vs_all_no_rank",
            accepted_missing_rank_gain,
            all_no_rank,
            open_rows,
            args.open_start,
            args.open_end,
            args.top_tokens,
            args.open_candidate_limit,
        ),
    ]

    payload = {
        "artifacts": {
            "bridge_audit": str(args.bridge_audit),
            "promoted_audit": str(args.promoted_audit) if args.promoted_audit else None,
        },
        "claim_status": (
            "ACCEPTED_FORM_CONTRAST_HAS_REPEATED_NONPRIORITY_RANK_GAIN"
            if len(accepted_10_14_rank_gain) >= 2
            else "ACCEPTED_FORM_CONTRAST_HAS_PRIORITY_SINGLETON_ONLY"
            if accepted_priority_rank_gain
            else "ACCEPTED_FORM_CONTRAST_HAS_NO_PRIORITY_RANK_GAIN"
        ),
        "contrasts": contrasts,
        "created_at": now_iso(),
        "honesty_boundary": [
            "TOY-EVIDENCE: controlled toy-prime ECDLP harness only.",
            "Public contrast tokens exclude direct accepted-form and accepted-column labels.",
            "Direct diagnostic tokens are posthoc labels and must not be used as future selectors.",
            "Open promoted candidates are not progress until direct certificate export and rank audit match them.",
            "No target descent, deployed-curve break, or large-field/FFE speedup is claimed.",
        ],
        "parameters": {
            "audit_end": args.audit_end,
            "audit_start": args.audit_start,
            "open_end": args.open_end,
            "open_start": args.open_start,
            "priority_columns": sorted(priorities),
            "top_tokens": args.top_tokens,
        },
        "schema": SCHEMA,
        "summary": {
            "accepted_10_14_rank_gain_count": len(accepted_10_14_rank_gain),
            "accepted_missing_rank_gain_count": len(accepted_missing_rank_gain),
            "accepted_priority_rank_gain_count": len(accepted_priority_rank_gain),
            "certificate_count": len(certificates),
            "form_rank_gain_summaries": summarize_forms(certificates),
            "no_rank_selected_priority_control_count": len(no_rank_selected_priority),
            "rank_gain_certificate_count": sum(1 for row in certificates if rank_gain(row) > 0),
        },
    }
    write_json(args.out, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
