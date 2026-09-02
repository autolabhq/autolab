#!/usr/bin/env python3
"""Audit the repeated priority-column accepted-form bridge.

The accepted priority-column path was singleton-bound until the later direct
frontier produced repeats at the same public top-k16/full-support carrier.  This
script isolates that recurrence, mines public salt/support tokens against
selected-priority no-rank controls, and emits exact-row replay candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_priority_recurrence_audit.v1"


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


def rank_gain(row: dict[str, Any]) -> int:
    return int_value((row.get("rank_score") or {}).get("rank_gain"))


def direct_rank_gain(row: dict[str, Any]) -> int:
    return int_value(row.get("direct_audit_rank_gain"))


def unique_gain(row: dict[str, Any]) -> int:
    return int_value((row.get("rank_score") or {}).get("unique_factor_relation_gain"))


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


def public_tokens(row: dict[str, Any]) -> set[str]:
    selected_support = support_tuple(row.get("selected_term_support"))
    selected_set = set(selected_support)
    selector = str(row.get("selector"))
    top_k = int_value(row.get("top_k"))
    support = support_key(selected_support)
    salts = salts_from_row_keys(row.get("row_keys") or [])
    tokens = {
        f"cost_bucket={cost_bucket(row.get('direct_ops_over_rho'))}",
        f"selected_support={support}",
        f"selector={selector}",
        f"selector_topk={selector}|{top_k}",
        f"selector_topk_support={selector}|{top_k}|{support}",
        f"support_size={len(selected_support)}",
        f"top_k={top_k}",
        f"topk_support={top_k}|{support}",
    }
    for column in selected_support:
        tokens.add(f"selected_has={column}")
    for left, right in [(1, 5), (10, 13), (10, 14), (11, 15), (13, 15)]:
        if left in selected_set and right in selected_set:
            tokens.add(f"selected_has_{left}_and_{right}")
    if salts:
        gap = salts[-1] - salts[0] if len(salts) >= 2 else 0
        tokens.update(
            {
                f"salt_count={len(salts)}",
                f"salt_gap={gap}",
                f"salt_min={salts[0]}",
                f"salt_max={salts[-1]}",
                f"salt_pair={','.join(str(salt) for salt in salts)}",
                f"salt_pair_mod4={','.join(str(salt % 4) for salt in salts)}",
                f"salt_sum_mod4={sum(salts) % 4}",
            }
        )
        for salt in salts:
            tokens.add(f"salt_has={salt}")
        tokens.add(f"salt_adjacent={gap == 1}")
    return tokens


def has_selected_priority(row: dict[str, Any], priorities: set[int]) -> bool:
    selected = {int_value(item) for item in row.get("selected_priority_columns") or []}
    return bool(selected & priorities)


def has_accepted_priority(row: dict[str, Any], priorities: set[int]) -> bool:
    accepted = {int_value(item) for item in row.get("accepted_priority_columns") or []}
    return bool(accepted & priorities)


def row_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_missing_columns": sorted(int_value(item) for item in row.get("accepted_missing_columns") or []),
        "accepted_priority_columns": sorted(int_value(item) for item in row.get("accepted_priority_columns") or []),
        "classification": row.get("classification"),
        "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
        "form_supports": [list(support_tuple(item)) for item in row.get("form_supports") or []],
        "rank_gain": rank_gain(row),
        "row_keys": row.get("row_keys") or [],
        "salts": salts_from_row_keys(row.get("row_keys") or []),
        "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
        "selector": row.get("selector"),
        "top_k": int_value(row.get("top_k")),
        "transfer_index": int_value(row.get("transfer_index")),
        "unique_factor_relation_gain": unique_gain(row),
    }


def token_stats(positive_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    pos_counts: Counter[str] = Counter()
    control_counts: Counter[str] = Counter()
    for row in positive_rows:
        pos_counts.update(public_tokens(row))
    for row in control_rows:
        control_counts.update(public_tokens(row))
    rows = []
    total_pos = len(positive_rows)
    total_controls = len(control_rows)
    for token in sorted(set(pos_counts) | set(control_counts)):
        pos = pos_counts[token]
        controls = control_counts[token]
        precision = pos / (pos + controls) if pos + controls else 0.0
        recall = pos / total_pos if total_pos else 0.0
        control_rate = controls / total_controls if total_controls else 0.0
        score = (precision * recall) + math.log2((pos + 0.5) / (controls + 0.5))
        rows.append(
            {
                "control_count": controls,
                "control_rate": round(control_rate, 8),
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
            int(item["control_count"]),
            str(item["token"]),
        )
    )
    return rows[:limit]


def merge_candidate(
    bucket: dict[tuple[Any, ...], dict[str, Any]],
    row: dict[str, Any],
    source: str,
    cert_by_identity: dict[tuple[Any, ...], dict[str, Any]],
) -> None:
    key = identity(row)
    item = bucket.setdefault(
        key,
        {
            "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
            "promoted_score": None,
            "range": row.get("range"),
            "row_keys": row.get("row_keys") or [],
            "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
            "selector": row.get("selector"),
            "soft_score": None,
            "sources": [],
            "top_k": int_value(row.get("top_k")),
            "transfer_index": int_value(row.get("transfer_index")),
        },
    )
    if source not in item["sources"]:
        item["sources"].append(source)
    cert = cert_by_identity.get(key)
    if cert:
        item["direct_audit_matched"] = True
        item["direct_audit_classification"] = cert.get("classification")
        item["direct_audit_rank_gain"] = rank_gain(cert)
    else:
        item.setdefault("direct_audit_matched", bool(row.get("direct_audit_matched", False)))
        item.setdefault("direct_audit_classification", row.get("direct_audit_classification", "NO_DIRECT_AUDIT_MATCH"))
        item.setdefault("direct_audit_rank_gain", row.get("direct_audit_rank_gain"))
    if source == "promoted":
        item["promoted_score"] = max(int_value(item.get("promoted_score")), int_value(row.get("score")))
    if source == "soft":
        item["soft_score"] = max(float_value(item.get("soft_score")) or 0.0, float_value(row.get("score")) or 0.0)


def candidate_pool(
    promoted: dict[str, Any],
    soft: dict[str, Any] | None,
    bridge_rows: list[dict[str, Any]],
    start: int | None,
    end: int | None,
) -> list[dict[str, Any]]:
    cert_by_identity = {identity(row): row for row in bridge_rows}
    bucket: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in promoted.get("audited_candidates") or []:
        if in_range(row, start, end):
            merge_candidate(bucket, row, "promoted", cert_by_identity)
    if soft:
        for row in soft.get("holdout_candidates") or []:
            if in_range(row, start, end):
                merge_candidate(bucket, row, "soft", cert_by_identity)
    return list(bucket.values())


def score_open_rows(open_rows: list[dict[str, Any]], stats: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    weights = {
        str(item["token"]): float(item["score"])
        for item in stats
        if int_value(item.get("positive_count")) > 0 and float(item.get("score")) and float(item["score"]) > 0.0
    }
    scored = []
    for row in open_rows:
        tokens = public_tokens(row)
        matched = sorted(token for token in tokens if token in weights)
        if not matched:
            continue
        score = sum(weights[token] for token in matched)
        if "promoted" in (row.get("sources") or []):
            score += min(8.0, int_value(row.get("promoted_score")) / 3.0)
        if "soft" in (row.get("sources") or []):
            score += min(12.0, (float_value(row.get("soft_score")) or 0.0) / 4.0)
        scored.append(
            {
                "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
                "matched_public_tokens": matched[:30],
                "priority_score": round(score, 8),
                "promoted_score": row.get("promoted_score"),
                "row_keys": row.get("row_keys") or [],
                "salts": salts_from_row_keys(row.get("row_keys") or []),
                "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
                "selector": row.get("selector"),
                "soft_score": row.get("soft_score"),
                "sources": sorted(row.get("sources") or []),
                "top_k": int_value(row.get("top_k")),
                "transfer_index": int_value(row.get("transfer_index")),
            }
        )
    scored.sort(
        key=lambda item: (
            -float(item["priority_score"]),
            float(item["direct_ops_over_rho"]) if item["direct_ops_over_rho"] is not None else 999.0,
            int(item["transfer_index"]),
        )
    )
    return scored[:limit]


def positive_stable_tokens(stats: list[dict[str, Any]], positive_count: int) -> list[str]:
    return [
        str(item["token"])
        for item in stats
        if int_value(item.get("positive_count")) == positive_count and (float_value(item.get("score")) or 0.0) > 0.0
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-audit", required=True, type=Path)
    parser.add_argument("--promoted-audit", required=True, type=Path)
    parser.add_argument("--soft-scorer", type=Path)
    parser.add_argument("--priority-columns", default="15")
    parser.add_argument("--audit-start", type=int)
    parser.add_argument("--audit-end", type=int)
    parser.add_argument("--open-start", type=int)
    parser.add_argument("--open-end", type=int)
    parser.add_argument("--top-tokens", type=int, default=40)
    parser.add_argument("--open-candidate-limit", type=int, default=24)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    priorities = {int(item.strip()) for item in args.priority_columns.split(",") if item.strip()}
    bridge = load_json(args.bridge_audit)
    promoted = load_json(args.promoted_audit)
    soft = load_json(args.soft_scorer) if args.soft_scorer else None
    certificates = [
        row
        for row in bridge.get("certificates") or []
        if in_range(row, args.audit_start, args.audit_end)
    ]
    priority_positive = [
        row
        for row in certificates
        if has_accepted_priority(row, priorities) and rank_gain(row) > 0
    ]
    priority_controls = [
        row
        for row in certificates
        if has_selected_priority(row, priorities) and rank_gain(row) <= 0
    ]
    stats = token_stats(priority_positive, priority_controls, args.top_tokens)
    pool = candidate_pool(promoted, soft, certificates, args.open_start, args.open_end)
    open_rows = [row for row in pool if not row.get("direct_audit_matched")]
    queue = score_open_rows(open_rows, stats, args.open_candidate_limit)
    stable_tokens = positive_stable_tokens(stats, len(priority_positive))
    stable_salt_tokens = [token for token in stable_tokens if token.startswith("salt_")]
    payload = {
        "artifacts": {
            "bridge_audit": str(args.bridge_audit),
            "promoted_audit": str(args.promoted_audit),
            "soft_scorer": str(args.soft_scorer) if args.soft_scorer else None,
        },
        "claim_status": (
            "PRIORITY_RECURRENCE_HAS_STABLE_PUBLIC_SALT_RULE_AND_OPEN_QUEUE"
            if len(priority_positive) >= 3 and stable_salt_tokens and queue
            else "PRIORITY_RECURRENCE_HAS_PUBLIC_CARRIER_AND_OPEN_QUEUE"
            if len(priority_positive) >= 3 and queue
            else "PRIORITY_RECURRENCE_HAS_REPEATS_NO_OPEN_QUEUE"
            if len(priority_positive) >= 3
            else "PRIORITY_RECURRENCE_NOT_YET_REPEATED"
        ),
        "created_at": now_iso(),
        "honesty_boundary": [
            "TOY-EVIDENCE: controlled toy-prime ECDLP harness only.",
            "Accepted priority-column labels seed and evaluate the recurrence; open-candidate ordering uses public row tokens.",
            "Open queue entries are work orders until direct certificate export and rank audit match them.",
            "No target descent, deployed-curve break, large-field/FFE result, or Pollard-rho speedup is claimed.",
        ],
        "open_priority_replay_queue": queue,
        "parameters": {
            "audit_end": args.audit_end,
            "audit_start": args.audit_start,
            "open_end": args.open_end,
            "open_start": args.open_start,
            "priority_columns": sorted(priorities),
            "top_tokens": args.top_tokens,
        },
        "priority_positive_rows": [row_summary(row) for row in priority_positive],
        "public_token_stats": stats,
        "schema": SCHEMA,
        "summary": {
            "open_candidate_count": len(open_rows),
            "open_queue_count": len(queue),
            "priority_positive_count": len(priority_positive),
            "priority_positive_rank_gain_total": sum(rank_gain(row) for row in priority_positive),
            "priority_positive_transfers": sorted({int_value(row.get("transfer_index")) for row in priority_positive}),
            "priority_positive_unique_gain_total": sum(unique_gain(row) for row in priority_positive),
            "selected_priority_no_rank_control_count": len(priority_controls),
            "stable_public_tokens": stable_tokens,
            "stable_salt_tokens": stable_salt_tokens,
            "top_open_transfers": [row["transfer_index"] for row in queue[:12]],
        },
    }
    write_json(args.out, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
