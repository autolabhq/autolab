#!/usr/bin/env python3
"""Mine public-token splits for a repeated accepted-form family.

The accepted-form contrast miner reports that a direct form family such as
[10,14] repeats with rank gain.  This script asks a narrower question: can
small conjunctions of public selector/support/salt tokens split that family
from selected-family no-rank controls without using direct accepted-form labels
as features?
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_family_public_split_miner.v1"


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


def parse_family(raw: str) -> tuple[int, ...]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) < 2:
        raise ValueError("--family must contain at least two comma-separated columns")
    return tuple(sorted(values))


def support_tuple(raw: Any) -> tuple[int, ...]:
    return tuple(sorted(int_value(item) for item in (raw or [])))


def support_key(raw: Any) -> str:
    return ",".join(str(item) for item in support_tuple(raw))


def rank_gain(row: dict[str, Any]) -> int:
    return int_value((row.get("rank_score") or {}).get("rank_gain"))


def unique_gain(row: dict[str, Any]) -> int:
    return int_value((row.get("rank_score") or {}).get("unique_factor_relation_gain"))


def in_range(row: dict[str, Any], start: int | None, end: int | None) -> bool:
    transfer = int_value(row.get("transfer_index"))
    return (start is None or transfer >= start) and (end is None or transfer <= end)


def salts_from_row_keys(row_keys: Any) -> list[int]:
    salts = []
    for row_key in row_keys or []:
        match = re.search(r"salt(\d+)", str(row_key))
        if match:
            salts.append(int(match.group(1)))
    return sorted(salts)


def bucket_gap(gap: int) -> str:
    if gap <= 1:
        return str(gap)
    if gap <= 3:
        return "2_3"
    if gap <= 7:
        return "4_7"
    if gap <= 11:
        return "8_11"
    if gap <= 15:
        return "12_15"
    return "ge16"


def has_form(row: dict[str, Any], family: tuple[int, ...]) -> bool:
    return any(support_tuple(item) == family for item in row.get("form_supports") or [])


def selected_has_family(row: dict[str, Any], family: tuple[int, ...]) -> bool:
    selected = set(support_tuple(row.get("selected_term_support")))
    return set(family) <= selected


def public_tokens(row: dict[str, Any], family: tuple[int, ...]) -> set[str]:
    selected_support = support_tuple(row.get("selected_term_support"))
    selected_set = set(selected_support)
    selector = str(row.get("selector"))
    top_k = int_value(row.get("top_k"))
    support = support_key(selected_support)
    family_key = ",".join(str(item) for item in family)
    tokens = {
        f"family_columns_present={sum(1 for item in family if item in selected_set)}",
        f"family_selected={set(family) <= selected_set}",
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
    watched_pairs = [
        (0, 15),
        (1, 5),
        (2, 4),
        (8, 11),
        (9, 11),
        (10, 13),
        (10, 14),
        (11, 15),
        (13, 15),
        tuple(family[:2]),
    ]
    for left, right in sorted(set(tuple(sorted(pair)) for pair in watched_pairs)):
        if left in selected_set and right in selected_set:
            tokens.add(f"selected_has_{left}_and_{right}")
    if set(family) <= selected_set:
        tokens.add(f"selected_has_family={family_key}")

    salts = salts_from_row_keys(row.get("row_keys") or [])
    if salts:
        gap = salts[-1] - salts[0] if len(salts) >= 2 else 0
        tokens.update(
            {
                f"salt_adjacent={gap == 1}",
                f"salt_count={len(salts)}",
                f"salt_gap={gap}",
                f"salt_gap_bucket={bucket_gap(gap)}",
                f"salt_max={salts[-1]}",
                f"salt_max_mod4={salts[-1] % 4}",
                f"salt_max_mod8={salts[-1] % 8}",
                f"salt_min={salts[0]}",
                f"salt_min_mod4={salts[0] % 4}",
                f"salt_min_mod8={salts[0] % 8}",
                f"salt_pair={','.join(str(salt) for salt in salts)}",
                f"salt_pair_mod4={','.join(str(salt % 4) for salt in salts)}",
                f"salt_pair_mod8={','.join(str(salt % 8) for salt in salts)}",
                f"salt_sum_mod4={sum(salts) % 4}",
                f"salt_sum_mod8={sum(salts) % 8}",
            }
        )
        for salt in salts:
            tokens.add(f"salt_has={salt}")
    return tokens


def label_row(row: dict[str, Any], family: tuple[int, ...]) -> str:
    if has_form(row, family) and row.get("classification") == "ACCEPTED_MISSING_COLUMN_RANK_GAIN" and rank_gain(row) > 0:
        return "target_family_positive"
    if selected_has_family(row, family) and rank_gain(row) <= 0:
        return "selected_family_no_rank_control"
    if selected_has_family(row, family) and rank_gain(row) > 0:
        return "selected_family_other_rank_gain"
    return "other"


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


def exactish_tokens(tokens: tuple[str, ...]) -> list[str]:
    prefixes = ("salt_pair=", "salt_min=", "salt_max=", "salt_has=")
    return [token for token in tokens if token.startswith(prefixes)]


def count_rules(
    positives: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    family: tuple[int, ...],
    max_size: int,
) -> tuple[dict[tuple[str, ...], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    for label, rows in [("positive", positives), ("control", controls)]:
        for index, row in enumerate(rows):
            records.append(
                {
                    "index": f"{label}:{index}",
                    "label": label,
                    "row": row,
                    "tokens": public_tokens(row, family),
                }
            )
    buckets: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in records:
        tokens = sorted(record["tokens"])
        for size in range(1, max_size + 1):
            for combo in itertools.combinations(tokens, size):
                bucket = buckets.setdefault(
                    combo,
                    {
                        "control_indices": set(),
                        "control_transfers": set(),
                        "positive_indices": set(),
                        "positive_transfers": set(),
                    },
                )
                if record["label"] == "positive":
                    bucket["positive_indices"].add(record["index"])
                    bucket["positive_transfers"].add(int_value(record["row"].get("transfer_index")))
                else:
                    bucket["control_indices"].add(record["index"])
                    bucket["control_transfers"].add(int_value(record["row"].get("transfer_index")))
    positive_records = [record for record in records if record["label"] == "positive"]
    control_records = [record for record in records if record["label"] == "control"]
    return buckets, positive_records, control_records


def summarize_rule(combo: tuple[str, ...], bucket: dict[str, Any], total_pos: int, total_controls: int) -> dict[str, Any]:
    pos = len(bucket["positive_indices"])
    controls = len(bucket["control_indices"])
    precision = pos / (pos + controls) if pos + controls else 0.0
    recall = pos / total_pos if total_pos else 0.0
    base_rate = total_pos / (total_pos + total_controls) if total_pos + total_controls else 0.0
    lift = precision / base_rate if base_rate else 0.0
    score = (precision * recall) + math.log2((pos + 0.5) / (controls + 0.5)) - (0.03 * len(combo))
    exactish = exactish_tokens(combo)
    return {
        "control_count": controls,
        "control_transfers": sorted(bucket["control_transfers"]),
        "exactish_tokens": exactish,
        "is_exactish": bool(exactish),
        "jackknife_train_positive_min": max(0, pos - 1),
        "lift_over_base": round(lift, 8),
        "positive_count": pos,
        "positive_transfers": sorted(bucket["positive_transfers"]),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "score": round(score, 8),
        "tokens": list(combo),
    }


def mine_rules(
    positives: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    family: tuple[int, ...],
    max_size: int,
    min_positive_support: int,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    buckets, _, _ = count_rules(positives, controls, family, max_size)
    all_rules = [
        summarize_rule(combo, bucket, len(positives), len(controls))
        for combo, bucket in buckets.items()
        if len(bucket["positive_indices"]) > 0
    ]
    all_rules.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["precision"]),
            -float(item["recall"]),
            int(item["control_count"]),
            len(item["tokens"]),
            str(item["tokens"]),
        )
    )
    zero_control = [
        item
        for item in all_rules
        if item["control_count"] == 0 and item["positive_count"] >= min_positive_support
    ]
    robust_zero_control = [
        item
        for item in zero_control
        if item["jackknife_train_positive_min"] >= min_positive_support and not item["is_exactish"]
    ]
    high_recall = [
        item
        for item in all_rules
        if item["positive_count"] == len(positives)
    ]
    high_recall.sort(
        key=lambda item: (
            int(item["control_count"]),
            len(item["tokens"]),
            float(item["precision"]) * -1,
            str(item["tokens"]),
        )
    )
    return (
        all_rules[:limit],
        zero_control[:limit],
        robust_zero_control[:limit],
        high_recall[:limit],
        {
            "all_rule_count": len(all_rules),
            "high_recall_rule_count": len(high_recall),
            "robust_zero_control_rule_count": len(robust_zero_control),
            "zero_control_rule_count": len(zero_control),
        },
    )


def eval_rule(rule: dict[str, Any], positives: list[dict[str, Any]], controls: list[dict[str, Any]], family: tuple[int, ...]) -> dict[str, Any]:
    tokens = set(rule.get("tokens") or [])
    covered_pos = [row for row in positives if tokens <= public_tokens(row, family)]
    covered_controls = [row for row in controls if tokens <= public_tokens(row, family)]
    pos = len(covered_pos)
    control_count = len(covered_controls)
    precision = pos / (pos + control_count) if pos + control_count else 0.0
    recall = pos / len(positives) if positives else 0.0
    return {
        "control_count": control_count,
        "control_transfers": sorted({int_value(row.get("transfer_index")) for row in covered_controls}),
        "positive_count": pos,
        "positive_transfers": sorted({int_value(row.get("transfer_index")) for row in covered_pos}),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "source_rule_tokens": rule.get("tokens") or [],
    }


def eval_rules(
    rules: list[dict[str, Any]],
    positives: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    family: tuple[int, ...],
    limit: int,
) -> list[dict[str, Any]]:
    return [eval_rule(rule, positives, controls, family) for rule in rules[:limit]]


def replay_payload(
    positives: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    family: tuple[int, ...],
    replay_start: int | None,
    max_rule_size: int,
    min_positive_support: int,
    limit: int,
) -> dict[str, Any] | None:
    if replay_start is None:
        return None
    train_pos = [row for row in positives if int_value(row.get("transfer_index")) < replay_start]
    train_controls = [row for row in controls if int_value(row.get("transfer_index")) < replay_start]
    replay_pos = [row for row in positives if int_value(row.get("transfer_index")) >= replay_start]
    replay_controls = [row for row in controls if int_value(row.get("transfer_index")) >= replay_start]
    if not train_pos:
        return {
            "status": "NO_PREFIX_POSITIVES",
            "replay_start": replay_start,
            "replay_positive_count": len(replay_pos),
            "train_positive_count": 0,
        }
    _, _, robust, high_recall, counts = mine_rules(
        train_pos,
        train_controls,
        family,
        max_rule_size,
        min_positive_support,
        limit,
    )
    replay_eval_limit = min(10, limit)
    robust_eval = eval_rules(robust, replay_pos, replay_controls, family, replay_eval_limit)
    high_recall_eval = eval_rules(high_recall, replay_pos, replay_controls, family, replay_eval_limit)
    validated_robust = [
        item
        for item in robust_eval
        if item["positive_count"] > 0 and item["control_count"] == 0
    ]
    validated_carrier = [
        item
        for item in high_recall_eval
        if item["positive_count"] == len(replay_pos) and item["control_count"] <= 1
    ]
    return {
        "high_recall_rule_replay": high_recall_eval,
        "prefix_rule_counts": counts,
        "replay_no_rank_control_count": len(replay_controls),
        "replay_positive_count": len(replay_pos),
        "replay_positive_transfers": sorted({int_value(row.get("transfer_index")) for row in replay_pos}),
        "replay_start": replay_start,
        "status": (
            "PREFIX_ROBUST_ZERO_CONTROL_RULE_VALIDATED_ON_TAIL"
            if validated_robust
            else "PREFIX_HIGH_RECALL_CARRIER_VALIDATED_ON_TAIL"
            if validated_carrier
            else "PREFIX_RULES_DID_NOT_VALIDATE_ON_TAIL"
        ),
        "train_no_rank_control_count": len(train_controls),
        "train_positive_count": len(train_pos),
        "train_positive_transfers": sorted({int_value(row.get("transfer_index")) for row in train_pos}),
        "validated_high_recall_rules": validated_carrier,
        "validated_robust_zero_control_rules": validated_robust,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-audit", required=True, type=Path)
    parser.add_argument("--family", default="10,14")
    parser.add_argument("--audit-start", type=int)
    parser.add_argument("--audit-end", type=int)
    parser.add_argument("--replay-start", type=int)
    parser.add_argument("--max-rule-size", type=int, default=3)
    parser.add_argument("--min-positive-support", type=int, default=2)
    parser.add_argument("--rule-limit", type=int, default=40)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    family = parse_family(args.family)
    bridge = load_json(args.bridge_audit)
    certificates = [
        row for row in bridge.get("certificates") or []
        if in_range(row, args.audit_start, args.audit_end)
    ]
    labels = Counter(label_row(row, family) for row in certificates)
    positives = [row for row in certificates if label_row(row, family) == "target_family_positive"]
    controls = [row for row in certificates if label_row(row, family) == "selected_family_no_rank_control"]
    other_rank_gain = [row for row in certificates if label_row(row, family) == "selected_family_other_rank_gain"]
    positive_selected_family_count = sum(1 for row in positives if selected_has_family(row, family))
    selected_family_warning = (
        "Some target-family positives do not contain every family column in selected_term_support; controls are not directly comparable."
        if positive_selected_family_count < len(positives)
        else None
    )
    all_rules, zero_control, robust_zero_control, high_recall, rule_counts = mine_rules(
        positives,
        controls,
        family,
        max(1, args.max_rule_size),
        max(1, args.min_positive_support),
        max(1, args.rule_limit),
    )
    replay = replay_payload(
        positives,
        controls,
        family,
        args.replay_start,
        max(1, args.max_rule_size),
        max(1, args.min_positive_support),
        max(1, args.rule_limit),
    )
    robust_full_recall = any(item.get("positive_count") == len(positives) for item in robust_zero_control)
    replay_status = (replay or {}).get("status")
    claim_status = (
        "FAMILY_PUBLIC_SPLIT_HAS_SELECTED_CARRIER_MISMATCH"
        if selected_family_warning
        else "FAMILY_PUBLIC_SPLIT_HAS_PREFIX_VALIDATED_ZERO_CONTROL_RULE"
        if replay_status == "PREFIX_ROBUST_ZERO_CONTROL_RULE_VALIDATED_ON_TAIL"
        else "FAMILY_PUBLIC_SPLIT_HAS_ROBUST_ZERO_CONTROL_FULL_RECALL_RULE"
        if robust_full_recall
        else "FAMILY_PUBLIC_SPLIT_HAS_ROBUST_ZERO_CONTROL_SUBCARRIER"
        if robust_zero_control
        else "FAMILY_PUBLIC_SPLIT_HAS_ZERO_CONTROL_OVERFIT_RULES"
        if zero_control
        else "FAMILY_PUBLIC_SPLIT_REMAINS_BROAD_CARRIER"
        if high_recall
        else "FAMILY_PUBLIC_SPLIT_HAS_NO_RECALL_CARRIER"
    )
    payload = {
        "artifacts": {
            "bridge_audit": str(args.bridge_audit),
        },
        "claim_status": claim_status,
        "created_at": now_iso(),
        "family_positive_rows": [row_summary(row) for row in positives],
        "high_recall_rules": high_recall,
        "honesty_boundary": [
            "TOY-EVIDENCE: controlled toy-prime ECDLP harness only.",
            "Public split rules use selector, selected support, and row-key salt tokens only.",
            "Direct accepted-form labels are used only to label/evaluate positives and controls.",
            "Prefix replay rules are still local toy evidence until future direct/rank tails validate them again.",
            "No target descent, deployed-curve break, large-field/FFE result, or Pollard-rho speedup is claimed.",
        ],
        "parameters": {
            "audit_end": args.audit_end,
            "audit_start": args.audit_start,
            "family": list(family),
            "max_rule_size": args.max_rule_size,
            "min_positive_support": args.min_positive_support,
            "replay_start": args.replay_start,
            "rule_limit": args.rule_limit,
        },
        "prefix_replay": replay,
        "robust_zero_control_rules": robust_zero_control,
        "schema": SCHEMA,
        "selected_family_no_rank_control_count": len(controls),
        "summary": {
            "best_high_recall_control_count": high_recall[0]["control_count"] if high_recall else None,
            "best_high_recall_precision": high_recall[0]["precision"] if high_recall else None,
            "best_robust_zero_control_recall": robust_zero_control[0]["recall"] if robust_zero_control else None,
            "best_robust_zero_control_tokens": robust_zero_control[0]["tokens"] if robust_zero_control else [],
            "family": list(family),
            "label_counts": dict(labels),
            "positive_not_selected_family_count": len(positives) - positive_selected_family_count,
            "positive_selected_family_count": positive_selected_family_count,
            "robust_zero_control_rule_count": rule_counts["robust_zero_control_rule_count"],
            "selected_family_warning": selected_family_warning,
            "selected_family_no_rank_control_count": len(controls),
            "selected_family_other_rank_gain_count": len(other_rank_gain),
            "target_family_positive_count": len(positives),
            "target_family_rank_gain_total": sum(rank_gain(row) for row in positives),
            "target_family_transfers": sorted({int_value(row.get("transfer_index")) for row in positives}),
            "target_family_unique_gain_total": sum(unique_gain(row) for row in positives),
            "top_high_recall_rule_tokens": high_recall[0]["tokens"] if high_recall else [],
            "zero_control_rule_count": rule_counts["zero_control_rule_count"],
        },
        "top_rules": all_rules,
        "zero_control_rules": zero_control,
    }
    write_json(args.out, payload)
    print(json.dumps({"claim_status": claim_status, "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
