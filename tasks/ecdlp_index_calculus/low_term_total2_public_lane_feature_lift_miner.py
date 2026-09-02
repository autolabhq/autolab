#!/usr/bin/env python3
"""Mine public-token sublanes inside a low-term total2 work-order lane.

The broad selected_has=13 carrier has both exported rank-gain comparators and
missing full-family targets.  This miner asks which public/source-side tokens
raise the exported rank-gain rate while still leaving missing targets to
validate.  It does not create new direct/rank evidence.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_public_lane_feature_lift_miner.v1"
DEFAULT_BASE_TOKEN = "selected_has=13"


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


def family_key(raw: Any) -> str:
    return ",".join(str(item) for item in support_tuple(raw))


def rank_gain(row: dict[str, Any]) -> int:
    return int_value((row.get("best_direct_certificate") or {}).get("rank_gain"))


def unique_gain(row: dict[str, Any]) -> int:
    return int_value((row.get("best_direct_certificate") or {}).get("unique_factor_relation_gain"))


def classification(row: dict[str, Any]) -> str:
    return str((row.get("best_direct_certificate") or {}).get("classification") or "missing")


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
    if cost < 0.60:
        return "lt_0.60"
    if cost < 0.65:
        return "0.60_0.65"
    if cost < 0.70:
        return "0.65_0.70"
    if cost < 0.75:
        return "0.70_0.75"
    if cost < 0.85:
        return "0.75_0.85"
    return "ge_0.85"


def gap_bucket(gap: int) -> str:
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


def parse_tokens(raw: str) -> tuple[str, ...]:
    tokens = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not tokens:
        raise ValueError("At least one base token is required")
    return tokens


def row_tokens(row: dict[str, Any]) -> set[str]:
    selected = support_tuple(row.get("selected_term_support"))
    selected_set = set(selected)
    families = [family_key(item) for item in row.get("matched_families") or []]
    family_set = set(families)
    selector = str(row.get("selector"))
    top_k = int_value(row.get("top_k"))
    salts = salts_from_row_keys(row.get("row_keys") or [])
    priority_hits = {int_value(item) for item in row.get("priority_hits") or []}
    tokens = {
        f"cost_bucket={cost_bucket(row.get('direct_ops_over_rho'))}",
        f"family_count={len(families)}",
        f"public_product_gate={bool(row.get('public_product_gate_selected'))}",
        f"selector={selector}",
        f"selector_topk={selector}|{top_k}",
        f"support_size={len(selected)}",
        f"top_k={top_k}",
        f"topk_support={top_k}|{support_key(selected)}",
    }
    if len(families) >= 3:
        tokens.add("full_family_match")
    for family in families:
        tokens.add(f"family_has={family}")
    for watched in ("0,5", "10,14", "11,15"):
        tokens.add(f"family_has_{watched}={watched in family_set}")
    for column in selected:
        tokens.add(f"selected_has={column}")
    for left, right in (
        (0, 5),
        (1, 5),
        (2, 4),
        (3, 5),
        (5, 6),
        (8, 11),
        (10, 13),
        (10, 14),
        (11, 15),
        (13, 15),
    ):
        if left in selected_set and right in selected_set:
            tokens.add(f"selected_has_{left}_and_{right}")
    for hit in priority_hits:
        tokens.add(f"priority_hit={hit}")
    tokens.add(f"priority_hit_15={15 in priority_hits}")
    if salts:
        gap = salts[-1] - salts[0] if len(salts) >= 2 else 0
        tokens.update(
            {
                f"salt_adjacent={gap == 1}",
                f"salt_count={len(salts)}",
                f"salt_gap={gap}",
                f"salt_gap_bucket={gap_bucket(gap)}",
                f"salt_max_mod4={salts[-1] % 4}",
                f"salt_max_mod8={salts[-1] % 8}",
                f"salt_min_mod4={salts[0] % 4}",
                f"salt_min_mod8={salts[0] % 8}",
                f"salt_pair_mod4={','.join(str(salt % 4) for salt in salts)}",
                f"salt_pair_mod8={','.join(str(salt % 8) for salt in salts)}",
                f"salt_sum_mod4={sum(salts) % 4}",
                f"salt_sum_mod8={sum(salts) % 8}",
            }
        )
        for salt in salts:
            tokens.add(f"salt_has={salt}")
    return tokens


def exactish_tokens(tokens: tuple[str, ...]) -> list[str]:
    exact_prefixes = (
        "salt_has=",
        "topk_support=",
    )
    return [token for token in tokens if token.startswith(exact_prefixes)]


def row_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification": classification(row),
        "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
        "direct_status": row.get("direct_status"),
        "matched_families": row.get("matched_families") or [],
        "rank_gain": rank_gain(row),
        "range": row.get("range"),
        "row_keys": row.get("row_keys") or [],
        "salts": salts_from_row_keys(row.get("row_keys") or []),
        "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
        "selector": row.get("selector"),
        "top_k": int_value(row.get("top_k")),
        "transfer_index": int_value(row.get("transfer_index")),
        "unique_factor_relation_gain": unique_gain(row),
    }


def sorted_unique(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for row in sorted(
        rows,
        key=lambda item: (
            int_value(item.get("transfer_index")),
            str(item.get("direct_status")),
            str(item.get("selected_term_support")),
            str(item.get("row_keys")),
        ),
    ):
        key = (
            int_value(row.get("transfer_index")),
            str(row.get("direct_status")),
            tuple(row.get("selected_term_support") or []),
            tuple(row.get("row_keys") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def rule_matches(rule: tuple[str, ...], token_set: set[str]) -> bool:
    return set(rule) <= token_set


def summarize_rule(
    rule: tuple[str, ...],
    exported_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    row_token_sets: dict[int, set[str]],
    total_positive: int,
    total_exported: int,
) -> dict[str, Any]:
    exported_match = [row for row in exported_rows if rule_matches(rule, row_token_sets[id(row)])]
    missing_match = [row for row in missing_rows if rule_matches(rule, row_token_sets[id(row)])]
    positive_rows = [row for row in exported_match if rank_gain(row) > 0]
    controls = [row for row in exported_match if rank_gain(row) <= 0]
    pos = len(positive_rows)
    control_count = len(controls)
    exported_count = len(exported_match)
    precision = pos / exported_count if exported_count else 0.0
    recall = pos / total_positive if total_positive else 0.0
    base_rate = total_positive / total_exported if total_exported else 0.0
    lift = precision / base_rate if base_rate else 0.0
    missing_full = [row for row in missing_match if len(row.get("matched_families") or []) >= 3]
    missing_transfers = sorted({int_value(row.get("transfer_index")) for row in missing_match})
    missing_full_transfers = sorted({int_value(row.get("transfer_index")) for row in missing_full})
    exactish = exactish_tokens(rule)
    score = (
        math.log2((pos + 0.5) / (control_count + 0.5))
        + (0.9 * recall)
        + (0.08 * len(missing_full_transfers))
        - (0.04 * len(rule))
        - (0.25 if exactish else 0.0)
    )
    return {
        "control_count": control_count,
        "control_transfers": sorted({int_value(row.get("transfer_index")) for row in controls}),
        "exactish_tokens": exactish,
        "exported_count": exported_count,
        "lift_over_base": round(lift, 8),
        "missing_full_family_count": len(missing_full),
        "missing_full_family_transfers": missing_full_transfers,
        "missing_row_count": len(missing_match),
        "missing_transfers": missing_transfers,
        "positive_count": pos,
        "positive_transfers": sorted({int_value(row.get("transfer_index")) for row in positive_rows}),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "score": round(score, 8),
        "tokens": list(rule),
        "top_missing_rows": [row_summary(row) for row in missing_full[:10]],
        "top_positive_rows": [row_summary(row) for row in positive_rows[:10]],
    }


def mine_rules(
    rows: list[dict[str, Any]],
    base_tokens: tuple[str, ...],
    max_size: int,
    min_positive_support: int,
    min_missing_transfers: int,
    limit: int,
) -> dict[str, Any]:
    row_token_sets = {id(row): row_tokens(row) for row in rows}
    base_rows = [row for row in rows if set(base_tokens) <= row_token_sets[id(row)]]
    exported_rows = [row for row in base_rows if row.get("direct_status") == "direct_certificate_exported"]
    missing_rows = [row for row in base_rows if row.get("direct_status") == "direct_certificate_missing"]
    total_positive = sum(1 for row in exported_rows if rank_gain(row) > 0)
    total_exported = len(exported_rows)
    candidate_tokens = sorted({token for row in exported_rows for token in row_token_sets[id(row)]})
    candidate_tokens = [token for token in candidate_tokens if token not in set(base_tokens)]

    rules: list[dict[str, Any]] = []
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(candidate_tokens, size):
            rule = tuple(sorted((*base_tokens, *combo)))
            summary = summarize_rule(rule, exported_rows, missing_rows, row_token_sets, total_positive, total_exported)
            if summary["positive_count"] < min_positive_support:
                continue
            if len(summary["missing_full_family_transfers"]) < min_missing_transfers:
                continue
            rules.append(summary)

    rules.sort(
        key=lambda item: (
            bool(item.get("exactish_tokens")),
            -float(item.get("score") or 0),
            -float(item.get("precision") or 0),
            -int_value(item.get("positive_count")),
            -len(item.get("missing_full_family_transfers") or []),
            item.get("tokens") or [],
        )
    )
    robust_rules = [
        rule
        for rule in rules
        if not rule.get("exactish_tokens")
        and float(rule.get("precision") or 0) >= 0.60
        and int_value(rule.get("positive_count")) >= min_positive_support
        and len(rule.get("missing_full_family_transfers") or []) >= min_missing_transfers
    ]
    return {
        "base_lane": summarize_lane(base_rows),
        "candidate_token_count": len(candidate_tokens),
        "promoted_rule": robust_rules[0] if robust_rules else (rules[0] if rules else None),
        "robust_rule_count": len(robust_rules),
        "robust_rules": robust_rules[:limit],
        "top_rules": rules[:limit],
    }


def summarize_lane(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exported = [row for row in rows if row.get("direct_status") == "direct_certificate_exported"]
    missing = [row for row in rows if row.get("direct_status") == "direct_certificate_missing"]
    positives = [row for row in exported if rank_gain(row) > 0]
    full_missing = [row for row in missing if len(row.get("matched_families") or []) >= 3]
    classes = Counter(classification(row) for row in rows)
    return {
        "classification_counts": dict(sorted(classes.items())),
        "exported_count": len(exported),
        "exported_rank_gain_count": len(positives),
        "exported_rank_gain_rate": round(len(positives) / len(exported), 8) if exported else 0.0,
        "missing_count": len(missing),
        "missing_full_family_count": len(full_missing),
        "missing_full_family_transfers": sorted({int_value(row.get("transfer_index")) for row in full_missing}),
        "row_count": len(rows),
        "unique_transfer_count": len({int_value(row.get("transfer_index")) for row in rows}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workorder", required=True, type=Path)
    parser.add_argument("--base-tokens", default=DEFAULT_BASE_TOKEN)
    parser.add_argument("--max-size", type=int, default=2)
    parser.add_argument("--min-positive-support", type=int, default=3)
    parser.add_argument("--min-missing-transfers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workorder = load_json(args.workorder)
    raw_rows = list(workorder.get("candidate_rows") or [])
    rows = sorted_unique(raw_rows)
    base_tokens = parse_tokens(args.base_tokens)
    result = mine_rules(
        rows,
        base_tokens=base_tokens,
        max_size=max(1, args.max_size),
        min_positive_support=max(1, args.min_positive_support),
        min_missing_transfers=max(0, args.min_missing_transfers),
        limit=max(1, args.limit),
    )
    payload = {
        "artifacts": {
            "workorder": str(args.workorder),
        },
        "claim_status": (
            "PUBLIC_LANE_HAS_PROMOTABLE_SUBLANE_WITH_MISSING_TARGETS"
            if result.get("promoted_rule")
            else "PUBLIC_LANE_HAS_NO_PROMOTABLE_SUBLANE"
        ),
        "created_at": now_iso(),
        "honesty_boundary": [
            "This miner scores public-token sublanes against already exported comparator labels.",
            "It does not generate direct/rank evidence for missing rows.",
            "Rules with exactish salt/support tokens are reported but not promoted as structural claims.",
            "Missing full-family transfers remain validation targets before any speedup claim.",
        ],
        "parameters": {
            "base_tokens": list(base_tokens),
            "limit": args.limit,
            "max_size": args.max_size,
            "min_missing_transfers": args.min_missing_transfers,
            "min_positive_support": args.min_positive_support,
        },
        "schema": SCHEMA,
        "summary": {
            "base_lane": result["base_lane"],
            "candidate_token_count": result["candidate_token_count"],
            "promoted_rule_tokens": (result.get("promoted_rule") or {}).get("tokens"),
            "raw_workorder_row_count": len(raw_rows),
            "robust_rule_count": result["robust_rule_count"],
            "unique_workorder_row_count": len(rows),
        },
        "promoted_rule": result.get("promoted_rule"),
        "robust_rules": result["robust_rules"],
        "top_rules": result["top_rules"],
    }
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "claim_status": payload["claim_status"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
