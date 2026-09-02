#!/usr/bin/env python3
"""Materialize direct/rank validation rows for mined public-lane rules.

Feature-lift miners rank public/source-side sublanes using exported comparator
rows.  This script turns those rules into a concrete validation manifest by
scanning support-scout rows, marking whether bridge/direct evidence already
exists, and prioritizing rows that still need direct/rank export.
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


SCHEMA = "ecdlp.low_term_total2_public_lane_validation_manifest.v1"
DEFAULT_FAMILIES = "11,15;10,14;0,5"
DEFAULT_BROAD_TOKENS = ("selected_has=13", "salt_adjacent=False")


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
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        matches = sorted(Path(path) for path in glob.glob(chunk))
        paths.extend(matches or [Path(chunk)])
    seen: set[str] = set()
    unique = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def parse_families(raw: str) -> list[tuple[int, ...]]:
    families: list[tuple[int, ...]] = []
    for chunk in raw.split(";"):
        family = tuple(sorted(int(part.strip()) for part in chunk.split(",") if part.strip()))
        if family:
            families.append(family)
    if not families:
        raise ValueError("At least one family is required")
    return families


def range_from_path(path: Path) -> tuple[int, int] | None:
    match = re.search(r"_(\d+)_(\d+)_probe\.json$", path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def range_label(item: tuple[int, int] | None) -> str | None:
    if item is None:
        return None
    return f"{item[0]}_{item[1]}"


def range_end_from_artifact(path: Path, fallback: int | None = None) -> int | None:
    item = range_from_path(path)
    if item is not None:
        return item[1]
    return fallback


def support_tuple(raw: Any) -> tuple[int, ...]:
    return tuple(sorted(int_value(item) for item in (raw or [])))


def support_key(raw: Any) -> str:
    return ",".join(str(item) for item in support_tuple(raw))


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


def support_report_key(row: dict[str, Any]) -> tuple[int, int, tuple[int, ...]]:
    return (
        int_value(row.get("transfer_index")),
        int_value(row.get("top_k")),
        support_tuple(row.get("selected_term_support")),
    )


def bridge_support_index(bridge: dict[str, Any]) -> dict[tuple[int, int, tuple[int, ...]], dict[str, Any]]:
    return {support_report_key(row): row for row in bridge.get("support_reports") or []}


def certificate_index(bridge: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in bridge.get("certificates") or []:
        index[int_value(row.get("transfer_index"))].append(row)
    return index


def rank_gain(row: dict[str, Any]) -> int:
    return int_value((row.get("rank_score") or {}).get("rank_gain"))


def unique_gain(row: dict[str, Any]) -> int:
    return int_value((row.get("rank_score") or {}).get("unique_factor_relation_gain"))


def direct_status(row: dict[str, Any], support_index: dict[tuple[int, int, tuple[int, ...]], dict[str, Any]]) -> str:
    support_row = support_index.get(support_report_key(row))
    if support_row is None:
        return "support_report_missing"
    if support_row.get("direct_certificate_exported"):
        return "direct_certificate_exported"
    return "direct_certificate_missing"


def public_tokens(row: dict[str, Any], families: list[tuple[int, ...]]) -> set[str]:
    selected = support_tuple(row.get("selected_term_support"))
    selected_set = set(selected)
    family_matches = compatible_families(row, families)
    family_set = {support_key(family) for family in family_matches}
    selector = str(row.get("selector"))
    top_k = int_value(row.get("top_k"))
    priority_hits = {int_value(item) for item in row.get("priority_hits") or []}
    salts = salts_from_row_keys(row.get("row_keys") or [])
    tokens = {
        f"cost_bucket={cost_bucket(row.get('direct_ops_over_rho'))}",
        f"family_count={len(family_matches)}",
        f"public_product_gate={bool(row.get('public_product_gate_selected'))}",
        f"selector={selector}",
        f"selector_topk={selector}|{top_k}",
        f"support_size={len(selected)}",
        f"top_k={top_k}",
        f"topk_support={top_k}|{support_key(selected)}",
    }
    if len(family_matches) >= len(families):
        tokens.add("full_family_match")
    for family in family_matches:
        tokens.add(f"family_has={support_key(family)}")
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


def compatible_families(row: dict[str, Any], families: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    selected = set(support_tuple(row.get("selected_term_support")))
    return [family for family in families if set(family) <= selected]


def load_scout_rows(paths: list[Path], start: int | None, end: int | None) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        item_range = range_from_path(path)
        if item_range is not None:
            if start is not None and item_range[1] < start:
                continue
            if end is not None and item_range[0] > end:
                continue
        payload = load_json(path)
        for row in payload.get("case_reports") or []:
            transfer = int_value(row.get("transfer_index"))
            if start is not None and transfer < start:
                continue
            if end is not None and transfer > end:
                continue
            rows.append({**row, "artifact": str(path), "range": range_label(item_range)})
    return rows


def sorted_unique(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for row in sorted(
        rows,
        key=lambda item: (
            int_value(item.get("transfer_index")),
            str(item.get("selector")),
            int_value(item.get("top_k")),
            str(item.get("selected_term_support")),
            str(item.get("row_keys")),
        ),
    ):
        key = (
            int_value(row.get("transfer_index")),
            str(row.get("selector")),
            int_value(row.get("top_k")),
            tuple(row.get("selected_term_support") or []),
            tuple(row.get("row_keys") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def load_rules(feature_lift: dict[str, Any]) -> list[dict[str, Any]]:
    rules = []
    promoted = feature_lift.get("promoted_rule") or {}
    if promoted.get("tokens"):
        rules.append({"name": "sharp_promoted", "tokens": tuple(sorted(promoted["tokens"]))})
    robust = feature_lift.get("robust_rules") or []
    broad = next(
        (
            rule
            for rule in robust
            if set(rule.get("tokens") or []) == set(DEFAULT_BROAD_TOKENS)
        ),
        None,
    )
    if broad:
        rules.append({"name": "broad_nonadjacent", "tokens": tuple(sorted(broad["tokens"]))})
    else:
        rules.append({"name": "broad_nonadjacent", "tokens": tuple(sorted(DEFAULT_BROAD_TOKENS))})
    secondary = []
    for rule in robust:
        tokens = tuple(sorted(rule.get("tokens") or []))
        if not tokens or any(item["tokens"] == tokens for item in rules):
            continue
        if rule.get("exactish_tokens"):
            continue
        secondary.append({"name": f"secondary_{len(secondary) + 1}", "tokens": tokens})
        if len(secondary) >= 3:
            break
    return rules + secondary


def best_certificate(transfer: int, certs_by_transfer: dict[int, list[dict[str, Any]]]) -> dict[str, Any] | None:
    certs = certs_by_transfer.get(transfer, [])
    if not certs:
        return None
    return max(certs, key=lambda item: (rank_gain(item), unique_gain(item)))


def certificate_brief(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "accepted_missing_columns": sorted(int_value(item) for item in row.get("accepted_missing_columns") or []),
        "classification": row.get("classification"),
        "form_supports": [list(support_tuple(item)) for item in row.get("form_supports") or []],
        "rank_gain": rank_gain(row),
        "unique_factor_relation_gain": unique_gain(row),
    }


def row_brief(
    row: dict[str, Any],
    status: str,
    matched_rules: list[str],
    matched_rule_tokens: list[list[str]],
    families: list[tuple[int, ...]],
    certs_by_transfer: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    transfer = int_value(row.get("transfer_index"))
    return {
        "artifact": row.get("artifact"),
        "best_direct_certificate": certificate_brief(best_certificate(transfer, certs_by_transfer)),
        "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
        "direct_status": status,
        "matched_families": [list(family) for family in compatible_families(row, families)],
        "matched_rule_tokens": matched_rule_tokens,
        "matched_rules": matched_rules,
        "priority_hits": sorted(int_value(item) for item in row.get("priority_hits") or []),
        "public_product_gate_selected": bool(row.get("public_product_gate_selected")),
        "range": row.get("range"),
        "row_keys": row.get("row_keys") or [],
        "salts": salts_from_row_keys(row.get("row_keys") or []),
        "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
        "selector": row.get("selector"),
        "shared_product_ops_over_rho": float_value(row.get("shared_product_ops_over_rho")),
        "target": row.get("target"),
        "top_k": int_value(row.get("top_k")),
        "transfer_index": transfer,
    }


def priority_score(item: dict[str, Any]) -> tuple[int, int, int, float, int]:
    status = item.get("direct_status")
    status_score = 3 if status == "direct_certificate_missing" else 2 if status == "support_report_missing" else 0
    rule_score = 4 if "sharp_promoted" in item.get("matched_rules", []) else 2
    family_score = len(item.get("matched_families") or [])
    cost = float_value(item.get("direct_ops_over_rho"))
    cost_score = -cost if cost is not None else -9.0
    return (status_score, rule_score, family_score, cost_score, -int_value(item.get("transfer_index")))


def summarize(rows: list[dict[str, Any]], rules: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(str(row.get("direct_status")) for row in rows)
    by_rule: dict[str, dict[str, Any]] = {}
    for rule in rules:
        name = rule["name"]
        matched = [row for row in rows if name in row.get("matched_rules", [])]
        needing = [row for row in matched if row.get("direct_status") in {"direct_certificate_missing", "support_report_missing"}]
        full_needing = [row for row in needing if len(row.get("matched_families") or []) >= 3]
        by_rule[name] = {
            "full_family_needing_validation_count": len(full_needing),
            "full_family_needing_validation_transfers": sorted({int_value(row.get("transfer_index")) for row in full_needing}),
            "needing_validation_count": len(needing),
            "needing_validation_status_counts": dict(sorted(Counter(str(row.get("direct_status")) for row in needing).items())),
            "needing_validation_transfers": sorted({int_value(row.get("transfer_index")) for row in needing}),
            "row_count": len(matched),
            "tokens": list(rule["tokens"]),
        }
    return {
        "direct_status_counts": dict(sorted(by_status.items())),
        "rules": by_rule,
        "total_needing_validation_count": sum(
            1 for row in rows if row.get("direct_status") in {"direct_certificate_missing", "support_report_missing"}
        ),
        "unique_transfer_count": len({int_value(row.get("transfer_index")) for row in rows}),
    }


def build_transfer_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_transfer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_transfer[int_value(row.get("transfer_index"))].append(row)
    queue = []
    for transfer, transfer_rows in by_transfer.items():
        ranked = sorted(transfer_rows, key=priority_score, reverse=True)
        best = ranked[0]
        queue.append(
            {
                "best_row": best,
                "direct_statuses": sorted({str(row.get("direct_status")) for row in transfer_rows}),
                "matched_rules": sorted({rule for row in transfer_rows for rule in row.get("matched_rules", [])}),
                "row_variant_count": len(transfer_rows),
                "transfer_index": transfer,
            }
        )
    queue.sort(key=lambda item: priority_score(item["best_row"]), reverse=True)
    return queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-scouts", required=True)
    parser.add_argument("--bridge-audit", required=True, type=Path)
    parser.add_argument("--feature-lift", required=True, type=Path)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    families = parse_families(args.families)
    paths = parse_paths(args.support_scouts)
    bridge = load_json(args.bridge_audit)
    feature_lift = load_json(args.feature_lift)
    rules = load_rules(feature_lift)
    support_index = bridge_support_index(bridge)
    certs_by_transfer = certificate_index(bridge)
    raw_rows = load_scout_rows(paths, args.start, args.end)
    available_scout_ranges = [range_from_path(path) for path in paths]
    available_scout_ranges = [item for item in available_scout_ranges if item is not None]
    scanned_scout_ranges = [
        item
        for item in available_scout_ranges
        if not (args.start is not None and item[1] < args.start)
        and not (args.end is not None and item[0] > args.end)
    ]

    materialized = []
    for row in sorted_unique(raw_rows):
        tokens = public_tokens(row, families)
        matched = [rule for rule in rules if set(rule["tokens"]) <= tokens]
        if not matched:
            continue
        status = direct_status(row, support_index)
        matched_names = [rule["name"] for rule in matched]
        matched_tokens = [list(rule["tokens"]) for rule in matched]
        materialized.append(row_brief(row, status, matched_names, matched_tokens, families, certs_by_transfer))

    needing_validation = [
        row
        for row in materialized
        if row.get("direct_status") in {"direct_certificate_missing", "support_report_missing"}
    ]
    needing_validation.sort(key=priority_score, reverse=True)
    transfer_queue = build_transfer_queue(needing_validation)
    payload = {
        "artifacts": {
            "bridge_audit": str(args.bridge_audit),
            "feature_lift": str(args.feature_lift),
            "support_scouts": args.support_scouts,
        },
        "claim_status": (
            "PUBLIC_LANE_VALIDATION_MANIFEST_HAS_DIRECT_RANK_BACKFILL_TARGETS"
            if needing_validation
            else "PUBLIC_LANE_VALIDATION_MANIFEST_HAS_NO_BACKFILL_TARGETS"
        ),
        "created_at": now_iso(),
        "direct_rank_export_request": needing_validation[: max(1, args.limit)],
        "direct_rank_transfer_queue": transfer_queue[: max(1, args.limit)],
        "honesty_boundary": [
            "This manifest materializes validation targets only.",
            "support_report_missing means the support scout exists beyond the available bridge/direct-rank audit.",
            "direct_certificate_missing means the bridge audit saw the row but no direct certificate was exported.",
            "No new rank-gain or ECDLP speedup is claimed until direct/rank export validates these rows.",
        ],
        "parameters": {
            "end": args.end,
            "families": [list(family) for family in families],
            "limit": args.limit,
            "start": args.start,
        },
        "rules": [{"name": rule["name"], "tokens": list(rule["tokens"])} for rule in rules],
        "schema": SCHEMA,
        "summary": {
            **summarize(materialized, rules),
            "latest_available_scout_end": max((item[1] for item in available_scout_ranges), default=None),
            "latest_bridge_end": range_end_from_artifact(args.bridge_audit, int_value((bridge.get("parameters") or {}).get("end"))),
            "latest_scanned_scout_end": max((item[1] for item in scanned_scout_ranges), default=None),
            "materialized_row_count": len(materialized),
            "raw_scout_row_count": len(raw_rows),
            "selected_export_request_count": min(len(needing_validation), max(1, args.limit)),
            "selected_transfer_queue_count": min(len(transfer_queue), max(1, args.limit)),
            "support_scout_file_count": len(paths),
        },
        "validation_rows": materialized,
    }
    write_json(args.out, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
