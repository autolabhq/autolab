#!/usr/bin/env python3
"""Build a public-token replay plan for repeated accepted-form families.

The bridge and contrast artifacts show repeated below-rho accepted-form rank
gain, but they do not yet turn that signal into a concrete next-row queue.  This
planner keeps the split explicit:

* direct bridge labels are used to identify seed families and evaluate carriers,
* future candidate scoring uses public row tokens, selected support, salts, and
  frozen/soft selector scores,
* unmatched candidates are work orders, not progress claims.
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


SCHEMA = "ecdlp.low_term_total2_family_replay_planner.v1"


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


def parse_families(raw: str) -> list[tuple[int, ...]]:
    families = []
    for chunk in raw.split(";"):
        parts = [part.strip() for part in chunk.split(",") if part.strip()]
        if parts:
            families.append(tuple(sorted(int(part) for part in parts)))
    return families


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


def direct_unique_gain(row: dict[str, Any]) -> int:
    return int_value(row.get("direct_audit_unique_factor_relation_gain"))


def has_form(row: dict[str, Any], family: tuple[int, ...]) -> bool:
    return any(support_tuple(item) == family for item in (row.get("form_supports") or []))


def family_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("selector")),
            str(int_value(row.get("top_k"))),
            support_key(row.get("selected_term_support")),
        ]
    )


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


def public_tokens(row: dict[str, Any]) -> set[str]:
    selected_support = support_tuple(row.get("selected_term_support"))
    selected_set = set(selected_support)
    selector = str(row.get("selector"))
    top_k = int_value(row.get("top_k"))
    support = ",".join(str(item) for item in selected_support)
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
    watched = {0, 1, 2, 4, 5, 8, 9, 10, 11, 13, 14, 15}
    for left in selected_support:
        for right in selected_support:
            if left < right and (left in watched or right in watched):
                tokens.add(f"selected_pair={left}:{right}")
    for left, right in [(8, 11), (9, 11), (10, 13), (10, 14), (11, 13), (11, 15)]:
        if left in selected_set and right in selected_set:
            tokens.add(f"selected_has_{left}_and_{right}")
    if salts:
        gap = salts[-1] - salts[0] if len(salts) >= 2 else 0
        tokens.update(
            {
                f"salt_count={len(salts)}",
                f"salt_gap={gap}",
                f"salt_gap_bucket={salt_gap_bucket(gap)}",
                f"salt_max_bucket={(salts[-1] // 4) * 4}",
                f"salt_max_mod4={salts[-1] % 4}",
                f"salt_min_bucket={(salts[0] // 4) * 4}",
                f"salt_min_mod4={salts[0] % 4}",
                f"salt_pair={','.join(str(salt) for salt in salts)}",
                f"salt_pair_mod4={','.join(str(salt % 4) for salt in salts)}",
                f"salt_sum_mod4={sum(salts) % 4}",
            }
        )
    return tokens


def row_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_missing_columns": sorted(int_value(item) for item in (row.get("accepted_missing_columns") or [])),
        "accepted_priority_columns": sorted(int_value(item) for item in (row.get("accepted_priority_columns") or [])),
        "classification": row.get("classification"),
        "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
        "form_supports": [list(support_tuple(item)) for item in (row.get("form_supports") or [])],
        "rank_gain": rank_gain(row),
        "row_keys": row.get("row_keys") or [],
        "salts": salts_from_row_keys(row.get("row_keys") or []),
        "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
        "selector": row.get("selector"),
        "top_k": int_value(row.get("top_k")),
        "transfer_index": int_value(row.get("transfer_index")),
        "unique_factor_relation_gain": unique_gain(row),
    }


def family_summary(
    family: tuple[int, ...],
    positive_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    seed_rows = [row for row in positive_rows if has_form(row, family)]
    all_family_rank_gain = [row for row in all_rows if has_form(row, family) and rank_gain(row) > 0]
    token_counts = Counter()
    selector_carriers = Counter()
    salt_pairs = Counter()
    for row in seed_rows:
        token_counts.update(public_tokens(row))
        selector_carriers[family_key(row)] += 1
        salt_pairs[",".join(str(item) for item in salts_from_row_keys(row.get("row_keys") or []))] += 1
    return {
        "accepted_missing_rank_gain_count": len(seed_rows),
        "family": list(family),
        "public_signature_tokens": [
            {"count": count, "token": token}
            for token, count in token_counts.most_common(20)
        ],
        "rank_gain_total": sum(rank_gain(row) for row in all_family_rank_gain),
        "seed_rows": [row_summary(row) for row in seed_rows],
        "selector_carriers": [
            {"count": count, "family_key": key}
            for key, count in selector_carriers.most_common()
        ],
        "salt_pairs": [
            {"count": count, "salts": [int(item) for item in key.split(",") if item]}
            for key, count in salt_pairs.most_common()
        ],
        "transfers": sorted({int_value(row.get("transfer_index")) for row in seed_rows}),
        "unique_factor_relation_gain_total": sum(unique_gain(row) for row in all_family_rank_gain),
    }


def carrier_summaries(candidates: list[dict[str, Any]], families: list[tuple[int, ...]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        buckets[family_key(row)].append(row)
    summaries = []
    for key, rows in buckets.items():
        first = rows[0]
        matched_rows = [row for row in rows if row.get("direct_audit_matched")]
        rank_gain_rows = [row for row in rows if direct_rank_gain(row) > 0]
        accepted_missing_rows = [
            row
            for row in rows
            if row.get("direct_audit_classification") == "ACCEPTED_MISSING_COLUMN_RANK_GAIN"
        ]
        target_counts = {}
        for family in families:
            count = sum(1 for row in accepted_missing_rows if has_form(row, family))
            if count:
                target_counts[support_key(family)] = count
        direct_matches = len(matched_rows)
        summaries.append(
            {
                "accepted_missing_rank_gain_count": len(accepted_missing_rows),
                "candidate_count": len(rows),
                "direct_match_count": direct_matches,
                "direct_rank_gain_count": len(rank_gain_rows),
                "family_key": key,
                "precision_among_direct_matches": round(len(accepted_missing_rows) / direct_matches, 8)
                if direct_matches
                else None,
                "selected_term_support": list(support_tuple(first.get("selected_term_support"))),
                "selector": first.get("selector"),
                "target_family_accepts": target_counts,
                "top_k": int_value(first.get("top_k")),
                "transfers": sorted({int_value(row.get("transfer_index")) for row in rows}),
                "unmatched_count": sum(1 for row in rows if not row.get("direct_audit_matched")),
            }
        )
    summaries.sort(
        key=lambda item: (
            -int_value(item.get("accepted_missing_rank_gain_count")),
            -int_value(item.get("direct_rank_gain_count")),
            -int_value(item.get("direct_match_count")),
            -int_value(item.get("candidate_count")),
            str(item.get("family_key")),
        )
    )
    return summaries


def token_weights(
    positive_rows: list[dict[str, Any]],
    negative_rows: list[dict[str, Any]],
) -> dict[str, float]:
    positive_counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    for row in positive_rows:
        positive_counts.update(public_tokens(row))
    for row in negative_rows:
        negative_counts.update(public_tokens(row))
    weights = {}
    total_positive = len(positive_rows)
    for token in set(positive_counts) | set(negative_counts):
        positives = positive_counts[token]
        negatives = negative_counts[token]
        if positives <= 0:
            continue
        precision = positives / (positives + negatives)
        recall = positives / total_positive if total_positive else 0.0
        score = (precision * recall) + math.log2((positives + 0.5) / (negatives + 0.5))
        if score > 0.0:
            weights[token] = score
    return weights


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
            "matched_rule_tokens": [],
            "promoted_score": None,
            "range": row.get("range"),
            "row_keys": row.get("row_keys") or [],
            "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
            "selector": row.get("selector"),
            "soft_positive_tokens": [],
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
        item["direct_audit_classification"] = cert.get("classification")
        item["direct_audit_matched"] = True
        item["direct_audit_rank_gain"] = rank_gain(cert)
        item["direct_audit_unique_factor_relation_gain"] = unique_gain(cert)
        item["form_supports"] = cert.get("form_supports") or []
    else:
        item.setdefault("direct_audit_classification", row.get("direct_audit_classification", "NO_DIRECT_AUDIT_MATCH"))
        item.setdefault("direct_audit_matched", bool(row.get("direct_audit_matched", False)))
        item.setdefault("direct_audit_rank_gain", row.get("direct_audit_rank_gain"))
        item.setdefault("direct_audit_unique_factor_relation_gain", row.get("direct_audit_unique_factor_relation_gain"))
        item.setdefault("form_supports", row.get("form_supports") or [])
    if source == "promoted":
        item["promoted_score"] = max(int_value(item.get("promoted_score")), int_value(row.get("score")))
        item["matched_rule_tokens"] = sorted(set(item["matched_rule_tokens"]) | set(row.get("matched_rule_tokens") or []))
    if source == "soft":
        current = float_value(item.get("soft_score")) or 0.0
        item["soft_score"] = max(current, float_value(row.get("score")) or 0.0)
        soft_tokens = [
            str(token.get("token"))
            for token in (row.get("matched_positive_evidence") or [])[:12]
            if token.get("token")
        ]
        item["soft_positive_tokens"] = sorted(set(item["soft_positive_tokens"]) | set(soft_tokens))


def build_candidate_pool(
    promoted: dict[str, Any],
    soft: dict[str, Any] | None,
    bridge_rows: list[dict[str, Any]],
    open_start: int | None,
    open_end: int | None,
) -> list[dict[str, Any]]:
    cert_by_identity = {identity(row): row for row in bridge_rows}
    bucket: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in promoted.get("audited_candidates") or []:
        if in_range(row, open_start, open_end):
            merge_candidate(bucket, row, "promoted", cert_by_identity)
    if soft:
        for row in soft.get("holdout_candidates") or []:
            if in_range(row, open_start, open_end):
                merge_candidate(bucket, row, "soft", cert_by_identity)
    return list(bucket.values())


def score_candidate(
    row: dict[str, Any],
    families: list[tuple[int, ...]],
    weights_by_family: dict[tuple[int, ...], dict[str, float]],
    positive_salt_pairs_by_family: dict[tuple[int, ...], set[tuple[int, ...]]],
    carrier_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected = set(support_tuple(row.get("selected_term_support")))
    tokens = public_tokens(row)
    reasons = set()
    matched_target_families = []
    score = 0.0
    for family in families:
        if set(family).issubset(selected):
            matched_target_families.append(list(family))
            reasons.add(f"selected_contains_family={support_key(family)}")
        weights = weights_by_family.get(family, {})
        matched_tokens = [token for token in tokens if token in weights]
        if matched_tokens:
            score += sum(weights[token] for token in matched_tokens)
            reasons.add(f"public_tokens_match_family={support_key(family)}")
        salts = tuple(salts_from_row_keys(row.get("row_keys") or []))
        if salts and salts in positive_salt_pairs_by_family.get(family, set()):
            score += 8.0
            reasons.add(f"shares_positive_salt_pair={support_key(family)}")
    carrier = carrier_by_key.get(family_key(row))
    if carrier and int_value(carrier.get("accepted_missing_rank_gain_count")) > 0:
        count = int_value(carrier.get("accepted_missing_rank_gain_count"))
        score += min(20.0, count * 3.0)
        reasons.add("carrier_has_accepted_missing_rank_gain")
    if "promoted" in (row.get("sources") or []):
        score += min(8.0, int_value(row.get("promoted_score")) / 3.0)
        reasons.add("frozen_public_selector_promoted")
    if "soft" in (row.get("sources") or []):
        score += min(12.0, (float_value(row.get("soft_score")) or 0.0) / 4.0)
        reasons.add("soft_geometry_candidate")
    direct_cost = float_value(row.get("direct_ops_over_rho"))
    if direct_cost is not None and direct_cost < 1.0:
        score += 3.0
        reasons.add("direct_cost_below_rho")
    return {
        "direct_ops_over_rho": direct_cost,
        "matched_rule_tokens": row.get("matched_rule_tokens") or [],
        "matched_target_families": matched_target_families,
        "planner_score": round(score, 8),
        "promoted_score": row.get("promoted_score"),
        "range": row.get("range"),
        "reason_tags": sorted(reasons),
        "row_keys": row.get("row_keys") or [],
        "salts": salts_from_row_keys(row.get("row_keys") or []),
        "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
        "selector": row.get("selector"),
        "soft_positive_tokens": row.get("soft_positive_tokens") or [],
        "soft_score": row.get("soft_score"),
        "sources": sorted(row.get("sources") or []),
        "top_k": int_value(row.get("top_k")),
        "transfer_index": int_value(row.get("transfer_index")),
    }


def family_replay_queues(
    open_rows: list[dict[str, Any]],
    families: list[tuple[int, ...]],
    weights_by_family: dict[tuple[int, ...], dict[str, float]],
    positive_salt_pairs_by_family: dict[tuple[int, ...], set[tuple[int, ...]]],
    carrier_by_key: dict[str, dict[str, Any]],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    queues: dict[str, list[dict[str, Any]]] = {}
    for family in families:
        scored = [
            score_candidate(row, [family], weights_by_family, positive_salt_pairs_by_family, carrier_by_key)
            for row in open_rows
        ]
        scored = [row for row in scored if row["planner_score"] > 0.0]
        scored.sort(
            key=lambda item: (
                -float(item["planner_score"]),
                float(item["direct_ops_over_rho"]) if item["direct_ops_over_rho"] is not None else 999.0,
                int(item["transfer_index"]),
                str(item["selector"]),
            )
        )
        queues[support_key(family)] = scored[:limit]
    return queues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-audit", required=True, type=Path)
    parser.add_argument("--promoted-audit", required=True, type=Path)
    parser.add_argument("--soft-scorer", type=Path)
    parser.add_argument("--families", default="10,13;11,15;10,14;9,11;8,11")
    parser.add_argument("--audit-start", type=int)
    parser.add_argument("--audit-end", type=int)
    parser.add_argument("--open-start", type=int)
    parser.add_argument("--open-end", type=int)
    parser.add_argument("--open-candidate-limit", type=int, default=24)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    families = parse_families(args.families)
    bridge = load_json(args.bridge_audit)
    promoted = load_json(args.promoted_audit)
    soft = load_json(args.soft_scorer) if args.soft_scorer else None
    certificates = [
        row
        for row in bridge.get("certificates") or []
        if in_range(row, args.audit_start, args.audit_end)
    ]
    accepted_missing_rank_gain = [
        row
        for row in certificates
        if row.get("classification") == "ACCEPTED_MISSING_COLUMN_RANK_GAIN" and rank_gain(row) > 0
    ]
    no_rank_controls = [row for row in certificates if rank_gain(row) <= 0]
    family_summaries = [
        family_summary(family, accepted_missing_rank_gain, certificates)
        for family in families
    ]
    weights_by_family = {
        family: token_weights([row for row in accepted_missing_rank_gain if has_form(row, family)], no_rank_controls)
        for family in families
    }
    positive_salt_pairs_by_family = {
        family: {
            tuple(salts_from_row_keys(row.get("row_keys") or []))
            for row in accepted_missing_rank_gain
            if has_form(row, family)
        }
        for family in families
    }
    carriers = carrier_summaries(promoted.get("audited_candidates") or [], families)
    carrier_by_key = {str(item["family_key"]): item for item in carriers}
    candidate_pool = build_candidate_pool(
        promoted,
        soft,
        certificates,
        args.open_start,
        args.open_end,
    )
    open_rows = [
        row
        for row in candidate_pool
        if not row.get("direct_audit_matched")
    ]
    scored = [
        score_candidate(row, families, weights_by_family, positive_salt_pairs_by_family, carrier_by_key)
        for row in open_rows
    ]
    scored = [row for row in scored if row["planner_score"] > 0.0]
    scored.sort(
        key=lambda item: (
            -float(item["planner_score"]),
            float(item["direct_ops_over_rho"]) if item["direct_ops_over_rho"] is not None else 999.0,
            int(item["transfer_index"]),
            str(item["selector"]),
        )
    )
    per_family_queues = family_replay_queues(
        open_rows,
        families,
        weights_by_family,
        positive_salt_pairs_by_family,
        carrier_by_key,
        args.open_candidate_limit,
    )
    payload = {
        "artifacts": {
            "bridge_audit": str(args.bridge_audit),
            "promoted_audit": str(args.promoted_audit),
            "soft_scorer": str(args.soft_scorer) if args.soft_scorer else None,
        },
        "claim_status": (
            "FAMILY_REPLAY_PLANNER_HAS_REPEATED_TARGET_FAMILIES_AND_OPEN_QUEUE"
            if any(item["accepted_missing_rank_gain_count"] >= 2 for item in family_summaries) and scored
            else "FAMILY_REPLAY_PLANNER_HAS_SEED_FAMILIES_ONLY"
            if any(item["accepted_missing_rank_gain_count"] for item in family_summaries)
            else "FAMILY_REPLAY_PLANNER_HAS_NO_TARGET_SEEDS"
        ),
        "created_at": now_iso(),
        "family_summaries": family_summaries,
        "honesty_boundary": [
            "TOY-EVIDENCE: controlled toy-prime ECDLP harness only.",
            "Direct bridge labels seed and evaluate families; open-candidate ordering uses public row tokens and selector scores.",
            "An open candidate is unmatched by exact row identity; another row at the same transfer may already have a direct certificate.",
            "Open queue entries are work orders until direct certificate export and rank audit match them.",
            "No target descent, deployed-curve break, large-field/FFE result, or Pollard-rho speedup is claimed.",
        ],
        "next_work_orders": [
            "Direct-certificate export the top open_replay_queue rows before treating them as progress.",
            "Prioritize top-k16 broad-support rows that share [10,13] tokens and high-salt public signatures.",
            "Split the new [11,15] priority-column repeat from the broader [10,13] row-family path.",
        ],
        "open_replay_queue": scored[: args.open_candidate_limit],
        "parameters": {
            "audit_end": args.audit_end,
            "audit_start": args.audit_start,
            "families": [list(family) for family in families],
            "open_candidate_limit": args.open_candidate_limit,
            "open_end": args.open_end,
            "open_start": args.open_start,
        },
        "per_family_replay_queues": per_family_queues,
        "public_carrier_summaries": carriers,
        "schema": SCHEMA,
        "summary": {
            "accepted_missing_rank_gain_count": len(accepted_missing_rank_gain),
            "certificate_count": len(certificates),
            "open_candidate_count": len(open_rows),
            "open_queue_count": min(len(scored), args.open_candidate_limit),
            "per_family_top_transfers": {
                key: [row["transfer_index"] for row in rows[: min(8, len(rows))]]
                for key, rows in per_family_queues.items()
            },
            "repeated_target_family_count": sum(
                1 for item in family_summaries if item["accepted_missing_rank_gain_count"] >= 2
            ),
            "top_open_transfers": [row["transfer_index"] for row in scored[: min(12, args.open_candidate_limit)]],
        },
    }
    write_json(args.out, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
