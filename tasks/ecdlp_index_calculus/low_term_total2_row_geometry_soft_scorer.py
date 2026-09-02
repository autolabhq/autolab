#!/usr/bin/env python3
"""Score accepted-form bridge candidates with public row-key geometry.

The strict accepted-form miner only promotes tokens with zero nonpositive
calibration examples.  That is useful but brittle.  This scorer keeps the same
honesty boundary, but ranks scout-only rows with smoothed public token evidence
so the next direct replay/export work order can be chosen deliberately.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_row_geometry_soft_scorer.v1"


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
    unique = []
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
    return ",".join(str(item) for item in sorted(int_value(item) for item in (raw or [])))


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


def label_for_cert(cert: dict[str, Any]) -> str:
    if cert.get("classification") == "ACCEPTED_MISSING_COLUMN_RANK_GAIN":
        return "positive_rank_gain"
    if cert.get("classification") == "ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN":
        return "accepted_missing_dependent"
    if cert.get("classification") == "SELECTED_MISSING_COLLAPSED_TO_SATURATED":
        return "saturated_collapse"
    return "other_nonpositive"


def public_tokens(row: dict[str, Any]) -> list[str]:
    selected_support = sorted(int_value(item) for item in (row.get("selected_term_support") or []))
    selected_set = set(selected_support)
    selector = str(row.get("selector"))
    top_k = int_value(row.get("top_k"))
    support = support_key(selected_support)
    salts = salts_from_row_keys(row.get("row_keys") or [])
    tokens = [
        f"selector={selector}",
        f"top_k={top_k}",
        f"selector_topk={selector}|{top_k}",
        f"selected_support={support}",
        f"topk_support={top_k}|{support}",
        f"selector_topk_support={selector}|{top_k}|{support}",
        f"support_size={len(selected_support)}",
        f"cost_bucket={cost_bucket(row.get('direct_ops_over_rho'))}",
        f"public_product_gate={bool(row.get('public_product_gate_selected'))}",
        f"priority_hit_15={15 in set(int_value(item) for item in (row.get('priority_hits') or []))}",
    ]
    for column in selected_support:
        tokens.append(f"selected_has={column}")
    for left in selected_support:
        for right in selected_support:
            if left < right and (left in {0, 10, 13, 15} or right in {0, 10, 13, 15}):
                tokens.append(f"selected_pair={left}:{right}")
    if 0 in selected_set and 15 in selected_set:
        tokens.append("selected_has_0_and_15")
    if 10 in selected_set and 13 in selected_set:
        tokens.append("selected_has_10_and_13")
    if 11 in selected_set and 15 in selected_set:
        tokens.append("selected_has_11_and_15")
    if 13 in selected_set:
        tokens.append("selected_has_13")
    if salts:
        gap = salts[-1] - salts[0] if len(salts) >= 2 else 0
        tokens.extend(
            [
                f"salt_count={len(salts)}",
                f"salt_gap={gap}",
                f"salt_gap_bucket={salt_gap_bucket(gap)}",
                f"salt_min_mod4={salts[0] % 4}",
                f"salt_max_mod4={salts[-1] % 4}",
                f"salt_sum_mod4={sum(salts) % 4}",
                f"salt_pair_mod4={','.join(str(salt % 4) for salt in salts)}",
                f"salt_min_bucket={(salts[0] // 4) * 4}",
                f"salt_max_bucket={(salts[-1] // 4) * 4}",
            ]
        )
        if len(salts) >= 2:
            tokens.append(f"salt_adjacent={gap == 1}")
    return tokens


def build_labeled_rows(bridge_audit: Path, start: int, end: int | None) -> list[dict[str, Any]]:
    payload = load_json(bridge_audit)
    rows = []
    for cert in payload.get("certificates") or []:
        if not in_range(cert, start, end):
            continue
        rows.append({**cert, "_label": label_for_cert(cert)})
    return rows


def train_weights(rows: list[dict[str, Any]], alpha: float) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    total_pos = sum(1 for row in rows if row["_label"] == "positive_rank_gain")
    total_neg = len(rows) - total_pos
    counts: dict[str, dict[str, Any]] = defaultdict(lambda: {"neg": 0, "pos": 0})
    for row in rows:
        seen = set(public_tokens(row))
        for token in seen:
            if row["_label"] == "positive_rank_gain":
                counts[token]["pos"] += 1
            else:
                counts[token]["neg"] += 1
    weights: dict[str, dict[str, Any]] = {}
    for token, item in counts.items():
        pos = int_value(item["pos"])
        neg = int_value(item["neg"])
        p_token_pos = (pos + alpha) / (total_pos + 2 * alpha) if total_pos else alpha
        p_token_neg = (neg + alpha) / (total_neg + 2 * alpha) if total_neg else alpha
        weight = math.log2(p_token_pos / p_token_neg)
        weights[token] = {
            "negative_count": neg,
            "positive_count": pos,
            "precision": round(pos / (pos + neg), 8) if pos + neg else None,
            "recall": round(pos / total_pos, 8) if total_pos else None,
            "token": token,
            "weight": round(weight, 8),
        }
    meta = {
        "alpha": alpha,
        "base_positive_rate": round(total_pos / len(rows), 8) if rows else None,
        "negative_count": total_neg,
        "positive_count": total_pos,
        "row_count": len(rows),
    }
    return weights, meta


def score_row(row: dict[str, Any], weights: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tokens = sorted(set(public_tokens(row)))
    evidence = [weights[token] for token in tokens if token in weights]
    score = sum(float_value(item.get("weight")) or 0.0 for item in evidence)
    positive_evidence = sorted(evidence, key=lambda item: (-(float_value(item.get("weight")) or 0.0), item["token"]))[:12]
    negative_evidence = sorted(evidence, key=lambda item: ((float_value(item.get("weight")) or 0.0), item["token"]))[:8]
    return {
        "evidence_token_count": len(evidence),
        "matched_positive_evidence": positive_evidence,
        "matched_negative_evidence": negative_evidence,
        "score": round(score, 8),
    }


def score_rows(rows: list[dict[str, Any]], weights: dict[str, dict[str, Any]], include_label: bool) -> list[dict[str, Any]]:
    scored = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        row_identity = identity(row)
        if row_identity in seen:
            continue
        seen.add(row_identity)
        score = score_row(row, weights)
        item = {
            **score,
            "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
            "direct_public_key_verified_posthoc": bool(row.get("direct_public_key_verified")),
            "range": row.get("_range") or row.get("range"),
            "row_keys": row.get("row_keys") or [],
            "selected_term_support": sorted(int_value(item) for item in (row.get("selected_term_support") or [])),
            "selector": row.get("selector"),
            "shared_product_public_key_verified_posthoc": bool(row.get("shared_product_public_key_verified")),
            "top_k": int_value(row.get("top_k")),
            "transfer_index": int_value(row.get("transfer_index")),
        }
        if include_label:
            rank_score = row.get("rank_score") or {}
            item.update(
                {
                    "accepted_missing_columns": row.get("accepted_missing_columns") or [],
                    "classification": row.get("classification"),
                    "form_supports": row.get("form_supports") or [],
                    "label": row.get("_label"),
                    "rank_gain": rank_score.get("rank_gain"),
                    "unique_factor_relation_gain": rank_score.get("unique_factor_relation_gain"),
                }
            )
        scored.append(item)
    scored.sort(
        key=lambda item: (
            -(float_value(item.get("score")) or 0.0),
            float_value(item.get("direct_ops_over_rho")) or 999.0,
            int_value(item.get("transfer_index")),
            int_value(item.get("top_k")),
            row_key_key(item.get("row_keys")),
        )
    )
    return scored


def summarize_scored(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    top_rows = rows[:top_k]
    return {
        "candidate_count": len(rows),
        "direct_verified_posthoc_count": sum(1 for row in rows if row.get("direct_public_key_verified_posthoc")),
        "ranges": dict(Counter(str(row.get("range")) for row in rows)),
        "shared_verified_posthoc_count": sum(1 for row in rows if row.get("shared_product_public_key_verified_posthoc")),
        "top_score": top_rows[0].get("score") if top_rows else None,
        "top_transfers": sorted({int_value(row.get("transfer_index")) for row in top_rows}),
        "transfers": sorted({int_value(row.get("transfer_index")) for row in rows}),
    }


def validation_summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    top_rows = rows[:top_k]
    return {
        "accepted_missing_rank_gain_in_top_k": sum(
            1 for row in top_rows if row.get("classification") == "ACCEPTED_MISSING_COLUMN_RANK_GAIN"
        ),
        "classification_counts": dict(Counter(str(row.get("classification")) for row in rows)),
        "positive_rank_gain_count": sum(1 for row in rows if row.get("label") == "positive_rank_gain"),
        "rank_gain_in_top_k": sum(1 for row in top_rows if int_value(row.get("rank_gain")) > 0),
        "row_count": len(rows),
        "top_k": top_k,
        "top_labels": dict(Counter(str(row.get("label")) for row in top_rows)),
        "top_transfers": sorted({int_value(row.get("transfer_index")) for row in top_rows}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-audit", required=True, type=Path)
    parser.add_argument("--support-scouts", required=True)
    parser.add_argument("--train-start", type=int, default=5984)
    parser.add_argument("--train-end", type=int, default=6087)
    parser.add_argument("--prefix-end", type=int, default=6023)
    parser.add_argument("--validation-start", type=int, default=6024)
    parser.add_argument("--validation-end", type=int, default=6087)
    parser.add_argument("--holdout-start", type=int, default=6088)
    parser.add_argument("--holdout-end", type=int, default=6095)
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument("--top-k-summary", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    support_paths = parse_paths(args.support_scouts)
    train_rows = build_labeled_rows(args.bridge_audit, args.train_start, args.train_end)
    prefix_rows = build_labeled_rows(args.bridge_audit, args.train_start, args.prefix_end)
    validation_rows = build_labeled_rows(args.bridge_audit, args.validation_start, args.validation_end)
    holdout_reports = load_support_reports(support_paths, args.holdout_start, args.holdout_end)

    full_weights, full_meta = train_weights(train_rows, args.alpha)
    prefix_weights, prefix_meta = train_weights(prefix_rows, args.alpha)
    scored_holdout = score_rows(holdout_reports, full_weights, include_label=False)[: args.candidate_limit]
    scored_validation = score_rows(validation_rows, prefix_weights, include_label=True)

    top_positive_tokens = sorted(
        full_weights.values(),
        key=lambda item: (-(float_value(item.get("weight")) or 0.0), -int_value(item.get("positive_count")), item["token"]),
    )[:40]
    payload = {
        "artifacts": {
            "bridge_audit": str(args.bridge_audit),
            "support_scouts": [str(path) for path in support_paths],
        },
        "claim_status": (
            "SOFT_ROW_GEOMETRY_SCORER_HAS_FORWARD_CANDIDATES"
            if scored_holdout
            else "SOFT_ROW_GEOMETRY_SCORER_HAS_NO_FORWARD_CANDIDATES"
        ),
        "created_at": now_iso(),
        "holdout_candidates": scored_holdout,
        "honesty_boundary": [
            "TOY-EVIDENCE: controlled toy-prime ECDLP harness only.",
            "Soft scores are work-order rankings, not zero-false-positive selector rules.",
            "Posthoc verification fields are reported but are not used as features.",
            "Scout-only rows require direct certificate export and rank audit before progress is counted.",
            "This is not target descent or a deployed-curve speedup claim.",
        ],
        "parameters": {
            "alpha": args.alpha,
            "candidate_limit": args.candidate_limit,
            "holdout_end": args.holdout_end,
            "holdout_start": args.holdout_start,
            "prefix_end": args.prefix_end,
            "top_k_summary": args.top_k_summary,
            "train_end": args.train_end,
            "train_start": args.train_start,
            "validation_end": args.validation_end,
            "validation_start": args.validation_start,
        },
        "schema": SCHEMA,
        "summary": {
            "full_training": full_meta,
            "holdout": summarize_scored(scored_holdout, args.top_k_summary),
            "prefix_training": prefix_meta,
            "top_positive_tokens": top_positive_tokens,
            "validation_with_prefix_model": validation_summary(scored_validation, args.top_k_summary),
        },
        "validation_rows_scored_by_prefix_model": scored_validation[: args.candidate_limit],
    }
    write_json(args.out, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(json.dumps({"claim_status": payload["claim_status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
