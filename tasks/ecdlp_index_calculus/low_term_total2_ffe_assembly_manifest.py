#!/usr/bin/env python3
"""Assemble selected13 public-lane rows into an FFE relation manifest.

This is the handoff between the low-term total2 direct/rank audits and an
FFE/summation-polynomial implementation.  It does not compute new relations.
Instead, it materializes the row-key/salt streams, selected supports, bridge
certificates, and direct-missing targets that a lower-level FFE executor should
replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


SCHEMA = "ecdlp.low_term_total2_ffe_assembly_manifest.v1"


LANE_DEFS = (
    {
        "name": "selected13_all",
        "tokens": ("selected_has=13",),
        "description": "All materialized selected13 rows in the validation manifest.",
    },
    {
        "name": "selected13_nonadjacent",
        "tokens": ("selected_has=13", "salt_adjacent=False"),
        "description": "Selected13 rows with non-adjacent public salt pairs.",
    },
    {
        "name": "selected13_sharp_minmod3",
        "tokens": ("selected_has=13", "salt_adjacent=False", "salt_min_mod4=3"),
        "description": "Promoted selected13 sublane with non-adjacent salts and min salt mod 4 equal to 3.",
    },
)


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


def row_key_tuple(raw: Any) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in (raw or [])))


def rank_gain(cert: dict[str, Any] | None) -> int:
    if cert is None:
        return 0
    if "rank_score" in cert:
        return int_value((cert.get("rank_score") or {}).get("rank_gain"))
    return int_value(cert.get("rank_gain"))


def unique_gain(cert: dict[str, Any] | None) -> int:
    if cert is None:
        return 0
    if "rank_score" in cert:
        return int_value((cert.get("rank_score") or {}).get("unique_factor_relation_gain"))
    return int_value(cert.get("unique_factor_relation_gain"))


def certificate_sort_key(cert: dict[str, Any]) -> tuple[int, int, int, float]:
    return (
        rank_gain(cert),
        unique_gain(cert),
        len(cert.get("accepted_missing_columns") or []),
        -(float_value(cert.get("direct_ops_over_rho")) or 9.0),
    )


def certificate_brief(cert: dict[str, Any] | None) -> dict[str, Any] | None:
    if cert is None:
        return None
    return {
        "accepted_missing_columns": sorted(int_value(item) for item in cert.get("accepted_missing_columns") or []),
        "accepted_missing_not_selected_columns": sorted(
            int_value(item) for item in cert.get("accepted_missing_not_selected_columns") or []
        ),
        "accepted_priority_columns": sorted(int_value(item) for item in cert.get("accepted_priority_columns") or []),
        "artifact": cert.get("artifact"),
        "classification": cert.get("classification"),
        "direct_ops_over_rho": float_value(cert.get("direct_ops_over_rho")),
        "form_supports": [list(support_tuple(item)) for item in cert.get("form_supports") or []],
        "rank": int_value(cert.get("rank")),
        "rank_gain": rank_gain(cert),
        "rank_score": cert.get("rank_score"),
        "row_keys": list(row_key_tuple(cert.get("row_keys") or [])),
        "saturated_form_supports": [list(support_tuple(item)) for item in cert.get("saturated_form_supports") or []],
        "selected_missing_columns": sorted(int_value(item) for item in cert.get("selected_missing_columns") or []),
        "selected_priority_columns": sorted(int_value(item) for item in cert.get("selected_priority_columns") or []),
        "selected_term_support": list(support_tuple(cert.get("selected_term_support"))),
        "selector": cert.get("selector"),
        "target": cert.get("target"),
        "top_k": int_value(cert.get("top_k")),
        "transfer_index": int_value(cert.get("transfer_index")),
        "unique_factor_relation_gain": unique_gain(cert),
    }


def exact_certificate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_value(row.get("transfer_index")),
        str(row.get("selector")),
        int_value(row.get("top_k")),
        support_tuple(row.get("selected_term_support")),
        row_key_tuple(row.get("row_keys") or []),
    )


def same_salts_certificate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_value(row.get("transfer_index")),
        row_key_tuple(row.get("row_keys") or []),
    )


def build_certificate_indexes(
    bridge: dict[str, Any],
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]], dict[int, dict[str, Any]]]:
    exact: dict[tuple[Any, ...], dict[str, Any]] = {}
    same_salts: dict[tuple[Any, ...], dict[str, Any]] = {}
    by_transfer: dict[int, dict[str, Any]] = {}
    for cert in bridge.get("certificates") or []:
        exact_key = exact_certificate_key(cert)
        salt_key = same_salts_certificate_key(cert)
        transfer = int_value(cert.get("transfer_index"))
        if exact_key not in exact or certificate_sort_key(cert) > certificate_sort_key(exact[exact_key]):
            exact[exact_key] = cert
        if salt_key not in same_salts or certificate_sort_key(cert) > certificate_sort_key(same_salts[salt_key]):
            same_salts[salt_key] = cert
        if transfer not in by_transfer or certificate_sort_key(cert) > certificate_sort_key(by_transfer[transfer]):
            by_transfer[transfer] = cert
    return exact, same_salts, by_transfer


def lane_tokens(row: dict[str, Any]) -> set[str]:
    selected = set(support_tuple(row.get("selected_term_support")))
    salts = [int_value(item) for item in row.get("salts") or []]
    tokens = {f"selected_has={item}" for item in selected}
    tokens.add(f"salt_count={len(salts)}")
    if salts:
        gap = max(salts) - min(salts)
        tokens.add(f"salt_adjacent={gap == 1}")
        tokens.add(f"salt_min_mod4={min(salts) % 4}")
        tokens.add(f"salt_max_mod4={max(salts) % 4}")
    for rule_tokens in row.get("matched_rule_tokens") or []:
        tokens.update(str(item) for item in rule_tokens)
    return tokens


def matched_lanes(row: dict[str, Any]) -> list[str]:
    tokens = lane_tokens(row)
    matched = []
    for lane in LANE_DEFS:
        if set(lane["tokens"]) <= tokens:
            matched.append(str(lane["name"]))
    return matched


def stable_row_id(row: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "row_keys": row_key_tuple(row.get("row_keys") or []),
            "selected_term_support": support_tuple(row.get("selected_term_support")),
            "selector": row.get("selector"),
            "top_k": int_value(row.get("top_k")),
            "transfer_index": int_value(row.get("transfer_index")),
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
    return f"t{int_value(row.get('transfer_index'))}_{digest}"


def choose_score_certificate(
    exact: dict[str, Any] | None,
    same_salts: dict[str, Any] | None,
    best_transfer: dict[str, Any] | None,
) -> tuple[str, dict[str, Any] | None]:
    if exact is not None:
        return "exact_bridge_certificate", exact
    if same_salts is not None:
        return "same_salt_bridge_certificate", same_salts
    if best_transfer is not None:
        return "best_transfer_bridge_certificate", best_transfer
    return "none", None


def assembly_row(
    row: dict[str, Any],
    exact_index: dict[tuple[Any, ...], dict[str, Any]],
    same_salts_index: dict[tuple[Any, ...], dict[str, Any]],
    transfer_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    exact = exact_index.get(exact_certificate_key(row))
    same_salts = same_salts_index.get(same_salts_certificate_key(row))
    best_transfer = transfer_index.get(int_value(row.get("transfer_index")))
    score_source, score_cert = choose_score_certificate(exact, same_salts, best_transfer)
    score_rank_gain = rank_gain(score_cert)
    score_unique_gain = unique_gain(score_cert)
    exact_rank_gain = rank_gain(exact)
    lane_names = matched_lanes(row)
    return {
        "accepted_missing_rank_gain": (
            score_rank_gain > 0
            and (score_cert or {}).get("classification") == "ACCEPTED_MISSING_COLUMN_RANK_GAIN"
        ),
        "best_transfer_bridge_certificate": certificate_brief(best_transfer),
        "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
        "direct_status": row.get("direct_status"),
        "evidence_positive": row.get("direct_status") == "direct_certificate_exported" and score_rank_gain > 0,
        "exact_bridge_certificate": certificate_brief(exact),
        "exact_rank_gain": exact_rank_gain,
        "matched_families": row.get("matched_families") or [],
        "matched_lane_names": lane_names,
        "matched_validation_rules": row.get("matched_rules") or [],
        "priority_hits": row.get("priority_hits") or [],
        "public_product_gate_selected": bool(row.get("public_product_gate_selected")),
        "range": row.get("range"),
        "row_id": stable_row_id(row),
        "row_keys": row.get("row_keys") or [],
        "same_salt_bridge_certificate": certificate_brief(same_salts),
        "salts": row.get("salts") or [],
        "score_certificate_source": score_source,
        "score_rank_gain": score_rank_gain,
        "score_unique_factor_relation_gain": score_unique_gain,
        "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
        "selector": row.get("selector"),
        "shared_product_ops_over_rho": float_value(row.get("shared_product_ops_over_rho")),
        "target": row.get("target"),
        "top_k": int_value(row.get("top_k")),
        "transfer_index": int_value(row.get("transfer_index")),
        "validation_best_direct_certificate": row.get("best_direct_certificate"),
    }


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"max": None, "median": None, "min": None}
    return {
        "max": round(max(values), 8),
        "median": round(float(median(values)), 8),
        "min": round(min(values), 8),
    }


def support_counts(rows: list[dict[str, Any]], certificate_field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        cert = row.get(certificate_field) or {}
        for support in cert.get("form_supports") or []:
            counter[support_key(support)] += 1
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def lane_priority(row: dict[str, Any]) -> tuple[int, int, int, float, int]:
    missing_score = 3 if row.get("direct_status") in {"direct_certificate_missing", "support_report_missing"} else 0
    sharp_score = 2 if "selected13_sharp_minmod3" in row.get("matched_lane_names", []) else 0
    family_score = len(row.get("matched_families") or [])
    cost = float_value(row.get("direct_ops_over_rho"))
    cost_score = -(cost if cost is not None else 9.0)
    return (missing_score, sharp_score, family_score, cost_score, -int_value(row.get("transfer_index")))


def summarize_lane(rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    positives = [row for row in rows if row.get("evidence_positive")]
    exact_positives = [row for row in rows if int_value(row.get("exact_rank_gain")) > 0]
    missing = [
        row
        for row in rows
        if row.get("direct_status") in {"direct_certificate_missing", "support_report_missing"}
    ]
    full_family_missing = [row for row in missing if len(row.get("matched_families") or []) >= 3]
    direct_costs = [float(row["direct_ops_over_rho"]) for row in rows if row.get("direct_ops_over_rho") is not None]
    best_by_transfer: dict[int, dict[str, Any]] = {}
    for row in positives:
        transfer = int_value(row.get("transfer_index"))
        old = best_by_transfer.get(transfer)
        if old is None or (
            int_value(row.get("score_rank_gain")),
            int_value(row.get("score_unique_factor_relation_gain")),
        ) > (
            int_value(old.get("score_rank_gain")),
            int_value(old.get("score_unique_factor_relation_gain")),
        ):
            best_by_transfer[transfer] = row
    return {
        "accepted_missing_rank_gain_count": sum(1 for row in positives if row.get("accepted_missing_rank_gain")),
        "direct_ops_over_rho": percentile_summary(direct_costs),
        "direct_status_counts": dict(sorted(Counter(str(row.get("direct_status")) for row in rows).items())),
        "exact_rank_gain_positive_count": len(exact_positives),
        "exact_rank_gain_total": sum(int_value(row.get("exact_rank_gain")) for row in exact_positives),
        "exact_rank_gain_transfers": sorted({int_value(row.get("transfer_index")) for row in exact_positives}),
        "form_support_counts": support_counts(positives, "best_transfer_bridge_certificate"),
        "full_family_missing_validation_count": len(full_family_missing),
        "full_family_missing_validation_transfers": sorted(
            {int_value(row.get("transfer_index")) for row in full_family_missing}
        ),
        "missing_validation_count": len(missing),
        "missing_validation_transfers": sorted({int_value(row.get("transfer_index")) for row in missing}),
        "positive_row_ids": [row["row_id"] for row in sorted(positives, key=lane_priority, reverse=True)[:limit]],
        "positive_transfer_count": len({int_value(row.get("transfer_index")) for row in positives}),
        "positive_transfers": sorted({int_value(row.get("transfer_index")) for row in positives}),
        "rank_gain_positive_count": len(positives),
        "relation_row_ids": [row["row_id"] for row in sorted(rows, key=lane_priority, reverse=True)],
        "row_count": len(rows),
        "score_certificate_source_counts": dict(
            sorted(Counter(str(row.get("score_certificate_source")) for row in rows).items())
        ),
        "score_rank_gain_total": sum(int_value(row.get("score_rank_gain")) for row in positives),
        "score_unique_factor_relation_gain_total": sum(
            int_value(row.get("score_unique_factor_relation_gain")) for row in positives
        ),
        "selected_support_counts": dict(
            sorted(Counter(support_key(row.get("selected_term_support")) for row in rows).items())
        ),
        "top_missing_row_ids": [row["row_id"] for row in sorted(missing, key=lane_priority, reverse=True)[:limit]],
        "transfer_dedup_score_rank_gain_total": sum(
            int_value(row.get("score_rank_gain")) for row in best_by_transfer.values()
        ),
        "transfer_dedup_score_unique_factor_relation_gain_total": sum(
            int_value(row.get("score_unique_factor_relation_gain")) for row in best_by_transfer.values()
        ),
        "transfer_count": len({int_value(row.get("transfer_index")) for row in rows}),
        "transfers": sorted({int_value(row.get("transfer_index")) for row in rows}),
    }


def build_lane_payloads(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    lane_payloads = []
    for lane in LANE_DEFS:
        lane_rows = [row for row in rows if lane["name"] in row.get("matched_lane_names", [])]
        lane_payloads.append(
            {
                "description": lane["description"],
                "name": lane["name"],
                "summary": summarize_lane(lane_rows, limit),
                "tokens": list(lane["tokens"]),
            }
        )
    return lane_payloads


def validation_rule_targets(validation: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rows = validation.get("validation_rows") or []
    for lane in LANE_DEFS:
        lane_name = str(lane["name"])
        tokens = set(str(token) for token in lane["tokens"])
        matched = [row for row in rows if tokens <= lane_tokens(row)]
        needing = [
            row
            for row in matched
            if row.get("direct_status") in {"direct_certificate_missing", "support_report_missing"}
        ]
        full_family = [row for row in needing if len(row.get("matched_families") or []) >= 3]
        source_rules = sorted({rule for row in matched for rule in row.get("matched_rules") or []})
        out[lane_name] = {
            "full_family_needing_validation_transfers": sorted(
                {int_value(row.get("transfer_index")) for row in full_family}
            ),
            "needing_validation_transfers": sorted({int_value(row.get("transfer_index")) for row in needing}),
            "source": "validation_rows_token_match",
            "source_rules": source_rules,
        }
    return out


def build_backfill_queue(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    missing = [
        row
        for row in rows
        if row.get("direct_status") in {"direct_certificate_missing", "support_report_missing"}
    ]
    queue = []
    for row in sorted(missing, key=lane_priority, reverse=True)[:limit]:
        queue.append(
            {
                "direct_ops_over_rho": row.get("direct_ops_over_rho"),
                "lane_names": row.get("matched_lane_names"),
                "matched_families": row.get("matched_families"),
                "range": row.get("range"),
                "row_id": row.get("row_id"),
                "row_keys": row.get("row_keys"),
                "salts": row.get("salts"),
                "selected_term_support": row.get("selected_term_support"),
                "selector": row.get("selector"),
                "top_k": row.get("top_k"),
                "transfer_index": row.get("transfer_index"),
            }
        )
    return queue


def claim_status(rows: list[dict[str, Any]]) -> str:
    has_positive = any(row.get("evidence_positive") for row in rows)
    has_missing = any(row.get("direct_status") in {"direct_certificate_missing", "support_report_missing"} for row in rows)
    if has_positive and has_missing:
        return "FFE_ASSEMBLY_MANIFEST_HAS_VALIDATED_RANK_GAIN_STREAM_AND_BACKFILL_TARGETS"
    if has_positive:
        return "FFE_ASSEMBLY_MANIFEST_HAS_VALIDATED_RANK_GAIN_STREAM"
    if has_missing:
        return "FFE_ASSEMBLY_MANIFEST_HAS_BACKFILL_TARGETS_ONLY"
    return "FFE_ASSEMBLY_MANIFEST_HAS_NO_RANK_GAIN_STREAM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-audit", required=True, type=Path)
    parser.add_argument("--comparator-audit", required=True, type=Path)
    parser.add_argument("--feature-lift", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bridge = load_json(args.bridge_audit)
    comparator = load_json(args.comparator_audit)
    feature_lift = load_json(args.feature_lift)
    validation = load_json(args.validation_manifest)
    exact_index, same_salts_index, transfer_index = build_certificate_indexes(bridge)
    rows = [
        assembly_row(row, exact_index, same_salts_index, transfer_index)
        for row in validation.get("validation_rows") or []
        if 13 in set(support_tuple(row.get("selected_term_support")))
    ]
    rows.sort(key=lambda row: (int_value(row.get("transfer_index")), row.get("row_id")))
    lanes = build_lane_payloads(rows, max(1, args.limit))
    payload = {
        "artifacts": {
            "bridge_audit": str(args.bridge_audit),
            "comparator_audit": str(args.comparator_audit),
            "feature_lift": str(args.feature_lift),
            "validation_manifest": str(args.validation_manifest),
        },
        "assembly_contract": {
            "executor_input_fields": [
                "target",
                "transfer_index",
                "row_keys",
                "salts",
                "selected_term_support",
                "matched_families",
                "accepted_missing_columns",
                "form_supports",
            ],
            "ffe_role": (
                "Replay public row-key salt pairs as source-side FFE/summation-polynomial surfaces, "
                "then export direct/rank rows for the missing queue."
            ),
            "relation_streams": [lane["name"] for lane in LANE_DEFS],
            "rank_update_role": (
                "Use accepted missing columns and form supports as the sparse factor-row interface; "
                "rank_gain fields are bridge evidence, not newly computed here."
            ),
        },
        "backfill_queue": build_backfill_queue(rows, max(1, args.limit)),
        "claim_status": claim_status(rows),
        "created_at": now_iso(),
        "honesty_boundary": [
            "This is a relation assembly manifest only; no ECDLP recovery claim is made.",
            "No new direct relation, rank-gain, large-field FFE result, or Pollard-rho speedup is computed here.",
            "Rows with score_certificate_source other than exact_bridge_certificate inherit transfer/salt-level bridge evidence and must be replayed before promotion.",
            "Missing rows are backfill targets for direct/rank export, not successes.",
        ],
        "lanes": lanes,
        "parameters": {
            "limit": args.limit,
        },
        "promoted_rule": feature_lift.get("promoted_rule"),
        "relation_rows": rows,
        "schema": SCHEMA,
        "source_summaries": {
            "bridge": bridge.get("summary"),
            "comparator": comparator.get("summary"),
            "feature_lift": feature_lift.get("summary"),
            "validation": validation.get("summary"),
            "validation_rule_targets": validation_rule_targets(validation),
        },
        "summary": {
            "direct_status_counts": dict(sorted(Counter(str(row.get("direct_status")) for row in rows).items())),
            "lane_count": len(lanes),
            "positive_transfer_union": sorted(
                {int_value(row.get("transfer_index")) for row in rows if row.get("evidence_positive")}
            ),
            "relation_row_count": len(rows),
            "score_certificate_source_counts": dict(
                sorted(Counter(str(row.get("score_certificate_source")) for row in rows).items())
            ),
            "unique_transfer_count": len({int_value(row.get("transfer_index")) for row in rows}),
        },
    }
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "claim_status": payload["claim_status"],
                "summary": payload["summary"],
                "lane_summaries": {lane["name"]: lane["summary"] for lane in lanes},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
