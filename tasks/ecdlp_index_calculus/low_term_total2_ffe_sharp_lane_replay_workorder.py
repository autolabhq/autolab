#!/usr/bin/env python3
"""Build a sharp selected13 FFE replay workorder from an assembly manifest.

The assembly manifest identifies a public selected13 lane with exact bridge
positives and direct-missing targets.  This script turns that lane into a
kernel-facing replay contract: public first-pass groups, exact-positive replay
checks, inherited row-promotion checks, and direct/rank backfill targets.

It does not evaluate summation polynomials.  The output is the specification a
lower-level FFE/summation-polynomial implementation must satisfy before any
lane promotion can be claimed.
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


SCHEMA = "ecdlp.low_term_total2_ffe_sharp_lane_replay_workorder.v1"
DEFAULT_LANE = "selected13_sharp_minmod3"
SHARP_TOKENS = ("selected_has=13", "salt_adjacent=False", "salt_min_mod4=3")
FULL_FAMILIES = ((11, 15), (10, 14), (0, 5))


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


def stable_id(prefix: str, material: Any) -> str:
    blob = json.dumps(material, sort_keys=True)
    return f"{prefix}_{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:12]}"


def cert_rank_gain(cert: dict[str, Any] | None) -> int:
    if not cert:
        return 0
    return int_value(cert.get("rank_gain"))


def cert_unique_gain(cert: dict[str, Any] | None) -> int:
    if not cert:
        return 0
    return int_value(cert.get("unique_factor_relation_gain"))


def row_priority(row: dict[str, Any]) -> tuple[int, int, float, str]:
    exact = 2 if row.get("score_certificate_source") == "exact_bridge_certificate" else 0
    positive = 1 if row.get("evidence_positive") else 0
    cost = float_value(row.get("direct_ops_over_rho"))
    return (exact, positive, -(cost if cost is not None else 9.0), str(row.get("row_id")))


def group_priority(group: dict[str, Any]) -> tuple[int, int, float, int]:
    class_score = {
        "exact_positive_replay": 3,
        "direct_rank_backfill": 2,
        "inherited_positive_replay": 1,
    }.get(group.get("replay_class"), 0)
    cost = float_value(group.get("min_direct_ops_over_rho"))
    return (
        class_score,
        int_value(group.get("full_family_row_count")),
        -(cost if cost is not None else 9.0),
        -int_value(group.get("transfer_index")),
    )


def exact_certificate(row: dict[str, Any]) -> dict[str, Any] | None:
    cert = row.get("exact_bridge_certificate")
    if isinstance(cert, dict) and cert_rank_gain(cert) > 0:
        return cert
    return None


def group_key(row: dict[str, Any]) -> tuple[int, tuple[str, ...]]:
    return (int_value(row.get("transfer_index")), row_key_tuple(row.get("row_keys") or []))


def rows_for_lane(manifest: dict[str, Any], lane: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in manifest.get("relation_rows") or []
        if lane in set(row.get("matched_lane_names") or [])
    ]
    return sorted(rows, key=lambda row: (int_value(row.get("transfer_index")), str(row.get("row_id"))))


def lane_summary(manifest: dict[str, Any], lane: str) -> dict[str, Any]:
    for item in manifest.get("lanes") or []:
        if item.get("name") == lane:
            return item.get("summary") or {}
    return {}


def validate_sharp_rows(rows: list[dict[str, Any]], lane: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        salts = [int_value(item) for item in row.get("salts") or []]
        selected = set(support_tuple(row.get("selected_term_support")))
        if len(salts) != 2:
            failures.append({"code": "bad_salt_count", "row_id": row.get("row_id"), "salts": salts})
            continue
        if 13 not in selected:
            failures.append({"code": "missing_selected13", "row_id": row.get("row_id")})
        if max(salts) - min(salts) == 1:
            failures.append({"code": "salt_pair_is_adjacent", "row_id": row.get("row_id"), "salts": salts})
        if min(salts) % 4 != 3:
            failures.append({"code": "salt_min_mod4_not_3", "row_id": row.get("row_id"), "salts": salts})
        if lane not in set(row.get("matched_lane_names") or []):
            failures.append({"code": "lane_membership_missing", "row_id": row.get("row_id"), "lane": lane})
    return failures


def row_contract(row: dict[str, Any]) -> dict[str, Any]:
    exact = exact_certificate(row)
    score_cert = row.get("best_transfer_bridge_certificate") or row.get("same_salt_bridge_certificate") or exact
    return {
        "accepted_missing_columns": (score_cert or {}).get("accepted_missing_columns"),
        "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
        "direct_status": row.get("direct_status"),
        "evidence_positive": bool(row.get("evidence_positive")),
        "exact_bridge_certificate": exact,
        "exact_rank_gain": int_value(row.get("exact_rank_gain")),
        "expected_form_supports": (score_cert or {}).get("form_supports"),
        "matched_families": row.get("matched_families") or [],
        "row_id": row.get("row_id"),
        "score_certificate_source": row.get("score_certificate_source"),
        "score_rank_gain": int_value(row.get("score_rank_gain")),
        "score_unique_factor_relation_gain": int_value(row.get("score_unique_factor_relation_gain")),
        "selected_term_support": list(support_tuple(row.get("selected_term_support"))),
        "selector": row.get("selector"),
        "target": row.get("target"),
        "top_k": int_value(row.get("top_k")),
        "transfer_index": int_value(row.get("transfer_index")),
    }


def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, tuple[str, ...]], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)

    groups = []
    for (transfer, row_keys), items in sorted(grouped.items()):
        items = sorted(items, key=row_priority, reverse=True)
        exact_rows = [row for row in items if exact_certificate(row) is not None]
        inherited_positive = [
            row
            for row in items
            if row.get("evidence_positive") and row.get("score_certificate_source") != "exact_bridge_certificate"
        ]
        missing_rows = [
            row
            for row in items
            if row.get("direct_status") in {"direct_certificate_missing", "support_report_missing"}
        ]
        assigned_row_ids = {
            str(row.get("row_id"))
            for row in exact_rows + inherited_positive + missing_rows
        }
        neutral_rows = [row for row in items if str(row.get("row_id")) not in assigned_row_ids]
        full_family_rows = [row for row in items if len(row.get("matched_families") or []) >= len(FULL_FAMILIES)]
        costs = [float(row["direct_ops_over_rho"]) for row in items if row.get("direct_ops_over_rho") is not None]
        replay_class = "direct_rank_backfill"
        if exact_rows:
            replay_class = "exact_positive_replay"
        elif inherited_positive:
            replay_class = "inherited_positive_replay"
        elif neutral_rows and not missing_rows:
            replay_class = "neutral_exported_replay"
        group = {
            "backfill_required": bool(missing_rows),
            "direct_statuses": sorted({str(row.get("direct_status")) for row in items}),
            "exact_positive_row_count": len(exact_rows),
            "exact_positive_rows": [row_contract(row) for row in exact_rows],
            "full_family_row_count": len(full_family_rows),
            "group_id": stable_id("sharp_group", {"transfer": transfer, "row_keys": row_keys}),
            "inherited_positive_row_count": len(inherited_positive),
            "inherited_positive_rows": [row_contract(row) for row in inherited_positive],
            "max_direct_ops_over_rho": round(max(costs), 8) if costs else None,
            "min_direct_ops_over_rho": round(min(costs), 8) if costs else None,
            "missing_row_count": len(missing_rows),
            "missing_rows": [row_contract(row) for row in missing_rows],
            "neutral_row_count": len(neutral_rows),
            "neutral_rows": [row_contract(row) for row in neutral_rows],
            "public_first_pass": {
                "row_keys": list(row_keys),
                "salt_gap": max(items[0].get("salts") or [0]) - min(items[0].get("salts") or [0]),
                "salt_min_mod4": min(items[0].get("salts") or [0]) % 4,
                "salts": items[0].get("salts") or [],
                "target": items[0].get("target"),
                "transfer_index": transfer,
            },
            "replay_class": replay_class,
            "row_count": len(items),
            "second_pass_row_ids": [str(row.get("row_id")) for row in items],
            "selected_supports": sorted({support_key(row.get("selected_term_support")) for row in items}),
            "transfer_index": transfer,
        }
        groups.append(group)
    return sorted(groups, key=group_priority, reverse=True)


def build_queues(groups: list[dict[str, Any]]) -> dict[str, Any]:
    exact_groups = [group for group in groups if group.get("replay_class") == "exact_positive_replay"]
    backfill_groups = [group for group in groups if group.get("backfill_required")]
    inherited_groups = [group for group in groups if int_value(group.get("inherited_positive_row_count")) > 0]
    return {
        "direct_rank_backfill_queue": [
            {
                "full_family_row_ids": [
                    row["row_id"]
                    for row in group.get("missing_rows") or []
                    if len(row.get("matched_families") or []) >= len(FULL_FAMILIES)
                ],
                "group_id": group.get("group_id"),
                "min_direct_ops_over_rho": group.get("min_direct_ops_over_rho"),
                "public_first_pass": group.get("public_first_pass"),
                "row_count": group.get("row_count"),
                "transfer_index": group.get("transfer_index"),
            }
            for group in backfill_groups
        ],
        "exact_positive_replay_queue": [
            {
                "expected_exact_rows": group.get("exact_positive_rows"),
                "group_id": group.get("group_id"),
                "public_first_pass": group.get("public_first_pass"),
                "transfer_index": group.get("transfer_index"),
            }
            for group in exact_groups
        ],
        "inherited_promotion_queue": [
            {
                "group_id": group.get("group_id"),
                "inherited_positive_rows": group.get("inherited_positive_rows"),
                "public_first_pass": group.get("public_first_pass"),
                "transfer_index": group.get("transfer_index"),
            }
            for group in inherited_groups
        ],
    }


def summarize(rows: list[dict[str, Any]], groups: list[dict[str, Any]], lane_info: dict[str, Any]) -> dict[str, Any]:
    costs = [float(row["direct_ops_over_rho"]) for row in rows if row.get("direct_ops_over_rho") is not None]
    exact_groups = [group for group in groups if group.get("replay_class") == "exact_positive_replay"]
    missing_groups = [group for group in groups if group.get("backfill_required")]
    inherited_rows = [
        row
        for group in groups
        for row in group.get("inherited_positive_rows") or []
    ]
    neutral_rows = [
        row
        for group in groups
        for row in group.get("neutral_rows") or []
    ]
    return {
        "direct_status_counts": dict(sorted(Counter(str(row.get("direct_status")) for row in rows).items())),
        "direct_ops_over_rho_median": round(float(median(costs)), 8) if costs else None,
        "exact_positive_group_count": len(exact_groups),
        "exact_positive_transfers": sorted(int_value(group.get("transfer_index")) for group in exact_groups),
        "full_family_missing_transfers": lane_info.get("full_family_missing_validation_transfers") or [],
        "group_count": len(groups),
        "inherited_positive_row_count": len(inherited_rows),
        "lane_positive_transfers": lane_info.get("positive_transfers") or [],
        "missing_backfill_group_count": len(missing_groups),
        "missing_backfill_transfers": sorted(int_value(group.get("transfer_index")) for group in missing_groups),
        "neutral_exported_row_count": len(neutral_rows),
        "neutral_exported_transfers": sorted({int_value(row.get("transfer_index")) for row in neutral_rows}),
        "row_count": len(rows),
        "score_certificate_source_counts": dict(
            sorted(Counter(str(row.get("score_certificate_source")) for row in rows).items())
        ),
        "transfer_count": len({int_value(row.get("transfer_index")) for row in rows}),
    }


def cross_check_lane_summary(summary: dict[str, Any], lane_info: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    expected_positive = sorted(int_value(item) for item in lane_info.get("exact_rank_gain_transfers") or [])
    observed_positive = summary.get("exact_positive_transfers") or []
    if observed_positive != expected_positive:
        failures.append(
            {
                "code": "exact_positive_transfer_mismatch",
                "expected": expected_positive,
                "observed": observed_positive,
            }
        )
    expected_missing = sorted(int_value(item) for item in lane_info.get("full_family_missing_validation_transfers") or [])
    observed_missing = summary.get("missing_backfill_transfers") or []
    if observed_missing != expected_missing:
        failures.append(
            {
                "code": "full_family_missing_transfer_mismatch",
                "expected": expected_missing,
                "observed": observed_missing,
            }
        )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-manifest", required=True, type=Path)
    parser.add_argument("--lane", default=DEFAULT_LANE)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_json(args.assembly_manifest)
    rows = rows_for_lane(manifest, args.lane)
    lane_info = lane_summary(manifest, args.lane)
    groups = group_rows(rows)
    summary = summarize(rows, groups, lane_info)
    failures = validate_sharp_rows(rows, args.lane) + cross_check_lane_summary(summary, lane_info)
    queues = build_queues(groups)
    payload = {
        "artifacts": {
            "assembly_manifest": str(args.assembly_manifest),
        },
        "claim_status": (
            "FFE_SHARP_LANE_REPLAY_WORKORDER_READY"
            if rows and not failures
            else "FFE_SHARP_LANE_REPLAY_WORKORDER_FAILED_CHECK"
        ),
        "created_at": now_iso(),
        "failures": failures,
        "honesty_boundary": [
            "This workorder specifies replay and backfill targets only.",
            "It does not evaluate summation polynomials or compute new rank-gain evidence.",
            "Inherited positives must become exact row-level certificates before promotion.",
            "Backfill rows are not successes until direct/rank export validates them.",
        ],
        "kernel_contract": {
            "first_pass_group_key": ["target", "transfer_index", "row_keys", "salts"],
            "required_lane_tokens": list(SHARP_TOKENS),
            "required_positive_check": (
                "For exact_positive_replay groups, reproduce the exact bridge certificate fields "
                "for row_keys, selector, top_k, selected support, accepted missing columns, "
                "form supports, rank gain, and unique factor-relation gain."
            ),
            "required_backfill_check": (
                "For direct_rank_backfill groups, export direct/rank rows for the full-family "
                "selected support first, then attempt lower-support variants with inherited evidence kept separate."
            ),
        },
        "lane": args.lane,
        "queues": queues,
        "replay_groups": groups,
        "schema": SCHEMA,
        "source_lane_summary": lane_info,
        "summary": summary,
    }
    write_json(args.out, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "summary": summary, "failures": failures}, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
