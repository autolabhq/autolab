#!/usr/bin/env python3
"""Build a strict kernel contract from the sharp-lane replay workorder.

The replay workorder is intentionally human-readable.  This script lowers it
into an ABI-style contract for a future FFE/summation-polynomial worker:
stable first-pass group ids, second-pass row checks, exact certificate hashes,
inherited-promotion gates, and direct/rank backfill rows.

No summation polynomial is evaluated here.  The output is a fail-closed
contract that says what a lower-level kernel must reproduce before any sharp
selected13 lane promotion is credited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_ffe_sharp_lane_kernel_contract.v1"
DEFAULT_LANE = "selected13_sharp_minmod3"
DEFAULT_EXPECTED_EXACT = 6
DEFAULT_EXPECTED_BACKFILL = 12
DEFAULT_EXPECTED_INHERITED_ROWS = 12
DIRECT_MISSING_STATUSES = {"direct_certificate_missing", "support_report_missing"}
EXACT_SOURCE = "exact_bridge_certificate"
FULL_FAMILY_COUNT = 3


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


def sorted_ints(raw: Any) -> list[int]:
    return sorted(int_value(item) for item in (raw or []))


def sorted_strings(raw: Any) -> list[str]:
    return sorted(str(item) for item in (raw or []))


def sorted_supports(raw: Any) -> list[list[int]]:
    return sorted([sorted_ints(item) for item in (raw or [])])


def canonical_digest(material: Any, prefix: str) -> str:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]}"


def certificate_digest(cert: dict[str, Any]) -> str:
    return canonical_digest(canonical_certificate_material(cert), "cert")


def row_digest(row: dict[str, Any], public_first_pass: dict[str, Any], check_class: str) -> str:
    return canonical_digest(row_check_material(row, public_first_pass, check_class), "row")


def first_pass_material(group: dict[str, Any]) -> dict[str, Any]:
    public = group.get("public_first_pass") or {}
    return {
        "row_keys": sorted_strings(public.get("row_keys")),
        "salts": sorted_ints(public.get("salts")),
        "target": public.get("target"),
        "transfer_index": int_value(public.get("transfer_index")),
    }


def first_pass_id(group: dict[str, Any]) -> str:
    return canonical_digest(first_pass_material(group), "first")


def canonical_certificate_material(cert: dict[str, Any]) -> dict[str, Any]:
    rank_score = cert.get("rank_score") or {}
    return {
        "accepted_missing_columns": sorted_ints(cert.get("accepted_missing_columns")),
        "accepted_priority_columns": sorted_ints(cert.get("accepted_priority_columns")),
        "artifact": cert.get("artifact"),
        "classification": cert.get("classification"),
        "direct_ops_over_rho": float_value(cert.get("direct_ops_over_rho")),
        "form_supports": sorted_supports(cert.get("form_supports")),
        "rank": int_value(cert.get("rank")),
        "rank_gain": int_value(cert.get("rank_gain")),
        "rank_score": {
            "artifact": rank_score.get("artifact"),
            "claim_status": rank_score.get("claim_status"),
            "rank_after": int_value(rank_score.get("rank_after")),
            "rank_before": int_value(rank_score.get("rank_before")),
            "rank_gain": int_value(rank_score.get("rank_gain")),
            "unique_factor_relation_gain": int_value(rank_score.get("unique_factor_relation_gain")),
        },
        "row_keys": sorted_strings(cert.get("row_keys")),
        "selected_term_support": sorted_ints(cert.get("selected_term_support")),
        "selector": cert.get("selector"),
        "target": cert.get("target"),
        "top_k": int_value(cert.get("top_k")),
        "transfer_index": int_value(cert.get("transfer_index")),
        "unique_factor_relation_gain": int_value(cert.get("unique_factor_relation_gain")),
    }


def row_check_material(row: dict[str, Any], public_first_pass: dict[str, Any], check_class: str) -> dict[str, Any]:
    return {
        "accepted_missing_columns": sorted_ints(row.get("accepted_missing_columns")),
        "check_class": check_class,
        "direct_ops_over_rho": float_value(row.get("direct_ops_over_rho")),
        "direct_status": row.get("direct_status"),
        "evidence_positive": bool(row.get("evidence_positive")),
        "expected_form_supports": sorted_supports(row.get("expected_form_supports")),
        "matched_families": sorted_supports(row.get("matched_families")),
        "public_first_pass": {
            "row_keys": sorted_strings(public_first_pass.get("row_keys")),
            "salts": sorted_ints(public_first_pass.get("salts")),
            "target": public_first_pass.get("target"),
            "transfer_index": int_value(public_first_pass.get("transfer_index")),
        },
        "row_id": row.get("row_id"),
        "score_certificate_source": row.get("score_certificate_source"),
        "score_rank_gain": int_value(row.get("score_rank_gain")),
        "score_unique_factor_relation_gain": int_value(row.get("score_unique_factor_relation_gain")),
        "selected_term_support": sorted_ints(row.get("selected_term_support")),
        "selector": row.get("selector"),
        "top_k": int_value(row.get("top_k")),
    }


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("row_id")),
        str(row.get("selector")),
        int_value(row.get("top_k")),
        tuple(sorted_ints(row.get("selected_term_support"))),
    )


def full_family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if len(row.get("matched_families") or []) >= FULL_FAMILY_COUNT]


def group_row_sets(group: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "exact": list(group.get("exact_positive_rows") or []),
        "inherited": list(group.get("inherited_positive_rows") or []),
        "missing": list(group.get("missing_rows") or []),
        "neutral": list(group.get("neutral_rows") or []),
    }


def group_all_rows(group: dict[str, Any]) -> list[dict[str, Any]]:
    row_sets = group_row_sets(group)
    rows = row_sets["exact"] + row_sets["inherited"] + row_sets["missing"] + row_sets["neutral"]
    return sorted(rows, key=row_key)


def replay_groups(workorder: dict[str, Any]) -> list[dict[str, Any]]:
    groups = [group for group in workorder.get("replay_groups") or [] if isinstance(group, dict)]
    return sorted(groups, key=lambda group: (int_value(group.get("transfer_index")), str(group.get("group_id"))))


def public_first_pass_contract(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for group in groups:
        public = group.get("public_first_pass") or {}
        first_id = first_pass_id(group)
        row_sets = group_row_sets(group)
        out.append(
            {
                "direct_statuses": sorted_strings(group.get("direct_statuses")),
                "first_pass_id": first_id,
                "group_id": group.get("group_id"),
                "group_signature": canonical_digest(
                    {
                        "first_pass": first_pass_material(group),
                        "replay_class": group.get("replay_class"),
                        "second_pass_row_ids": sorted_strings(group.get("second_pass_row_ids")),
                    },
                    "group",
                ),
                "public_first_pass": {
                    "row_keys": sorted_strings(public.get("row_keys")),
                    "salt_gap": int_value(public.get("salt_gap")),
                    "salt_min_mod4": int_value(public.get("salt_min_mod4")),
                    "salts": sorted_ints(public.get("salts")),
                    "target": public.get("target"),
                    "transfer_index": int_value(public.get("transfer_index")),
                },
                "replay_class": group.get("replay_class"),
                "row_counts": {
                    "exact": len(row_sets["exact"]),
                    "inherited": len(row_sets["inherited"]),
                    "missing": len(row_sets["missing"]),
                    "neutral": len(row_sets["neutral"]),
                    "total": len(group_all_rows(group)),
                },
                "second_pass_row_ids": sorted_strings(group.get("second_pass_row_ids")),
                "selected_supports": sorted_strings(group.get("selected_supports")),
                "transfer_index": int_value(group.get("transfer_index")),
            }
        )
    return out


def second_pass_row_checks(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for group in groups:
        public = group.get("public_first_pass") or {}
        first_id = first_pass_id(group)
        row_sets = group_row_sets(group)
        row_classes = (
            [("exact_positive_row", row) for row in row_sets["exact"]]
            + [("inherited_promotion_row", row) for row in row_sets["inherited"]]
            + [("direct_rank_backfill_row", row) for row in row_sets["missing"]]
            + [("neutral_exported_row", row) for row in row_sets["neutral"]]
        )
        for check_class, row in sorted(row_classes, key=lambda item: (item[0], row_key(item[1]))):
            check = {
                "check_class": check_class,
                "direct_status": row.get("direct_status"),
                "evidence_positive": bool(row.get("evidence_positive")),
                "first_pass_id": first_id,
                "group_id": group.get("group_id"),
                "matched_families": sorted_supports(row.get("matched_families")),
                "row_id": row.get("row_id"),
                "score_certificate_source": row.get("score_certificate_source"),
                "selected_term_support": sorted_ints(row.get("selected_term_support")),
                "selector": row.get("selector"),
                "top_k": int_value(row.get("top_k")),
                "transfer_index": int_value(group.get("transfer_index")),
            }
            check["row_check_hash"] = row_digest(row, public, check_class)
            checks.append(check)
    return sorted(checks, key=lambda check: (int_value(check.get("transfer_index")), str(check.get("row_id"))))


def exact_certificate_checks(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for group in groups:
        public = group.get("public_first_pass") or {}
        for row in group.get("exact_positive_rows") or []:
            cert = row.get("exact_bridge_certificate")
            if not isinstance(cert, dict):
                cert = {}
            material = canonical_certificate_material(cert)
            checks.append(
                {
                    "accepted_missing_columns": sorted_ints(row.get("accepted_missing_columns")),
                    "certificate_hash": certificate_digest(cert) if cert else None,
                    "certificate_material": material if cert else None,
                    "expected_form_supports": sorted_supports(row.get("expected_form_supports")),
                    "first_pass_id": first_pass_id(group),
                    "group_id": group.get("group_id"),
                    "row_check_hash": row_digest(row, public, "exact_positive_row"),
                    "row_id": row.get("row_id"),
                    "selector": row.get("selector"),
                    "top_k": int_value(row.get("top_k")),
                    "transfer_index": int_value(group.get("transfer_index")),
                }
            )
    return sorted(checks, key=lambda check: (int_value(check.get("transfer_index")), str(check.get("row_id"))))


def inherited_promotion_checks(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for group in groups:
        public = group.get("public_first_pass") or {}
        for row in group.get("inherited_positive_rows") or []:
            checks.append(
                {
                    "accepted_missing_columns": sorted_ints(row.get("accepted_missing_columns")),
                    "blocking_condition": "requires_exact_row_level_certificate_before_promotion",
                    "expected_form_supports": sorted_supports(row.get("expected_form_supports")),
                    "first_pass_id": first_pass_id(group),
                    "group_id": group.get("group_id"),
                    "has_exact_bridge_certificate": isinstance(row.get("exact_bridge_certificate"), dict),
                    "row_check_hash": row_digest(row, public, "inherited_promotion_row"),
                    "row_id": row.get("row_id"),
                    "score_certificate_source": row.get("score_certificate_source"),
                    "selector": row.get("selector"),
                    "top_k": int_value(row.get("top_k")),
                    "transfer_index": int_value(group.get("transfer_index")),
                }
            )
    return sorted(checks, key=lambda check: (int_value(check.get("transfer_index")), str(check.get("row_id"))))


def direct_rank_backfill_manifest(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue = []
    for group in groups:
        missing = list(group.get("missing_rows") or [])
        if not missing:
            continue
        public = group.get("public_first_pass") or {}
        full_rows = full_family_rows(missing)
        queue.append(
            {
                "first_pass_id": first_pass_id(group),
                "full_family_row_ids": sorted_strings(row.get("row_id") for row in full_rows),
                "group_id": group.get("group_id"),
                "min_direct_ops_over_rho": float_value(group.get("min_direct_ops_over_rho")),
                "public_first_pass": {
                    "row_keys": sorted_strings(public.get("row_keys")),
                    "salt_gap": int_value(public.get("salt_gap")),
                    "salt_min_mod4": int_value(public.get("salt_min_mod4")),
                    "salts": sorted_ints(public.get("salts")),
                    "target": public.get("target"),
                    "transfer_index": int_value(public.get("transfer_index")),
                },
                "queue_order": len(queue),
                "row_checks": [
                    {
                        "direct_status": row.get("direct_status"),
                        "is_full_family": len(row.get("matched_families") or []) >= FULL_FAMILY_COUNT,
                        "matched_families": sorted_supports(row.get("matched_families")),
                        "row_check_hash": row_digest(row, public, "direct_rank_backfill_row"),
                        "row_id": row.get("row_id"),
                        "selected_term_support": sorted_ints(row.get("selected_term_support")),
                        "selector": row.get("selector"),
                        "top_k": int_value(row.get("top_k")),
                    }
                    for row in sorted(missing, key=row_key)
                ],
                "transfer_index": int_value(group.get("transfer_index")),
            }
        )
    return sorted(queue, key=lambda item: int_value(item.get("transfer_index")))


def queue_group_ids(queue: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("group_id")) for item in queue}


def validate_contract(
    workorder: dict[str, Any],
    groups: list[dict[str, Any]],
    exact_checks: list[dict[str, Any]],
    inherited_checks: list[dict[str, Any]],
    backfill_manifest: list[dict[str, Any]],
    lane: str,
    expected_exact: int,
    expected_backfill: int,
    expected_inherited_rows: int,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    summary = workorder.get("summary") or {}
    queues = workorder.get("queues") or {}

    def fail(code: str, **details: Any) -> None:
        failures.append({"code": code, **details})

    if workorder.get("claim_status") != "FFE_SHARP_LANE_REPLAY_WORKORDER_READY":
        fail("workorder_not_ready", claim_status=workorder.get("claim_status"))
    if workorder.get("failures"):
        fail("workorder_has_failures", failures=workorder.get("failures"))
    if workorder.get("lane") != lane:
        fail("workorder_lane_mismatch", expected=lane, observed=workorder.get("lane"))

    exact_groups = [group for group in groups if group.get("exact_positive_rows")]
    backfill_groups = [group for group in groups if group.get("missing_rows")]
    inherited_rows = [row for group in groups for row in group.get("inherited_positive_rows") or []]
    neutral_rows = [row for group in groups for row in group.get("neutral_rows") or []]
    all_rows = [row for group in groups for row in group_all_rows(group)]

    if len(exact_groups) != expected_exact:
        fail("exact_group_count_mismatch", expected=expected_exact, observed=len(exact_groups))
    if len(backfill_groups) != expected_backfill:
        fail("backfill_group_count_mismatch", expected=expected_backfill, observed=len(backfill_groups))
    if len(inherited_rows) != expected_inherited_rows:
        fail("inherited_row_count_mismatch", expected=expected_inherited_rows, observed=len(inherited_rows))
    if int_value(summary.get("exact_positive_group_count")) != len(exact_groups):
        fail("summary_exact_group_count_mismatch", summary=summary.get("exact_positive_group_count"), observed=len(exact_groups))
    if int_value(summary.get("missing_backfill_group_count")) != len(backfill_groups):
        fail("summary_backfill_group_count_mismatch", summary=summary.get("missing_backfill_group_count"), observed=len(backfill_groups))
    if int_value(summary.get("inherited_positive_row_count")) != len(inherited_rows):
        fail("summary_inherited_row_count_mismatch", summary=summary.get("inherited_positive_row_count"), observed=len(inherited_rows))
    if int_value(summary.get("neutral_exported_row_count")) != len(neutral_rows):
        fail("summary_neutral_row_count_mismatch", summary=summary.get("neutral_exported_row_count"), observed=len(neutral_rows))
    if int_value(summary.get("row_count")) != len(all_rows):
        fail("summary_row_count_mismatch", summary=summary.get("row_count"), observed=len(all_rows))

    group_ids = [str(group.get("group_id")) for group in groups]
    if len(set(group_ids)) != len(group_ids):
        fail("duplicate_group_ids", duplicates=sorted(item for item, count in Counter(group_ids).items() if count > 1))
    first_ids = [first_pass_id(group) for group in groups]
    if len(set(first_ids)) != len(first_ids):
        fail("duplicate_first_pass_ids", duplicates=sorted(item for item, count in Counter(first_ids).items() if count > 1))

    row_ids = [str(row.get("row_id")) for row in all_rows]
    if len(set(row_ids)) != len(row_ids):
        fail("duplicate_row_ids", duplicates=sorted(item for item, count in Counter(row_ids).items() if count > 1))

    for group in groups:
        public = group.get("public_first_pass") or {}
        salts = sorted_ints(public.get("salts"))
        if len(salts) != 2:
            fail("first_pass_bad_salt_count", group_id=group.get("group_id"), salts=salts)
        else:
            if max(salts) - min(salts) == 1:
                fail("first_pass_salts_adjacent", group_id=group.get("group_id"), salts=salts)
            if min(salts) % 4 != 3:
                fail("first_pass_min_salt_mod4_not_3", group_id=group.get("group_id"), salts=salts)
        listed = sorted_strings(group.get("second_pass_row_ids"))
        observed = sorted_strings(row.get("row_id") for row in group_all_rows(group))
        if listed != observed:
            fail("second_pass_row_id_coverage_mismatch", group_id=group.get("group_id"), listed=listed, observed=observed)

    for group in exact_groups:
        public = group.get("public_first_pass") or {}
        for row in group.get("exact_positive_rows") or []:
            cert = row.get("exact_bridge_certificate")
            if not isinstance(cert, dict):
                fail("exact_row_missing_certificate", group_id=group.get("group_id"), row_id=row.get("row_id"))
                continue
            if row.get("direct_status") != "direct_certificate_exported":
                fail("exact_row_not_exported", group_id=group.get("group_id"), row_id=row.get("row_id"), direct_status=row.get("direct_status"))
            if not row.get("evidence_positive"):
                fail("exact_row_not_positive", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if row.get("score_certificate_source") != EXACT_SOURCE:
                fail("exact_row_wrong_source", group_id=group.get("group_id"), row_id=row.get("row_id"), source=row.get("score_certificate_source"))
            if int_value(cert.get("rank_gain")) <= 0 or int_value(cert.get("unique_factor_relation_gain")) <= 0:
                fail("exact_row_nonpositive_certificate_gain", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if sorted_ints(cert.get("accepted_missing_columns")) != sorted_ints(row.get("accepted_missing_columns")):
                fail("exact_row_accepted_missing_mismatch", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if sorted_supports(cert.get("form_supports")) != sorted_supports(row.get("expected_form_supports")):
                fail("exact_row_form_support_mismatch", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if sorted_strings(cert.get("row_keys")) != sorted_strings(public.get("row_keys")):
                fail("exact_row_key_mismatch", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if int_value(cert.get("transfer_index")) != int_value(public.get("transfer_index")):
                fail("exact_transfer_mismatch", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if sorted_ints(cert.get("selected_term_support")) != sorted_ints(row.get("selected_term_support")):
                fail("exact_selected_support_mismatch", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if cert.get("selector") != row.get("selector") or int_value(cert.get("top_k")) != int_value(row.get("top_k")):
                fail("exact_selector_topk_mismatch", group_id=group.get("group_id"), row_id=row.get("row_id"))

    for group in groups:
        for row in group.get("inherited_positive_rows") or []:
            if row.get("exact_bridge_certificate") is not None:
                fail("inherited_row_has_exact_certificate", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if row.get("score_certificate_source") == EXACT_SOURCE:
                fail("inherited_row_marked_exact_source", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if not row.get("evidence_positive"):
                fail("inherited_row_not_positive", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if int_value(row.get("score_rank_gain")) <= 0:
                fail("inherited_row_without_score_gain", group_id=group.get("group_id"), row_id=row.get("row_id"))

    for group in backfill_groups:
        missing = list(group.get("missing_rows") or [])
        full_rows = full_family_rows(missing)
        if not full_rows:
            fail("backfill_group_has_no_full_family_row", group_id=group.get("group_id"), transfer_index=group.get("transfer_index"))
        for row in missing:
            if row.get("direct_status") not in DIRECT_MISSING_STATUSES:
                fail("backfill_row_not_direct_missing", group_id=group.get("group_id"), row_id=row.get("row_id"), direct_status=row.get("direct_status"))
        for row in full_rows:
            if row.get("direct_status") not in DIRECT_MISSING_STATUSES:
                fail("full_family_backfill_row_not_direct_missing", group_id=group.get("group_id"), row_id=row.get("row_id"))

    for group in groups:
        for row in group.get("neutral_rows") or []:
            if row.get("direct_status") in DIRECT_MISSING_STATUSES:
                fail("neutral_row_is_direct_missing", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if row.get("exact_bridge_certificate") is not None:
                fail("neutral_row_has_exact_certificate", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if row.get("evidence_positive"):
                fail("neutral_row_marked_positive", group_id=group.get("group_id"), row_id=row.get("row_id"))
            if int_value(row.get("score_rank_gain")) > 0:
                fail("neutral_row_has_score_gain", group_id=group.get("group_id"), row_id=row.get("row_id"))

    exact_queue_ids = queue_group_ids(queues.get("exact_positive_replay_queue") or [])
    observed_exact_ids = {str(group.get("group_id")) for group in exact_groups}
    if exact_queue_ids != observed_exact_ids:
        fail("exact_queue_group_mismatch", queue=sorted(exact_queue_ids), observed=sorted(observed_exact_ids))
    backfill_queue_ids = queue_group_ids(queues.get("direct_rank_backfill_queue") or [])
    observed_backfill_ids = {str(group.get("group_id")) for group in backfill_groups}
    if backfill_queue_ids != observed_backfill_ids:
        fail("backfill_queue_group_mismatch", queue=sorted(backfill_queue_ids), observed=sorted(observed_backfill_ids))
    inherited_queue_ids = queue_group_ids(queues.get("inherited_promotion_queue") or [])
    observed_inherited_ids = {str(group.get("group_id")) for group in groups if group.get("inherited_positive_rows")}
    if inherited_queue_ids != observed_inherited_ids:
        fail("inherited_queue_group_mismatch", queue=sorted(inherited_queue_ids), observed=sorted(observed_inherited_ids))

    groups_by_id = {str(group.get("group_id")): group for group in groups}
    for queue_item in queues.get("exact_positive_replay_queue") or []:
        group = groups_by_id.get(str(queue_item.get("group_id")))
        expected_rows = sorted_strings(row.get("row_id") for row in (group or {}).get("exact_positive_rows") or [])
        queued_rows = sorted_strings(row.get("row_id") for row in queue_item.get("expected_exact_rows") or [])
        if expected_rows != queued_rows:
            fail("exact_queue_row_mismatch", group_id=queue_item.get("group_id"), queue=queued_rows, observed=expected_rows)
    for queue_item in queues.get("inherited_promotion_queue") or []:
        group = groups_by_id.get(str(queue_item.get("group_id")))
        expected_rows = sorted_strings(row.get("row_id") for row in (group or {}).get("inherited_positive_rows") or [])
        queued_rows = sorted_strings(row.get("row_id") for row in queue_item.get("inherited_positive_rows") or [])
        if expected_rows != queued_rows:
            fail("inherited_queue_row_mismatch", group_id=queue_item.get("group_id"), queue=queued_rows, observed=expected_rows)
    for queue_item in queues.get("direct_rank_backfill_queue") or []:
        group = groups_by_id.get(str(queue_item.get("group_id")))
        expected_rows = sorted_strings(row.get("row_id") for row in full_family_rows((group or {}).get("missing_rows") or []))
        queued_rows = sorted_strings(queue_item.get("full_family_row_ids") or [])
        if expected_rows != queued_rows:
            fail("backfill_queue_full_family_row_mismatch", group_id=queue_item.get("group_id"), queue=queued_rows, observed=expected_rows)

    if len(exact_checks) != expected_exact:
        fail("exact_check_count_mismatch", expected=expected_exact, observed=len(exact_checks))
    if len(inherited_checks) != expected_inherited_rows:
        fail("inherited_check_count_mismatch", expected=expected_inherited_rows, observed=len(inherited_checks))
    if len(backfill_manifest) != expected_backfill:
        fail("backfill_manifest_count_mismatch", expected=expected_backfill, observed=len(backfill_manifest))

    return failures


def summarize(groups: list[dict[str, Any]], exact_checks: list[dict[str, Any]], inherited_checks: list[dict[str, Any]], backfill_manifest: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = [row for group in groups for row in group_all_rows(group)]
    row_statuses = Counter(str(row.get("direct_status")) for row in all_rows)
    exact_transfers = sorted({int_value(check.get("transfer_index")) for check in exact_checks})
    backfill_transfers = sorted({int_value(item.get("transfer_index")) for item in backfill_manifest})
    return {
        "backfill_group_count": len(backfill_manifest),
        "backfill_transfers": backfill_transfers,
        "direct_status_counts": dict(sorted(row_statuses.items())),
        "exact_certificate_check_count": len(exact_checks),
        "exact_positive_transfers": exact_transfers,
        "first_pass_group_count": len(groups),
        "inherited_promotion_check_count": len(inherited_checks),
        "neutral_exported_row_count": sum(1 for group in groups for _row in group.get("neutral_rows") or []),
        "neutral_exported_transfers": sorted(
            {
                int_value(row.get("transfer_index"))
                for group in groups
                for row in group.get("neutral_rows") or []
            }
        ),
        "row_check_count": len(all_rows),
        "transfer_count": len({int_value(group.get("transfer_index")) for group in groups}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workorder", required=True, type=Path)
    parser.add_argument("--lane", default=DEFAULT_LANE)
    parser.add_argument("--expected-exact-groups", default=DEFAULT_EXPECTED_EXACT, type=int)
    parser.add_argument("--expected-backfill-groups", default=DEFAULT_EXPECTED_BACKFILL, type=int)
    parser.add_argument("--expected-inherited-rows", default=DEFAULT_EXPECTED_INHERITED_ROWS, type=int)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workorder = load_json(args.workorder)
    groups = replay_groups(workorder)
    first_pass = public_first_pass_contract(groups)
    second_pass = second_pass_row_checks(groups)
    exact_checks = exact_certificate_checks(groups)
    inherited_checks = inherited_promotion_checks(groups)
    backfill_manifest = direct_rank_backfill_manifest(groups)
    failures = validate_contract(
        workorder,
        groups,
        exact_checks,
        inherited_checks,
        backfill_manifest,
        args.lane,
        args.expected_exact_groups,
        args.expected_backfill_groups,
        args.expected_inherited_rows,
    )
    summary = summarize(groups, exact_checks, inherited_checks, backfill_manifest)
    payload = {
        "artifacts": {
            "replay_workorder": str(args.workorder),
        },
        "claim_status": "FFE_SHARP_LANE_KERNEL_CONTRACT_READY" if not failures else "FFE_SHARP_LANE_KERNEL_CONTRACT_FAILED_CHECK",
        "contract": {
            "direct_rank_backfill_manifest": backfill_manifest,
            "exact_certificate_checks": exact_checks,
            "first_pass_contract": first_pass,
            "inherited_promotion_checks": inherited_checks,
            "second_pass_contract": second_pass,
        },
        "created_at": now_iso(),
        "failures": failures,
        "honesty_boundary": [
            "This is a kernel contract and validation gate only.",
            "It does not evaluate summation polynomials, export new direct/rank rows, recover an ECDLP scalar, or claim a Pollard-rho speedup.",
            "Inherited positives remain blocked until an exact row-level certificate is reproduced by the lower-level worker.",
            "Backfill transfers remain missing until direct/rank export validates them.",
        ],
        "lane": args.lane,
        "parameters": {
            "expected_backfill_groups": args.expected_backfill_groups,
            "expected_exact_groups": args.expected_exact_groups,
            "expected_inherited_rows": args.expected_inherited_rows,
        },
        "schema": SCHEMA,
        "source_summary": workorder.get("summary"),
        "summary": summary,
    }
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "claim_status": payload["claim_status"],
                "failures": failures,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
