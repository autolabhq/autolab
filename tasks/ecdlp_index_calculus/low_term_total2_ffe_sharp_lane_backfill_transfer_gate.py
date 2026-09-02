#!/usr/bin/env python3
"""Bind selected13 sidecar coefficient forms to backfill row gates.

The sidecar coefficient preflight extracts source relation forms for the two
priority backfill transfers.  This script turns those forms into a stricter
worker input contract: every form is bound to the missing backfill row hash,
grouped into per-mask replay lanes, and emitted as a native-checkable C header.

This is a transfer gate only.  It does not accept source coefficients as proof,
re-evaluate a summation polynomial, export a direct/rank row, solve ECDLP, or
claim a Pollard-rho speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_ffe_sharp_lane_backfill_transfer_gate.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SIDECAR_COEFF = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_sidecar_coeff_preflight_selected13_9696_9999_probe.json"
)
DEFAULT_CONTRACT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9999_probe.json"
)
DEFAULT_WORKLIST = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_execution_worklist_selected13_9696_9999_probe.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_backfill_transfer_gate_selected13_9696_9999_probe.json"
)
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_backfill_transfer_gate_selected13_9696_9999_probe.h"
)

ORDER = 11779
COEFF_COUNT = 17
FULL_FAMILY_MASKS = [33, 17408, 34816]
PRIMARY_BACKFILL_TRANSFERS = [9981, 9943]
EXPECTED_FORM_COUNT = 12
EXPECTED_MASK_LANE_COUNT = 6
EXPECTED_TRANSFER_GATE_COUNT = 2
TIER_CODES = {
    "same_row_key_exact_support": 1,
    "one_salt_neighbor_exact_support": 2,
    "support_span_only": 3,
}
LANE_CLASSES = {
    "same_row_key_exact_support": "same_row_key_replay_lane",
    "one_salt_neighbor_exact_support": "one_salt_neighbor_mutation_lane",
    "support_span_only": "support_span_only_source_hint_lane",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def digest_u64(raw: Any) -> int:
    text = str(raw or "")
    suffix = text.rsplit("_", 1)[-1]
    if suffix and all(ch in "0123456789abcdefABCDEF" for ch in suffix):
        return int(suffix[-16:], 16)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def canonical_json(raw: Any) -> str:
    return json.dumps(raw, sort_keys=True, separators=(",", ":"))


def support_mask_from_coeffs(coeffs: list[Any], order: int = ORDER) -> int:
    mask = 0
    for index, coeff in enumerate(coeffs[1:]):
        if as_int(coeff) % order != 0:
            mask |= 1 << index
    return mask


def sorted_ints(raw: Any) -> list[int]:
    return sorted(as_int(item) for item in raw or [])


def sorted_strings(raw: Any) -> list[str]:
    return sorted(str(item) for item in raw or [])


def primary_work_items(worklist: dict[str, Any]) -> dict[int, dict[str, Any]]:
    items = {}
    for item in worklist.get("work_items") or []:
        if not isinstance(item, dict):
            continue
        transfer = as_int(item.get("transfer_index"), -1)
        if transfer not in PRIMARY_BACKFILL_TRANSFERS:
            continue
        if item.get("phase") != "direct_rank_backfill_export" or not bool(item.get("is_full_family_backfill")):
            continue
        items[transfer] = item
    return items


def contract_backfill_manifest(contract: dict[str, Any]) -> dict[int, dict[str, Any]]:
    manifest = ((contract.get("contract") or {}).get("direct_rank_backfill_manifest")) or []
    return {
        as_int(item.get("transfer_index"), -1): item
        for item in manifest
        if isinstance(item, dict) and as_int(item.get("transfer_index"), -1) in PRIMARY_BACKFILL_TRANSFERS
    }


def transfer_order(transfer: int) -> int:
    try:
        return PRIMARY_BACKFILL_TRANSFERS.index(transfer)
    except ValueError:
        return len(PRIMARY_BACKFILL_TRANSFERS)


def coeff_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        transfer_order(as_int(row.get("backfill_transfer_index"), -1)),
        as_int(row.get("source_form_support_mask"), -1),
        as_int(row.get("source_tier_code"), -1),
        as_int(row.get("source_transfer_index"), -1),
        as_int(row.get("form_hash_u64"), -1),
    )


def source_salt_delta(source_salts: list[int], backfill_salts: list[int]) -> dict[str, Any]:
    source_set = set(source_salts)
    backfill_set = set(backfill_salts)
    shared = sorted(source_set & backfill_set)
    if len(shared) != 1:
        return {
            "backfill_other_salt": None,
            "other_salt_delta": None,
            "record_other_salt": None,
            "shared_salt": None,
        }
    shared_salt = shared[0]
    source_other = sorted(source_set - {shared_salt})
    backfill_other = sorted(backfill_set - {shared_salt})
    if len(source_other) != 1 or len(backfill_other) != 1:
        return {
            "backfill_other_salt": None,
            "other_salt_delta": None,
            "record_other_salt": None,
            "shared_salt": shared_salt,
        }
    return {
        "backfill_other_salt": backfill_other[0],
        "other_salt_delta": backfill_other[0] - source_other[0],
        "record_other_salt": source_other[0],
        "shared_salt": shared_salt,
    }


def candidate_hash_u64(row: dict[str, Any], work_item: dict[str, Any]) -> int:
    payload = {
        "backfill_row_check_hash": work_item.get("row_check_hash"),
        "backfill_transfer_index": as_int(row.get("backfill_transfer_index"), -1),
        "form_hash_u64": as_int(row.get("form_hash_u64")),
        "source_form_support_mask": as_int(row.get("source_form_support_mask")),
        "source_tier": row.get("source_tier"),
        "source_transfer_index": as_int(row.get("source_transfer_index"), -1),
    }
    return int(hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16], 16)


def build_candidate_forms(
    sidecar_coeff: dict[str, Any],
    work_items: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(sorted(sidecar_coeff.get("coefficient_forms") or [], key=coeff_sort_key)):
        transfer = as_int(row.get("backfill_transfer_index"), -1)
        work_item = work_items.get(transfer) or {}
        coeffs = [as_int(value) % ORDER for value in row.get("coeffs") or []]
        while len(coeffs) < COEFF_COUNT:
            coeffs.append(0)
        coeffs = coeffs[:COEFF_COUNT]
        source_salts = sorted_ints(row.get("source_salts"))
        backfill_salts = sorted_ints(row.get("backfill_salts"))
        rows.append(
            {
                "backfill_row_check_hash": work_item.get("row_check_hash") or row.get("backfill_row_check_hash"),
                "backfill_row_check_hash_u64": as_int(work_item.get("row_check_hash_u64"))
                or as_int(row.get("backfill_row_check_hash_u64"))
                or digest_u64(row.get("backfill_row_check_hash")),
                "backfill_row_id": work_item.get("row_id"),
                "backfill_row_id_u64": as_int(work_item.get("row_id_u64")) or digest_u64(work_item.get("row_id")),
                "backfill_salts": backfill_salts,
                "backfill_transfer_index": transfer,
                "candidate_form_hash_u64": candidate_hash_u64(row, work_item),
                "candidate_form_index": index,
                "coeff_count": COEFF_COUNT,
                "coeffs": coeffs,
                "form_hash_u64": as_int(row.get("form_hash_u64")),
                "rhs": as_int(row.get("rhs")) % ORDER,
                "salt_delta": source_salt_delta(source_salts, backfill_salts),
                "source_certificate_hash": row.get("source_certificate_hash"),
                "source_certificate_hash_u64": as_int(row.get("source_certificate_hash_u64")),
                "source_derived_secret": as_int(row.get("source_derived_secret")),
                "source_form_support_mask": as_int(row.get("source_form_support_mask")),
                "source_public_key_verified": bool(row.get("source_public_key_verified")),
                "source_row_keys": sorted_strings(row.get("source_row_keys")),
                "source_salts": source_salts,
                "source_tier": row.get("source_tier"),
                "source_tier_code": as_int(row.get("source_tier_code")),
                "source_transfer_index": as_int(row.get("source_transfer_index"), -1),
                "terms": [as_int(value) for value in row.get("terms") or []],
                "worker_acceptance_gate": "must_reemit_and_verify_backfill_direct_rank_row",
            }
        )
    return rows


def build_mask_lanes(forms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in forms:
        by_key[(as_int(row.get("backfill_transfer_index"), -1), as_int(row.get("source_form_support_mask")))].append(row)

    lanes = []
    for transfer in PRIMARY_BACKFILL_TRANSFERS:
        for mask in FULL_FAMILY_MASKS:
            rows = sorted(by_key.get((transfer, mask), []), key=lambda item: as_int(item.get("candidate_form_index"), -1))
            tier_counts = Counter(str(row.get("source_tier")) for row in rows)
            tier = rows[0].get("source_tier") if rows else None
            source_salts = rows[0].get("source_salts") if rows else []
            backfill_salts = rows[0].get("backfill_salts") if rows else []
            lanes.append(
                {
                    "backfill_transfer_index": transfer,
                    "candidate_form_count": len(rows),
                    "candidate_form_indices": [as_int(row.get("candidate_form_index"), -1) for row in rows],
                    "candidate_form_start": as_int(rows[0].get("candidate_form_index"), -1) if rows else -1,
                    "family_mask": mask,
                    "lane_class": LANE_CLASSES.get(str(tier), "unknown_lane"),
                    "lane_index": len(lanes),
                    "salt_delta": source_salt_delta(sorted_ints(source_salts), sorted_ints(backfill_salts)),
                    "source_secret_set": sorted({as_int(row.get("source_derived_secret"), -1) for row in rows}),
                    "source_tier": tier,
                    "source_tier_code": TIER_CODES.get(str(tier), 0),
                    "source_tier_counts": dict(sorted(tier_counts.items())),
                    "source_transfers": sorted({as_int(row.get("source_transfer_index"), -1) for row in rows}),
                }
            )
    return lanes


def row_check_for_manifest_item(item: dict[str, Any]) -> dict[str, Any]:
    full_rows = [row for row in item.get("row_checks") or [] if bool(row.get("is_full_family"))]
    return full_rows[0] if full_rows else {}


def gate_status_for_transfer(transfer: int, tier_counts: Counter[str]) -> str:
    if transfer == 9981 and tier_counts.get("same_row_key_exact_support", 0) >= 4:
        return "BACKFILL_SAME_ROW_KEY_REPLAY_FIRST"
    if transfer == 9943 and tier_counts.get("one_salt_neighbor_exact_support", 0) >= 4:
        return "BACKFILL_SALT_NEIGHBOR_MUTATION_SECOND"
    return "BACKFILL_DIRECT_EXPORT_REQUIRED"


def build_transfer_gates(
    forms: list[dict[str, Any]],
    mask_lanes: list[dict[str, Any]],
    work_items: dict[int, dict[str, Any]],
    manifest: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    gates = []
    by_transfer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    lanes_by_transfer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in forms:
        by_transfer[as_int(row.get("backfill_transfer_index"), -1)].append(row)
    for lane in mask_lanes:
        lanes_by_transfer[as_int(lane.get("backfill_transfer_index"), -1)].append(lane)

    for gate_index, transfer in enumerate(PRIMARY_BACKFILL_TRANSFERS):
        rows = by_transfer.get(transfer, [])
        work_item = work_items.get(transfer) or {}
        item = manifest.get(transfer) or {}
        manifest_full_row = row_check_for_manifest_item(item)
        tier_counts = Counter(str(row.get("source_tier")) for row in rows)
        source_secrets = sorted({as_int(row.get("source_derived_secret"), -1) for row in rows})
        gates.append(
            {
                "accepted_backfill_export_count": 0,
                "backfill_row_check_hash": work_item.get("row_check_hash"),
                "backfill_row_check_hash_u64": as_int(work_item.get("row_check_hash_u64")) or digest_u64(
                    work_item.get("row_check_hash")
                ),
                "backfill_row_id": work_item.get("row_id"),
                "backfill_row_id_u64": as_int(work_item.get("row_id_u64")) or digest_u64(work_item.get("row_id")),
                "candidate_form_count": len(rows),
                "covered_family_masks": sorted({as_int(row.get("source_form_support_mask")) for row in rows}),
                "direct_status": work_item.get("direct_status"),
                "family_mask_count": len(lanes_by_transfer.get(transfer, [])),
                "full_family_row_id_matches_contract": manifest_full_row.get("row_id") == work_item.get("row_id"),
                "gate_index": gate_index,
                "gate_status": gate_status_for_transfer(transfer, tier_counts),
                "min_direct_ops_over_rho": item.get("min_direct_ops_over_rho"),
                "min_direct_ops_over_rho_scaled": int(round(float(item.get("min_direct_ops_over_rho") or 0.0) * 100000000)),
                "one_salt_neighbor_form_count": tier_counts.get("one_salt_neighbor_exact_support", 0),
                "primary_order": gate_index,
                "required_output": {
                    "must_match_backfill_row_check_hash": work_item.get("row_check_hash"),
                    "must_match_backfill_row_id": work_item.get("row_id"),
                    "must_set_backfill_transfer_index": transfer,
                    "must_verify_public_key": True,
                    "must_write_direct_rank_export": True,
                },
                "same_row_key_form_count": tier_counts.get("same_row_key_exact_support", 0),
                "source_secret_count": len(source_secrets),
                "source_secrets": source_secrets,
                "support_span_only_form_count": tier_counts.get("support_span_only", 0),
                "target": work_item.get("target"),
                "transfer_index": transfer,
            }
        )
    return gates


def validate_inputs(sidecar_coeff: dict[str, Any], contract: dict[str, Any], worklist: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if sidecar_coeff.get("claim_status") != "FFE_SHARP_LANE_SIDECAR_COEFF_PREFLIGHT_READY":
        failures.append({"code": "sidecar_coeff_not_ready", "claim_status": sidecar_coeff.get("claim_status")})
    if sidecar_coeff.get("failures"):
        failures.append({"code": "sidecar_coeff_has_failures", "failures": sidecar_coeff.get("failures")})
    if (sidecar_coeff.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "sidecar_coeff_not_verified"})
    if contract.get("claim_status") != "FFE_SHARP_LANE_KERNEL_CONTRACT_READY":
        failures.append({"code": "contract_not_ready", "claim_status": contract.get("claim_status")})
    if contract.get("failures"):
        failures.append({"code": "contract_has_failures", "failures": contract.get("failures")})
    if worklist.get("claim_status") != "FFE_SHARP_LANE_EXECUTION_WORKLIST_READY":
        failures.append({"code": "worklist_not_ready", "claim_status": worklist.get("claim_status")})
    if worklist.get("failures"):
        failures.append({"code": "worklist_has_failures", "failures": worklist.get("failures")})
    return failures


def validate_gate_shape(
    forms: list[dict[str, Any]],
    mask_lanes: list[dict[str, Any]],
    transfer_gates: list[dict[str, Any]],
    work_items: dict[int, dict[str, Any]],
    manifest: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(work_items) != EXPECTED_TRANSFER_GATE_COUNT:
        failures.append({"code": "primary_full_family_work_item_count_mismatch", "observed": len(work_items)})
    if len(manifest) != EXPECTED_TRANSFER_GATE_COUNT:
        failures.append({"code": "primary_contract_manifest_count_mismatch", "observed": len(manifest)})
    if len(forms) != EXPECTED_FORM_COUNT:
        failures.append({"code": "candidate_form_count_mismatch", "observed": len(forms)})
    if len(mask_lanes) != EXPECTED_MASK_LANE_COUNT:
        failures.append({"code": "mask_lane_count_mismatch", "observed": len(mask_lanes)})
    if len(transfer_gates) != EXPECTED_TRANSFER_GATE_COUNT:
        failures.append({"code": "transfer_gate_count_mismatch", "observed": len(transfer_gates)})
    if [as_int(gate.get("transfer_index"), -1) for gate in transfer_gates] != PRIMARY_BACKFILL_TRANSFERS:
        failures.append({"code": "transfer_gate_order_mismatch"})

    for row in forms:
        coeffs = [as_int(value) for value in row.get("coeffs") or []]
        if len(coeffs) != COEFF_COUNT or as_int(row.get("coeff_count")) != COEFF_COUNT:
            failures.append({"code": "coefficient_width_mismatch", "candidate_form_index": row.get("candidate_form_index")})
        if support_mask_from_coeffs(coeffs) != as_int(row.get("source_form_support_mask")):
            failures.append({"code": "candidate_support_mask_mismatch", "candidate_form_index": row.get("candidate_form_index")})
        if as_int(row.get("source_tier_code")) != TIER_CODES.get(str(row.get("source_tier")), 0):
            failures.append({"code": "candidate_tier_code_mismatch", "candidate_form_index": row.get("candidate_form_index")})
        if as_int(row.get("backfill_row_check_hash_u64")) == 0 or as_int(row.get("candidate_form_hash_u64")) == 0:
            failures.append({"code": "candidate_hash_zero", "candidate_form_index": row.get("candidate_form_index")})
        if row.get("worker_acceptance_gate") != "must_reemit_and_verify_backfill_direct_rank_row":
            failures.append({"code": "candidate_acceptance_gate_missing", "candidate_form_index": row.get("candidate_form_index")})

    for lane in mask_lanes:
        if as_int(lane.get("candidate_form_count")) != 2:
            failures.append({"code": "mask_lane_form_count_mismatch", "lane_index": lane.get("lane_index")})
        if as_int(lane.get("family_mask")) not in FULL_FAMILY_MASKS:
            failures.append({"code": "mask_lane_unknown_family_mask", "lane_index": lane.get("lane_index")})
        if as_int(lane.get("source_tier_code")) not in set(TIER_CODES.values()):
            failures.append({"code": "mask_lane_unknown_tier", "lane_index": lane.get("lane_index")})

    for gate in transfer_gates:
        if gate.get("direct_status") != "direct_certificate_missing":
            failures.append({"code": "gate_not_missing_direct_certificate", "transfer_index": gate.get("transfer_index")})
        if as_int(gate.get("accepted_backfill_export_count")) != 0:
            failures.append({"code": "gate_claims_backfill_export", "transfer_index": gate.get("transfer_index")})
        if gate.get("covered_family_masks") != FULL_FAMILY_MASKS:
            failures.append({"code": "gate_family_mask_coverage_mismatch", "transfer_index": gate.get("transfer_index")})
        if not bool(gate.get("full_family_row_id_matches_contract")):
            failures.append({"code": "gate_full_family_row_id_not_in_contract", "transfer_index": gate.get("transfer_index")})

    tier_counts = Counter(str(row.get("source_tier")) for row in forms)
    if tier_counts.get("same_row_key_exact_support", 0) != 4:
        failures.append({"code": "same_row_key_candidate_form_count_mismatch"})
    if tier_counts.get("one_salt_neighbor_exact_support", 0) != 4:
        failures.append({"code": "one_salt_neighbor_candidate_form_count_mismatch"})
    if tier_counts.get("support_span_only", 0) != 4:
        failures.append({"code": "support_span_only_candidate_form_count_mismatch"})
    return failures


def c_u64_array(values: list[int]) -> str:
    return "{" + ", ".join(f"{as_int(value)}ULL" for value in values) + "}"


def render_c_header(forms: list[dict[str, Any]], lanes: list[dict[str, Any]], gates: list[dict[str, Any]]) -> str:
    form_lines = []
    for row in forms:
        coeffs = [as_int(value) for value in row.get("coeffs") or []]
        while len(coeffs) < COEFF_COUNT:
            coeffs.append(0)
        form_lines.append(
            "  {"
            f"{as_int(row.get('candidate_form_index'))}ULL, "
            f"{as_int(row.get('backfill_transfer_index'))}ULL, "
            f"{as_int(row.get('backfill_row_id_u64'))}ULL, "
            f"{as_int(row.get('backfill_row_check_hash_u64'))}ULL, "
            f"{as_int(row.get('source_transfer_index'))}ULL, "
            f"{as_int(row.get('source_tier_code'))}ULL, "
            f"{as_int(row.get('source_form_support_mask'))}ULL, "
            f"{as_int(row.get('form_hash_u64'))}ULL, "
            f"{as_int(row.get('candidate_form_hash_u64'))}ULL, "
            f"{as_int(row.get('source_derived_secret'))}ULL, "
            f"{as_int(row.get('source_certificate_hash_u64'))}ULL, "
            f"{as_int(row.get('rhs'))}ULL, "
            f"{c_u64_array(coeffs[:COEFF_COUNT])}"
            "},"
        )
    lane_lines = []
    for lane in lanes:
        lane_lines.append(
            "  {"
            f"{as_int(lane.get('lane_index'))}ULL, "
            f"{as_int(lane.get('backfill_transfer_index'))}ULL, "
            f"{as_int(lane.get('family_mask'))}ULL, "
            f"{as_int(lane.get('source_tier_code'))}ULL, "
            f"{as_int(lane.get('candidate_form_start'))}ULL, "
            f"{as_int(lane.get('candidate_form_count'))}ULL"
            "},"
        )
    gate_lines = []
    for gate in gates:
        gate_lines.append(
            "  {"
            f"{as_int(gate.get('gate_index'))}ULL, "
            f"{as_int(gate.get('transfer_index'))}ULL, "
            f"{as_int(gate.get('backfill_row_id_u64'))}ULL, "
            f"{as_int(gate.get('backfill_row_check_hash_u64'))}ULL, "
            f"{as_int(gate.get('candidate_form_count'))}ULL, "
            f"{as_int(gate.get('family_mask_count'))}ULL, "
            f"{as_int(gate.get('same_row_key_form_count'))}ULL, "
            f"{as_int(gate.get('one_salt_neighbor_form_count'))}ULL, "
            f"{as_int(gate.get('support_span_only_form_count'))}ULL, "
            f"{as_int(gate.get('source_secret_count'))}ULL, "
            f"{as_int(gate.get('accepted_backfill_export_count'))}ULL, "
            f"{as_int(gate.get('min_direct_ops_over_rho_scaled'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_FFE_SHARP_LANE_BACKFILL_TRANSFER_GATE_H
#define LOW_TERM_TOTAL2_FFE_SHARP_LANE_BACKFILL_TRANSFER_GATE_H

#include <stdint.h>

#define SELECTED13_BACKFILL_GATE_FORM_COUNT {len(forms)}
#define SELECTED13_BACKFILL_GATE_MASK_LANE_COUNT {len(lanes)}
#define SELECTED13_BACKFILL_GATE_TRANSFER_COUNT {len(gates)}
#define SELECTED13_BACKFILL_GATE_COEFF_WIDTH {COEFF_COUNT}
#define SELECTED13_BACKFILL_GATE_ACCEPTED_EXPORT_COUNT 0

#define SELECTED13_SOURCE_TIER_SAME_ROW_KEY 1ULL
#define SELECTED13_SOURCE_TIER_ONE_SALT_NEIGHBOR 2ULL
#define SELECTED13_SOURCE_TIER_SUPPORT_SPAN_ONLY 3ULL

typedef struct {{
  uint64_t candidate_form_index;
  uint64_t backfill_transfer_index;
  uint64_t backfill_row_id_u64;
  uint64_t backfill_row_check_hash_u64;
  uint64_t source_transfer_index;
  uint64_t source_tier_code;
  uint64_t support_mask;
  uint64_t source_form_hash_u64;
  uint64_t candidate_form_hash_u64;
  uint64_t source_derived_secret;
  uint64_t source_certificate_hash_u64;
  uint64_t rhs;
  uint64_t coeffs[SELECTED13_BACKFILL_GATE_COEFF_WIDTH];
}} selected13_backfill_gate_form_t;

typedef struct {{
  uint64_t lane_index;
  uint64_t backfill_transfer_index;
  uint64_t family_mask;
  uint64_t source_tier_code;
  uint64_t candidate_form_start;
  uint64_t candidate_form_count;
}} selected13_backfill_mask_lane_t;

typedef struct {{
  uint64_t gate_index;
  uint64_t backfill_transfer_index;
  uint64_t backfill_row_id_u64;
  uint64_t backfill_row_check_hash_u64;
  uint64_t candidate_form_count;
  uint64_t family_mask_count;
  uint64_t same_row_key_form_count;
  uint64_t one_salt_neighbor_form_count;
  uint64_t support_span_only_form_count;
  uint64_t source_secret_count;
  uint64_t accepted_backfill_export_count;
  uint64_t min_direct_ops_over_rho_scaled;
}} selected13_backfill_transfer_gate_t;

static const selected13_backfill_gate_form_t SELECTED13_BACKFILL_GATE_FORMS[] = {{
{chr(10).join(form_lines)}
}};

static const selected13_backfill_mask_lane_t SELECTED13_BACKFILL_MASK_LANES[] = {{
{chr(10).join(lane_lines)}
}};

static const selected13_backfill_transfer_gate_t SELECTED13_BACKFILL_TRANSFER_GATES[] = {{
{chr(10).join(gate_lines)}
}};

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

static uint64_t support_mask_from_coeffs(const uint64_t coeffs[SELECTED13_BACKFILL_GATE_COEFF_WIDTH]) {{
  uint64_t mask = 0;
  for (uint64_t i = 1; i < SELECTED13_BACKFILL_GATE_COEFF_WIDTH; i++) {{
    if (coeffs[i] % 11779ULL != 0) {{
      mask |= (1ULL << (i - 1));
    }}
  }}
  return mask;
}}

int main(void) {{
  uint64_t failure_count = 0;
  uint64_t transfer9981_form_count = 0;
  uint64_t transfer9943_form_count = 0;
  uint64_t same_row_key_form_count = 0;
  uint64_t one_salt_neighbor_form_count = 0;
  uint64_t support_span_only_form_count = 0;
  uint64_t accepted_export_count = 0;
  const size_t form_count =
      sizeof(SELECTED13_BACKFILL_GATE_FORMS) / sizeof(SELECTED13_BACKFILL_GATE_FORMS[0]);
  const size_t lane_count =
      sizeof(SELECTED13_BACKFILL_MASK_LANES) / sizeof(SELECTED13_BACKFILL_MASK_LANES[0]);
  const size_t gate_count =
      sizeof(SELECTED13_BACKFILL_TRANSFER_GATES) / sizeof(SELECTED13_BACKFILL_TRANSFER_GATES[0]);

  if (form_count != SELECTED13_BACKFILL_GATE_FORM_COUNT) failure_count++;
  if (lane_count != SELECTED13_BACKFILL_GATE_MASK_LANE_COUNT) failure_count++;
  if (gate_count != SELECTED13_BACKFILL_GATE_TRANSFER_COUNT) failure_count++;
  if (SELECTED13_BACKFILL_GATE_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;

  for (size_t i = 0; i < form_count; i++) {{
    const selected13_backfill_gate_form_t *form = &SELECTED13_BACKFILL_GATE_FORMS[i];
    if (form->candidate_form_index != i) failure_count++;
    if (form->backfill_row_id_u64 == 0 || form->backfill_row_check_hash_u64 == 0 ||
        form->source_form_hash_u64 == 0 || form->candidate_form_hash_u64 == 0 ||
        form->source_certificate_hash_u64 == 0) {{
      failure_count++;
    }}
    if (support_mask_from_coeffs(form->coeffs) != form->support_mask) failure_count++;
    if (form->rhs >= 11779ULL) failure_count++;
    if (form->backfill_transfer_index == 9981ULL) {{
      transfer9981_form_count++;
    }} else if (form->backfill_transfer_index == 9943ULL) {{
      transfer9943_form_count++;
    }} else {{
      failure_count++;
    }}
    if (form->source_tier_code == SELECTED13_SOURCE_TIER_SAME_ROW_KEY) {{
      same_row_key_form_count++;
    }} else if (form->source_tier_code == SELECTED13_SOURCE_TIER_ONE_SALT_NEIGHBOR) {{
      one_salt_neighbor_form_count++;
    }} else if (form->source_tier_code == SELECTED13_SOURCE_TIER_SUPPORT_SPAN_ONLY) {{
      support_span_only_form_count++;
    }} else {{
      failure_count++;
    }}
  }}

  for (size_t i = 0; i < lane_count; i++) {{
    const selected13_backfill_mask_lane_t *lane = &SELECTED13_BACKFILL_MASK_LANES[i];
    if (lane->lane_index != i) failure_count++;
    if (lane->candidate_form_count != 2ULL) failure_count++;
    if (lane->candidate_form_start + lane->candidate_form_count > form_count) failure_count++;
  }}

  for (size_t i = 0; i < gate_count; i++) {{
    const selected13_backfill_transfer_gate_t *gate = &SELECTED13_BACKFILL_TRANSFER_GATES[i];
    if (gate->gate_index != i) failure_count++;
    if (gate->candidate_form_count != 6ULL) failure_count++;
    if (gate->family_mask_count != 3ULL) failure_count++;
    if (gate->accepted_backfill_export_count != 0ULL) failure_count++;
    accepted_export_count += gate->accepted_backfill_export_count;
  }}

  if (transfer9981_form_count != 6ULL || transfer9943_form_count != 6ULL) failure_count++;
  if (same_row_key_form_count != 4ULL) failure_count++;
  if (one_salt_neighbor_form_count != 4ULL) failure_count++;
  if (support_span_only_form_count != 4ULL) failure_count++;
  if (accepted_export_count != 0ULL) failure_count++;

  printf("{{");
  printf("\\\"form_count\\\":%llu,", (unsigned long long)form_count);
  printf("\\\"mask_lane_count\\\":%llu,", (unsigned long long)lane_count);
  printf("\\\"transfer_gate_count\\\":%llu,", (unsigned long long)gate_count);
  printf("\\\"transfer9981_form_count\\\":%llu,", (unsigned long long)transfer9981_form_count);
  printf("\\\"transfer9943_form_count\\\":%llu,", (unsigned long long)transfer9943_form_count);
  printf("\\\"same_row_key_form_count\\\":%llu,", (unsigned long long)same_row_key_form_count);
  printf("\\\"one_salt_neighbor_form_count\\\":%llu,", (unsigned long long)one_salt_neighbor_form_count);
  printf("\\\"support_span_only_form_count\\\":%llu,", (unsigned long long)support_span_only_form_count);
  printf("\\\"accepted_export_count\\\":%llu,", (unsigned long long)accepted_export_count);
  printf("\\\"failure_count\\\":%llu", (unsigned long long)failure_count);
  printf("}}\\n");
  return failure_count == 0 ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path, compiler: str) -> dict[str, Any]:
    c_source = render_preflight_c(header_path.name)
    source_hash = hashlib.sha256(c_source.encode("utf-8")).hexdigest()
    temp_root = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path(tempfile.gettempdir())
    env = os.environ.copy()
    env["TMPDIR"] = str(temp_root)
    with tempfile.TemporaryDirectory(prefix="ecdlp_selected13_backfill_gate_", dir=str(temp_root)) as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "selected13_backfill_gate_preflight.c"
        exe_path = temp_path / "selected13_backfill_gate_preflight"
        c_path.write_text(c_source)
        command = [
            compiler,
            "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            str(header_path.parent),
            str(c_path),
            "-o",
            str(exe_path),
        ]
        compile_run = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
        if compile_run.returncode != 0:
            return {
                "c_source_sha256": source_hash,
                "compile_command": command,
                "compile_returncode": compile_run.returncode,
                "compile_stderr": compile_run.stderr,
                "compile_stdout": compile_run.stdout,
                "compiled": False,
                "executed": False,
            }
        native_run = subprocess.run([str(exe_path)], capture_output=True, text=True, check=False, env=env)
        try:
            native_summary = json.loads(native_run.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            native_summary = None
        return {
            "c_source_sha256": source_hash,
            "compile_command": command,
            "compile_returncode": compile_run.returncode,
            "compile_stderr": compile_run.stderr,
            "compile_stdout": compile_run.stdout,
            "compiled": True,
            "executed": True,
            "native_summary": native_summary,
            "run_returncode": native_run.returncode,
            "run_stderr": native_run.stderr,
            "run_stdout": native_run.stdout,
        }


def compare_native(native: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not native.get("compiled"):
        failures.append({"code": "native_preflight_compile_failed", "compile_stderr": native.get("compile_stderr")})
        return failures
    if as_int(native.get("run_returncode"), -1) != 0:
        failures.append({"code": "native_preflight_run_failed", "run_stdout": native.get("run_stdout")})
    summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    if not summary:
        failures.append({"code": "native_preflight_stdout_not_json", "run_stdout": native.get("run_stdout")})
        return failures
    if as_int(summary.get("failure_count"), -1) != 0:
        failures.append({"code": "native_preflight_reported_failures", "native_summary": summary})
    return failures


def summarize(forms: list[dict[str, Any]], lanes: list[dict[str, Any]], gates: list[dict[str, Any]], native: dict[str, Any]) -> dict[str, Any]:
    native_summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    tier_counts = Counter(str(row.get("source_tier")) for row in forms)
    return {
        "accepted_backfill_export_count": sum(as_int(gate.get("accepted_backfill_export_count")) for gate in gates),
        "candidate_form_count": len(forms),
        "mask_lane_count": len(lanes),
        "native_preflight_failure_count": as_int(native_summary.get("failure_count"), -1),
        "native_preflight_verified": as_int(native_summary.get("failure_count"), -1) == 0,
        "one_salt_neighbor_form_count": tier_counts.get("one_salt_neighbor_exact_support", 0),
        "primary_backfill_transfers": PRIMARY_BACKFILL_TRANSFERS,
        "same_row_key_form_count": tier_counts.get("same_row_key_exact_support", 0),
        "support_span_only_form_count": tier_counts.get("support_span_only", 0),
        "transfer_gate_count": len(gates),
        "transfer_gate_interpretation": (
            "Sidecar forms are now bound to missing full-family backfill row hashes; "
            "a worker must still re-emit and verify direct/rank rows for 9981 and 9943."
        ),
        "worker_required_direct_export_count": len(gates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-coeff", type=Path, default=DEFAULT_SIDECAR_COEFF)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--worklist", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--no-c-header", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sidecar_coeff = load_json(args.sidecar_coeff)
    contract = load_json(args.contract)
    worklist = load_json(args.worklist)

    failures = validate_inputs(sidecar_coeff, contract, worklist)
    work_items = primary_work_items(worklist)
    manifest = contract_backfill_manifest(contract)
    forms = build_candidate_forms(sidecar_coeff, work_items)
    lanes = build_mask_lanes(forms)
    gates = build_transfer_gates(forms, lanes, work_items, manifest)
    failures.extend(validate_gate_shape(forms, lanes, gates, work_items, manifest))

    native: dict[str, Any] = {"compiled": False, "executed": False, "native_summary": {}}
    if not args.no_c_header:
        args.c_header_out.parent.mkdir(parents=True, exist_ok=True)
        args.c_header_out.write_text(render_c_header(forms, lanes, gates))
        native = run_native_preflight(args.c_header_out, args.cc)
        failures.extend(compare_native(native))

    verified = not failures
    payload = {
        "artifacts": {
            "c_header": None if args.no_c_header else str(args.c_header_out),
            "contract": str(args.contract),
            "sidecar_coeff": str(args.sidecar_coeff),
            "worklist": str(args.worklist),
        },
        "candidate_forms": forms,
        "claim_status": (
            "FFE_SHARP_LANE_BACKFILL_TRANSFER_GATE_READY"
            if verified
            else "FFE_SHARP_LANE_BACKFILL_TRANSFER_GATE_FAILED_CHECK"
        ),
        "created_at": now_iso(),
        "failures": failures,
        "honesty_boundary": [
            "This binds source coefficient forms to missing backfill row hashes for worker input only.",
            "It does not accept source coefficients as a backfill direct/rank certificate.",
            "It does not evaluate summation polynomials, solve finite-field equations, solve ECDLP, or claim a Pollard-rho speedup.",
        ],
        "mask_lanes": lanes,
        "native_preflight": native,
        "parameters": {
            "coefficient_width": COEFF_COUNT,
            "full_family_masks": FULL_FAMILY_MASKS,
            "primary_backfill_transfers": PRIMARY_BACKFILL_TRANSFERS,
            "source_tier_codes": TIER_CODES,
        },
        "schema": SCHEMA,
        "summary": summarize(forms, lanes, gates, native),
        "transfer_gates": gates,
    }
    payload["summary"]["verified"] = verified
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "claim_status": payload["claim_status"],
                "failures": failures,
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
