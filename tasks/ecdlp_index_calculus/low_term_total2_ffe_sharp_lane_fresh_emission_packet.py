#!/usr/bin/env python3
"""Emit the selected13 fresh FFE row-emission packet.

The replay-algebra gate proved that copied source coefficient systems only
replay old source secrets or become inconsistent.  This packet is the next
worker-facing boundary: it binds the priority backfill rows, row hashes, family
lanes, source hints, and replay-negative controls that a lower-level
FFE/summation-polynomial worker must satisfy with fresh direct/rank row
emission.

This is still a pre-execution contract.  It does not evaluate summation
polynomials, export backfill rows, derive an ECDLP scalar, or claim a
Pollard-rho speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_ffe_sharp_lane_fresh_emission_packet.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_CONTRACT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9999_probe.json"
)
DEFAULT_WORKLIST = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_execution_worklist_selected13_9696_9999_probe.json"
)
DEFAULT_BACKFILL_GATE = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_backfill_transfer_gate_selected13_9696_9999_probe.json"
)
DEFAULT_REPLAY_ALGEBRA = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_backfill_replay_algebra_selected13_9696_9999_probe.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_fresh_emission_packet_selected13_9696_9999_probe.json"
)
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_fresh_emission_packet_selected13_9696_9999_probe.h"
)

PRIMARY_TRANSFERS = [9981, 9943]
EXPECTED_TARGET_COUNT = 2
EXPECTED_ROW_SLOT_COUNT = 6
EXPECTED_FAMILY_LANE_COUNT = 6
EXPECTED_ROW_SLOTS_PER_TARGET = 3
EXPECTED_FAMILY_LANES_PER_TARGET = 3

RESULT_CODES = {
    "SOURCE_SECRET_REPLAY_ONLY": 1,
    "SOURCE_FORM_SYSTEM_INCONSISTENT": 2,
    "NO_UNIQUE_SECRET_FROM_REPLAY_FORMS": 3,
    "UNVERIFIED_NEW_LINEAR_DERIVATION": 4,
}
SOURCE_TIER_CODES = {
    "same_row_key_exact_support": 1,
    "one_salt_neighbor_exact_support": 2,
    "support_span_only": 3,
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
    if raw is None:
        return 0
    return int(hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:16], 16)


def family_terms(mask: int) -> list[int]:
    return [index for index in range(64) if mask & (1 << index)]


def stable_hash_u64(material: Any) -> int:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def transfer_order(transfer: int) -> int:
    try:
        return PRIMARY_TRANSFERS.index(transfer)
    except ValueError:
        return len(PRIMARY_TRANSFERS)


def direct_backfill_by_transfer(contract: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        as_int(item.get("transfer_index"), -1): item
        for item in (contract.get("contract") or {}).get("direct_rank_backfill_manifest") or []
        if isinstance(item, dict)
    }


def work_items_by_row_id(worklist: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("row_id")): item
        for item in worklist.get("work_items") or []
        if isinstance(item, dict) and item.get("row_id") is not None
    }


def transfer_gates_by_transfer(backfill_gate: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        as_int(item.get("transfer_index"), -1): item
        for item in backfill_gate.get("transfer_gates") or []
        if isinstance(item, dict)
    }


def mask_lanes_by_transfer(backfill_gate: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for lane in backfill_gate.get("mask_lanes") or []:
        if not isinstance(lane, dict):
            continue
        out.setdefault(as_int(lane.get("backfill_transfer_index"), -1), []).append(lane)
    for lanes in out.values():
        lanes.sort(key=lambda item: as_int(item.get("family_mask")))
    return out


def replay_transfer_cases(replay: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        as_int(case.get("backfill_transfer_index"), -1): case
        for case in replay.get("replay_cases") or []
        if isinstance(case, dict) and case.get("case_kind") == "transfer_gate"
    }


def replay_mask_cases(replay: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (as_int(case.get("backfill_transfer_index"), -1), as_int(case.get("family_mask"))): case
        for case in replay.get("replay_cases") or []
        if isinstance(case, dict) and case.get("case_kind") == "mask_lane"
    }


def row_slots_for_target(
    queue_item: dict[str, Any],
    work_item_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    slots = []
    for offset, row in enumerate(queue_item.get("row_checks") or []):
        row_id = str(row.get("row_id") or "")
        work_item = work_item_index.get(row_id) or {}
        slots.append(
            {
                "direct_status": row.get("direct_status"),
                "is_full_family": bool(row.get("is_full_family")),
                "matched_families": row.get("matched_families") or [],
                "packet_index": as_int(work_item.get("packet_index"), -1),
                "packet_row_offset": as_int(work_item.get("packet_row_offset"), offset),
                "requires_direct_rank_export": bool(work_item.get("requires_direct_rank_export")),
                "row_check_hash": row.get("row_check_hash"),
                "row_check_hash_u64": as_int(work_item.get("row_check_hash_u64")) or digest_u64(row.get("row_check_hash")),
                "row_id": row_id,
                "row_id_u64": as_int(work_item.get("row_id_u64")) or digest_u64(row_id),
                "selected_term_support": row.get("selected_term_support") or [],
                "selector": row.get("selector"),
                "top_k": as_int(row.get("top_k")),
                "work_item_index": as_int(work_item.get("work_item_index"), -1),
            }
        )
    return sorted(slots, key=lambda item: (not item["is_full_family"], item["packet_row_offset"], item["row_id"]))


def build_family_lane(
    lane: dict[str, Any],
    replay_case: dict[str, Any] | None,
) -> dict[str, Any]:
    source_tier = str(lane.get("source_tier") or "")
    family_mask = as_int(lane.get("family_mask"))
    result_status = str((replay_case or {}).get("result_status") or "")
    return {
        "candidate_form_count": as_int(lane.get("candidate_form_count")),
        "candidate_form_indices": [as_int(item) for item in lane.get("candidate_form_indices") or []],
        "family_mask": family_mask,
        "family_terms": family_terms(family_mask),
        "fresh_source_solve_required": source_tier == "support_span_only",
        "lane_class": lane.get("lane_class"),
        "lane_hash_u64": stable_hash_u64(
            {
                "candidate_form_indices": lane.get("candidate_form_indices"),
                "family_mask": family_mask,
                "source_tier": source_tier,
                "transfer": lane.get("backfill_transfer_index"),
            }
        ),
        "lane_index": as_int(lane.get("lane_index"), -1),
        "one_salt_neighbor_hint": source_tier == "one_salt_neighbor_exact_support",
        "replay_result_code": RESULT_CODES.get(result_status, 0),
        "replay_result_status": result_status,
        "same_row_key_hint": source_tier == "same_row_key_exact_support",
        "source_secret_count": len(lane.get("source_secret_set") or []),
        "source_secret_set": [as_int(item) for item in lane.get("source_secret_set") or []],
        "source_tier": source_tier,
        "source_tier_code": SOURCE_TIER_CODES.get(source_tier, 0),
        "source_transfers": [as_int(item) for item in lane.get("source_transfers") or []],
        "worker_action": (
            "fresh_support_solve_and_emit_direct_rank_row"
            if source_tier == "support_span_only"
            else "fresh_reemit_relation_against_backfill_row_hash"
        ),
    }


def build_emission_target(
    transfer: int,
    queue_item: dict[str, Any],
    gate: dict[str, Any],
    lanes: list[dict[str, Any]],
    replay_case: dict[str, Any],
    replay_mask_index: dict[tuple[int, int], dict[str, Any]],
    work_item_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row_slots = row_slots_for_target(queue_item, work_item_index)
    full_family_slots = [slot for slot in row_slots if slot.get("is_full_family")]
    full_family = full_family_slots[0] if full_family_slots else {}
    family_lanes = [
        build_family_lane(lane, replay_mask_index.get((transfer, as_int(lane.get("family_mask")))))
        for lane in lanes
    ]
    salts = [as_int(item) for item in (queue_item.get("public_first_pass") or {}).get("salts") or []]
    return {
        "accepted_backfill_export_count": 0,
        "backfill_gate_status": gate.get("gate_status"),
        "backfill_row_check_hash": gate.get("backfill_row_check_hash"),
        "backfill_row_id": gate.get("backfill_row_id"),
        "candidate_priority": "first" if transfer_order(transfer) == 0 else "second",
        "family_lane_count": len(family_lanes),
        "family_lanes": family_lanes,
        "first_pass_id": queue_item.get("first_pass_id"),
        "fresh_emission_required": True,
        "full_family_row": full_family,
        "full_gate_replay_result": replay_case.get("result_status"),
        "full_gate_source_secret_set": replay_case.get("source_secret_set") or [],
        "group_id": queue_item.get("group_id"),
        "min_direct_ops_over_rho": queue_item.get("min_direct_ops_over_rho"),
        "priority_order": transfer_order(transfer),
        "public_first_pass": queue_item.get("public_first_pass"),
        "required_output": {
            "must_match_backfill_row_check_hash": gate.get("backfill_row_check_hash"),
            "must_match_backfill_row_id": gate.get("backfill_row_id"),
            "must_not_use_copied_source_form_solve": True,
            "must_recompute_candidate_equality": True,
            "must_set_backfill_transfer_index": transfer,
            "must_verify_public_key": True,
            "must_write_direct_rank_export": True,
        },
        "row_slot_count": len(row_slots),
        "row_slots": row_slots,
        "salt_gap": max(salts) - min(salts) if len(salts) == 2 else None,
        "source_secret_count": as_int(gate.get("source_secret_count")),
        "transfer_index": transfer,
        "worker_acceptance_gate": "fresh_ffe_summation_polynomial_direct_rank_export_only",
    }


def validate_sources(
    contract: dict[str, Any],
    worklist: dict[str, Any],
    gate: dict[str, Any],
    replay: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if contract.get("claim_status") != "FFE_SHARP_LANE_KERNEL_CONTRACT_READY":
        failures.append({"code": "kernel_contract_not_ready", "claim_status": contract.get("claim_status")})
    if contract.get("failures"):
        failures.append({"code": "kernel_contract_has_failures", "failures": contract.get("failures")})
    if worklist.get("claim_status") != "FFE_SHARP_LANE_EXECUTION_WORKLIST_READY":
        failures.append({"code": "execution_worklist_not_ready", "claim_status": worklist.get("claim_status")})
    if worklist.get("failures"):
        failures.append({"code": "execution_worklist_has_failures", "failures": worklist.get("failures")})
    if gate.get("claim_status") != "FFE_SHARP_LANE_BACKFILL_TRANSFER_GATE_READY":
        failures.append({"code": "backfill_gate_not_ready", "claim_status": gate.get("claim_status")})
    if gate.get("failures"):
        failures.append({"code": "backfill_gate_has_failures", "failures": gate.get("failures")})
    if as_int((gate.get("summary") or {}).get("accepted_backfill_export_count"), -1) != 0:
        failures.append({"code": "backfill_gate_claims_export", "summary": gate.get("summary")})
    if replay.get("claim_status") != "FFE_SHARP_LANE_BACKFILL_REPLAY_ALGEBRA_READY":
        failures.append({"code": "replay_algebra_not_ready", "claim_status": replay.get("claim_status")})
    if replay.get("failures"):
        failures.append({"code": "replay_algebra_has_failures", "failures": replay.get("failures")})
    if (replay.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "replay_algebra_not_verified", "summary": replay.get("summary")})
    return failures


def validate_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(targets) != EXPECTED_TARGET_COUNT:
        failures.append({"code": "target_count_mismatch", "observed": len(targets), "expected": EXPECTED_TARGET_COUNT})
    row_slot_count = sum(as_int(target.get("row_slot_count")) for target in targets)
    if row_slot_count != EXPECTED_ROW_SLOT_COUNT:
        failures.append({"code": "row_slot_count_mismatch", "observed": row_slot_count})
    family_lane_count = sum(as_int(target.get("family_lane_count")) for target in targets)
    if family_lane_count != EXPECTED_FAMILY_LANE_COUNT:
        failures.append({"code": "family_lane_count_mismatch", "observed": family_lane_count})
    for target in targets:
        transfer = as_int(target.get("transfer_index"), -1)
        if as_int(target.get("accepted_backfill_export_count"), -1) != 0:
            failures.append({"code": "target_claims_backfill_export", "transfer_index": transfer})
        if target.get("full_gate_replay_result") != "SOURCE_FORM_SYSTEM_INCONSISTENT":
            failures.append(
                {
                    "code": "full_gate_replay_not_negative_control",
                    "transfer_index": transfer,
                    "result": target.get("full_gate_replay_result"),
                }
            )
        if as_int(target.get("row_slot_count")) != EXPECTED_ROW_SLOTS_PER_TARGET:
            failures.append({"code": "row_slots_per_target_mismatch", "transfer_index": transfer})
        if as_int(target.get("family_lane_count")) != EXPECTED_FAMILY_LANES_PER_TARGET:
            failures.append({"code": "family_lanes_per_target_mismatch", "transfer_index": transfer})
        full = target.get("full_family_row") or {}
        if full.get("row_id") != target.get("backfill_row_id"):
            failures.append(
                {
                    "code": "full_family_row_id_mismatch",
                    "transfer_index": transfer,
                    "row_id": full.get("row_id"),
                    "gate_row_id": target.get("backfill_row_id"),
                }
            )
        if full.get("row_check_hash") != target.get("backfill_row_check_hash"):
            failures.append(
                {
                    "code": "full_family_row_hash_mismatch",
                    "transfer_index": transfer,
                    "row_check_hash": full.get("row_check_hash"),
                    "gate_hash": target.get("backfill_row_check_hash"),
                }
            )
        for slot in target.get("row_slots") or []:
            if slot.get("direct_status") != "direct_certificate_missing":
                failures.append({"code": "row_slot_not_missing", "transfer_index": transfer, "row_id": slot.get("row_id")})
            if slot.get("requires_direct_rank_export") is not True:
                failures.append(
                    {"code": "row_slot_not_marked_direct_rank_export", "transfer_index": transfer, "row_id": slot.get("row_id")}
                )
            if as_int(slot.get("row_check_hash_u64")) == 0:
                failures.append({"code": "row_slot_hash_missing", "transfer_index": transfer, "row_id": slot.get("row_id")})
        lane_masks = sorted(as_int(lane.get("family_mask")) for lane in target.get("family_lanes") or [])
        if lane_masks != [33, 17408, 34816]:
            failures.append({"code": "family_lane_mask_mismatch", "transfer_index": transfer, "lane_masks": lane_masks})
        for lane in target.get("family_lanes") or []:
            if lane.get("replay_result_status") != "SOURCE_SECRET_REPLAY_ONLY":
                failures.append(
                    {
                        "code": "family_lane_not_source_secret_negative_control",
                        "transfer_index": transfer,
                        "family_mask": lane.get("family_mask"),
                        "result": lane.get("replay_result_status"),
                    }
                )
            if as_int(lane.get("candidate_form_count")) <= 0:
                failures.append(
                    {
                        "code": "family_lane_without_source_forms",
                        "transfer_index": transfer,
                        "family_mask": lane.get("family_mask"),
                    }
                )
    return failures


def render_c_header(targets: list[dict[str, Any]]) -> str:
    target_rows = []
    family_rows = []
    for target in targets:
        full = target.get("full_family_row") or {}
        public = target.get("public_first_pass") or {}
        salts = [as_int(item) for item in public.get("salts") or []]
        while len(salts) < 2:
            salts.append(0)
        target_rows.append(
            "  {"
            f"{as_int(target.get('transfer_index'))}ULL, "
            f"{as_int(target.get('priority_order'))}ULL, "
            f"{as_int(target.get('row_slot_count'))}ULL, "
            f"{as_int(target.get('family_lane_count'))}ULL, "
            f"{as_int(full.get('row_id_u64'))}ULL, "
            f"{as_int(full.get('row_check_hash_u64'))}ULL, "
            f"{as_int(salts[0])}ULL, "
            f"{as_int(salts[1])}ULL, "
            f"{as_int(target.get('salt_gap'))}ULL, "
            f"{RESULT_CODES.get(str(target.get('full_gate_replay_result')), 0)}ULL, "
            f"{as_int(target.get('accepted_backfill_export_count'))}ULL, "
            "1ULL"
            "},"
        )
        for lane in target.get("family_lanes") or []:
            family_rows.append(
                "  {"
                f"{as_int(target.get('transfer_index'))}ULL, "
                f"{as_int(lane.get('family_mask'))}ULL, "
                f"{as_int(lane.get('source_tier_code'))}ULL, "
                f"{as_int(lane.get('source_secret_count'))}ULL, "
                f"{as_int(lane.get('candidate_form_count'))}ULL, "
                f"{as_int(lane.get('replay_result_code'))}ULL, "
                f"{1 if lane.get('same_row_key_hint') else 0}ULL, "
                f"{1 if lane.get('one_salt_neighbor_hint') else 0}ULL, "
                f"{1 if lane.get('fresh_source_solve_required') else 0}ULL, "
                f"{as_int(lane.get('lane_hash_u64'))}ULL"
                "},"
            )
    return f"""#ifndef LOW_TERM_TOTAL2_FFE_SHARP_LANE_FRESH_EMISSION_PACKET_H
#define LOW_TERM_TOTAL2_FFE_SHARP_LANE_FRESH_EMISSION_PACKET_H

#include <stdint.h>

#define SELECTED13_FRESH_EMISSION_TARGET_COUNT {len(targets)}
#define SELECTED13_FRESH_EMISSION_ROW_SLOT_COUNT {sum(as_int(target.get('row_slot_count')) for target in targets)}
#define SELECTED13_FRESH_EMISSION_FAMILY_LANE_COUNT {sum(as_int(target.get('family_lane_count')) for target in targets)}
#define SELECTED13_FRESH_EMISSION_ACCEPTED_EXPORT_COUNT 0

#define SELECTED13_REPLAY_RESULT_SOURCE_SECRET_ONLY 1ULL
#define SELECTED13_REPLAY_RESULT_INCONSISTENT 2ULL
#define SELECTED13_REPLAY_RESULT_NO_UNIQUE_SECRET 3ULL
#define SELECTED13_REPLAY_RESULT_UNVERIFIED_NEW_DERIVATION 4ULL

typedef struct {{
  uint64_t transfer_index;
  uint64_t priority_order;
  uint64_t row_slot_count;
  uint64_t family_lane_count;
  uint64_t full_family_row_id_u64;
  uint64_t full_family_row_hash_u64;
  uint64_t salt0;
  uint64_t salt1;
  uint64_t salt_gap;
  uint64_t full_gate_replay_result_code;
  uint64_t accepted_backfill_export_count;
  uint64_t fresh_emission_required;
}} selected13_fresh_emission_target_t;

typedef struct {{
  uint64_t transfer_index;
  uint64_t family_mask;
  uint64_t source_tier_code;
  uint64_t source_secret_count;
  uint64_t candidate_form_count;
  uint64_t replay_result_code;
  uint64_t same_row_key_hint;
  uint64_t one_salt_neighbor_hint;
  uint64_t fresh_source_solve_required;
  uint64_t lane_hash_u64;
}} selected13_fresh_emission_family_lane_t;

static const selected13_fresh_emission_target_t SELECTED13_FRESH_EMISSION_TARGETS[] = {{
{chr(10).join(target_rows)}
}};

static const selected13_fresh_emission_family_lane_t SELECTED13_FRESH_EMISSION_FAMILY_LANES[] = {{
{chr(10).join(family_rows)}
}};

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

int main(void) {{
  uint64_t failure_count = 0;
  uint64_t target_count =
      sizeof(SELECTED13_FRESH_EMISSION_TARGETS) / sizeof(SELECTED13_FRESH_EMISSION_TARGETS[0]);
  uint64_t lane_count =
      sizeof(SELECTED13_FRESH_EMISSION_FAMILY_LANES) / sizeof(SELECTED13_FRESH_EMISSION_FAMILY_LANES[0]);
  uint64_t row_slot_count = 0;
  uint64_t family_lane_count = 0;
  uint64_t accepted_export_count = 0;
  uint64_t fresh_required_count = 0;
  uint64_t full_gate_inconsistent_count = 0;
  uint64_t source_secret_lane_count = 0;

  if (target_count != SELECTED13_FRESH_EMISSION_TARGET_COUNT) failure_count++;
  if (lane_count != SELECTED13_FRESH_EMISSION_FAMILY_LANE_COUNT) failure_count++;
  if (SELECTED13_FRESH_EMISSION_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;

  for (size_t i = 0; i < target_count; i++) {{
    const selected13_fresh_emission_target_t *target = &SELECTED13_FRESH_EMISSION_TARGETS[i];
    row_slot_count += target->row_slot_count;
    family_lane_count += target->family_lane_count;
    accepted_export_count += target->accepted_backfill_export_count;
    fresh_required_count += target->fresh_emission_required;
    if (target->full_gate_replay_result_code == SELECTED13_REPLAY_RESULT_INCONSISTENT) {{
      full_gate_inconsistent_count++;
    }}
    if (target->row_slot_count != {EXPECTED_ROW_SLOTS_PER_TARGET}ULL) failure_count++;
    if (target->family_lane_count != {EXPECTED_FAMILY_LANES_PER_TARGET}ULL) failure_count++;
    if (target->full_family_row_id_u64 == 0ULL || target->full_family_row_hash_u64 == 0ULL) failure_count++;
    if (target->accepted_backfill_export_count != 0ULL) failure_count++;
    if (target->fresh_emission_required != 1ULL) failure_count++;
  }}
  for (size_t i = 0; i < lane_count; i++) {{
    const selected13_fresh_emission_family_lane_t *lane = &SELECTED13_FRESH_EMISSION_FAMILY_LANES[i];
    if (lane->family_mask == 0ULL || lane->candidate_form_count == 0ULL || lane->lane_hash_u64 == 0ULL) {{
      failure_count++;
    }}
    if (lane->replay_result_code == SELECTED13_REPLAY_RESULT_SOURCE_SECRET_ONLY) {{
      source_secret_lane_count++;
    }}
  }}

  if (row_slot_count != SELECTED13_FRESH_EMISSION_ROW_SLOT_COUNT) failure_count++;
  if (family_lane_count != SELECTED13_FRESH_EMISSION_FAMILY_LANE_COUNT) failure_count++;
  if (accepted_export_count != 0ULL) failure_count++;
  if (fresh_required_count != SELECTED13_FRESH_EMISSION_TARGET_COUNT) failure_count++;
  if (full_gate_inconsistent_count != SELECTED13_FRESH_EMISSION_TARGET_COUNT) failure_count++;
  if (source_secret_lane_count != SELECTED13_FRESH_EMISSION_FAMILY_LANE_COUNT) failure_count++;

  printf("{{");
  printf("\\\"target_count\\\":%llu,", (unsigned long long)target_count);
  printf("\\\"row_slot_count\\\":%llu,", (unsigned long long)row_slot_count);
  printf("\\\"family_lane_count\\\":%llu,", (unsigned long long)family_lane_count);
  printf("\\\"accepted_export_count\\\":%llu,", (unsigned long long)accepted_export_count);
  printf("\\\"fresh_required_count\\\":%llu,", (unsigned long long)fresh_required_count);
  printf("\\\"full_gate_inconsistent_count\\\":%llu,", (unsigned long long)full_gate_inconsistent_count);
  printf("\\\"source_secret_lane_count\\\":%llu,", (unsigned long long)source_secret_lane_count);
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
    with tempfile.TemporaryDirectory(prefix="ecdlp_selected13_fresh_emission_", dir=str(temp_root)) as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "selected13_fresh_emission_packet.c"
        exe_path = temp_path / "selected13_fresh_emission_packet"
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


def build_targets(
    contract: dict[str, Any],
    worklist: dict[str, Any],
    gate: dict[str, Any],
    replay: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    queue_index = direct_backfill_by_transfer(contract)
    work_item_index = work_items_by_row_id(worklist)
    gate_index = transfer_gates_by_transfer(gate)
    lane_index = mask_lanes_by_transfer(gate)
    replay_transfer_index = replay_transfer_cases(replay)
    replay_mask_index = replay_mask_cases(replay)
    targets = []
    for transfer in PRIMARY_TRANSFERS:
        queue_item = queue_index.get(transfer)
        gate_item = gate_index.get(transfer)
        replay_case = replay_transfer_index.get(transfer)
        lanes = lane_index.get(transfer, [])
        if queue_item is None:
            failures.append({"code": "missing_contract_backfill_queue_item", "transfer_index": transfer})
            continue
        if gate_item is None:
            failures.append({"code": "missing_backfill_transfer_gate", "transfer_index": transfer})
            continue
        if replay_case is None:
            failures.append({"code": "missing_replay_transfer_case", "transfer_index": transfer})
            continue
        targets.append(
            build_emission_target(
                transfer,
                queue_item,
                gate_item,
                lanes,
                replay_case,
                replay_mask_index,
                work_item_index,
            )
        )
    targets.sort(key=lambda item: as_int(item.get("priority_order")))
    return targets, failures


def summarize(targets: list[dict[str, Any]], native: dict[str, Any]) -> dict[str, Any]:
    native_summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    full_gate_counts = Counter(str(target.get("full_gate_replay_result")) for target in targets)
    lane_tiers = Counter(
        str(lane.get("source_tier"))
        for target in targets
        for lane in target.get("family_lanes") or []
    )
    row_slots = [slot for target in targets for slot in target.get("row_slots") or []]
    return {
        "accepted_backfill_export_count": sum(as_int(target.get("accepted_backfill_export_count")) for target in targets),
        "family_lane_count": sum(as_int(target.get("family_lane_count")) for target in targets),
        "full_family_target_count": sum(1 for target in targets if (target.get("full_family_row") or {}).get("row_id")),
        "full_gate_replay_result_counts": dict(sorted(full_gate_counts.items())),
        "native_preflight_failure_count": as_int(native_summary.get("failure_count"), -1),
        "native_preflight_verified": as_int(native_summary.get("failure_count"), -1) == 0,
        "primary_backfill_transfers": [as_int(target.get("transfer_index")) for target in targets],
        "row_slot_count": len(row_slots),
        "row_slot_direct_status_counts": dict(sorted(Counter(str(slot.get("direct_status")) for slot in row_slots).items())),
        "source_tier_counts": dict(sorted(lane_tiers.items())),
        "target_count": len(targets),
        "worker_interpretation": (
            "The packet binds priority backfill rows to row hashes and replay-negative controls; "
            "a lower-level FFE/summation-polynomial worker must freshly emit direct/rank rows before any export is accepted."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--worklist", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--backfill-gate", type=Path, default=DEFAULT_BACKFILL_GATE)
    parser.add_argument("--replay-algebra", type=Path, default=DEFAULT_REPLAY_ALGEBRA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--no-c-header", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_json(args.contract)
    worklist = load_json(args.worklist)
    gate = load_json(args.backfill_gate)
    replay = load_json(args.replay_algebra)
    failures = validate_sources(contract, worklist, gate, replay)
    targets, target_build_failures = build_targets(contract, worklist, gate, replay)
    failures.extend(target_build_failures)
    failures.extend(validate_targets(targets))

    native: dict[str, Any] = {"compiled": False, "executed": False, "native_summary": {}}
    if not args.no_c_header:
        args.c_header_out.parent.mkdir(parents=True, exist_ok=True)
        args.c_header_out.write_text(render_c_header(targets))
        native = run_native_preflight(args.c_header_out, args.cc)
        failures.extend(compare_native(native))

    verified = not failures
    payload = {
        "artifacts": {
            "backfill_gate": str(args.backfill_gate),
            "c_header": None if args.no_c_header else str(args.c_header_out),
            "contract": str(args.contract),
            "replay_algebra": str(args.replay_algebra),
            "worklist": str(args.worklist),
        },
        "claim_status": (
            "FFE_SHARP_LANE_FRESH_EMISSION_PACKET_READY"
            if verified
            else "FFE_SHARP_LANE_FRESH_EMISSION_PACKET_FAILED_CHECK"
        ),
        "created_at": now_iso(),
        "emission_targets": targets,
        "failures": failures,
        "honesty_boundary": [
            "This packet describes fresh FFE/summation-polynomial emission obligations only.",
            "It does not evaluate summation polynomials or emit direct/rank rows.",
            "Replay-algebra source-secret lanes are negative controls, not accepted exports.",
            "A promoted result must match the bound row_id and row_check_hash and verify the public key.",
        ],
        "native_preflight": native,
        "schema": SCHEMA,
        "summary": summarize(targets, native),
    }
    payload["summary"]["verified"] = verified
    write_json(args.out, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "failures": failures, "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
