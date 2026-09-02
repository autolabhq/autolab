#!/usr/bin/env python3
"""Lower the selected13 priority-0 row into an FFE direct-rank emission ABI.

The priority-0 direct-rank worker packet identifies transfer 10376 as the next
fresh selected13 materialization target.  This script turns that packet into a
native-checkable FFE/summation-polynomial emission manifest: one target row
slot, three residual source-delta lanes, three family lanes, and the 10595
relation-derived positive controls.

This remains a worker contract.  It does not evaluate summation polynomials,
emit a target direct/rank row, derive a new ECDLP scalar, or claim a
Pollard-rho speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_ffe_direct_rank_emission_packet.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_WORKER_PACKET = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_direct_rank_worker_packet_10376_probe.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_direct_rank_emission_packet_10376_probe.json"
)
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_direct_rank_emission_packet_10376_probe.h"
)

SELECTED13_MASK = 1 << 13
TARGET_PHASE_CODE = 1
RESIDUAL_HINT_PHASE_CODE = 2
FAMILY_LANE_PHASE_CODE = 3
POSITIVE_CONTROL_PHASE_CODE = 4


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


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def stable_hash_u64(material: Any) -> int:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def terms_from_mask(mask: int) -> list[int]:
    return [index for index in range(64) if as_int(mask) & (1 << index)]


def gf2_rank64(values: list[int]) -> int:
    basis = [0] * 64
    rank = 0
    for value in values:
        x = as_int(value) & ((1 << 64) - 1)
        while x:
            bit = x.bit_length() - 1
            if basis[bit]:
                x ^= basis[bit]
                continue
            basis[bit] = x
            rank += 1
            break
    return rank


def unique_first_seen(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out = []
    for value in values:
        value = as_int(value)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def c_u64_array(values: list[int]) -> str:
    if not values:
        return "{0ULL}"
    return "{" + ", ".join(f"{as_int(value)}ULL" for value in values) + "}"


def build_target_slot(packet: dict[str, Any]) -> dict[str, Any]:
    target = packet.get("target") or {}
    family_masks = [as_int(lane.get("family_mask")) for lane in target.get("family_lanes") or []]
    residual_masks = unique_first_seen(
        [as_int(hint.get("missing_target_support_mask")) for hint in packet.get("hint_lanes") or []]
    )
    salts = [as_int(salt) for salt in target.get("target_salts") or []][:2]
    while len(salts) < 2:
        salts.append(0)
    return {
        "candidate_case_arg": target.get("candidate_case_arg"),
        "classification": target.get("classification"),
        "classification_code": as_int(target.get("classification_code")),
        "combined_family_residual_rank": gf2_rank64(family_masks + residual_masks),
        "direct_ops_over_rho": as_float(target.get("direct_ops_over_rho")),
        "direct_ops_over_rho_scaled": as_int(target.get("direct_ops_over_rho_scaled")),
        "family_mask_rank": gf2_rank64(family_masks),
        "family_masks": family_masks,
        "phase": "target_direct_rank_export",
        "phase_code": TARGET_PHASE_CODE,
        "priority_rank": as_int(target.get("priority_rank"), -1),
        "priority_score": as_int(target.get("priority_score")),
        "required_output": {
            "must_emit_target_direct_rank_export": True,
            "must_evaluate_fresh_ffe_or_summation_polynomial_system": True,
            "must_match_row_request_id": target.get("row_request_id"),
            "must_reject_source_hint_replay_as_target_export": True,
            "must_set_relation_derived_ecdlp": True,
            "must_verify_public_key": True,
        },
        "residual_mask_rank": gf2_rank64(residual_masks),
        "residual_masks": residual_masks,
        "row_keys": target.get("row_keys") or [],
        "row_request_id": target.get("row_request_id"),
        "row_request_id_u64": as_int(target.get("row_request_id_u64")),
        "selected_support_mask": as_int(target.get("selected_support_mask")),
        "selected_term_support": target.get("selected_term_support") or [],
        "selector": target.get("selector"),
        "target": target.get("target"),
        "target_salts": salts,
        "top_k": as_int(target.get("top_k")),
        "transfer_index": as_int(target.get("transfer_index"), -1),
        "worker_acceptance_gate": "fresh_ffe_direct_rank_export_and_relation_derived_ecdlp_only",
        "worker_class": target.get("worker_class"),
        "worker_class_code": as_int(target.get("worker_class_code")),
    }


def build_residual_lanes(packet: dict[str, Any], target_slot: dict[str, Any]) -> list[dict[str, Any]]:
    target_mask = as_int(target_slot.get("selected_support_mask"))
    lanes = []
    for index, hint in enumerate(packet.get("hint_lanes") or []):
        source_mask = as_int(hint.get("source_selected_support_mask"))
        missing_mask = as_int(hint.get("missing_target_support_mask"))
        lanes.append(
            {
                "direct_ops_over_rho": as_float(hint.get("direct_ops_over_rho")),
                "direct_ops_over_rho_scaled": as_int(hint.get("direct_ops_over_rho_scaled")),
                "direct_public_key_verified": bool(hint.get("direct_public_key_verified")),
                "extra_source_support_mask": as_int(hint.get("extra_source_support_mask")),
                "hint_hash_u64": as_int(hint.get("hint_hash_u64")),
                "hint_local_index": as_int(hint.get("hint_local_index"), index),
                "missing_target_support_mask": missing_mask,
                "missing_target_terms": hint.get("missing_target_terms") or terms_from_mask(missing_mask),
                "phase": "same_salt_residual_delta_guard",
                "phase_code": RESIDUAL_HINT_PHASE_CODE,
                "public_product_gate_selected": bool(hint.get("public_product_gate_selected")),
                "residual_lane_hash_u64": stable_hash_u64(
                    {
                        "hint_hash_u64": hint.get("hint_hash_u64"),
                        "missing_target_support_mask": missing_mask,
                        "source_selected_support_mask": source_mask,
                        "target_selected_support_mask": target_mask,
                    }
                ),
                "same_salt_pair": bool(hint.get("same_salt_pair")),
                "shared_product_public_key_verified": bool(hint.get("shared_product_public_key_verified")),
                "source_row_hash_u64": as_int(hint.get("source_row_hash_u64")),
                "source_row_keys": hint.get("source_row_keys") or [],
                "source_selected_support_mask": source_mask,
                "source_selector": hint.get("source_selector"),
                "source_selector_u64": as_int(hint.get("source_selector_u64")),
                "source_support_is_target_subset": (source_mask & ~target_mask) == 0,
                "source_terms": hint.get("source_terms") or terms_from_mask(source_mask),
                "source_top_k": as_int(hint.get("source_top_k")),
                "target_gap_exact": (source_mask | missing_mask) == target_mask and (source_mask & missing_mask) == 0,
                "target_selected_support_mask": target_mask,
                "worker_action": "freshly_solve_missing_target_terms_against_bound_target_row",
            }
        )
    return lanes


def build_family_lanes(packet: dict[str, Any]) -> list[dict[str, Any]]:
    lanes = []
    for lane in (packet.get("target") or {}).get("family_lanes") or []:
        family_mask = as_int(lane.get("family_mask"))
        lanes.append(
            {
                "family_index": as_int(lane.get("family_index")),
                "family_lane_hash_u64": stable_hash_u64(
                    {
                        "family_index": lane.get("family_index"),
                        "family_mask": family_mask,
                        "family_terms": lane.get("family_terms"),
                    }
                ),
                "family_mask": family_mask,
                "family_terms": lane.get("family_terms") or terms_from_mask(family_mask),
                "phase": "family_mask_direct_rank_lane",
                "phase_code": FAMILY_LANE_PHASE_CODE,
                "selected_support_covers_family": bool(lane.get("selected_support_covers_family")),
                "worker_action": "include_family_mask_in_fresh_rank_export",
            }
        )
    return lanes


def build_control_slots(packet: dict[str, Any]) -> list[dict[str, Any]]:
    slots = []
    for index, control in enumerate(packet.get("positive_controls") or []):
        slots.append(
            {
                "control_index": index,
                "control_hash_u64": stable_hash_u64(control),
                "derived_secret": as_int(control.get("derived_secret")),
                "phase": "positive_control_relation_derived_guard",
                "phase_code": POSITIVE_CONTROL_PHASE_CODE,
                "relation_derived_ecdlp": bool(control.get("relation_derived_ecdlp")),
                "row_request_id": control.get("row_request_id"),
                "row_request_id_u64": as_int(control.get("row_request_id_u64")),
                "selected_support_mask": as_int(control.get("selected_support_mask")),
                "transfer_index": as_int(control.get("transfer_index"), -1),
            }
        )
    return slots


def validate_packet_source(packet: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if packet.get("claim_status") != "SELECTED13_PRIORITY0_DIRECT_RANK_WORKER_PACKET_READY":
        failures.append({"code": "worker_packet_not_ready", "claim_status": packet.get("claim_status")})
    if packet.get("failures"):
        failures.append({"code": "worker_packet_has_failures", "failures": packet.get("failures")})
    summary = packet.get("summary") or {}
    if summary.get("verified") is not True:
        failures.append({"code": "worker_packet_summary_not_verified", "summary": summary})
    if as_int(summary.get("accepted_relation_export_count"), -1) != 0:
        failures.append({"code": "source_packet_claims_relation_export", "summary": summary})
    if summary.get("relation_derived_ecdlp") is not False:
        failures.append({"code": "source_packet_claims_relation_derived_ecdlp", "summary": summary})
    return failures


def validate_emission_shape(
    target_slot: dict[str, Any],
    residual_lanes: list[dict[str, Any]],
    family_lanes: list[dict[str, Any]],
    control_slots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    target_mask = as_int(target_slot.get("selected_support_mask"))
    if as_int(target_slot.get("priority_rank"), -1) != 0:
        failures.append({"code": "target_priority_rank_not_zero", "priority_rank": target_slot.get("priority_rank")})
    if as_int(target_slot.get("transfer_index"), -1) != 10376:
        failures.append({"code": "target_transfer_unexpected", "transfer_index": target_slot.get("transfer_index")})
    if not (target_mask & SELECTED13_MASK):
        failures.append({"code": "target_support_missing_selected13"})
    if as_int(target_slot.get("combined_family_residual_rank")) <= as_int(target_slot.get("family_mask_rank")):
        failures.append(
            {
                "code": "residual_masks_add_no_rank",
                "family_rank": target_slot.get("family_mask_rank"),
                "combined_rank": target_slot.get("combined_family_residual_rank"),
            }
        )
    if len(residual_lanes) != 3:
        failures.append({"code": "residual_lane_count_unexpected", "observed": len(residual_lanes)})
    if len(family_lanes) != 3:
        failures.append({"code": "family_lane_count_unexpected", "observed": len(family_lanes)})
    if sorted(as_int(lane.get("family_mask")) for lane in family_lanes) != [33, 17408, 34816]:
        failures.append({"code": "family_lane_masks_unexpected"})
    for lane in family_lanes:
        if not lane.get("selected_support_covers_family"):
            failures.append({"code": "family_lane_not_covered", "family_mask": lane.get("family_mask")})
    for lane in residual_lanes:
        source_mask = as_int(lane.get("source_selected_support_mask"))
        missing_mask = as_int(lane.get("missing_target_support_mask"))
        if not lane.get("direct_public_key_verified"):
            failures.append({"code": "residual_lane_source_not_direct_verified", "hint": lane.get("hint_local_index")})
        if not lane.get("same_salt_pair"):
            failures.append({"code": "residual_lane_not_same_salt", "hint": lane.get("hint_local_index")})
        if as_int(lane.get("extra_source_support_mask")) != 0:
            failures.append({"code": "residual_lane_has_extra_source_support", "hint": lane.get("hint_local_index")})
        if (source_mask & ~target_mask) != 0:
            failures.append({"code": "residual_lane_source_not_subset", "hint": lane.get("hint_local_index")})
        if (source_mask | missing_mask) != target_mask or (source_mask & missing_mask) != 0:
            failures.append({"code": "residual_lane_not_exact_gap", "hint": lane.get("hint_local_index")})
        if not (missing_mask & SELECTED13_MASK):
            failures.append({"code": "residual_lane_missing_selected13_delta", "hint": lane.get("hint_local_index")})
    if len(control_slots) < 3:
        failures.append({"code": "positive_control_count_below_expected", "observed": len(control_slots)})
    for control in control_slots:
        if not control.get("relation_derived_ecdlp") or as_int(control.get("derived_secret")) == 0:
            failures.append({"code": "positive_control_not_relation_derived", "control": control.get("row_request_id")})
    return failures


def render_c_header(
    target_slot: dict[str, Any],
    residual_lanes: list[dict[str, Any]],
    family_lanes: list[dict[str, Any]],
    control_slots: list[dict[str, Any]],
) -> str:
    salts = [as_int(salt) for salt in target_slot.get("target_salts") or []][:2]
    while len(salts) < 2:
        salts.append(0)
    family_rows = []
    for lane in family_lanes:
        family_rows.append(
            "  {"
            f"{as_int(lane.get('family_index'))}ULL, "
            f"{as_int(lane.get('family_mask'))}ULL, "
            f"{as_int(lane.get('family_lane_hash_u64'))}ULL, "
            f"{1 if lane.get('selected_support_covers_family') else 0}ULL"
            "},"
        )
    residual_rows = []
    for lane in residual_lanes:
        residual_rows.append(
            "  {"
            f"{as_int(lane.get('hint_local_index'))}ULL, "
            f"{as_int(lane.get('hint_hash_u64'))}ULL, "
            f"{as_int(lane.get('source_row_hash_u64'))}ULL, "
            f"{as_int(lane.get('residual_lane_hash_u64'))}ULL, "
            f"{as_int(lane.get('source_selected_support_mask'))}ULL, "
            f"{as_int(lane.get('missing_target_support_mask'))}ULL, "
            f"{as_int(lane.get('extra_source_support_mask'))}ULL, "
            f"{as_int(lane.get('source_selector_u64'))}ULL, "
            f"{as_int(lane.get('source_top_k'))}ULL, "
            f"{1 if lane.get('same_salt_pair') else 0}ULL, "
            f"{1 if lane.get('direct_public_key_verified') else 0}ULL, "
            f"{1 if lane.get('target_gap_exact') else 0}ULL"
            "},"
        )
    control_rows = []
    for control in control_slots:
        control_rows.append(
            "  {"
            f"{as_int(control.get('control_index'))}ULL, "
            f"{as_int(control.get('transfer_index'))}ULL, "
            f"{as_int(control.get('row_request_id_u64'))}ULL, "
            f"{as_int(control.get('selected_support_mask'))}ULL, "
            f"{as_int(control.get('derived_secret'))}ULL, "
            f"{as_int(control.get('control_hash_u64'))}ULL, "
            f"{1 if control.get('relation_derived_ecdlp') else 0}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_FFE_DIRECT_RANK_EMISSION_PACKET_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_FFE_DIRECT_RANK_EMISSION_PACKET_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_FFE_TRANSFER {as_int(target_slot.get('transfer_index'))}ULL
#define SELECTED13_PRIORITY0_FFE_ROW_REQUEST_U64 {as_int(target_slot.get('row_request_id_u64'))}ULL
#define SELECTED13_PRIORITY0_FFE_SELECTED_SUPPORT_MASK {as_int(target_slot.get('selected_support_mask'))}ULL
#define SELECTED13_PRIORITY0_FFE_RESIDUAL_LANE_COUNT {len(residual_lanes)}
#define SELECTED13_PRIORITY0_FFE_FAMILY_LANE_COUNT {len(family_lanes)}
#define SELECTED13_PRIORITY0_FFE_CONTROL_COUNT {len(control_slots)}
#define SELECTED13_PRIORITY0_FFE_ACCEPTED_EXPORT_COUNT 0ULL
#define SELECTED13_PRIORITY0_FFE_RELATION_DERIVED_ECDLP 0ULL
#define SELECTED13_PRIORITY0_FFE_FAMILY_MASK_RANK {as_int(target_slot.get('family_mask_rank'))}ULL
#define SELECTED13_PRIORITY0_FFE_RESIDUAL_MASK_RANK {as_int(target_slot.get('residual_mask_rank'))}ULL
#define SELECTED13_PRIORITY0_FFE_COMBINED_MASK_RANK {as_int(target_slot.get('combined_family_residual_rank'))}ULL

typedef struct {{
  uint64_t transfer_index;
  uint64_t row_request_id_u64;
  uint64_t selected_support_mask;
  uint64_t target_salts[2];
  uint64_t direct_ops_over_rho_scaled;
  uint64_t family_mask_rank;
  uint64_t residual_mask_rank;
  uint64_t combined_mask_rank;
  uint64_t phase_code;
}} selected13_priority0_ffe_target_t;

typedef struct {{
  uint64_t family_index;
  uint64_t family_mask;
  uint64_t family_lane_hash_u64;
  uint64_t selected_support_covers_family;
}} selected13_priority0_ffe_family_lane_t;

typedef struct {{
  uint64_t hint_local_index;
  uint64_t hint_hash_u64;
  uint64_t source_row_hash_u64;
  uint64_t residual_lane_hash_u64;
  uint64_t source_selected_support_mask;
  uint64_t missing_target_support_mask;
  uint64_t extra_source_support_mask;
  uint64_t source_selector_u64;
  uint64_t source_top_k;
  uint64_t same_salt_pair;
  uint64_t direct_public_key_verified;
  uint64_t target_gap_exact;
}} selected13_priority0_ffe_residual_lane_t;

typedef struct {{
  uint64_t control_index;
  uint64_t transfer_index;
  uint64_t row_request_id_u64;
  uint64_t selected_support_mask;
  uint64_t derived_secret;
  uint64_t control_hash_u64;
  uint64_t relation_derived_ecdlp;
}} selected13_priority0_ffe_control_t;

static const selected13_priority0_ffe_target_t SELECTED13_PRIORITY0_FFE_TARGET = {{
  {as_int(target_slot.get('transfer_index'))}ULL,
  {as_int(target_slot.get('row_request_id_u64'))}ULL,
  {as_int(target_slot.get('selected_support_mask'))}ULL,
  {c_u64_array(salts)},
  {as_int(target_slot.get('direct_ops_over_rho_scaled'))}ULL,
  {as_int(target_slot.get('family_mask_rank'))}ULL,
  {as_int(target_slot.get('residual_mask_rank'))}ULL,
  {as_int(target_slot.get('combined_family_residual_rank'))}ULL,
  {TARGET_PHASE_CODE}ULL
}};

static const selected13_priority0_ffe_family_lane_t SELECTED13_PRIORITY0_FFE_FAMILY_LANES[] = {{
{chr(10).join(family_rows)}
}};

static const selected13_priority0_ffe_residual_lane_t SELECTED13_PRIORITY0_FFE_RESIDUAL_LANES[] = {{
{chr(10).join(residual_rows)}
}};

static const selected13_priority0_ffe_control_t SELECTED13_PRIORITY0_FFE_CONTROLS[] = {{
{chr(10).join(control_rows)}
}};

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

int main(void) {{
  const uint64_t selected13_mask = {SELECTED13_MASK}ULL;
  const uint64_t target_mask = SELECTED13_PRIORITY0_FFE_TARGET.selected_support_mask;
  uint64_t failure_count = 0;
  uint64_t exact_gap_count = 0;
  uint64_t direct_verified_count = 0;
  uint64_t same_salt_count = 0;
  uint64_t residual_contains_selected13_count = 0;
  uint64_t covered_family_count = 0;
  uint64_t verified_control_count = 0;

  const size_t residual_count =
      sizeof(SELECTED13_PRIORITY0_FFE_RESIDUAL_LANES) / sizeof(SELECTED13_PRIORITY0_FFE_RESIDUAL_LANES[0]);
  const size_t family_count =
      sizeof(SELECTED13_PRIORITY0_FFE_FAMILY_LANES) / sizeof(SELECTED13_PRIORITY0_FFE_FAMILY_LANES[0]);
  const size_t control_count =
      sizeof(SELECTED13_PRIORITY0_FFE_CONTROLS) / sizeof(SELECTED13_PRIORITY0_FFE_CONTROLS[0]);

  if (SELECTED13_PRIORITY0_FFE_TARGET.transfer_index != SELECTED13_PRIORITY0_FFE_TRANSFER) failure_count++;
  if (SELECTED13_PRIORITY0_FFE_TARGET.row_request_id_u64 != SELECTED13_PRIORITY0_FFE_ROW_REQUEST_U64) failure_count++;
  if ((target_mask & selected13_mask) == 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_FFE_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_FFE_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;
  if (residual_count != SELECTED13_PRIORITY0_FFE_RESIDUAL_LANE_COUNT) failure_count++;
  if (family_count != SELECTED13_PRIORITY0_FFE_FAMILY_LANE_COUNT) failure_count++;
  if (control_count != SELECTED13_PRIORITY0_FFE_CONTROL_COUNT) failure_count++;
  if (SELECTED13_PRIORITY0_FFE_TARGET.combined_mask_rank <= SELECTED13_PRIORITY0_FFE_TARGET.family_mask_rank) failure_count++;

  for (size_t i = 0; i < family_count; i++) {{
    const selected13_priority0_ffe_family_lane_t *lane = &SELECTED13_PRIORITY0_FFE_FAMILY_LANES[i];
    if (lane->family_mask == 0ULL || lane->family_lane_hash_u64 == 0ULL) failure_count++;
    if ((lane->family_mask & ~target_mask) != 0ULL) failure_count++;
    if (lane->selected_support_covers_family) covered_family_count++;
  }}
  for (size_t i = 0; i < residual_count; i++) {{
    const selected13_priority0_ffe_residual_lane_t *lane = &SELECTED13_PRIORITY0_FFE_RESIDUAL_LANES[i];
    if (lane->hint_hash_u64 == 0ULL || lane->source_row_hash_u64 == 0ULL ||
        lane->residual_lane_hash_u64 == 0ULL) failure_count++;
    if (lane->extra_source_support_mask != 0ULL) failure_count++;
    if ((lane->source_selected_support_mask & ~target_mask) != 0ULL) failure_count++;
    if ((lane->source_selected_support_mask & lane->missing_target_support_mask) != 0ULL) failure_count++;
    if ((lane->source_selected_support_mask | lane->missing_target_support_mask) != target_mask) failure_count++;
    if ((lane->missing_target_support_mask & selected13_mask) != 0ULL) residual_contains_selected13_count++;
    if (lane->target_gap_exact) exact_gap_count++;
    if (lane->direct_public_key_verified) direct_verified_count++;
    if (lane->same_salt_pair) same_salt_count++;
  }}
  for (size_t i = 0; i < control_count; i++) {{
    const selected13_priority0_ffe_control_t *control = &SELECTED13_PRIORITY0_FFE_CONTROLS[i];
    if (control->row_request_id_u64 == 0ULL || control->derived_secret == 0ULL ||
        control->control_hash_u64 == 0ULL) failure_count++;
    if (control->relation_derived_ecdlp) verified_control_count++;
  }}

  if (covered_family_count != family_count) failure_count++;
  if (exact_gap_count != residual_count) failure_count++;
  if (direct_verified_count != residual_count) failure_count++;
  if (same_salt_count != residual_count) failure_count++;
  if (residual_contains_selected13_count != residual_count) failure_count++;
  if (verified_control_count != control_count) failure_count++;

  printf("selected13_priority0_ffe_direct_rank_emission_preflight transfer=%llu residual_lanes=%llu family_lanes=%llu controls=%llu combined_rank=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_FFE_TRANSFER,
         (unsigned long long)residual_count,
         (unsigned long long)family_count,
         (unsigned long long)control_count,
         (unsigned long long)SELECTED13_PRIORITY0_FFE_TARGET.combined_mask_rank,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path, compiler: str) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    temp_root = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path(tempfile.gettempdir())
    env = os.environ.copy()
    env["TMPDIR"] = str(temp_root)
    with tempfile.TemporaryDirectory(prefix="selected13_priority0_ffe_emission_", dir=str(temp_root)) as tmp:
        tmp_path = Path(tmp)
        c_path = tmp_path / "preflight.c"
        exe_path = tmp_path / "preflight"
        c_path.write_text(source)
        command = [
            compiler,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-O2",
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
                "verified": False,
            }
        native_run = subprocess.run([str(exe_path)], capture_output=True, text=True, check=False, env=env)
        return {
            "c_source_sha256": source_hash,
            "compile_command": command,
            "compile_returncode": compile_run.returncode,
            "compile_stderr": compile_run.stderr,
            "compile_stdout": compile_run.stdout,
            "compiled": True,
            "executed": True,
            "preflight_returncode": native_run.returncode,
            "preflight_stderr": native_run.stderr.strip(),
            "preflight_stdout": native_run.stdout.strip(),
            "verified": native_run.returncode == 0,
        }


def summarize(
    target_slot: dict[str, Any],
    residual_lanes: list[dict[str, Any]],
    family_lanes: list[dict[str, Any]],
    control_slots: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    native_preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "accepted_relation_export_count": 0,
        "combined_family_residual_rank": as_int(target_slot.get("combined_family_residual_rank")),
        "control_relation_derived_row_count": sum(1 for control in control_slots if control.get("relation_derived_ecdlp")),
        "control_slot_count": len(control_slots),
        "failure_count": len(failures),
        "family_lane_count": len(family_lanes),
        "family_mask_rank": as_int(target_slot.get("family_mask_rank")),
        "native_preflight_verified": bool(native_preflight.get("verified")),
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "residual_lane_count": len(residual_lanes),
        "residual_mask_rank": as_int(target_slot.get("residual_mask_rank")),
        "residual_missing_target_term_sets": [lane.get("missing_target_terms") for lane in residual_lanes],
        "target_row_request_id": target_slot.get("row_request_id"),
        "target_transfer_index": as_int(target_slot.get("transfer_index"), -1),
        "verified": not failures,
        "worker_interpretation": (
            "The manifest binds the 10376 target row to exact source-support residual gaps. "
            "A lower-level FFE/summation-polynomial worker must emit a fresh target direct/rank "
            "export and verify the public key before relation_derived_ecdlp can become true."
        ),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    worker_packet_path = Path(args.worker_packet)
    worker_packet = load_json(worker_packet_path)
    failures = validate_packet_source(worker_packet)
    target_slot = build_target_slot(worker_packet)
    residual_lanes = build_residual_lanes(worker_packet, target_slot)
    family_lanes = build_family_lanes(worker_packet)
    control_slots = build_control_slots(worker_packet)
    failures.extend(validate_emission_shape(target_slot, residual_lanes, family_lanes, control_slots))
    packet_hash = stable_hash_u64(
        {
            "control_slots": control_slots,
            "family_lanes": family_lanes,
            "residual_lanes": residual_lanes,
            "target_slot": target_slot,
        }
    )
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_FFE_DIRECT_RANK_EMISSION_PACKET_READY"
            if not failures
            else "SELECTED13_PRIORITY0_FFE_DIRECT_RANK_EMISSION_PACKET_FAILED"
        ),
        "parameters": {
            "worker_packet": str(worker_packet_path),
        },
        "packet_hash_u64": packet_hash,
        "target_slot": target_slot,
        "residual_lanes": residual_lanes,
        "family_lanes": family_lanes,
        "control_slots": control_slots,
        "required_worker_outputs": {
            "must_emit_target_direct_rank_export": True,
            "must_evaluate_fresh_ffe_or_summation_polynomial_system": True,
            "must_match_row_request_id": target_slot.get("row_request_id"),
            "must_reject_source_hint_replay_as_target_export": True,
            "must_set_relation_derived_ecdlp": True,
            "must_verify_public_key": True,
        },
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "fresh_direct_rank_worker_required": True,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "residual_source_hints_are_not_target_exports": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-packet", default=str(DEFAULT_WORKER_PACKET), help="Priority-0 worker packet JSON")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output emission packet JSON")
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT), help="Output C preflight header")
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"), help="C compiler for native preflight")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(
        render_c_header(
            payload["target_slot"],
            payload["residual_lanes"],
            payload["family_lanes"],
            payload["control_slots"],
        )
    )
    payload["artifacts"] = {"c_header": str(header_path)}
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_FFE_DIRECT_RANK_EMISSION_PACKET_FAILED"
    payload["summary"] = summarize(
        payload["target_slot"],
        payload["residual_lanes"],
        payload["family_lanes"],
        payload["control_slots"],
        payload["failures"],
        native_preflight,
    )
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
