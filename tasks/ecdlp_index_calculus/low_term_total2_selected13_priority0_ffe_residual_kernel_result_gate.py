#!/usr/bin/env python3
"""Emit result-completion gates for the selected13 priority-0 residual kernel.

The residual kernel contract describes the fused FFE/summation-polynomial work
surface.  This script emits the next ABI layer: stable completion tokens for
each fused term group, each lane, and the final target export gate.  A future
kernel result must bind to these tokens before it can be accepted.

This is a result gate only.  It does not evaluate summation polynomials,
materialize coefficients, emit a target direct/rank row, solve ECDLP, or claim
a Pollard-rho speedup.
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


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_ffe_residual_kernel_result_gate.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_KERNEL_CONTRACT = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_contract_10376_probe.json"
)
DEFAULT_TARGET_GATE = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_target_export_verifier_gate_10376_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_result_gate_10376_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_result_gate_10376_probe.h"

EXPECTED_TRANSFER = 10376
EXPECTED_ROW_REQUEST_ID = "selected13_matmiss_10376_0_d352d0d05610"
EXPECTED_SECRET = 5859
EXPECTED_GROUP_COUNT = 7
EXPECTED_LANE_COUNT = 3
EXPECTED_TERM_SLOT_COUNT = 20
EXPECTED_SYNTHESIS_SLOT_COUNT = 19
EXPECTED_HYBRID_SLOT_COUNT = 14
EXPECTED_GUARD_SLOT_COUNT = 1

GROUP_CLASS_CODES = {
    "plain_and_hybrid_residual": 1,
    "hybrid_residual": 2,
    "guarded_hybrid_residual": 3,
}
LANE_CLASS_CODES = {
    "source_coeff_guard_partial_residual_synthesis": 1,
    "shared_product_verified_coeff_materialization": 2,
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


def stable_hash_u64(material: Any) -> int:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def validate_sources(kernel: dict[str, Any], gate: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if kernel.get("claim_status") != "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_CONTRACT_READY":
        failures.append({"code": "kernel_contract_not_ready", "claim_status": kernel.get("claim_status")})
    if kernel.get("failures"):
        failures.append({"code": "kernel_contract_has_failures", "failures": kernel.get("failures")})
    if (kernel.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "kernel_contract_summary_not_verified", "summary": kernel.get("summary")})
    if (kernel.get("native_preflight") or {}).get("verified") is not True:
        failures.append({"code": "kernel_contract_native_preflight_not_verified"})
    if gate.get("claim_status") != "SELECTED13_PRIORITY0_TARGET_EXPORT_VERIFIER_GATE_READY":
        failures.append({"code": "target_gate_not_ready", "claim_status": gate.get("claim_status")})
    if gate.get("failures"):
        failures.append({"code": "target_gate_has_failures", "failures": gate.get("failures")})
    target = kernel.get("target_slot") or {}
    if as_int(target.get("transfer_index"), -1) != EXPECTED_TRANSFER:
        failures.append({"code": "target_transfer_unexpected", "observed": target.get("transfer_index")})
    if target.get("row_request_id") != EXPECTED_ROW_REQUEST_ID:
        failures.append({"code": "target_row_request_unexpected", "observed": target.get("row_request_id")})
    if as_int((kernel.get("kernel_contract") or {}).get("expected_secret")) != EXPECTED_SECRET:
        failures.append({"code": "kernel_expected_secret_unexpected", "kernel_contract": kernel.get("kernel_contract")})
    if as_int((gate.get("source_consensus") or {}).get("expected_secret")) != EXPECTED_SECRET:
        failures.append({"code": "target_gate_expected_secret_unexpected", "source_consensus": gate.get("source_consensus")})
    return failures


def group_required_flags(group: dict[str, Any]) -> dict[str, bool]:
    return {
        "must_apply_lane_fanout": as_int(group.get("lane_count")) > 1,
        "must_materialize_hybrid_coefficients": as_int(group.get("hybrid_materialization_slot_count")) > 0,
        "must_preserve_source_guard": as_int(group.get("source_guard_slot_count")) > 0,
        "must_synthesize_residual_term": as_int(group.get("synthesis_required_slot_count")) > 0,
    }


def build_group_completion_gates(kernel: dict[str, Any]) -> list[dict[str, Any]]:
    gates = []
    target = kernel.get("target_slot") or {}
    packet_hash = as_int(kernel.get("packet_hash_u64"))
    for group in kernel.get("fused_term_groups") or []:
        material = {
            "contract_packet_hash_u64": packet_hash,
            "group_index": group.get("group_index"),
            "row_request_id": target.get("row_request_id"),
            "term_group_hash_u64": group.get("term_group_hash_u64"),
        }
        flags = group_required_flags(group)
        gates.append(
            {
                "accepted": False,
                "completion_token_hash_u64": stable_hash_u64(material),
                "expected_hybrid_materialization_slot_count": as_int(group.get("hybrid_materialization_slot_count")),
                "expected_lane_bitmap": as_int(group.get("lane_bitmap")),
                "expected_plain_synthesis_slot_count": as_int(group.get("plain_synthesis_slot_count")),
                "expected_source_guard_slot_count": as_int(group.get("source_guard_slot_count")),
                "expected_synthesis_required_slot_count": as_int(group.get("synthesis_required_slot_count")),
                "expected_term_slot_count": as_int(group.get("term_slot_count")),
                "group_index": as_int(group.get("group_index"), -1),
                "pending_result": True,
                "required_result_flags": flags,
                "term": as_int(group.get("term"), -1),
                "term_group_class": group.get("term_group_class"),
                "term_group_class_code": GROUP_CLASS_CODES.get(str(group.get("term_group_class")), 0),
                "term_group_hash_u64": as_int(group.get("term_group_hash_u64")),
                "term_mask": as_int(group.get("term_mask")),
            }
        )
    return gates


def build_lane_completion_gates(kernel: dict[str, Any]) -> list[dict[str, Any]]:
    gates = []
    target = kernel.get("target_slot") or {}
    packet_hash = as_int(kernel.get("packet_hash_u64"))
    for lane in kernel.get("kernel_lanes") or []:
        material = {
            "contract_packet_hash_u64": packet_hash,
            "hint_local_index": lane.get("hint_local_index"),
            "kernel_lane_hash_u64": lane.get("kernel_lane_hash_u64"),
            "row_request_id": target.get("row_request_id"),
        }
        gates.append(
            {
                "accepted": False,
                "completion_token_hash_u64": stable_hash_u64(material),
                "expected_materialization_required": bool(lane.get("materialization_required")),
                "expected_source_public_key_verified": bool(lane.get("source_public_key_verified")),
                "expected_source_secret": as_int(lane.get("source_voter_secret")),
                "expected_synthesis_required_slot_count": as_int(lane.get("synthesis_required_slot_count")),
                "expected_term_slot_count": as_int(lane.get("term_slot_count")),
                "hint_local_index": as_int(lane.get("hint_local_index"), -1),
                "kernel_lane_hash_u64": as_int(lane.get("kernel_lane_hash_u64")),
                "lane_class": lane.get("lane_class"),
                "lane_class_code": LANE_CLASS_CODES.get(str(lane.get("lane_class")), 0),
                "pending_result": True,
                "source_voter_hash_u64": as_int(lane.get("source_voter_hash_u64")),
            }
        )
    return gates


def target_export_completion_gate(kernel: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    target = kernel.get("target_slot") or {}
    material = {
        "contract_packet_hash_u64": kernel.get("packet_hash_u64"),
        "expected_secret": EXPECTED_SECRET,
        "row_request_id": target.get("row_request_id"),
        "target_gate_packet_hash_u64": gate.get("packet_hash_u64"),
    }
    return {
        "accepted": False,
        "completion_token_hash_u64": stable_hash_u64(material),
        "expected_secret": EXPECTED_SECRET,
        "must_bind_all_group_completion_tokens": True,
        "must_bind_all_lane_completion_tokens": True,
        "must_emit_target_direct_rank_export": True,
        "must_match_row_request_id": target.get("row_request_id"),
        "must_preserve_kernel_contract_packet_hash_u64": as_int(kernel.get("packet_hash_u64")),
        "must_preserve_target_gate_packet_hash_u64": as_int(gate.get("packet_hash_u64")),
        "must_reject_pending_or_source_replay_result": True,
        "must_set_relation_derived_ecdlp": True,
        "must_verify_public_key": True,
        "pending_result": True,
        "relation_derived_ecdlp": False,
        "target_row_request_id_u64": as_int(target.get("row_request_id_u64")),
    }


def validate_result_gates(
    kernel: dict[str, Any],
    gate: dict[str, Any],
    group_gates: list[dict[str, Any]],
    lane_gates: list[dict[str, Any]],
    export_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = validate_sources(kernel, gate)
    if len(group_gates) != EXPECTED_GROUP_COUNT:
        failures.append({"code": "group_completion_gate_count_unexpected", "observed": len(group_gates)})
    if len(lane_gates) != EXPECTED_LANE_COUNT:
        failures.append({"code": "lane_completion_gate_count_unexpected", "observed": len(lane_gates)})
    if len({as_int(item.get("completion_token_hash_u64")) for item in group_gates + lane_gates}) != len(group_gates) + len(lane_gates):
        failures.append({"code": "completion_tokens_not_unique"})
    if sum(as_int(item.get("expected_term_slot_count")) for item in group_gates) != EXPECTED_TERM_SLOT_COUNT:
        failures.append({"code": "group_gate_term_slot_sum_unexpected"})
    if sum(as_int(item.get("expected_synthesis_required_slot_count")) for item in group_gates) != EXPECTED_SYNTHESIS_SLOT_COUNT:
        failures.append({"code": "group_gate_synthesis_slot_sum_unexpected"})
    if sum(as_int(item.get("expected_hybrid_materialization_slot_count")) for item in group_gates) != EXPECTED_HYBRID_SLOT_COUNT:
        failures.append({"code": "group_gate_hybrid_slot_sum_unexpected"})
    if sum(as_int(item.get("expected_source_guard_slot_count")) for item in group_gates) != EXPECTED_GUARD_SLOT_COUNT:
        failures.append({"code": "group_gate_guard_slot_sum_unexpected"})
    if sum(as_int(item.get("expected_term_slot_count")) for item in lane_gates) != EXPECTED_TERM_SLOT_COUNT:
        failures.append({"code": "lane_gate_term_slot_sum_unexpected"})
    if sum(as_int(item.get("expected_synthesis_required_slot_count")) for item in lane_gates) != EXPECTED_SYNTHESIS_SLOT_COUNT:
        failures.append({"code": "lane_gate_synthesis_slot_sum_unexpected"})
    if sum(1 for item in lane_gates if item.get("expected_materialization_required")) != 2:
        failures.append({"code": "lane_gate_materialization_count_unexpected"})
    if sum(1 for item in group_gates if item.get("accepted")) != 0:
        failures.append({"code": "group_gate_already_accepted"})
    if sum(1 for item in lane_gates if item.get("accepted")) != 0:
        failures.append({"code": "lane_gate_already_accepted"})
    if export_gate.get("accepted") is not False or export_gate.get("relation_derived_ecdlp") is not False:
        failures.append({"code": "target_export_gate_already_accepted", "target_export_completion_gate": export_gate})
    for item in group_gates:
        if as_int(item.get("completion_token_hash_u64")) == 0 or as_int(item.get("term_group_hash_u64")) == 0:
            failures.append({"code": "group_completion_token_missing", "group": item})
        if as_int(item.get("term_group_class_code")) == 0:
            failures.append({"code": "group_class_code_missing", "group": item})
    for item in lane_gates:
        if as_int(item.get("completion_token_hash_u64")) == 0 or as_int(item.get("kernel_lane_hash_u64")) == 0:
            failures.append({"code": "lane_completion_token_missing", "lane": item})
        if as_int(item.get("lane_class_code")) == 0:
            failures.append({"code": "lane_class_code_missing", "lane": item})
        if as_int(item.get("expected_source_secret")) != EXPECTED_SECRET:
            failures.append({"code": "lane_source_secret_mismatch", "lane": item})
    if as_int(export_gate.get("completion_token_hash_u64")) == 0:
        failures.append({"code": "target_export_completion_token_missing"})
    return failures


def render_c_header(
    group_gates: list[dict[str, Any]],
    lane_gates: list[dict[str, Any]],
    export_gate: dict[str, Any],
    target: dict[str, Any],
) -> str:
    group_lines = []
    for item in group_gates:
        flags = item.get("required_result_flags") or {}
        group_lines.append(
            "  {"
            f"{as_int(item.get('group_index'))}ULL, "
            f"{as_int(item.get('term'))}ULL, "
            f"{as_int(item.get('term_mask'))}ULL, "
            f"{as_int(item.get('term_group_class_code'))}ULL, "
            f"{as_int(item.get('term_group_hash_u64'))}ULL, "
            f"{as_int(item.get('completion_token_hash_u64'))}ULL, "
            f"{as_int(item.get('expected_lane_bitmap'))}ULL, "
            f"{as_int(item.get('expected_term_slot_count'))}ULL, "
            f"{as_int(item.get('expected_synthesis_required_slot_count'))}ULL, "
            f"{as_int(item.get('expected_plain_synthesis_slot_count'))}ULL, "
            f"{as_int(item.get('expected_hybrid_materialization_slot_count'))}ULL, "
            f"{as_int(item.get('expected_source_guard_slot_count'))}ULL, "
            f"{1 if flags.get('must_materialize_hybrid_coefficients') else 0}ULL, "
            f"{1 if flags.get('must_preserve_source_guard') else 0}ULL, "
            f"{1 if item.get('accepted') else 0}ULL"
            "},"
        )
    lane_lines = []
    for item in lane_gates:
        lane_lines.append(
            "  {"
            f"{as_int(item.get('hint_local_index'))}ULL, "
            f"{as_int(item.get('lane_class_code'))}ULL, "
            f"{as_int(item.get('kernel_lane_hash_u64'))}ULL, "
            f"{as_int(item.get('source_voter_hash_u64'))}ULL, "
            f"{as_int(item.get('completion_token_hash_u64'))}ULL, "
            f"{as_int(item.get('expected_source_secret'))}ULL, "
            f"{1 if item.get('expected_materialization_required') else 0}ULL, "
            f"{1 if item.get('expected_source_public_key_verified') else 0}ULL, "
            f"{as_int(item.get('expected_term_slot_count'))}ULL, "
            f"{as_int(item.get('expected_synthesis_required_slot_count'))}ULL, "
            f"{1 if item.get('accepted') else 0}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_GATE_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_GATE_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_RESULT_GATE_TRANSFER {as_int(target.get('transfer_index'))}ULL
#define SELECTED13_PRIORITY0_RESULT_GATE_ROW_REQUEST_U64 {as_int(target.get('row_request_id_u64'))}ULL
#define SELECTED13_PRIORITY0_RESULT_GATE_EXPECTED_SECRET {EXPECTED_SECRET}ULL
#define SELECTED13_PRIORITY0_RESULT_GATE_GROUP_COUNT {len(group_gates)}ULL
#define SELECTED13_PRIORITY0_RESULT_GATE_LANE_COUNT {len(lane_gates)}ULL
#define SELECTED13_PRIORITY0_RESULT_GATE_TERM_SLOT_COUNT {sum(as_int(item.get('expected_term_slot_count')) for item in group_gates)}ULL
#define SELECTED13_PRIORITY0_RESULT_GATE_SYNTHESIS_SLOT_COUNT {sum(as_int(item.get('expected_synthesis_required_slot_count')) for item in group_gates)}ULL
#define SELECTED13_PRIORITY0_RESULT_GATE_HYBRID_SLOT_COUNT {sum(as_int(item.get('expected_hybrid_materialization_slot_count')) for item in group_gates)}ULL
#define SELECTED13_PRIORITY0_RESULT_GATE_GUARD_SLOT_COUNT {sum(as_int(item.get('expected_source_guard_slot_count')) for item in group_gates)}ULL
#define SELECTED13_PRIORITY0_RESULT_GATE_TARGET_EXPORT_TOKEN {as_int(export_gate.get('completion_token_hash_u64'))}ULL
#define SELECTED13_PRIORITY0_RESULT_GATE_ACCEPTED_EXPORT_COUNT 0ULL
#define SELECTED13_PRIORITY0_RESULT_GATE_RELATION_DERIVED_ECDLP 0ULL
#define SELECTED13_PRIORITY0_RESULT_GATE_POLLARD_RHO_SPEEDUP_CLAIMED 0ULL

#define SELECTED13_RESULT_GATE_GROUP_PLAIN_AND_HYBRID 1ULL
#define SELECTED13_RESULT_GATE_GROUP_HYBRID 2ULL
#define SELECTED13_RESULT_GATE_GROUP_GUARDED_HYBRID 3ULL
#define SELECTED13_RESULT_GATE_LANE_SOURCE_GUARD 1ULL
#define SELECTED13_RESULT_GATE_LANE_SHARED_PRODUCT 2ULL

typedef struct {{
  uint64_t group_index;
  uint64_t term;
  uint64_t term_mask;
  uint64_t term_group_class_code;
  uint64_t term_group_hash_u64;
  uint64_t completion_token_hash_u64;
  uint64_t expected_lane_bitmap;
  uint64_t expected_term_slot_count;
  uint64_t expected_synthesis_required_slot_count;
  uint64_t expected_plain_synthesis_slot_count;
  uint64_t expected_hybrid_materialization_slot_count;
  uint64_t expected_source_guard_slot_count;
  uint64_t must_materialize_hybrid_coefficients;
  uint64_t must_preserve_source_guard;
  uint64_t accepted;
}} selected13_priority0_result_group_gate_t;

typedef struct {{
  uint64_t hint_local_index;
  uint64_t lane_class_code;
  uint64_t kernel_lane_hash_u64;
  uint64_t source_voter_hash_u64;
  uint64_t completion_token_hash_u64;
  uint64_t expected_source_secret;
  uint64_t expected_materialization_required;
  uint64_t expected_source_public_key_verified;
  uint64_t expected_term_slot_count;
  uint64_t expected_synthesis_required_slot_count;
  uint64_t accepted;
}} selected13_priority0_result_lane_gate_t;

static const selected13_priority0_result_group_gate_t SELECTED13_PRIORITY0_RESULT_GROUP_GATES[] = {{
{chr(10).join(group_lines)}
}};

static const selected13_priority0_result_lane_gate_t SELECTED13_PRIORITY0_RESULT_LANE_GATES[] = {{
{chr(10).join(lane_lines)}
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
  uint64_t group_slot_sum = 0;
  uint64_t group_synthesis_sum = 0;
  uint64_t group_hybrid_sum = 0;
  uint64_t group_guard_sum = 0;
  uint64_t group_accepted_sum = 0;
  uint64_t lane_slot_sum = 0;
  uint64_t lane_synthesis_sum = 0;
  uint64_t lane_materialization_count = 0;
  uint64_t lane_verified_count = 0;
  uint64_t lane_secret_match_count = 0;
  uint64_t lane_accepted_sum = 0;

  const size_t group_count = sizeof(SELECTED13_PRIORITY0_RESULT_GROUP_GATES) / sizeof(SELECTED13_PRIORITY0_RESULT_GROUP_GATES[0]);
  const size_t lane_count = sizeof(SELECTED13_PRIORITY0_RESULT_LANE_GATES) / sizeof(SELECTED13_PRIORITY0_RESULT_LANE_GATES[0]);
  if (SELECTED13_PRIORITY0_RESULT_GATE_TRANSFER != {EXPECTED_TRANSFER}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESULT_GATE_ROW_REQUEST_U64 == 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESULT_GATE_EXPECTED_SECRET != {EXPECTED_SECRET}ULL) failure_count++;
  if (group_count != SELECTED13_PRIORITY0_RESULT_GATE_GROUP_COUNT || group_count != {EXPECTED_GROUP_COUNT}ULL) failure_count++;
  if (lane_count != SELECTED13_PRIORITY0_RESULT_GATE_LANE_COUNT || lane_count != {EXPECTED_LANE_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESULT_GATE_TARGET_EXPORT_TOKEN == 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESULT_GATE_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESULT_GATE_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESULT_GATE_POLLARD_RHO_SPEEDUP_CLAIMED != 0ULL) failure_count++;

  for (size_t i = 0; i < group_count; i++) {{
    const selected13_priority0_result_group_gate_t *gate = &SELECTED13_PRIORITY0_RESULT_GROUP_GATES[i];
    if (gate->term_mask == 0ULL || gate->term_group_hash_u64 == 0ULL || gate->completion_token_hash_u64 == 0ULL) failure_count++;
    if (gate->term_group_class_code == 0ULL) failure_count++;
    if (gate->expected_synthesis_required_slot_count == 0ULL) failure_count++;
    group_slot_sum += gate->expected_term_slot_count;
    group_synthesis_sum += gate->expected_synthesis_required_slot_count;
    group_hybrid_sum += gate->expected_hybrid_materialization_slot_count;
    group_guard_sum += gate->expected_source_guard_slot_count;
    group_accepted_sum += gate->accepted;
  }}

  for (size_t i = 0; i < lane_count; i++) {{
    const selected13_priority0_result_lane_gate_t *gate = &SELECTED13_PRIORITY0_RESULT_LANE_GATES[i];
    if (gate->kernel_lane_hash_u64 == 0ULL || gate->source_voter_hash_u64 == 0ULL || gate->completion_token_hash_u64 == 0ULL) failure_count++;
    if (gate->lane_class_code == 0ULL) failure_count++;
    if (gate->expected_source_secret == SELECTED13_PRIORITY0_RESULT_GATE_EXPECTED_SECRET) lane_secret_match_count++;
    if (gate->expected_materialization_required) lane_materialization_count++;
    if (gate->expected_source_public_key_verified) lane_verified_count++;
    lane_slot_sum += gate->expected_term_slot_count;
    lane_synthesis_sum += gate->expected_synthesis_required_slot_count;
    lane_accepted_sum += gate->accepted;
  }}
  if (group_slot_sum != SELECTED13_PRIORITY0_RESULT_GATE_TERM_SLOT_COUNT || group_slot_sum != {EXPECTED_TERM_SLOT_COUNT}ULL) failure_count++;
  if (group_synthesis_sum != SELECTED13_PRIORITY0_RESULT_GATE_SYNTHESIS_SLOT_COUNT || group_synthesis_sum != {EXPECTED_SYNTHESIS_SLOT_COUNT}ULL) failure_count++;
  if (group_hybrid_sum != SELECTED13_PRIORITY0_RESULT_GATE_HYBRID_SLOT_COUNT || group_hybrid_sum != {EXPECTED_HYBRID_SLOT_COUNT}ULL) failure_count++;
  if (group_guard_sum != SELECTED13_PRIORITY0_RESULT_GATE_GUARD_SLOT_COUNT || group_guard_sum != {EXPECTED_GUARD_SLOT_COUNT}ULL) failure_count++;
  if (lane_slot_sum != SELECTED13_PRIORITY0_RESULT_GATE_TERM_SLOT_COUNT) failure_count++;
  if (lane_synthesis_sum != SELECTED13_PRIORITY0_RESULT_GATE_SYNTHESIS_SLOT_COUNT) failure_count++;
  if (lane_materialization_count != 2ULL) failure_count++;
  if (lane_verified_count != 3ULL) failure_count++;
  if (lane_secret_match_count != 3ULL) failure_count++;
  if (group_accepted_sum != 0ULL) failure_count++;
  if (lane_accepted_sum != 0ULL) failure_count++;

  printf("selected13_priority0_ffe_residual_kernel_result_gate_preflight transfer=%llu groups=%llu lanes=%llu target_token=%llu slots=%llu synthesis_slots=%llu accepted_groups=%llu accepted_lanes=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_RESULT_GATE_TRANSFER,
         (unsigned long long)group_count,
         (unsigned long long)lane_count,
         (unsigned long long)SELECTED13_PRIORITY0_RESULT_GATE_TARGET_EXPORT_TOKEN,
         (unsigned long long)SELECTED13_PRIORITY0_RESULT_GATE_TERM_SLOT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_RESULT_GATE_SYNTHESIS_SLOT_COUNT,
         (unsigned long long)group_accepted_sum,
         (unsigned long long)lane_accepted_sum,
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
    with tempfile.TemporaryDirectory(prefix="selected13_priority0_result_gate_", dir=str(temp_root)) as tmp:
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
    group_gates: list[dict[str, Any]],
    lane_gates: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    native_preflight: dict[str, Any],
) -> dict[str, Any]:
    group_classes = Counter(str(item.get("term_group_class")) for item in group_gates)
    return {
        "accepted_group_completion_count": sum(1 for item in group_gates if item.get("accepted")),
        "accepted_lane_completion_count": sum(1 for item in lane_gates if item.get("accepted")),
        "accepted_relation_export_count": 0,
        "failure_count": len(failures),
        "group_completion_gate_count": len(group_gates),
        "lane_completion_gate_count": len(lane_gates),
        "native_preflight_verified": bool(native_preflight.get("verified")),
        "pending_group_completion_count": sum(1 for item in group_gates if item.get("pending_result")),
        "pending_lane_completion_count": sum(1 for item in lane_gates if item.get("pending_result")),
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "result_gate_status": "pending_external_kernel_outputs",
        "term_group_class_counts": dict(sorted(group_classes.items())),
        "verified": not failures,
        "worker_interpretation": (
            "The result gate assigns stable completion tokens to seven fused term groups, "
            "three lanes, and the final target export. All completions are pending; a future "
            "kernel must bind evidence to these tokens before the target export verifier gate "
            "can accept relation_derived_ecdlp."
        ),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    kernel_path = Path(args.kernel_contract)
    gate_path = Path(args.target_gate)
    kernel = load_json(kernel_path)
    target_gate = load_json(gate_path)
    group_gates = build_group_completion_gates(kernel)
    lane_gates = build_lane_completion_gates(kernel)
    export_gate = target_export_completion_gate(kernel, target_gate)
    failures = validate_result_gates(kernel, target_gate, group_gates, lane_gates, export_gate)
    target = kernel.get("target_slot") or {}
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_GATE_READY"
            if not failures
            else "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_GATE_FAILED"
        ),
        "parameters": {
            "kernel_contract": str(kernel_path),
            "target_gate": str(gate_path),
        },
        "packet_hash_u64": stable_hash_u64({"target": target, "group_gates": group_gates, "lane_gates": lane_gates, "export_gate": export_gate}),
        "target_slot": target,
        "source_kernel_contract": {
            "packet_hash_u64": as_int(kernel.get("packet_hash_u64")),
            "claim_status": kernel.get("claim_status"),
            "summary": kernel.get("summary"),
        },
        "source_target_gate": {
            "packet_hash_u64": as_int(target_gate.get("packet_hash_u64")),
            "claim_status": target_gate.get("claim_status"),
            "source_consensus": target_gate.get("source_consensus"),
        },
        "group_completion_gates": group_gates,
        "lane_completion_gates": lane_gates,
        "target_export_completion_gate": export_gate,
        "result_acceptance_contract": {
            "accepted_relation_export_count": 0,
            "expected_secret": EXPECTED_SECRET,
            "must_bind_group_completion_count": len(group_gates),
            "must_bind_lane_completion_count": len(lane_gates),
            "must_emit_target_direct_rank_export": True,
            "must_match_row_request_id": target.get("row_request_id"),
            "must_materialize_hybrid_slot_count": EXPECTED_HYBRID_SLOT_COUNT,
            "must_preserve_source_guard_slot_count": EXPECTED_GUARD_SLOT_COUNT,
            "must_reject_pending_or_source_replay_result": True,
            "must_set_relation_derived_ecdlp": True,
            "must_synthesize_required_slot_count": EXPECTED_SYNTHESIS_SLOT_COUNT,
            "must_verify_public_key": True,
            "relation_derived_ecdlp": False,
        },
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "all_completion_tokens_pending": True,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "summation_polynomial_evaluated": False,
            "target_export_not_emitted_by_this_gate": True,
        },
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-contract", default=str(DEFAULT_KERNEL_CONTRACT))
    parser.add_argument("--target-gate", default=str(DEFAULT_TARGET_GATE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(
        render_c_header(
            payload["group_completion_gates"],
            payload["lane_completion_gates"],
            payload["target_export_completion_gate"],
            payload["target_slot"],
        )
    )
    payload["artifacts"] = {"c_header": str(header_path)}
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_GATE_FAILED"
    payload["summary"] = summarize(
        payload["group_completion_gates"],
        payload["lane_completion_gates"],
        payload["failures"],
        native_preflight,
    )
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
