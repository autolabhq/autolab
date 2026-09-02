#!/usr/bin/env python3
"""Emit a fused FFE residual kernel contract for selected13 priority-0.

The target export verifier gate fixes the acceptance boundary for transfer
10376.  This script lowers the 20 term slots into a smaller fused
FFE/summation-polynomial kernel surface: seven unique residual term groups,
their lane fanout, source-guard slots, and hybrid coefficient materialization
requirements.

This is an execution contract only.  It does not evaluate summation
polynomials, materialize coefficients, emit a target direct/rank row, solve
ECDLP, or claim a Pollard-rho speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_ffe_residual_kernel_contract.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_TARGET_GATE = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_target_export_verifier_gate_10376_probe.json"
)
DEFAULT_SYNTHESIS_WORKLIST = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_residual_synthesis_worklist_10376_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_contract_10376_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_contract_10376_probe.h"
DEFAULT_TASK_DIR = Path(os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus"))

EXPECTED_TRANSFER = 10376
EXPECTED_ROW_REQUEST_ID = "selected13_matmiss_10376_0_d352d0d05610"
EXPECTED_SECRET = 5859
EXPECTED_TERM_SLOT_COUNT = 20
EXPECTED_SYNTHESIS_SLOT_COUNT = 19
EXPECTED_FUSED_TERM_GROUP_COUNT = 7
TERM_STATUS_SOURCE_GUARD = "source_coefficient_guard_available"
TERM_STATUS_PLAIN_SYNTHESIS = "needs_residual_synthesis"
TERM_STATUS_HYBRID_SYNTHESIS = "needs_hybrid_coefficient_materialization_and_residual_synthesis"

LANE_CLASS_CODES = {
    "source_coeff_guard_partial_residual_synthesis": 1,
    "shared_product_verified_coeff_materialization": 2,
}
TERM_GROUP_CLASS_CODES = {
    "plain_and_hybrid_residual": 1,
    "hybrid_residual": 2,
    "guarded_hybrid_residual": 3,
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


def parse_target(raw: Any) -> tuple[str, int]:
    label, sep, prime = str(raw or "").partition("@")
    if not sep or not label:
        raise ValueError(f"target must be label@prime, got {raw!r}")
    return label, int(prime)


def load_verifier_module(task_dir: Path) -> Any:
    env_main = task_dir / "environment" / "main.py"
    data_path = task_dir / "environment" / "lmfdb_curves.json"
    if not env_main.is_file():
        raise FileNotFoundError(f"verifier environment missing: {env_main}")
    if not data_path.is_file():
        raise FileNotFoundError(f"LMFDB data missing: {data_path}")
    old_app_dir = os.environ.get("APP_DIR")
    old_lmfdb_data = os.environ.get("LMFDB_DATA")
    os.environ["APP_DIR"] = str(env_main.parent)
    os.environ["LMFDB_DATA"] = str(data_path)
    try:
        spec = importlib.util.spec_from_file_location("ecdlp_verifier_main_for_selected13_residual_kernel", env_main)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import verifier helpers from {env_main}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if old_app_dir is None:
            os.environ.pop("APP_DIR", None)
        else:
            os.environ["APP_DIR"] = old_app_dir
        if old_lmfdb_data is None:
            os.environ.pop("LMFDB_DATA", None)
        else:
            os.environ["LMFDB_DATA"] = old_lmfdb_data


def target_context(target_slot: dict[str, Any], task_dir: Path) -> dict[str, Any]:
    target = str(target_slot.get("target") or "")
    label, p = parse_target(target)
    verifier = load_verifier_module(task_dir)
    records = verifier.load_records()
    by_label = {str(record["label"]): record for record in records}
    record = by_label.get(label)
    if record is None:
        return {"context_status": "missing_verifier_record", "label": label, "p": p, "target": target}
    inv = verifier.reduction_invariants(record, p)
    base_order = int(inv["base_order"])
    return {
        "ainvs": [int(value) for value in record["ainvs"]],
        "ainvs_mod_p": [int(value) % p for value in record["ainvs"]],
        "base": verifier.point_to_json(inv["base"]),
        "base_order": base_order,
        "context_status": "verified",
        "generic_rho_steps": math.ceil(math.sqrt(math.pi * base_order / 2.0)),
        "group_order": int(inv["order"]),
        "label": label,
        "p": p,
        "precomputed_target": bool(inv.get("precomputed_target")),
        "target": target,
    }


def term_group_class(status_counts: Counter[str]) -> str:
    has_guard = status_counts.get(TERM_STATUS_SOURCE_GUARD, 0) > 0
    has_plain = status_counts.get(TERM_STATUS_PLAIN_SYNTHESIS, 0) > 0
    has_hybrid = status_counts.get(TERM_STATUS_HYBRID_SYNTHESIS, 0) > 0
    if has_guard and has_hybrid:
        return "guarded_hybrid_residual"
    if has_plain and has_hybrid:
        return "plain_and_hybrid_residual"
    if has_hybrid:
        return "hybrid_residual"
    return "plain_and_hybrid_residual"


def worker_action_for_group(klass: str) -> str:
    if klass == "guarded_hybrid_residual":
        return "reuse_source_guard_then_materialize_hybrid_coefficients_and_synthesize_residual_term"
    if klass == "hybrid_residual":
        return "materialize_hybrid_coefficients_and_synthesize_residual_term"
    return "synthesize_residual_term_once_and_fan_out_to_plain_and_hybrid_lanes"


def all_term_slots(worklist: dict[str, Any]) -> list[dict[str, Any]]:
    slots = []
    for lane in worklist.get("lane_work_items") or []:
        hint = as_int(lane.get("hint_local_index"), -1)
        for slot in lane.get("term_slots") or []:
            item = dict(slot)
            item["lane_class"] = lane.get("lane_class")
            item["lane_work_hash_u64"] = as_int(lane.get("lane_work_hash_u64"))
            item["source_row_hash_u64"] = as_int(lane.get("source_row_hash_u64"))
            item["hint_local_index"] = hint
            slots.append(item)
    return slots


def build_kernel_lanes(worklist: dict[str, Any], gate: dict[str, Any]) -> list[dict[str, Any]]:
    voters = {as_int(voter.get("hint_local_index"), -1): voter for voter in gate.get("source_voters") or []}
    lanes = []
    for lane in worklist.get("lane_work_items") or []:
        hint = as_int(lane.get("hint_local_index"), -1)
        term_statuses = Counter(str(slot.get("status")) for slot in lane.get("term_slots") or [])
        voter = voters.get(hint, {})
        item = {
            "coefficient_form_count": as_int(lane.get("coefficient_form_count")),
            "hint_local_index": hint,
            "kernel_lane_hash_u64": 0,
            "lane_class": lane.get("lane_class"),
            "lane_class_code": LANE_CLASS_CODES.get(str(lane.get("lane_class")), 0),
            "lane_work_hash_u64": as_int(lane.get("lane_work_hash_u64")),
            "materialization_required": bool(voter.get("materialization_required")),
            "source_public_key_verified": bool(voter.get("source_public_key_verified")),
            "source_row_hash_u64": as_int(lane.get("source_row_hash_u64")),
            "source_voter_hash_u64": as_int(voter.get("source_voter_hash_u64")),
            "source_voter_secret": as_int(voter.get("derived_secret")),
            "synthesis_required_slot_count": sum(
                count for status, count in term_statuses.items() if status != TERM_STATUS_SOURCE_GUARD
            ),
            "term_slot_count": as_int(lane.get("term_slot_count")),
            "term_status_counts": dict(sorted(term_statuses.items())),
        }
        item["kernel_lane_hash_u64"] = stable_hash_u64({key: value for key, value in item.items() if key != "kernel_lane_hash_u64"})
        lanes.append(item)
    return lanes


def build_term_groups(slots: list[dict[str, Any]], selected_support_mask: int) -> list[dict[str, Any]]:
    by_term: dict[int, list[dict[str, Any]]] = {}
    for slot in slots:
        by_term.setdefault(as_int(slot.get("term"), -1), []).append(slot)
    groups = []
    for group_index, term in enumerate(sorted(by_term)):
        group_slots = sorted(by_term[term], key=lambda slot: as_int(slot.get("hint_local_index"), -1))
        statuses = Counter(str(slot.get("status")) for slot in group_slots)
        lane_bitmap = 0
        lane_hashes = []
        slot_hashes = []
        for slot in group_slots:
            hint = as_int(slot.get("hint_local_index"), -1)
            if hint >= 0:
                lane_bitmap |= 1 << hint
            lane_hashes.append(as_int(slot.get("lane_work_hash_u64")))
            slot_hashes.append(as_int(slot.get("slot_hash_u64")))
        klass = term_group_class(statuses)
        material = {
            "group_index": group_index,
            "lane_bitmap": lane_bitmap,
            "lane_hashes": lane_hashes,
            "slot_hashes": slot_hashes,
            "term": term,
            "term_mask": 1 << term,
            "term_status_counts": dict(sorted(statuses.items())),
        }
        groups.append(
            {
                "ffe_materialization_required": statuses.get(TERM_STATUS_HYBRID_SYNTHESIS, 0) > 0,
                "group_index": group_index,
                "hybrid_materialization_slot_count": statuses.get(TERM_STATUS_HYBRID_SYNTHESIS, 0),
                "lane_bitmap": lane_bitmap,
                "lane_count": len({as_int(slot.get("hint_local_index"), -1) for slot in group_slots}),
                "participating_hint_local_indices": [as_int(slot.get("hint_local_index"), -1) for slot in group_slots],
                "plain_synthesis_slot_count": statuses.get(TERM_STATUS_PLAIN_SYNTHESIS, 0),
                "residual_synthesis_required": any(status != TERM_STATUS_SOURCE_GUARD for status in statuses),
                "selected_support_member": bool((1 << term) & selected_support_mask),
                "source_guard_slot_count": statuses.get(TERM_STATUS_SOURCE_GUARD, 0),
                "synthesis_required_slot_count": sum(
                    count for status, count in statuses.items() if status != TERM_STATUS_SOURCE_GUARD
                ),
                "term": term,
                "term_group_class": klass,
                "term_group_class_code": TERM_GROUP_CLASS_CODES.get(klass, 0),
                "term_group_hash_u64": stable_hash_u64(material),
                "term_mask": 1 << term,
                "term_slot_count": len(group_slots),
                "term_status_counts": dict(sorted(statuses.items())),
                "worker_action": worker_action_for_group(klass),
            }
        )
    return groups


def validate_sources(gate: dict[str, Any], worklist: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if gate.get("claim_status") != "SELECTED13_PRIORITY0_TARGET_EXPORT_VERIFIER_GATE_READY":
        failures.append({"code": "target_gate_not_ready", "claim_status": gate.get("claim_status")})
    if gate.get("failures"):
        failures.append({"code": "target_gate_has_failures", "failures": gate.get("failures")})
    if (gate.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "target_gate_summary_not_verified", "summary": gate.get("summary")})
    if (gate.get("native_preflight") or {}).get("verified") is not True:
        failures.append({"code": "target_gate_native_preflight_not_verified"})
    if worklist.get("claim_status") != "SELECTED13_PRIORITY0_RESIDUAL_SYNTHESIS_WORKLIST_READY":
        failures.append({"code": "synthesis_worklist_not_ready", "claim_status": worklist.get("claim_status")})
    if worklist.get("failures"):
        failures.append({"code": "synthesis_worklist_has_failures", "failures": worklist.get("failures")})
    target = worklist.get("target_slot") or {}
    if as_int(target.get("transfer_index"), -1) != EXPECTED_TRANSFER:
        failures.append({"code": "target_transfer_unexpected", "observed": target.get("transfer_index")})
    if target.get("row_request_id") != EXPECTED_ROW_REQUEST_ID:
        failures.append({"code": "target_row_request_unexpected", "observed": target.get("row_request_id")})
    gate_accept = gate.get("target_export_acceptance_gate") or {}
    if as_int(gate_accept.get("expected_secret")) != EXPECTED_SECRET:
        failures.append({"code": "gate_expected_secret_unexpected", "observed": gate_accept.get("expected_secret")})
    if gate_accept.get("must_verify_public_key") is not True:
        failures.append({"code": "gate_public_key_verifier_missing"})
    return failures


def validate_contract(
    gate: dict[str, Any],
    worklist: dict[str, Any],
    context: dict[str, Any],
    lanes: list[dict[str, Any]],
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures = validate_sources(gate, worklist)
    summary = worklist.get("summary") or {}
    if context.get("context_status") != "verified":
        failures.append({"code": "target_context_not_verified", "context": context})
    if as_int(gate.get("source_consensus", {}).get("expected_secret")) >= as_int(context.get("base_order")):
        failures.append({"code": "expected_secret_outside_base_order", "context": context})
    if len(lanes) != 3:
        failures.append({"code": "kernel_lane_count_unexpected", "observed": len(lanes)})
    if len(groups) != EXPECTED_FUSED_TERM_GROUP_COUNT:
        failures.append({"code": "fused_term_group_count_unexpected", "observed": len(groups)})
    if sum(as_int(group.get("term_slot_count")) for group in groups) != EXPECTED_TERM_SLOT_COUNT:
        failures.append({"code": "grouped_term_slot_count_unexpected"})
    if sum(as_int(group.get("synthesis_required_slot_count")) for group in groups) != EXPECTED_SYNTHESIS_SLOT_COUNT:
        failures.append({"code": "grouped_synthesis_slot_count_unexpected"})
    if sum(as_int(group.get("hybrid_materialization_slot_count")) for group in groups) != 14:
        failures.append({"code": "grouped_hybrid_slot_count_unexpected"})
    if sum(as_int(group.get("source_guard_slot_count")) for group in groups) != 1:
        failures.append({"code": "grouped_source_guard_slot_count_unexpected"})
    if as_int(summary.get("term_slot_count")) != EXPECTED_TERM_SLOT_COUNT:
        failures.append({"code": "source_worklist_term_count_unexpected", "summary": summary})
    if as_int(summary.get("synthesis_required_term_count")) != EXPECTED_SYNTHESIS_SLOT_COUNT:
        failures.append({"code": "source_worklist_synthesis_count_unexpected", "summary": summary})
    if as_int(summary.get("shared_product_verified_lane_count")) != 2:
        failures.append({"code": "source_worklist_hybrid_lane_count_unexpected", "summary": summary})
    for lane in lanes:
        if as_int(lane.get("kernel_lane_hash_u64")) == 0 or as_int(lane.get("lane_class_code")) == 0:
            failures.append({"code": "kernel_lane_hash_or_class_missing", "lane": lane})
        if as_int(lane.get("source_voter_secret")) != EXPECTED_SECRET:
            failures.append({"code": "kernel_lane_secret_mismatch", "lane": lane})
        if lane.get("source_public_key_verified") is not True:
            failures.append({"code": "kernel_lane_source_not_verified", "lane": lane})
    for group in groups:
        if as_int(group.get("term_group_hash_u64")) == 0 or as_int(group.get("term_mask")) == 0:
            failures.append({"code": "term_group_hash_or_mask_missing", "group": group})
        if group.get("selected_support_member") is not True:
            failures.append({"code": "term_group_not_in_selected_support", "group": group})
        if as_int(group.get("synthesis_required_slot_count")) == 0:
            failures.append({"code": "term_group_without_synthesis_requirement", "group": group})
    return failures


def render_c_header(lanes: list[dict[str, Any]], groups: list[dict[str, Any]], target: dict[str, Any]) -> str:
    lane_lines = []
    for lane in lanes:
        lane_lines.append(
            "  {"
            f"{as_int(lane.get('hint_local_index'))}ULL, "
            f"{as_int(lane.get('lane_class_code'))}ULL, "
            f"{as_int(lane.get('kernel_lane_hash_u64'))}ULL, "
            f"{as_int(lane.get('lane_work_hash_u64'))}ULL, "
            f"{as_int(lane.get('source_voter_hash_u64'))}ULL, "
            f"{as_int(lane.get('source_row_hash_u64'))}ULL, "
            f"{1 if lane.get('source_public_key_verified') else 0}ULL, "
            f"{as_int(lane.get('source_voter_secret'))}ULL, "
            f"{1 if lane.get('materialization_required') else 0}ULL, "
            f"{as_int(lane.get('term_slot_count'))}ULL, "
            f"{as_int(lane.get('synthesis_required_slot_count'))}ULL"
            "},"
        )
    group_lines = []
    for group in groups:
        group_lines.append(
            "  {"
            f"{as_int(group.get('group_index'))}ULL, "
            f"{as_int(group.get('term'))}ULL, "
            f"{as_int(group.get('term_mask'))}ULL, "
            f"{as_int(group.get('term_group_class_code'))}ULL, "
            f"{as_int(group.get('term_group_hash_u64'))}ULL, "
            f"{as_int(group.get('lane_bitmap'))}ULL, "
            f"{as_int(group.get('term_slot_count'))}ULL, "
            f"{as_int(group.get('synthesis_required_slot_count'))}ULL, "
            f"{as_int(group.get('plain_synthesis_slot_count'))}ULL, "
            f"{as_int(group.get('hybrid_materialization_slot_count'))}ULL, "
            f"{as_int(group.get('source_guard_slot_count'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_CONTRACT_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_CONTRACT_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TRANSFER {as_int(target.get('transfer_index'))}ULL
#define SELECTED13_PRIORITY0_RESIDUAL_KERNEL_ROW_REQUEST_U64 {as_int(target.get('row_request_id_u64'))}ULL
#define SELECTED13_PRIORITY0_RESIDUAL_KERNEL_EXPECTED_SECRET {EXPECTED_SECRET}ULL
#define SELECTED13_PRIORITY0_RESIDUAL_KERNEL_LANE_COUNT {len(lanes)}ULL
#define SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TERM_GROUP_COUNT {len(groups)}ULL
#define SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TERM_SLOT_COUNT {sum(as_int(group.get('term_slot_count')) for group in groups)}ULL
#define SELECTED13_PRIORITY0_RESIDUAL_KERNEL_SYNTHESIS_SLOT_COUNT {sum(as_int(group.get('synthesis_required_slot_count')) for group in groups)}ULL
#define SELECTED13_PRIORITY0_RESIDUAL_KERNEL_HYBRID_SLOT_COUNT {sum(as_int(group.get('hybrid_materialization_slot_count')) for group in groups)}ULL
#define SELECTED13_PRIORITY0_RESIDUAL_KERNEL_GUARD_SLOT_COUNT {sum(as_int(group.get('source_guard_slot_count')) for group in groups)}ULL
#define SELECTED13_PRIORITY0_RESIDUAL_KERNEL_ACCEPTED_EXPORT_COUNT 0ULL
#define SELECTED13_PRIORITY0_RESIDUAL_KERNEL_RELATION_DERIVED_ECDLP 0ULL
#define SELECTED13_PRIORITY0_RESIDUAL_KERNEL_POLLARD_RHO_SPEEDUP_CLAIMED 0ULL

#define SELECTED13_RESIDUAL_KERNEL_LANE_SOURCE_GUARD 1ULL
#define SELECTED13_RESIDUAL_KERNEL_LANE_SHARED_PRODUCT 2ULL
#define SELECTED13_RESIDUAL_KERNEL_GROUP_PLAIN_AND_HYBRID 1ULL
#define SELECTED13_RESIDUAL_KERNEL_GROUP_HYBRID 2ULL
#define SELECTED13_RESIDUAL_KERNEL_GROUP_GUARDED_HYBRID 3ULL

typedef struct {{
  uint64_t hint_local_index;
  uint64_t lane_class_code;
  uint64_t kernel_lane_hash_u64;
  uint64_t lane_work_hash_u64;
  uint64_t source_voter_hash_u64;
  uint64_t source_row_hash_u64;
  uint64_t source_public_key_verified;
  uint64_t source_voter_secret;
  uint64_t materialization_required;
  uint64_t term_slot_count;
  uint64_t synthesis_required_slot_count;
}} selected13_priority0_residual_kernel_lane_t;

typedef struct {{
  uint64_t group_index;
  uint64_t term;
  uint64_t term_mask;
  uint64_t term_group_class_code;
  uint64_t term_group_hash_u64;
  uint64_t lane_bitmap;
  uint64_t term_slot_count;
  uint64_t synthesis_required_slot_count;
  uint64_t plain_synthesis_slot_count;
  uint64_t hybrid_materialization_slot_count;
  uint64_t source_guard_slot_count;
}} selected13_priority0_residual_kernel_term_group_t;

static const selected13_priority0_residual_kernel_lane_t SELECTED13_PRIORITY0_RESIDUAL_KERNEL_LANES[] = {{
{chr(10).join(lane_lines)}
}};

static const selected13_priority0_residual_kernel_term_group_t SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TERM_GROUPS[] = {{
{chr(10).join(group_lines)}
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
  uint64_t source_guard_lane_count = 0;
  uint64_t shared_product_lane_count = 0;
  uint64_t materialization_lane_count = 0;
  uint64_t verified_lane_count = 0;
  uint64_t matching_secret_lane_count = 0;
  uint64_t grouped_slot_count = 0;
  uint64_t grouped_synthesis_count = 0;
  uint64_t grouped_plain_count = 0;
  uint64_t grouped_hybrid_count = 0;
  uint64_t grouped_guard_count = 0;
  uint64_t guarded_hybrid_group_count = 0;

  const size_t lane_count = sizeof(SELECTED13_PRIORITY0_RESIDUAL_KERNEL_LANES) / sizeof(SELECTED13_PRIORITY0_RESIDUAL_KERNEL_LANES[0]);
  const size_t group_count = sizeof(SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TERM_GROUPS) / sizeof(SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TERM_GROUPS[0]);
  if (SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TRANSFER != {EXPECTED_TRANSFER}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESIDUAL_KERNEL_ROW_REQUEST_U64 == 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESIDUAL_KERNEL_EXPECTED_SECRET != {EXPECTED_SECRET}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESIDUAL_KERNEL_LANE_COUNT != 3ULL || lane_count != 3ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TERM_GROUP_COUNT != {EXPECTED_FUSED_TERM_GROUP_COUNT}ULL || group_count != {EXPECTED_FUSED_TERM_GROUP_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TERM_SLOT_COUNT != {EXPECTED_TERM_SLOT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESIDUAL_KERNEL_SYNTHESIS_SLOT_COUNT != {EXPECTED_SYNTHESIS_SLOT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESIDUAL_KERNEL_HYBRID_SLOT_COUNT != 14ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESIDUAL_KERNEL_GUARD_SLOT_COUNT != 1ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESIDUAL_KERNEL_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESIDUAL_KERNEL_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESIDUAL_KERNEL_POLLARD_RHO_SPEEDUP_CLAIMED != 0ULL) failure_count++;

  for (size_t i = 0; i < lane_count; i++) {{
    const selected13_priority0_residual_kernel_lane_t *lane = &SELECTED13_PRIORITY0_RESIDUAL_KERNEL_LANES[i];
    if (lane->kernel_lane_hash_u64 == 0ULL || lane->lane_work_hash_u64 == 0ULL || lane->source_voter_hash_u64 == 0ULL) failure_count++;
    if (lane->source_row_hash_u64 == 0ULL) failure_count++;
    if (lane->source_public_key_verified) verified_lane_count++;
    if (lane->source_voter_secret == SELECTED13_PRIORITY0_RESIDUAL_KERNEL_EXPECTED_SECRET) matching_secret_lane_count++;
    if (lane->lane_class_code == SELECTED13_RESIDUAL_KERNEL_LANE_SOURCE_GUARD) {{
      source_guard_lane_count++;
      if (lane->materialization_required != 0ULL) failure_count++;
    }} else if (lane->lane_class_code == SELECTED13_RESIDUAL_KERNEL_LANE_SHARED_PRODUCT) {{
      shared_product_lane_count++;
      if (lane->materialization_required == 0ULL) failure_count++;
      materialization_lane_count++;
    }} else {{
      failure_count++;
    }}
  }}

  for (size_t i = 0; i < group_count; i++) {{
    const selected13_priority0_residual_kernel_term_group_t *group = &SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TERM_GROUPS[i];
    if (group->term_mask == 0ULL || group->term_group_hash_u64 == 0ULL || group->lane_bitmap == 0ULL) failure_count++;
    if (group->synthesis_required_slot_count == 0ULL) failure_count++;
    grouped_slot_count += group->term_slot_count;
    grouped_synthesis_count += group->synthesis_required_slot_count;
    grouped_plain_count += group->plain_synthesis_slot_count;
    grouped_hybrid_count += group->hybrid_materialization_slot_count;
    grouped_guard_count += group->source_guard_slot_count;
    if (group->term_group_class_code == SELECTED13_RESIDUAL_KERNEL_GROUP_GUARDED_HYBRID) guarded_hybrid_group_count++;
  }}
  if (source_guard_lane_count != 1ULL) failure_count++;
  if (shared_product_lane_count != 2ULL) failure_count++;
  if (materialization_lane_count != 2ULL) failure_count++;
  if (verified_lane_count != 3ULL) failure_count++;
  if (matching_secret_lane_count != 3ULL) failure_count++;
  if (grouped_slot_count != SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TERM_SLOT_COUNT) failure_count++;
  if (grouped_synthesis_count != SELECTED13_PRIORITY0_RESIDUAL_KERNEL_SYNTHESIS_SLOT_COUNT) failure_count++;
  if (grouped_plain_count != 5ULL) failure_count++;
  if (grouped_hybrid_count != SELECTED13_PRIORITY0_RESIDUAL_KERNEL_HYBRID_SLOT_COUNT) failure_count++;
  if (grouped_guard_count != SELECTED13_PRIORITY0_RESIDUAL_KERNEL_GUARD_SLOT_COUNT) failure_count++;
  if (guarded_hybrid_group_count != 1ULL) failure_count++;

  printf("selected13_priority0_ffe_residual_kernel_contract_preflight transfer=%llu lanes=%llu term_groups=%llu slots=%llu synthesis_slots=%llu hybrid_slots=%llu guard_slots=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TRANSFER,
         (unsigned long long)lane_count,
         (unsigned long long)group_count,
         (unsigned long long)SELECTED13_PRIORITY0_RESIDUAL_KERNEL_TERM_SLOT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_RESIDUAL_KERNEL_SYNTHESIS_SLOT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_RESIDUAL_KERNEL_HYBRID_SLOT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_RESIDUAL_KERNEL_GUARD_SLOT_COUNT,
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
    with tempfile.TemporaryDirectory(prefix="selected13_priority0_residual_kernel_", dir=str(temp_root)) as tmp:
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
    lanes: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    context: dict[str, Any],
    failures: list[dict[str, Any]],
    native_preflight: dict[str, Any],
) -> dict[str, Any]:
    group_classes = Counter(str(group.get("term_group_class")) for group in groups)
    return {
        "accepted_relation_export_count": 0,
        "failure_count": len(failures),
        "fused_term_group_count": len(groups),
        "generic_rho_steps": as_int(context.get("generic_rho_steps")),
        "hybrid_materialization_slot_count": sum(as_int(group.get("hybrid_materialization_slot_count")) for group in groups),
        "kernel_lane_count": len(lanes),
        "native_preflight_verified": bool(native_preflight.get("verified")),
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "rho_comparison_status": "contract_only_no_executable_ops_over_rho_claim",
        "source_guard_slot_count": sum(as_int(group.get("source_guard_slot_count")) for group in groups),
        "structural_fanout_saving_slots": EXPECTED_SYNTHESIS_SLOT_COUNT
        - sum(1 for group in groups if as_int(group.get("synthesis_required_slot_count")) > 0),
        "term_group_class_counts": dict(sorted(group_classes.items())),
        "term_slot_count": sum(as_int(group.get("term_slot_count")) for group in groups),
        "synthesis_required_slot_count": sum(as_int(group.get("synthesis_required_slot_count")) for group in groups),
        "verified": not failures,
        "worker_interpretation": (
            "The contract fuses 19 synthesis-required residual slots into seven unique term "
            "groups while preserving lane fanout, one source guard, and fourteen hybrid "
            "coefficient-materialization slots. It remains a pre-export FFE/summation worker "
            "contract."
        ),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    target_gate_path = Path(args.target_gate)
    synthesis_path = Path(args.synthesis_worklist)
    gate = load_json(target_gate_path)
    worklist = load_json(synthesis_path)
    target = worklist.get("target_slot") or {}
    context = target_context(target, Path(args.task_dir))
    lanes = build_kernel_lanes(worklist, gate)
    groups = build_term_groups(all_term_slots(worklist), as_int(target.get("selected_support_mask")))
    failures = validate_contract(gate, worklist, context, lanes, groups)
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_CONTRACT_READY"
            if not failures
            else "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_CONTRACT_FAILED"
        ),
        "parameters": {
            "synthesis_worklist": str(synthesis_path),
            "target_gate": str(target_gate_path),
            "task_dir": str(args.task_dir),
        },
        "packet_hash_u64": stable_hash_u64({"target": target, "lanes": lanes, "term_groups": groups}),
        "target_slot": target,
        "target_context": context,
        "kernel_lanes": lanes,
        "fused_term_groups": groups,
        "kernel_contract": {
            "accepted_relation_export_count": 0,
            "expected_secret": EXPECTED_SECRET,
            "fresh_ffe_or_summation_polynomial_required": True,
            "fused_residual_term_group_count": len(groups),
            "must_emit_target_direct_rank_export": True,
            "must_match_row_request_id": target.get("row_request_id"),
            "must_materialize_hybrid_coeff_lane_count": 2,
            "must_preserve_target_export_verifier_gate": True,
            "must_reject_source_hint_replay_as_target_export": True,
            "must_set_relation_derived_ecdlp": True,
            "must_synthesize_required_slot_count": sum(as_int(group.get("synthesis_required_slot_count")) for group in groups),
            "must_verify_public_key": True,
            "relation_derived_ecdlp": False,
            "target_export_verifier_gate": str(target_gate_path),
        },
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "summation_polynomial_evaluated": False,
            "target_export_not_emitted_by_this_contract": True,
        },
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-gate", default=str(DEFAULT_TARGET_GATE))
    parser.add_argument("--synthesis-worklist", default=str(DEFAULT_SYNTHESIS_WORKLIST))
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["kernel_lanes"], payload["fused_term_groups"], payload["target_slot"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_CONTRACT_FAILED"
    payload["summary"] = summarize(
        payload["kernel_lanes"],
        payload["fused_term_groups"],
        payload["target_context"],
        payload["failures"],
        native_preflight,
    )
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
