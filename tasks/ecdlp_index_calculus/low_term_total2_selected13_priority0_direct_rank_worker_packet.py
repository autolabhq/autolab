#!/usr/bin/env python3
"""Emit a focused direct-rank worker packet for the top materialization row.

The bridge-extension queue ranks transfer 10376 as the next selected13
materialization target.  This script extracts one priority row, its direct-
verified source hints, full-family masks, and the 10595 positive controls into a
small native-checkable packet for the lower-level FFE/summation-polynomial
direct-rank worker.

This is still a worker input contract.  It does not evaluate summation
polynomials, emit a direct/rank row, derive a new ECDLP secret, or claim a
Pollard-rho speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_direct_rank_worker_packet.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_QUEUE = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_bridge_extension_queue_112_full147_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_direct_rank_worker_packet_10376_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_direct_rank_worker_packet_10376_probe.h"
SELECTED13_MASK = 1 << 13


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


def stable_hash_u64(raw: Any) -> int:
    blob = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def scaled_ops(value: Any) -> int:
    number = as_float(value)
    return 0 if number is None else round(number * 100_000_000)


def terms_from_mask(mask: int) -> list[int]:
    return [index for index in range(64) if as_int(mask) & (1 << index)]


def support_mask(raw: Any) -> int:
    mask = 0
    for item in raw or []:
        value = as_int(item, -1)
        if value >= 0:
            mask |= 1 << value
    return mask


def c_u64_array(values: list[int]) -> str:
    if not values:
        return "{0ULL}"
    return "{" + ", ".join(f"{as_int(value)}ULL" for value in values) + "}"


def select_case(queue: dict[str, Any], priority_rank: int) -> dict[str, Any] | None:
    for case in queue.get("extension_cases") or []:
        if isinstance(case, dict) and as_int(case.get("priority_rank"), -1) == priority_rank:
            return case
    return None


def build_target(case: dict[str, Any]) -> dict[str, Any]:
    selected_support_mask = as_int(case.get("selected_support_mask"))
    family_masks = [as_int(mask) for mask in case.get("matched_family_masks") or []]
    return {
        "candidate_case_arg": case.get("candidate_case_arg"),
        "classification": case.get("classification"),
        "classification_code": as_int(case.get("classification_code")),
        "direct_ops_over_rho": as_float(case.get("direct_ops_over_rho")),
        "direct_ops_over_rho_scaled": scaled_ops(case.get("direct_ops_over_rho")),
        "family_lanes": [
            {
                "family_index": index,
                "family_mask": mask,
                "family_terms": terms_from_mask(mask),
                "selected_support_covers_family": (selected_support_mask & mask) == mask,
            }
            for index, mask in enumerate(family_masks)
        ],
        "global_row_index": as_int(case.get("global_row_index"), -1),
        "is_best_manifest_row": bool(case.get("is_best_manifest_row")),
        "packet_index": as_int(case.get("packet_index"), -1),
        "priority_rank": as_int(case.get("priority_rank"), -1),
        "priority_score": as_int(case.get("priority_score")),
        "required_output": {
            "must_emit_direct_rank_export": True,
            "must_match_row_request_id": case.get("row_request_id"),
            "must_set_relation_derived_ecdlp": True,
            "must_verify_public_key": True,
            "no_claim_from_source_hint_alone": True,
        },
        "row_keys": case.get("row_keys") or [],
        "row_request_id": case.get("row_request_id"),
        "row_request_id_u64": as_int(case.get("row_request_id_u64")),
        "selected_support_mask": selected_support_mask,
        "selected_term_support": [as_int(term) for term in case.get("selected_term_support") or []],
        "selector": case.get("selector"),
        "target": case.get("target"),
        "target_salts": [as_int(salt) for salt in case.get("target_salts") or []],
        "top_k": as_int(case.get("top_k")),
        "transfer_index": as_int(case.get("transfer_index"), -1),
        "worker_acceptance_gate": case.get("worker_acceptance_gate"),
        "worker_class": case.get("worker_class"),
        "worker_class_code": as_int(case.get("worker_class_code")),
    }


def build_hint_lanes(case: dict[str, Any]) -> list[dict[str, Any]]:
    target_mask = as_int(case.get("selected_support_mask"))
    lanes = []
    for hint in case.get("hints") or []:
        if not isinstance(hint, dict):
            continue
        source_mask = as_int(hint.get("source_selected_support_mask"))
        missing_mask = target_mask & ~source_mask
        extra_mask = source_mask & ~target_mask
        lanes.append(
            {
                "direct_ops_over_rho": as_float(hint.get("direct_ops_over_rho")),
                "direct_ops_over_rho_scaled": as_int(hint.get("direct_ops_over_rho_scaled"))
                or scaled_ops(hint.get("direct_ops_over_rho")),
                "direct_public_key_verified": bool(hint.get("direct_public_key_verified")),
                "extra_source_support_mask": extra_mask,
                "extra_source_terms": terms_from_mask(extra_mask),
                "hint_hash_u64": as_int(hint.get("hint_hash_u64")),
                "hint_local_index": as_int(hint.get("hint_local_index"), -1),
                "missing_target_support_mask": missing_mask,
                "missing_target_terms": terms_from_mask(missing_mask),
                "public_product_gate_selected": bool(hint.get("public_product_gate_selected")),
                "salt_delta_min": as_int(hint.get("salt_delta_min")),
                "salt_overlap_count": as_int(hint.get("salt_overlap_count")),
                "same_salt_pair": as_int(hint.get("salt_overlap_count")) == len(case.get("target_salts") or []),
                "shared_product_public_key_verified": bool(hint.get("shared_product_public_key_verified")),
                "source_row_hash_u64": as_int(hint.get("source_row_hash_u64")),
                "source_row_keys": hint.get("source_row_keys") or [],
                "source_selected_support_mask": source_mask,
                "source_selector": hint.get("source_selector"),
                "source_selector_u64": as_int(hint.get("source_selector_u64")) or stable_hash_u64(hint.get("source_selector")),
                "source_support_delta_popcount": as_int(hint.get("source_support_delta_popcount")),
                "source_support_jaccard_scaled_1e6": as_int(hint.get("source_support_jaccard_scaled_1e6")),
                "source_support_overlap_count": as_int(hint.get("source_support_overlap_count")),
                "source_terms": terms_from_mask(source_mask),
                "source_top_k": as_int(hint.get("source_top_k")),
                "target_selector": case.get("selector"),
                "target_top_k": as_int(case.get("top_k")),
            }
        )
    return lanes


def build_controls(queue: dict[str, Any]) -> list[dict[str, Any]]:
    controls = []
    for row in queue.get("control_cases") or []:
        if not isinstance(row, dict):
            continue
        controls.append(
            {
                "classification": row.get("classification"),
                "derived_secret": as_int(row.get("derived_secret")),
                "relation_derived_ecdlp": bool(row.get("relation_derived_ecdlp")),
                "row_request_id": row.get("row_request_id"),
                "row_request_id_u64": as_int(row.get("row_request_id_u64")),
                "selected_support_mask": as_int(row.get("selected_support_mask")),
                "transfer_index": as_int(row.get("transfer_index"), -1),
            }
        )
    return controls


def validate_packet(target: dict[str, Any], hints: list[dict[str, Any]], controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if as_int(target.get("priority_rank"), -1) != 0:
        failures.append({"code": "target_priority_rank_not_zero", "priority_rank": target.get("priority_rank")})
    if not target.get("is_best_manifest_row"):
        failures.append({"code": "target_not_best_manifest_row"})
    if target.get("worker_class") != "hinted_delta_source_solve":
        failures.append({"code": "target_worker_class_unexpected", "worker_class": target.get("worker_class")})
    if target.get("classification") != "FRESH_DIRECT_RELATION_REQUIRED_NO_RELATIONS":
        failures.append({"code": "target_classification_unexpected", "classification": target.get("classification")})
    if not (as_int(target.get("selected_support_mask")) & SELECTED13_MASK):
        failures.append({"code": "target_support_missing_selected13"})
    if len(hints) != 3:
        failures.append({"code": "hint_lane_count_unexpected", "observed": len(hints)})
    if len(target.get("family_lanes") or []) != 3:
        failures.append({"code": "family_lane_count_unexpected", "observed": len(target.get("family_lanes") or [])})
    for hint in hints:
        if not hint.get("direct_public_key_verified"):
            failures.append({"code": "hint_not_direct_verified", "hint_local_index": hint.get("hint_local_index")})
        if as_int(hint.get("missing_target_support_mask")) == 0:
            failures.append({"code": "hint_has_no_target_delta", "hint_local_index": hint.get("hint_local_index")})
        if not hint.get("same_salt_pair"):
            failures.append({"code": "hint_not_same_salt_pair", "hint_local_index": hint.get("hint_local_index")})
        if as_int(hint.get("extra_source_support_mask")) != 0:
            failures.append({"code": "hint_source_has_extra_support", "hint_local_index": hint.get("hint_local_index")})
    if len(controls) < 3:
        failures.append({"code": "control_count_below_expected", "observed": len(controls)})
    for control in controls:
        if not control.get("relation_derived_ecdlp") or as_int(control.get("derived_secret")) == 0:
            failures.append({"code": "control_not_relation_derived", "row_request_id": control.get("row_request_id")})
    return failures


def summarize(target: dict[str, Any], hints: list[dict[str, Any]], controls: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "accepted_relation_export_count": 0,
        "control_relation_derived_row_count": len(controls),
        "control_relation_derived_transfers": sorted({as_int(control.get("transfer_index"), -1) for control in controls}),
        "failure_count": len(failures),
        "family_lane_count": len(target.get("family_lanes") or []),
        "hint_lane_count": len(hints),
        "hint_missing_target_term_sets": [hint.get("missing_target_terms") for hint in hints],
        "hint_same_salt_pair_count": sum(1 for hint in hints if hint.get("same_salt_pair")),
        "pollard_rho_speedup_claimed": False,
        "priority_rank": as_int(target.get("priority_rank"), -1),
        "relation_derived_ecdlp": False,
        "target_row_request_id": target.get("row_request_id"),
        "target_selected_support_mask": as_int(target.get("selected_support_mask")),
        "target_transfer_index": as_int(target.get("transfer_index"), -1),
        "verified": not failures,
        "worker_interpretation": (
            "Execute the target row with fresh FFE/summation-polynomial direct-rank "
            "emission.  The source hints are same-salt direct-verified rows whose "
            "support deltas identify missing target terms, but they are not target "
            "relation exports."
        ),
    }


def render_c_header(target: dict[str, Any], hints: list[dict[str, Any]], controls: list[dict[str, Any]]) -> str:
    family_lines = []
    for lane in target.get("family_lanes") or []:
        family_lines.append(
            "  {"
            f"{as_int(lane.get('family_index'))}ULL, "
            f"{as_int(lane.get('family_mask'))}ULL, "
            f"{1 if lane.get('selected_support_covers_family') else 0}ULL"
            "},"
        )
    hint_lines = []
    for hint in hints:
        hint_lines.append(
            "  {"
            f"{as_int(hint.get('hint_local_index'))}ULL, "
            f"{as_int(hint.get('hint_hash_u64'))}ULL, "
            f"{as_int(hint.get('source_row_hash_u64'))}ULL, "
            f"{as_int(hint.get('source_selector_u64'))}ULL, "
            f"{as_int(hint.get('source_selected_support_mask'))}ULL, "
            f"{as_int(hint.get('missing_target_support_mask'))}ULL, "
            f"{as_int(hint.get('extra_source_support_mask'))}ULL, "
            f"{as_int(hint.get('source_support_delta_popcount'))}ULL, "
            f"{as_int(hint.get('source_support_overlap_count'))}ULL, "
            f"{as_int(hint.get('source_support_jaccard_scaled_1e6'))}ULL, "
            f"{as_int(hint.get('salt_overlap_count'))}ULL, "
            f"{as_int(hint.get('salt_delta_min'))}ULL, "
            f"{1 if hint.get('direct_public_key_verified') else 0}ULL, "
            f"{1 if hint.get('public_product_gate_selected') else 0}ULL, "
            f"{1 if hint.get('shared_product_public_key_verified') else 0}ULL, "
            f"{as_int(hint.get('direct_ops_over_rho_scaled'))}ULL"
            "},"
        )
    control_lines = []
    for control in controls:
        control_lines.append(
            "  {"
            f"{as_int(control.get('transfer_index'))}ULL, "
            f"{as_int(control.get('row_request_id_u64'))}ULL, "
            f"{as_int(control.get('selected_support_mask'))}ULL, "
            f"{as_int(control.get('derived_secret'))}ULL, "
            f"{1 if control.get('relation_derived_ecdlp') else 0}ULL"
            "},"
        )
    target_salts = [as_int(salt) for salt in target.get("target_salts") or []][:2]
    while len(target_salts) < 2:
        target_salts.append(0)
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_DIRECT_RANK_WORKER_PACKET_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_DIRECT_RANK_WORKER_PACKET_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_TRANSFER {as_int(target.get('transfer_index'))}
#define SELECTED13_PRIORITY0_ROW_REQUEST_U64 {as_int(target.get('row_request_id_u64'))}ULL
#define SELECTED13_PRIORITY0_SELECTED_SUPPORT_MASK {as_int(target.get('selected_support_mask'))}ULL
#define SELECTED13_PRIORITY0_HINT_COUNT {len(hints)}
#define SELECTED13_PRIORITY0_FAMILY_LANE_COUNT {len(target.get('family_lanes') or [])}
#define SELECTED13_PRIORITY0_CONTROL_COUNT {len(controls)}
#define SELECTED13_PRIORITY0_RELATION_EXPORT_COUNT 0
#define SELECTED13_PRIORITY0_RELATION_DERIVED_ECDLP 0

typedef struct {{
  uint64_t priority_rank;
  uint64_t priority_score;
  uint64_t packet_index;
  uint64_t global_row_index;
  uint64_t transfer_index;
  uint64_t row_request_id_u64;
  uint64_t selected_support_mask;
  uint64_t target_salts[2];
  uint64_t direct_ops_over_rho_scaled;
  uint64_t worker_class_code;
  uint64_t classification_code;
}} selected13_priority0_target_t;

typedef struct {{
  uint64_t family_index;
  uint64_t family_mask;
  uint64_t selected_support_covers_family;
}} selected13_priority0_family_lane_t;

typedef struct {{
  uint64_t hint_local_index;
  uint64_t hint_hash_u64;
  uint64_t source_row_hash_u64;
  uint64_t source_selector_u64;
  uint64_t source_selected_support_mask;
  uint64_t missing_target_support_mask;
  uint64_t extra_source_support_mask;
  uint64_t support_delta_popcount;
  uint64_t support_overlap_count;
  uint64_t support_jaccard_scaled_1e6;
  uint64_t salt_overlap_count;
  uint64_t salt_delta_min;
  uint64_t direct_public_key_verified;
  uint64_t public_product_gate_selected;
  uint64_t shared_product_public_key_verified;
  uint64_t direct_ops_over_rho_scaled;
}} selected13_priority0_hint_lane_t;

typedef struct {{
  uint64_t transfer_index;
  uint64_t row_request_id_u64;
  uint64_t selected_support_mask;
  uint64_t derived_secret;
  uint64_t relation_derived_ecdlp;
}} selected13_priority0_control_t;

static const selected13_priority0_target_t SELECTED13_PRIORITY0_TARGET = {{
  {as_int(target.get('priority_rank'))}ULL,
  {as_int(target.get('priority_score'))}ULL,
  {as_int(target.get('packet_index'))}ULL,
  {as_int(target.get('global_row_index'))}ULL,
  {as_int(target.get('transfer_index'))}ULL,
  {as_int(target.get('row_request_id_u64'))}ULL,
  {as_int(target.get('selected_support_mask'))}ULL,
  {c_u64_array(target_salts)},
  {as_int(target.get('direct_ops_over_rho_scaled'))}ULL,
  {as_int(target.get('worker_class_code'))}ULL,
  {as_int(target.get('classification_code'))}ULL
}};

static const selected13_priority0_family_lane_t SELECTED13_PRIORITY0_FAMILY_LANES[] = {{
{chr(10).join(family_lines)}
}};

static const selected13_priority0_hint_lane_t SELECTED13_PRIORITY0_HINT_LANES[] = {{
{chr(10).join(hint_lines)}
}};

static const selected13_priority0_control_t SELECTED13_PRIORITY0_CONTROLS[] = {{
{chr(10).join(control_lines)}
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
  const uint64_t selected13_mask = {SELECTED13_MASK}ULL;
  const size_t family_count = sizeof(SELECTED13_PRIORITY0_FAMILY_LANES) / sizeof(SELECTED13_PRIORITY0_FAMILY_LANES[0]);
  const size_t hint_count = sizeof(SELECTED13_PRIORITY0_HINT_LANES) / sizeof(SELECTED13_PRIORITY0_HINT_LANES[0]);
  const size_t control_count = sizeof(SELECTED13_PRIORITY0_CONTROLS) / sizeof(SELECTED13_PRIORITY0_CONTROLS[0]);
  uint64_t direct_verified_hints = 0;
  uint64_t same_salt_hints = 0;
  uint64_t missing_delta_hints = 0;
  uint64_t verified_controls = 0;

  if (SELECTED13_PRIORITY0_TARGET.priority_rank != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TARGET.transfer_index != SELECTED13_PRIORITY0_TRANSFER) failure_count++;
  if (SELECTED13_PRIORITY0_TARGET.row_request_id_u64 != SELECTED13_PRIORITY0_ROW_REQUEST_U64) failure_count++;
  if ((SELECTED13_PRIORITY0_TARGET.selected_support_mask & selected13_mask) == 0ULL) failure_count++;
  if (family_count != SELECTED13_PRIORITY0_FAMILY_LANE_COUNT) failure_count++;
  if (hint_count != SELECTED13_PRIORITY0_HINT_COUNT) failure_count++;
  if (control_count != SELECTED13_PRIORITY0_CONTROL_COUNT) failure_count++;
  if (SELECTED13_PRIORITY0_RELATION_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;

  for (size_t i = 0; i < family_count; i++) {{
    const selected13_priority0_family_lane_t *lane = &SELECTED13_PRIORITY0_FAMILY_LANES[i];
    if (lane->family_mask == 0ULL || lane->selected_support_covers_family == 0ULL) failure_count++;
  }}
  for (size_t i = 0; i < hint_count; i++) {{
    const selected13_priority0_hint_lane_t *hint = &SELECTED13_PRIORITY0_HINT_LANES[i];
    if (hint->hint_hash_u64 == 0ULL || hint->source_row_hash_u64 == 0ULL) failure_count++;
    if (hint->source_selected_support_mask == 0ULL) failure_count++;
    if (hint->extra_source_support_mask != 0ULL) failure_count++;
    if (hint->missing_target_support_mask != 0ULL) missing_delta_hints++;
    if (hint->salt_overlap_count == 2ULL && hint->salt_delta_min == 0ULL) same_salt_hints++;
    if (hint->direct_public_key_verified) direct_verified_hints++;
  }}
  for (size_t i = 0; i < control_count; i++) {{
    const selected13_priority0_control_t *control = &SELECTED13_PRIORITY0_CONTROLS[i];
    if (control->row_request_id_u64 == 0ULL || control->derived_secret == 0ULL) failure_count++;
    if (control->relation_derived_ecdlp) verified_controls++;
  }}
  if (direct_verified_hints != hint_count) failure_count++;
  if (same_salt_hints != hint_count) failure_count++;
  if (missing_delta_hints != hint_count) failure_count++;
  if (verified_controls != control_count) failure_count++;

  printf("selected13_priority0_direct_rank_worker_packet_preflight transfer=%llu hints=%llu family_lanes=%llu same_salt_hints=%llu controls=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_TRANSFER,
         (unsigned long long)hint_count,
         (unsigned long long)family_count,
         (unsigned long long)same_salt_hints,
         (unsigned long long)control_count,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_priority0_direct_rank_worker_packet_") as tmp:
        tmp_path = Path(tmp)
        c_path = tmp_path / "preflight.c"
        exe_path = tmp_path / "preflight"
        local_header = tmp_path / header_path.name
        c_path.write_text(source)
        local_header.write_text(header_path.read_text())
        compile_cmd = ["cc", "-std=c99", "-Wall", "-Wextra", "-O2", str(c_path), "-o", str(exe_path)]
        compile_run = subprocess.run(compile_cmd, text=True, capture_output=True, check=False)
        if compile_run.returncode != 0:
            return {
                "compile_command": compile_cmd,
                "compile_returncode": compile_run.returncode,
                "compile_stderr": compile_run.stderr,
                "verified": False,
            }
        preflight_run = subprocess.run([str(exe_path)], text=True, capture_output=True, check=False)
        return {
            "compile_command": compile_cmd,
            "compile_returncode": compile_run.returncode,
            "preflight_returncode": preflight_run.returncode,
            "preflight_stdout": preflight_run.stdout.strip(),
            "preflight_stderr": preflight_run.stderr.strip(),
            "verified": preflight_run.returncode == 0,
        }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    queue_path = Path(args.queue)
    queue = load_json(queue_path)
    failures: list[dict[str, Any]] = []
    if queue.get("claim_status") != "SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_QUEUE_READY":
        failures.append({"code": "queue_not_ready", "claim_status": queue.get("claim_status")})
    if queue.get("failures"):
        failures.append({"code": "queue_has_failures", "failures": queue.get("failures")})
    case = select_case(queue, args.priority_rank)
    if case is None:
        failures.append({"code": "priority_case_missing", "priority_rank": args.priority_rank})
        case = {}
    target = build_target(case)
    hint_lanes = build_hint_lanes(case)
    controls = build_controls(queue)
    failures.extend(validate_packet(target, hint_lanes, controls))
    summary = summarize(target, hint_lanes, controls, failures)
    packet_hash = stable_hash_u64(
        {
            "controls": controls,
            "hint_lanes": hint_lanes,
            "target": target,
        }
    )
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_DIRECT_RANK_WORKER_PACKET_READY"
            if not failures
            else "SELECTED13_PRIORITY0_DIRECT_RANK_WORKER_PACKET_FAILED"
        ),
        "parameters": {
            "priority_rank": args.priority_rank,
            "queue": str(queue_path),
        },
        "packet_hash_u64": packet_hash,
        "source_summary": queue.get("summary"),
        "summary": summary,
        "target": target,
        "hint_lanes": hint_lanes,
        "positive_controls": controls,
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "fresh_direct_rank_worker_required": True,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "source_hints_are_not_target_row_exports": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE), help="Bridge-extension queue JSON")
    parser.add_argument("--priority-rank", type=int, default=0, help="Queue priority rank to lower")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output worker packet JSON")
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT), help="Output C preflight header")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    out_path = Path(args.out)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["target"], payload["hint_lanes"], payload["positive_controls"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["claim_status"] = "SELECTED13_PRIORITY0_DIRECT_RANK_WORKER_PACKET_FAILED"
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
    write_json(out_path, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(out_path), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
