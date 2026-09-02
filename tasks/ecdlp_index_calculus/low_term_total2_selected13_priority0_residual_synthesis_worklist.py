#!/usr/bin/env python3
"""Emit per-term residual synthesis work items for selected13 priority-0.

The residual coefficient-gap manifest shows that transfer 10376 has one
coefficient-backed residual lane and two shared-product verified lanes whose
coefficient forms are not materialized.  This script lowers that into a
term-level worklist for a lower-level FFE/summation-polynomial worker:
coefficient materialization tasks, residual-term synthesis slots, and the
shared-product row profile that tells the worker where the verified hybrid
source relation came from.

This is a synthesis worklist only.  It does not materialize missing
coefficients, evaluate summation polynomials, emit a target direct/rank row,
solve ECDLP, or claim a Pollard-rho speedup.
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


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_residual_synthesis_worklist.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_COEFF_GAP = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_residual_coeff_gap_manifest_10376_probe.json"
DEFAULT_SHARED_PRODUCT_GATE = Path(
    "/Volumes/Volume/autolab/ecdlp_index_calculus_state/"
    "low_term_total2_fixed_leaf_shared_product_gate_10376_10383_col15_selector_expanded_density_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_residual_synthesis_worklist_10376_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_residual_synthesis_worklist_10376_probe.h"

SELECTED13_MASK = 1 << 13
TERM_STATUS_CODES = {
    "source_coefficient_guard_available": 1,
    "needs_residual_synthesis": 2,
    "needs_hybrid_coefficient_materialization_and_residual_synthesis": 3,
}
LANE_CLASS_CODES = {
    "source_coeff_guard_partial_residual_synthesis": 1,
    "shared_product_verified_coeff_materialization": 2,
    "unverified_or_missing_source": 3,
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


def normalize_row_keys(raw: Any) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in raw or []))


def lane_match_key(lane: dict[str, Any]) -> tuple[Any, ...]:
    return (
        lane.get("source_selector"),
        as_int(lane.get("source_top_k")),
        normalize_row_keys((lane.get("target_slot") or {}).get("row_keys") or lane.get("source_row_keys")),
    )


def shared_case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        case.get("selector"),
        as_int(case.get("top_k")),
        normalize_row_keys(case.get("row_keys")),
    )


def terms_from_mask(mask: int) -> list[int]:
    return [index for index in range(64) if as_int(mask) & (1 << index)]


def support_mask(raw: Any) -> int:
    mask = 0
    for item in raw or []:
        value = as_int(item, -1)
        if value >= 0:
            mask |= 1 << value
    return mask


def shared_product_cases_by_key(shared_product_gate: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    out = {}
    for case in shared_product_gate.get("cases") or []:
        if isinstance(case, dict):
            out[shared_case_key(case)] = case
    return out


def row_profile_summary(case: dict[str, Any]) -> list[dict[str, Any]]:
    profile = case.get("shared_product_profile") or {}
    rows = []
    for row in profile.get("row_shared_product_profiles") or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "degree_eval_cost": as_int(row.get("degree_eval_cost")),
                "hit_root_values": as_int(row.get("hit_root_values")),
                "leaf_hits": as_int(row.get("leaf_hits")),
                "node_evals": as_int(row.get("node_evals")),
                "row_key": row.get("row_key"),
                "selected_hit_root_values": as_int(row.get("selected_hit_root_values")),
                "shared_product_gcd_degree": as_int(row.get("shared_product_gcd_degree")),
                "x_matches": as_int(row.get("x_matches")),
            }
        )
    return rows


def lane_class(lane: dict[str, Any], shared_case: dict[str, Any] | None) -> str:
    if as_int(lane.get("coefficient_form_count")) > 0:
        return "source_coeff_guard_partial_residual_synthesis"
    if shared_case and shared_case.get("shared_product_union_public_key_verified"):
        return "shared_product_verified_coeff_materialization"
    return "unverified_or_missing_source"


def build_term_slots(lane: dict[str, Any], klass: str) -> list[dict[str, Any]]:
    residual_terms = lane.get("missing_target_terms") or terms_from_mask(as_int(lane.get("missing_target_support_mask")))
    overlap_terms = set(as_int(term) for term in lane.get("residual_overlap_union_terms") or [])
    slots = []
    for offset, term in enumerate(residual_terms):
        term = as_int(term)
        if term in overlap_terms:
            status = "source_coefficient_guard_available"
        elif klass == "shared_product_verified_coeff_materialization":
            status = "needs_hybrid_coefficient_materialization_and_residual_synthesis"
        else:
            status = "needs_residual_synthesis"
        slots.append(
            {
                "hint_local_index": as_int(lane.get("hint_local_index"), -1),
                "slot_hash_u64": stable_hash_u64(
                    {
                        "hint_local_index": lane.get("hint_local_index"),
                        "status": status,
                        "term": term,
                    }
                ),
                "slot_offset": offset,
                "status": status,
                "status_code": TERM_STATUS_CODES[status],
                "term": term,
                "term_mask": 1 << term,
            }
        )
    return slots


def build_lane_work_items(coeff_gap: dict[str, Any], shared_product_gate: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    shared_index = shared_product_cases_by_key(shared_product_gate)
    target = coeff_gap.get("target_slot") or {}
    out = []
    for lane in coeff_gap.get("residual_coeff_lanes") or []:
        lane = dict(lane)
        lane["target_slot"] = target
        shared_case = shared_index.get(lane_match_key(lane))
        klass = lane_class(lane, shared_case)
        if lane.get("source_class") == "shared_product_verified_coefficients_missing" and shared_case is None:
            failures.append({"code": "shared_product_case_missing", "hint_local_index": lane.get("hint_local_index")})
        term_slots = build_term_slots(lane, klass)
        profile_rows = row_profile_summary(shared_case or {})
        hit_rows = [row for row in profile_rows if as_int(row.get("selected_hit_root_values")) > 0]
        item = {
            "coefficient_form_count": as_int(lane.get("coefficient_form_count")),
            "coefficient_rank_mod_order": as_int(lane.get("coefficient_rank_mod_order")),
            "direct_certificate_artifact": lane.get("direct_certificate_artifact"),
            "hint_hash_u64": as_int(lane.get("hint_hash_u64")),
            "hint_local_index": as_int(lane.get("hint_local_index"), -1),
            "lane_class": klass,
            "lane_class_code": LANE_CLASS_CODES[klass],
            "missing_coefficient_materialization_required": klass == "shared_product_verified_coeff_materialization",
            "residual_synthesis_required": any(slot.get("status") != "source_coefficient_guard_available" for slot in term_slots),
            "shared_product_case_hash_u64": stable_hash_u64(shared_case) if shared_case else 0,
            "shared_product_direct_ops_over_rho": as_float((shared_case or {}).get("direct_verifier_replay_ops_over_rho")),
            "shared_product_hit_row_count": len(hit_rows),
            "shared_product_hit_rows": hit_rows,
            "shared_product_ledger_ops_over_rho": as_float((shared_case or {}).get("shared_product_ledger_ops_over_rho")),
            "shared_product_profile_rows": profile_rows,
            "shared_product_union_derived_secret": (
                None if shared_case is None else as_int(shared_case.get("shared_product_union_derived_secret"))
            ),
            "shared_product_union_public_key_verified": bool((shared_case or {}).get("shared_product_union_public_key_verified")),
            "shared_product_union_rank": as_int((shared_case or {}).get("shared_product_union_rank")),
            "shared_product_union_relation_count": as_int((shared_case or {}).get("shared_product_union_relation_count")),
            "source_class": lane.get("source_class"),
            "source_row_hash_u64": as_int(lane.get("source_row_hash_u64")),
            "source_selected_support_mask": as_int(lane.get("source_selected_support_mask")),
            "source_selector": lane.get("source_selector"),
            "source_top_k": as_int(lane.get("source_top_k")),
            "term_slots": term_slots,
            "term_slot_count": len(term_slots),
            "uncovered_residual_mask": as_int(lane.get("uncovered_residual_mask")),
            "uncovered_residual_terms": lane.get("uncovered_residual_terms") or [],
            "worker_action": (
                "materialize_shared_product_coefficients_then_synthesize_residual_terms"
                if klass == "shared_product_verified_coeff_materialization"
                else "synthesize_uncovered_residual_terms_against_source_coefficient_guard"
            ),
        }
        item["lane_work_hash_u64"] = stable_hash_u64(item)
        out.append(item)
    return out, failures


def validate_sources(coeff_gap: dict[str, Any], shared_product_gate: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if coeff_gap.get("claim_status") != "SELECTED13_PRIORITY0_RESIDUAL_COEFF_GAP_MANIFEST_READY":
        failures.append({"code": "coeff_gap_not_ready", "claim_status": coeff_gap.get("claim_status")})
    if coeff_gap.get("failures"):
        failures.append({"code": "coeff_gap_has_failures", "failures": coeff_gap.get("failures")})
    if (coeff_gap.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "coeff_gap_summary_not_verified", "summary": coeff_gap.get("summary")})
    if (shared_product_gate.get("summary") or {}).get("claim_status") != "PUBLIC_GATE_VALIDATION_POSITIVE":
        failures.append(
            {
                "code": "shared_product_gate_not_positive",
                "claim_status": (shared_product_gate.get("summary") or {}).get("claim_status"),
            }
        )
    return failures


def validate_work_items(items: list[dict[str, Any]], target: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(items) != 3:
        failures.append({"code": "lane_work_item_count_unexpected", "observed": len(items)})
    class_counts = Counter(str(item.get("lane_class")) for item in items)
    if class_counts.get("source_coeff_guard_partial_residual_synthesis", 0) != 1:
        failures.append({"code": "source_coeff_guard_lane_count_unexpected", "class_counts": dict(class_counts)})
    if class_counts.get("shared_product_verified_coeff_materialization", 0) != 2:
        failures.append({"code": "shared_product_lane_count_unexpected", "class_counts": dict(class_counts)})
    term_slots = [slot for item in items for slot in item.get("term_slots") or []]
    if len(term_slots) != 20:
        failures.append({"code": "term_slot_count_unexpected", "observed": len(term_slots)})
    if sum(1 for slot in term_slots if slot.get("status") == "source_coefficient_guard_available") != 1:
        failures.append({"code": "coefficient_guard_term_count_unexpected"})
    if sum(1 for slot in term_slots if slot.get("status") != "source_coefficient_guard_available") != 19:
        failures.append({"code": "synthesis_term_count_unexpected"})
    if sum(1 for slot in term_slots if as_int(slot.get("term_mask")) & SELECTED13_MASK) != 3:
        failures.append({"code": "selected13_term_slot_count_unexpected"})
    for item in items:
        if as_int(item.get("lane_work_hash_u64")) == 0:
            failures.append({"code": "lane_work_hash_missing", "hint": item.get("hint_local_index")})
        if item.get("lane_class") == "shared_product_verified_coeff_materialization":
            if item.get("shared_product_union_public_key_verified") is not True:
                failures.append({"code": "shared_product_lane_not_verified", "hint": item.get("hint_local_index")})
            if as_int(item.get("shared_product_union_rank")) != 2:
                failures.append({"code": "shared_product_rank_unexpected", "hint": item.get("hint_local_index")})
            if as_int(item.get("shared_product_hit_row_count")) != 1:
                failures.append({"code": "shared_product_hit_row_count_unexpected", "hint": item.get("hint_local_index")})
            if as_int(item.get("shared_product_case_hash_u64")) == 0:
                failures.append({"code": "shared_product_case_hash_missing", "hint": item.get("hint_local_index")})
        for slot in item.get("term_slots") or []:
            if as_int(slot.get("slot_hash_u64")) == 0 or as_int(slot.get("term_mask")) == 0:
                failures.append({"code": "term_slot_hash_or_mask_missing", "hint": item.get("hint_local_index")})
    if as_int(target.get("transfer_index"), -1) != 10376:
        failures.append({"code": "target_transfer_unexpected", "transfer": target.get("transfer_index")})
    return failures


def render_c_header(items: list[dict[str, Any]], target: dict[str, Any]) -> str:
    lane_lines = []
    term_lines = []
    term_index = 0
    for item in items:
        start = term_index
        slots = item.get("term_slots") or []
        for slot in slots:
            term_lines.append(
                "  {"
                f"{as_int(item.get('hint_local_index'))}ULL, "
                f"{as_int(slot.get('slot_offset'))}ULL, "
                f"{as_int(slot.get('term'))}ULL, "
                f"{as_int(slot.get('term_mask'))}ULL, "
                f"{as_int(slot.get('status_code'))}ULL, "
                f"{as_int(slot.get('slot_hash_u64'))}ULL"
                "},"
            )
            term_index += 1
        lane_lines.append(
            "  {"
            f"{as_int(item.get('hint_local_index'))}ULL, "
            f"{as_int(item.get('lane_class_code'))}ULL, "
            f"{as_int(item.get('lane_work_hash_u64'))}ULL, "
            f"{as_int(item.get('source_row_hash_u64'))}ULL, "
            f"{as_int(item.get('coefficient_form_count'))}ULL, "
            f"{as_int(item.get('coefficient_rank_mod_order'))}ULL, "
            f"{1 if item.get('missing_coefficient_materialization_required') else 0}ULL, "
            f"{1 if item.get('residual_synthesis_required') else 0}ULL, "
            f"{as_int(item.get('shared_product_union_rank'))}ULL, "
            f"{as_int(item.get('shared_product_union_relation_count'))}ULL, "
            f"{as_int(item.get('shared_product_union_derived_secret'))}ULL, "
            f"{1 if item.get('shared_product_union_public_key_verified') else 0}ULL, "
            f"{as_int(item.get('shared_product_hit_row_count'))}ULL, "
            f"{start}ULL, "
            f"{len(slots)}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_RESIDUAL_SYNTHESIS_WORKLIST_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_RESIDUAL_SYNTHESIS_WORKLIST_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_SYNTHESIS_TRANSFER {as_int(target.get('transfer_index'))}ULL
#define SELECTED13_PRIORITY0_SYNTHESIS_ROW_REQUEST_U64 {as_int(target.get('row_request_id_u64'))}ULL
#define SELECTED13_PRIORITY0_SYNTHESIS_LANE_COUNT {len(items)}
#define SELECTED13_PRIORITY0_SYNTHESIS_TERM_SLOT_COUNT {term_index}
#define SELECTED13_PRIORITY0_SYNTHESIS_ACCEPTED_EXPORT_COUNT 0ULL
#define SELECTED13_PRIORITY0_SYNTHESIS_RELATION_DERIVED_ECDLP 0ULL

#define SELECTED13_SYNTHESIS_LANE_SOURCE_GUARD 1ULL
#define SELECTED13_SYNTHESIS_LANE_SHARED_PRODUCT_MATERIALIZE 2ULL
#define SELECTED13_SYNTHESIS_TERM_SOURCE_GUARD 1ULL
#define SELECTED13_SYNTHESIS_TERM_NEEDS_SYNTHESIS 2ULL
#define SELECTED13_SYNTHESIS_TERM_NEEDS_HYBRID_COEFFS 3ULL

typedef struct {{
  uint64_t hint_local_index;
  uint64_t lane_class_code;
  uint64_t lane_work_hash_u64;
  uint64_t source_row_hash_u64;
  uint64_t coefficient_form_count;
  uint64_t coefficient_rank_mod_order;
  uint64_t missing_coefficient_materialization_required;
  uint64_t residual_synthesis_required;
  uint64_t shared_product_union_rank;
  uint64_t shared_product_union_relation_count;
  uint64_t shared_product_union_derived_secret;
  uint64_t shared_product_union_public_key_verified;
  uint64_t shared_product_hit_row_count;
  uint64_t term_start;
  uint64_t term_count;
}} selected13_priority0_synthesis_lane_t;

typedef struct {{
  uint64_t hint_local_index;
  uint64_t slot_offset;
  uint64_t term;
  uint64_t term_mask;
  uint64_t status_code;
  uint64_t slot_hash_u64;
}} selected13_priority0_synthesis_term_t;

static const selected13_priority0_synthesis_lane_t SELECTED13_PRIORITY0_SYNTHESIS_LANES[] = {{
{chr(10).join(lane_lines)}
}};

static const selected13_priority0_synthesis_term_t SELECTED13_PRIORITY0_SYNTHESIS_TERMS[] = {{
{chr(10).join(term_lines)}
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
  uint64_t failure_count = 0;
  uint64_t source_guard_lane_count = 0;
  uint64_t shared_product_lane_count = 0;
  uint64_t shared_verified_lane_count = 0;
  uint64_t source_guard_term_count = 0;
  uint64_t synthesis_term_count = 0;
  uint64_t selected13_term_count = 0;
  uint64_t lane_term_sum = 0;

  const size_t lane_count = sizeof(SELECTED13_PRIORITY0_SYNTHESIS_LANES) / sizeof(SELECTED13_PRIORITY0_SYNTHESIS_LANES[0]);
  const size_t term_count = sizeof(SELECTED13_PRIORITY0_SYNTHESIS_TERMS) / sizeof(SELECTED13_PRIORITY0_SYNTHESIS_TERMS[0]);
  if (lane_count != SELECTED13_PRIORITY0_SYNTHESIS_LANE_COUNT) failure_count++;
  if (term_count != SELECTED13_PRIORITY0_SYNTHESIS_TERM_SLOT_COUNT) failure_count++;
  if (SELECTED13_PRIORITY0_SYNTHESIS_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_SYNTHESIS_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;

  for (size_t i = 0; i < lane_count; i++) {{
    const selected13_priority0_synthesis_lane_t *lane = &SELECTED13_PRIORITY0_SYNTHESIS_LANES[i];
    if (lane->lane_work_hash_u64 == 0ULL || lane->source_row_hash_u64 == 0ULL) failure_count++;
    if (lane->residual_synthesis_required == 0ULL) failure_count++;
    if (lane->term_start + lane->term_count > term_count) failure_count++;
    lane_term_sum += lane->term_count;
    if (lane->lane_class_code == SELECTED13_SYNTHESIS_LANE_SOURCE_GUARD) {{
      source_guard_lane_count++;
      if (lane->coefficient_form_count == 0ULL || lane->coefficient_rank_mod_order == 0ULL) failure_count++;
      if (lane->missing_coefficient_materialization_required != 0ULL) failure_count++;
    }} else if (lane->lane_class_code == SELECTED13_SYNTHESIS_LANE_SHARED_PRODUCT_MATERIALIZE) {{
      shared_product_lane_count++;
      if (lane->missing_coefficient_materialization_required == 0ULL) failure_count++;
      if (lane->shared_product_union_public_key_verified) shared_verified_lane_count++;
      if (lane->shared_product_union_rank != 2ULL || lane->shared_product_hit_row_count != 1ULL) failure_count++;
    }} else {{
      failure_count++;
    }}
  }}
  for (size_t i = 0; i < term_count; i++) {{
    const selected13_priority0_synthesis_term_t *term = &SELECTED13_PRIORITY0_SYNTHESIS_TERMS[i];
    if (term->term_mask == 0ULL || term->slot_hash_u64 == 0ULL) failure_count++;
    if (term->status_code == SELECTED13_SYNTHESIS_TERM_SOURCE_GUARD) source_guard_term_count++;
    if (term->status_code != SELECTED13_SYNTHESIS_TERM_SOURCE_GUARD) synthesis_term_count++;
    if (term->term_mask & selected13_mask) selected13_term_count++;
  }}
  if (lane_term_sum != term_count) failure_count++;
  if (source_guard_lane_count != 1ULL) failure_count++;
  if (shared_product_lane_count != 2ULL) failure_count++;
  if (shared_verified_lane_count != 2ULL) failure_count++;
  if (source_guard_term_count != 1ULL) failure_count++;
  if (synthesis_term_count != 19ULL) failure_count++;
  if (selected13_term_count != 3ULL) failure_count++;

  printf("selected13_priority0_residual_synthesis_worklist_preflight transfer=%llu lanes=%llu terms=%llu source_guard_terms=%llu synthesis_terms=%llu shared_verified_lanes=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_SYNTHESIS_TRANSFER,
         (unsigned long long)lane_count,
         (unsigned long long)term_count,
         (unsigned long long)source_guard_term_count,
         (unsigned long long)synthesis_term_count,
         (unsigned long long)shared_verified_lane_count,
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
    with tempfile.TemporaryDirectory(prefix="selected13_priority0_synthesis_", dir=str(temp_root)) as tmp:
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


def summarize(items: list[dict[str, Any]], failures: list[dict[str, Any]], native_preflight: dict[str, Any]) -> dict[str, Any]:
    lane_classes = Counter(str(item.get("lane_class")) for item in items)
    term_slots = [slot for item in items for slot in item.get("term_slots") or []]
    term_statuses = Counter(str(slot.get("status")) for slot in term_slots)
    return {
        "accepted_relation_export_count": 0,
        "failure_count": len(failures),
        "lane_class_counts": dict(sorted(lane_classes.items())),
        "lane_count": len(items),
        "native_preflight_verified": bool(native_preflight.get("verified")),
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "shared_product_verified_lane_count": sum(1 for item in items if item.get("shared_product_union_public_key_verified")),
        "synthesis_required_term_count": sum(
            1 for slot in term_slots if slot.get("status") != "source_coefficient_guard_available"
        ),
        "term_slot_count": len(term_slots),
        "term_status_counts": dict(sorted(term_statuses.items())),
        "verified": not failures,
        "worker_interpretation": (
            "The worklist converts the 10376 residual gap into per-term synthesis slots. "
            "It binds the two hybrid lanes to shared-product verified rank-2 source cases, "
            "but still requires coefficient materialization and target-row direct/rank export."
        ),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    coeff_gap_path = Path(args.coeff_gap)
    shared_gate_path = Path(args.shared_product_gate)
    coeff_gap = load_json(coeff_gap_path)
    shared_gate = load_json(shared_gate_path)
    failures = validate_sources(coeff_gap, shared_gate)
    work_items, item_failures = build_lane_work_items(coeff_gap, shared_gate)
    failures.extend(item_failures)
    target = coeff_gap.get("target_slot") or {}
    failures.extend(validate_work_items(work_items, target))
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_RESIDUAL_SYNTHESIS_WORKLIST_READY"
            if not failures
            else "SELECTED13_PRIORITY0_RESIDUAL_SYNTHESIS_WORKLIST_FAILED"
        ),
        "parameters": {
            "coeff_gap": str(coeff_gap_path),
            "shared_product_gate": str(shared_gate_path),
        },
        "packet_hash_u64": stable_hash_u64({"target": target, "work_items": work_items}),
        "target_slot": target,
        "lane_work_items": work_items,
        "required_worker_outputs": {
            "must_materialize_shared_product_coefficients_for_hybrid_lanes": True,
            "must_synthesize_all_non_guard_residual_terms": True,
            "must_emit_target_direct_rank_export": True,
            "must_verify_public_key": True,
            "must_set_relation_derived_ecdlp": True,
        },
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "fresh_direct_rank_worker_required": True,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "shared_product_cases_are_source_guards_not_target_exports": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coeff-gap", default=str(DEFAULT_COEFF_GAP))
    parser.add_argument("--shared-product-gate", default=str(DEFAULT_SHARED_PRODUCT_GATE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["lane_work_items"], payload["target_slot"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_RESIDUAL_SYNTHESIS_WORKLIST_FAILED"
    payload["summary"] = summarize(payload["lane_work_items"], payload["failures"], native_preflight)
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
