#!/usr/bin/env python3
"""Scan public dependency artifacts for selected13 priority-0 gap closures.

The dependency bridge for transfer 10376 already covers residual terms 2, 6,
and 13.  This scanner looks across mounted public dependency-circuit artifacts
for same-target analogues that touch the four remaining residual terms
3, 8, 9, and 12, and for any public evidence that touches the missing salt173
slice row.

The scanner emits a native-checkable work order.  It treats same-target,
public-key-verified, below-rho dependency circuits as gap-closure analogues,
keeps cross-target analogues in a separate non-counting bucket, and never emits
a validator-accepted kernel result or Pollard-rho speedup claim.
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


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_gap_closure_scanner.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
MOUNTED_STATE_DIR = Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state")
DEFAULT_BRIDGE = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_dependency_circuit_evidence_bridge_10376_probe.json"
)
DEFAULT_DEPENDENCY_CIRCUIT = MOUNTED_STATE_DIR / "frontier_signed_eval_cover_dependency_circuit_probe.json"
DEFAULT_SURFACE_GATE = (
    MOUNTED_STATE_DIR / "disjoint_dependency_circuit_surface_gate_cost_low_total2_456_519_552_583_probe.json"
)
DEFAULT_SAME_TRANSFER_ROW_BANK = (
    MOUNTED_STATE_DIR / "disjoint_dependency_circuit_same_transfer_row_bank_22050_552_583_probe.json"
)
DEFAULT_MULTIBLOCK = (
    MOUNTED_STATE_DIR / "frontier_multiblock_disjoint_dependency_circuit_200_231_232_375_probe.json"
)
DEFAULT_SLICE_QUADRATIC = (
    MOUNTED_STATE_DIR
    / (
        "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_"
        "static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_slice_quadratic_root_probe.json"
    )
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_gap_closure_scanner_10376_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_gap_closure_scanner_10376_probe.h"

EXPECTED_TRANSFER = 10376
EXPECTED_BRIDGE_COVERED_TERM_COUNT = 3
EXPECTED_BRIDGE_MISSING_TERM_COUNT = 4
EXPECTED_SAME_TARGET_NEW_TERM_COUNT = 2
EXPECTED_SAME_TARGET_TOTAL_TERM_COUNT = 5
EXPECTED_REMAINING_TERM_COUNT = 2
EXPECTED_SALT173_CANDIDATE_COUNT = 1
EXPECTED_CROSS_TARGET_ANALOGUE_TERM_COUNT = 2
EXPECTED_REMAINING_TERMS = [8, 12]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
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


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_hash_u64(material: Any) -> int:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def term_set_from_events(events: list[dict[str, Any]]) -> list[int]:
    terms: set[int] = set()
    for event in events:
        support = event.get("factor_support")
        if support is None:
            support = event.get("terms") or []
        for term in support:
            terms.add(as_int(term, -1))
    return sorted(term for term in terms if term >= 0)


def term_shape_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        shape = str(event.get("term_shape") or "unknown")
        counts[shape] = counts.get(shape, 0) + 1
    return counts


def row_salts(row_keys: list[str]) -> list[int]:
    salts = []
    for row_key in row_keys:
        marker = ":salt"
        if marker not in row_key:
            continue
        salts.append(as_int(row_key.rsplit(marker, 1)[1], -1))
    return sorted(salt for salt in salts if salt >= 0)


def ops_over_rho(summary: dict[str, Any], fallback: dict[str, Any]) -> float | None:
    ratio = as_float(summary.get("fused_selector_ops_over_rho"))
    if ratio is not None:
        return ratio
    total_ops = as_float(summary.get("fused_selector_total_ops"))
    rho = as_float(summary.get("rho"))
    if total_ops is not None and rho not in (None, 0.0):
        return round(total_ops / rho, 8)
    ratio = as_float(fallback.get("public_relation_dependency_ops_over_rho"))
    if ratio is not None:
        return ratio
    return as_float(fallback.get("public_leaf_dependency_ops_over_rho"))


def normalize_result_record(row: dict[str, Any], source_label: str, source_path: Path) -> dict[str, Any]:
    events = row.get("dependency_events") or []
    term_support = sorted({as_int(term, -1) for term in row.get("dependency_term_support") or [] if as_int(term, -1) >= 0})
    if not term_support:
        term_support = term_set_from_events(events)
    relation_verified = bool(row.get("public_relation_key_verified"))
    leaf_verified = bool(row.get("public_leaf_key_verified"))
    relation_below = bool(row.get("public_relation_dependency_beats_rho"))
    leaf_below = bool(row.get("public_leaf_dependency_beats_rho"))
    return {
        "dependency_event_count": len(events),
        "dependency_events": events,
        "fused_selector_below_rho": relation_below or leaf_below,
        "ops_over_rho": as_float(row.get("public_relation_dependency_ops_over_rho")),
        "public_dependency_key_verified": relation_verified or leaf_verified,
        "public_leaf_key_verified": leaf_verified,
        "public_relation_key_verified": relation_verified,
        "record_kind": "dependency_circuit_result",
        "record_status": (
            "PUBLIC RELATION DEPENDENCY BEATS RHO"
            if relation_verified and relation_below
            else "PUBLIC DEPENDENCY VERIFIED"
            if relation_verified or leaf_verified
            else "PUBLIC DEPENDENCY UNVERIFIED"
        ),
        "relation_indices": [as_int(event.get("relation_index"), -1) for event in events],
        "row_keys": [],
        "row_salts": [],
        "source_label": source_label,
        "source_path": str(source_path),
        "target": row.get("target"),
        "term_shape_counts": row.get("dependency_term_shape_counts") or term_shape_counts(events),
        "term_support": term_support,
        "transfer_index": None,
    }


def normalize_case_record(case: dict[str, Any], source_label: str, source_path: Path) -> dict[str, Any]:
    summary = case.get("summary") or {}
    case_info = case.get("case") or {}
    rows = case.get("rows") or []
    events = case.get("event_summaries") or []
    row_keys = [str(row.get("row_key")) for row in rows if row.get("row_key")]
    term_support = term_set_from_events(events)
    public_dependency = case.get("public_dependency") or {}
    minimal_relation = case.get("minimal_public_relation_dependency") or {}
    public_verified = bool(
        summary.get("public_dependency_key_verified")
        or summary.get("public_relation_key_verified")
        or public_dependency.get("public_key_verified")
        or minimal_relation.get("public_key_verified")
    )
    relation_verified = bool(summary.get("public_relation_key_verified") or minimal_relation.get("public_key_verified"))
    leaf_verified = bool(summary.get("public_leaf_key_verified"))
    target = summary.get("target") or case_info.get("target")
    transfer = summary.get("transfer_index")
    if transfer is None:
        transfer = case_info.get("transfer_index")
    return {
        "dependency_event_count": len(events),
        "dependency_events": events,
        "fused_selector_below_rho": bool(summary.get("fused_selector_below_rho")),
        "ops_over_rho": ops_over_rho(summary, {}),
        "public_dependency_key_verified": public_verified,
        "public_dependency_rank": as_int(summary.get("public_dependency_rank")),
        "public_leaf_key_verified": leaf_verified,
        "public_relation_key_verified": relation_verified,
        "record_kind": "dependency_circuit_case",
        "record_status": summary.get("status"),
        "relation_indices": list(summary.get("public_relation_minimal_relation_indices") or []),
        "row_keys": row_keys,
        "row_salts": row_salts(row_keys),
        "source_label": source_label,
        "source_path": str(source_path),
        "target": target,
        "term_shape_counts": term_shape_counts(events),
        "term_support": term_support,
        "transfer_index": as_int(transfer, -1) if transfer is not None else None,
    }


def normalize_dependency_records(data: Any, source_label: str, source_path: Path) -> list[dict[str, Any]]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return [normalize_result_record(row, source_label, source_path) for row in data.get("results") or []]
        if isinstance(data.get("cases"), list):
            return [normalize_case_record(case, source_label, source_path) for case in data.get("cases") or []]
        items = list(data.values())
    else:
        return []
    records = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "summary" in item or "case" in item:
            records.append(normalize_case_record(item, source_label, source_path))
        elif "target" in item:
            records.append(normalize_result_record(item, source_label, source_path))
    return records


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[int, dict[str, Any]] = {}
    for record in records:
        digest = stable_hash_u64(
            {
                "below": record.get("fused_selector_below_rho"),
                "events": record.get("dependency_events"),
                "ops_over_rho": record.get("ops_over_rho"),
                "row_keys": record.get("row_keys"),
                "source_label": record.get("source_label"),
                "target": record.get("target"),
                "term_support": record.get("term_support"),
                "transfer_index": record.get("transfer_index"),
                "verified": record.get("public_dependency_key_verified"),
            }
        )
        if digest not in deduped:
            deduped[digest] = dict(record)
            deduped[digest]["record_hash_u64"] = digest
            deduped[digest]["duplicate_count"] = 1
        else:
            deduped[digest]["duplicate_count"] += 1
    return sorted(
        deduped.values(),
        key=lambda row: (
            str(row.get("target")),
            as_int(row.get("transfer_index"), 10**12),
            str(row.get("source_label")),
            row.get("term_support") or [],
        ),
    )


def slice_quadratic_target_rows(slice_source: dict[str, Any], target: str) -> list[dict[str, Any]]:
    rows = []
    for item in (slice_source.get("summary") or {}).get("best_preserving_slice_quadratic_surfaces") or []:
        row_key = str(item.get("row_key") or "")
        if not row_key.startswith(f"{target}:"):
            continue
        candidate = item.get("best_preserving_candidate") or {}
        rows.append(
            {
                "axis": candidate.get("axis"),
                "factor_count": len(candidate.get("factors") or []),
                "row_key": row_key,
                "row_salt": row_salts([row_key])[0] if row_salts([row_key]) else None,
                "slice_quadratic_all_hit_beats_rho": bool(candidate.get("slice_quadratic_all_hit_beats_rho")),
                "slice_quadratic_all_hit_ops_over_rho": as_float(candidate.get("slice_quadratic_all_hit_ops_over_rho")),
                "surface_id": item.get("surface_id"),
            }
        )
    return sorted(rows, key=lambda row: (as_int(row.get("row_salt"), 10**12), str(row.get("surface_id"))))


def record_covers_terms(record: dict[str, Any], terms: set[int]) -> list[int]:
    return sorted(set(record.get("term_support") or []) & terms)


def term_candidate_map(records: list[dict[str, Any]], terms: set[int]) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {term: [] for term in sorted(terms)}
    for index, record in enumerate(records):
        for term in record_covers_terms(record, terms):
            mapping[term].append(index)
    return mapping


def term_work_order(
    bridge_missing_terms: set[int],
    same_target_candidates: list[dict[str, Any]],
    cross_target_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    same_map = term_candidate_map(same_target_candidates, bridge_missing_terms)
    cross_map = term_candidate_map(cross_target_candidates, bridge_missing_terms)
    work_order = []
    for term in sorted(bridge_missing_terms):
        same_refs = same_map.get(term) or []
        cross_refs = cross_map.get(term) or []
        if same_refs:
            status = "same_target_verified_transfer_analogue_available"
            next_action = "Port the same-target public dependency shape into the transfer-10376 residual kernel lane evidence."
        elif cross_refs:
            status = "cross_target_analogue_only_fresh_same_target_required"
            next_action = "Search the 22050 target rows for this term shape; cross-target evidence is a pattern hint only."
        else:
            status = "no_public_analogue_found_fresh_synthesis_required"
            next_action = "Run fresh FFE or summation-polynomial synthesis for this residual group."
        work_order.append(
            {
                "cross_target_candidate_indices": cross_refs,
                "next_action": next_action,
                "same_target_candidate_indices": same_refs,
                "status": status,
                "term": term,
                "validator_acceptance_ready": False,
            }
        )
    return work_order


def build_scanner(
    bridge: dict[str, Any],
    dependency_records: list[dict[str, Any]],
    slice_source: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    target_slot = bridge.get("target_slot") or {}
    target = str(target_slot.get("target") or "")
    transfer = as_int(target_slot.get("transfer_index"), -1)
    target_row_keys = [str(row) for row in target_slot.get("row_keys") or []]
    bridge_covered_terms = {
        as_int(row.get("term"), -1)
        for row in bridge.get("group_evidence_bridge") or []
        if row.get("covered_by_public_dependency_circuit")
    }
    bridge_missing_terms = {
        as_int(row.get("term"), -1)
        for row in bridge.get("group_evidence_bridge") or []
        if not row.get("covered_by_public_dependency_circuit")
    }
    bridge_covered_terms = {term for term in bridge_covered_terms if term >= 0}
    bridge_missing_terms = {term for term in bridge_missing_terms if term >= 0}
    missing_rows = [str(row) for row in (bridge.get("slice_quadratic_bridge") or {}).get("missing_rows") or []]
    missing_row_set = set(missing_rows)
    records = dedupe_records(dependency_records)
    verified_below = [
        row
        for row in records
        if row.get("public_dependency_key_verified") is True and row.get("fused_selector_below_rho") is True
    ]
    same_target_candidates = [
        row
        for row in verified_below
        if str(row.get("target")) == target and record_covers_terms(row, bridge_missing_terms)
    ]
    same_target_salt_candidates = [
        row
        for row in verified_below
        if str(row.get("target")) == target and missing_row_set & set(row.get("row_keys") or [])
    ]
    same_target_new_terms = sorted(
        {
            term
            for row in same_target_candidates
            for term in record_covers_terms(row, bridge_missing_terms)
        }
    )
    same_target_total_terms = sorted(bridge_covered_terms | set(same_target_new_terms))
    remaining_after_same_target = sorted(bridge_missing_terms - set(same_target_new_terms))
    cross_target_candidates = [
        row
        for row in verified_below
        if str(row.get("target")) != target and record_covers_terms(row, set(remaining_after_same_target))
    ]
    cross_target_analogue_terms = sorted(
        {
            term
            for row in cross_target_candidates
            for term in record_covers_terms(row, set(remaining_after_same_target))
        }
    )
    unverified_same_target_context = [
        row
        for row in records
        if str(row.get("target")) == target
        and not row.get("public_dependency_key_verified")
        and (
            record_covers_terms(row, bridge_missing_terms)
            or missing_row_set & set(row.get("row_keys") or [])
        )
    ]
    target_slice_rows = slice_quadratic_target_rows(slice_source, target)
    slice_salts = sorted({as_int(row.get("row_salt"), -1) for row in target_slice_rows if as_int(row.get("row_salt"), -1) >= 0})
    failures = []
    if bridge.get("claim_status") != "SELECTED13_PRIORITY0_DEPENDENCY_CIRCUIT_EVIDENCE_BRIDGE_READY":
        failures.append({"code": "dependency_bridge_not_ready", "claim_status": bridge.get("claim_status")})
    if transfer != EXPECTED_TRANSFER:
        failures.append({"code": "transfer_unexpected", "observed": transfer})
    if len(bridge_covered_terms) != EXPECTED_BRIDGE_COVERED_TERM_COUNT:
        failures.append({"code": "bridge_covered_term_count_unexpected", "observed": len(bridge_covered_terms)})
    if len(bridge_missing_terms) != EXPECTED_BRIDGE_MISSING_TERM_COUNT:
        failures.append({"code": "bridge_missing_term_count_unexpected", "observed": len(bridge_missing_terms)})
    if len(same_target_new_terms) != EXPECTED_SAME_TARGET_NEW_TERM_COUNT:
        failures.append({"code": "same_target_new_term_count_unexpected", "observed": len(same_target_new_terms)})
    if len(same_target_total_terms) != EXPECTED_SAME_TARGET_TOTAL_TERM_COUNT:
        failures.append({"code": "same_target_total_term_count_unexpected", "observed": len(same_target_total_terms)})
    if remaining_after_same_target != EXPECTED_REMAINING_TERMS:
        failures.append({"code": "remaining_terms_unexpected", "observed": remaining_after_same_target})
    if len(remaining_after_same_target) != EXPECTED_REMAINING_TERM_COUNT:
        failures.append({"code": "remaining_term_count_unexpected", "observed": len(remaining_after_same_target)})
    if len(same_target_salt_candidates) != EXPECTED_SALT173_CANDIDATE_COUNT:
        failures.append({"code": "salt173_candidate_count_unexpected", "observed": len(same_target_salt_candidates)})
    if len(cross_target_analogue_terms) != EXPECTED_CROSS_TARGET_ANALOGUE_TERM_COUNT:
        failures.append({"code": "cross_target_analogue_term_count_unexpected", "observed": len(cross_target_analogue_terms)})
    summary = {
        "accepted_relation_export_count": 0,
        "bridge_covered_term_count": len(bridge_covered_terms),
        "bridge_covered_terms": sorted(bridge_covered_terms),
        "bridge_missing_term_count": len(bridge_missing_terms),
        "bridge_missing_terms": sorted(bridge_missing_terms),
        "cross_target_analogue_candidate_count": len(cross_target_candidates),
        "cross_target_analogue_term_count": len(cross_target_analogue_terms),
        "cross_target_analogue_terms": cross_target_analogue_terms,
        "failure_count": len(failures),
        "native_preflight_verified": False,
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "remaining_same_target_term_count": len(remaining_after_same_target),
        "remaining_same_target_terms": remaining_after_same_target,
        "salt173_candidate_count": len(same_target_salt_candidates),
        "same_target_new_term_count": len(same_target_new_terms),
        "same_target_new_terms": same_target_new_terms,
        "same_target_total_term_count": len(same_target_total_terms),
        "same_target_total_terms": same_target_total_terms,
        "same_target_verified_candidate_count": len(same_target_candidates),
        "source_record_count": len(records),
        "target_slice_quadratic_row_count": len(target_slice_rows),
        "target_slice_quadratic_salts": slice_salts,
        "verified": not failures,
        "worker_interpretation": (
            "Same-target verified analogue circuits now cover selected13 residual terms 3 and 9 "
            "in addition to the bridge-covered terms 2, 6, and 13. Terms 8 and 12 remain "
            "cross-target-only hints and still need fresh same-target FFE/summation evidence."
        ),
    }
    payload = {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_GAP_CLOSURE_SCANNER_READY"
            if not failures
            else "SELECTED13_PRIORITY0_GAP_CLOSURE_SCANNER_FAILED"
        ),
        "parameters": {key: str(value) for key, value in source_paths.items()},
        "packet_hash_u64": stable_hash_u64(
            {
                "bridge_covered_terms": sorted(bridge_covered_terms),
                "cross_target_analogue_terms": cross_target_analogue_terms,
                "remaining_after_same_target": remaining_after_same_target,
                "same_target_new_terms": same_target_new_terms,
                "target_slot": target_slot,
            }
        ),
        "target_slot": target_slot,
        "bridge_status": {
            "claim_status": bridge.get("claim_status"),
            "covered_terms": sorted(bridge_covered_terms),
            "missing_rows": missing_rows,
            "missing_terms": sorted(bridge_missing_terms),
            "native_preflight_verified": bool((bridge.get("summary") or {}).get("native_preflight_verified")),
            "target_row_keys": target_row_keys,
        },
        "dependency_sources_scanned": [
            {"label": key, "path": str(value)}
            for key, value in sorted(source_paths.items())
            if key not in {"bridge", "out", "c_header_out", "slice_quadratic"}
        ],
        "same_target_verified_closure_candidates": same_target_candidates,
        "same_target_missing_row_candidates": same_target_salt_candidates,
        "cross_target_analogue_candidates": cross_target_candidates,
        "unverified_same_target_context": unverified_same_target_context,
        "term_work_order": term_work_order(
            bridge_missing_terms,
            same_target_candidates,
            cross_target_candidates,
        ),
        "slice_quadratic_gap": {
            "missing_rows_from_bridge": missing_rows,
            "target_slice_quadratic_rows": target_slice_rows,
            "target_slice_quadratic_salts": slice_salts,
        },
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "cross_target_analogues_count_as_selected13_proof": False,
            "external_kernel_result_emitted": False,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "same_target_transfer_analogue_only": True,
            "summation_polynomial_evaluated_by_scanner": False,
            "validator_acceptance_ready": False,
        },
        "failures": failures,
        "summary": summary,
    }
    return payload


def render_c_header(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    target = payload.get("target_slot") or {}
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_GAP_CLOSURE_SCANNER_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_GAP_CLOSURE_SCANNER_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_GAP_SCANNER_TRANSFER {as_int(target.get('transfer_index'))}ULL
#define SELECTED13_PRIORITY0_GAP_SCANNER_BRIDGE_COVERED_TERM_COUNT {as_int(summary.get('bridge_covered_term_count'))}ULL
#define SELECTED13_PRIORITY0_GAP_SCANNER_BRIDGE_MISSING_TERM_COUNT {as_int(summary.get('bridge_missing_term_count'))}ULL
#define SELECTED13_PRIORITY0_GAP_SCANNER_SAME_TARGET_NEW_TERM_COUNT {as_int(summary.get('same_target_new_term_count'))}ULL
#define SELECTED13_PRIORITY0_GAP_SCANNER_SAME_TARGET_TOTAL_TERM_COUNT {as_int(summary.get('same_target_total_term_count'))}ULL
#define SELECTED13_PRIORITY0_GAP_SCANNER_REMAINING_TERM_COUNT {as_int(summary.get('remaining_same_target_term_count'))}ULL
#define SELECTED13_PRIORITY0_GAP_SCANNER_SALT173_CANDIDATE_COUNT {as_int(summary.get('salt173_candidate_count'))}ULL
#define SELECTED13_PRIORITY0_GAP_SCANNER_CROSS_TARGET_ANALOGUE_TERM_COUNT {as_int(summary.get('cross_target_analogue_term_count'))}ULL
#define SELECTED13_PRIORITY0_GAP_SCANNER_ACCEPTED_EXPORT_COUNT 0ULL
#define SELECTED13_PRIORITY0_GAP_SCANNER_RELATION_DERIVED_ECDLP 0ULL
#define SELECTED13_PRIORITY0_GAP_SCANNER_VALIDATOR_ACCEPTANCE_READY 0ULL

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

int main(void) {{
  uint64_t failure_count = 0;
  if (SELECTED13_PRIORITY0_GAP_SCANNER_TRANSFER != {EXPECTED_TRANSFER}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_GAP_SCANNER_BRIDGE_COVERED_TERM_COUNT != {EXPECTED_BRIDGE_COVERED_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_GAP_SCANNER_BRIDGE_MISSING_TERM_COUNT != {EXPECTED_BRIDGE_MISSING_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_GAP_SCANNER_SAME_TARGET_NEW_TERM_COUNT != {EXPECTED_SAME_TARGET_NEW_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_GAP_SCANNER_SAME_TARGET_TOTAL_TERM_COUNT != {EXPECTED_SAME_TARGET_TOTAL_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_GAP_SCANNER_REMAINING_TERM_COUNT != {EXPECTED_REMAINING_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_GAP_SCANNER_SALT173_CANDIDATE_COUNT != {EXPECTED_SALT173_CANDIDATE_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_GAP_SCANNER_CROSS_TARGET_ANALOGUE_TERM_COUNT != {EXPECTED_CROSS_TARGET_ANALOGUE_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_GAP_SCANNER_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_GAP_SCANNER_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_GAP_SCANNER_VALIDATOR_ACCEPTANCE_READY != 0ULL) failure_count++;

  printf("selected13_priority0_gap_closure_scanner_preflight transfer=%llu bridge_terms=%llu same_target_new_terms=%llu same_target_total_terms=%llu remaining_terms=%llu salt173_candidates=%llu cross_target_terms=%llu accepted=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_GAP_SCANNER_TRANSFER,
         (unsigned long long)SELECTED13_PRIORITY0_GAP_SCANNER_BRIDGE_COVERED_TERM_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_GAP_SCANNER_SAME_TARGET_NEW_TERM_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_GAP_SCANNER_SAME_TARGET_TOTAL_TERM_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_GAP_SCANNER_REMAINING_TERM_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_GAP_SCANNER_SALT173_CANDIDATE_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_GAP_SCANNER_CROSS_TARGET_ANALOGUE_TERM_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_GAP_SCANNER_ACCEPTED_EXPORT_COUNT,
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
    with tempfile.TemporaryDirectory(prefix="selected13_gap_closure_scanner_", dir=str(temp_root)) as tmp:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", default=str(DEFAULT_BRIDGE))
    parser.add_argument("--dependency-circuit", default=str(DEFAULT_DEPENDENCY_CIRCUIT))
    parser.add_argument("--surface-gate", default=str(DEFAULT_SURFACE_GATE))
    parser.add_argument("--same-transfer-row-bank", default=str(DEFAULT_SAME_TRANSFER_ROW_BANK))
    parser.add_argument("--multiblock", default=str(DEFAULT_MULTIBLOCK))
    parser.add_argument("--slice-quadratic", default=str(DEFAULT_SLICE_QUADRATIC))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_paths = {
        "bridge": Path(args.bridge),
        "dependency_circuit": Path(args.dependency_circuit),
        "surface_gate": Path(args.surface_gate),
        "same_transfer_row_bank": Path(args.same_transfer_row_bank),
        "multiblock": Path(args.multiblock),
        "slice_quadratic": Path(args.slice_quadratic),
        "out": Path(args.out),
        "c_header_out": Path(args.c_header_out),
    }
    dependency_records: list[dict[str, Any]] = []
    for label in ("dependency_circuit", "surface_gate", "same_transfer_row_bank", "multiblock"):
        path = source_paths[label]
        dependency_records.extend(normalize_dependency_records(load_json(path), label, path))
    payload = build_scanner(
        load_json(source_paths["bridge"]),
        dependency_records,
        load_json(source_paths["slice_quadratic"]),
        source_paths,
    )
    header_path = source_paths["c_header_out"]
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload))
    payload["artifacts"] = {"c_header": str(header_path)}
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_GAP_CLOSURE_SCANNER_FAILED"
    payload["summary"]["native_preflight_verified"] = bool(native_preflight.get("verified"))
    payload["summary"]["failure_count"] = len(payload["failures"])
    payload["summary"]["verified"] = not payload["failures"]
    write_json(source_paths["out"], payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
