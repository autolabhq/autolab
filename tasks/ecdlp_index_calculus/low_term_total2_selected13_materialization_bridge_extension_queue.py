#!/usr/bin/env python3
"""Build a prioritized bridge-extension queue for selected13 materialization rows.

The materialization bridge packet exposes 45 support-missing rows.  The direct
evidence audit has already promoted the three 10595 rows, leaving 42 rows that
need fresh FFE/summation-polynomial direct-rank emission.  This script binds the
remaining rows to exact source hints, per-row evidence classifications, support
and salt deltas, and a native-checkable priority queue for the next worker.

It does not evaluate summation polynomials, export new direct/rank rows, derive
new ECDLP secrets, or claim a Pollard-rho speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_selected13_materialization_bridge_extension_queue.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_BRIDGE_PACKET = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_bridge_packet_111_full147_probe.json"
DEFAULT_SOURCE_HINT = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_source_hint_adapter_111_full147_probe.json"
DEFAULT_DIRECT_EVIDENCE = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_direct_evidence_audit_111_full147_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_bridge_extension_queue_112_full147_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_bridge_extension_queue_112_full147_probe.h"

CLASS_CODES = {
    "DIRECT_UNION_DERIVED_SECRET_PRESENT": 1,
    "DIRECT_CERTIFICATE_DERIVED_SECRET_PRESENT": 2,
    "DIRECT_VERIFIER_SECRET_MISSING": 3,
    "FRESH_DIRECT_RELATION_REQUIRED_RANK_DEFICIENT": 4,
    "FRESH_DIRECT_RELATION_REQUIRED_NO_RELATIONS": 5,
    "DIRECT_VERIFIER_EVIDENCE_MISSING": 6,
}
WORKER_CLASS_CODES = {
    "control_promoted_direct_union": 1,
    "hinted_delta_rank_bridge": 2,
    "hinted_delta_source_solve": 3,
    "rank_deficient_unhinted_bridge": 4,
    "no_relation_unhinted_source_solve": 5,
}
ROW_KEY_SALT_RE = re.compile(r":salt(\d+)$")


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


def popcount(value: int) -> int:
    return int(value).bit_count()


def salts_from_row_keys(row_keys: Any) -> list[int]:
    salts = []
    for row_key in row_keys or []:
        match = ROW_KEY_SALT_RE.search(str(row_key))
        if match:
            salts.append(int(match.group(1)))
    return sorted(salts)


def salt_delta(target_salts: list[int], source_salts: list[int]) -> dict[str, Any]:
    target_set = set(target_salts)
    source_set = set(source_salts)
    shared = sorted(target_set & source_set)
    target_only = sorted(target_set - source_set)
    source_only = sorted(source_set - target_set)
    deltas = [abs(a - b) for a in target_only for b in source_only]
    return {
        "salt_delta_min": min(deltas) if deltas else 0,
        "salt_overlap_count": len(shared),
        "source_only_salts": source_only,
        "target_only_salts": target_only,
    }


def rows_by_request(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("row_request_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("row_request_id") is not None
    }


def source_hints_for_row(source_hint: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    start = as_int(row.get("hint_start"), -1)
    count = as_int(row.get("hint_count"))
    if start < 0 or count <= 0:
        return []
    hints = source_hint.get("source_hints") or []
    return [hint for hint in hints[start : start + count] if isinstance(hint, dict)]


def classify_worker(row: dict[str, Any], audit: dict[str, Any], hints: list[dict[str, Any]]) -> str:
    if bool(audit.get("relation_derived_ecdlp")):
        return "control_promoted_direct_union"
    classification = str(audit.get("classification") or "")
    if hints and classification == "FRESH_DIRECT_RELATION_REQUIRED_NO_RELATIONS":
        return "hinted_delta_source_solve"
    if hints:
        return "hinted_delta_rank_bridge"
    if classification == "FRESH_DIRECT_RELATION_REQUIRED_NO_RELATIONS":
        return "no_relation_unhinted_source_solve"
    return "rank_deficient_unhinted_bridge"


def priority_score(row: dict[str, Any], audit: dict[str, Any], hints: list[dict[str, Any]]) -> int:
    if bool(audit.get("relation_derived_ecdlp")):
        return -1
    score = 0
    if hints:
        score += 1_000_000
    if bool(row.get("is_best_manifest_row")):
        score += 350_000
    classification = str(audit.get("classification") or "")
    if classification == "FRESH_DIRECT_RELATION_REQUIRED_NO_RELATIONS":
        score += 175_000
    score += 50_000 * min(8, as_int(audit.get("product_rank")))
    score += 20_000 * min(8, as_int(audit.get("product_relation_count")))
    if hints:
        score += 1000 * max(as_int(hint.get("support_overlap_count")) for hint in hints)
        score += max(as_int(hint.get("support_jaccard_scaled_1e6")) for hint in hints) // 1000
        score += 500 * max(as_int(hint.get("salt_overlap_count")) for hint in hints)
        if any(hint.get("public_product_gate_selected") for hint in hints):
            score += 7500
        if any(hint.get("shared_product_public_key_verified") for hint in hints):
            score += 5000
    direct_ops = as_float(row.get("direct_ops_over_rho"))
    if direct_ops is not None and direct_ops < 1.0:
        score += max(0, round((1.0 - direct_ops) * 10_000))
    return score


def build_hint_case(target_row: dict[str, Any], hint: dict[str, Any], hint_local_index: int) -> dict[str, Any]:
    target_mask = as_int(target_row.get("selected_support_mask"))
    source_mask = as_int(hint.get("selected_support_mask"))
    target_salts = salts_from_row_keys(target_row.get("row_keys"))
    source_salts = salts_from_row_keys(hint.get("row_keys"))
    delta = salt_delta(target_salts, source_salts)
    return {
        "direct_ops_over_rho": as_float(hint.get("direct_ops_over_rho")),
        "direct_ops_over_rho_scaled": scaled_ops(hint.get("direct_ops_over_rho")),
        "direct_public_key_verified": bool(hint.get("direct_public_key_verified")),
        "extra_source_support_mask": source_mask & ~target_mask,
        "hint_hash_u64": as_int(hint.get("hint_hash_u64")),
        "hint_local_index": hint_local_index,
        "missing_target_support_mask": target_mask & ~source_mask,
        "public_product_gate_selected": bool(hint.get("public_product_gate_selected")),
        "salt_delta_min": as_int(delta.get("salt_delta_min")),
        "salt_overlap_count": as_int(hint.get("salt_overlap_count")),
        "shared_product_public_key_verified": bool(hint.get("shared_product_public_key_verified")),
        "source_row_hash_u64": as_int(hint.get("source_row_hash_u64")),
        "source_row_keys": hint.get("row_keys") or [],
        "source_selected_support_mask": source_mask,
        "source_selector": hint.get("selector"),
        "source_selector_u64": stable_hash_u64(hint.get("selector")),
        "source_support_delta_popcount": popcount(target_mask ^ source_mask),
        "source_support_jaccard_scaled_1e6": as_int(hint.get("support_jaccard_scaled_1e6")),
        "source_support_overlap_count": as_int(hint.get("support_overlap_count")),
        "source_top_k": as_int(hint.get("top_k")),
        "target_only_salts": delta.get("target_only_salts"),
        "source_only_salts": delta.get("source_only_salts"),
    }


def build_cases(
    bridge: dict[str, Any],
    source_hint: dict[str, Any],
    direct_evidence: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    hint_rows = rows_by_request(source_hint.get("rows") or [])
    audit_rows = rows_by_request(direct_evidence.get("row_audits") or [])
    cases = []
    controls = []
    for bridge_row in bridge.get("rows") or []:
        if not isinstance(bridge_row, dict):
            continue
        request_id = str(bridge_row.get("row_request_id") or "")
        hint_row = hint_rows.get(request_id)
        audit = audit_rows.get(request_id)
        if hint_row is None:
            failures.append({"code": "source_hint_row_missing", "row_request_id": request_id})
            hint_row = {}
        if audit is None:
            failures.append({"code": "direct_evidence_row_missing", "row_request_id": request_id})
            audit = {}
        hints = source_hints_for_row(source_hint, hint_row)
        worker_class = classify_worker(bridge_row, audit, hints)
        hint_cases = [build_hint_case(bridge_row, hint, index) for index, hint in enumerate(hints)]
        target_salts = salts_from_row_keys(bridge_row.get("row_keys"))
        case = {
            "accepted_relation_export": False,
            "bridge_worker_required": not bool(audit.get("relation_derived_ecdlp")),
            "candidate_case_arg": (
                f"{bridge_row.get('target')}|{as_int(bridge_row.get('transfer_index'))}|"
                f"{bridge_row.get('selector')}|{as_int(bridge_row.get('top_k'))}"
            ),
            "classification": audit.get("classification"),
            "classification_code": as_int(audit.get("classification_code")) or CLASS_CODES.get(str(audit.get("classification") or ""), 0),
            "direct_ops_over_rho": as_float(bridge_row.get("direct_ops_over_rho")),
            "direct_ops_over_rho_scaled": scaled_ops(bridge_row.get("direct_ops_over_rho")),
            "exact_product_gate_row_count": as_int(audit.get("exact_product_gate_row_count")),
            "exact_source_policy_row_count": as_int(audit.get("exact_source_policy_row_count")),
            "global_row_index": as_int(bridge_row.get("global_row_index"), -1),
            "hint_count": len(hint_cases),
            "hints": hint_cases,
            "is_best_manifest_row": bool(bridge_row.get("is_best_manifest_row")),
            "matched_family_count": as_int(bridge_row.get("matched_family_count")),
            "matched_family_masks": [as_int(item) for item in bridge_row.get("matched_family_masks") or []],
            "packet_index": as_int(bridge_row.get("packet_index"), -1),
            "priority_score": priority_score(bridge_row, audit, hints),
            "product_rank": as_int(audit.get("product_rank")),
            "product_relation_count": as_int(audit.get("product_relation_count")),
            "relation_derived_ecdlp": False,
            "row_keys": bridge_row.get("row_keys") or [],
            "row_request_id": request_id,
            "row_request_id_u64": as_int(bridge_row.get("row_request_id_u64")),
            "selected_support_mask": as_int(bridge_row.get("selected_support_mask")),
            "selected_term_support": [as_int(item) for item in bridge_row.get("selected_term_support") or []],
            "selector": bridge_row.get("selector"),
            "source_rank": as_int(audit.get("source_rank")),
            "source_relation_count": as_int(audit.get("source_relation_count")),
            "target": bridge_row.get("target"),
            "target_salts": target_salts,
            "top_k": as_int(bridge_row.get("top_k")),
            "transfer_index": as_int(bridge_row.get("transfer_index"), -1),
            "worker_acceptance_gate": "fresh_direct_rank_export_and_relation_derived_ecdlp_only",
            "worker_class": worker_class,
            "worker_class_code": WORKER_CLASS_CODES.get(worker_class, 0),
            "worker_obligation": (
                "control_row_already_promoted_by_direct_evidence"
                if worker_class == "control_promoted_direct_union"
                else "fresh_ffe_summation_polynomial_direct_rank_bridge_extension"
            ),
        }
        if worker_class == "control_promoted_direct_union":
            case["accepted_relation_export"] = True
            case["derived_secret"] = audit.get("derived_secret")
            case["relation_derived_ecdlp"] = bool(audit.get("relation_derived_ecdlp"))
            controls.append(case)
        else:
            cases.append(case)
    cases.sort(
        key=lambda row: (
            -as_int(row.get("priority_score")),
            0 if row.get("is_best_manifest_row") else 1,
            as_int(row.get("transfer_index")),
            as_int(row.get("global_row_index")),
        )
    )
    for priority_rank, case in enumerate(cases):
        case["priority_rank"] = priority_rank
    for priority_rank, case in enumerate(controls):
        case["priority_rank"] = priority_rank
    return cases, controls, failures


def summarize_transfer_cases(cases: list[dict[str, Any]], controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for case in [*cases, *controls]:
        grouped[as_int(case.get("transfer_index"), -1)].append(case)
    out = []
    for transfer in sorted(grouped):
        rows = grouped[transfer]
        actionable = [row for row in rows if row.get("bridge_worker_required")]
        out.append(
            {
                "actionable_row_count": len(actionable),
                "best_row_request_ids": [row.get("row_request_id") for row in actionable if row.get("is_best_manifest_row")],
                "classification_counts": dict(Counter(str(row.get("classification")) for row in rows)),
                "control_row_count": sum(1 for row in rows if not row.get("bridge_worker_required")),
                "hinted_actionable_row_count": sum(1 for row in actionable if as_int(row.get("hint_count")) > 0),
                "max_priority_score": max((as_int(row.get("priority_score")) for row in actionable), default=0),
                "top_worker_class": max(
                    (str(row.get("worker_class")) for row in actionable),
                    key=lambda item: Counter(str(row.get("worker_class")) for row in actionable).get(item, 0),
                    default="control_only",
                ),
                "transfer_index": transfer,
            }
        )
    return out


def summarize(cases: list[dict[str, Any]], controls: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = Counter(str(row.get("classification")) for row in cases)
    worker_counts = Counter(str(row.get("worker_class")) for row in cases)
    hinted_rows = [row for row in cases if as_int(row.get("hint_count")) > 0]
    no_relation_rows = [row for row in cases if row.get("classification") == "FRESH_DIRECT_RELATION_REQUIRED_NO_RELATIONS"]
    best_rows = [row for row in cases if row.get("is_best_manifest_row")]
    return {
        "accepted_relation_export_count": 0,
        "actionable_row_count": len(cases),
        "actionable_transfer_count": len({as_int(row.get("transfer_index"), -1) for row in cases}),
        "best_actionable_row_count": len(best_rows),
        "bridge_worker_required_row_count": len(cases),
        "classification_counts": dict(sorted(class_counts.items())),
        "control_relation_derived_row_count": len(controls),
        "control_relation_derived_transfers": sorted({as_int(row.get("transfer_index"), -1) for row in controls}),
        "failure_count": len(failures),
        "hinted_actionable_row_count": len(hinted_rows),
        "hinted_actionable_transfer_count": len({as_int(row.get("transfer_index"), -1) for row in hinted_rows}),
        "hinted_actionable_transfers": sorted({as_int(row.get("transfer_index"), -1) for row in hinted_rows}),
        "no_relation_actionable_row_count": len(no_relation_rows),
        "no_relation_actionable_transfers": sorted({as_int(row.get("transfer_index"), -1) for row in no_relation_rows}),
        "pollard_rho_speedup_claimed": False,
        "priority_frontier": [
            {
                "classification": row.get("classification"),
                "hint_count": as_int(row.get("hint_count")),
                "is_best_manifest_row": bool(row.get("is_best_manifest_row")),
                "priority_rank": as_int(row.get("priority_rank")),
                "priority_score": as_int(row.get("priority_score")),
                "row_request_id": row.get("row_request_id"),
                "transfer_index": as_int(row.get("transfer_index")),
                "worker_class": row.get("worker_class"),
            }
            for row in cases[:12]
        ],
        "relation_derived_ecdlp": False,
        "verified": not failures,
        "worker_class_counts": dict(sorted(worker_counts.items())),
        "worker_interpretation": (
            "Actionable rows are prioritized for fresh FFE/summation-polynomial "
            "direct-rank bridge emission.  Source hints are direct-verified "
            "nearby rows, not accepted target-row relations."
        ),
    }


def validate_sources(bridge: dict[str, Any], source_hint: dict[str, Any], direct_evidence: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if bridge.get("claim_status") != "SELECTED13_MATERIALIZATION_BRIDGE_PACKET_READY":
        failures.append({"code": "bridge_packet_not_ready", "claim_status": bridge.get("claim_status")})
    if source_hint.get("claim_status") not in {
        "SELECTED13_MATERIALIZATION_SOURCE_HINTS_WITH_EXACT_DIRECT_PROMOTION",
        "SELECTED13_MATERIALIZATION_SOURCE_HINTS_READY",
    }:
        failures.append({"code": "source_hint_not_ready", "claim_status": source_hint.get("claim_status")})
    if direct_evidence.get("claim_status") != "SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_HAS_DERIVED_ROWS":
        failures.append({"code": "direct_evidence_status_unexpected", "claim_status": direct_evidence.get("claim_status")})
    for name, payload in [("bridge", bridge), ("source_hint", source_hint), ("direct_evidence", direct_evidence)]:
        if payload.get("failures"):
            failures.append({"code": f"{name}_has_failures", "failures": payload.get("failures")})
        if (payload.get("summary") or {}).get("verified") is not True:
            failures.append({"code": f"{name}_not_verified"})
    return failures


def render_c_header(cases: list[dict[str, Any]], controls: list[dict[str, Any]]) -> str:
    rows = []
    hints = []
    hint_offset = 0
    for case in cases:
        case_hints = case.get("hints") or []
        rows.append(
            "  {"
            f"{as_int(case.get('priority_rank'))}ULL, "
            f"{as_int(case.get('priority_score'))}ULL, "
            f"{as_int(case.get('packet_index'))}ULL, "
            f"{as_int(case.get('global_row_index'))}ULL, "
            f"{as_int(case.get('transfer_index'))}ULL, "
            f"{as_int(case.get('row_request_id_u64'))}ULL, "
            f"{as_int(case.get('selected_support_mask'))}ULL, "
            f"{as_int(case.get('classification_code'))}ULL, "
            f"{as_int(case.get('worker_class_code'))}ULL, "
            f"{as_int(case.get('source_rank'))}ULL, "
            f"{as_int(case.get('product_rank'))}ULL, "
            f"{1 if case.get('is_best_manifest_row') else 0}ULL, "
            f"{hint_offset}ULL, "
            f"{len(case_hints)}ULL"
            "},"
        )
        for hint in case_hints:
            hints.append(
                "  {"
                f"{as_int(case.get('priority_rank'))}ULL, "
                f"{as_int(hint.get('hint_local_index'))}ULL, "
                f"{as_int(hint.get('hint_hash_u64'))}ULL, "
                f"{as_int(hint.get('source_row_hash_u64'))}ULL, "
                f"{as_int(hint.get('source_selected_support_mask'))}ULL, "
                f"{as_int(hint.get('missing_target_support_mask'))}ULL, "
                f"{as_int(hint.get('extra_source_support_mask'))}ULL, "
                f"{as_int(hint.get('source_support_delta_popcount'))}ULL, "
                f"{as_int(hint.get('source_support_overlap_count'))}ULL, "
                f"{as_int(hint.get('source_support_jaccard_scaled_1e6'))}ULL, "
                f"{as_int(hint.get('salt_overlap_count'))}ULL, "
                f"{as_int(hint.get('salt_delta_min'))}ULL, "
                f"{1 if hint.get('direct_public_key_verified') else 0}ULL, "
                f"{as_int(hint.get('direct_ops_over_rho_scaled'))}ULL"
                "},"
            )
        hint_offset += len(case_hints)
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_QUEUE_H
#define LOW_TERM_TOTAL2_SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_QUEUE_H

#include <stdint.h>

#define SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_CASE_COUNT {len(cases)}
#define SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_HINT_COUNT {sum(len(case.get('hints') or []) for case in cases)}
#define SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_CONTROL_COUNT {len(controls)}
#define SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_RELATION_EXPORT_COUNT 0
#define SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_RELATION_DERIVED_ECDLP 0

typedef struct {{
  uint64_t priority_rank;
  uint64_t priority_score;
  uint64_t packet_index;
  uint64_t global_row_index;
  uint64_t transfer_index;
  uint64_t row_request_id_u64;
  uint64_t selected_support_mask;
  uint64_t classification_code;
  uint64_t worker_class_code;
  uint64_t source_rank;
  uint64_t product_rank;
  uint64_t is_best_manifest_row;
  uint64_t hint_offset;
  uint64_t hint_count;
}} selected13_materialization_bridge_extension_case_t;

typedef struct {{
  uint64_t priority_rank;
  uint64_t hint_local_index;
  uint64_t hint_hash_u64;
  uint64_t source_row_hash_u64;
  uint64_t source_selected_support_mask;
  uint64_t missing_target_support_mask;
  uint64_t extra_source_support_mask;
  uint64_t support_delta_popcount;
  uint64_t support_overlap_count;
  uint64_t support_jaccard_scaled_1e6;
  uint64_t salt_overlap_count;
  uint64_t salt_delta_min;
  uint64_t direct_public_key_verified;
  uint64_t direct_ops_over_rho_scaled;
}} selected13_materialization_bridge_extension_hint_t;

static const selected13_materialization_bridge_extension_case_t SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_CASES[] = {{
{chr(10).join(rows)}
}};

static const selected13_materialization_bridge_extension_hint_t SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_HINTS[] = {{
{chr(10).join(hints)}
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
  uint64_t case_count =
      sizeof(SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_CASES) / sizeof(SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_CASES[0]);
  uint64_t hint_count =
      sizeof(SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_HINTS) / sizeof(SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_HINTS[0]);
  uint64_t observed_hints = 0;
  uint64_t hinted_cases = 0;
  uint64_t best_cases = 0;
  uint64_t previous_score = UINT64_MAX;

  if (case_count != SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_CASE_COUNT) failure_count++;
  if (hint_count != SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_HINT_COUNT) failure_count++;
  if (SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_RELATION_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;
  if (case_count == 0ULL || SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_CONTROL_COUNT == 0ULL) failure_count++;

  for (size_t i = 0; i < case_count; i++) {{
    const selected13_materialization_bridge_extension_case_t *row = &SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_CASES[i];
    if (row->priority_rank != i) failure_count++;
    if (row->priority_score > previous_score) failure_count++;
    previous_score = row->priority_score;
    if (row->row_request_id_u64 == 0ULL || row->selected_support_mask == 0ULL) failure_count++;
    if (row->classification_code == 0ULL || row->worker_class_code == 0ULL) failure_count++;
    if (row->hint_offset + row->hint_count > hint_count) failure_count++;
    if (row->hint_count > 0ULL) hinted_cases++;
    if (row->is_best_manifest_row) best_cases++;
    observed_hints += row->hint_count;
  }}
  for (size_t i = 0; i < hint_count; i++) {{
    const selected13_materialization_bridge_extension_hint_t *hint = &SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_HINTS[i];
    if (hint->hint_hash_u64 == 0ULL || hint->source_row_hash_u64 == 0ULL) failure_count++;
    if (hint->source_selected_support_mask == 0ULL) failure_count++;
    if (hint->direct_public_key_verified == 0ULL) failure_count++;
  }}
  if (observed_hints != hint_count) failure_count++;

  printf("selected13_materialization_bridge_extension_queue_preflight cases=%llu hints=%llu hinted_cases=%llu best_cases=%llu controls=%llu failures=%llu\\n",
         (unsigned long long)case_count,
         (unsigned long long)hint_count,
         (unsigned long long)hinted_cases,
         (unsigned long long)best_cases,
         (unsigned long long)SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_CONTROL_COUNT,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_materialization_bridge_extension_queue_") as tmp:
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
    bridge_path = Path(args.bridge_packet)
    source_hint_path = Path(args.source_hint)
    direct_evidence_path = Path(args.direct_evidence)
    bridge = load_json(bridge_path)
    source_hint = load_json(source_hint_path)
    direct_evidence = load_json(direct_evidence_path)
    failures = validate_sources(bridge, source_hint, direct_evidence)
    cases, controls, case_failures = build_cases(bridge, source_hint, direct_evidence)
    failures.extend(case_failures)
    summary = summarize(cases, controls, failures)
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_QUEUE_READY"
            if not failures
            else "SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_QUEUE_FAILED"
        ),
        "parameters": {
            "bridge_packet": str(bridge_path),
            "direct_evidence": str(direct_evidence_path),
            "source_hint": str(source_hint_path),
        },
        "source_summary": {
            "bridge_packet": bridge.get("summary"),
            "direct_evidence": direct_evidence.get("summary"),
            "source_hint": source_hint.get("summary"),
        },
        "summary": summary,
        "transfer_queue": summarize_transfer_cases(cases, controls),
        "extension_cases": cases,
        "control_cases": controls,
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "control_cases_are_existing_10595_direct_evidence": True,
            "fresh_worker_output_required_for_promotion": True,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "source_hints_are_not_target_row_exports": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-packet", default=str(DEFAULT_BRIDGE_PACKET), help="Materialization bridge packet JSON")
    parser.add_argument("--source-hint", default=str(DEFAULT_SOURCE_HINT), help="Materialization source-hint JSON")
    parser.add_argument("--direct-evidence", default=str(DEFAULT_DIRECT_EVIDENCE), help="Materialization direct-evidence JSON")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output bridge-extension queue JSON")
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT), help="Output C preflight header")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    out_path = Path(args.out)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["extension_cases"], payload["control_cases"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["claim_status"] = "SELECTED13_MATERIALIZATION_BRIDGE_EXTENSION_QUEUE_FAILED"
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
    write_json(out_path, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(out_path), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
