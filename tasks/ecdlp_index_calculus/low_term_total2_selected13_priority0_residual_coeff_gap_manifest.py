#!/usr/bin/env python3
"""Bind priority-0 residual lanes to available source coefficient material.

The 10376 FFE direct-rank emission packet has three same-salt residual lanes.
One lane has direct-certificate coefficient forms in the mounted source
sidecar; the two hybrid lanes are verified in the scout/shared-product layer
but do not have coefficient forms in the direct-certificate artifact.  This
manifest makes that split native-checkable for the next lower-level
FFE/summation-polynomial worker.

This is a coefficient-gap manifest only.  It does not synthesize new residual
forms, evaluate summation polynomials, emit a target direct/rank row, solve
ECDLP, or claim a Pollard-rho speedup.
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


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_residual_coeff_gap_manifest.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_EMISSION_PACKET = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_direct_rank_emission_packet_10376_probe.json"
)
DEFAULT_SOURCE_HINT = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_source_hint_adapter_111_full147_probe.json"
)
DEFAULT_DIRECT_EVIDENCE = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_direct_evidence_audit_111_full147_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_residual_coeff_gap_manifest_10376_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_residual_coeff_gap_manifest_10376_probe.h"

SELECTED13_MASK = 1 << 13
COEFF_COUNT = 17
SOURCE_CLASS_CODES = {
    "direct_certificate_coefficients_available": 1,
    "shared_product_verified_coefficients_missing": 2,
    "direct_verified_coefficients_missing": 3,
    "source_unverified": 4,
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


def terms_from_mask(mask: int) -> list[int]:
    return [index for index in range(64) if as_int(mask) & (1 << index)]


def support_mask(raw: Any) -> int:
    mask = 0
    for item in raw or []:
        value = as_int(item, -1)
        if value >= 0:
            mask |= 1 << value
    return mask


def support_mask_from_coeffs(coeffs: list[Any], order: int) -> int:
    mask = 0
    for index, coeff in enumerate(coeffs[1:]):
        if order and as_int(coeff) % order != 0:
            mask |= 1 << index
    return mask


def normalize_row_keys(raw: Any) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in raw or []))


def lane_match_key(lane: dict[str, Any]) -> tuple[Any, ...]:
    return (
        lane.get("source_selector"),
        as_int(lane.get("source_top_k")),
        normalize_row_keys(lane.get("source_row_keys")),
        as_int(lane.get("source_selected_support_mask")),
    )


def adapter_hint_match_key(hint: dict[str, Any]) -> tuple[Any, ...]:
    selected_support_mask = as_int(hint.get("selected_support_mask"))
    if selected_support_mask == 0 and hint.get("selected_term_support") is not None:
        selected_support_mask = support_mask(hint.get("selected_term_support"))
    return (
        hint.get("selector"),
        as_int(hint.get("top_k")),
        normalize_row_keys(hint.get("row_keys")),
        selected_support_mask,
    )


def cert_match_key(cert: dict[str, Any]) -> tuple[Any, ...]:
    selected = cert.get("selected") or {}
    return (
        selected.get("selector"),
        as_int(selected.get("top_k")),
        normalize_row_keys(selected.get("row_keys")),
        support_mask(cert.get("selected_term_support") or []),
    )


def resolve_path(raw: Any, fallback_state_dir: Path) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == fallback_state_dir.name:
        return fallback_state_dir.parent / path
    return fallback_state_dir / path.name


def direct_certificate_paths(direct_evidence: dict[str, Any], target_row_id: str, fallback_state_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for item in all_objects(direct_evidence):
        if item.get("row_request_id") != target_row_id:
            continue
        artifacts = item.get("source_artifacts") or {}
        for raw in artifacts.get("direct_certificates") or []:
            path = resolve_path(raw, fallback_state_dir)
            if path is not None:
                paths.append(path)
    return sorted(dict.fromkeys(paths))


def all_objects(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stack = [raw]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            out.append(item)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return out


def load_certificates(paths: list[Path]) -> list[dict[str, Any]]:
    certs = []
    for path in paths:
        if not path.is_file():
            continue
        payload = load_json(path)
        for offset, cert in enumerate(payload.get("certificates") or []):
            if not isinstance(cert, dict):
                continue
            item = dict(cert)
            item["artifact"] = str(path)
            item["artifact_offset"] = offset
            certs.append(item)
    return certs


def hints_by_key(source_hint: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    out = {}
    for hint in source_hint.get("source_hints") or []:
        if isinstance(hint, dict):
            out[adapter_hint_match_key(hint)] = hint
    return out


def scout_case_for_hint(adapter_hint: dict[str, Any]) -> dict[str, Any]:
    source_artifact = adapter_hint.get("source_artifact")
    if not source_artifact:
        return {}
    path = Path(str(source_artifact))
    if not path.is_file():
        return {}
    payload = load_json(path)
    key = adapter_hint_match_key(adapter_hint)
    for case in payload.get("case_reports") or []:
        if not isinstance(case, dict):
            continue
        if adapter_hint_match_key(case) == key:
            return case
    return {}


def coeff_rank_mod_prime(rows: list[list[int]], order: int) -> int:
    if order <= 1:
        return 0
    matrix = [[as_int(value) % order for value in row] for row in rows]
    rank = 0
    col_count = max((len(row) for row in matrix), default=0)
    for col in range(col_count):
        pivot = None
        for row_index in range(rank, len(matrix)):
            if matrix[row_index][col] % order:
                pivot = row_index
                break
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        inv = pow(matrix[rank][col], -1, order)
        matrix[rank] = [(value * inv) % order for value in matrix[rank]]
        for row_index in range(len(matrix)):
            if row_index == rank:
                continue
            factor = matrix[row_index][col] % order
            if factor:
                matrix[row_index] = [
                    (value - factor * matrix[rank][offset]) % order
                    for offset, value in enumerate(matrix[row_index])
                ]
        rank += 1
    return rank


def form_hash_u64(form: dict[str, Any], order: int) -> int:
    coeffs = [as_int(value) % order for value in form.get("coeffs") or []]
    while len(coeffs) < COEFF_COUNT:
        coeffs.append(0)
    material = {
        "coeffs": coeffs[:COEFF_COUNT],
        "order": order,
        "rhs": as_int(form.get("rhs")) % order if order else as_int(form.get("rhs")),
        "terms": [as_int(term) for term in form.get("terms") or []],
    }
    return stable_hash_u64(material)


def build_coeff_forms(cert: dict[str, Any], residual_mask: int) -> list[dict[str, Any]]:
    order = as_int(cert.get("order"))
    rows = []
    for form_index, form in enumerate(cert.get("forms") or []):
        coeffs = [as_int(value) % order for value in form.get("coeffs") or []]
        while len(coeffs) < COEFF_COUNT:
            coeffs.append(0)
        coeffs = coeffs[:COEFF_COUNT]
        form_mask = support_mask_from_coeffs(coeffs, order)
        residual_overlap_mask = form_mask & residual_mask
        rows.append(
            {
                "coeff_count": len(coeffs),
                "coeffs": coeffs,
                "form_hash_u64": form_hash_u64(form, order),
                "form_index": form_index,
                "form_support_mask": form_mask,
                "form_support_terms": terms_from_mask(form_mask),
                "residual_overlap_mask": residual_overlap_mask,
                "residual_overlap_terms": terms_from_mask(residual_overlap_mask),
                "rhs": as_int(form.get("rhs")) % order if order else as_int(form.get("rhs")),
                "terms": [as_int(term) for term in form.get("terms") or []],
            }
        )
    return rows


def source_class(cert: dict[str, Any] | None, adapter_hint: dict[str, Any], scout_case: dict[str, Any]) -> str:
    if cert is not None:
        return "direct_certificate_coefficients_available"
    if scout_case.get("shared_product_public_key_verified") or adapter_hint.get("shared_product_public_key_verified"):
        return "shared_product_verified_coefficients_missing"
    if scout_case.get("direct_public_key_verified") or adapter_hint.get("direct_public_key_verified"):
        return "direct_verified_coefficients_missing"
    return "source_unverified"


def build_lanes(
    emission_packet: dict[str, Any],
    source_hint: dict[str, Any],
    direct_evidence: dict[str, Any],
    fallback_state_dir: Path,
) -> tuple[list[dict[str, Any]], list[Path], list[dict[str, Any]]]:
    target = emission_packet.get("target_slot") or {}
    target_row_id = str(target.get("row_request_id") or "")
    cert_paths = direct_certificate_paths(direct_evidence, target_row_id, fallback_state_dir)
    certs = load_certificates(cert_paths)
    cert_index = {cert_match_key(cert): cert for cert in certs}
    hint_index = hints_by_key(source_hint)
    failures = []
    lanes = []
    for lane in emission_packet.get("residual_lanes") or []:
        key = lane_match_key(lane)
        adapter_hint = hint_index.get(key, {})
        scout_case = scout_case_for_hint(adapter_hint) if adapter_hint else {}
        cert = cert_index.get(key)
        residual_mask = as_int(lane.get("missing_target_support_mask"))
        coeff_forms = build_coeff_forms(cert, residual_mask) if cert is not None else []
        coeff_rows = [form.get("coeffs") or [] for form in coeff_forms]
        order = as_int((cert or {}).get("order"))
        coeff_union_mask = 0
        residual_overlap_union_mask = 0
        for form in coeff_forms:
            coeff_union_mask |= as_int(form.get("form_support_mask"))
            residual_overlap_union_mask |= as_int(form.get("residual_overlap_mask"))
        uncovered_residual_mask = residual_mask & ~residual_overlap_union_mask
        classification = source_class(cert, adapter_hint, scout_case)
        if not adapter_hint:
            failures.append({"code": "source_hint_adapter_match_missing", "hint_local_index": lane.get("hint_local_index")})
        lanes.append(
            {
                "adapter_source_artifact": adapter_hint.get("source_artifact"),
                "coefficient_form_count": len(coeff_forms),
                "coefficient_rank_mod_order": coeff_rank_mod_prime(coeff_rows, order) if coeff_forms else 0,
                "coefficient_union_mask": coeff_union_mask,
                "coefficient_union_terms": terms_from_mask(coeff_union_mask),
                "direct_certificate_artifact": None if cert is None else cert.get("artifact"),
                "direct_certificate_offset": -1 if cert is None else as_int(cert.get("artifact_offset"), -1),
                "direct_public_key_verified": bool(lane.get("direct_public_key_verified")),
                "direct_source_derived_secret": None if cert is None else as_int(cert.get("derived_secret")),
                "forms": coeff_forms,
                "hint_hash_u64": as_int(lane.get("hint_hash_u64")),
                "hint_local_index": as_int(lane.get("hint_local_index"), -1),
                "missing_target_support_mask": residual_mask,
                "missing_target_terms": lane.get("missing_target_terms") or terms_from_mask(residual_mask),
                "order": order,
                "residual_overlap_union_mask": residual_overlap_union_mask,
                "residual_overlap_union_terms": terms_from_mask(residual_overlap_union_mask),
                "same_salt_pair": bool(lane.get("same_salt_pair")),
                "scout_direct_public_key_verified": bool(scout_case.get("direct_public_key_verified")),
                "scout_shared_product_public_key_verified": bool(scout_case.get("shared_product_public_key_verified")),
                "source_class": classification,
                "source_class_code": SOURCE_CLASS_CODES[classification],
                "source_row_hash_u64": as_int(lane.get("source_row_hash_u64")),
                "source_selected_support_mask": as_int(lane.get("source_selected_support_mask")),
                "source_selector": lane.get("source_selector"),
                "source_top_k": as_int(lane.get("source_top_k")),
                "target_gap_exact": bool(lane.get("target_gap_exact")),
                "uncovered_residual_mask": uncovered_residual_mask,
                "uncovered_residual_terms": terms_from_mask(uncovered_residual_mask),
                "worker_action": (
                    "reuse_source_coefficients_as_guard_then_synthesize_uncovered_residual_terms"
                    if cert is not None
                    else "materialize_missing_coefficients_then_synthesize_residual_terms"
                ),
            }
        )
    return lanes, cert_paths, failures


def validate_sources(emission_packet: dict[str, Any], source_hint: dict[str, Any], direct_evidence: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if emission_packet.get("claim_status") != "SELECTED13_PRIORITY0_FFE_DIRECT_RANK_EMISSION_PACKET_READY":
        failures.append({"code": "emission_packet_not_ready", "claim_status": emission_packet.get("claim_status")})
    if emission_packet.get("failures"):
        failures.append({"code": "emission_packet_has_failures", "failures": emission_packet.get("failures")})
    if (emission_packet.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "emission_packet_summary_not_verified", "summary": emission_packet.get("summary")})
    if source_hint.get("claim_status") not in {
        "SELECTED13_MATERIALIZATION_SOURCE_HINT_ADAPTER_READY",
        "SELECTED13_MATERIALIZATION_SOURCE_HINTS_WITH_EXACT_DIRECT_PROMOTION",
    }:
        failures.append({"code": "source_hint_adapter_not_ready", "claim_status": source_hint.get("claim_status")})
    if source_hint.get("failures"):
        failures.append({"code": "source_hint_adapter_has_failures", "failures": source_hint.get("failures")})
    if (source_hint.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "source_hint_adapter_summary_not_verified", "summary": source_hint.get("summary")})
    if direct_evidence.get("claim_status") not in {
        "SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_AUDIT_READY",
        "SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_HAS_DERIVED_ROWS",
    }:
        failures.append({"code": "direct_evidence_audit_not_ready", "claim_status": direct_evidence.get("claim_status")})
    if direct_evidence.get("failures"):
        failures.append({"code": "direct_evidence_has_failures", "failures": direct_evidence.get("failures")})
    if (direct_evidence.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "direct_evidence_summary_not_verified", "summary": direct_evidence.get("summary")})
    return failures


def validate_lanes(lanes: list[dict[str, Any]], target: dict[str, Any], cert_paths: list[Path]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(lanes) != 3:
        failures.append({"code": "lane_count_unexpected", "observed": len(lanes)})
    if not cert_paths:
        failures.append({"code": "direct_certificate_artifact_missing"})
    coefficient_lanes = [lane for lane in lanes if as_int(lane.get("coefficient_form_count")) > 0]
    if len(coefficient_lanes) != 1:
        failures.append({"code": "coefficient_backed_lane_count_unexpected", "observed": len(coefficient_lanes)})
    for lane in lanes:
        residual_mask = as_int(lane.get("missing_target_support_mask"))
        if not lane.get("direct_public_key_verified"):
            failures.append({"code": "lane_not_direct_verified", "hint": lane.get("hint_local_index")})
        if not lane.get("same_salt_pair"):
            failures.append({"code": "lane_not_same_salt", "hint": lane.get("hint_local_index")})
        if not lane.get("target_gap_exact"):
            failures.append({"code": "lane_not_exact_target_gap", "hint": lane.get("hint_local_index")})
        if not (residual_mask & SELECTED13_MASK):
            failures.append({"code": "lane_residual_missing_selected13", "hint": lane.get("hint_local_index")})
        if as_int(lane.get("coefficient_form_count")):
            if as_int(lane.get("order")) <= 1:
                failures.append({"code": "coefficient_lane_order_missing", "hint": lane.get("hint_local_index")})
            if as_int(lane.get("coefficient_rank_mod_order")) == 0:
                failures.append({"code": "coefficient_lane_rank_zero", "hint": lane.get("hint_local_index")})
            if as_int(lane.get("uncovered_residual_mask")) == 0:
                failures.append({"code": "coefficient_lane_claims_full_residual_coverage", "hint": lane.get("hint_local_index")})
            for form in lane.get("forms") or []:
                if as_int(form.get("form_hash_u64")) == 0:
                    failures.append({"code": "form_hash_missing", "hint": lane.get("hint_local_index")})
    if as_int(target.get("transfer_index"), -1) != 10376:
        failures.append({"code": "target_transfer_unexpected", "transfer_index": target.get("transfer_index")})
    return failures


def render_c_header(lanes: list[dict[str, Any]], target: dict[str, Any]) -> str:
    lane_lines = []
    form_lines = []
    form_index = 0
    for lane in lanes:
        form_start = form_index
        forms = lane.get("forms") or []
        for form in forms:
            form_lines.append(
                "  {"
                f"{as_int(lane.get('hint_local_index'))}ULL, "
                f"{as_int(form.get('form_index'))}ULL, "
                f"{as_int(form.get('form_hash_u64'))}ULL, "
                f"{as_int(form.get('form_support_mask'))}ULL, "
                f"{as_int(form.get('residual_overlap_mask'))}ULL, "
                f"{as_int(form.get('rhs'))}ULL"
                "},"
            )
            form_index += 1
        lane_lines.append(
            "  {"
            f"{as_int(lane.get('hint_local_index'))}ULL, "
            f"{as_int(lane.get('source_class_code'))}ULL, "
            f"{as_int(lane.get('hint_hash_u64'))}ULL, "
            f"{as_int(lane.get('source_row_hash_u64'))}ULL, "
            f"{as_int(lane.get('missing_target_support_mask'))}ULL, "
            f"{as_int(lane.get('coefficient_union_mask'))}ULL, "
            f"{as_int(lane.get('residual_overlap_union_mask'))}ULL, "
            f"{as_int(lane.get('uncovered_residual_mask'))}ULL, "
            f"{as_int(lane.get('coefficient_form_count'))}ULL, "
            f"{as_int(lane.get('coefficient_rank_mod_order'))}ULL, "
            f"{as_int(lane.get('order'))}ULL, "
            f"{form_start}ULL, "
            f"{len(forms)}ULL, "
            f"{1 if lane.get('same_salt_pair') else 0}ULL, "
            f"{1 if lane.get('direct_public_key_verified') else 0}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_RESIDUAL_COEFF_GAP_MANIFEST_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_RESIDUAL_COEFF_GAP_MANIFEST_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_COEFF_GAP_TRANSFER {as_int(target.get('transfer_index'))}ULL
#define SELECTED13_PRIORITY0_COEFF_GAP_ROW_REQUEST_U64 {as_int(target.get('row_request_id_u64'))}ULL
#define SELECTED13_PRIORITY0_COEFF_GAP_LANE_COUNT {len(lanes)}
#define SELECTED13_PRIORITY0_COEFF_GAP_FORM_COUNT {form_index}
#define SELECTED13_PRIORITY0_COEFF_GAP_ACCEPTED_EXPORT_COUNT 0ULL
#define SELECTED13_PRIORITY0_COEFF_GAP_RELATION_DERIVED_ECDLP 0ULL

#define SELECTED13_SOURCE_CLASS_DIRECT_CERT 1ULL
#define SELECTED13_SOURCE_CLASS_SHARED_PRODUCT_MISSING_COEFFS 2ULL
#define SELECTED13_SOURCE_CLASS_DIRECT_MISSING_COEFFS 3ULL
#define SELECTED13_SOURCE_CLASS_UNVERIFIED 4ULL

typedef struct {{
  uint64_t hint_local_index;
  uint64_t source_class_code;
  uint64_t hint_hash_u64;
  uint64_t source_row_hash_u64;
  uint64_t residual_mask;
  uint64_t coefficient_union_mask;
  uint64_t residual_overlap_union_mask;
  uint64_t uncovered_residual_mask;
  uint64_t coefficient_form_count;
  uint64_t coefficient_rank_mod_order;
  uint64_t order;
  uint64_t form_start;
  uint64_t form_count;
  uint64_t same_salt_pair;
  uint64_t direct_public_key_verified;
}} selected13_priority0_coeff_gap_lane_t;

typedef struct {{
  uint64_t hint_local_index;
  uint64_t form_index;
  uint64_t form_hash_u64;
  uint64_t form_support_mask;
  uint64_t residual_overlap_mask;
  uint64_t rhs;
}} selected13_priority0_coeff_gap_form_t;

static const selected13_priority0_coeff_gap_lane_t SELECTED13_PRIORITY0_COEFF_GAP_LANES[] = {{
{chr(10).join(lane_lines)}
}};

static const selected13_priority0_coeff_gap_form_t SELECTED13_PRIORITY0_COEFF_GAP_FORMS[] = {{
{chr(10).join(form_lines)}
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
  uint64_t coefficient_lane_count = 0;
  uint64_t missing_coeff_lane_count = 0;
  uint64_t verified_lane_count = 0;
  uint64_t same_salt_lane_count = 0;
  uint64_t uncovered_lane_count = 0;
  uint64_t form_count_sum = 0;

  const size_t lane_count = sizeof(SELECTED13_PRIORITY0_COEFF_GAP_LANES) / sizeof(SELECTED13_PRIORITY0_COEFF_GAP_LANES[0]);
  const size_t form_count = sizeof(SELECTED13_PRIORITY0_COEFF_GAP_FORMS) / sizeof(SELECTED13_PRIORITY0_COEFF_GAP_FORMS[0]);
  if (lane_count != SELECTED13_PRIORITY0_COEFF_GAP_LANE_COUNT) failure_count++;
  if (form_count != SELECTED13_PRIORITY0_COEFF_GAP_FORM_COUNT) failure_count++;
  if (SELECTED13_PRIORITY0_COEFF_GAP_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_COEFF_GAP_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;

  for (size_t i = 0; i < lane_count; i++) {{
    const selected13_priority0_coeff_gap_lane_t *lane = &SELECTED13_PRIORITY0_COEFF_GAP_LANES[i];
    if (lane->hint_hash_u64 == 0ULL || lane->source_row_hash_u64 == 0ULL) failure_count++;
    if ((lane->residual_mask & selected13_mask) == 0ULL) failure_count++;
    if (lane->same_salt_pair) same_salt_lane_count++;
    if (lane->direct_public_key_verified) verified_lane_count++;
    if (lane->coefficient_form_count > 0ULL) {{
      coefficient_lane_count++;
      if (lane->source_class_code != SELECTED13_SOURCE_CLASS_DIRECT_CERT) failure_count++;
      if (lane->coefficient_rank_mod_order == 0ULL || lane->order <= 1ULL) failure_count++;
      if (lane->uncovered_residual_mask == 0ULL) failure_count++;
    }} else {{
      missing_coeff_lane_count++;
      if (lane->source_class_code == SELECTED13_SOURCE_CLASS_DIRECT_CERT) failure_count++;
    }}
    if (lane->uncovered_residual_mask != 0ULL) uncovered_lane_count++;
    if (lane->form_start + lane->form_count > form_count) failure_count++;
    form_count_sum += lane->form_count;
  }}
  for (size_t i = 0; i < form_count; i++) {{
    const selected13_priority0_coeff_gap_form_t *form = &SELECTED13_PRIORITY0_COEFF_GAP_FORMS[i];
    if (form->form_hash_u64 == 0ULL || form->form_support_mask == 0ULL) failure_count++;
  }}
  if (form_count_sum != form_count) failure_count++;
  if (coefficient_lane_count != 1ULL) failure_count++;
  if (missing_coeff_lane_count != 2ULL) failure_count++;
  if (same_salt_lane_count != lane_count) failure_count++;
  if (verified_lane_count != lane_count) failure_count++;
  if (uncovered_lane_count != lane_count) failure_count++;

  printf("selected13_priority0_residual_coeff_gap_preflight transfer=%llu lanes=%llu coeff_lanes=%llu missing_coeff_lanes=%llu forms=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_COEFF_GAP_TRANSFER,
         (unsigned long long)lane_count,
         (unsigned long long)coefficient_lane_count,
         (unsigned long long)missing_coeff_lane_count,
         (unsigned long long)form_count,
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
    with tempfile.TemporaryDirectory(prefix="selected13_priority0_coeff_gap_", dir=str(temp_root)) as tmp:
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


def summarize(lanes: list[dict[str, Any]], failures: list[dict[str, Any]], native_preflight: dict[str, Any]) -> dict[str, Any]:
    classes = Counter(str(lane.get("source_class")) for lane in lanes)
    return {
        "accepted_relation_export_count": 0,
        "coefficient_backed_lane_count": sum(1 for lane in lanes if as_int(lane.get("coefficient_form_count")) > 0),
        "coefficient_form_count": sum(as_int(lane.get("coefficient_form_count")) for lane in lanes),
        "failure_count": len(failures),
        "lane_count": len(lanes),
        "native_preflight_verified": bool(native_preflight.get("verified")),
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "source_class_counts": dict(sorted(classes.items())),
        "uncovered_residual_term_sets": [lane.get("uncovered_residual_terms") for lane in lanes],
        "verified": not failures,
        "worker_interpretation": (
            "Only one 10376 residual lane has direct-certificate coefficient forms. "
            "The two hybrid lanes remain verified source guards but need fresh coefficient "
            "materialization before a target direct/rank export can be accepted."
        ),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    emission_path = Path(args.emission_packet)
    source_hint_path = Path(args.source_hint)
    direct_evidence_path = Path(args.direct_evidence)
    emission_packet = load_json(emission_path)
    source_hint = load_json(source_hint_path)
    direct_evidence = load_json(direct_evidence_path)
    failures = validate_sources(emission_packet, source_hint, direct_evidence)
    lanes, cert_paths, lane_failures = build_lanes(
        emission_packet,
        source_hint,
        direct_evidence,
        fallback_state_dir=emission_path.parent,
    )
    failures.extend(lane_failures)
    target = emission_packet.get("target_slot") or {}
    failures.extend(validate_lanes(lanes, target, cert_paths))
    payload_hash = stable_hash_u64({"lanes": lanes, "target": target})
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_RESIDUAL_COEFF_GAP_MANIFEST_READY"
            if not failures
            else "SELECTED13_PRIORITY0_RESIDUAL_COEFF_GAP_MANIFEST_FAILED"
        ),
        "parameters": {
            "direct_evidence": str(direct_evidence_path),
            "emission_packet": str(emission_path),
            "source_hint": str(source_hint_path),
        },
        "artifacts": {
            "direct_certificate_paths": [str(path) for path in cert_paths],
        },
        "packet_hash_u64": payload_hash,
        "target_slot": target,
        "residual_coeff_lanes": lanes,
        "required_worker_outputs": {
            "must_materialize_missing_hybrid_coefficients": True,
            "must_synthesize_uncovered_residual_terms": True,
            "must_emit_target_direct_rank_export": True,
            "must_verify_public_key": True,
            "must_set_relation_derived_ecdlp": True,
        },
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "coefficients_are_source_guards_not_target_exports": True,
            "fresh_direct_rank_worker_required": True,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emission-packet", default=str(DEFAULT_EMISSION_PACKET))
    parser.add_argument("--source-hint", default=str(DEFAULT_SOURCE_HINT))
    parser.add_argument("--direct-evidence", default=str(DEFAULT_DIRECT_EVIDENCE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["residual_coeff_lanes"], payload["target_slot"]))
    payload["artifacts"]["c_header"] = str(header_path)
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_RESIDUAL_COEFF_GAP_MANIFEST_FAILED"
    payload["summary"] = summarize(payload["residual_coeff_lanes"], payload["failures"], native_preflight)
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
