#!/usr/bin/env python3
"""Bridge mounted 10376 direct certificates into the selected13 priority-0 lane.

The mounted direct-relation certificate file has fresh public-key-verified
evidence for transfer 10376 and secret 5859.  This bridge records that evidence
as a positive control for the selected13 worker, while preserving the stricter
target-export gate: the exact selected13 row request still needs a fresh
FFE/summation-polynomial result with `relation_derived_ecdlp=true`.
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


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_direct_certificate_bridge.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
MOUNTED_STATE_DIR = Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state")
DEFAULT_TARGET_GATE = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_target_export_verifier_gate_10376_probe.json"
DEFAULT_VALIDATOR = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_result_validator_10376_probe.json"
DEFAULT_DIRECT_CERTIFICATE = (
    MOUNTED_STATE_DIR
    / "low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_10376_10383_probe.json"
)
DEFAULT_SUPPORT_SCOUT = (
    MOUNTED_STATE_DIR
    / "low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_10376_10383_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_direct_certificate_bridge_10376_probe.json"
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_direct_certificate_bridge_10376_probe.h"
)

EXPECTED_TRANSFER = 10376
EXPECTED_SECRET = 5859
EXPECTED_DIRECT_CERT_COUNT = 3
EXPECTED_DIRECT_CERT_PASSING_COUNT = 3
EXPECTED_PUBLIC_VERIFIED_CERT_COUNT = 3
EXPECTED_BELOW_RHO_CERT_COUNT = 3
EXPECTED_TARGET_PAIR_CERT_COUNT = 1
EXPECTED_EXACT_TARGET_SLOT_CERT_COUNT = 0
EXPECTED_ROWS_10376_COUNT = 40
EXPECTED_TARGET_PAIR_ROW_COUNT = 20
EXPECTED_TARGET_PAIR_DIRECT_VERIFIED_COUNT = 10
EXPECTED_TARGET_PAIR_SHARED_VERIFIED_COUNT = 8
EXPECTED_TARGET_PAIR_TERM8_ROW_COUNT = 3
EXPECTED_TARGET_PAIR_TERM12_ROW_COUNT = 6
EXPECTED_TARGET_PAIR_TERM8_DIRECT_VERIFIED_COUNT = 0
EXPECTED_TARGET_PAIR_TERM12_DIRECT_VERIFIED_COUNT = 0
EXPECTED_EXACT_TARGET_SLOT_ROW_COUNT = 1
EXPECTED_EXACT_TARGET_SLOT_DIRECT_VERIFIED_COUNT = 0
EXPECTED_EXACT_TARGET_SLOT_SHARED_VERIFIED_COUNT = 0


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


def row_key_set(row_keys: Any) -> set[str]:
    return {str(row) for row in row_keys or []}


def same_row_pair(row_keys: Any, target_rows: set[str]) -> bool:
    return row_key_set(row_keys) == target_rows


def term_support(value: dict[str, Any]) -> list[int]:
    return [as_int(term, -1) for term in value.get("selected_term_support") or []]


def same_exact_target_slot(row: dict[str, Any], target_slot: dict[str, Any]) -> bool:
    return (
        same_row_pair(row.get("row_keys"), row_key_set(target_slot.get("row_keys")))
        and str(row.get("selector")) == str(target_slot.get("selector"))
        and as_int(row.get("top_k"), -1) == as_int(target_slot.get("top_k"), -1)
        and term_support(row) == term_support(target_slot)
    )


def cert_selected(cert: dict[str, Any]) -> dict[str, Any]:
    selected = cert.get("selected")
    return selected if isinstance(selected, dict) else {}


def cert_rows(cert: dict[str, Any]) -> Any:
    selected = cert_selected(cert)
    return selected.get("row_keys") or cert.get("row_keys") or []


def cert_transfer(cert: dict[str, Any]) -> int:
    selected = cert_selected(cert)
    return as_int(selected.get("transfer_index", cert.get("transfer_index")), -1)


def cert_selector(cert: dict[str, Any]) -> str:
    selected = cert_selected(cert)
    return str(selected.get("selector", cert.get("selector", "")))


def cert_top_k(cert: dict[str, Any]) -> int:
    selected = cert_selected(cert)
    return as_int(selected.get("top_k", cert.get("top_k")), -1)


def cert_secret(cert: dict[str, Any]) -> int:
    selected = cert_selected(cert)
    return as_int(cert.get("derived_secret", selected.get("secret", cert.get("secret"))), -1)


def cert_ops_over_rho(cert: dict[str, Any]) -> float | None:
    selected = cert_selected(cert)
    return as_float(selected.get("direct_ops_over_rho", cert.get("direct_ops_over_rho")))


def cert_rho(cert: dict[str, Any]) -> int:
    selected = cert_selected(cert)
    return as_int(selected.get("rho", cert.get("rho")), -1)


def summarize_certificate(cert: dict[str, Any], target_slot: dict[str, Any]) -> dict[str, Any]:
    forms = cert.get("forms") or []
    support = term_support(cert)
    row_pair_match = same_row_pair(cert_rows(cert), row_key_set(target_slot.get("row_keys")))
    exact_target_slot_match = (
        row_pair_match
        and cert_selector(cert) == str(target_slot.get("selector"))
        and cert_top_k(cert) == as_int(target_slot.get("top_k"), -1)
        and support == term_support(target_slot)
    )
    ops_over_rho = cert_ops_over_rho(cert)
    return {
        "below_rho": ops_over_rho is not None and ops_over_rho < 1.0,
        "certificate_status": cert.get("certificate_status"),
        "clause": cert.get("clause"),
        "direct_ops_over_rho": ops_over_rho,
        "exact_target_slot_match": exact_target_slot_match,
        "expected_secret": as_int(cert.get("expected_secret"), -1),
        "form_count": len(forms),
        "form_rhs_values": [as_int(form.get("rhs"), -1) for form in forms if isinstance(form, dict)],
        "form_term_supports": [form.get("terms") for form in forms if isinstance(form, dict)],
        "public_key_verified": bool(cert.get("public_key_verified")),
        "rho": cert_rho(cert),
        "row_pair_matches_target": row_pair_match,
        "row_keys": list(cert_rows(cert)),
        "secret": cert_secret(cert),
        "secret_matches_expected": cert_secret(cert) == EXPECTED_SECRET,
        "selected_term_support": support,
        "selector": cert_selector(cert),
        "term8_in_support": 8 in support,
        "term12_in_support": 12 in support,
        "top_k": cert_top_k(cert),
        "transfer_index": cert_transfer(cert),
    }


def summarize_scout_row(row: dict[str, Any], target_slot: dict[str, Any]) -> dict[str, Any]:
    support = term_support(row)
    return {
        "direct_ops_over_rho": as_float(row.get("direct_ops_over_rho")),
        "direct_public_key_verified": bool(row.get("direct_public_key_verified")),
        "exact_target_slot_match": same_exact_target_slot(row, target_slot),
        "priority_hits": row.get("priority_hits") or [],
        "public_product_gate_selected": bool(row.get("public_product_gate_selected")),
        "row_keys": row.get("row_keys") or [],
        "selected_term_support": support,
        "selector": row.get("selector"),
        "shared_product_ops_over_rho": as_float(row.get("shared_product_ops_over_rho")),
        "shared_product_public_key_verified": bool(row.get("shared_product_public_key_verified")),
        "shared_product_rank": as_int(row.get("shared_product_rank"), -1),
        "term8_in_support": 8 in support,
        "term12_in_support": 12 in support,
        "top_k": as_int(row.get("top_k"), -1),
        "transfer_index": as_int(row.get("transfer_index"), -1),
    }


def build_bridge(
    target_gate: dict[str, Any],
    validator: dict[str, Any],
    direct_certificate: dict[str, Any],
    support_scout: dict[str, Any],
    direct_certificate_path: Path,
    support_scout_path: Path,
) -> dict[str, Any]:
    target_slot = target_gate.get("target_slot") or {}
    target_rows = row_key_set(target_slot.get("row_keys"))
    target_support = term_support(target_slot)
    transfer = as_int(target_slot.get("transfer_index"), -1)
    certificates = [
        summarize_certificate(cert, target_slot)
        for cert in direct_certificate.get("certificates") or []
        if cert_transfer(cert) == transfer
    ]
    passing_certs = [
        cert
        for cert in certificates
        if cert.get("certificate_status") == "PUBLIC_DIRECT_RELATION_EQUATIONS_VERIFY_PUBLIC_KEY"
        and cert.get("public_key_verified")
        and cert.get("secret_matches_expected")
    ]
    target_pair_certs = [cert for cert in certificates if cert.get("row_pair_matches_target")]
    exact_target_slot_certs = [cert for cert in certificates if cert.get("exact_target_slot_match")]

    scout_rows_10376 = [
        row for row in support_scout.get("case_reports") or [] if as_int(row.get("transfer_index"), -1) == transfer
    ]
    target_pair_rows = [summarize_scout_row(row, target_slot) for row in scout_rows_10376 if same_row_pair(row.get("row_keys"), target_rows)]
    exact_target_slot_rows = [row for row in target_pair_rows if row.get("exact_target_slot_match")]
    target_pair_direct_verified_rows = [row for row in target_pair_rows if row.get("direct_public_key_verified")]
    target_pair_shared_verified_rows = [row for row in target_pair_rows if row.get("shared_product_public_key_verified")]
    target_pair_term8_rows = [row for row in target_pair_rows if row.get("term8_in_support")]
    target_pair_term12_rows = [row for row in target_pair_rows if row.get("term12_in_support")]
    target_pair_term8_direct_rows = [
        row for row in target_pair_term8_rows if row.get("direct_public_key_verified")
    ]
    target_pair_term12_direct_rows = [
        row for row in target_pair_term12_rows if row.get("direct_public_key_verified")
    ]
    exact_direct_rows = [row for row in exact_target_slot_rows if row.get("direct_public_key_verified")]
    exact_shared_rows = [row for row in exact_target_slot_rows if row.get("shared_product_public_key_verified")]

    below_rho_certs = [cert for cert in certificates if cert.get("below_rho")]
    public_verified_certs = [cert for cert in certificates if cert.get("public_key_verified")]
    secret_match_certs = [cert for cert in certificates if cert.get("secret_matches_expected")]
    source_summary = direct_certificate.get("summary") or {}
    validator_summary = validator.get("summary") or {}
    gate_summary = target_gate.get("summary") or {}
    summary = {
        "accepted_relation_export_count": as_int(gate_summary.get("accepted_relation_export_count")),
        "below_rho_certificate_count": len(below_rho_certs),
        "certificate_count": len(certificates),
        "direct_certificate_claim_status": source_summary.get("claim_status"),
        "exact_target_slot_certificate_count": len(exact_target_slot_certs),
        "exact_target_slot_direct_verified_count": len(exact_direct_rows),
        "exact_target_slot_row_count": len(exact_target_slot_rows),
        "exact_target_slot_shared_verified_count": len(exact_shared_rows),
        "expected_secret": EXPECTED_SECRET,
        "failure_count": 0,
        "native_preflight_verified": False,
        "passing_certificate_count": len(passing_certs),
        "pollard_rho_speedup_claimed": False,
        "public_key_verified_certificate_count": len(public_verified_certs),
        "relation_derived_ecdlp": bool(gate_summary.get("relation_derived_ecdlp")),
        "rows_10376_count": len(scout_rows_10376),
        "secret_match_certificate_count": len(secret_match_certs),
        "target_pair_certificate_count": len(target_pair_certs),
        "target_pair_direct_verified_count": len(target_pair_direct_verified_rows),
        "target_pair_row_count": len(target_pair_rows),
        "target_pair_shared_verified_count": len(target_pair_shared_verified_rows),
        "target_pair_term12_direct_verified_count": len(target_pair_term12_direct_rows),
        "target_pair_term12_row_count": len(target_pair_term12_rows),
        "target_pair_term8_direct_verified_count": len(target_pair_term8_direct_rows),
        "target_pair_term8_row_count": len(target_pair_term8_rows),
        "target_slot_selector": target_slot.get("selector"),
        "target_slot_support_count": len(target_support),
        "target_slot_top_k": as_int(target_slot.get("top_k"), -1),
        "target_transfer": transfer,
        "validator_target_export_accepted": bool(validator_summary.get("target_export_accepted")),
        "verified": True,
        "worker_interpretation": (
            "Mounted direct certificates prove the same transfer and secret under the public key, "
            "including one same target-row pair at top_k=12.  They remain positive controls only: "
            "the exact selected13 top_k=16 target slot is not direct-verified and the validator has "
            "no accepted relation-derived export."
        ),
    }

    failures = []
    checks = [
        ("transfer_unexpected", transfer == EXPECTED_TRANSFER, transfer),
        ("secret_unexpected", summary["expected_secret"] == EXPECTED_SECRET, summary["expected_secret"]),
        ("certificate_count_unexpected", summary["certificate_count"] == EXPECTED_DIRECT_CERT_COUNT, summary["certificate_count"]),
        (
            "passing_certificate_count_unexpected",
            summary["passing_certificate_count"] == EXPECTED_DIRECT_CERT_PASSING_COUNT,
            summary["passing_certificate_count"],
        ),
        (
            "public_verified_certificate_count_unexpected",
            summary["public_key_verified_certificate_count"] == EXPECTED_PUBLIC_VERIFIED_CERT_COUNT,
            summary["public_key_verified_certificate_count"],
        ),
        (
            "below_rho_certificate_count_unexpected",
            summary["below_rho_certificate_count"] == EXPECTED_BELOW_RHO_CERT_COUNT,
            summary["below_rho_certificate_count"],
        ),
        (
            "target_pair_certificate_count_unexpected",
            summary["target_pair_certificate_count"] == EXPECTED_TARGET_PAIR_CERT_COUNT,
            summary["target_pair_certificate_count"],
        ),
        (
            "exact_target_slot_certificate_count_unexpected",
            summary["exact_target_slot_certificate_count"] == EXPECTED_EXACT_TARGET_SLOT_CERT_COUNT,
            summary["exact_target_slot_certificate_count"],
        ),
        ("rows_10376_count_unexpected", summary["rows_10376_count"] == EXPECTED_ROWS_10376_COUNT, summary["rows_10376_count"]),
        (
            "target_pair_row_count_unexpected",
            summary["target_pair_row_count"] == EXPECTED_TARGET_PAIR_ROW_COUNT,
            summary["target_pair_row_count"],
        ),
        (
            "target_pair_direct_verified_count_unexpected",
            summary["target_pair_direct_verified_count"] == EXPECTED_TARGET_PAIR_DIRECT_VERIFIED_COUNT,
            summary["target_pair_direct_verified_count"],
        ),
        (
            "target_pair_shared_verified_count_unexpected",
            summary["target_pair_shared_verified_count"] == EXPECTED_TARGET_PAIR_SHARED_VERIFIED_COUNT,
            summary["target_pair_shared_verified_count"],
        ),
        (
            "target_pair_term8_row_count_unexpected",
            summary["target_pair_term8_row_count"] == EXPECTED_TARGET_PAIR_TERM8_ROW_COUNT,
            summary["target_pair_term8_row_count"],
        ),
        (
            "target_pair_term12_row_count_unexpected",
            summary["target_pair_term12_row_count"] == EXPECTED_TARGET_PAIR_TERM12_ROW_COUNT,
            summary["target_pair_term12_row_count"],
        ),
        (
            "target_pair_term8_direct_verified_count_unexpected",
            summary["target_pair_term8_direct_verified_count"] == EXPECTED_TARGET_PAIR_TERM8_DIRECT_VERIFIED_COUNT,
            summary["target_pair_term8_direct_verified_count"],
        ),
        (
            "target_pair_term12_direct_verified_count_unexpected",
            summary["target_pair_term12_direct_verified_count"] == EXPECTED_TARGET_PAIR_TERM12_DIRECT_VERIFIED_COUNT,
            summary["target_pair_term12_direct_verified_count"],
        ),
        (
            "exact_target_slot_row_count_unexpected",
            summary["exact_target_slot_row_count"] == EXPECTED_EXACT_TARGET_SLOT_ROW_COUNT,
            summary["exact_target_slot_row_count"],
        ),
        (
            "exact_target_slot_direct_verified_count_unexpected",
            summary["exact_target_slot_direct_verified_count"] == EXPECTED_EXACT_TARGET_SLOT_DIRECT_VERIFIED_COUNT,
            summary["exact_target_slot_direct_verified_count"],
        ),
        (
            "exact_target_slot_shared_verified_count_unexpected",
            summary["exact_target_slot_shared_verified_count"] == EXPECTED_EXACT_TARGET_SLOT_SHARED_VERIFIED_COUNT,
            summary["exact_target_slot_shared_verified_count"],
        ),
        (
            "accepted_relation_export_count_nonzero",
            summary["accepted_relation_export_count"] == 0,
            summary["accepted_relation_export_count"],
        ),
        ("relation_derived_ecdlp_unexpected", summary["relation_derived_ecdlp"] is False, summary["relation_derived_ecdlp"]),
        (
            "validator_target_export_accepted_unexpected",
            summary["validator_target_export_accepted"] is False,
            summary["validator_target_export_accepted"],
        ),
    ]
    for code, ok, observed in checks:
        if not ok:
            failures.append({"code": code, "observed": observed})

    summary["failure_count"] = len(failures)
    summary["verified"] = not failures
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_DIRECT_CERTIFICATE_BRIDGE_READY"
            if not failures
            else "SELECTED13_PRIORITY0_DIRECT_CERTIFICATE_BRIDGE_FAILED"
        ),
        "parameters": {
            "direct_certificate": str(direct_certificate_path),
            "support_scout": str(support_scout_path),
            "target_gate": str(DEFAULT_TARGET_GATE),
            "validator": str(DEFAULT_VALIDATOR),
        },
        "packet_hash_u64": stable_hash_u64(
            {
                "certificates": certificates,
                "target_pair_rows": target_pair_rows,
                "target_slot": target_slot,
                "validator_summary": validator_summary,
            }
        ),
        "target_slot": target_slot,
        "direct_certificate_bridge": {
            "certificates": certificates,
            "exact_target_slot_certificates": exact_target_slot_certs,
            "source_summary": source_summary,
            "target_pair_certificates": target_pair_certs,
        },
        "support_scout_bridge": {
            "exact_target_slot_rows": exact_target_slot_rows,
            "source_summary": support_scout.get("summary") or {},
            "target_pair_direct_verified_rows": target_pair_direct_verified_rows,
            "target_pair_rows": target_pair_rows,
            "target_pair_term12_rows": target_pair_term12_rows,
            "target_pair_term8_rows": target_pair_term8_rows,
        },
        "term812_boundary": {
            "exact_target_slot_has_term8": any(row.get("term8_in_support") for row in exact_target_slot_rows),
            "exact_target_slot_has_term12": any(row.get("term12_in_support") for row in exact_target_slot_rows),
            "target_pair_term8_direct_verified_count": len(target_pair_term8_direct_rows),
            "target_pair_term12_direct_verified_count": len(target_pair_term12_direct_rows),
            "terms_8_12_still_need_residual_synthesis": True,
        },
        "validator_bridge_status": {
            "target_export_gate_claim_status": target_gate.get("claim_status"),
            "target_export_gate_summary": gate_summary,
            "validator_claim_status": validator.get("claim_status"),
            "validator_summary": validator_summary,
        },
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "direct_certificates_are_positive_controls_only": True,
            "exact_target_slot_certificate_present": False,
            "exact_target_slot_direct_verified": False,
            "external_kernel_result_emitted": False,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "selected13_validator_acceptance_ready": False,
        },
        "failures": failures,
        "summary": summary,
    }


def render_c_header(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_DIRECT_CERTIFICATE_BRIDGE_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_DIRECT_CERTIFICATE_BRIDGE_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TRANSFER {as_int(summary.get('target_transfer'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_EXPECTED_SECRET {as_int(summary.get('expected_secret'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_CERT_COUNT {as_int(summary.get('certificate_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_PASSING_CERT_COUNT {as_int(summary.get('passing_certificate_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_PUBLIC_VERIFIED_CERT_COUNT {as_int(summary.get('public_key_verified_certificate_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_BELOW_RHO_CERT_COUNT {as_int(summary.get('below_rho_certificate_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_CERT_COUNT {as_int(summary.get('target_pair_certificate_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_EXACT_TARGET_SLOT_CERT_COUNT {as_int(summary.get('exact_target_slot_certificate_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_ROWS_10376 {as_int(summary.get('rows_10376_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_ROWS {as_int(summary.get('target_pair_row_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_DIRECT_VERIFIED {as_int(summary.get('target_pair_direct_verified_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_SHARED_VERIFIED {as_int(summary.get('target_pair_shared_verified_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_TERM8_ROWS {as_int(summary.get('target_pair_term8_row_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_TERM12_ROWS {as_int(summary.get('target_pair_term12_row_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_TERM8_DIRECT_VERIFIED {as_int(summary.get('target_pair_term8_direct_verified_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_TERM12_DIRECT_VERIFIED {as_int(summary.get('target_pair_term12_direct_verified_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_EXACT_TARGET_SLOT_ROWS {as_int(summary.get('exact_target_slot_row_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_EXACT_TARGET_SLOT_DIRECT_VERIFIED {as_int(summary.get('exact_target_slot_direct_verified_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_EXACT_TARGET_SLOT_SHARED_VERIFIED {as_int(summary.get('exact_target_slot_shared_verified_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_ACCEPTED_EXPORT_COUNT {as_int(summary.get('accepted_relation_export_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_RELATION_DERIVED_ECDLP {1 if summary.get('relation_derived_ecdlp') else 0}ULL

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

int main(void) {{
  uint64_t failure_count = 0;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TRANSFER != {EXPECTED_TRANSFER}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_EXPECTED_SECRET != {EXPECTED_SECRET}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_CERT_COUNT != {EXPECTED_DIRECT_CERT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_PASSING_CERT_COUNT != {EXPECTED_DIRECT_CERT_PASSING_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_PUBLIC_VERIFIED_CERT_COUNT != {EXPECTED_PUBLIC_VERIFIED_CERT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_BELOW_RHO_CERT_COUNT != {EXPECTED_BELOW_RHO_CERT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_CERT_COUNT != {EXPECTED_TARGET_PAIR_CERT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_EXACT_TARGET_SLOT_CERT_COUNT != {EXPECTED_EXACT_TARGET_SLOT_CERT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_ROWS_10376 != {EXPECTED_ROWS_10376_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_ROWS != {EXPECTED_TARGET_PAIR_ROW_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_DIRECT_VERIFIED != {EXPECTED_TARGET_PAIR_DIRECT_VERIFIED_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_SHARED_VERIFIED != {EXPECTED_TARGET_PAIR_SHARED_VERIFIED_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_TERM8_ROWS != {EXPECTED_TARGET_PAIR_TERM8_ROW_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_TERM12_ROWS != {EXPECTED_TARGET_PAIR_TERM12_ROW_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_TERM8_DIRECT_VERIFIED != {EXPECTED_TARGET_PAIR_TERM8_DIRECT_VERIFIED_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_TERM12_DIRECT_VERIFIED != {EXPECTED_TARGET_PAIR_TERM12_DIRECT_VERIFIED_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_EXACT_TARGET_SLOT_ROWS != {EXPECTED_EXACT_TARGET_SLOT_ROW_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_EXACT_TARGET_SLOT_DIRECT_VERIFIED != {EXPECTED_EXACT_TARGET_SLOT_DIRECT_VERIFIED_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_EXACT_TARGET_SLOT_SHARED_VERIFIED != {EXPECTED_EXACT_TARGET_SLOT_SHARED_VERIFIED_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;

  printf("selected13_priority0_direct_certificate_bridge_preflight transfer=%llu certs=%llu passing=%llu target_pair_certs=%llu exact_slot_certs=%llu target_pair_direct=%llu target_pair_term8_direct=%llu target_pair_term12_direct=%llu exact_slot_direct=%llu accepted=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TRANSFER,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_CERT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_PASSING_CERT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_CERT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_EXACT_TARGET_SLOT_CERT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_DIRECT_VERIFIED,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_TERM8_DIRECT_VERIFIED,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_TARGET_PAIR_TERM12_DIRECT_VERIFIED,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_EXACT_TARGET_SLOT_DIRECT_VERIFIED,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_BRIDGE_ACCEPTED_EXPORT_COUNT,
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
    with tempfile.TemporaryDirectory(prefix="selected13_direct_cert_bridge_", dir=str(temp_root)) as tmp:
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
    parser.add_argument("--target-gate", default=str(DEFAULT_TARGET_GATE))
    parser.add_argument("--validator", default=str(DEFAULT_VALIDATOR))
    parser.add_argument("--direct-certificate", default=str(DEFAULT_DIRECT_CERTIFICATE))
    parser.add_argument("--support-scout", default=str(DEFAULT_SUPPORT_SCOUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_gate_path = Path(args.target_gate)
    validator_path = Path(args.validator)
    direct_certificate_path = Path(args.direct_certificate)
    support_scout_path = Path(args.support_scout)
    payload = build_bridge(
        load_json(target_gate_path),
        load_json(validator_path),
        load_json(direct_certificate_path),
        load_json(support_scout_path),
        direct_certificate_path,
        support_scout_path,
    )
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload))
    payload["artifacts"] = {"c_header": str(header_path)}
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_DIRECT_CERTIFICATE_BRIDGE_FAILED"
    payload["summary"]["native_preflight_verified"] = bool(native_preflight.get("verified"))
    payload["summary"]["failure_count"] = len(payload["failures"])
    payload["summary"]["verified"] = not payload["failures"]
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
