#!/usr/bin/env python3
"""Audit selected13 materialized targets against direct verifier evidence.

The fresh materialization adapter binds packet row identities to support-scout
cases.  This audit follows the scout's upstream artifacts back to the direct
verifier labels and equation-certificate files, separating three states:

* exact target direct certificate already exists,
* exact target has verifier replay relations but is rank-deficient, or
* exact target has no direct relations and needs fresh FFE emission.

It is deliberately an audit/worker gate.  It does not invent relation forms,
derive an ECDLP scalar, or claim a Pollard-rho speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_selected13_direct_verification_audit.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SUPPORT_STATE_DIR = Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state")
DEFAULT_MATERIALIZATION = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_fresh_materialization_adapter_9981_9943_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_direct_verification_audit_9981_9943_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_direct_verification_audit_9981_9943_probe.h"

CLASS_CODES = {
    "DIRECT_CERTIFICATE_PRESENT": 1,
    "FRESH_DIRECT_RELATION_REQUIRED_NO_RELATIONS": 2,
    "FRESH_DIRECT_RELATION_REQUIRED_RANK_DEFICIENT": 3,
    "FRESH_DIRECT_VERIFICATION_REQUIRED_SECRET_MISSING": 4,
    "DIRECT_VERIFIER_EVIDENCE_MISSING": 5,
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


def digest_u64(raw: Any) -> int:
    if raw is None:
        return 0
    text = str(raw)
    suffix = text.rsplit("_", 1)[-1]
    if suffix and all(ch in "0123456789abcdefABCDEF" for ch in suffix):
        return int(suffix[-16:], 16)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def row_key_tuple(raw: Any) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in (raw or [])))


def support_tuple(raw: Any) -> tuple[int, ...]:
    return tuple(sorted(as_int(item) for item in (raw or [])))


def exact_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item.get("target") or ""),
        as_int(item.get("transfer_index"), -1),
        str(item.get("selector") or item.get("row_selector") or ""),
        as_int(item.get("top_k"), -1),
        row_key_tuple(item.get("row_keys")),
    )


def case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(case.get("target") or ""),
        as_int(case.get("transfer_index"), -1),
        str(case.get("selector") or ""),
        as_int(case.get("top_k"), -1),
        row_key_tuple(case.get("row_keys")),
    )


def range_from_path(path: Path) -> str | None:
    match = re.search(r"_(\d{4})_(\d{4})(?:_|_probe)", path.name)
    if not match:
        return None
    return f"{match.group(1)}_{match.group(2)}"


def resolve_state_artifact(raw: Any, support_state_dir: Path) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == support_state_dir.name:
        return support_state_dir.parent / path
    return support_state_dir / path.name


def all_objects(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(value, dict):
        out.append(value)
        for child in value.values():
            out.extend(all_objects(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(all_objects(child))
    return out


def find_rows(payload: dict[str, Any], key: tuple[Any, ...]) -> list[dict[str, Any]]:
    rows = []
    for item in all_objects(payload):
        if "transfer_index" not in item or "row_keys" not in item:
            continue
        if exact_key(item) == key:
            rows.append(item)
    return rows


def find_same_transfer_rows(payload: dict[str, Any], target: str, transfer: int) -> list[dict[str, Any]]:
    return [
        item
        for item in all_objects(payload)
        if str(item.get("target") or "") == target
        and as_int(item.get("transfer_index"), -1) == transfer
        and item.get("row_keys") is not None
    ]


def certificate_paths_for_scout(scout_path: Path, support_state_dir: Path) -> list[Path]:
    range_label = range_from_path(scout_path)
    if not range_label:
        return []
    return sorted(
        support_state_dir.glob(
            f"low_term_total2_direct_relation_equation_certificate_*_{range_label}_probe.json"
        )
    )


def cert_selected_key(cert: dict[str, Any]) -> tuple[Any, ...]:
    selected = cert.get("selected") or {}
    return exact_key(selected)


def load_certificates(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    certs: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        payload = load_json(path)
        for offset, cert in enumerate(payload.get("certificates") or []):
            item = dict(cert)
            item["artifact"] = str(path)
            item["artifact_offset"] = offset
            certs.append(item)
    return certs, missing


def compact_gate_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "derived_secret": row.get("direct_union_derived_secret", row.get("derived_secret")),
        "direct_ops_over_rho": as_float(row.get("direct_verifier_replay_ops_over_rho", row.get("ops_over_rho"))),
        "public_key_verified": bool(row.get("direct_union_public_key_verified", row.get("public_key_verified"))),
        "rank": as_int(row.get("direct_union_rank", row.get("rank"))),
        "relation_count": as_int(row.get("direct_union_relation_count", row.get("relation_count"))),
        "selector": row.get("selector") or row.get("row_selector"),
        "top_k": as_int(row.get("top_k"), -1),
    }


def compact_source_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "below_rho": bool(row.get("below_rho")),
        "derived_secret": row.get("derived_secret"),
        "ops_over_rho": as_float(row.get("ops_over_rho")),
        "public_key_verified": bool(row.get("public_key_verified")),
        "rank": as_int(row.get("rank")),
        "relation_count": as_int(row.get("relation_count")),
        "selector": row.get("selector") or row.get("row_selector"),
        "top_k": as_int(row.get("top_k"), -1),
    }


def best_rank(rows: list[dict[str, Any]]) -> int:
    return max((as_int(row.get("direct_union_rank", row.get("rank"))) for row in rows), default=0)


def best_relation_count(rows: list[dict[str, Any]]) -> int:
    return max((as_int(row.get("direct_union_relation_count", row.get("relation_count"))) for row in rows), default=0)


def classify(exact_certs: list[dict[str, Any]], gate_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> str:
    if any(bool(cert.get("public_key_verified")) for cert in exact_certs):
        return "DIRECT_CERTIFICATE_PRESENT"
    relation_count = max(best_relation_count(gate_rows), best_relation_count(source_rows))
    rank = max(best_rank(gate_rows), best_rank(source_rows))
    derived_secret = any(
        row.get("direct_union_derived_secret") is not None or row.get("derived_secret") is not None
        for row in [*gate_rows, *source_rows]
    )
    if relation_count <= 0 or rank <= 0:
        return "FRESH_DIRECT_RELATION_REQUIRED_NO_RELATIONS"
    if not derived_secret:
        return "FRESH_DIRECT_RELATION_REQUIRED_RANK_DEFICIENT"
    return "FRESH_DIRECT_VERIFICATION_REQUIRED_SECRET_MISSING"


def build_case_audit(case: dict[str, Any], support_state_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    target = str(case.get("target") or "")
    transfer = as_int(case.get("transfer_index"), -1)
    observed = case.get("observed_support_scout") or {}
    scout_path = Path(str(observed.get("artifact") or ""))
    if not scout_path.exists():
        failures.append({"code": "support_scout_missing", "transfer_index": transfer, "path": str(scout_path)})
        scout = {}
    else:
        scout = load_json(scout_path)

    artifacts = scout.get("artifacts") or {}
    source_path = resolve_state_artifact(artifacts.get("source"), support_state_dir)
    product_path = resolve_state_artifact(artifacts.get("product"), support_state_dir)
    source_payload = load_json(source_path) if source_path and source_path.exists() else {}
    product_payload = load_json(product_path) if product_path and product_path.exists() else {}
    if not source_payload:
        failures.append({"code": "source_policy_missing", "transfer_index": transfer, "path": str(source_path)})
    if not product_payload:
        failures.append({"code": "shared_product_gate_missing", "transfer_index": transfer, "path": str(product_path)})

    key = case_key(case)
    source_rows = find_rows(source_payload, key)
    product_rows = find_rows(product_payload, key)
    same_transfer_source_rows = find_same_transfer_rows(source_payload, target, transfer)
    same_transfer_product_rows = find_same_transfer_rows(product_payload, target, transfer)
    cert_paths = certificate_paths_for_scout(scout_path, support_state_dir)
    certs, missing_certs = load_certificates(cert_paths)
    for path in missing_certs:
        failures.append({"code": "certificate_source_missing", "transfer_index": transfer, "path": path})
    exact_certs = [cert for cert in certs if cert_selected_key(cert) == key]
    positive_controls = [
        cert
        for cert in certs
        if bool(cert.get("public_key_verified"))
        and str((cert.get("selected") or {}).get("target") or "") == target
    ]

    if not source_rows:
        failures.append({"code": "exact_source_policy_row_missing", "transfer_index": transfer})
    if not product_rows:
        failures.append({"code": "exact_shared_product_gate_row_missing", "transfer_index": transfer})

    classification = classify(exact_certs, product_rows, source_rows)
    case_audit = {
        "accepted_relation_export": False,
        "backfill_row_check_hash": case.get("backfill_row_check_hash"),
        "backfill_row_check_hash_u64": as_int(case.get("backfill_row_check_hash_u64")) or digest_u64(case.get("backfill_row_check_hash")),
        "backfill_row_id": case.get("backfill_row_id"),
        "backfill_row_id_u64": as_int(case.get("backfill_row_id_u64")) or digest_u64(case.get("backfill_row_id")),
        "classification": classification,
        "classification_code": CLASS_CODES.get(classification, 0),
        "direct_certificate_count": len(exact_certs),
        "direct_public_key_verified": any(bool(cert.get("public_key_verified")) for cert in exact_certs)
        or any(bool(row.get("direct_union_public_key_verified", row.get("public_key_verified"))) for row in [*product_rows, *source_rows]),
        "exact_product_gate_rows": [compact_gate_row(row) for row in product_rows],
        "exact_source_policy_rows": [compact_source_row(row) for row in source_rows],
        "family_masks": case.get("family_masks") or [],
        "fresh_direct_verification_required": True,
        "materialization_direct_public_key_verified": bool(case.get("direct_public_key_verified")),
        "positive_control_certificates": [
            {
                "artifact": cert.get("artifact"),
                "derived_secret": cert.get("derived_secret"),
                "forms_count": as_int(cert.get("forms_count")),
                "public_key_verified": bool(cert.get("public_key_verified")),
                "rank": as_int(cert.get("rank")),
                "selector": (cert.get("selected") or {}).get("selector"),
                "top_k": as_int((cert.get("selected") or {}).get("top_k")),
                "transfer_index": as_int((cert.get("selected") or {}).get("transfer_index"), -1),
            }
            for cert in positive_controls
        ],
        "relation_derived_ecdlp": False,
        "same_transfer_best_product_rank": best_rank(same_transfer_product_rows),
        "same_transfer_best_product_relation_count": best_relation_count(same_transfer_product_rows),
        "same_transfer_best_source_rank": best_rank(same_transfer_source_rows),
        "same_transfer_best_source_relation_count": best_relation_count(same_transfer_source_rows),
        "source_artifacts": {
            "direct_certificates": [str(path) for path in cert_paths],
            "frontier_public_leaf_policy": str(source_path) if source_path else None,
            "shared_product_gate": str(product_path) if product_path else None,
            "support_scout": str(scout_path),
        },
        "target": target,
        "transfer_index": transfer,
        "worker_obligation": (
            "emit_new_direct_relations_for_rank"
            if classification == "FRESH_DIRECT_RELATION_REQUIRED_NO_RELATIONS"
            else "emit_additional_independent_direct_relation"
            if classification == "FRESH_DIRECT_RELATION_REQUIRED_RANK_DEFICIENT"
            else "verify_or_export_direct_certificate"
        ),
    }
    return case_audit, failures


def validate_materialization(materialization: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if materialization.get("claim_status") not in {
        "SELECTED13_FRESH_MATERIALIZATION_NEEDS_FRESH_DIRECT_VERIFICATION",
        "SELECTED13_FRESH_MATERIALIZATION_DIRECT_VERIFIED_READY",
    }:
        failures.append({"code": "materialization_claim_status_unexpected", "claim_status": materialization.get("claim_status")})
    if materialization.get("failures"):
        failures.append({"code": "materialization_has_failures", "failures": materialization.get("failures")})
    return failures


def claim_status(failures: list[dict[str, Any]], case_audits: list[dict[str, Any]]) -> str:
    if failures:
        return "SELECTED13_DIRECT_VERIFICATION_AUDIT_FAILED"
    if any(bool(case.get("direct_public_key_verified")) for case in case_audits):
        return "SELECTED13_DIRECT_VERIFICATION_FOUND_EXPORTABLE_CERTIFICATE"
    return "SELECTED13_DIRECT_VERIFICATION_REQUIRES_FRESH_RELATIONS"


def render_c_header(case_audits: list[dict[str, Any]]) -> str:
    rows = []
    for case in case_audits:
        rows.append(
            "  {"
            f"{as_int(case.get('transfer_index'))}ULL, "
            f"{as_int(case.get('backfill_row_id_u64'))}ULL, "
            f"{as_int(case.get('backfill_row_check_hash_u64'))}ULL, "
            f"{as_int(case.get('classification_code'))}ULL, "
            f"{1 if case.get('direct_public_key_verified') else 0}ULL, "
            f"{as_int(case.get('direct_certificate_count'))}ULL, "
            f"{as_int(case.get('same_transfer_best_product_rank'))}ULL, "
            f"{as_int(case.get('same_transfer_best_product_relation_count'))}ULL, "
            f"{as_int(case.get('same_transfer_best_source_rank'))}ULL, "
            f"{as_int(case.get('same_transfer_best_source_relation_count'))}ULL, "
            f"{1 if case.get('fresh_direct_verification_required') else 0}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_DIRECT_VERIFICATION_AUDIT_H
#define LOW_TERM_TOTAL2_SELECTED13_DIRECT_VERIFICATION_AUDIT_H

#include <stdint.h>

#define SELECTED13_DIRECT_VERIFICATION_AUDIT_TARGET_COUNT {len(case_audits)}
#define SELECTED13_DIRECT_VERIFICATION_AUDIT_VERIFIED_COUNT {sum(1 for case in case_audits if case.get('direct_public_key_verified'))}
#define SELECTED13_DIRECT_VERIFICATION_AUDIT_RELATION_EXPORT_COUNT 0
#define SELECTED13_DIRECT_VERIFICATION_AUDIT_RELATION_DERIVED_ECDLP 0

#define SELECTED13_DIRECT_CLASS_CERT_PRESENT 1ULL
#define SELECTED13_DIRECT_CLASS_NO_RELATIONS 2ULL
#define SELECTED13_DIRECT_CLASS_RANK_DEFICIENT 3ULL
#define SELECTED13_DIRECT_CLASS_SECRET_MISSING 4ULL
#define SELECTED13_DIRECT_CLASS_EVIDENCE_MISSING 5ULL

typedef struct {{
  uint64_t transfer_index;
  uint64_t backfill_row_id_u64;
  uint64_t backfill_row_check_hash_u64;
  uint64_t classification_code;
  uint64_t direct_public_key_verified;
  uint64_t direct_certificate_count;
  uint64_t best_product_rank;
  uint64_t best_product_relation_count;
  uint64_t best_source_rank;
  uint64_t best_source_relation_count;
  uint64_t fresh_direct_verification_required;
}} selected13_direct_verification_audit_target_t;

static const selected13_direct_verification_audit_target_t SELECTED13_DIRECT_VERIFICATION_AUDIT_TARGETS[] = {{
{chr(10).join(rows)}
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
      sizeof(SELECTED13_DIRECT_VERIFICATION_AUDIT_TARGETS) / sizeof(SELECTED13_DIRECT_VERIFICATION_AUDIT_TARGETS[0]);
  uint64_t verified_count = 0;
  uint64_t fresh_required_count = 0;
  uint64_t no_relation_count = 0;
  uint64_t rank_deficient_count = 0;

  if (target_count != SELECTED13_DIRECT_VERIFICATION_AUDIT_TARGET_COUNT) failure_count++;
  if (SELECTED13_DIRECT_VERIFICATION_AUDIT_RELATION_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_DIRECT_VERIFICATION_AUDIT_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;
  if (target_count == 0ULL) failure_count++;

  for (size_t i = 0; i < target_count; i++) {{
    const selected13_direct_verification_audit_target_t *target = &SELECTED13_DIRECT_VERIFICATION_AUDIT_TARGETS[i];
    verified_count += target->direct_public_key_verified;
    fresh_required_count += target->fresh_direct_verification_required;
    if (target->classification_code == SELECTED13_DIRECT_CLASS_NO_RELATIONS) no_relation_count++;
    if (target->classification_code == SELECTED13_DIRECT_CLASS_RANK_DEFICIENT) rank_deficient_count++;
    if (target->backfill_row_id_u64 == 0ULL) failure_count++;
    if (target->backfill_row_check_hash_u64 == 0ULL) failure_count++;
    if (target->classification_code == 0ULL) failure_count++;
  }}

  if (verified_count != SELECTED13_DIRECT_VERIFICATION_AUDIT_VERIFIED_COUNT) failure_count++;

  printf("selected13_direct_verification_audit_preflight targets=%llu verified=%llu fresh_required=%llu no_relation=%llu rank_deficient=%llu failures=%llu\\n",
         (unsigned long long)target_count,
         (unsigned long long)verified_count,
         (unsigned long long)fresh_required_count,
         (unsigned long long)no_relation_count,
         (unsigned long long)rank_deficient_count,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_direct_audit_preflight_") as tmp:
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
    materialization_path = Path(args.materialization)
    materialization = load_json(materialization_path)
    failures = validate_materialization(materialization)
    support_state_dir = Path(args.support_state_dir)
    case_audits = []
    for case in materialization.get("worker_cases") or []:
        if not isinstance(case, dict):
            continue
        audit, audit_failures = build_case_audit(case, support_state_dir)
        case_audits.append(audit)
        failures.extend(audit_failures)

    classification_counts = Counter(str(case.get("classification")) for case in case_audits)
    summary = {
        "accepted_relation_export_count": 0,
        "classification_counts": dict(sorted(classification_counts.items())),
        "direct_certificate_target_count": sum(1 for case in case_audits if as_int(case.get("direct_certificate_count")) > 0),
        "direct_public_key_verified_count": sum(1 for case in case_audits if case.get("direct_public_key_verified")),
        "failure_count": len(failures),
        "fresh_direct_verification_required_count": sum(1 for case in case_audits if case.get("fresh_direct_verification_required")),
        "positive_control_certificate_count": sum(len(case.get("positive_control_certificates") or []) for case in case_audits),
        "pollard_rho_speedup_claimed": False,
        "rank_deficient_target_count": classification_counts.get("FRESH_DIRECT_RELATION_REQUIRED_RANK_DEFICIENT", 0),
        "relation_derived_ecdlp": False,
        "target_count": len(case_audits),
        "target_without_relation_count": classification_counts.get("FRESH_DIRECT_RELATION_REQUIRED_NO_RELATIONS", 0),
        "verified": not failures,
        "worker_interpretation": (
            "The materialized selected13 targets do not have direct certificates. "
            "9981 carries rank-deficient verifier evidence; 9943 has no exact "
            "direct relation for the materialized row. Fresh FFE relation emission "
            "is still required before any export or speedup claim."
        ),
    }
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, case_audits),
        "parameters": {
            "materialization": str(materialization_path),
            "support_state_dir": str(support_state_dir),
        },
        "summary": summary,
        "target_audits": case_audits,
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "materialized_below_rho_labels_are_diagnostic_only": True,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization", default=str(DEFAULT_MATERIALIZATION))
    parser.add_argument("--support-state-dir", default=str(DEFAULT_SUPPORT_STATE_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT), type=Path)
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["target_audits"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
        payload["claim_status"] = claim_status(payload["failures"], payload["target_audits"])
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(args.out), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
