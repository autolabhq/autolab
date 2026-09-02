#!/usr/bin/env python3
"""Audit materialization source hints against exact direct-verifier evidence.

The source-hint adapter can see exact support-scout rows, but a scout boolean is
not itself a relation export.  This audit follows each exact scout row back to
its upstream source policy, shared-product gate, and direct equation-certificate
files.  Rows with an exact direct-union derived secret are separated from rows
that still need fresh FFE/summation-polynomial direct-rank emission.

This is an evidence audit.  It does not synthesize new relations or claim a
general Pollard-rho speedup.
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


SCHEMA = "ecdlp.low_term_total2_selected13_materialization_direct_evidence_audit.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SUPPORT_STATE_DIR = Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state")
DEFAULT_SOURCE_HINT = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_source_hint_adapter_111_full147_probe.json"
DEFAULT_WORKORDER = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_miss_workorder_111_full147_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_direct_evidence_audit_111_full147_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_direct_evidence_audit_111_full147_probe.h"

CLASS_CODES = {
    "DIRECT_UNION_DERIVED_SECRET_PRESENT": 1,
    "DIRECT_CERTIFICATE_DERIVED_SECRET_PRESENT": 2,
    "DIRECT_VERIFIER_SECRET_MISSING": 3,
    "FRESH_DIRECT_RELATION_REQUIRED_RANK_DEFICIENT": 4,
    "FRESH_DIRECT_RELATION_REQUIRED_NO_RELATIONS": 5,
    "DIRECT_VERIFIER_EVIDENCE_MISSING": 6,
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
    return int(hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:16], 16)


def row_key_tuple(raw: Any) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in raw or []))


def support_tuple(raw: Any) -> tuple[int, ...]:
    return tuple(sorted(as_int(item) for item in raw or []))


def support_mask(raw: Any) -> int:
    mask = 0
    for term in raw or []:
        value = as_int(term, -1)
        if value >= 0:
            mask |= 1 << value
    return mask


def exact_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("target") or ""),
        as_int(row.get("transfer_index"), -1),
        str(row.get("selector") or row.get("row_selector") or ""),
        as_int(row.get("top_k"), -1),
        row_key_tuple(row.get("row_keys")),
    )


def selected_from_cert(cert: dict[str, Any]) -> dict[str, Any]:
    selected = dict(cert.get("selected") or {})
    if not selected.get("selected_term_support") and cert.get("selected_term_support"):
        selected["selected_term_support"] = cert.get("selected_term_support")
    return selected


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


def exact_rows(payload: dict[str, Any], target_row: dict[str, Any]) -> list[dict[str, Any]]:
    key = exact_key(target_row)
    target_support = support_tuple(target_row.get("selected_term_support"))
    rows = []
    for item in all_objects(payload):
        if "transfer_index" not in item or "row_keys" not in item:
            continue
        if exact_key(item) != key:
            continue
        item_support = support_tuple(item.get("selected_term_support"))
        if item_support and item_support != target_support:
            continue
        rows.append(item)
    return rows


def range_from_path(path: Path) -> str | None:
    match = re.search(r"_(\d{4,5})_(\d{4,5})(?:_|_probe)", path.name)
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


def certificate_paths_for_scout(scout_path: Path, support_state_dir: Path) -> list[Path]:
    range_label = range_from_path(scout_path)
    if not range_label:
        return []
    return sorted(
        support_state_dir.glob(f"low_term_total2_direct_relation_equation_certificate_*_{range_label}_probe.json")
    )


def exact_certs(paths: list[Path], target_row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    key = exact_key(target_row)
    support = support_tuple(target_row.get("selected_term_support"))
    for path in paths:
        if not path.exists():
            continue
        payload = load_json(path)
        for offset, cert in enumerate(payload.get("certificates") or []):
            selected = selected_from_cert(cert)
            if exact_key(selected) != key:
                continue
            selected_support = support_tuple(selected.get("selected_term_support"))
            if selected_support and selected_support != support:
                continue
            item = dict(cert)
            item["artifact"] = str(path)
            item["artifact_offset"] = offset
            out.append(item)
    return out


def workorder_rows_by_request(workorder: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for item in workorder.get("work_items") or []:
        for row in item.get("candidate_rows") or []:
            if isinstance(row, dict) and row.get("row_request_id"):
                out[str(row.get("row_request_id"))] = row
    return out


def best_int(rows: list[dict[str, Any]], *names: str) -> int:
    return max(
        (
            as_int(row.get(name))
            for row in rows
            for name in names
            if row.get(name) is not None
        ),
        default=0,
    )


def first_present(rows: list[dict[str, Any]], *names: str) -> Any:
    for row in rows:
        for name in names:
            if row.get(name) is not None:
                return row.get(name)
    return None


def evidence_class(product_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], certs: list[dict[str, Any]]) -> str:
    if any(row.get("direct_union_public_key_verified") and row.get("direct_union_derived_secret") is not None for row in product_rows):
        return "DIRECT_UNION_DERIVED_SECRET_PRESENT"
    if any(cert.get("public_key_verified") and cert.get("derived_secret") is not None for cert in certs):
        return "DIRECT_CERTIFICATE_DERIVED_SECRET_PRESENT"
    if any(row.get("public_key_verified") or row.get("direct_union_public_key_verified") for row in [*product_rows, *source_rows]):
        return "DIRECT_VERIFIER_SECRET_MISSING"
    relation_count = max(
        best_int(product_rows, "direct_union_relation_count", "relation_count"),
        best_int(source_rows, "relation_count", "direct_union_relation_count"),
    )
    rank = max(
        best_int(product_rows, "direct_union_rank", "rank"),
        best_int(source_rows, "rank", "direct_union_rank"),
    )
    if relation_count > 0 or rank > 0:
        return "FRESH_DIRECT_RELATION_REQUIRED_RANK_DEFICIENT"
    if product_rows or source_rows or certs:
        return "FRESH_DIRECT_RELATION_REQUIRED_NO_RELATIONS"
    return "DIRECT_VERIFIER_EVIDENCE_MISSING"


def build_row_audit(
    row: dict[str, Any],
    workorder_row: dict[str, Any],
    support_state_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    scout_path = Path(str(row.get("source_artifact") or ""))
    if not scout_path.exists():
        failures.append({"code": "support_scout_missing", "row_request_id": row.get("row_request_id"), "path": str(scout_path)})
        scout = {}
    else:
        scout = load_json(scout_path)
    artifacts = scout.get("artifacts") or {}
    source_path = resolve_state_artifact(artifacts.get("source"), support_state_dir)
    product_path = resolve_state_artifact(artifacts.get("product"), support_state_dir)
    source_payload = load_json(source_path) if source_path and source_path.exists() else {}
    product_payload = load_json(product_path) if product_path and product_path.exists() else {}
    if not source_payload:
        failures.append({"code": "source_policy_missing", "row_request_id": row.get("row_request_id"), "path": str(source_path)})
    if not product_payload:
        failures.append({"code": "shared_product_gate_missing", "row_request_id": row.get("row_request_id"), "path": str(product_path)})

    target_row = {
        "row_keys": workorder_row.get("row_keys"),
        "selected_term_support": workorder_row.get("selected_term_support"),
        "selector": workorder_row.get("selector"),
        "target": workorder_row.get("target"),
        "top_k": workorder_row.get("top_k"),
        "transfer_index": workorder_row.get("transfer_index"),
    }
    source_rows = exact_rows(source_payload, target_row)
    product_rows = exact_rows(product_payload, target_row)
    cert_paths = certificate_paths_for_scout(scout_path, support_state_dir)
    certs = exact_certs(cert_paths, target_row)
    classification = evidence_class(product_rows, source_rows, certs)
    derived_secret = first_present(product_rows, "direct_union_derived_secret", "derived_secret")
    if derived_secret is None:
        derived_secret = first_present(certs, "derived_secret")
    public_key_verified = classification in {
        "DIRECT_UNION_DERIVED_SECRET_PRESENT",
        "DIRECT_CERTIFICATE_DERIVED_SECRET_PRESENT",
    }
    direct_ops = first_present(product_rows, "direct_verifier_replay_ops_over_rho", "ops_over_rho")
    if direct_ops is None:
        direct_ops = row.get("direct_ops_over_rho")
    return (
        {
            "accepted_relation_export": public_key_verified,
            "bridge_worker_required": not public_key_verified,
            "classification": classification,
            "classification_code": CLASS_CODES.get(classification, 0),
            "derived_secret": derived_secret,
            "direct_certificate_count": len(certs),
            "direct_ops_over_rho": as_float(direct_ops),
            "direct_public_key_verified": public_key_verified,
            "exact_product_gate_row_count": len(product_rows),
            "exact_source_policy_row_count": len(source_rows),
            "global_row_index": as_int(row.get("global_row_index"), -1),
            "is_best_manifest_row": bool(row.get("is_best_manifest_row")),
            "packet_index": as_int(row.get("packet_index"), -1),
            "product_rank": best_int(product_rows, "direct_union_rank", "rank"),
            "product_relation_count": best_int(product_rows, "direct_union_relation_count", "relation_count"),
            "relation_derived_ecdlp": public_key_verified and derived_secret is not None,
            "row_request_id": row.get("row_request_id"),
            "row_request_id_u64": as_int(row.get("row_request_id_u64")),
            "selected_support_mask": support_mask(workorder_row.get("selected_term_support")),
            "source_artifacts": {
                "direct_certificates": [str(path) for path in cert_paths],
                "frontier_public_leaf_policy": str(source_path) if source_path else None,
                "shared_product_gate": str(product_path) if product_path else None,
                "support_scout": str(scout_path),
            },
            "source_rank": best_int(source_rows, "rank", "direct_union_rank"),
            "source_relation_count": best_int(source_rows, "relation_count", "direct_union_relation_count"),
            "target": workorder_row.get("target"),
            "transfer_index": as_int(row.get("transfer_index"), -1),
            "worker_obligation": "promote_direct_union_derived_row"
            if public_key_verified
            else "fresh_ffe_summation_polynomial_direct_rank_bridge_extension",
        },
        failures,
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    source_hint_path = Path(args.source_hint)
    workorder_path = Path(args.workorder)
    source_hint = load_json(source_hint_path)
    workorder = load_json(workorder_path)
    support_state_dir = Path(args.support_state_dir)
    failures: list[dict[str, Any]] = []
    if source_hint.get("claim_status") not in {
        "SELECTED13_MATERIALIZATION_SOURCE_HINTS_WITH_EXACT_DIRECT_PROMOTION",
        "SELECTED13_MATERIALIZATION_SOURCE_HINTS_READY",
    }:
        failures.append({"code": "source_hint_claim_status_unexpected", "claim_status": source_hint.get("claim_status")})
    if source_hint.get("failures"):
        failures.append({"code": "source_hint_has_failures", "failures": source_hint.get("failures")})
    if workorder.get("claim_status") != "SELECTED13_MATERIALIZATION_MISS_WORKORDER_READY":
        failures.append({"code": "workorder_not_ready", "claim_status": workorder.get("claim_status")})

    workorder_index = workorder_rows_by_request(workorder)
    row_audits = []
    for row in source_hint.get("rows") or []:
        if not isinstance(row, dict):
            continue
        request_id = str(row.get("row_request_id") or "")
        workorder_row = workorder_index.get(request_id)
        if workorder_row is None:
            failures.append({"code": "workorder_row_missing", "row_request_id": request_id})
            continue
        audit, audit_failures = build_row_audit(row, workorder_row, support_state_dir)
        row_audits.append(audit)
        failures.extend(audit_failures)

    class_counts = Counter(str(row.get("classification")) for row in row_audits)
    derived_rows = [row for row in row_audits if row.get("relation_derived_ecdlp")]
    derived_transfers = sorted({as_int(row.get("transfer_index"), -1) for row in derived_rows})
    summary = {
        "accepted_relation_export_count": len(derived_rows),
        "bridge_worker_required_row_count": sum(1 for row in row_audits if row.get("bridge_worker_required")),
        "classification_counts": dict(sorted(class_counts.items())),
        "derived_secret_values": sorted({as_int(row.get("derived_secret")) for row in derived_rows}),
        "direct_union_derived_row_count": class_counts.get("DIRECT_UNION_DERIVED_SECRET_PRESENT", 0),
        "direct_union_derived_transfer_count": len(derived_transfers),
        "direct_union_derived_transfers": derived_transfers,
        "failure_count": len(failures),
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": bool(derived_rows),
        "relation_derived_row_count": len(derived_rows),
        "relation_derived_transfer_count": len(derived_transfers),
        "row_count": len(row_audits),
        "verified": not failures,
        "worker_interpretation": (
            "Exact direct-union evidence promotes rows whose upstream product gate already derived the ECDLP secret; "
            "remaining rows still require fresh FFE/summation-polynomial direct-rank emission."
        ),
    }
    claim = (
        "SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_HAS_DERIVED_ROWS"
        if not failures and derived_rows
        else "SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_REQUIRES_FRESH_ROWS"
        if not failures
        else "SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_AUDIT_FAILED"
    )
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim,
        "parameters": {
            "source_hint": str(source_hint_path),
            "support_state_dir": str(support_state_dir),
            "workorder": str(workorder_path),
        },
        "source_summary": source_hint.get("summary"),
        "summary": summary,
        "row_audits": row_audits,
        "failures": failures,
        "honesty_boundary": {
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": bool(derived_rows),
            "source_evidence_audit_only": True,
        },
    }


def render_c_header(rows: list[dict[str, Any]]) -> str:
    row_lines = []
    for row in rows:
        row_lines.append(
            "  {"
            f"{as_int(row.get('packet_index'))}ULL, "
            f"{as_int(row.get('global_row_index'))}ULL, "
            f"{as_int(row.get('transfer_index'))}ULL, "
            f"{as_int(row.get('row_request_id_u64'))}ULL, "
            f"{as_int(row.get('selected_support_mask'))}ULL, "
            f"{as_int(row.get('classification_code'))}ULL, "
            f"{1 if row.get('relation_derived_ecdlp') else 0}ULL, "
            f"{1 if row.get('bridge_worker_required') else 0}ULL, "
            f"{1 if row.get('is_best_manifest_row') else 0}ULL, "
            f"{as_int(row.get('derived_secret'))}ULL, "
            f"{as_int(row.get('product_rank'))}ULL, "
            f"{as_int(row.get('product_relation_count'))}ULL, "
            f"{as_int(row.get('source_rank'))}ULL, "
            f"{as_int(row.get('source_relation_count'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_AUDIT_H
#define LOW_TERM_TOTAL2_SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_AUDIT_H

#include <stdint.h>

#define SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_ROW_COUNT {len(rows)}
#define SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_DERIVED_ROW_COUNT {sum(1 for row in rows if row.get('relation_derived_ecdlp'))}
#define SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_BRIDGE_REQUIRED_ROW_COUNT {sum(1 for row in rows if row.get('bridge_worker_required'))}

typedef struct {{
  uint64_t packet_index;
  uint64_t global_row_index;
  uint64_t transfer_index;
  uint64_t row_request_id_u64;
  uint64_t selected_support_mask;
  uint64_t classification_code;
  uint64_t relation_derived_ecdlp;
  uint64_t bridge_worker_required;
  uint64_t is_best_manifest_row;
  uint64_t derived_secret;
  uint64_t product_rank;
  uint64_t product_relation_count;
  uint64_t source_rank;
  uint64_t source_relation_count;
}} selected13_materialization_direct_evidence_row_t;

static const selected13_materialization_direct_evidence_row_t SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_ROWS[] = {{
{chr(10).join(row_lines)}
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
  uint64_t row_count = sizeof(SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_ROWS) / sizeof(SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_ROWS[0]);
  uint64_t derived_rows = 0;
  uint64_t bridge_rows = 0;
  uint64_t derived_best_rows = 0;
  uint64_t last_derived_transfer = 0;
  uint64_t derived_transfers = 0;

  if (row_count != SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_ROW_COUNT) failure_count++;
  if (row_count == 0ULL) failure_count++;
  for (size_t i = 0; i < row_count; i++) {{
    const selected13_materialization_direct_evidence_row_t *row = &SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_ROWS[i];
    if (row->row_request_id_u64 == 0ULL || row->selected_support_mask == 0ULL || row->classification_code == 0ULL) {{
      failure_count++;
    }}
    if (row->relation_derived_ecdlp && row->bridge_worker_required) failure_count++;
    if (!row->relation_derived_ecdlp && !row->bridge_worker_required) failure_count++;
    if (row->relation_derived_ecdlp) {{
      derived_rows++;
      if (row->derived_secret == 0ULL || row->product_rank == 0ULL || row->product_relation_count == 0ULL) {{
        failure_count++;
      }}
      if (row->is_best_manifest_row) derived_best_rows++;
      if (row->transfer_index != last_derived_transfer) derived_transfers++;
      last_derived_transfer = row->transfer_index;
    }}
    if (row->bridge_worker_required) bridge_rows++;
  }}
  if (derived_rows != SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_DERIVED_ROW_COUNT) failure_count++;
  if (bridge_rows != SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_BRIDGE_REQUIRED_ROW_COUNT) failure_count++;
  if (derived_rows + bridge_rows != row_count) failure_count++;

  printf("selected13_materialization_direct_evidence_preflight rows=%llu derived_rows=%llu derived_transfers=%llu derived_best_rows=%llu needs_bridge=%llu failures=%llu\\n",
         (unsigned long long)row_count,
         (unsigned long long)derived_rows,
         (unsigned long long)derived_transfers,
         (unsigned long long)derived_best_rows,
         (unsigned long long)bridge_rows,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_mat_direct_evidence_") as tmp:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-hint", type=Path, default=DEFAULT_SOURCE_HINT)
    parser.add_argument("--workorder", type=Path, default=DEFAULT_WORKORDER)
    parser.add_argument("--support-state-dir", type=Path, default=DEFAULT_SUPPORT_STATE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["row_audits"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
        payload["claim_status"] = "SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_AUDIT_FAILED"
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(args.out), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
