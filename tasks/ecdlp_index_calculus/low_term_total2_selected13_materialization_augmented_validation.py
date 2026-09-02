#!/usr/bin/env python3
"""Overlay materialization direct evidence onto selected13 full-queue validation.

The frozen selected13 full-queue public policy currently validates 111/147
transfers.  A separate materialization evidence audit found direct-union
relation-derived rows for transfer 10595, one of the original materialization
misses.  This script produces a native-checkable augmented validation artifact
that promotes only transfers with audited direct evidence and keeps provenance
separate from the frozen one-leaf replay.

It does not rerun the policy, synthesize relations, or claim a general ECDLP
speedup.  It is an evidence overlay over same-target-family validation state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_selected13_materialization_augmented_validation.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SEED_VALIDATION = (
    DEFAULT_STATE_DIR
    / "low_term_total2_selected13_second_stage_public_root2_span_le6_active111_then_span_le6_position_validation_10056_10607_full147_probe.json"
)
DEFAULT_DIRECT_EVIDENCE = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_direct_evidence_audit_111_full147_probe.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR
    / "low_term_total2_selected13_second_stage_public_root2_span_le6_active111_plus_materialization_direct_validation_10056_10607_full147_probe.json"
)
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR
    / "low_term_total2_selected13_second_stage_public_root2_span_le6_active111_plus_materialization_direct_validation_10056_10607_full147_probe.h"
)

MATERIALIZATION_STATUS = "ACCEPTED_MATERIALIZATION_DIRECT_UNION_BELOW_RHO"
MATERIALIZATION_STATUS_CODE = 31


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


def canonical_json(raw: Any) -> str:
    return json.dumps(raw, sort_keys=True, separators=(",", ":"))


def digest_u64(raw: Any) -> int:
    return int(hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()[:16], 16)


def evidence_by_transfer(direct_evidence: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for row in direct_evidence.get("row_audits") or []:
        if not isinstance(row, dict) or not row.get("relation_derived_ecdlp"):
            continue
        out.setdefault(as_int(row.get("transfer_index"), -1), []).append(row)
    for rows in out.values():
        rows.sort(
            key=lambda row: (
                0 if row.get("is_best_manifest_row") else 1,
                as_float(row.get("direct_ops_over_rho")) if as_float(row.get("direct_ops_over_rho")) is not None else 9.0,
                as_int(row.get("global_row_index"), 9999),
            )
        )
    return out


def choose_evidence_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows else {}


def overlay_record(record: dict[str, Any], row: dict[str, Any], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct_ops = as_float(row.get("direct_ops_over_rho"))
    updated = deepcopy(record)
    original = {
        "accepted_relation_export": bool(record.get("accepted_relation_export")),
        "below_rho": bool(record.get("below_rho")),
        "derived_secret": record.get("derived_secret"),
        "ops_over_rho": record.get("ops_over_rho"),
        "public_key_verified": bool(record.get("public_key_verified")),
        "rank": as_int(record.get("rank")),
        "relation_count": as_int(record.get("relation_count")),
        "relation_derived_ecdlp": bool(record.get("relation_derived_ecdlp")),
        "status": record.get("status"),
        "status_code": as_int(record.get("status_code")),
    }
    materialized_rows = [
        {
            "classification": item.get("classification"),
            "derived_secret": item.get("derived_secret"),
            "direct_ops_over_rho": item.get("direct_ops_over_rho"),
            "global_row_index": item.get("global_row_index"),
            "is_best_manifest_row": bool(item.get("is_best_manifest_row")),
            "product_rank": item.get("product_rank"),
            "product_relation_count": item.get("product_relation_count"),
            "row_request_id": item.get("row_request_id"),
            "source_rank": item.get("source_rank"),
            "source_relation_count": item.get("source_relation_count"),
        }
        for item in all_rows
    ]
    updated.update(
        {
            "accepted_relation_export": True,
            "below_rho": bool(direct_ops is not None and direct_ops < 1.0),
            "candidate_class": "selected13_materialization_direct_union_overlay",
            "candidate_class_code": 31,
            "candidate_id": f"mat_direct_union_{digest_u64([record.get('candidate_id'), row.get('row_request_id')]):016x}",
            "candidate_id_u64": digest_u64(["materialization_direct_union", record.get("candidate_id"), row.get("row_request_id")]),
            "derived_secret": as_int(row.get("derived_secret")),
            "direct_replay_original": original,
            "materialization_direct_evidence": {
                "all_relation_derived_rows": materialized_rows,
                "chosen_row_policy": "best_manifest_row_then_lowest_ops",
                "direct_ops_over_rho": direct_ops,
                "is_best_manifest_row": bool(row.get("is_best_manifest_row")),
                "product_rank": as_int(row.get("product_rank")),
                "product_relation_count": as_int(row.get("product_relation_count")),
                "relation_derived_row_count": len(all_rows),
                "row_request_id": row.get("row_request_id"),
                "row_request_id_u64": as_int(row.get("row_request_id_u64")),
                "source_artifacts": row.get("source_artifacts"),
                "source_rank": as_int(row.get("source_rank")),
                "source_relation_count": as_int(row.get("source_relation_count")),
            },
            "ops_over_rho": round(direct_ops, 8) if direct_ops is not None else record.get("ops_over_rho"),
            "public_key_verified": True,
            "rank": as_int(row.get("product_rank")),
            "relation_count": as_int(row.get("product_relation_count")),
            "relation_derived_ecdlp": True,
            "second_stage_selector_id": f"{record.get('second_stage_selector_id')}_plus_materialization_direct",
            "status": MATERIALIZATION_STATUS,
            "status_code": MATERIALIZATION_STATUS_CODE,
        }
    )
    return updated


def augment_records(seed: dict[str, Any], direct_evidence: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    overlays: list[dict[str, Any]] = []
    evidence_index = evidence_by_transfer(direct_evidence)
    records = [deepcopy(row) for row in seed.get("direct_replay_records") or [] if isinstance(row, dict)]
    if not records:
        failures.append({"code": "seed_validation_records_missing"})
        return records, overlays, failures
    record_by_transfer = {as_int(row.get("transfer_index"), -1): row for row in records}
    for transfer, rows in sorted(evidence_index.items()):
        record = record_by_transfer.get(transfer)
        if record is None:
            failures.append({"code": "evidence_transfer_missing_from_seed_validation", "transfer_index": transfer})
            continue
        if record.get("relation_derived_ecdlp") and record.get("below_rho"):
            failures.append({"code": "evidence_transfer_already_accepted_in_seed", "transfer_index": transfer})
            continue
        chosen = choose_evidence_row(rows)
        if as_float(chosen.get("direct_ops_over_rho")) is None or as_float(chosen.get("direct_ops_over_rho")) >= 1.0:
            failures.append({"code": "evidence_row_not_below_rho", "transfer_index": transfer})
            continue
        updated = overlay_record(record, chosen, rows)
        record_by_transfer[transfer] = updated
        overlays.append(
            {
                "chosen_row_is_best_manifest_row": bool(chosen.get("is_best_manifest_row")),
                "chosen_row_request_id": chosen.get("row_request_id"),
                "derived_secret": as_int(chosen.get("derived_secret")),
                "direct_ops_over_rho": as_float(chosen.get("direct_ops_over_rho")),
                "evidence_row_count": len(rows),
                "original_status": record.get("status"),
                "transfer_index": transfer,
            }
        )
    augmented = [record_by_transfer[as_int(row.get("transfer_index"), -1)] for row in records]
    return augmented, overlays, failures


def best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row
        for row in records
        if row.get("accepted_relation_export") and row.get("below_rho")
    ]
    candidates.sort(
        key=lambda row: (
            as_float(row.get("ops_over_rho")) if as_float(row.get("ops_over_rho")) is not None else 9.0,
            as_int(row.get("transfer_index")),
        )
    )
    return candidates[0] if candidates else {}


def summarize(
    records: list[dict[str, Any]],
    overlays: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    seed_summary: dict[str, Any],
) -> dict[str, Any]:
    accepted = [row for row in records if row.get("accepted_relation_export")]
    below = [row for row in accepted if row.get("below_rho")]
    statuses = Counter(str(row.get("status")) for row in records)
    selected_transfers = sorted(as_int(row.get("transfer_index")) for row in records)
    accepted_transfers = sorted(as_int(row.get("transfer_index")) for row in below)
    missing = [
        as_int(row.get("transfer_index"))
        for row in records
        if not (row.get("accepted_relation_export") and row.get("below_rho"))
    ]
    best = best_record(records)
    overlay_transfers = sorted(as_int(item.get("transfer_index")) for item in overlays)
    return {
        "accepted_relation_export_count": len(accepted),
        "augmented_from_seed": as_int(seed_summary.get("accepted_relation_export_count")),
        "below_rho_accepted_relation_export_count": len(below),
        "below_rho_accepted_relation_export_transfers": accepted_transfers,
        "best_candidate_id": best.get("candidate_id"),
        "best_derived_secret": best.get("derived_secret"),
        "best_ops_over_rho": best.get("ops_over_rho"),
        "best_transfer_index": best.get("transfer_index"),
        "failure_count": len(failures),
        "general_ecdlp_algorithm_claimed": False,
        "materialization_overlay_count": len(overlays),
        "materialization_overlay_transfers": overlay_transfers,
        "missing_below_rho_transfers": sorted(missing),
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": bool(below),
        "selected_validation_transfer_count": len(records),
        "selected_validation_transfers": selected_transfers,
        "seed_accepted_relation_export_count": as_int(seed_summary.get("accepted_relation_export_count")),
        "seed_below_rho_accepted_relation_export_count": as_int(seed_summary.get("below_rho_accepted_relation_export_count")),
        "status_counts": dict(sorted(statuses.items())),
        "training_overlap_count": as_int(seed_summary.get("training_overlap_count")),
        "training_overlap_transfers": seed_summary.get("training_overlap_transfers") or [],
        "verified": not failures,
        "worker_interpretation": (
            "This is the frozen selected13 full-queue validation augmented only with audited "
            "materialization direct-union evidence.  It is not a fresh policy rerun and not "
            "a general ECDLP speedup claim."
        ),
    }


def claim_status(failures: list[dict[str, Any]], overlays: list[dict[str, Any]]) -> str:
    if failures:
        return "SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_FAILED"
    if overlays:
        return "SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_BELOW_RHO_EXPORT"
    return "SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_NO_NEW_EXPORT"


def render_c_header(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = []
    for record in records:
        ops_scaled = 0
        ops_over_rho = as_float(record.get("ops_over_rho"))
        if ops_over_rho is not None:
            ops_scaled = int(round(ops_over_rho * 1_000_000))
        rows.append(
            "  {"
            f"{as_int(record.get('candidate_index'))}ULL, "
            f"{as_int(record.get('candidate_id_u64'))}ULL, "
            f"{as_int(record.get('transfer_index'))}ULL, "
            f"{as_int(record.get('manifest_queue_position'))}ULL, "
            f"{1 if record.get('below_rho') else 0}ULL, "
            f"{1 if record.get('public_key_verified') else 0}ULL, "
            f"{1 if record.get('relation_derived_ecdlp') else 0}ULL, "
            f"{1 if record.get('materialization_direct_evidence') else 0}ULL, "
            f"{as_int(record.get('rank'))}ULL, "
            f"{as_int(record.get('relation_count'))}ULL, "
            f"{as_int(record.get('derived_secret'))}ULL, "
            f"{ops_scaled}ULL, "
            f"{as_int(record.get('status_code'))}ULL"
            "},"
        )
    accepted_count = sum(1 for record in records if record.get("relation_derived_ecdlp"))
    accepted_below_count = sum(1 for record in records if record.get("relation_derived_ecdlp") and record.get("below_rho"))
    overlay_count = sum(1 for record in records if record.get("materialization_direct_evidence"))
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_H
#define LOW_TERM_TOTAL2_SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_H

#include <stdint.h>

#define SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_RECORD_COUNT {len(records)}
#define SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_SELECTED_TRANSFER_COUNT {as_int(summary.get("selected_validation_transfer_count"))}
#define SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_TRAINING_OVERLAP_COUNT {as_int(summary.get("training_overlap_count"))}
#define SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_ACCEPTED_COUNT {accepted_count}
#define SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_ACCEPTED_BELOW_COUNT {accepted_below_count}
#define SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_OVERLAY_COUNT {overlay_count}

typedef struct {{
  uint64_t candidate_index;
  uint64_t candidate_id_u64;
  uint64_t transfer_index;
  uint64_t manifest_queue_position;
  uint64_t below_rho;
  uint64_t public_key_verified;
  uint64_t relation_derived_ecdlp;
  uint64_t materialization_overlay;
  uint64_t rank;
  uint64_t relation_count;
  uint64_t derived_secret;
  uint64_t ops_over_rho_scaled_1e6;
  uint64_t status_code;
}} selected13_materialization_augmented_validation_record_t;

static const selected13_materialization_augmented_validation_record_t SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_RECORDS[] = {{
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
  uint64_t record_count =
      sizeof(SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_RECORDS) /
      sizeof(SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_RECORDS[0]);
  uint64_t accepted = 0;
  uint64_t accepted_below = 0;
  uint64_t overlay = 0;

  if (record_count != SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_RECORD_COUNT) failure_count++;
  if (record_count == 0ULL) failure_count++;
  if (record_count != SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_SELECTED_TRANSFER_COUNT) failure_count++;
  if (SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_TRAINING_OVERLAP_COUNT != 0ULL) failure_count++;

  for (size_t i = 0; i < record_count; i++) {{
    const selected13_materialization_augmented_validation_record_t *record =
        &SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_RECORDS[i];
    if (record->candidate_id_u64 == 0ULL) failure_count++;
    if (record->status_code == 0ULL) failure_count++;
    if (record->relation_derived_ecdlp && !record->public_key_verified) failure_count++;
    if (record->relation_derived_ecdlp && record->derived_secret == 0ULL) failure_count++;
    if (record->relation_derived_ecdlp) {{
      accepted++;
      if (record->below_rho) accepted_below++;
    }}
    if (record->materialization_overlay) {{
      overlay++;
      if (!record->relation_derived_ecdlp || !record->below_rho || record->rank == 0ULL ||
          record->relation_count == 0ULL) {{
        failure_count++;
      }}
    }}
  }}

  if (accepted != SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_ACCEPTED_COUNT) failure_count++;
  if (accepted_below != SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_ACCEPTED_BELOW_COUNT) failure_count++;
  if (overlay != SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_OVERLAY_COUNT) failure_count++;

  printf("selected13_materialization_augmented_validation_preflight records=%llu accepted=%llu accepted_below=%llu overlays=%llu overlap=%llu failures=%llu\\n",
         (unsigned long long)record_count,
         (unsigned long long)accepted,
         (unsigned long long)accepted_below,
         (unsigned long long)overlay,
         (unsigned long long)SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_TRAINING_OVERLAP_COUNT,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_mat_augmented_validation_") as tmp:
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
    seed_path = Path(args.seed_validation)
    evidence_path = Path(args.direct_evidence)
    seed = load_json(seed_path)
    direct_evidence = load_json(evidence_path)
    failures: list[dict[str, Any]] = []
    if seed.get("claim_status") != "SELECTED13_SECOND_STAGE_DISJOINT_VALIDATION_BELOW_RHO_EXPORT":
        failures.append({"code": "seed_validation_claim_status_unexpected", "claim_status": seed.get("claim_status")})
    if seed.get("failures"):
        failures.append({"code": "seed_validation_has_failures", "failures": seed.get("failures")})
    if direct_evidence.get("claim_status") != "SELECTED13_MATERIALIZATION_DIRECT_EVIDENCE_HAS_DERIVED_ROWS":
        failures.append(
            {"code": "direct_evidence_claim_status_unexpected", "claim_status": direct_evidence.get("claim_status")}
        )
    if direct_evidence.get("failures"):
        failures.append({"code": "direct_evidence_has_failures", "failures": direct_evidence.get("failures")})

    records, overlays, overlay_failures = augment_records(seed, direct_evidence)
    failures.extend(overlay_failures)
    summary = summarize(records, overlays, failures, seed.get("summary") or {})
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, overlays),
        "parameters": {
            "direct_evidence": str(evidence_path),
            "seed_validation": str(seed_path),
        },
        "source_summary": {
            "direct_evidence": direct_evidence.get("summary"),
            "seed_validation": seed.get("summary"),
        },
        "summary": summary,
        "materialization_overlays": overlays,
        "direct_replay_records": records,
        "failures": failures,
        "honesty_boundary": {
            "general_ecdlp_algorithm_claimed": False,
            "materialization_overlay_only": True,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": bool(overlays),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-validation", type=Path, default=DEFAULT_SEED_VALIDATION)
    parser.add_argument("--direct-evidence", type=Path, default=DEFAULT_DIRECT_EVIDENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["direct_replay_records"], payload["summary"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["claim_status"] = "SELECTED13_MATERIALIZATION_AUGMENTED_VALIDATION_FAILED"
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(args.out), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
