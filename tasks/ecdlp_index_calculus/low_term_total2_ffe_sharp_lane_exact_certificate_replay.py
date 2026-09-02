#!/usr/bin/env python3
"""Replay exact selected13 sharp-lane certificate material.

The row execution worklist carries the six exact certificate hashes that must
be reproduced before inherited promotions or direct/rank backfill exports are
credited.  This script consumes the strict kernel contract and row worklist,
recomputes the canonical certificate hashes from embedded certificate material,
checks the exact row/work-item linkage, emits a compact C header, and compiles
a native preflight over the exact replay records.

This is an exact-certificate replay gate only.  It does not evaluate summation
polynomials, regenerate the original direct-relation sidecar artifacts, export
direct/rank rows, solve ECDLP, or claim a Pollard-rho speedup.
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

import low_term_total2_ffe_sharp_lane_kernel_contract as kernel_contract


SCHEMA = "ecdlp.low_term_total2_ffe_sharp_lane_exact_certificate_replay.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_CONTRACT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9999_probe.json"
)
DEFAULT_WORKLIST = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_execution_worklist_selected13_9696_9999_probe.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_exact_certificate_replay_selected13_9696_9999_probe.json"
)
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_exact_certificate_replay_selected13_9696_9999_probe.h"
)
SELECTED13_MASK = 1 << 13
MAX_FORM_MASKS = 4
CLASSIFICATION_CODES = {
    "ACCEPTED_MISSING_COLUMN_RANK_GAIN": 1,
    "RANK_GAIN_WITHOUT_ACCEPTED_MISSING_COLUMN": 2,
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


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def support_mask(raw: Any) -> int:
    mask = 0
    for item in raw or []:
        value = as_int(item)
        if value < 0 or value >= 64:
            raise ValueError(f"support index outside u64 mask range: {value}")
        mask |= 1 << value
    return mask


def support_masks(raw: Any) -> list[int]:
    masks = [support_mask(item) for item in raw or []]
    while len(masks) < MAX_FORM_MASKS:
        masks.append(0)
    return masks[:MAX_FORM_MASKS]


def digest_u64(raw: Any) -> int:
    text = str(raw or "")
    suffix = text.rsplit("_", 1)[-1]
    if suffix and all(ch in "0123456789abcdefABCDEF" for ch in suffix):
        return int(suffix[-16:], 16)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def sorted_ints(raw: Any) -> list[int]:
    return sorted(as_int(item) for item in raw or [])


def work_items_by_row(worklist: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("row_id")): item
        for item in worklist.get("work_items") or []
        if isinstance(item, dict) and item.get("row_id") is not None
    }


def exact_checks(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in (contract.get("contract") or {}).get("exact_certificate_checks") or []
        if isinstance(item, dict)
    ]


def build_records(contract: dict[str, Any], worklist: dict[str, Any]) -> list[dict[str, Any]]:
    by_row = work_items_by_row(worklist)
    records = []
    for exact_index, cert in enumerate(exact_checks(contract)):
        material = cert.get("certificate_material") or {}
        recomputed_material = kernel_contract.canonical_certificate_material(material)
        recomputed_hash = kernel_contract.canonical_digest(recomputed_material, "cert")
        work_item = by_row.get(str(cert.get("row_id"))) or {}
        form_masks = support_masks(material.get("form_supports") or [])
        accepted_missing_mask = support_mask(material.get("accepted_missing_columns") or [])
        accepted_priority_mask = support_mask(material.get("accepted_priority_columns") or [])
        selected_mask = support_mask(material.get("selected_term_support") or [])
        direct_ops_scaled = round(as_float(material.get("direct_ops_over_rho")) * 100_000_000)
        records.append(
            {
                "accepted_missing_columns": sorted_ints(material.get("accepted_missing_columns")),
                "accepted_missing_mask": accepted_missing_mask,
                "accepted_priority_columns": sorted_ints(material.get("accepted_priority_columns")),
                "accepted_priority_mask": accepted_priority_mask,
                "artifact": material.get("artifact"),
                "artifact_available_in_worktree": Path(str(material.get("artifact") or "")).is_file(),
                "certificate_hash": cert.get("certificate_hash"),
                "certificate_hash_u64": digest_u64(cert.get("certificate_hash")),
                "classification": material.get("classification"),
                "classification_code": CLASSIFICATION_CODES.get(str(material.get("classification") or ""), 0),
                "direct_ops_over_rho": as_float(material.get("direct_ops_over_rho")),
                "direct_ops_over_rho_scaled": direct_ops_scaled,
                "exact_index": exact_index,
                "expected_form_supports": kernel_contract.sorted_supports(cert.get("expected_form_supports")),
                "first_pass_id": cert.get("first_pass_id"),
                "form_count": len(material.get("form_supports") or []),
                "form_masks": form_masks,
                "form_supports": kernel_contract.sorted_supports(material.get("form_supports")),
                "group_id": cert.get("group_id"),
                "rank": as_int(material.get("rank")),
                "rank_gain": as_int(material.get("rank_gain")),
                "rank_score": material.get("rank_score") or {},
                "recomputed_certificate_hash": recomputed_hash,
                "recomputed_certificate_material": recomputed_material,
                "row_check_hash": cert.get("row_check_hash"),
                "row_check_hash_u64": digest_u64(cert.get("row_check_hash")),
                "row_id": cert.get("row_id"),
                "row_keys": kernel_contract.sorted_strings(material.get("row_keys")),
                "salts": sorted_ints(key.rsplit("salt", 1)[-1] for key in material.get("row_keys") or []),
                "selected_support_mask": selected_mask,
                "selector": cert.get("selector"),
                "target": material.get("target"),
                "top_k": as_int(cert.get("top_k")),
                "transfer_index": as_int(cert.get("transfer_index")),
                "unique_factor_relation_gain": as_int(material.get("unique_factor_relation_gain")),
                "work_item": {
                    "accepted_exact_certificate_hash": work_item.get("accepted_exact_certificate_hash"),
                    "accepted_exact_certificate_hash_u64": as_int(
                        work_item.get("accepted_exact_certificate_hash_u64")
                    ),
                    "phase": work_item.get("phase"),
                    "row_check_hash": work_item.get("row_check_hash"),
                    "row_check_hash_u64": as_int(work_item.get("row_check_hash_u64")),
                    "selected_support_mask": as_int(work_item.get("selected_support_mask")),
                    "work_item_index": as_int(work_item.get("work_item_index"), -1),
                },
            }
        )
    return records


def validate_sources(contract: dict[str, Any], worklist: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if contract.get("claim_status") != "FFE_SHARP_LANE_KERNEL_CONTRACT_READY":
        failures.append({"code": "contract_not_ready", "claim_status": contract.get("claim_status")})
    if contract.get("failures"):
        failures.append({"code": "contract_has_failures", "failures": contract.get("failures")})
    if worklist.get("claim_status") != "FFE_SHARP_LANE_EXECUTION_WORKLIST_READY":
        failures.append({"code": "worklist_not_ready", "claim_status": worklist.get("claim_status")})
    if worklist.get("failures"):
        failures.append({"code": "worklist_has_failures", "failures": worklist.get("failures")})
    return failures


def validate_records(records: list[dict[str, Any]], expected_exact_count: int) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(records) != expected_exact_count:
        failures.append({"code": "exact_record_count_mismatch", "expected": expected_exact_count, "observed": len(records)})
    for record in records:
        row_id = record.get("row_id")
        if record.get("certificate_hash") != record.get("recomputed_certificate_hash"):
            failures.append(
                {
                    "code": "certificate_hash_mismatch",
                    "row_id": row_id,
                    "certificate_hash": record.get("certificate_hash"),
                    "recomputed": record.get("recomputed_certificate_hash"),
                }
            )
        if record.get("form_supports") != record.get("expected_form_supports"):
            failures.append(
                {
                    "code": "form_support_mismatch",
                    "row_id": row_id,
                    "form_supports": record.get("form_supports"),
                    "expected": record.get("expected_form_supports"),
                }
            )
        if as_int(record.get("classification_code")) == 0:
            failures.append({"code": "unknown_exact_classification", "row_id": row_id, "classification": record.get("classification")})
        if as_int(record.get("rank")) != as_int(record.get("form_count")) + 1:
            failures.append(
                {
                    "code": "rank_form_count_mismatch",
                    "row_id": row_id,
                    "rank": record.get("rank"),
                    "form_count": record.get("form_count"),
                }
            )
        if as_int(record.get("rank_gain")) < 1:
            failures.append({"code": "nonpositive_rank_gain", "row_id": row_id})
        if as_int(record.get("unique_factor_relation_gain")) < as_int(record.get("rank_gain")):
            failures.append({"code": "unique_gain_below_rank_gain", "row_id": row_id})
        if as_int(record.get("accepted_missing_mask")) & ~as_int(record.get("accepted_priority_mask")):
            failures.append({"code": "accepted_missing_not_priority_subset", "row_id": row_id})
        if not (as_int(record.get("selected_support_mask")) & SELECTED13_MASK):
            failures.append({"code": "certificate_missing_selected13", "row_id": row_id})
        work_item = record.get("work_item") or {}
        if as_int(work_item.get("work_item_index"), -1) < 0:
            failures.append({"code": "missing_exact_work_item", "row_id": row_id})
            continue
        if work_item.get("phase") != "exact_certificate_replay":
            failures.append({"code": "work_item_phase_not_exact", "row_id": row_id, "phase": work_item.get("phase")})
        if work_item.get("accepted_exact_certificate_hash") != record.get("certificate_hash"):
            failures.append({"code": "work_item_certificate_hash_mismatch", "row_id": row_id})
        if work_item.get("row_check_hash") != record.get("row_check_hash"):
            failures.append({"code": "work_item_row_hash_mismatch", "row_id": row_id})
        if as_int(work_item.get("selected_support_mask")) != as_int(record.get("selected_support_mask")):
            failures.append({"code": "work_item_selected_mask_mismatch", "row_id": row_id})
    return failures


def c_u64_array(values: list[int]) -> str:
    if not values:
        return "{0ULL}"
    return "{" + ", ".join(f"{as_int(value)}ULL" for value in values) + "}"


def render_c_header(records: list[dict[str, Any]]) -> str:
    record_lines = []
    for record in records:
        salts = [as_int(value) for value in record.get("salts") or []][:2]
        while len(salts) < 2:
            salts.append(0)
        form_masks = [as_int(value) for value in record.get("form_masks") or []][:MAX_FORM_MASKS]
        while len(form_masks) < MAX_FORM_MASKS:
            form_masks.append(0)
        work_item = record.get("work_item") or {}
        record_lines.append(
            "  {"
            f"{as_int(record.get('exact_index'))}ULL, "
            f"{as_int(work_item.get('work_item_index'), -1)}ULL, "
            f"{as_int(record.get('transfer_index'))}ULL, "
            f"{c_u64_array(salts)}, "
            f"{as_int(record.get('row_check_hash_u64'))}ULL, "
            f"{as_int(record.get('certificate_hash_u64'))}ULL, "
            f"{as_int(record.get('selected_support_mask'))}ULL, "
            f"{c_u64_array(form_masks)}, "
            f"{as_int(record.get('form_count'))}ULL, "
            f"{as_int(record.get('accepted_missing_mask'))}ULL, "
            f"{as_int(record.get('accepted_priority_mask'))}ULL, "
            f"{as_int(record.get('rank'))}ULL, "
            f"{as_int(record.get('rank_gain'))}ULL, "
            f"{as_int(record.get('unique_factor_relation_gain'))}ULL, "
            f"{as_int(record.get('classification_code'))}ULL, "
            f"{as_int(record.get('direct_ops_over_rho_scaled'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_FFE_SHARP_LANE_EXACT_CERTIFICATE_REPLAY_H
#define LOW_TERM_TOTAL2_FFE_SHARP_LANE_EXACT_CERTIFICATE_REPLAY_H

#include <stdint.h>

#define SELECTED13_EXACT_CERTIFICATE_REPLAY_COUNT {len(records)}
#define SELECTED13_EXACT_CERTIFICATE_MAX_FORM_MASKS {MAX_FORM_MASKS}

typedef struct {{
  uint64_t exact_index;
  uint64_t work_item_index;
  uint64_t transfer_index;
  uint64_t salts[2];
  uint64_t row_check_hash_u64;
  uint64_t certificate_hash_u64;
  uint64_t selected_support_mask;
  uint64_t form_masks[SELECTED13_EXACT_CERTIFICATE_MAX_FORM_MASKS];
  uint64_t form_count;
  uint64_t accepted_missing_mask;
  uint64_t accepted_priority_mask;
  uint64_t rank;
  uint64_t rank_gain;
  uint64_t unique_factor_relation_gain;
  uint64_t classification_code;
  uint64_t direct_ops_over_rho_scaled;
}} selected13_exact_certificate_replay_t;

static const selected13_exact_certificate_replay_t SELECTED13_EXACT_CERTIFICATE_REPLAYS[] = {{
{chr(10).join(record_lines)}
}};

#endif
"""


def render_preflight_c(header_basename: str, records: list[dict[str, Any]]) -> str:
    expected_record_count = len(records)
    expected_rank_gain_sum = sum(as_int(record.get("rank_gain")) for record in records)
    expected_form_count_sum = sum(as_int(record.get("form_count")) for record in records)
    expected_accepted_missing_count = sum(1 for record in records if as_int(record.get("accepted_missing_mask")) != 0)
    return f"""#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

int main(void) {{
  const uint64_t selected13_mask = {SELECTED13_MASK}ULL;
  uint64_t failure_count = 0;
  uint64_t selected13_missing_count = 0;
  uint64_t zero_cert_hash_count = 0;
  uint64_t zero_row_hash_count = 0;
  uint64_t rank_gain_sum = 0;
  uint64_t form_count_sum = 0;
  uint64_t accepted_missing_record_count = 0;
  uint64_t classification_counts[3] = {{0, 0, 0}};
  const size_t record_count =
      sizeof(SELECTED13_EXACT_CERTIFICATE_REPLAYS) / sizeof(SELECTED13_EXACT_CERTIFICATE_REPLAYS[0]);
  if (record_count != SELECTED13_EXACT_CERTIFICATE_REPLAY_COUNT) failure_count++;

  for (size_t i = 0; i < record_count; i++) {{
    const selected13_exact_certificate_replay_t *record = &SELECTED13_EXACT_CERTIFICATE_REPLAYS[i];
    if (record->exact_index != i) failure_count++;
    if ((record->selected_support_mask & selected13_mask) == 0) selected13_missing_count++;
    if (record->certificate_hash_u64 == 0) zero_cert_hash_count++;
    if (record->row_check_hash_u64 == 0) zero_row_hash_count++;
    if (record->work_item_index >= 60) failure_count++;
    if (record->form_count == 0 || record->form_count > SELECTED13_EXACT_CERTIFICATE_MAX_FORM_MASKS) failure_count++;
    if (record->rank != record->form_count + 1) failure_count++;
    if (record->rank_gain == 0) failure_count++;
    if (record->unique_factor_relation_gain < record->rank_gain) failure_count++;
    if ((record->accepted_missing_mask & ~record->accepted_priority_mask) != 0) failure_count++;
    if (record->accepted_missing_mask != 0) accepted_missing_record_count++;
    if (record->classification_code >= 3 || record->classification_code == 0) {{
      failure_count++;
    }} else {{
      classification_counts[record->classification_code]++;
    }}
    for (uint64_t j = 0; j < record->form_count; j++) {{
      if (record->form_masks[j] == 0) failure_count++;
    }}
    for (uint64_t j = record->form_count; j < SELECTED13_EXACT_CERTIFICATE_MAX_FORM_MASKS; j++) {{
      if (record->form_masks[j] != 0) failure_count++;
    }}
    rank_gain_sum += record->rank_gain;
    form_count_sum += record->form_count;
  }}

  if (selected13_missing_count != 0) failure_count++;
  if (zero_cert_hash_count != 0) failure_count++;
  if (zero_row_hash_count != 0) failure_count++;
  if (record_count != {expected_record_count}) failure_count++;
  if (rank_gain_sum != {expected_rank_gain_sum}) failure_count++;
  if (form_count_sum != {expected_form_count_sum}) failure_count++;
  if (accepted_missing_record_count != {expected_accepted_missing_count}) failure_count++;

  printf("{{");
  printf("\\\"record_count\\\":%llu,", (unsigned long long)record_count);
  printf("\\\"failure_count\\\":%llu,", (unsigned long long)failure_count);
  printf("\\\"selected13_missing_count\\\":%llu,", (unsigned long long)selected13_missing_count);
  printf("\\\"zero_cert_hash_count\\\":%llu,", (unsigned long long)zero_cert_hash_count);
  printf("\\\"zero_row_hash_count\\\":%llu,", (unsigned long long)zero_row_hash_count);
  printf("\\\"rank_gain_sum\\\":%llu,", (unsigned long long)rank_gain_sum);
  printf("\\\"form_count_sum\\\":%llu,", (unsigned long long)form_count_sum);
  printf("\\\"accepted_missing_record_count\\\":%llu,", (unsigned long long)accepted_missing_record_count);
  printf("\\\"classification_counts\\\":{{\\\"1\\\":%llu,\\\"2\\\":%llu}}",
         (unsigned long long)classification_counts[1],
         (unsigned long long)classification_counts[2]);
  printf("}}\\n");
  return failure_count == 0 ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path, compiler: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    c_source = render_preflight_c(header_path.name, records)
    source_hash = hashlib.sha256(c_source.encode("utf-8")).hexdigest()
    temp_root = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path(tempfile.gettempdir())
    env = os.environ.copy()
    env["TMPDIR"] = str(temp_root)
    with tempfile.TemporaryDirectory(prefix="ecdlp_selected13_exact_", dir=str(temp_root)) as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "selected13_exact_certificate_preflight.c"
        exe_path = temp_path / "selected13_exact_certificate_preflight"
        c_path.write_text(c_source)
        command = [
            compiler,
            "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            str(header_path.parent),
            str(c_path),
            "-o",
            str(exe_path),
        ]
        compile_run = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
        if compile_run.returncode != 0:
            return {
                "compiled": False,
                "compile_command": command,
                "compile_returncode": compile_run.returncode,
                "compile_stdout": compile_run.stdout,
                "compile_stderr": compile_run.stderr,
                "executed": False,
                "c_source_sha256": source_hash,
            }
        native_run = subprocess.run([str(exe_path)], capture_output=True, text=True, check=False, env=env)
        try:
            native_summary = json.loads(native_run.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            native_summary = None
        return {
            "compiled": True,
            "compile_command": command,
            "compile_returncode": compile_run.returncode,
            "compile_stdout": compile_run.stdout,
            "compile_stderr": compile_run.stderr,
            "executed": True,
            "run_returncode": native_run.returncode,
            "run_stdout": native_run.stdout,
            "run_stderr": native_run.stderr,
            "native_summary": native_summary,
            "c_source_sha256": source_hash,
        }


def compare_native(records: list[dict[str, Any]], native: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not native.get("compiled"):
        failures.append({"code": "native_preflight_compile_failed", "compile_stderr": native.get("compile_stderr")})
        return failures
    if as_int(native.get("run_returncode"), -1) != 0:
        failures.append({"code": "native_preflight_run_failed", "run_stdout": native.get("run_stdout")})
    summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    if not summary:
        failures.append({"code": "native_preflight_stdout_not_json", "run_stdout": native.get("run_stdout")})
        return failures
    if as_int(summary.get("record_count"), -1) != len(records):
        failures.append({"code": "native_record_count_mismatch"})
    if as_int(summary.get("failure_count"), -1) != 0:
        failures.append({"code": "native_preflight_reported_failures", "native_summary": summary})
    expected_class_counts = Counter(as_int(record.get("classification_code")) for record in records)
    if (summary.get("classification_counts") or {}) != {
        "1": int(expected_class_counts.get(1, 0)),
        "2": int(expected_class_counts.get(2, 0)),
    }:
        failures.append({"code": "native_classification_counts_mismatch", "native": summary.get("classification_counts")})
    return failures


def summarize(records: list[dict[str, Any]], native_summary: dict[str, Any]) -> dict[str, Any]:
    class_counts = Counter(str(record.get("classification")) for record in records)
    missing_sidecars = [record.get("artifact") for record in records if not record.get("artifact_available_in_worktree")]
    return {
        "accepted_missing_record_count": sum(1 for record in records if as_int(record.get("accepted_missing_mask")) != 0),
        "artifact_sidecar_available_count": len(records) - len(missing_sidecars),
        "artifact_sidecar_missing_count": len(missing_sidecars),
        "classification_counts": dict(sorted(class_counts.items())),
        "embedded_material_replay_count": len(records),
        "form_count_sum": sum(as_int(record.get("form_count")) for record in records),
        "native_preflight_failure_count": as_int(native_summary.get("failure_count"), -1),
        "native_preflight_verified": as_int(native_summary.get("failure_count"), -1) == 0,
        "rank_gain_sum": sum(as_int(record.get("rank_gain")) for record in records),
        "record_count": len(records),
        "selected13_missing_count": sum(1 for record in records if not (as_int(record.get("selected_support_mask")) & SELECTED13_MASK)),
        "transfers": [as_int(record.get("transfer_index")) for record in records],
        "unique_certificate_hash_count": len({record.get("certificate_hash") for record in records}),
        "unique_row_hash_count": len({record.get("row_check_hash") for record in records}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--worklist", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--no-c-header", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_json(args.contract)
    worklist = load_json(args.worklist)
    records = build_records(contract, worklist)
    expected_exact_count = as_int((contract.get("summary") or {}).get("exact_certificate_check_count"), -1)
    failures = validate_sources(contract, worklist)
    failures.extend(validate_records(records, expected_exact_count))
    native: dict[str, Any] = {"compiled": False, "executed": False, "native_summary": {}}
    if not args.no_c_header:
        args.c_header_out.parent.mkdir(parents=True, exist_ok=True)
        args.c_header_out.write_text(render_c_header(records))
        native = run_native_preflight(args.c_header_out, args.cc, records)
        failures.extend(compare_native(records, native))

    native_summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    verified = not failures
    payload = {
        "artifacts": {
            "c_header": None if args.no_c_header else str(args.c_header_out),
            "contract": str(args.contract),
            "worklist": str(args.worklist),
        },
        "claim_status": (
            "FFE_SHARP_LANE_EXACT_CERTIFICATE_REPLAY_READY"
            if verified
            else "FFE_SHARP_LANE_EXACT_CERTIFICATE_REPLAY_FAILED_CHECK"
        ),
        "created_at": now_iso(),
        "failures": failures,
        "honesty_boundary": [
            "This replays embedded exact certificate material and native preflight records only.",
            "It does not evaluate summation polynomials, regenerate missing sidecar certificate artifacts, export direct/rank rows, solve ECDLP, or claim a Pollard-rho speedup.",
        ],
        "native_preflight": native,
        "records": records,
        "schema": SCHEMA,
        "summary": summarize(records, native_summary),
    }
    payload["summary"]["verified"] = verified
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "claim_status": payload["claim_status"],
                "failures": failures,
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
