#!/usr/bin/env python3
"""Replay selected13 backfill-gate forms through modular linear algebra.

The backfill transfer gate binds source-side coefficient forms to missing
backfill row hashes.  This probe tests the algebraic replay boundary before a
lower-level FFE/summation-polynomial worker tries to re-emit rows: each mask
lane, each source tier group, and each full transfer gate is reduced modulo the
target order and classified as source-secret replay, inconsistent source mix,
or no unique secret.

This is a negative/triage gate for coefficient reuse.  It does not verify
candidate points, export direct/rank rows, solve ECDLP, or claim a Pollard-rho
speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_ffe_sharp_lane_backfill_replay_algebra_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_BACKFILL_GATE = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_backfill_transfer_gate_selected13_9696_9999_probe.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_backfill_replay_algebra_selected13_9696_9999_probe.json"
)
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_backfill_replay_algebra_selected13_9696_9999_probe.h"
)

ORDER = 11779
COEFF_COUNT = 17
PRIMARY_BACKFILL_TRANSFERS = [9981, 9943]
EXPECTED_MASK_LANE_CASES = 6
EXPECTED_TIER_GROUP_CASES = 4
EXPECTED_TRANSFER_GATE_CASES = 2
EXPECTED_REPLAY_CASES = EXPECTED_MASK_LANE_CASES + EXPECTED_TIER_GROUP_CASES + EXPECTED_TRANSFER_GATE_CASES

CASE_KIND_CODES = {
    "mask_lane": 1,
    "tier_group": 2,
    "transfer_gate": 3,
}
RESULT_CODES = {
    "SOURCE_SECRET_REPLAY_ONLY": 1,
    "SOURCE_FORM_SYSTEM_INCONSISTENT": 2,
    "NO_UNIQUE_SECRET_FROM_REPLAY_FORMS": 3,
    "UNVERIFIED_NEW_LINEAR_DERIVATION": 4,
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


def inv_mod(value: int, modulus: int) -> int:
    return pow(value % modulus, -1, modulus)


def rref_mod(rows: list[list[int]], modulus: int) -> tuple[list[list[int]], list[int], bool]:
    if not rows:
        return [], [], True
    rows = [[value % modulus for value in row] for row in rows]
    nvars = len(rows[0]) - 1
    pivot_cols: list[int] = []
    pivot_row = 0
    for col in range(nvars):
        pivot = None
        for row_index in range(pivot_row, len(rows)):
            if rows[row_index][col] % modulus:
                pivot = row_index
                break
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inverse = inv_mod(rows[pivot_row][col], modulus)
        rows[pivot_row] = [(value * inverse) % modulus for value in rows[pivot_row]]
        for row_index in range(len(rows)):
            if row_index == pivot_row:
                continue
            scale = rows[row_index][col] % modulus
            if scale:
                rows[row_index] = [
                    (rows[row_index][i] - scale * rows[pivot_row][i]) % modulus
                    for i in range(nvars + 1)
                ]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    for row in rows:
        if all(row[col] % modulus == 0 for col in range(nvars)) and row[-1] % modulus:
            return rows, pivot_cols, False
    return rows, pivot_cols, True


def derive_secret_from_gate_forms(forms: list[dict[str, Any]], modulus: int = ORDER) -> dict[str, Any]:
    if not forms:
        return {
            "consistent": True,
            "derived": False,
            "derived_secret": None,
            "pivot_cols": [],
            "rank": 0,
        }
    rows = []
    for form in forms:
        coeffs = [as_int(value) % modulus for value in form.get("coeffs") or []]
        while len(coeffs) < COEFF_COUNT:
            coeffs.append(0)
        rows.append(coeffs[:COEFF_COUNT] + [as_int(form.get("rhs")) % modulus])

    reduced, pivot_cols, consistent = rref_mod(rows, modulus)
    if not consistent or 0 not in pivot_cols:
        return {
            "consistent": consistent,
            "derived": False,
            "derived_secret": None,
            "pivot_cols": pivot_cols,
            "rank": len(pivot_cols),
        }
    free_cols = set(range(COEFF_COUNT)) - set(pivot_cols)
    row_index = pivot_cols.index(0)
    if any(reduced[row_index][col] % modulus for col in free_cols):
        return {
            "consistent": consistent,
            "derived": False,
            "derived_secret": None,
            "pivot_cols": pivot_cols,
            "rank": len(pivot_cols),
        }
    return {
        "consistent": consistent,
        "derived": True,
        "derived_secret": reduced[row_index][-1] % modulus,
        "pivot_cols": pivot_cols,
        "rank": len(pivot_cols),
    }


def result_status(derived: dict[str, Any], source_secret_set: list[int]) -> str:
    if not bool(derived.get("consistent")):
        return "SOURCE_FORM_SYSTEM_INCONSISTENT"
    if bool(derived.get("derived")):
        secret = as_int(derived.get("derived_secret"), -1)
        if secret in set(source_secret_set):
            return "SOURCE_SECRET_REPLAY_ONLY"
        return "UNVERIFIED_NEW_LINEAR_DERIVATION"
    return "NO_UNIQUE_SECRET_FROM_REPLAY_FORMS"


def forms_by_index(gate: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        as_int(row.get("candidate_form_index"), -1): row
        for row in gate.get("candidate_forms") or []
        if isinstance(row, dict)
    }


def transfer_order(transfer: int) -> int:
    try:
        return PRIMARY_BACKFILL_TRANSFERS.index(transfer)
    except ValueError:
        return len(PRIMARY_BACKFILL_TRANSFERS)


def case_hash_u64(case: dict[str, Any]) -> int:
    payload = {
        "candidate_form_indices": case.get("candidate_form_indices"),
        "case_kind": case.get("case_kind"),
        "family_mask": case.get("family_mask"),
        "source_tier": case.get("source_tier"),
        "transfer_index": case.get("backfill_transfer_index"),
    }
    return int(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16], 16)


def build_case(
    case_kind: str,
    backfill_transfer: int,
    forms: list[dict[str, Any]],
    *,
    family_mask: int = 0,
    source_tier: str = "",
    lane_index: int = -1,
) -> dict[str, Any]:
    ordered = sorted(forms, key=lambda row: as_int(row.get("candidate_form_index"), -1))
    source_secret_set = sorted({as_int(row.get("source_derived_secret"), -1) for row in ordered})
    source_transfers = sorted({as_int(row.get("source_transfer_index"), -1) for row in ordered})
    row_hashes = sorted({str(row.get("backfill_row_check_hash") or "") for row in ordered})
    derived = derive_secret_from_gate_forms(ordered)
    status = result_status(derived, source_secret_set)
    case = {
        "accepted_backfill_export_count": 0,
        "backfill_row_check_hashes": row_hashes,
        "backfill_transfer_index": backfill_transfer,
        "candidate_form_count": len(ordered),
        "candidate_form_indices": [as_int(row.get("candidate_form_index"), -1) for row in ordered],
        "case_kind": case_kind,
        "case_kind_code": CASE_KIND_CODES.get(case_kind, 0),
        "derived": derived,
        "derived_secret_matches_source": bool(derived.get("derived"))
        and as_int(derived.get("derived_secret"), -1) in set(source_secret_set),
        "family_mask": family_mask,
        "lane_index": lane_index,
        "result_code": RESULT_CODES[status],
        "result_status": status,
        "source_secret_count": len(source_secret_set),
        "source_secret_set": source_secret_set,
        "source_tier": source_tier,
        "source_transfers": source_transfers,
        "worker_obligation": (
            "reemit_and_publicly_verify_backfill_direct_rank_row"
            if status != "UNVERIFIED_NEW_LINEAR_DERIVATION"
            else "public_verify_before_any_backfill_claim"
        ),
    }
    case["case_hash_u64"] = case_hash_u64(case)
    return case


def build_replay_cases(gate: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = forms_by_index(gate)
    cases: list[dict[str, Any]] = []

    for lane in sorted(gate.get("mask_lanes") or [], key=lambda item: as_int(item.get("lane_index"), -1)):
        forms = [lookup[index] for index in lane.get("candidate_form_indices") or [] if index in lookup]
        cases.append(
            build_case(
                "mask_lane",
                as_int(lane.get("backfill_transfer_index"), -1),
                forms,
                family_mask=as_int(lane.get("family_mask")),
                lane_index=as_int(lane.get("lane_index"), -1),
                source_tier=str(lane.get("source_tier") or ""),
            )
        )

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gate.get("candidate_forms") or []:
        if not isinstance(row, dict):
            continue
        grouped[(as_int(row.get("backfill_transfer_index"), -1), str(row.get("source_tier") or ""))].append(row)
    for key in sorted(grouped, key=lambda item: (transfer_order(item[0]), item[1])):
        transfer, tier = key
        cases.append(build_case("tier_group", transfer, grouped[key], source_tier=tier))

    by_transfer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in gate.get("candidate_forms") or []:
        if not isinstance(row, dict):
            continue
        by_transfer[as_int(row.get("backfill_transfer_index"), -1)].append(row)
    for transfer in PRIMARY_BACKFILL_TRANSFERS:
        cases.append(build_case("transfer_gate", transfer, by_transfer.get(transfer, [])))

    for index, case in enumerate(cases):
        case["replay_case_index"] = index
    return cases


def validate_inputs(gate: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if gate.get("claim_status") != "FFE_SHARP_LANE_BACKFILL_TRANSFER_GATE_READY":
        failures.append({"code": "backfill_transfer_gate_not_ready", "claim_status": gate.get("claim_status")})
    if gate.get("failures"):
        failures.append({"code": "backfill_transfer_gate_has_failures", "failures": gate.get("failures")})
    if (gate.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "backfill_transfer_gate_not_verified"})
    if as_int((gate.get("summary") or {}).get("accepted_backfill_export_count"), -1) != 0:
        failures.append({"code": "backfill_transfer_gate_already_claims_export"})
    return failures


def validate_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(cases) != EXPECTED_REPLAY_CASES:
        failures.append({"code": "replay_case_count_mismatch", "observed": len(cases)})
    counts = Counter(str(case.get("case_kind")) for case in cases)
    if counts.get("mask_lane", 0) != EXPECTED_MASK_LANE_CASES:
        failures.append({"code": "mask_lane_case_count_mismatch", "observed": counts.get("mask_lane", 0)})
    if counts.get("tier_group", 0) != EXPECTED_TIER_GROUP_CASES:
        failures.append({"code": "tier_group_case_count_mismatch", "observed": counts.get("tier_group", 0)})
    if counts.get("transfer_gate", 0) != EXPECTED_TRANSFER_GATE_CASES:
        failures.append({"code": "transfer_gate_case_count_mismatch", "observed": counts.get("transfer_gate", 0)})
    for case in cases:
        if as_int(case.get("accepted_backfill_export_count"), -1) != 0:
            failures.append({"code": "replay_case_claims_backfill_export", "replay_case_index": case.get("replay_case_index")})
        if as_int(case.get("candidate_form_count")) <= 0:
            failures.append({"code": "replay_case_without_forms", "replay_case_index": case.get("replay_case_index")})
        if case.get("result_status") == "UNVERIFIED_NEW_LINEAR_DERIVATION":
            failures.append(
                {
                    "code": "unverified_new_linear_derivation_requires_public_verify",
                    "derived": case.get("derived"),
                    "replay_case_index": case.get("replay_case_index"),
                }
            )
    return failures


def render_c_header(cases: list[dict[str, Any]]) -> str:
    rows = []
    for case in cases:
        derived = case.get("derived") or {}
        rows.append(
            "  {"
            f"{as_int(case.get('replay_case_index'))}ULL, "
            f"{as_int(case.get('case_kind_code'))}ULL, "
            f"{as_int(case.get('result_code'))}ULL, "
            f"{as_int(case.get('backfill_transfer_index'))}ULL, "
            f"{as_int(case.get('family_mask'))}ULL, "
            f"{as_int(case.get('candidate_form_count'))}ULL, "
            f"{as_int(case.get('source_secret_count'))}ULL, "
            f"{1 if bool(derived.get('consistent')) else 0}ULL, "
            f"{1 if bool(derived.get('derived')) else 0}ULL, "
            f"{as_int(derived.get('derived_secret'))}ULL, "
            f"{as_int(derived.get('rank'))}ULL, "
            f"{as_int(case.get('accepted_backfill_export_count'))}ULL, "
            f"{as_int(case.get('case_hash_u64'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_FFE_SHARP_LANE_BACKFILL_REPLAY_ALGEBRA_H
#define LOW_TERM_TOTAL2_FFE_SHARP_LANE_BACKFILL_REPLAY_ALGEBRA_H

#include <stdint.h>

#define SELECTED13_BACKFILL_REPLAY_CASE_COUNT {len(cases)}
#define SELECTED13_BACKFILL_REPLAY_ACCEPTED_EXPORT_COUNT 0

#define SELECTED13_REPLAY_CASE_MASK_LANE 1ULL
#define SELECTED13_REPLAY_CASE_TIER_GROUP 2ULL
#define SELECTED13_REPLAY_CASE_TRANSFER_GATE 3ULL

#define SELECTED13_REPLAY_RESULT_SOURCE_SECRET_ONLY 1ULL
#define SELECTED13_REPLAY_RESULT_INCONSISTENT 2ULL
#define SELECTED13_REPLAY_RESULT_NO_UNIQUE_SECRET 3ULL
#define SELECTED13_REPLAY_RESULT_UNVERIFIED_NEW_DERIVATION 4ULL

typedef struct {{
  uint64_t replay_case_index;
  uint64_t case_kind_code;
  uint64_t result_code;
  uint64_t backfill_transfer_index;
  uint64_t family_mask;
  uint64_t candidate_form_count;
  uint64_t source_secret_count;
  uint64_t consistent;
  uint64_t derived;
  uint64_t derived_secret;
  uint64_t rank;
  uint64_t accepted_backfill_export_count;
  uint64_t case_hash_u64;
}} selected13_backfill_replay_case_t;

static const selected13_backfill_replay_case_t SELECTED13_BACKFILL_REPLAY_CASES[] = {{
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
  uint64_t mask_lane_count = 0;
  uint64_t tier_group_count = 0;
  uint64_t transfer_gate_count = 0;
  uint64_t source_secret_only_count = 0;
  uint64_t inconsistent_count = 0;
  uint64_t no_unique_count = 0;
  uint64_t unverified_new_derivation_count = 0;
  uint64_t accepted_export_count = 0;
  const size_t case_count =
      sizeof(SELECTED13_BACKFILL_REPLAY_CASES) / sizeof(SELECTED13_BACKFILL_REPLAY_CASES[0]);
  if (case_count != SELECTED13_BACKFILL_REPLAY_CASE_COUNT) failure_count++;
  if (SELECTED13_BACKFILL_REPLAY_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;

  for (size_t i = 0; i < case_count; i++) {{
    const selected13_backfill_replay_case_t *entry = &SELECTED13_BACKFILL_REPLAY_CASES[i];
    if (entry->replay_case_index != i) failure_count++;
    if (entry->candidate_form_count == 0 || entry->case_hash_u64 == 0) failure_count++;
    if (entry->accepted_backfill_export_count != 0ULL) failure_count++;
    accepted_export_count += entry->accepted_backfill_export_count;
    if (entry->case_kind_code == SELECTED13_REPLAY_CASE_MASK_LANE) {{
      mask_lane_count++;
    }} else if (entry->case_kind_code == SELECTED13_REPLAY_CASE_TIER_GROUP) {{
      tier_group_count++;
    }} else if (entry->case_kind_code == SELECTED13_REPLAY_CASE_TRANSFER_GATE) {{
      transfer_gate_count++;
    }} else {{
      failure_count++;
    }}
    if (entry->result_code == SELECTED13_REPLAY_RESULT_SOURCE_SECRET_ONLY) {{
      source_secret_only_count++;
    }} else if (entry->result_code == SELECTED13_REPLAY_RESULT_INCONSISTENT) {{
      inconsistent_count++;
    }} else if (entry->result_code == SELECTED13_REPLAY_RESULT_NO_UNIQUE_SECRET) {{
      no_unique_count++;
    }} else if (entry->result_code == SELECTED13_REPLAY_RESULT_UNVERIFIED_NEW_DERIVATION) {{
      unverified_new_derivation_count++;
    }} else {{
      failure_count++;
    }}
  }}

  if (mask_lane_count != {EXPECTED_MASK_LANE_CASES}ULL) failure_count++;
  if (tier_group_count != {EXPECTED_TIER_GROUP_CASES}ULL) failure_count++;
  if (transfer_gate_count != {EXPECTED_TRANSFER_GATE_CASES}ULL) failure_count++;
  if (accepted_export_count != 0ULL) failure_count++;
  if (unverified_new_derivation_count != 0ULL) failure_count++;

  printf("{{");
  printf("\\\"case_count\\\":%llu,", (unsigned long long)case_count);
  printf("\\\"mask_lane_count\\\":%llu,", (unsigned long long)mask_lane_count);
  printf("\\\"tier_group_count\\\":%llu,", (unsigned long long)tier_group_count);
  printf("\\\"transfer_gate_count\\\":%llu,", (unsigned long long)transfer_gate_count);
  printf("\\\"source_secret_only_count\\\":%llu,", (unsigned long long)source_secret_only_count);
  printf("\\\"inconsistent_count\\\":%llu,", (unsigned long long)inconsistent_count);
  printf("\\\"no_unique_count\\\":%llu,", (unsigned long long)no_unique_count);
  printf("\\\"unverified_new_derivation_count\\\":%llu,", (unsigned long long)unverified_new_derivation_count);
  printf("\\\"accepted_export_count\\\":%llu,", (unsigned long long)accepted_export_count);
  printf("\\\"failure_count\\\":%llu", (unsigned long long)failure_count);
  printf("}}\\n");
  return failure_count == 0 ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path, compiler: str) -> dict[str, Any]:
    c_source = render_preflight_c(header_path.name)
    source_hash = hashlib.sha256(c_source.encode("utf-8")).hexdigest()
    temp_root = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path(tempfile.gettempdir())
    env = os.environ.copy()
    env["TMPDIR"] = str(temp_root)
    with tempfile.TemporaryDirectory(prefix="ecdlp_selected13_replay_algebra_", dir=str(temp_root)) as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "selected13_backfill_replay_algebra.c"
        exe_path = temp_path / "selected13_backfill_replay_algebra"
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
                "c_source_sha256": source_hash,
                "compile_command": command,
                "compile_returncode": compile_run.returncode,
                "compile_stderr": compile_run.stderr,
                "compile_stdout": compile_run.stdout,
                "compiled": False,
                "executed": False,
            }
        native_run = subprocess.run([str(exe_path)], capture_output=True, text=True, check=False, env=env)
        try:
            native_summary = json.loads(native_run.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            native_summary = None
        return {
            "c_source_sha256": source_hash,
            "compile_command": command,
            "compile_returncode": compile_run.returncode,
            "compile_stderr": compile_run.stderr,
            "compile_stdout": compile_run.stdout,
            "compiled": True,
            "executed": True,
            "native_summary": native_summary,
            "run_returncode": native_run.returncode,
            "run_stderr": native_run.stderr,
            "run_stdout": native_run.stdout,
        }


def compare_native(native: dict[str, Any]) -> list[dict[str, Any]]:
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
    if as_int(summary.get("failure_count"), -1) != 0:
        failures.append({"code": "native_preflight_reported_failures", "native_summary": summary})
    return failures


def summarize(cases: list[dict[str, Any]], native: dict[str, Any]) -> dict[str, Any]:
    result_counts = Counter(str(case.get("result_status")) for case in cases)
    case_counts = Counter(str(case.get("case_kind")) for case in cases)
    native_summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    transfer_cases = [case for case in cases if case.get("case_kind") == "transfer_gate"]
    return {
        "accepted_backfill_export_count": sum(as_int(case.get("accepted_backfill_export_count")) for case in cases),
        "case_kind_counts": dict(sorted(case_counts.items())),
        "native_preflight_failure_count": as_int(native_summary.get("failure_count"), -1),
        "native_preflight_verified": as_int(native_summary.get("failure_count"), -1) == 0,
        "primary_backfill_transfers": PRIMARY_BACKFILL_TRANSFERS,
        "replay_case_count": len(cases),
        "result_counts": dict(sorted(result_counts.items())),
        "transfer_gate_statuses": [
            {
                "backfill_transfer_index": as_int(case.get("backfill_transfer_index"), -1),
                "candidate_form_count": as_int(case.get("candidate_form_count")),
                "result_status": case.get("result_status"),
                "source_secret_set": case.get("source_secret_set"),
            }
            for case in transfer_cases
        ],
        "unverified_new_linear_derivation_count": result_counts.get("UNVERIFIED_NEW_LINEAR_DERIVATION", 0),
        "worker_interpretation": (
            "The bound sidecar forms replay only source secrets or inconsistent source mixes; "
            "no accepted backfill export is produced without fresh FFE/summation-polynomial row emission."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill-gate", type=Path, default=DEFAULT_BACKFILL_GATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--no-c-header", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gate = load_json(args.backfill_gate)
    failures = validate_inputs(gate)
    cases = build_replay_cases(gate)
    failures.extend(validate_cases(cases))

    native: dict[str, Any] = {"compiled": False, "executed": False, "native_summary": {}}
    if not args.no_c_header:
        args.c_header_out.parent.mkdir(parents=True, exist_ok=True)
        args.c_header_out.write_text(render_c_header(cases))
        native = run_native_preflight(args.c_header_out, args.cc)
        failures.extend(compare_native(native))

    verified = not failures
    payload = {
        "artifacts": {
            "backfill_gate": str(args.backfill_gate),
            "c_header": None if args.no_c_header else str(args.c_header_out),
        },
        "claim_status": (
            "FFE_SHARP_LANE_BACKFILL_REPLAY_ALGEBRA_READY"
            if verified
            else "FFE_SHARP_LANE_BACKFILL_REPLAY_ALGEBRA_FAILED_CHECK"
        ),
        "created_at": now_iso(),
        "failures": failures,
        "honesty_boundary": [
            "This tests source coefficient replay algebra only.",
            "Source-secret replay or source-form inconsistency is not a backfill direct/rank export.",
            "A promoted result still needs fresh FFE/summation-polynomial row emission and public-key verification against the backfill row hash.",
        ],
        "native_preflight": native,
        "parameters": {
            "coefficient_width": COEFF_COUNT,
            "modulus": ORDER,
            "primary_backfill_transfers": PRIMARY_BACKFILL_TRANSFERS,
        },
        "replay_cases": cases,
        "schema": SCHEMA,
        "summary": summarize(cases, native),
    }
    payload["summary"]["verified"] = verified
    write_json(args.out, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "failures": failures, "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
