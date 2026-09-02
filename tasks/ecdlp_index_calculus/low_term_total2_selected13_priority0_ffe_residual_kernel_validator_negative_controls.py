#!/usr/bin/env python3
"""Run negative controls for the selected13 priority-0 result validator.

The result validator is the acceptance boundary for external FFE/summation
kernel output.  This harness feeds it deliberately invalid result JSONs:
template placeholders, token-only evidence shells, missing tokens, wrong target
token, wrong row id, and a source-replay-like non-export.  Every case must be
rejected without crashing.

This is a negative-control artifact only.  It does not evaluate summation
polynomials, emit a target direct/rank row, solve ECDLP, or claim a Pollard-rho
speedup.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_ffe_residual_kernel_validator_negative_controls.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_VALIDATOR = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_result_validator_10376_probe.json"
)
DEFAULT_RESULT_GATE = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_result_gate_10376_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_validator_negative_controls_10376_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_validator_negative_controls_10376_probe.h"
DEFAULT_VALIDATOR_SCRIPT = (
    WORKTREE_ROOT / "tasks/ecdlp_index_calculus/low_term_total2_selected13_priority0_ffe_residual_kernel_result_validator.py"
)

EXPECTED_TRANSFER = 10376
EXPECTED_SECRET = 5859
EXPECTED_TARGET_TOKEN = 4590949340060852637
EXPECTED_NEGATIVE_CASE_COUNT = 6


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


def stable_hash_u64(material: Any) -> int:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def template(validator: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(validator.get("expected_kernel_result_template") or {})


def negative_cases(validator: dict[str, Any]) -> list[dict[str, Any]]:
    base = template(validator)
    cases = []

    malformed_public = copy.deepcopy(base)
    cases.append(
        {
            "case_id": "malformed_template_public_key",
            "expected_failure_codes": [
                "target_public_key_not_verified",
                "group_completion_evidence_hash_missing",
                "lane_completion_evidence_hash_missing",
                "target_export_direct_rank_hash_missing",
            ],
            "kernel_result": malformed_public,
            "why_invalid": "template placeholder public_key strings must not parse as a curve point",
        }
    )

    token_only = copy.deepcopy(base)
    for item in token_only.get("group_completions") or []:
        for field in (
            "group_evidence_hash_u64",
            "residual_relation_hash_u64",
            "residual_synthesis_evaluated",
            "lane_fanout_applied",
            "hybrid_coefficients_materialized",
            "source_guard_preserved",
            "worker_evidence_kind",
        ):
            item.pop(field, None)
    for item in token_only.get("lane_completions") or []:
        for field in (
            "lane_evidence_hash_u64",
            "residual_synthesis_evaluated",
            "coefficient_materialization_evaluated",
            "source_coefficient_guard_reused",
            "worker_evidence_kind",
        ):
            item.pop(field, None)
    for field in (
        "fresh_ffe_or_summation_polynomial_evaluated",
        "group_evidence_digest_u64",
        "kernel_contract_packet_hash_u64",
        "lane_evidence_digest_u64",
        "result_gate_packet_hash_u64",
        "target_direct_rank_export_evaluated",
        "target_direct_rank_export_hash_u64",
        "target_gate_packet_hash_u64",
        "worker_evidence_kind",
    ):
        token_only.setdefault("target_export", {}).pop(field, None)
    cases.append(
        {
            "case_id": "token_only_no_evidence_contract",
            "expected_failure_codes": [
                "group_completion_evidence_hash_missing",
                "group_completion_residual_relation_hash_missing",
                "lane_completion_evidence_hash_missing",
                "target_export_group_evidence_digest_mismatch",
                "target_export_direct_rank_hash_missing",
            ],
            "kernel_result": token_only,
            "why_invalid": "all completion tokens are present but no FFE/summation evidence is bound to them",
        }
    )

    missing_group = copy.deepcopy(base)
    missing_group["group_completions"] = missing_group.get("group_completions", [])[:-1]
    cases.append(
        {
            "case_id": "missing_group_completion_token",
            "expected_failure_codes": [
                "group_completion_token_set_mismatch",
                "accepted_group_count_unexpected",
                "target_export_group_evidence_digest_mismatch",
            ],
            "kernel_result": missing_group,
            "why_invalid": "one fused term group token is absent",
        }
    )

    wrong_token = copy.deepcopy(base)
    wrong_token.setdefault("target_export", {})["completion_token_hash_u64"] = EXPECTED_TARGET_TOKEN + 1
    cases.append(
        {
            "case_id": "wrong_target_export_token",
            "expected_failure_codes": ["target_export_token_mismatch", "target_export_group_evidence_digest_mismatch"],
            "kernel_result": wrong_token,
            "why_invalid": "target export token does not match the result gate",
        }
    )

    wrong_row = copy.deepcopy(base)
    wrong_row["row_request_id"] = f"{wrong_row.get('row_request_id')}_wrong"
    cases.append(
        {
            "case_id": "wrong_row_request_id",
            "expected_failure_codes": ["kernel_result_row_request_mismatch", "target_export_group_evidence_digest_mismatch"],
            "kernel_result": wrong_row,
            "why_invalid": "result is bound to the wrong target row request",
        }
    )

    source_replay = copy.deepcopy(base)
    source_replay["target_export"] = {
        "accepted": False,
        "completion_token_hash_u64": EXPECTED_TARGET_TOKEN,
        "derived_secret": EXPECTED_SECRET,
        "public_key": None,
        "relation_derived_ecdlp": False,
        "source_replay_only": True,
    }
    cases.append(
        {
            "case_id": "source_replay_without_target_export",
            "expected_failure_codes": [
                "target_public_key_not_verified",
                "target_export_not_accepted",
                "target_relation_not_derived",
                "target_export_source_replay_flag_set",
            ],
            "kernel_result": source_replay,
            "why_invalid": "source replay and pending target export cannot satisfy the target gate",
        }
    )
    return cases


def run_validator_case(
    validator_script: Path,
    result_gate: Path,
    case: dict[str, Any],
    tmp_path: Path,
    compiler: str,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    kernel_result_path = tmp_path / f"{case_id}.json"
    validator_out = tmp_path / f"{case_id}_validator.json"
    validator_header = tmp_path / f"{case_id}_validator.h"
    write_json(kernel_result_path, case["kernel_result"])
    command = [
        "python3",
        str(validator_script),
        "--result-gate",
        str(result_gate),
        "--kernel-result",
        str(kernel_result_path),
        "--out",
        str(validator_out),
        "--c-header-out",
        str(validator_header),
        "--cc",
        compiler,
    ]
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    parsed: dict[str, Any] | None = None
    parse_error = None
    if validator_out.is_file():
        try:
            parsed = load_json(validator_out)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
    observed_codes = sorted(
        {
            str(item.get("code"))
            for item in (parsed or {}).get("failures") or []
            if isinstance(item, dict) and item.get("code")
        }
    )
    expected_codes = sorted(str(item) for item in case.get("expected_failure_codes") or [])
    rejected = bool(parsed) and (parsed.get("claim_status") == "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_VALIDATOR_FAILED")
    crashed = parsed is None
    expected_code_coverage = all(code in observed_codes for code in expected_codes)
    relation_derived = bool(((parsed or {}).get("summary") or {}).get("relation_derived_ecdlp"))
    accepted_export_count = as_int(((parsed or {}).get("summary") or {}).get("accepted_relation_export_count"))
    return {
        "case_hash_u64": stable_hash_u64(case),
        "case_id": case_id,
        "command": command,
        "expected_failure_codes": expected_codes,
        "expected_failure_codes_covered": expected_code_coverage,
        "kernel_result_path": str(kernel_result_path),
        "observed_failure_codes": observed_codes,
        "parse_error": parse_error,
        "relation_derived_ecdlp": relation_derived,
        "rejected": rejected,
        "returncode": run.returncode,
        "stderr": run.stderr.strip(),
        "stdout": run.stdout.strip(),
        "validator_crashed_or_missed_output": crashed,
        "validator_out": str(validator_out),
        "accepted_relation_export_count": accepted_export_count,
        "why_invalid": case.get("why_invalid"),
    }


def validate_sources(validator: dict[str, Any], result_gate: dict[str, Any], validator_script: Path) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if validator.get("claim_status") != "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_VALIDATOR_AWAITING_KERNEL_OUTPUT":
        failures.append({"code": "validator_not_awaiting_output", "claim_status": validator.get("claim_status")})
    if (validator.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "validator_summary_not_verified", "summary": validator.get("summary")})
    if (validator.get("native_preflight") or {}).get("verified") is not True:
        failures.append({"code": "validator_native_preflight_not_verified"})
    if result_gate.get("claim_status") != "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_GATE_READY":
        failures.append({"code": "result_gate_not_ready", "claim_status": result_gate.get("claim_status")})
    if as_int((result_gate.get("target_export_completion_gate") or {}).get("completion_token_hash_u64")) != EXPECTED_TARGET_TOKEN:
        failures.append({"code": "target_token_unexpected", "target_export_completion_gate": result_gate.get("target_export_completion_gate")})
    if not validator_script.is_file():
        failures.append({"code": "validator_script_missing", "validator_script": str(validator_script)})
    return failures


def validate_case_results(case_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(case_results) != EXPECTED_NEGATIVE_CASE_COUNT:
        failures.append({"code": "negative_case_count_unexpected", "observed": len(case_results)})
    for result in case_results:
        if result.get("validator_crashed_or_missed_output"):
            failures.append({"code": "negative_control_validator_crashed", "case_id": result.get("case_id")})
        if result.get("rejected") is not True:
            failures.append({"code": "negative_control_not_rejected", "case_id": result.get("case_id")})
        if result.get("expected_failure_codes_covered") is not True:
            failures.append(
                {
                    "code": "negative_control_expected_codes_missing",
                    "case_id": result.get("case_id"),
                    "expected": result.get("expected_failure_codes"),
                    "observed": result.get("observed_failure_codes"),
                }
            )
        if result.get("relation_derived_ecdlp") or as_int(result.get("accepted_relation_export_count")):
            failures.append({"code": "negative_control_claimed_relation", "case_id": result.get("case_id")})
    return failures


def render_c_header(case_results: list[dict[str, Any]]) -> str:
    case_lines = []
    for index, result in enumerate(case_results):
        case_lines.append(
            "  {"
            f"{index}ULL, "
            f"{as_int(result.get('case_hash_u64'))}ULL, "
            f"{1 if result.get('rejected') else 0}ULL, "
            f"{1 if result.get('validator_crashed_or_missed_output') else 0}ULL, "
            f"{1 if result.get('relation_derived_ecdlp') else 0}ULL, "
            f"{as_int(result.get('accepted_relation_export_count'))}ULL, "
            f"{1 if result.get('expected_failure_codes_covered') else 0}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_VALIDATOR_NEGATIVE_CONTROLS_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_VALIDATOR_NEGATIVE_CONTROLS_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_TRANSFER {EXPECTED_TRANSFER}ULL
#define SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_CASE_COUNT {len(case_results)}ULL
#define SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_REJECTED_COUNT {sum(1 for item in case_results if item.get('rejected'))}ULL
#define SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_CRASH_COUNT {sum(1 for item in case_results if item.get('validator_crashed_or_missed_output'))}ULL
#define SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_RELATION_DERIVED_COUNT {sum(1 for item in case_results if item.get('relation_derived_ecdlp'))}ULL
#define SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_ACCEPTED_EXPORT_COUNT {sum(as_int(item.get('accepted_relation_export_count')) for item in case_results)}ULL
#define SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_CODE_COVERAGE_COUNT {sum(1 for item in case_results if item.get('expected_failure_codes_covered'))}ULL

typedef struct {{
  uint64_t case_index;
  uint64_t case_hash_u64;
  uint64_t rejected;
  uint64_t crashed;
  uint64_t relation_derived_ecdlp;
  uint64_t accepted_relation_export_count;
  uint64_t expected_failure_codes_covered;
}} selected13_priority0_negative_control_case_t;

static const selected13_priority0_negative_control_case_t SELECTED13_PRIORITY0_NEGATIVE_CONTROL_CASES[] = {{
{chr(10).join(case_lines)}
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
  uint64_t rejected_count = 0;
  uint64_t crash_count = 0;
  uint64_t relation_count = 0;
  uint64_t accepted_export_count = 0;
  uint64_t coverage_count = 0;
  const size_t case_count = sizeof(SELECTED13_PRIORITY0_NEGATIVE_CONTROL_CASES) / sizeof(SELECTED13_PRIORITY0_NEGATIVE_CONTROL_CASES[0]);
  if (SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_TRANSFER != {EXPECTED_TRANSFER}ULL) failure_count++;
  if (case_count != SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_CASE_COUNT) failure_count++;
  if (case_count != {EXPECTED_NEGATIVE_CASE_COUNT}ULL) failure_count++;
  for (size_t i = 0; i < case_count; i++) {{
    const selected13_priority0_negative_control_case_t *item = &SELECTED13_PRIORITY0_NEGATIVE_CONTROL_CASES[i];
    if (item->case_hash_u64 == 0ULL) failure_count++;
    rejected_count += item->rejected;
    crash_count += item->crashed;
    relation_count += item->relation_derived_ecdlp;
    accepted_export_count += item->accepted_relation_export_count;
    coverage_count += item->expected_failure_codes_covered;
  }}
  if (rejected_count != SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_REJECTED_COUNT) failure_count++;
  if (crash_count != SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_CRASH_COUNT) failure_count++;
  if (relation_count != SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_RELATION_DERIVED_COUNT) failure_count++;
  if (accepted_export_count != SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_ACCEPTED_EXPORT_COUNT) failure_count++;
  if (coverage_count != SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_CODE_COVERAGE_COUNT) failure_count++;
  if (rejected_count != {EXPECTED_NEGATIVE_CASE_COUNT}ULL) failure_count++;
  if (crash_count != 0ULL) failure_count++;
  if (relation_count != 0ULL) failure_count++;
  if (accepted_export_count != 0ULL) failure_count++;
  if (coverage_count != {EXPECTED_NEGATIVE_CASE_COUNT}ULL) failure_count++;
  printf("selected13_priority0_ffe_residual_kernel_validator_negative_controls_preflight transfer=%llu cases=%llu rejected=%llu crashes=%llu accepted=%llu code_coverage=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_NEGATIVE_CONTROLS_TRANSFER,
         (unsigned long long)case_count,
         (unsigned long long)rejected_count,
         (unsigned long long)crash_count,
         (unsigned long long)accepted_export_count,
         (unsigned long long)coverage_count,
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
    with tempfile.TemporaryDirectory(prefix="selected13_priority0_negative_controls_preflight_", dir=str(temp_root)) as tmp:
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


def summarize(case_results: list[dict[str, Any]], failures: list[dict[str, Any]], native_preflight: dict[str, Any]) -> dict[str, Any]:
    rejected = sum(1 for item in case_results if item.get("rejected"))
    crashes = sum(1 for item in case_results if item.get("validator_crashed_or_missed_output"))
    accepted = sum(as_int(item.get("accepted_relation_export_count")) for item in case_results)
    code_coverage = sum(1 for item in case_results if item.get("expected_failure_codes_covered"))
    return {
        "accepted_relation_export_count": accepted,
        "case_count": len(case_results),
        "code_coverage_count": code_coverage,
        "crash_count": crashes,
        "failure_count": len(failures),
        "native_preflight_verified": bool(native_preflight.get("verified")),
        "pollard_rho_speedup_claimed": False,
        "rejected_count": rejected,
        "relation_derived_ecdlp": False,
        "verified": not failures,
        "worker_interpretation": (
            "The negative controls prove the result validator rejects malformed template output, "
            "missing completion tokens, wrong target tokens, wrong row ids, and source-replay-only "
            "results without accepting a relation export; it also rejects token-only shells that "
            "lack FFE/summation evidence binding."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator", default=str(DEFAULT_VALIDATOR))
    parser.add_argument("--result-gate", default=str(DEFAULT_RESULT_GATE))
    parser.add_argument("--validator-script", default=str(DEFAULT_VALIDATOR_SCRIPT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validator_path = Path(args.validator)
    result_gate_path = Path(args.result_gate)
    validator_script = Path(args.validator_script)
    validator = load_json(validator_path)
    result_gate = load_json(result_gate_path)
    failures = validate_sources(validator, result_gate, validator_script)
    temp_root = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path(tempfile.gettempdir())
    case_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="selected13_priority0_negative_controls_", dir=str(temp_root)) as tmp:
        tmp_path = Path(tmp)
        for case in negative_cases(validator):
            case_results.append(run_validator_case(validator_script, result_gate_path, case, tmp_path, args.cc))
    failures.extend(validate_case_results(case_results))
    payload = {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_VALIDATOR_NEGATIVE_CONTROLS_READY"
            if not failures
            else "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_VALIDATOR_NEGATIVE_CONTROLS_FAILED"
        ),
        "parameters": {
            "result_gate": str(result_gate_path),
            "validator": str(validator_path),
            "validator_script": str(validator_script),
        },
        "packet_hash_u64": stable_hash_u64({"validator": validator.get("packet_hash_u64"), "case_results": case_results}),
        "case_results": case_results,
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "general_ecdlp_algorithm_claimed": False,
            "negative_controls_only": True,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "summation_polynomial_evaluated": False,
        },
        "failures": failures,
    }
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(case_results))
    payload["artifacts"] = {"c_header": str(header_path)}
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_VALIDATOR_NEGATIVE_CONTROLS_FAILED"
    payload["summary"] = summarize(payload["case_results"], payload["failures"], native_preflight)
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
