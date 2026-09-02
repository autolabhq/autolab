#!/usr/bin/env python3
"""Compile and run a native C reference for fused first-pass field replays.

The field replay artifact is verifier-independent, but it is still executed by
Python.  This probe lowers the same first-pass candidate-point instruction
stream into a standalone C reference program, compiles it, runs it, and checks
that the native modular arithmetic reproduces every recorded register value and
candidate-point output.

This is not an optimized implementation yet.  It is the native/FFE promotion
gate: a future hand-written or generated kernel must match this C reference and
then feed the unchanged affine, ABI, contract, and second-pass relation gates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_FIELD_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_field_replay_target67_672_744.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_native_reference.json"

OP_IDS = {
    "add": "OP_ADD",
    "sub": "OP_SUB",
    "mul": "OP_MUL",
    "neg": "OP_NEG",
    "inv": "OP_INV",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_field(raw: Any) -> int:
    match = re.fullmatch(r"GF\((\d+)\)", str(raw or ""))
    if match is None:
        raise ValueError(f"field must be GF(p), got {raw!r}")
    return int(match.group(1))


def c_string(value: Any) -> str:
    return json.dumps(str(value))


def compact_event_key(raw: Any) -> str:
    values = list(raw or [])
    if len(values) != 3:
        return "?:-1:-1"
    return f"{values[0]}:{as_int(values[1], -1)}:{as_int(values[2], -1)}"


def point_pair(raw: Any) -> tuple[int, int]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"point must be [x,y], got {raw!r}")
    return as_int(raw[0]), as_int(raw[1])


def instruction_arg(instruction: dict[str, Any], index: int) -> int:
    args = instruction.get("args") or []
    if index >= len(args):
        return 0
    return as_int(args[index])


def out_kind(raw: Any) -> int:
    name = str(raw or "")
    if name == "x3":
        return 1
    if name == "y3":
        return 2
    return 0


def collect_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for record in records:
        source_name = str(record.get("source_name") or "")
        for group in record.get("trace_groups") or []:
            if not isinstance(group, dict):
                continue
            replay = group.get("representative_kernel_replay") or {}
            instructions = replay.get("instruction_stream") or []
            if not instructions:
                continue
            result = group.get("replayed_candidate_point") or replay.get("result")
            expected_x, expected_y = point_pair(result)
            cases.append(
                {
                    "source_name": source_name,
                    "event_key": compact_event_key(group.get("event_key")),
                    "fanout": as_int(group.get("fanout")),
                    "candidate_point_reused": bool(group.get("candidate_point_reused")),
                    "p": parse_field(replay.get("field")),
                    "expected_x": expected_x,
                    "expected_y": expected_y,
                    "instructions": instructions,
                }
            )
    return cases


def render_c_source(cases: list[dict[str, Any]]) -> str:
    instruction_arrays = []
    case_entries = []
    for case_index, case in enumerate(cases):
        rows = []
        for instruction in case["instructions"]:
            op = str(instruction.get("op") or "")
            if op not in OP_IDS:
                raise ValueError(f"unsupported op {op!r}")
            rows.append(
                "  {"
                f"{OP_IDS[op]}, "
                f"{instruction_arg(instruction, 0)}ULL, "
                f"{instruction_arg(instruction, 1)}ULL, "
                f"{as_int(instruction.get('value'))}ULL, "
                f"{out_kind(instruction.get('out'))}"
                "},"
            )
        instruction_arrays.append(
            f"static const instr_t instr_{case_index}[] = {{\n" + "\n".join(rows) + "\n};"
        )
        case_entries.append(
            "  {"
            f"{c_string(case['source_name'])}, "
            f"{c_string(case['event_key'])}, "
            f"{as_int(case['fanout'])}, "
            f"{1 if case['candidate_point_reused'] else 0}, "
            f"{as_int(case['p'])}ULL, "
            f"{as_int(case['expected_x'])}ULL, "
            f"{as_int(case['expected_y'])}ULL, "
            f"instr_{case_index}, "
            f"sizeof(instr_{case_index}) / sizeof(instr_{case_index}[0])"
            "},"
        )

    return """#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef enum {
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_NEG,
  OP_INV
} op_t;

typedef struct {
  op_t op;
  uint64_t a;
  uint64_t b;
  uint64_t expected;
  int out_kind;
} instr_t;

typedef struct {
  const char *source_name;
  const char *event_key;
  int fanout;
  int reused;
  uint64_t p;
  uint64_t expected_x;
  uint64_t expected_y;
  const instr_t *instructions;
  size_t instruction_count;
} case_t;

static uint64_t mod_add(uint64_t a, uint64_t b, uint64_t p) {
  return (a % p + b % p) % p;
}

static uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t p) {
  return (a % p + p - (b % p)) % p;
}

static uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t p) {
  return ((a % p) * (b % p)) % p;
}

static uint64_t mod_neg(uint64_t a, uint64_t p) {
  return (p - (a % p)) % p;
}

static uint64_t mod_inv(uint64_t a, uint64_t p, int *ok) {
  int64_t t = 0;
  int64_t new_t = 1;
  int64_t r = (int64_t)p;
  int64_t new_r = (int64_t)(a % p);

  while (new_r != 0) {
    int64_t q = r / new_r;
    int64_t next_t = t - q * new_t;
    int64_t next_r = r - q * new_r;
    t = new_t;
    new_t = next_t;
    r = new_r;
    new_r = next_r;
  }
  if (r != 1) {
    *ok = 0;
    return 0;
  }
  if (t < 0) {
    t += (int64_t)p;
  }
  *ok = 1;
  return (uint64_t)t;
}

""" + "\n\n".join(instruction_arrays) + """

static const case_t cases[] = {
""" + "\n".join(case_entries) + """
};

int main(void) {
  size_t case_count = sizeof(cases) / sizeof(cases[0]);
  size_t verified_count = 0;
  size_t failure_count = 0;
  size_t instruction_count = 0;
  size_t result_match_count = 0;
  size_t reused_case_count = 0;
  size_t op_add = 0;
  size_t op_sub = 0;
  size_t op_mul = 0;
  size_t op_neg = 0;
  size_t op_inv = 0;

  for (size_t i = 0; i < case_count; ++i) {
    const case_t *c = &cases[i];
    uint64_t x3 = UINT64_MAX;
    uint64_t y3 = UINT64_MAX;
    int case_failed = 0;
    if (c->reused) {
      reused_case_count += 1;
    }

    for (size_t j = 0; j < c->instruction_count; ++j) {
      const instr_t *instr = &c->instructions[j];
      uint64_t value = 0;
      int ok = 1;
      switch (instr->op) {
        case OP_ADD:
          value = mod_add(instr->a, instr->b, c->p);
          op_add += 1;
          break;
        case OP_SUB:
          value = mod_sub(instr->a, instr->b, c->p);
          op_sub += 1;
          break;
        case OP_MUL:
          value = mod_mul(instr->a, instr->b, c->p);
          op_mul += 1;
          break;
        case OP_NEG:
          value = mod_neg(instr->a, c->p);
          op_neg += 1;
          break;
        case OP_INV:
          value = mod_inv(instr->a, c->p, &ok);
          op_inv += 1;
          break;
      }

      instruction_count += 1;
      if (!ok || value != instr->expected) {
        if (!case_failed) {
          fprintf(
            stderr,
            "case failure source=%s event=%s instr=%zu expected=%" PRIu64 " got=%" PRIu64 "\\n",
            c->source_name,
            c->event_key,
            j,
            instr->expected,
            value
          );
        }
        case_failed = 1;
        failure_count += 1;
      }
      if (instr->out_kind == 1) {
        x3 = value;
      } else if (instr->out_kind == 2) {
        y3 = value;
      }
    }

    if (x3 == c->expected_x && y3 == c->expected_y) {
      result_match_count += 1;
    } else {
      if (!case_failed) {
        fprintf(
          stderr,
          "result failure source=%s event=%s expected=(%" PRIu64 ",%" PRIu64 ") got=(%" PRIu64 ",%" PRIu64 ")\\n",
          c->source_name,
          c->event_key,
          c->expected_x,
          c->expected_y,
          x3,
          y3
        );
      }
      case_failed = 1;
      failure_count += 1;
    }

    if (!case_failed) {
      verified_count += 1;
    }
  }

  printf(
    "{"
    "\\"case_count\\":%zu,"
    "\\"verified_count\\":%zu,"
    "\\"failure_count\\":%zu,"
    "\\"instruction_count\\":%zu,"
    "\\"result_match_count\\":%zu,"
    "\\"reused_case_count\\":%zu,"
    "\\"field_op_counts\\":{"
    "\\"add\\":%zu,"
    "\\"sub\\":%zu,"
    "\\"mul\\":%zu,"
    "\\"neg\\":%zu,"
    "\\"inv\\":%zu,"
    "\\"total\\":%zu"
    "}"
    "}\\n",
    case_count,
    verified_count,
    failure_count,
    instruction_count,
    result_match_count,
    reused_case_count,
    op_add,
    op_sub,
    op_mul,
    op_neg,
    op_inv,
    instruction_count
  );

  return failure_count == 0 ? 0 : 1;
}
"""


def compiler_command(compiler: str, c_path: Path, exe_path: Path) -> list[str]:
    return [
        compiler,
        "-std=c99",
        "-O2",
        "-Wall",
        "-Wextra",
        str(c_path),
        "-o",
        str(exe_path),
    ]


def run_native_reference(
    c_source: str,
    c_out: Path,
    compiler: str,
    keep_exe: bool,
) -> dict[str, Any]:
    c_out.parent.mkdir(parents=True, exist_ok=True)
    c_out.write_text(c_source)
    temp_root = Path("/private/tmp")
    exe_parent = c_out.parent if keep_exe else Path(tempfile.mkdtemp(prefix="ecdlp_native_ref_", dir=temp_root))
    exe_path = exe_parent / f"{c_out.stem}.bin"
    command = compiler_command(compiler, c_out, exe_path)
    compile_env = os.environ.copy()
    compile_env["TMPDIR"] = str(temp_root)
    compile_run = subprocess.run(command, capture_output=True, text=True, check=False, env=compile_env)
    if compile_run.returncode != 0:
        return {
            "compiled": False,
            "compile_command": command,
            "compile_returncode": compile_run.returncode,
            "compile_stdout": compile_run.stdout,
            "compile_stderr": compile_run.stderr,
            "executed": False,
            "executable": str(exe_path),
        }

    native_run = subprocess.run([str(exe_path)], capture_output=True, text=True, check=False, env=compile_env)
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
        "executable": str(exe_path),
    }


def source_record_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for record in records:
        groups = record.get("trace_groups") or []
        instruction_count = sum(
            as_int(((group.get("representative_kernel_replay") or {}).get("field_op_counts") or {}).get("total"))
            for group in groups
            if isinstance(group, dict)
        )
        summaries.append(
            {
                "source_name": record.get("source_name"),
                "public_group_key": record.get("public_group_key"),
                "native_reference_status": "pending_global_native_result",
                "field_replay_status": record.get("field_replay_status"),
                "case_count": len(groups),
                "instruction_count": instruction_count,
                "reused_case_count": sum(1 for group in groups if bool(group.get("candidate_point_reused"))),
                "matches_contract_below_rho": bool((record.get("checks") or {}).get("matches_contract_below_rho")),
                "matches_contract_event_reuse_target": bool(
                    (record.get("checks") or {}).get("matches_contract_event_reuse_target")
                ),
            }
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-source", type=Path, default=DEFAULT_FIELD_SOURCE)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-out", type=Path)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--keep-exe", action="store_true")
    args = parser.parse_args()

    source = load_json(args.field_source)
    wanted = set(args.source_name or [])
    records = []
    for record in source.get("records") or []:
        if not isinstance(record, dict):
            continue
        source_name = str(record.get("source_name") or "")
        if wanted and source_name not in wanted:
            continue
        records.append(record)

    cases = collect_cases(records)
    c_source = render_c_source(cases)
    c_out = args.c_out or args.out.with_suffix(".c")
    native = run_native_reference(c_source, c_out, args.cc, args.keep_exe)
    native_summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    global_ok = bool(
        native.get("compiled")
        and native.get("executed")
        and as_int(native.get("run_returncode"), -1) == 0
        and as_int(native_summary.get("case_count"), -1) == len(cases)
        and as_int(native_summary.get("verified_count"), -1) == len(cases)
        and as_int(native_summary.get("failure_count"), -1) == 0
    )
    record_summaries = source_record_summaries(records)
    for summary in record_summaries:
        summary["native_reference_status"] = (
            "native_reference_verified" if global_ok else "native_reference_failed_check"
        )

    below = [record for record in record_summaries if bool(record.get("matches_contract_below_rho"))]
    event_reuse = [
        record for record in below if bool(record.get("matches_contract_event_reuse_target"))
    ]
    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_kernel_native_reference_probe_v1",
        "method": "compiled_c_reference_for_first_pass_finite_field_instruction_stream",
        "parameters": {
            "field_source": str(args.field_source),
            "source_names": sorted(wanted),
            "compiler": args.cc,
            "c_out": str(c_out),
            "keep_exe": bool(args.keep_exe),
        },
        "summary": {
            "record_count": len(records),
            "native_reference_verified_count": len(records) if global_ok else 0,
            "below_rho_native_reference_verified_count": len(below) if global_ok else 0,
            "event_reuse_below_rho_native_reference_verified_count": len(event_reuse) if global_ok else 0,
            "native_case_count": len(cases),
            "native_verified_case_count": as_int(native_summary.get("verified_count")),
            "native_failure_count": as_int(native_summary.get("failure_count"), -1),
            "native_instruction_count": as_int(native_summary.get("instruction_count")),
            "native_result_match_count": as_int(native_summary.get("result_match_count")),
            "native_reused_case_count": as_int(native_summary.get("reused_case_count")),
            "native_field_op_counts": native_summary.get("field_op_counts"),
            "compiled": bool(native.get("compiled")),
            "executed": bool(native.get("executed")),
            "global_native_reference_verified": global_ok,
            "interpretation": (
                "The first-pass candidate-point instruction stream has been "
                "compiled into and verified by a standalone C reference kernel."
            ),
        },
        "records": record_summaries,
        "native_execution": native,
        "non_claims": [
            "This is a generated C reference for the first-pass field instruction stream, not an optimized kernel.",
            "It does not search for new relations and does not replace the second-pass relation predicate layer.",
            "A production FFE/summation-polynomial implementation must still match this C reference and pass the ABI/contract gates.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0 if global_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
