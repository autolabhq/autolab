#!/usr/bin/env python3
"""Compile and run a compact native kernel for fused candidate points.

The native reference probe compiled static field-instruction records.  This
probe goes one step closer to a real implementation: it consumes affine traces,
emits operand arrays, and verifies a reusable C kernel function that computes
the generalized-Weierstrass affine-add registers directly.

The generated C is still a reference harness, not an optimized FFE kernel.  It
is intentionally shaped like a promotion gate for that kernel: operands in,
register stream and candidate point out, then compare against the established
field/native-reference artifacts.
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
DEFAULT_AFFINE_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_affine_trace_target67_672_744.json"
)
DEFAULT_NATIVE_REFERENCE_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_native_reference_target67_672_744.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_native_kernel.json"


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


def op_kind(raw: Any) -> int:
    value = str(raw or "")
    if value == "addition":
        return 0
    if value == "doubling":
        return 1
    raise ValueError(f"unsupported operation {value!r}")


def collect_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for record in records:
        source_name = str(record.get("source_name") or "")
        for group in record.get("trace_groups") or []:
            if not isinstance(group, dict):
                continue
            trace = group.get("representative_affine_trace") or {}
            if str(trace.get("operation") or "") not in {"addition", "doubling"}:
                continue
            left_x, left_y = point_pair(trace.get("left"))
            right_x, right_y = point_pair(trace.get("right"))
            result_x, result_y = point_pair(trace.get("result"))
            ainvs = [as_int(value) for value in trace.get("ainvs_mod_p") or []]
            if len(ainvs) != 5:
                raise ValueError(f"expected five curve coefficients, got {ainvs!r}")
            cases.append(
                {
                    "source_name": source_name,
                    "event_key": compact_event_key(group.get("event_key")),
                    "fanout": as_int(group.get("fanout")),
                    "candidate_point_reused": bool(group.get("candidate_point_reused")),
                    "operation": str(trace.get("operation")),
                    "p": parse_field(trace.get("field")),
                    "ainvs": ainvs,
                    "left": (left_x, left_y),
                    "right": (right_x, right_y),
                    "expected_slope_numerator": as_int(trace.get("slope_numerator")),
                    "expected_slope_denominator": as_int(trace.get("slope_denominator")),
                    "expected_slope_denominator_inverse": as_int(trace.get("slope_denominator_inverse")),
                    "expected_slope": as_int(trace.get("slope")),
                    "expected_intercept": as_int(trace.get("intercept")),
                    "expected_x": result_x,
                    "expected_y": result_y,
                }
            )
    return cases


def render_c_source(cases: list[dict[str, Any]]) -> str:
    case_entries = []
    for case in cases:
        a1, a2, a3, a4, a6 = case["ainvs"]
        left_x, left_y = case["left"]
        right_x, right_y = case["right"]
        case_entries.append(
            "  {"
            f"{c_string(case['source_name'])}, "
            f"{c_string(case['event_key'])}, "
            f"{as_int(case['fanout'])}, "
            f"{1 if case['candidate_point_reused'] else 0}, "
            f"{op_kind(case['operation'])}, "
            f"{as_int(case['p'])}ULL, "
            f"{as_int(a1)}ULL, {as_int(a2)}ULL, {as_int(a3)}ULL, {as_int(a4)}ULL, {as_int(a6)}ULL, "
            f"{as_int(left_x)}ULL, {as_int(left_y)}ULL, "
            f"{as_int(right_x)}ULL, {as_int(right_y)}ULL, "
            f"{as_int(case['expected_slope_numerator'])}ULL, "
            f"{as_int(case['expected_slope_denominator'])}ULL, "
            f"{as_int(case['expected_slope_denominator_inverse'])}ULL, "
            f"{as_int(case['expected_slope'])}ULL, "
            f"{as_int(case['expected_intercept'])}ULL, "
            f"{as_int(case['expected_x'])}ULL, "
            f"{as_int(case['expected_y'])}ULL"
            "},"
        )

    return """#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

enum {
  OP_KIND_ADD = 0,
  OP_KIND_DOUBLE = 1
};

typedef struct {
  const char *source_name;
  const char *event_key;
  int fanout;
  int reused;
  int op_kind;
  uint64_t p;
  uint64_t a1;
  uint64_t a2;
  uint64_t a3;
  uint64_t a4;
  uint64_t a6;
  uint64_t left_x;
  uint64_t left_y;
  uint64_t right_x;
  uint64_t right_y;
  uint64_t expected_slope_numerator;
  uint64_t expected_slope_denominator;
  uint64_t expected_slope_denominator_inverse;
  uint64_t expected_slope;
  uint64_t expected_intercept;
  uint64_t expected_x;
  uint64_t expected_y;
} case_t;

typedef struct {
  uint64_t slope_numerator;
  uint64_t slope_denominator;
  uint64_t slope_denominator_inverse;
  uint64_t slope;
  uint64_t intercept;
  uint64_t x;
  uint64_t y;
} kernel_out_t;

typedef struct {
  uint64_t add;
  uint64_t sub;
  uint64_t mul;
  uint64_t neg;
  uint64_t inv;
} op_counts_t;

static uint64_t mod_add(uint64_t a, uint64_t b, uint64_t p, op_counts_t *counts) {
  counts->add += 1;
  return (a % p + b % p) % p;
}

static uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t p, op_counts_t *counts) {
  counts->sub += 1;
  return (a % p + p - (b % p)) % p;
}

static uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t p, op_counts_t *counts) {
  counts->mul += 1;
  return ((a % p) * (b % p)) % p;
}

static uint64_t mod_neg(uint64_t a, uint64_t p, op_counts_t *counts) {
  counts->neg += 1;
  return (p - (a % p)) % p;
}

static uint64_t mod_inv(uint64_t a, uint64_t p, int *ok, op_counts_t *counts) {
  int64_t t = 0;
  int64_t new_t = 1;
  int64_t r = (int64_t)p;
  int64_t new_r = (int64_t)(a % p);
  counts->inv += 1;

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

static int fused_candidate_point_kernel(const case_t *c, kernel_out_t *out, op_counts_t *counts) {
  int ok = 1;
  uint64_t slope_numerator = 0;
  uint64_t slope_denominator = 0;

  if (c->op_kind == OP_KIND_ADD) {
    slope_numerator = mod_sub(c->right_y, c->left_y, c->p, counts);
    slope_denominator = mod_sub(c->right_x, c->left_x, c->p, counts);
  } else if (c->op_kind == OP_KIND_DOUBLE) {
    uint64_t x_sq = mod_mul(c->left_x, c->left_x, c->p, counts);
    uint64_t three_x_sq = mod_mul(3, x_sq, c->p, counts);
    uint64_t two_a2_x = mod_mul((2 * c->a2) % c->p, c->left_x, c->p, counts);
    uint64_t numerator_tmp = mod_add(three_x_sq, two_a2_x, c->p, counts);
    numerator_tmp = mod_add(numerator_tmp, c->a4, c->p, counts);
    uint64_t a1_y = mod_mul(c->a1, c->left_y, c->p, counts);
    slope_numerator = mod_sub(numerator_tmp, a1_y, c->p, counts);
    uint64_t two_y = mod_mul(2, c->left_y, c->p, counts);
    uint64_t a1_x = mod_mul(c->a1, c->left_x, c->p, counts);
    uint64_t denominator_tmp = mod_add(two_y, a1_x, c->p, counts);
    slope_denominator = mod_add(denominator_tmp, c->a3, c->p, counts);
  } else {
    return 0;
  }

  uint64_t slope_denominator_inverse = mod_inv(slope_denominator, c->p, &ok, counts);
  if (!ok) {
    return 0;
  }
  uint64_t slope = mod_mul(slope_numerator, slope_denominator_inverse, c->p, counts);
  uint64_t slope_x1 = mod_mul(slope, c->left_x, c->p, counts);
  uint64_t intercept = mod_sub(c->left_y, slope_x1, c->p, counts);
  uint64_t slope_sq = mod_mul(slope, slope, c->p, counts);
  uint64_t a1_slope = mod_mul(c->a1, slope, c->p, counts);
  uint64_t x_tmp = mod_add(slope_sq, a1_slope, c->p, counts);
  x_tmp = mod_sub(x_tmp, c->a2, c->p, counts);
  x_tmp = mod_sub(x_tmp, c->left_x, c->p, counts);
  uint64_t x3 = mod_sub(x_tmp, c->right_x, c->p, counts);
  uint64_t slope_plus_a1 = mod_add(slope, c->a1, c->p, counts);
  uint64_t y_tmp = mod_mul(slope_plus_a1, x3, c->p, counts);
  y_tmp = mod_neg(y_tmp, c->p, counts);
  y_tmp = mod_sub(y_tmp, intercept, c->p, counts);
  uint64_t y3 = mod_sub(y_tmp, c->a3, c->p, counts);

  out->slope_numerator = slope_numerator;
  out->slope_denominator = slope_denominator;
  out->slope_denominator_inverse = slope_denominator_inverse;
  out->slope = slope;
  out->intercept = intercept;
  out->x = x3;
  out->y = y3;
  return 1;
}

static const case_t cases[] = {
""" + "\n".join(case_entries) + """
};

int main(void) {
  size_t case_count = sizeof(cases) / sizeof(cases[0]);
  size_t verified_count = 0;
  size_t failure_count = 0;
  size_t result_match_count = 0;
  size_t register_match_count = 0;
  size_t reused_case_count = 0;
  op_counts_t total_counts = {0, 0, 0, 0, 0};

  for (size_t i = 0; i < case_count; ++i) {
    const case_t *c = &cases[i];
    kernel_out_t out = {0, 0, 0, 0, 0, 0, 0};
    op_counts_t counts = {0, 0, 0, 0, 0};
    int ok = fused_candidate_point_kernel(c, &out, &counts);
    int case_failed = 0;
    if (c->reused) {
      reused_case_count += 1;
    }

    total_counts.add += counts.add;
    total_counts.sub += counts.sub;
    total_counts.mul += counts.mul;
    total_counts.neg += counts.neg;
    total_counts.inv += counts.inv;

    if (!ok) {
      fprintf(stderr, "kernel failed source=%s event=%s\\n", c->source_name, c->event_key);
      case_failed = 1;
      failure_count += 1;
    }

    if (
      out.slope_numerator == c->expected_slope_numerator &&
      out.slope_denominator == c->expected_slope_denominator &&
      out.slope_denominator_inverse == c->expected_slope_denominator_inverse &&
      out.slope == c->expected_slope &&
      out.intercept == c->expected_intercept
    ) {
      register_match_count += 1;
    } else {
      fprintf(
        stderr,
        "register mismatch source=%s event=%s got=(%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ")\\n",
        c->source_name,
        c->event_key,
        out.slope_numerator,
        out.slope_denominator,
        out.slope_denominator_inverse,
        out.slope,
        out.intercept
      );
      case_failed = 1;
      failure_count += 1;
    }

    if (out.x == c->expected_x && out.y == c->expected_y) {
      result_match_count += 1;
    } else {
      fprintf(
        stderr,
        "result mismatch source=%s event=%s expected=(%" PRIu64 ",%" PRIu64 ") got=(%" PRIu64 ",%" PRIu64 ")\\n",
        c->source_name,
        c->event_key,
        c->expected_x,
        c->expected_y,
        out.x,
        out.y
      );
      case_failed = 1;
      failure_count += 1;
    }

    if (!case_failed) {
      verified_count += 1;
    }
  }

  uint64_t total_ops = total_counts.add + total_counts.sub + total_counts.mul + total_counts.neg + total_counts.inv;
  printf(
    "{"
    "\\"case_count\\":%zu,"
    "\\"verified_count\\":%zu,"
    "\\"failure_count\\":%zu,"
    "\\"result_match_count\\":%zu,"
    "\\"register_match_count\\":%zu,"
    "\\"reused_case_count\\":%zu,"
    "\\"field_op_counts\\":{"
    "\\"add\\":%" PRIu64 ","
    "\\"sub\\":%" PRIu64 ","
    "\\"mul\\":%" PRIu64 ","
    "\\"neg\\":%" PRIu64 ","
    "\\"inv\\":%" PRIu64 ","
    "\\"total\\":%" PRIu64
    "}"
    "}\\n",
    case_count,
    verified_count,
    failure_count,
    result_match_count,
    register_match_count,
    reused_case_count,
    total_counts.add,
    total_counts.sub,
    total_counts.mul,
    total_counts.neg,
    total_counts.inv,
    total_ops
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


def run_native_kernel(c_source: str, c_out: Path, compiler: str, keep_exe: bool) -> dict[str, Any]:
    c_out.parent.mkdir(parents=True, exist_ok=True)
    c_out.write_text(c_source)
    temp_root = Path("/private/tmp")
    exe_parent = c_out.parent if keep_exe else Path(tempfile.mkdtemp(prefix="ecdlp_native_kernel_", dir=temp_root))
    exe_path = exe_parent / f"{c_out.stem}.bin"
    command = compiler_command(compiler, c_out, exe_path)
    env = os.environ.copy()
    env["TMPDIR"] = str(temp_root)
    compile_run = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
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
        "executable": str(exe_path),
    }


def source_record_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for record in records:
        groups = record.get("trace_groups") or []
        summaries.append(
            {
                "source_name": record.get("source_name"),
                "public_group_key": record.get("public_group_key"),
                "native_kernel_status": "pending_global_native_kernel_result",
                "affine_trace_status": record.get("affine_trace_status"),
                "case_count": len(groups),
                "reused_case_count": sum(1 for group in groups if bool(group.get("candidate_point_reused"))),
                "matches_contract_below_rho": bool((record.get("checks") or {}).get("matches_contract_below_rho")),
                "matches_contract_event_reuse_target": bool(
                    (record.get("checks") or {}).get("matches_contract_event_reuse_target")
                ),
            }
        )
    return summaries


def reference_counts(reference_source: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(reference_source, dict):
        return {}
    summary = reference_source.get("summary") or {}
    return {
        "case_count": as_int(summary.get("native_case_count"), -1),
        "field_op_counts": summary.get("native_field_op_counts"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affine-source", type=Path, default=DEFAULT_AFFINE_SOURCE)
    parser.add_argument("--native-reference-source", type=Path, default=DEFAULT_NATIVE_REFERENCE_SOURCE)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-out", type=Path)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--keep-exe", action="store_true")
    args = parser.parse_args()

    source = load_json(args.affine_source)
    native_reference = load_json(args.native_reference_source) if args.native_reference_source.exists() else None
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
    native = run_native_kernel(c_source, c_out, args.cc, args.keep_exe)
    native_summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    reference = reference_counts(native_reference)
    reference_ops = reference.get("field_op_counts") if isinstance(reference.get("field_op_counts"), dict) else {}
    native_ops = native_summary.get("field_op_counts") if isinstance(native_summary.get("field_op_counts"), dict) else {}
    matches_reference_counts = bool(
        as_int(reference.get("case_count"), len(cases)) == len(cases)
        and all(as_int(native_ops.get(key), -2) == as_int(reference_ops.get(key), -1) for key in ("add", "sub", "mul", "neg", "inv", "total"))
    ) if reference_ops else None
    global_ok = bool(
        native.get("compiled")
        and native.get("executed")
        and as_int(native.get("run_returncode"), -1) == 0
        and as_int(native_summary.get("case_count"), -1) == len(cases)
        and as_int(native_summary.get("verified_count"), -1) == len(cases)
        and as_int(native_summary.get("failure_count"), -1) == 0
        and as_int(native_summary.get("result_match_count"), -1) == len(cases)
        and as_int(native_summary.get("register_match_count"), -1) == len(cases)
        and (matches_reference_counts is not False)
    )
    record_summaries = source_record_summaries(records)
    for summary in record_summaries:
        summary["native_kernel_status"] = "native_kernel_verified" if global_ok else "native_kernel_failed_check"

    below = [record for record in record_summaries if bool(record.get("matches_contract_below_rho"))]
    event_reuse = [
        record for record in below if bool(record.get("matches_contract_event_reuse_target"))
    ]
    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_kernel_native_kernel_probe_v1",
        "method": "compiled_c_affine_add_kernel_for_first_pass_candidate_points",
        "parameters": {
            "affine_source": str(args.affine_source),
            "native_reference_source": str(args.native_reference_source),
            "source_names": sorted(wanted),
            "compiler": args.cc,
            "c_out": str(c_out),
            "keep_exe": bool(args.keep_exe),
        },
        "summary": {
            "record_count": len(records),
            "native_kernel_verified_count": len(records) if global_ok else 0,
            "below_rho_native_kernel_verified_count": len(below) if global_ok else 0,
            "event_reuse_below_rho_native_kernel_verified_count": len(event_reuse) if global_ok else 0,
            "native_case_count": len(cases),
            "native_verified_case_count": as_int(native_summary.get("verified_count")),
            "native_failure_count": as_int(native_summary.get("failure_count"), -1),
            "native_result_match_count": as_int(native_summary.get("result_match_count")),
            "native_register_match_count": as_int(native_summary.get("register_match_count")),
            "native_reused_case_count": as_int(native_summary.get("reused_case_count")),
            "native_field_op_counts": native_summary.get("field_op_counts"),
            "matches_native_reference_op_counts": matches_reference_counts,
            "compiled": bool(native.get("compiled")),
            "executed": bool(native.get("executed")),
            "global_native_kernel_verified": global_ok,
            "interpretation": (
                "The first-pass candidate-point layer has been executed by a "
                "compact native C affine-add kernel over operand arrays."
            ),
        },
        "records": record_summaries,
        "native_execution": native,
        "non_claims": [
            "This is a compact native reference kernel for first-pass candidate-point arithmetic, not an optimized FFE kernel.",
            "It does not search for new relations and does not replace the second-pass relation predicate layer.",
            "A production FFE/summation-polynomial implementation must still pass this kernel gate and the ABI/contract gates.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0 if global_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
