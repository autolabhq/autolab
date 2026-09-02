#!/usr/bin/env python3
"""Compile and run a grouped-left-y native batch kernel for fused candidates.

The shared-denominator kernel prepares one denominator/inverse per repeated
coordinate group, but still computes one candidate point per event case.  This
probe adds the next batch lane: within each shared denominator group, cases
with the same left y-coordinate produce the same candidate point, so the
generated C computes each unique left-y lane once and then verifies all event
cases against that lane output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_AFFINE_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_affine_trace_target67_672_744.json"
)
DEFAULT_SHARED_DENOMINATOR_SOURCE = (
    DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_shared_denominator_target67_672_744.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_public_repeated_coordinate_fused_kernel_batch_lane.json"


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


def collect_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for record in records:
        source_name = str(record.get("source_name") or "")
        for group in record.get("trace_groups") or []:
            if not isinstance(group, dict):
                continue
            trace = group.get("representative_affine_trace") or {}
            if str(trace.get("operation") or "") != "addition":
                raise ValueError("batch-lane probe currently expects affine addition cases")
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
                    "p": parse_field(trace.get("field")),
                    "ainvs": ainvs,
                    "left_x": left_x,
                    "left_y": left_y,
                    "right_x": right_x,
                    "right_y": right_y,
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


def group_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        as_int(case["p"]),
        tuple(as_int(value) for value in case["ainvs"]),
        as_int(case["left_x"]),
        as_int(case["right_x"]),
        as_int(case["right_y"]),
        as_int(case["expected_slope_denominator"]),
        as_int(case["expected_slope_denominator_inverse"]),
    )


def lane_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        as_int(case["left_y"]),
        as_int(case["expected_slope_numerator"]),
        as_int(case["expected_slope"]),
        as_int(case["expected_intercept"]),
        as_int(case["expected_x"]),
        as_int(case["expected_y"]),
    )


def build_layout(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_group[group_key(case)].append(case)

    groups = []
    lanes = []
    event_cases = []
    for group_index, key in enumerate(sorted(by_group, key=lambda item: (-len(by_group[item]), item))):
        group_cases = sorted(by_group[key], key=lambda item: (item["source_name"], item["event_key"]))
        first = group_cases[0]
        lane_start = len(lanes)
        by_lane: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for case in group_cases:
            by_lane[lane_key(case)].append(case)

        lane_index_by_key = {}
        for local_lane_index, lkey in enumerate(sorted(by_lane, key=lambda item: (-len(by_lane[item]), item))):
            lane_cases = by_lane[lkey]
            lane = lane_cases[0]
            lane_index = len(lanes)
            lane_index_by_key[lkey] = lane_index
            lanes.append(
                {
                    "group_index": group_index,
                    "left_y": lane["left_y"],
                    "expected_slope_numerator": lane["expected_slope_numerator"],
                    "expected_slope": lane["expected_slope"],
                    "expected_intercept": lane["expected_intercept"],
                    "expected_x": lane["expected_x"],
                    "expected_y": lane["expected_y"],
                    "case_count": len(lane_cases),
                    "local_lane_index": local_lane_index,
                }
            )

        for case in group_cases:
            event_cases.append({**case, "group_index": group_index, "lane_index": lane_index_by_key[lane_key(case)]})

        groups.append(
            {
                "group_index": group_index,
                "p": first["p"],
                "ainvs": first["ainvs"],
                "left_x": first["left_x"],
                "right_x": first["right_x"],
                "right_y": first["right_y"],
                "expected_slope_denominator": first["expected_slope_denominator"],
                "expected_slope_denominator_inverse": first["expected_slope_denominator_inverse"],
                "lane_start": lane_start,
                "lane_count": len(by_lane),
                "case_count": len(group_cases),
            }
        )
    return groups, lanes, event_cases


def render_c_source(groups: list[dict[str, Any]], lanes: list[dict[str, Any]], cases: list[dict[str, Any]]) -> str:
    group_entries = []
    for group in groups:
        a1, a2, a3, a4, a6 = group["ainvs"]
        group_entries.append(
            "  {"
            f"{as_int(group['p'])}ULL, "
            f"{as_int(a1)}ULL, {as_int(a2)}ULL, {as_int(a3)}ULL, {as_int(a4)}ULL, {as_int(a6)}ULL, "
            f"{as_int(group['left_x'])}ULL, {as_int(group['right_x'])}ULL, {as_int(group['right_y'])}ULL, "
            f"{as_int(group['expected_slope_denominator'])}ULL, "
            f"{as_int(group['expected_slope_denominator_inverse'])}ULL, "
            f"{as_int(group['lane_start'])}, {as_int(group['lane_count'])}, {as_int(group['case_count'])}"
            "},"
        )

    lane_entries = []
    for lane in lanes:
        lane_entries.append(
            "  {"
            f"{as_int(lane['group_index'])}, "
            f"{as_int(lane['left_y'])}ULL, "
            f"{as_int(lane['expected_slope_numerator'])}ULL, "
            f"{as_int(lane['expected_slope'])}ULL, "
            f"{as_int(lane['expected_intercept'])}ULL, "
            f"{as_int(lane['expected_x'])}ULL, "
            f"{as_int(lane['expected_y'])}ULL, "
            f"{as_int(lane['case_count'])}"
            "},"
        )

    case_entries = []
    for case in cases:
        case_entries.append(
            "  {"
            f"{c_string(case['source_name'])}, {c_string(case['event_key'])}, "
            f"{as_int(case['fanout'])}, {1 if case['candidate_point_reused'] else 0}, "
            f"{as_int(case['lane_index'])}, "
            f"{as_int(case['expected_slope_numerator'])}ULL, "
            f"{as_int(case['expected_slope'])}ULL, "
            f"{as_int(case['expected_intercept'])}ULL, "
            f"{as_int(case['expected_x'])}ULL, {as_int(case['expected_y'])}ULL"
            "},"
        )

    return """#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  uint64_t p;
  uint64_t a1;
  uint64_t a2;
  uint64_t a3;
  uint64_t a4;
  uint64_t a6;
  uint64_t left_x;
  uint64_t right_x;
  uint64_t right_y;
  uint64_t expected_slope_denominator;
  uint64_t expected_slope_denominator_inverse;
  size_t lane_start;
  size_t lane_count;
  size_t case_count;
} shared_group_t;

typedef struct {
  size_t group_index;
  uint64_t left_y;
  uint64_t expected_slope_numerator;
  uint64_t expected_slope;
  uint64_t expected_intercept;
  uint64_t expected_x;
  uint64_t expected_y;
  size_t case_count;
} lane_t;

typedef struct {
  const char *source_name;
  const char *event_key;
  int fanout;
  int reused;
  size_t lane_index;
  uint64_t expected_slope_numerator;
  uint64_t expected_slope;
  uint64_t expected_intercept;
  uint64_t expected_x;
  uint64_t expected_y;
} case_t;

typedef struct {
  uint64_t slope_denominator;
  uint64_t slope_denominator_inverse;
} shared_state_t;

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

static int prepare_shared_denominator(const shared_group_t *g, shared_state_t *state, op_counts_t *counts) {
  int ok = 1;
  state->slope_denominator = mod_sub(g->right_x, g->left_x, g->p, counts);
  state->slope_denominator_inverse = mod_inv(state->slope_denominator, g->p, &ok, counts);
  return ok;
}

static int batch_lane_candidate_kernel(
    const shared_group_t *g,
    const shared_state_t *state,
    const lane_t *lane,
    kernel_out_t *out,
    op_counts_t *counts) {
  uint64_t slope_numerator = mod_sub(g->right_y, lane->left_y, g->p, counts);
  uint64_t slope = mod_mul(slope_numerator, state->slope_denominator_inverse, g->p, counts);
  uint64_t slope_x1 = mod_mul(slope, g->left_x, g->p, counts);
  uint64_t intercept = mod_sub(lane->left_y, slope_x1, g->p, counts);
  uint64_t slope_sq = mod_mul(slope, slope, g->p, counts);
  uint64_t a1_slope = mod_mul(g->a1, slope, g->p, counts);
  uint64_t x_tmp = mod_add(slope_sq, a1_slope, g->p, counts);
  x_tmp = mod_sub(x_tmp, g->a2, g->p, counts);
  x_tmp = mod_sub(x_tmp, g->left_x, g->p, counts);
  uint64_t x3 = mod_sub(x_tmp, g->right_x, g->p, counts);
  uint64_t slope_plus_a1 = mod_add(slope, g->a1, g->p, counts);
  uint64_t y_tmp = mod_mul(slope_plus_a1, x3, g->p, counts);
  y_tmp = mod_neg(y_tmp, g->p, counts);
  y_tmp = mod_sub(y_tmp, intercept, g->p, counts);
  uint64_t y3 = mod_sub(y_tmp, g->a3, g->p, counts);

  out->slope_numerator = slope_numerator;
  out->slope_denominator = state->slope_denominator;
  out->slope_denominator_inverse = state->slope_denominator_inverse;
  out->slope = slope;
  out->intercept = intercept;
  out->x = x3;
  out->y = y3;
  return 1;
}

static const shared_group_t groups[] = {
""" + "\n".join(group_entries) + """
};

static const lane_t lanes[] = {
""" + "\n".join(lane_entries) + """
};

static const case_t cases[] = {
""" + "\n".join(case_entries) + """
};

int main(void) {
  size_t group_count = sizeof(groups) / sizeof(groups[0]);
  size_t lane_count = sizeof(lanes) / sizeof(lanes[0]);
  size_t case_count = sizeof(cases) / sizeof(cases[0]);
  kernel_out_t lane_outputs[sizeof(lanes) / sizeof(lanes[0])];
  size_t lane_ready[sizeof(lanes) / sizeof(lanes[0])];
  size_t lane_match_count = 0;
  size_t case_match_count = 0;
  size_t result_match_count = 0;
  size_t register_match_count = 0;
  size_t shared_prepare_match_count = 0;
  size_t reused_case_count = 0;
  size_t failure_count = 0;
  op_counts_t total_counts = {0, 0, 0, 0, 0};

  for (size_t i = 0; i < lane_count; ++i) {
    lane_ready[i] = 0;
  }

  for (size_t i = 0; i < group_count; ++i) {
    const shared_group_t *g = &groups[i];
    shared_state_t state = {0, 0};
    op_counts_t prepare_counts = {0, 0, 0, 0, 0};
    if (!prepare_shared_denominator(g, &state, &prepare_counts)) {
      fprintf(stderr, "shared denominator inverse failed group=%zu\\n", i);
      failure_count += 1;
      continue;
    }
    total_counts.add += prepare_counts.add;
    total_counts.sub += prepare_counts.sub;
    total_counts.mul += prepare_counts.mul;
    total_counts.neg += prepare_counts.neg;
    total_counts.inv += prepare_counts.inv;

    if (
      state.slope_denominator == g->expected_slope_denominator &&
      state.slope_denominator_inverse == g->expected_slope_denominator_inverse
    ) {
      shared_prepare_match_count += 1;
    } else {
      fprintf(stderr, "shared prepare mismatch group=%zu\\n", i);
      failure_count += 1;
    }

    for (size_t j = 0; j < g->lane_count; ++j) {
      size_t lane_index = g->lane_start + j;
      const lane_t *lane = &lanes[lane_index];
      kernel_out_t out = {0, 0, 0, 0, 0, 0, 0};
      op_counts_t counts = {0, 0, 0, 0, 0};
      if (!batch_lane_candidate_kernel(g, &state, lane, &out, &counts)) {
        fprintf(stderr, "lane kernel failed group=%zu lane=%zu\\n", i, j);
        failure_count += 1;
        continue;
      }
      total_counts.add += counts.add;
      total_counts.sub += counts.sub;
      total_counts.mul += counts.mul;
      total_counts.neg += counts.neg;
      total_counts.inv += counts.inv;
      lane_outputs[lane_index] = out;
      lane_ready[lane_index] = 1;

      if (
        out.slope_numerator == lane->expected_slope_numerator &&
        out.slope == lane->expected_slope &&
        out.intercept == lane->expected_intercept &&
        out.x == lane->expected_x &&
        out.y == lane->expected_y
      ) {
        lane_match_count += 1;
      } else {
        fprintf(stderr, "lane mismatch group=%zu lane=%zu\\n", i, j);
        failure_count += 1;
      }
    }
  }

  for (size_t i = 0; i < case_count; ++i) {
    const case_t *c = &cases[i];
    if (c->reused) {
      reused_case_count += 1;
    }
    if (c->lane_index >= lane_count || !lane_ready[c->lane_index]) {
      fprintf(stderr, "missing lane output source=%s event=%s\\n", c->source_name, c->event_key);
      failure_count += 1;
      continue;
    }
    const kernel_out_t *out = &lane_outputs[c->lane_index];
    if (
      out->slope_numerator == c->expected_slope_numerator &&
      out->slope == c->expected_slope &&
      out->intercept == c->expected_intercept
    ) {
      register_match_count += 1;
    } else {
      fprintf(stderr, "case register mismatch source=%s event=%s\\n", c->source_name, c->event_key);
      failure_count += 1;
    }
    if (out->x == c->expected_x && out->y == c->expected_y) {
      result_match_count += 1;
    } else {
      fprintf(stderr, "case result mismatch source=%s event=%s\\n", c->source_name, c->event_key);
      failure_count += 1;
    }
    if (
      out->slope_numerator == c->expected_slope_numerator &&
      out->slope == c->expected_slope &&
      out->intercept == c->expected_intercept &&
      out->x == c->expected_x &&
      out->y == c->expected_y
    ) {
      case_match_count += 1;
    }
  }

  uint64_t total_ops = total_counts.add + total_counts.sub + total_counts.mul + total_counts.neg + total_counts.inv;
  printf(
    "{"
    "\\"group_count\\":%zu,"
    "\\"lane_count\\":%zu,"
    "\\"case_count\\":%zu,"
    "\\"lane_match_count\\":%zu,"
    "\\"case_match_count\\":%zu,"
    "\\"failure_count\\":%zu,"
    "\\"shared_prepare_match_count\\":%zu,"
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
    group_count,
    lane_count,
    case_count,
    lane_match_count,
    case_match_count,
    failure_count,
    shared_prepare_match_count,
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
    return [compiler, "-std=c99", "-O2", "-Wall", "-Wextra", str(c_path), "-o", str(exe_path)]


def run_batch_kernel(c_source: str, c_out: Path, compiler: str, keep_exe: bool) -> dict[str, Any]:
    c_out.parent.mkdir(parents=True, exist_ok=True)
    c_out.write_text(c_source)
    temp_root = Path("/private/tmp")
    exe_parent = c_out.parent if keep_exe else Path(tempfile.mkdtemp(prefix="ecdlp_batch_lane_", dir=temp_root))
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


def shared_denominator_counts(shared_source: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(shared_source, dict):
        return {}
    summary = shared_source.get("summary") or {}
    return {
        "case_count": as_int(summary.get("native_case_count"), -1),
        "shared_group_count": as_int(summary.get("shared_group_count"), -1),
        "field_op_counts": summary.get("native_field_op_counts"),
    }


def source_record_summaries(records: list[dict[str, Any]], cases: list[dict[str, Any]], lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_cases: Counter[str] = Counter(case["source_name"] for case in cases)
    source_reuse: Counter[str] = Counter()
    source_lanes: dict[str, set[tuple[int, int, int, int]]] = defaultdict(set)
    for case in cases:
        if bool(case.get("candidate_point_reused")):
            source_reuse[case["source_name"]] += 1
        lane = lanes[case["lane_index"]]
        source_lanes[case["source_name"]].add(
            (
                as_int(lane["group_index"]),
                as_int(lane["left_y"]),
                as_int(lane["expected_x"]),
                as_int(lane["expected_y"]),
            )
        )
    summaries = []
    for record in records:
        source_name = str(record.get("source_name") or "")
        summaries.append(
            {
                "source_name": source_name,
                "public_group_key": record.get("public_group_key"),
                "batch_lane_status": "pending_global_batch_lane_result",
                "affine_trace_status": record.get("affine_trace_status"),
                "case_count": source_cases[source_name],
                "lane_count": len(source_lanes[source_name]),
                "reused_case_count": source_reuse[source_name],
                "matches_contract_below_rho": bool((record.get("checks") or {}).get("matches_contract_below_rho")),
                "matches_contract_event_reuse_target": bool(
                    (record.get("checks") or {}).get("matches_contract_event_reuse_target")
                ),
            }
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affine-source", type=Path, default=DEFAULT_AFFINE_SOURCE)
    parser.add_argument("--shared-denominator-source", type=Path, default=DEFAULT_SHARED_DENOMINATOR_SOURCE)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-out", type=Path)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--keep-exe", action="store_true")
    args = parser.parse_args()

    source = load_json(args.affine_source)
    shared_source = load_json(args.shared_denominator_source) if args.shared_denominator_source.exists() else None
    wanted = set(args.source_name or [])
    records = []
    for record in source.get("records") or []:
        if not isinstance(record, dict):
            continue
        source_name = str(record.get("source_name") or "")
        if wanted and source_name not in wanted:
            continue
        records.append(record)

    raw_cases = collect_cases(records)
    groups, lanes, cases = build_layout(raw_cases)
    c_source = render_c_source(groups, lanes, cases)
    c_out = args.c_out or args.out.with_suffix(".c")
    native = run_batch_kernel(c_source, c_out, args.cc, args.keep_exe)
    native_summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    baseline = shared_denominator_counts(shared_source)
    baseline_ops = baseline.get("field_op_counts") if isinstance(baseline.get("field_op_counts"), dict) else {}
    native_ops = native_summary.get("field_op_counts") if isinstance(native_summary.get("field_op_counts"), dict) else {}
    savings = {}
    if baseline_ops and native_ops:
        for key in ("add", "sub", "mul", "neg", "inv", "total"):
            savings[key] = as_int(baseline_ops.get(key)) - as_int(native_ops.get(key))
    matches_case_count = as_int(baseline.get("case_count"), len(cases)) == len(cases)
    matches_group_count = as_int(baseline.get("shared_group_count"), len(groups)) == len(groups)
    global_ok = bool(
        native.get("compiled")
        and native.get("executed")
        and as_int(native.get("run_returncode"), -1) == 0
        and as_int(native_summary.get("group_count"), -1) == len(groups)
        and as_int(native_summary.get("lane_count"), -1) == len(lanes)
        and as_int(native_summary.get("case_count"), -1) == len(cases)
        and as_int(native_summary.get("lane_match_count"), -1) == len(lanes)
        and as_int(native_summary.get("case_match_count"), -1) == len(cases)
        and as_int(native_summary.get("failure_count"), -1) == 0
        and as_int(native_summary.get("shared_prepare_match_count"), -1) == len(groups)
        and as_int(native_summary.get("result_match_count"), -1) == len(cases)
        and as_int(native_summary.get("register_match_count"), -1) == len(cases)
        and matches_case_count
        and matches_group_count
    )
    record_summaries = source_record_summaries(records, cases, lanes)
    for summary in record_summaries:
        summary["batch_lane_status"] = "batch_lane_native_kernel_verified" if global_ok else "batch_lane_native_kernel_failed_check"

    below = [record for record in record_summaries if bool(record.get("matches_contract_below_rho"))]
    event_reuse = [record for record in below if bool(record.get("matches_contract_event_reuse_target"))]
    output = {
        "schema": "ecdlp_public_repeated_coordinate_fused_kernel_batch_lane_probe_v1",
        "method": "compiled_c_grouped_left_y_batch_lane_kernel_for_repeated_coordinate_candidate_points",
        "parameters": {
            "affine_source": str(args.affine_source),
            "shared_denominator_source": str(args.shared_denominator_source),
            "source_names": sorted(wanted),
            "compiler": args.cc,
            "c_out": str(c_out),
            "keep_exe": bool(args.keep_exe),
        },
        "summary": {
            "record_count": len(records),
            "batch_lane_verified_count": len(records) if global_ok else 0,
            "below_rho_batch_lane_verified_count": len(below) if global_ok else 0,
            "event_reuse_below_rho_batch_lane_verified_count": len(event_reuse) if global_ok else 0,
            "shared_group_count": len(groups),
            "batch_lane_count": len(lanes),
            "native_case_count": len(cases),
            "native_lane_match_count": as_int(native_summary.get("lane_match_count")),
            "native_case_match_count": as_int(native_summary.get("case_match_count")),
            "native_failure_count": as_int(native_summary.get("failure_count"), -1),
            "native_shared_prepare_match_count": as_int(native_summary.get("shared_prepare_match_count")),
            "native_result_match_count": as_int(native_summary.get("result_match_count")),
            "native_register_match_count": as_int(native_summary.get("register_match_count")),
            "native_reused_case_count": as_int(native_summary.get("reused_case_count")),
            "native_field_op_counts": native_summary.get("field_op_counts"),
            "baseline_shared_denominator_field_op_counts": baseline_ops or None,
            "field_op_savings_vs_shared_denominator": savings or None,
            "compiled": bool(native.get("compiled")),
            "executed": bool(native.get("executed")),
            "global_batch_lane_verified": global_ok,
            "interpretation": (
                "Repeated-coordinate candidate points now batch by unique "
                "left-y lane inside each shared-denominator group."
            ),
        },
        "records": record_summaries,
        "shared_groups": [
            {
                "group_index": group["group_index"],
                "p": group["p"],
                "left_x": group["left_x"],
                "right": [group["right_x"], group["right_y"]],
                "slope_denominator": group["expected_slope_denominator"],
                "slope_denominator_inverse": group["expected_slope_denominator_inverse"],
                "case_count": group["case_count"],
                "lane_count": group["lane_count"],
            }
            for group in groups
        ],
        "batch_lanes": [
            {
                "lane_index": index,
                "group_index": lane["group_index"],
                "left_y": lane["left_y"],
                "candidate_point": [lane["expected_x"], lane["expected_y"]],
                "case_count": lane["case_count"],
            }
            for index, lane in enumerate(lanes)
        ],
        "native_execution": native,
        "non_claims": [
            "This specializes first-pass candidate-point arithmetic only; it is not a new relation search.",
            "It does not replace the second-pass relation predicate or modular linear algebra layers.",
            "A production FFE/summation-polynomial implementation must still pass the ABI and contract gates.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0 if global_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
