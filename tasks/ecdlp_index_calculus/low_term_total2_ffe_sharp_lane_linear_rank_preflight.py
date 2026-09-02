#!/usr/bin/env python3
"""Preflight GF(2) support-mask rank for selected13 sharp-lane rows.

The exact certificate replay exposes seven form-support masks from six embedded
certificates.  The row execution worklist exposes thirteen full-family
direct/rank backfill rows, each with three family masks.  This script checks
whether those backfill masks add any new GF(2) support-space rank before the
worker claims a direct/rank export.

This is a negative algebraic guardrail: support-mask span coverage is not a
summation-polynomial evaluation, finite-field equation solve, direct/rank row
export, ECDLP solve, or Pollard-rho speedup.
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


SCHEMA = "ecdlp.low_term_total2_ffe_sharp_lane_linear_rank_preflight.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_EXACT_REPLAY = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_exact_certificate_replay_selected13_9696_9999_probe.json"
)
DEFAULT_WORKLIST = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_execution_worklist_selected13_9696_9999_probe.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_linear_rank_preflight_selected13_9696_9999_probe.json"
)
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_linear_rank_preflight_selected13_9696_9999_probe.h"
)

EXPECTED_EXACT_FORM_MASK_COUNT = 7
EXPECTED_UNIQUE_EXACT_FORM_MASK_COUNT = 4
EXPECTED_EXACT_FORM_MASK_RANK = 4
EXPECTED_FULL_FAMILY_BACKFILL_COUNT = 13
EXPECTED_FAMILY_MASKS_PER_BACKFILL = 3
PRIMARY_BACKFILL_TRANSFERS = [9981, 9943]
SELECTED13_MASK = 1 << 13


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


def digest_u64(raw: Any) -> int:
    text = str(raw or "")
    suffix = text.rsplit("_", 1)[-1]
    if suffix and all(ch in "0123456789abcdefABCDEF" for ch in suffix):
        return int(suffix[-16:], 16)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def gf2_rank64(values: list[int]) -> int:
    basis = [0] * 64
    rank = 0
    for value in values:
        x = as_int(value) & ((1 << 64) - 1)
        while x:
            bit = x.bit_length() - 1
            if basis[bit]:
                x ^= basis[bit]
                continue
            basis[bit] = x
            rank += 1
            break
    return rank


def in_span64(basis_values: list[int], value: int) -> bool:
    return gf2_rank64(basis_values + [value]) == gf2_rank64(basis_values)


def unique_first_seen(values: list[int]) -> list[int]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def exact_records(exact_replay: dict[str, Any]) -> list[dict[str, Any]]:
    records = exact_replay.get("records")
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    legacy = exact_replay.get("exact_records")
    if isinstance(legacy, list):
        return [record for record in legacy if isinstance(record, dict)]
    return []


def work_items(worklist: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in worklist.get("work_items") or [] if isinstance(item, dict)]


def build_exact_mask_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for record in records:
        form_masks = [as_int(value) for value in record.get("form_masks") or []]
        form_count = as_int(record.get("form_count"), len([mask for mask in form_masks if mask]))
        form_supports = record.get("form_supports") or []
        for form_index, mask in enumerate(form_masks[:form_count]):
            out.append(
                {
                    "certificate_hash": record.get("certificate_hash"),
                    "exact_index": as_int(record.get("exact_index")),
                    "form_index": form_index,
                    "form_mask": mask,
                    "form_support": form_supports[form_index] if form_index < len(form_supports) else [],
                    "row_id": record.get("row_id"),
                    "transfer_index": as_int(record.get("transfer_index")),
                }
            )
    return out


def is_primary_backfill(item: dict[str, Any]) -> bool:
    return as_int(item.get("transfer_index"), -1) in PRIMARY_BACKFILL_TRANSFERS


def backfill_order_key(item: dict[str, Any]) -> tuple[int, int, int]:
    transfer = as_int(item.get("transfer_index"), -1)
    if transfer in PRIMARY_BACKFILL_TRANSFERS:
        return (0, PRIMARY_BACKFILL_TRANSFERS.index(transfer), as_int(item.get("work_item_index"), -1))
    return (1, transfer, as_int(item.get("work_item_index"), -1))


def build_backfill_items(items: list[dict[str, Any]], exact_masks: list[int]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        if item.get("phase") != "direct_rank_backfill_export":
            continue
        if not bool(item.get("is_full_family_backfill")):
            continue
        family_masks = [as_int(value) for value in (item.get("family_masks") or [])[:EXPECTED_FAMILY_MASKS_PER_BACKFILL]]
        while len(family_masks) < EXPECTED_FAMILY_MASKS_PER_BACKFILL:
            family_masks.append(0)
        span_membership = [in_span64(exact_masks, mask) for mask in family_masks]
        out.append(
            {
                "all_family_masks_in_exact_span": all(span_membership),
                "family_masks": family_masks,
                "family_masks_in_exact_span": span_membership,
                "is_primary": is_primary_backfill(item),
                "row_check_hash": item.get("row_check_hash"),
                "row_check_hash_u64": as_int(item.get("row_check_hash_u64")) or digest_u64(item.get("row_check_hash")),
                "row_id": item.get("row_id"),
                "transfer_index": as_int(item.get("transfer_index"), -1),
                "work_item_index": as_int(item.get("work_item_index"), -1),
            }
        )
    return sorted(out, key=backfill_order_key)


def neutral_controls(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "phase": item.get("phase"),
            "row_id": item.get("row_id"),
            "transfer_index": as_int(item.get("transfer_index"), -1),
            "work_item_index": as_int(item.get("work_item_index"), -1),
        }
        for item in items
        if item.get("phase") == "neutral_exported_control"
    ]


def validate_sources(exact_replay: dict[str, Any], worklist: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if exact_replay.get("claim_status") != "FFE_SHARP_LANE_EXACT_CERTIFICATE_REPLAY_READY":
        failures.append({"code": "exact_replay_not_ready", "claim_status": exact_replay.get("claim_status")})
    if exact_replay.get("failures"):
        failures.append({"code": "exact_replay_has_failures", "failures": exact_replay.get("failures")})
    if (exact_replay.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "exact_replay_summary_not_verified", "summary": exact_replay.get("summary")})
    if worklist.get("claim_status") != "FFE_SHARP_LANE_EXECUTION_WORKLIST_READY":
        failures.append({"code": "worklist_not_ready", "claim_status": worklist.get("claim_status")})
    if worklist.get("failures"):
        failures.append({"code": "worklist_has_failures", "failures": worklist.get("failures")})
    if (worklist.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "worklist_summary_not_verified", "summary": worklist.get("summary")})
    return failures


def validate_rank_shape(
    exact_masks: list[int],
    unique_exact_masks: list[int],
    exact_rank: int,
    backfill_items: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(exact_masks) != EXPECTED_EXACT_FORM_MASK_COUNT:
        failures.append(
            {
                "code": "exact_form_mask_count_mismatch",
                "expected": EXPECTED_EXACT_FORM_MASK_COUNT,
                "observed": len(exact_masks),
            }
        )
    if len(unique_exact_masks) != EXPECTED_UNIQUE_EXACT_FORM_MASK_COUNT:
        failures.append(
            {
                "code": "unique_exact_form_mask_count_mismatch",
                "expected": EXPECTED_UNIQUE_EXACT_FORM_MASK_COUNT,
                "observed": len(unique_exact_masks),
            }
        )
    if exact_rank != EXPECTED_EXACT_FORM_MASK_RANK:
        failures.append(
            {
                "code": "exact_form_mask_rank_mismatch",
                "expected": EXPECTED_EXACT_FORM_MASK_RANK,
                "observed": exact_rank,
            }
        )
    if any(mask == 0 for mask in exact_masks):
        failures.append({"code": "zero_exact_form_mask"})
    if len(backfill_items) != EXPECTED_FULL_FAMILY_BACKFILL_COUNT:
        failures.append(
            {
                "code": "full_family_backfill_count_mismatch",
                "expected": EXPECTED_FULL_FAMILY_BACKFILL_COUNT,
                "observed": len(backfill_items),
            }
        )
    for item in backfill_items:
        if len(item.get("family_masks") or []) != EXPECTED_FAMILY_MASKS_PER_BACKFILL:
            failures.append(
                {
                    "code": "family_mask_width_mismatch",
                    "work_item_index": item.get("work_item_index"),
                    "family_masks": item.get("family_masks"),
                }
            )
        if not item.get("all_family_masks_in_exact_span"):
            failures.append(
                {
                    "code": "backfill_family_mask_outside_exact_span",
                    "work_item_index": item.get("work_item_index"),
                    "family_masks": item.get("family_masks"),
                    "family_masks_in_exact_span": item.get("family_masks_in_exact_span"),
                }
            )
    primary = [as_int(item.get("transfer_index"), -1) for item in backfill_items if bool(item.get("is_primary"))]
    if primary != PRIMARY_BACKFILL_TRANSFERS:
        failures.append({"code": "primary_backfill_order_mismatch", "expected": PRIMARY_BACKFILL_TRANSFERS, "observed": primary})
    neutral_indices = {as_int(item.get("work_item_index"), -1) for item in controls}
    backfill_indices = {as_int(item.get("work_item_index"), -1) for item in backfill_items}
    if neutral_indices & backfill_indices:
        failures.append({"code": "neutral_control_included_in_backfill_claim", "indices": sorted(neutral_indices & backfill_indices)})
    if any(as_int(item.get("transfer_index"), -1) == 9969 for item in backfill_items):
        failures.append({"code": "neutral_transfer_included_in_backfill_claim"})
    return failures


def c_u64_array(values: list[int]) -> str:
    if not values:
        return "{0ULL}"
    return "{" + ", ".join(f"{as_int(value)}ULL" for value in values) + "}"


def render_c_header(exact_masks: list[int], unique_exact_masks: list[int], backfill_items: list[dict[str, Any]]) -> str:
    item_lines = []
    for item in backfill_items:
        family_masks = [as_int(value) for value in item.get("family_masks") or []][:EXPECTED_FAMILY_MASKS_PER_BACKFILL]
        while len(family_masks) < EXPECTED_FAMILY_MASKS_PER_BACKFILL:
            family_masks.append(0)
        item_lines.append(
            "  {"
            f"{as_int(item.get('work_item_index'))}ULL, "
            f"{as_int(item.get('transfer_index'))}ULL, "
            f"{as_int(item.get('row_check_hash_u64'))}ULL, "
            f"{c_u64_array(family_masks)}, "
            f"{1 if bool(item.get('is_primary')) else 0}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_FFE_SHARP_LANE_LINEAR_RANK_PREFLIGHT_H
#define LOW_TERM_TOTAL2_FFE_SHARP_LANE_LINEAR_RANK_PREFLIGHT_H

#include <stdint.h>

#define SELECTED13_LINEAR_RANK_EXACT_FORM_MASK_COUNT {len(exact_masks)}
#define SELECTED13_LINEAR_RANK_UNIQUE_EXACT_FORM_MASK_COUNT {len(unique_exact_masks)}
#define SELECTED13_LINEAR_RANK_EXPECTED_EXACT_FORM_MASK_RANK {gf2_rank64(exact_masks)}
#define SELECTED13_LINEAR_RANK_FULL_FAMILY_BACKFILL_COUNT {len(backfill_items)}
#define SELECTED13_LINEAR_RANK_FAMILY_MASK_COUNT {EXPECTED_FAMILY_MASKS_PER_BACKFILL}
#define SELECTED13_LINEAR_RANK_EXPECTED_SPAN_HIT_COUNT {len(backfill_items) * EXPECTED_FAMILY_MASKS_PER_BACKFILL}

typedef struct {{
  uint64_t work_item_index;
  uint64_t transfer_index;
  uint64_t row_check_hash_u64;
  uint64_t family_masks[SELECTED13_LINEAR_RANK_FAMILY_MASK_COUNT];
  uint64_t is_primary;
}} selected13_linear_rank_backfill_item_t;

static const uint64_t SELECTED13_LINEAR_RANK_EXACT_FORM_MASKS[] =
    {c_u64_array(exact_masks)};

static const uint64_t SELECTED13_LINEAR_RANK_UNIQUE_EXACT_FORM_MASKS[] =
    {c_u64_array(unique_exact_masks)};

static const selected13_linear_rank_backfill_item_t SELECTED13_LINEAR_RANK_BACKFILL_ITEMS[] = {{
{chr(10).join(item_lines)}
}};

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

static int highest_bit(uint64_t value) {{
  for (int bit = 63; bit >= 0; bit--) {{
    if ((value & (1ULL << bit)) != 0) return bit;
  }}
  return -1;
}}

static uint64_t gf2_rank64(const uint64_t *values, size_t count) {{
  uint64_t basis[64] = {{0}};
  uint64_t rank = 0;
  for (size_t i = 0; i < count; i++) {{
    uint64_t x = values[i];
    while (x != 0) {{
      int bit = highest_bit(x);
      if (bit < 0) break;
      if (basis[bit] != 0) {{
        x ^= basis[bit];
      }} else {{
        basis[bit] = x;
        rank++;
        break;
      }}
    }}
  }}
  return rank;
}}

static int in_exact_span(uint64_t value) {{
  uint64_t augmented[SELECTED13_LINEAR_RANK_EXACT_FORM_MASK_COUNT + 1];
  for (size_t i = 0; i < SELECTED13_LINEAR_RANK_EXACT_FORM_MASK_COUNT; i++) {{
    augmented[i] = SELECTED13_LINEAR_RANK_EXACT_FORM_MASKS[i];
  }}
  augmented[SELECTED13_LINEAR_RANK_EXACT_FORM_MASK_COUNT] = value;
  return gf2_rank64(SELECTED13_LINEAR_RANK_EXACT_FORM_MASKS,
                    SELECTED13_LINEAR_RANK_EXACT_FORM_MASK_COUNT) ==
         gf2_rank64(augmented, SELECTED13_LINEAR_RANK_EXACT_FORM_MASK_COUNT + 1);
}}

int main(void) {{
  uint64_t failure_count = 0;
  uint64_t unique_seen_count = 0;
  uint64_t exact_rank = gf2_rank64(SELECTED13_LINEAR_RANK_EXACT_FORM_MASKS,
                                  SELECTED13_LINEAR_RANK_EXACT_FORM_MASK_COUNT);
  uint64_t unique_rank = gf2_rank64(SELECTED13_LINEAR_RANK_UNIQUE_EXACT_FORM_MASKS,
                                   SELECTED13_LINEAR_RANK_UNIQUE_EXACT_FORM_MASK_COUNT);
  uint64_t span_hit_count = 0;
  uint64_t backfill_all_family_in_span_count = 0;
  uint64_t primary_count = 0;
  uint64_t primary_transfer0 = 0;
  uint64_t primary_transfer1 = 0;

  const size_t exact_mask_count =
      sizeof(SELECTED13_LINEAR_RANK_EXACT_FORM_MASKS) / sizeof(SELECTED13_LINEAR_RANK_EXACT_FORM_MASKS[0]);
  const size_t unique_exact_mask_count =
      sizeof(SELECTED13_LINEAR_RANK_UNIQUE_EXACT_FORM_MASKS) /
      sizeof(SELECTED13_LINEAR_RANK_UNIQUE_EXACT_FORM_MASKS[0]);
  const size_t backfill_count =
      sizeof(SELECTED13_LINEAR_RANK_BACKFILL_ITEMS) / sizeof(SELECTED13_LINEAR_RANK_BACKFILL_ITEMS[0]);

  if (exact_mask_count != SELECTED13_LINEAR_RANK_EXACT_FORM_MASK_COUNT) failure_count++;
  if (unique_exact_mask_count != SELECTED13_LINEAR_RANK_UNIQUE_EXACT_FORM_MASK_COUNT) failure_count++;
  if (backfill_count != SELECTED13_LINEAR_RANK_FULL_FAMILY_BACKFILL_COUNT) failure_count++;
  if (exact_rank != SELECTED13_LINEAR_RANK_EXPECTED_EXACT_FORM_MASK_RANK) failure_count++;
  if (unique_rank != exact_rank) failure_count++;

  for (size_t i = 0; i < exact_mask_count; i++) {{
    uint64_t seen = 0;
    if (SELECTED13_LINEAR_RANK_EXACT_FORM_MASKS[i] == 0) failure_count++;
    for (size_t j = 0; j < i; j++) {{
      if (SELECTED13_LINEAR_RANK_EXACT_FORM_MASKS[j] == SELECTED13_LINEAR_RANK_EXACT_FORM_MASKS[i]) {{
        seen = 1;
      }}
    }}
    if (seen == 0) unique_seen_count++;
  }}
  if (unique_seen_count != SELECTED13_LINEAR_RANK_UNIQUE_EXACT_FORM_MASK_COUNT) failure_count++;

  for (size_t i = 0; i < backfill_count; i++) {{
    const selected13_linear_rank_backfill_item_t *item = &SELECTED13_LINEAR_RANK_BACKFILL_ITEMS[i];
    uint64_t row_span_hits = 0;
    if (item->row_check_hash_u64 == 0) failure_count++;
    for (size_t j = 0; j < SELECTED13_LINEAR_RANK_FAMILY_MASK_COUNT; j++) {{
      if (item->family_masks[j] == 0) failure_count++;
      if (in_exact_span(item->family_masks[j])) {{
        row_span_hits++;
        span_hit_count++;
      }}
    }}
    if (row_span_hits == SELECTED13_LINEAR_RANK_FAMILY_MASK_COUNT) {{
      backfill_all_family_in_span_count++;
    }} else {{
      failure_count++;
    }}
    if (item->is_primary != 0) {{
      if (primary_count == 0) {{
        primary_transfer0 = item->transfer_index;
        if (item->transfer_index != {PRIMARY_BACKFILL_TRANSFERS[0]}ULL) failure_count++;
      }} else if (primary_count == 1) {{
        primary_transfer1 = item->transfer_index;
        if (item->transfer_index != {PRIMARY_BACKFILL_TRANSFERS[1]}ULL) failure_count++;
      }} else {{
        failure_count++;
      }}
      primary_count++;
    }}
  }}

  if (span_hit_count != SELECTED13_LINEAR_RANK_EXPECTED_SPAN_HIT_COUNT) failure_count++;
  if (backfill_all_family_in_span_count != SELECTED13_LINEAR_RANK_FULL_FAMILY_BACKFILL_COUNT) failure_count++;
  if (primary_count != 2) failure_count++;

  printf("{{");
  printf("\\\"exact_mask_count\\\":%llu,", (unsigned long long)exact_mask_count);
  printf("\\\"unique_exact_mask_count\\\":%llu,", (unsigned long long)unique_exact_mask_count);
  printf("\\\"unique_seen_count\\\":%llu,", (unsigned long long)unique_seen_count);
  printf("\\\"exact_rank\\\":%llu,", (unsigned long long)exact_rank);
  printf("\\\"unique_rank\\\":%llu,", (unsigned long long)unique_rank);
  printf("\\\"backfill_count\\\":%llu,", (unsigned long long)backfill_count);
  printf("\\\"span_hit_count\\\":%llu,", (unsigned long long)span_hit_count);
  printf("\\\"backfill_all_family_in_span_count\\\":%llu,", (unsigned long long)backfill_all_family_in_span_count);
  printf("\\\"primary_count\\\":%llu,", (unsigned long long)primary_count);
  printf("\\\"primary_transfer0\\\":%llu,", (unsigned long long)primary_transfer0);
  printf("\\\"primary_transfer1\\\":%llu,", (unsigned long long)primary_transfer1);
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
    with tempfile.TemporaryDirectory(prefix="ecdlp_selected13_linear_rank_", dir=str(temp_root)) as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "selected13_linear_rank_preflight.c"
        exe_path = temp_path / "selected13_linear_rank_preflight"
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


def compare_native(
    exact_masks: list[int],
    unique_exact_masks: list[int],
    exact_rank: int,
    backfill_items: list[dict[str, Any]],
    native: dict[str, Any],
) -> list[dict[str, Any]]:
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
    span_hit_count = sum(
        1
        for item in backfill_items
        for hit in item.get("family_masks_in_exact_span") or []
        if bool(hit)
    )
    all_span_count = sum(1 for item in backfill_items if bool(item.get("all_family_masks_in_exact_span")))
    if as_int(summary.get("exact_mask_count"), -1) != len(exact_masks):
        failures.append({"code": "native_exact_mask_count_mismatch"})
    if as_int(summary.get("unique_exact_mask_count"), -1) != len(unique_exact_masks):
        failures.append({"code": "native_unique_exact_mask_count_mismatch"})
    if as_int(summary.get("exact_rank"), -1) != exact_rank:
        failures.append({"code": "native_exact_rank_mismatch"})
    if as_int(summary.get("backfill_count"), -1) != len(backfill_items):
        failures.append({"code": "native_backfill_count_mismatch"})
    if as_int(summary.get("span_hit_count"), -1) != span_hit_count:
        failures.append({"code": "native_span_hit_count_mismatch"})
    if as_int(summary.get("backfill_all_family_in_span_count"), -1) != all_span_count:
        failures.append({"code": "native_all_span_count_mismatch"})
    if as_int(summary.get("primary_transfer0"), -1) != PRIMARY_BACKFILL_TRANSFERS[0]:
        failures.append({"code": "native_primary_transfer0_mismatch"})
    if as_int(summary.get("primary_transfer1"), -1) != PRIMARY_BACKFILL_TRANSFERS[1]:
        failures.append({"code": "native_primary_transfer1_mismatch"})
    if as_int(summary.get("failure_count"), -1) != 0:
        failures.append({"code": "native_preflight_reported_failures", "native_summary": summary})
    return failures


def summarize(
    exact_masks: list[int],
    unique_exact_masks: list[int],
    exact_rank: int,
    backfill_items: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    native_summary: dict[str, Any],
) -> dict[str, Any]:
    span_hit_count = sum(
        1
        for item in backfill_items
        for hit in item.get("family_masks_in_exact_span") or []
        if bool(hit)
    )
    all_span_count = sum(1 for item in backfill_items if bool(item.get("all_family_masks_in_exact_span")))
    return {
        "backfill_all_family_masks_in_exact_span_count": all_span_count,
        "backfill_family_mask_count": len(backfill_items) * EXPECTED_FAMILY_MASKS_PER_BACKFILL,
        "backfill_family_masks_in_exact_span_count": span_hit_count,
        "exact_form_mask_count": len(exact_masks),
        "exact_form_mask_rank": exact_rank,
        "full_family_backfill_count": len(backfill_items),
        "mask_span_interpretation": (
            "Full-family backfill support masks are already covered by the exact replay GF(2) support span; "
            "9981 and 9943 still require actual FFE/summation-polynomial direct/rank export."
        ),
        "native_preflight_failure_count": as_int(native_summary.get("failure_count"), -1),
        "native_preflight_verified": as_int(native_summary.get("failure_count"), -1) == 0,
        "neutral_included_in_backfill_rank_claim": False,
        "neutral_transfer_count": len({as_int(item.get("transfer_index"), -1) for item in controls}),
        "neutral_work_item_count": len(controls),
        "primary_backfill_transfers": [
            as_int(item.get("transfer_index"), -1) for item in backfill_items if bool(item.get("is_primary"))
        ],
        "unique_exact_form_mask_count": len(unique_exact_masks),
        "unique_exact_form_masks": sorted(unique_exact_masks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-replay", type=Path, default=DEFAULT_EXACT_REPLAY)
    parser.add_argument("--worklist", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--no-c-header", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exact_replay = load_json(args.exact_replay)
    worklist = load_json(args.worklist)
    records = exact_records(exact_replay)
    items = work_items(worklist)
    exact_mask_records = build_exact_mask_records(records)
    exact_masks = [as_int(record.get("form_mask")) for record in exact_mask_records]
    unique_exact_masks = unique_first_seen(exact_masks)
    exact_rank = gf2_rank64(exact_masks)
    backfill_items = build_backfill_items(items, exact_masks)
    controls = neutral_controls(items)

    failures = validate_sources(exact_replay, worklist)
    failures.extend(validate_rank_shape(exact_masks, unique_exact_masks, exact_rank, backfill_items, controls))

    native: dict[str, Any] = {"compiled": False, "executed": False, "native_summary": {}}
    if not args.no_c_header:
        args.c_header_out.parent.mkdir(parents=True, exist_ok=True)
        args.c_header_out.write_text(render_c_header(exact_masks, unique_exact_masks, backfill_items))
        native = run_native_preflight(args.c_header_out, args.cc)
        failures.extend(compare_native(exact_masks, unique_exact_masks, exact_rank, backfill_items, native))

    native_summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    verified = not failures
    payload = {
        "artifacts": {
            "c_header": None if args.no_c_header else str(args.c_header_out),
            "exact_replay": str(args.exact_replay),
            "worklist": str(args.worklist),
        },
        "backfill_mask_items": backfill_items,
        "claim_status": (
            "FFE_SHARP_LANE_LINEAR_RANK_PREFLIGHT_READY"
            if verified
            else "FFE_SHARP_LANE_LINEAR_RANK_PREFLIGHT_FAILED_CHECK"
        ),
        "created_at": now_iso(),
        "exact_mask_records": exact_mask_records,
        "failures": failures,
        "honesty_boundary": [
            "This is a GF(2) support-mask rank/span preflight for selected13 sharp-lane rows.",
            "It is a negative guard against confusing support-mask coverage with new direct/rank relation export.",
            "It does not evaluate summation polynomials, solve finite-field equations, export direct/rank rows, solve ECDLP, or claim a Pollard-rho speedup.",
        ],
        "native_preflight": native,
        "neutral_controls": controls,
        "parameters": {
            "expected_exact_form_mask_count": EXPECTED_EXACT_FORM_MASK_COUNT,
            "expected_exact_form_mask_rank": EXPECTED_EXACT_FORM_MASK_RANK,
            "expected_full_family_backfill_count": EXPECTED_FULL_FAMILY_BACKFILL_COUNT,
            "expected_unique_exact_form_mask_count": EXPECTED_UNIQUE_EXACT_FORM_MASK_COUNT,
            "primary_backfill_transfers": PRIMARY_BACKFILL_TRANSFERS,
            "selected13_mask": SELECTED13_MASK,
        },
        "schema": SCHEMA,
        "summary": summarize(exact_masks, unique_exact_masks, exact_rank, backfill_items, controls, native_summary),
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
