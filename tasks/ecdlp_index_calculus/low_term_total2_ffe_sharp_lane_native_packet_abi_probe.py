#!/usr/bin/env python3
"""Compile and verify the selected13 sharp-lane packet ABI header.

The packet manifest lowers the strict FFE kernel contract into a compact C
header.  This probe compiles a tiny native verifier against that header and
checks that the exported arrays preserve packet counts, class-code accounting,
row-window layout, selected13 support, backfill transfer coverage, full-family
row masks, and the neutral exported control packet.

This is an ABI gate only.  It does not evaluate summation polynomials, export
direct/rank rows, or claim an ECDLP recovery.
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


SCHEMA = "ecdlp.low_term_total2_ffe_sharp_lane_native_packet_abi_probe.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_PACKET_MANIFEST = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_kernel_packet_manifest_selected13_9696_9999_probe.json"
)
DEFAULT_C_HEADER = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_kernel_packet_manifest_selected13_9696_9999_probe.h"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_native_packet_abi_selected13_9696_9999_probe.json"
)
FULL_FAMILY_MASKS = [33, 17408, 34816]
SELECTED13_MASK = 1 << 13
REPLAY_CODE_NAMES = {
    1: "direct_rank_backfill",
    2: "inherited_positive_replay",
    3: "exact_positive_replay",
    4: "neutral_exported_replay",
}
ROW_CODE_NAMES = {
    1: "direct_rank_backfill_row",
    2: "inherited_promotion_row",
    3: "exact_positive_row",
    4: "neutral_exported_row",
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


def code_counter(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(as_int(item.get(key)) for item in items)
    return {str(code): int(counts.get(code, 0)) for code in range(1, 5)}


def nonzero_code_counter(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(as_int(item.get(key)) for item in items)
    return {str(code): int(count) for code, count in sorted(counts.items()) if code != 0 and count}


def packet_transfer(packet: dict[str, Any]) -> int:
    return as_int((packet.get("public_first_pass") or {}).get("transfer_index"), -1)


def expected_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    packets = [packet for packet in manifest.get("packets") or [] if isinstance(packet, dict)]
    rows = [row for row in manifest.get("rows") or [] if isinstance(row, dict)]
    backfill_transfers = [packet_transfer(packet) for packet in packets if as_int(packet.get("class_code")) == 1]
    exact_transfers = [packet_transfer(packet) for packet in packets if as_int(packet.get("class_code")) == 3]
    neutral_transfers = [packet_transfer(packet) for packet in packets if as_int(packet.get("class_code")) == 4]
    return {
        "backfill_transfers": backfill_transfers,
        "exact_transfers": exact_transfers,
        "full_family_backfill_transfers": [
            as_int(row.get("transfer_index"), -1)
            for row in rows
            if bool(row.get("is_full_family_backfill"))
        ],
        "neutral_transfers": neutral_transfers,
        "packet_count": len(packets),
        "replay_class_counts": code_counter(packets, "class_code"),
        "replay_class_counts_nonzero": nonzero_code_counter(packets, "class_code"),
        "row_class_counts": code_counter(rows, "class_code"),
        "row_class_counts_nonzero": nonzero_code_counter(rows, "class_code"),
        "row_count": len(rows),
    }


def json_manifest_failures(manifest: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    packets = [packet for packet in manifest.get("packets") or [] if isinstance(packet, dict)]
    rows = [row for row in manifest.get("rows") or [] if isinstance(row, dict)]
    if manifest.get("claim_status") != "FFE_SHARP_LANE_KERNEL_PACKET_MANIFEST_READY":
        failures.append({"code": "packet_manifest_not_ready", "claim_status": manifest.get("claim_status")})
    if manifest.get("failures"):
        failures.append({"code": "packet_manifest_has_failures", "failures": manifest.get("failures")})

    cursor = 0
    for packet_index, packet in enumerate(packets):
        offset = as_int(packet.get("global_row_offset"), -1)
        count = as_int(packet.get("row_count"), -1)
        if as_int(packet.get("packet_index"), -1) != packet_index:
            failures.append(
                {
                    "code": "json_packet_index_mismatch",
                    "expected": packet_index,
                    "observed": packet.get("packet_index"),
                }
            )
        if offset != cursor:
            failures.append(
                {
                    "code": "json_packet_row_offset_not_contiguous",
                    "expected": cursor,
                    "observed": offset,
                    "packet_index": packet_index,
                }
            )
        packet_rows = rows[offset : offset + count] if offset >= 0 and count >= 0 else []
        if len(packet_rows) != count:
            failures.append({"code": "json_packet_row_window_out_of_range", "packet_index": packet_index})
        for local_offset, row in enumerate(packet_rows):
            if as_int(row.get("packet_index"), -1) != packet_index:
                failures.append(
                    {
                        "code": "json_row_packet_index_mismatch",
                        "packet_index": packet_index,
                        "row_id": row.get("row_id"),
                    }
                )
            if as_int(row.get("global_row_offset"), -1) != offset + local_offset:
                failures.append(
                    {
                        "code": "json_row_global_offset_mismatch",
                        "packet_index": packet_index,
                        "row_id": row.get("row_id"),
                    }
                )
            if not (as_int(row.get("selected_support_mask")) & SELECTED13_MASK):
                failures.append(
                    {
                        "code": "json_row_missing_selected13",
                        "packet_index": packet_index,
                        "row_id": row.get("row_id"),
                    }
                )
        if as_int(packet.get("class_code")) == 1:
            full_family = [row for row in packet_rows if bool(row.get("is_full_family_backfill"))]
            if len(full_family) != 1:
                failures.append(
                    {
                        "code": "json_backfill_packet_full_family_slot_count_mismatch",
                        "full_family_count": len(full_family),
                        "packet_index": packet_index,
                        "transfer_index": packet_transfer(packet),
                    }
                )
            elif [as_int(value) for value in full_family[0].get("family_masks") or []] != FULL_FAMILY_MASKS:
                failures.append(
                    {
                        "code": "json_backfill_packet_full_family_masks_mismatch",
                        "family_masks": full_family[0].get("family_masks"),
                        "packet_index": packet_index,
                        "transfer_index": packet_transfer(packet),
                    }
                )
        if packet_transfer(packet) == 9969:
            for row in packet_rows:
                if as_int(row.get("class_code")) != 4:
                    failures.append({"code": "json_neutral_9969_row_class_mismatch", "row_id": row.get("row_id")})
                if row.get("requires_direct_rank_export") or row.get("requires_exact_certificate_before_promotion"):
                    failures.append({"code": "json_neutral_9969_row_has_export_or_promotion_gate", "row_id": row.get("row_id")})
        cursor += count

    if cursor != len(rows):
        failures.append({"code": "json_rows_not_fully_covered", "covered": cursor, "row_count": len(rows)})
    if expected["backfill_transfers"] != expected["full_family_backfill_transfers"]:
        failures.append(
            {
                "code": "json_backfill_transfers_do_not_match_full_family_rows",
                "backfill_transfers": expected["backfill_transfers"],
                "full_family_backfill_transfers": expected["full_family_backfill_transfers"],
            }
        )
    if expected["neutral_transfers"] != [9969]:
        failures.append({"code": "json_neutral_transfer_list_mismatch", "neutral_transfers": expected["neutral_transfers"]})
    return failures


def c_u64_array(values: list[int]) -> str:
    if not values:
        return "{0ULL}"
    return "{" + ", ".join(f"{as_int(value)}ULL" for value in values) + "}"


def render_c_source(header_basename: str, expected: dict[str, Any]) -> str:
    expected_backfills = [as_int(value) for value in expected["backfill_transfers"]]
    mask0, mask1, mask2 = FULL_FAMILY_MASKS
    backfill_count = len(expected_backfills)
    return f"""#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

static const uint64_t EXPECTED_BACKFILL_TRANSFERS[] = {c_u64_array(expected_backfills)};

int main(void) {{
  const uint64_t selected13_mask = {SELECTED13_MASK}ULL;
  const uint64_t full_family_masks[3] = {{{mask0}ULL, {mask1}ULL, {mask2}ULL}};
  const size_t packet_array_count =
      sizeof(SELECTED13_SHARP_FIRST_PASS_PACKETS) / sizeof(SELECTED13_SHARP_FIRST_PASS_PACKETS[0]);
  const size_t row_array_count =
      sizeof(SELECTED13_SHARP_SECOND_PASS_ROWS) / sizeof(SELECTED13_SHARP_SECOND_PASS_ROWS[0]);
  uint64_t replay_counts[5] = {{0, 0, 0, 0, 0}};
  uint64_t row_counts[5] = {{0, 0, 0, 0, 0}};
  uint64_t cursor = 0;
  uint64_t failure_count = 0;
  uint64_t row_index_mismatch_count = 0;
  uint64_t selected13_missing_count = 0;
  uint64_t backfill_transfer_mismatch_count = 0;
  uint64_t backfill_packet_seen_count = 0;
  uint64_t backfill_full_family_packet_count = 0;
  uint64_t neutral_9969_seen_count = 0;
  uint64_t neutral_9969_ok_count = 0;
  uint64_t neutral_bad_flag_count = 0;

  if (packet_array_count != SELECTED13_SHARP_PACKET_COUNT) failure_count++;
  if (row_array_count != SELECTED13_SHARP_ROW_COUNT) failure_count++;

  for (size_t i = 0; i < packet_array_count; i++) {{
    const selected13_sharp_first_pass_packet_t *packet = &SELECTED13_SHARP_FIRST_PASS_PACKETS[i];
    const uint64_t replay_code = packet->replay_class_code;
    if (replay_code < 5) replay_counts[replay_code]++;
    if (packet->row_offset != cursor) failure_count++;
    if (packet->row_offset + packet->row_count > row_array_count) failure_count++;

    if (replay_code == 1) {{
      uint64_t expected_transfer = 0;
      if (backfill_packet_seen_count < {backfill_count}ULL) {{
        expected_transfer = EXPECTED_BACKFILL_TRANSFERS[backfill_packet_seen_count];
      }}
      if (backfill_packet_seen_count >= {backfill_count}ULL ||
          packet->transfer_index != expected_transfer) {{
        backfill_transfer_mismatch_count++;
      }}
      backfill_packet_seen_count++;
    }}

    uint64_t packet_full_family_rows = 0;
    uint64_t packet_neutral_ok = 1;
    for (uint64_t j = 0; j < packet->row_count; j++) {{
      const uint64_t row_index = packet->row_offset + j;
      const selected13_sharp_second_pass_row_t *row = &SELECTED13_SHARP_SECOND_PASS_ROWS[row_index];
      const uint64_t row_code = row->class_code;
      if (row->packet_index != i) row_index_mismatch_count++;
      if ((row->selected_support_mask & selected13_mask) == 0) selected13_missing_count++;
      if (row_code < 5) row_counts[row_code]++;
      if (replay_code == 1 &&
          row->family_masks[0] == full_family_masks[0] &&
          row->family_masks[1] == full_family_masks[1] &&
          row->family_masks[2] == full_family_masks[2] &&
          row->class_code == 1 &&
          row->requires_direct_rank_export == 1 &&
          row->requires_exact_certificate_before_promotion == 0) {{
        packet_full_family_rows++;
      }}
      if (packet->transfer_index == 9969ULL) {{
        if (replay_code != 4 ||
            row->class_code != 4 ||
            row->requires_direct_rank_export != 0 ||
            row->requires_exact_certificate_before_promotion != 0) {{
          packet_neutral_ok = 0;
          neutral_bad_flag_count++;
        }}
      }}
    }}
    if (replay_code == 1 && packet_full_family_rows == 1) {{
      backfill_full_family_packet_count++;
    }} else if (replay_code == 1) {{
      failure_count++;
    }}
    if (packet->transfer_index == 9969ULL) {{
      neutral_9969_seen_count++;
      if (packet_neutral_ok) neutral_9969_ok_count++;
    }}
    cursor += packet->row_count;
  }}

  if (cursor != row_array_count) failure_count++;
  if (row_index_mismatch_count != 0) failure_count++;
  if (selected13_missing_count != 0) failure_count++;
  if (backfill_packet_seen_count != {backfill_count}ULL) failure_count++;
  if (backfill_transfer_mismatch_count != 0) failure_count++;
  if (backfill_full_family_packet_count != {backfill_count}ULL) failure_count++;
  if (neutral_9969_seen_count != 1 || neutral_9969_ok_count != 1 || neutral_bad_flag_count != 0) {{
    failure_count++;
  }}

  printf("{{");
  printf("\\\"packet_count\\\":%llu,", (unsigned long long)packet_array_count);
  printf("\\\"row_count\\\":%llu,", (unsigned long long)row_array_count);
  printf("\\\"covered_row_count\\\":%llu,", (unsigned long long)cursor);
  printf("\\\"failure_count\\\":%llu,", (unsigned long long)failure_count);
  printf("\\\"row_index_mismatch_count\\\":%llu,", (unsigned long long)row_index_mismatch_count);
  printf("\\\"selected13_missing_count\\\":%llu,", (unsigned long long)selected13_missing_count);
  printf("\\\"backfill_packet_seen_count\\\":%llu,", (unsigned long long)backfill_packet_seen_count);
  printf("\\\"backfill_transfer_mismatch_count\\\":%llu,", (unsigned long long)backfill_transfer_mismatch_count);
  printf("\\\"backfill_full_family_packet_count\\\":%llu,", (unsigned long long)backfill_full_family_packet_count);
  printf("\\\"neutral_9969_seen_count\\\":%llu,", (unsigned long long)neutral_9969_seen_count);
  printf("\\\"neutral_9969_ok_count\\\":%llu,", (unsigned long long)neutral_9969_ok_count);
  printf("\\\"neutral_bad_flag_count\\\":%llu,", (unsigned long long)neutral_bad_flag_count);
  printf("\\\"replay_class_counts\\\":{{\\\"1\\\":%llu,\\\"2\\\":%llu,\\\"3\\\":%llu,\\\"4\\\":%llu}},",
         (unsigned long long)replay_counts[1],
         (unsigned long long)replay_counts[2],
         (unsigned long long)replay_counts[3],
         (unsigned long long)replay_counts[4]);
  printf("\\\"row_class_counts\\\":{{\\\"1\\\":%llu,\\\"2\\\":%llu,\\\"3\\\":%llu,\\\"4\\\":%llu}}",
         (unsigned long long)row_counts[1],
         (unsigned long long)row_counts[2],
         (unsigned long long)row_counts[3],
         (unsigned long long)row_counts[4]);
  printf("}}\\n");
  return failure_count == 0 ? 0 : 1;
}}
"""


def run_native_probe(header: Path, expected: dict[str, Any], compiler: str) -> dict[str, Any]:
    c_source = render_c_source(header.name, expected)
    source_hash = hashlib.sha256(c_source.encode("utf-8")).hexdigest()
    temp_root = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path(tempfile.gettempdir())
    env = os.environ.copy()
    env["TMPDIR"] = str(temp_root)
    with tempfile.TemporaryDirectory(prefix="ecdlp_selected13_abi_", dir=str(temp_root)) as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "selected13_native_packet_abi_probe.c"
        exe_path = temp_path / "selected13_native_packet_abi_probe"
        c_path.write_text(c_source)
        command = [
            compiler,
            "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            str(header.parent),
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


def compare_native(expected: dict[str, Any], native: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    if not native.get("compiled"):
        failures.append(
            {
                "code": "native_abi_probe_compile_failed",
                "compile_returncode": native.get("compile_returncode"),
                "compile_stderr": native.get("compile_stderr"),
            }
        )
        return failures
    if not native.get("executed"):
        failures.append({"code": "native_abi_probe_not_executed"})
        return failures
    if as_int(native.get("run_returncode"), -1) != 0:
        failures.append(
            {
                "code": "native_abi_probe_run_failed",
                "run_returncode": native.get("run_returncode"),
                "run_stderr": native.get("run_stderr"),
                "run_stdout": native.get("run_stdout"),
            }
        )
    if not summary:
        failures.append({"code": "native_abi_probe_stdout_not_json", "run_stdout": native.get("run_stdout")})
        return failures
    if as_int(summary.get("packet_count"), -1) != as_int(expected.get("packet_count")):
        failures.append({"code": "native_packet_count_mismatch", "native": summary.get("packet_count"), "expected": expected.get("packet_count")})
    if as_int(summary.get("row_count"), -1) != as_int(expected.get("row_count")):
        failures.append({"code": "native_row_count_mismatch", "native": summary.get("row_count"), "expected": expected.get("row_count")})
    if as_int(summary.get("covered_row_count"), -1) != as_int(expected.get("row_count")):
        failures.append({"code": "native_row_coverage_mismatch", "native": summary.get("covered_row_count"), "expected": expected.get("row_count")})
    if (summary.get("replay_class_counts") or {}) != expected.get("replay_class_counts"):
        failures.append(
            {
                "code": "native_replay_class_counts_mismatch",
                "native": summary.get("replay_class_counts"),
                "expected": expected.get("replay_class_counts"),
            }
        )
    if (summary.get("row_class_counts") or {}) != expected.get("row_class_counts"):
        failures.append(
            {
                "code": "native_row_class_counts_mismatch",
                "native": summary.get("row_class_counts"),
                "expected": expected.get("row_class_counts"),
            }
        )
    if as_int(summary.get("failure_count"), -1) != 0:
        failures.append({"code": "native_reported_abi_failures", "native_summary": summary})
    if as_int(summary.get("backfill_full_family_packet_count"), -1) != len(expected.get("backfill_transfers") or []):
        failures.append(
            {
                "code": "native_backfill_full_family_count_mismatch",
                "native": summary.get("backfill_full_family_packet_count"),
                "expected": len(expected.get("backfill_transfers") or []),
            }
        )
    if as_int(summary.get("neutral_9969_ok_count"), -1) != 1:
        failures.append({"code": "native_neutral_9969_not_verified", "native": summary})
    return failures


def named_count_summary(code_counts: dict[str, int], names: dict[int, str]) -> dict[str, int]:
    return {
        names[as_int(code)]: count
        for code, count in sorted(code_counts.items(), key=lambda item: as_int(item[0]))
        if as_int(code) in names and count
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-manifest", type=Path, default=DEFAULT_PACKET_MANIFEST)
    parser.add_argument("--c-header", type=Path, default=DEFAULT_C_HEADER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_json(args.packet_manifest)
    expected = expected_from_manifest(manifest)
    native = run_native_probe(args.c_header, expected, args.cc)
    failures = json_manifest_failures(manifest, expected)
    failures.extend(compare_native(expected, native))
    native_summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    verified = not failures
    payload = {
        "artifacts": {
            "c_header": str(args.c_header),
            "packet_manifest": str(args.packet_manifest),
        },
        "claim_status": (
            "FFE_SHARP_LANE_NATIVE_PACKET_ABI_VERIFIED"
            if verified
            else "FFE_SHARP_LANE_NATIVE_PACKET_ABI_FAILED_CHECK"
        ),
        "created_at": now_iso(),
        "expected": {
            "backfill_transfers": expected["backfill_transfers"],
            "exact_transfers": expected["exact_transfers"],
            "full_family_backfill_transfers": expected["full_family_backfill_transfers"],
            "full_family_masks": FULL_FAMILY_MASKS,
            "neutral_transfers": expected["neutral_transfers"],
            "replay_class_counts": named_count_summary(expected["replay_class_counts"], REPLAY_CODE_NAMES),
            "row_class_counts": named_count_summary(expected["row_class_counts"], ROW_CODE_NAMES),
        },
        "failures": failures,
        "honesty_boundary": [
            "This verifies the native C ABI shape for selected13 sharp-lane packets only.",
            "It does not evaluate summation polynomials or finite-field equations.",
            "It does not export direct/rank rows, solve an ECDLP instance, or claim a Pollard-rho speedup.",
        ],
        "native_execution": native,
        "schema": SCHEMA,
        "summary": {
            "backfill_full_family_packet_count": as_int(native_summary.get("backfill_full_family_packet_count"), -1),
            "backfill_transfer_count": len(expected["backfill_transfers"]),
            "compiled": bool(native.get("compiled")),
            "exact_positive_packet_count": len(expected["exact_transfers"]),
            "native_failure_count": as_int(native_summary.get("failure_count"), -1),
            "native_packet_count": as_int(native_summary.get("packet_count"), -1),
            "native_row_count": as_int(native_summary.get("row_count"), -1),
            "neutral_9969_ok_count": as_int(native_summary.get("neutral_9969_ok_count"), -1),
            "neutral_packet_count": len(expected["neutral_transfers"]),
            "packet_count": expected["packet_count"],
            "replay_class_counts": named_count_summary(expected["replay_class_counts"], REPLAY_CODE_NAMES),
            "row_class_counts": named_count_summary(expected["row_class_counts"], ROW_CODE_NAMES),
            "row_count": expected["row_count"],
            "selected13_missing_count": as_int(native_summary.get("selected13_missing_count"), -1),
            "verified": verified,
        },
    }
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
