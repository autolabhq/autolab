#!/usr/bin/env python3
"""Emit a row-level execution worklist for the selected13 sharp-lane packets.

The packet ABI proves that a native worker can load the selected13 packet and
row arrays.  This script adds the next execution layer: row hashes, exact cert
hashes, phase flags, full-family backfill priorities, and inherited-promotion
dependencies.  It also emits an optional C header and compiles a native
preflight that checks the generated row work-item arrays.

This is still a worker contract.  It does not evaluate summation polynomials,
export direct/rank rows, solve an ECDLP instance, or claim a Pollard-rho
speedup.
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


SCHEMA = "ecdlp.low_term_total2_ffe_sharp_lane_execution_worklist.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_PACKET_MANIFEST = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_kernel_packet_manifest_selected13_9696_9999_probe.json"
)
DEFAULT_NATIVE_ABI = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_native_packet_abi_selected13_9696_9999_probe.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_execution_worklist_selected13_9696_9999_probe.json"
)
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_execution_worklist_selected13_9696_9999_probe.h"
)
FULL_FAMILY_MASKS = [33, 17408, 34816]
SELECTED13_MASK = 1 << 13
PRIMARY_BACKFILL_TRANSFERS = [9981, 9943]
PHASE_CODES = {
    "exact_certificate_replay": 1,
    "inherited_promotion_gate": 2,
    "direct_rank_backfill_export": 3,
    "neutral_exported_control": 4,
}
FLAG_BITS = {
    "requires_direct_rank_export": 1 << 0,
    "requires_exact_certificate_before_promotion": 1 << 1,
    "has_exact_certificate_hash": 1 << 2,
    "is_full_family_backfill": 1 << 3,
    "is_neutral_control": 1 << 4,
    "contains_selected13": 1 << 5,
    "is_primary_backfill": 1 << 6,
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


def sorted_ints(raw: Any) -> list[int]:
    return sorted(as_int(item) for item in (raw or []))


def digest_u64(raw: Any) -> int:
    text = str(raw or "")
    suffix = text.rsplit("_", 1)[-1]
    if suffix and all(ch in "0123456789abcdefABCDEF" for ch in suffix):
        return int(suffix[-16:], 16)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def phase_for_row(row: dict[str, Any]) -> str:
    check_class = str(row.get("check_class") or "")
    if check_class == "exact_positive_row":
        return "exact_certificate_replay"
    if check_class == "inherited_promotion_row":
        return "inherited_promotion_gate"
    if check_class == "direct_rank_backfill_row":
        return "direct_rank_backfill_export"
    if check_class == "neutral_exported_row":
        return "neutral_exported_control"
    return "unknown"


def row_flags(row: dict[str, Any], phase: str, transfer_index: int) -> int:
    flags = 0
    if bool(row.get("requires_direct_rank_export")):
        flags |= FLAG_BITS["requires_direct_rank_export"]
    if bool(row.get("requires_exact_certificate_before_promotion")):
        flags |= FLAG_BITS["requires_exact_certificate_before_promotion"]
    if row.get("accepted_exact_certificate_hash"):
        flags |= FLAG_BITS["has_exact_certificate_hash"]
    if bool(row.get("is_full_family_backfill")):
        flags |= FLAG_BITS["is_full_family_backfill"]
    if phase == "neutral_exported_control":
        flags |= FLAG_BITS["is_neutral_control"]
    if as_int(row.get("selected_support_mask")) & SELECTED13_MASK:
        flags |= FLAG_BITS["contains_selected13"]
    if transfer_index in PRIMARY_BACKFILL_TRANSFERS and bool(row.get("is_full_family_backfill")):
        flags |= FLAG_BITS["is_primary_backfill"]
    return flags


def packet_by_index(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        as_int(packet.get("packet_index"), -1): packet
        for packet in manifest.get("packets") or []
        if isinstance(packet, dict)
    }


def dependency_map(rows: list[dict[str, Any]]) -> dict[int, list[int]]:
    exact_by_packet: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        if str(row.get("check_class")) == "exact_positive_row":
            exact_by_packet.setdefault(as_int(row.get("packet_index"), -1), []).append(index)

    out: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        if bool(row.get("requires_exact_certificate_before_promotion")):
            packet_index = as_int(row.get("packet_index"), -1)
            out[index] = exact_by_packet.get(packet_index, [])
        else:
            out[index] = []
    return out


def build_dependencies(dep_map: dict[int, list[int]]) -> tuple[list[int], dict[int, tuple[int, int]]]:
    flat: list[int] = []
    spans: dict[int, tuple[int, int]] = {}
    for work_item_index in sorted(dep_map):
        deps = [as_int(item) for item in dep_map[work_item_index]]
        spans[work_item_index] = (len(flat), len(deps))
        flat.extend(deps)
    return flat, spans


def build_work_items(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    rows = [row for row in manifest.get("rows") or [] if isinstance(row, dict)]
    packets = packet_by_index(manifest)
    dep_map = dependency_map(rows)
    dependencies, dep_spans = build_dependencies(dep_map)
    work_items = []
    for index, row in enumerate(rows):
        packet = packets.get(as_int(row.get("packet_index"), -1), {})
        public = packet.get("public_first_pass") or {}
        salts = sorted_ints(public.get("salts"))
        while len(salts) < 2:
            salts.append(0)
        family_masks = [as_int(item) for item in (row.get("family_masks") or [])[:3]]
        while len(family_masks) < 3:
            family_masks.append(0)
        transfer_index = as_int(row.get("transfer_index"), -1)
        phase = phase_for_row(row)
        dep_start, dep_count = dep_spans[index]
        flags = row_flags(row, phase, transfer_index)
        work_items.append(
            {
                "accepted_exact_certificate_hash": row.get("accepted_exact_certificate_hash"),
                "accepted_exact_certificate_hash_u64": digest_u64(row.get("accepted_exact_certificate_hash"))
                if row.get("accepted_exact_certificate_hash")
                else 0,
                "class_code": as_int(row.get("class_code")),
                "dependency_count": dep_count,
                "dependency_indices": dep_map[index],
                "dependency_start": dep_start,
                "direct_status": row.get("direct_status"),
                "family_masks": family_masks,
                "first_pass_id": row.get("first_pass_id"),
                "flags": flags,
                "group_id": row.get("group_id"),
                "is_full_family_backfill": bool(row.get("is_full_family_backfill")),
                "packet_hash": packet.get("packet_hash"),
                "packet_hash_u64": digest_u64(packet.get("packet_hash")),
                "packet_index": as_int(row.get("packet_index"), -1),
                "packet_row_offset": as_int(row.get("packet_row_offset"), -1),
                "phase": phase,
                "phase_code": PHASE_CODES.get(phase, 0),
                "requires_direct_rank_export": bool(row.get("requires_direct_rank_export")),
                "requires_exact_certificate_before_promotion": bool(
                    row.get("requires_exact_certificate_before_promotion")
                ),
                "row_check_hash": row.get("row_check_hash"),
                "row_check_hash_u64": digest_u64(row.get("row_check_hash")),
                "row_id": row.get("row_id"),
                "row_id_u64": digest_u64(row.get("row_id")),
                "salt_gap": as_int(public.get("salt_gap")),
                "salt_min_mod4": as_int(public.get("salt_min_mod4")),
                "salts": salts[:2],
                "score_certificate_source": row.get("score_certificate_source"),
                "selected_support_mask": as_int(row.get("selected_support_mask")),
                "selector": row.get("selector"),
                "selector_u64": digest_u64(row.get("selector")),
                "target": public.get("target"),
                "top_k": as_int(row.get("top_k")),
                "transfer_index": transfer_index,
                "work_item_index": index,
            }
        )
    return work_items, dependencies


def execution_order_key(item: dict[str, Any]) -> tuple[int, int, int, int]:
    phase = str(item.get("phase") or "")
    transfer = as_int(item.get("transfer_index"), -1)
    full_family = bool(item.get("is_full_family_backfill"))
    primary = transfer in PRIMARY_BACKFILL_TRANSFERS and full_family
    phase_rank = {
        "exact_certificate_replay": 0,
        "inherited_promotion_gate": 1,
        "direct_rank_backfill_export": 2,
        "neutral_exported_control": 3,
    }.get(phase, 9)
    backfill_rank = 0 if primary else 1 if full_family else 2
    primary_rank = PRIMARY_BACKFILL_TRANSFERS.index(transfer) if transfer in PRIMARY_BACKFILL_TRANSFERS else 99
    return (phase_rank, backfill_rank, primary_rank, as_int(item.get("work_item_index"), -1))


def validate_sources(packet_manifest: dict[str, Any], native_abi: dict[str, Any] | None) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if packet_manifest.get("claim_status") != "FFE_SHARP_LANE_KERNEL_PACKET_MANIFEST_READY":
        failures.append(
            {
                "code": "packet_manifest_not_ready",
                "claim_status": packet_manifest.get("claim_status"),
            }
        )
    if packet_manifest.get("failures"):
        failures.append({"code": "packet_manifest_has_failures", "failures": packet_manifest.get("failures")})
    if native_abi is None:
        failures.append({"code": "native_abi_source_missing"})
    elif native_abi.get("claim_status") != "FFE_SHARP_LANE_NATIVE_PACKET_ABI_VERIFIED":
        failures.append({"code": "native_abi_not_verified", "claim_status": native_abi.get("claim_status")})
    elif (native_abi.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "native_abi_summary_not_verified", "summary": native_abi.get("summary")})
    return failures


def validate_worklist(work_items: list[dict[str, Any]], dependencies: list[int]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    row_hashes = [as_int(item.get("row_check_hash_u64")) for item in work_items]
    if len(row_hashes) != len(set(row_hashes)):
        failures.append({"code": "duplicate_row_check_hash_u64"})
    for expected_index, item in enumerate(work_items):
        index = as_int(item.get("work_item_index"), -1)
        phase = str(item.get("phase") or "")
        flags = as_int(item.get("flags"))
        if index != expected_index:
            failures.append({"code": "work_item_index_mismatch", "expected": expected_index, "observed": index})
        if as_int(item.get("row_check_hash_u64")) == 0:
            failures.append({"code": "zero_row_check_hash_u64", "work_item_index": index})
        if not (as_int(item.get("selected_support_mask")) & SELECTED13_MASK):
            failures.append({"code": "work_item_missing_selected13", "work_item_index": index})
        if not (flags & FLAG_BITS["contains_selected13"]):
            failures.append({"code": "work_item_missing_selected13_flag", "work_item_index": index})
        if phase == "exact_certificate_replay":
            if as_int(item.get("accepted_exact_certificate_hash_u64")) == 0:
                failures.append({"code": "exact_work_item_missing_cert_hash", "work_item_index": index})
            if bool(item.get("requires_direct_rank_export")):
                failures.append({"code": "exact_work_item_marked_for_export", "work_item_index": index})
        elif phase == "inherited_promotion_gate":
            if as_int(item.get("dependency_count")) < 1:
                failures.append({"code": "inherited_work_item_missing_dependency", "work_item_index": index})
            if not bool(item.get("requires_exact_certificate_before_promotion")):
                failures.append({"code": "inherited_work_item_missing_gate_flag", "work_item_index": index})
        elif phase == "direct_rank_backfill_export":
            if not bool(item.get("requires_direct_rank_export")):
                failures.append({"code": "backfill_work_item_missing_export_flag", "work_item_index": index})
            if as_int(item.get("accepted_exact_certificate_hash_u64")) != 0:
                failures.append({"code": "backfill_work_item_has_cert_hash", "work_item_index": index})
        elif phase == "neutral_exported_control":
            if bool(item.get("requires_direct_rank_export")) or bool(
                item.get("requires_exact_certificate_before_promotion")
            ):
                failures.append({"code": "neutral_work_item_has_action_gate", "work_item_index": index})
            if as_int(item.get("transfer_index")) != 9969:
                failures.append({"code": "neutral_work_item_wrong_transfer", "work_item_index": index})
        else:
            failures.append({"code": "unknown_work_item_phase", "work_item_index": index, "phase": phase})

        start = as_int(item.get("dependency_start"), -1)
        count = as_int(item.get("dependency_count"), -1)
        if start < 0 or count < 0 or start + count > len(dependencies):
            failures.append({"code": "dependency_span_out_of_range", "work_item_index": index})
        for dep in dependencies[start : start + count]:
            if dep < 0 or dep >= len(work_items):
                failures.append({"code": "dependency_index_out_of_range", "work_item_index": index, "dependency": dep})
            elif work_items[dep].get("phase") != "exact_certificate_replay":
                failures.append(
                    {
                        "code": "dependency_not_exact_replay",
                        "work_item_index": index,
                        "dependency": dep,
                        "dependency_phase": work_items[dep].get("phase"),
                    }
                )

    full_backfills = [
        item
        for item in work_items
        if item.get("phase") == "direct_rank_backfill_export" and bool(item.get("is_full_family_backfill"))
    ]
    primary_order = [
        as_int(item.get("transfer_index"), -1)
        for item in sorted(full_backfills, key=execution_order_key)
        if as_int(item.get("transfer_index"), -1) in PRIMARY_BACKFILL_TRANSFERS
    ]
    if primary_order != PRIMARY_BACKFILL_TRANSFERS:
        failures.append({"code": "primary_backfill_order_mismatch"})
    for item in full_backfills:
        if [as_int(value) for value in item.get("family_masks") or []] != FULL_FAMILY_MASKS:
            failures.append(
                {
                    "code": "full_family_backfill_masks_mismatch",
                    "work_item_index": item.get("work_item_index"),
                    "family_masks": item.get("family_masks"),
                }
            )
    return failures


def summarize(work_items: list[dict[str, Any]], dependencies: list[int], native_summary: dict[str, Any]) -> dict[str, Any]:
    phase_counts = Counter(str(item.get("phase")) for item in work_items)
    class_counts = Counter(as_int(item.get("class_code")) for item in work_items)
    full_backfills = [
        item
        for item in work_items
        if item.get("phase") == "direct_rank_backfill_export" and bool(item.get("is_full_family_backfill"))
    ]
    primary = [
        item
        for item in sorted(full_backfills, key=execution_order_key)
        if as_int(item.get("transfer_index"), -1) in PRIMARY_BACKFILL_TRANSFERS
    ]
    return {
        "backfill_transfer_count": len({as_int(item.get("transfer_index"), -1) for item in full_backfills}),
        "dependency_edge_count": len(dependencies),
        "execution_phase_counts": dict(sorted(phase_counts.items())),
        "full_family_backfill_work_item_count": len(full_backfills),
        "native_preflight_failure_count": as_int(native_summary.get("failure_count"), -1),
        "native_preflight_verified": as_int(native_summary.get("failure_count"), -1) == 0,
        "primary_backfill_transfers": [as_int(item.get("transfer_index"), -1) for item in primary],
        "row_class_code_counts": {str(code): int(count) for code, count in sorted(class_counts.items())},
        "selected13_missing_count": sum(
            1 for item in work_items if not (as_int(item.get("selected_support_mask")) & SELECTED13_MASK)
        ),
        "unique_row_hash_count": len({as_int(item.get("row_check_hash_u64")) for item in work_items}),
        "work_item_count": len(work_items),
    }


def c_u64_array(values: list[int]) -> str:
    if not values:
        return "{0ULL}"
    return "{" + ", ".join(f"{as_int(value)}ULL" for value in values) + "}"


def render_c_header(work_items: list[dict[str, Any]], dependencies: list[int]) -> str:
    item_lines = []
    for item in work_items:
        family_masks = [as_int(value) for value in item.get("family_masks") or []][:3]
        while len(family_masks) < 3:
            family_masks.append(0)
        salts = [as_int(value) for value in item.get("salts") or []][:2]
        while len(salts) < 2:
            salts.append(0)
        item_lines.append(
            "  {"
            f"{as_int(item.get('work_item_index'))}ULL, "
            f"{as_int(item.get('packet_index'))}ULL, "
            f"{as_int(item.get('transfer_index'))}ULL, "
            f"{c_u64_array(salts)}, "
            f"{as_int(item.get('class_code'))}ULL, "
            f"{as_int(item.get('phase_code'))}ULL, "
            f"{as_int(item.get('selected_support_mask'))}ULL, "
            f"{c_u64_array(family_masks)}, "
            f"{as_int(item.get('row_check_hash_u64'))}ULL, "
            f"{as_int(item.get('accepted_exact_certificate_hash_u64'))}ULL, "
            f"{as_int(item.get('packet_hash_u64'))}ULL, "
            f"{as_int(item.get('selector_u64'))}ULL, "
            f"{as_int(item.get('dependency_start'))}ULL, "
            f"{as_int(item.get('dependency_count'))}ULL, "
            f"{as_int(item.get('flags'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_FFE_SHARP_LANE_EXECUTION_WORKLIST_H
#define LOW_TERM_TOTAL2_FFE_SHARP_LANE_EXECUTION_WORKLIST_H

#include <stdint.h>

#define SELECTED13_SHARP_WORK_ITEM_COUNT {len(work_items)}
#define SELECTED13_SHARP_DEPENDENCY_COUNT {len(dependencies)}

#define SELECTED13_PHASE_EXACT_REPLAY 1ULL
#define SELECTED13_PHASE_INHERITED_GATE 2ULL
#define SELECTED13_PHASE_BACKFILL_EXPORT 3ULL
#define SELECTED13_PHASE_NEUTRAL_CONTROL 4ULL

#define SELECTED13_FLAG_REQUIRES_EXPORT 1ULL
#define SELECTED13_FLAG_REQUIRES_EXACT_BEFORE_PROMOTION 2ULL
#define SELECTED13_FLAG_HAS_EXACT_CERT_HASH 4ULL
#define SELECTED13_FLAG_FULL_FAMILY_BACKFILL 8ULL
#define SELECTED13_FLAG_NEUTRAL_CONTROL 16ULL
#define SELECTED13_FLAG_CONTAINS_SELECTED13 32ULL
#define SELECTED13_FLAG_PRIMARY_BACKFILL 64ULL

typedef struct {{
  uint64_t work_item_index;
  uint64_t packet_index;
  uint64_t transfer_index;
  uint64_t salts[2];
  uint64_t class_code;
  uint64_t phase_code;
  uint64_t selected_support_mask;
  uint64_t family_masks[3];
  uint64_t row_check_hash_u64;
  uint64_t accepted_exact_certificate_hash_u64;
  uint64_t packet_hash_u64;
  uint64_t selector_u64;
  uint64_t dependency_start;
  uint64_t dependency_count;
  uint64_t flags;
}} selected13_sharp_work_item_t;

static const selected13_sharp_work_item_t SELECTED13_SHARP_WORK_ITEMS[] = {{
{chr(10).join(item_lines)}
}};

static const uint64_t SELECTED13_SHARP_DEPENDENCIES[] = {c_u64_array(dependencies)};

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

int main(void) {{
  const uint64_t selected13_mask = {SELECTED13_MASK}ULL;
  const uint64_t full_masks[3] = {{{FULL_FAMILY_MASKS[0]}ULL, {FULL_FAMILY_MASKS[1]}ULL, {FULL_FAMILY_MASKS[2]}ULL}};
  uint64_t phase_counts[5] = {{0, 0, 0, 0, 0}};
  uint64_t class_counts[5] = {{0, 0, 0, 0, 0}};
  uint64_t full_family_backfill_count = 0;
  uint64_t primary_backfill_count = 0;
  uint64_t dependency_edge_count = 0;
  uint64_t selected13_missing_count = 0;
  uint64_t row_hash_zero_count = 0;
  uint64_t exact_cert_hash_count = 0;
  uint64_t neutral_control_count = 0;
  uint64_t failure_count = 0;

  const size_t item_count = sizeof(SELECTED13_SHARP_WORK_ITEMS) / sizeof(SELECTED13_SHARP_WORK_ITEMS[0]);
  const size_t dependency_count = sizeof(SELECTED13_SHARP_DEPENDENCIES) / sizeof(SELECTED13_SHARP_DEPENDENCIES[0]);
  if (item_count != SELECTED13_SHARP_WORK_ITEM_COUNT) failure_count++;
  if (dependency_count != SELECTED13_SHARP_DEPENDENCY_COUNT) failure_count++;

  for (size_t i = 0; i < item_count; i++) {{
    const selected13_sharp_work_item_t *item = &SELECTED13_SHARP_WORK_ITEMS[i];
    if (item->work_item_index != i) failure_count++;
    if (item->row_check_hash_u64 == 0) row_hash_zero_count++;
    if ((item->selected_support_mask & selected13_mask) == 0 ||
        (item->flags & SELECTED13_FLAG_CONTAINS_SELECTED13) == 0) {{
      selected13_missing_count++;
    }}
    if (item->phase_code < 5) phase_counts[item->phase_code]++;
    if (item->class_code < 5) class_counts[item->class_code]++;
    dependency_edge_count += item->dependency_count;
    if (item->dependency_start + item->dependency_count > dependency_count) failure_count++;

    if (item->phase_code == SELECTED13_PHASE_EXACT_REPLAY) {{
      if (item->accepted_exact_certificate_hash_u64 == 0 ||
          (item->flags & SELECTED13_FLAG_HAS_EXACT_CERT_HASH) == 0 ||
          (item->flags & SELECTED13_FLAG_REQUIRES_EXPORT) != 0) {{
        failure_count++;
      }}
      exact_cert_hash_count++;
    }} else if (item->phase_code == SELECTED13_PHASE_INHERITED_GATE) {{
      if (item->dependency_count == 0 ||
          (item->flags & SELECTED13_FLAG_REQUIRES_EXACT_BEFORE_PROMOTION) == 0 ||
          (item->flags & SELECTED13_FLAG_REQUIRES_EXPORT) != 0) {{
        failure_count++;
      }}
      for (uint64_t j = 0; j < item->dependency_count; j++) {{
        uint64_t dep = SELECTED13_SHARP_DEPENDENCIES[item->dependency_start + j];
        if (dep >= item_count ||
            SELECTED13_SHARP_WORK_ITEMS[dep].phase_code != SELECTED13_PHASE_EXACT_REPLAY) {{
          failure_count++;
        }}
      }}
    }} else if (item->phase_code == SELECTED13_PHASE_BACKFILL_EXPORT) {{
      if ((item->flags & SELECTED13_FLAG_REQUIRES_EXPORT) == 0 ||
          item->accepted_exact_certificate_hash_u64 != 0) {{
        failure_count++;
      }}
      if ((item->flags & SELECTED13_FLAG_FULL_FAMILY_BACKFILL) != 0) {{
        if (item->family_masks[0] != full_masks[0] ||
            item->family_masks[1] != full_masks[1] ||
            item->family_masks[2] != full_masks[2]) {{
          failure_count++;
        }}
        full_family_backfill_count++;
      }}
      if ((item->flags & SELECTED13_FLAG_PRIMARY_BACKFILL) != 0) {{
        if (item->transfer_index != 9981ULL && item->transfer_index != 9943ULL) {{
          failure_count++;
        }}
        primary_backfill_count++;
      }}
    }} else if (item->phase_code == SELECTED13_PHASE_NEUTRAL_CONTROL) {{
      if (item->transfer_index != 9969ULL ||
          (item->flags & SELECTED13_FLAG_NEUTRAL_CONTROL) == 0 ||
          (item->flags & SELECTED13_FLAG_REQUIRES_EXPORT) != 0 ||
          (item->flags & SELECTED13_FLAG_REQUIRES_EXACT_BEFORE_PROMOTION) != 0 ||
          item->accepted_exact_certificate_hash_u64 != 0) {{
        failure_count++;
      }}
      neutral_control_count++;
    }} else {{
      failure_count++;
    }}
  }}

  if (dependency_edge_count != dependency_count) failure_count++;
  if (row_hash_zero_count != 0) failure_count++;
  if (selected13_missing_count != 0) failure_count++;
  if (full_family_backfill_count != 13) failure_count++;
  if (primary_backfill_count != 2) failure_count++;
  if (exact_cert_hash_count != 6) failure_count++;
  if (neutral_control_count != 3) failure_count++;

  printf("{{");
  printf("\\\"work_item_count\\\":%llu,", (unsigned long long)item_count);
  printf("\\\"dependency_edge_count\\\":%llu,", (unsigned long long)dependency_edge_count);
  printf("\\\"failure_count\\\":%llu,", (unsigned long long)failure_count);
  printf("\\\"row_hash_zero_count\\\":%llu,", (unsigned long long)row_hash_zero_count);
  printf("\\\"selected13_missing_count\\\":%llu,", (unsigned long long)selected13_missing_count);
  printf("\\\"full_family_backfill_count\\\":%llu,", (unsigned long long)full_family_backfill_count);
  printf("\\\"primary_backfill_count\\\":%llu,", (unsigned long long)primary_backfill_count);
  printf("\\\"exact_cert_hash_count\\\":%llu,", (unsigned long long)exact_cert_hash_count);
  printf("\\\"neutral_control_count\\\":%llu,", (unsigned long long)neutral_control_count);
  printf("\\\"phase_counts\\\":{{\\\"1\\\":%llu,\\\"2\\\":%llu,\\\"3\\\":%llu,\\\"4\\\":%llu}},",
         (unsigned long long)phase_counts[1],
         (unsigned long long)phase_counts[2],
         (unsigned long long)phase_counts[3],
         (unsigned long long)phase_counts[4]);
  printf("\\\"class_counts\\\":{{\\\"1\\\":%llu,\\\"2\\\":%llu,\\\"3\\\":%llu,\\\"4\\\":%llu}}",
         (unsigned long long)class_counts[1],
         (unsigned long long)class_counts[2],
         (unsigned long long)class_counts[3],
         (unsigned long long)class_counts[4]);
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
    with tempfile.TemporaryDirectory(prefix="ecdlp_selected13_worklist_", dir=str(temp_root)) as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "selected13_execution_worklist_preflight.c"
        exe_path = temp_path / "selected13_execution_worklist_preflight"
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


def compare_native_preflight(
    work_items: list[dict[str, Any]],
    dependencies: list[int],
    native: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if not native.get("compiled"):
        failures.append(
            {
                "code": "native_preflight_compile_failed",
                "compile_returncode": native.get("compile_returncode"),
                "compile_stderr": native.get("compile_stderr"),
            }
        )
        return failures
    if as_int(native.get("run_returncode"), -1) != 0:
        failures.append(
            {
                "code": "native_preflight_run_failed",
                "run_returncode": native.get("run_returncode"),
                "run_stdout": native.get("run_stdout"),
                "run_stderr": native.get("run_stderr"),
            }
        )
    summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    if not summary:
        failures.append({"code": "native_preflight_stdout_not_json", "run_stdout": native.get("run_stdout")})
        return failures
    expected_phase_counts = Counter(as_int(item.get("phase_code")) for item in work_items)
    expected_class_counts = Counter(as_int(item.get("class_code")) for item in work_items)
    if as_int(summary.get("work_item_count"), -1) != len(work_items):
        failures.append({"code": "native_work_item_count_mismatch"})
    if as_int(summary.get("dependency_edge_count"), -1) != len(dependencies):
        failures.append({"code": "native_dependency_count_mismatch"})
    if as_int(summary.get("failure_count"), -1) != 0:
        failures.append({"code": "native_preflight_reported_failures", "native_summary": summary})
    if (summary.get("phase_counts") or {}) != {
        str(code): int(expected_phase_counts.get(code, 0)) for code in range(1, 5)
    }:
        failures.append({"code": "native_phase_counts_mismatch", "native": summary.get("phase_counts")})
    if (summary.get("class_counts") or {}) != {
        str(code): int(expected_class_counts.get(code, 0)) for code in range(1, 5)
    }:
        failures.append({"code": "native_class_counts_mismatch", "native": summary.get("class_counts")})
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-manifest", type=Path, default=DEFAULT_PACKET_MANIFEST)
    parser.add_argument("--native-abi", type=Path, default=DEFAULT_NATIVE_ABI)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--no-c-header", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet_manifest = load_json(args.packet_manifest)
    native_abi = load_json(args.native_abi) if args.native_abi.exists() else None
    work_items, dependencies = build_work_items(packet_manifest)
    execution_order = [as_int(item.get("work_item_index")) for item in sorted(work_items, key=execution_order_key)]
    failures = validate_sources(packet_manifest, native_abi)
    failures.extend(validate_worklist(work_items, dependencies))

    native_preflight: dict[str, Any] = {"compiled": False, "executed": False, "native_summary": {}}
    if not args.no_c_header:
        args.c_header_out.parent.mkdir(parents=True, exist_ok=True)
        args.c_header_out.write_text(render_c_header(work_items, dependencies))
        native_preflight = run_native_preflight(args.c_header_out, args.cc)
        failures.extend(compare_native_preflight(work_items, dependencies, native_preflight))

    native_summary = (
        native_preflight.get("native_summary")
        if isinstance(native_preflight.get("native_summary"), dict)
        else {}
    )
    verified = not failures
    payload = {
        "artifacts": {
            "c_header": None if args.no_c_header else str(args.c_header_out),
            "native_abi": str(args.native_abi),
            "packet_manifest": str(args.packet_manifest),
        },
        "claim_status": (
            "FFE_SHARP_LANE_EXECUTION_WORKLIST_READY"
            if verified
            else "FFE_SHARP_LANE_EXECUTION_WORKLIST_FAILED_CHECK"
        ),
        "created_at": now_iso(),
        "dependencies": dependencies,
        "execution_order": execution_order,
        "failures": failures,
        "honesty_boundary": [
            "This is a row-level worklist and native preflight for the selected13 sharp-lane FFE worker.",
            "It does not evaluate summation polynomials or finite-field equations.",
            "It does not export direct/rank rows, solve an ECDLP instance, or claim a Pollard-rho speedup.",
        ],
        "native_preflight": native_preflight,
        "parameters": {
            "full_family_masks": FULL_FAMILY_MASKS,
            "primary_backfill_transfers": PRIMARY_BACKFILL_TRANSFERS,
            "selected13_mask": SELECTED13_MASK,
        },
        "schema": SCHEMA,
        "summary": summarize(work_items, dependencies, native_summary),
        "work_items": work_items,
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
