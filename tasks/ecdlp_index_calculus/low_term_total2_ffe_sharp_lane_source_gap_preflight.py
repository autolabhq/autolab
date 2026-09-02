#!/usr/bin/env python3
"""Classify selected13 backfill masks by exact-source proximity.

The linear-rank preflight shows that the full-family backfill masks already sit
inside the exact replay support span.  This script adds the next worker-facing
distinction: which masks have an exact support witness on the same row-key pair,
which only have a one-salt-neighbor exact witness, and which are support-span
only.  That separates replay/carryover candidates from masks that still need a
fresh FFE/summation-polynomial direct/rank export.

This is a source-gap preflight only.  It does not regenerate missing sidecar
coefficients, evaluate summation polynomials, export direct/rank rows, solve
ECDLP, or claim a Pollard-rho speedup.
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


SCHEMA = "ecdlp.low_term_total2_ffe_sharp_lane_source_gap_preflight.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_EXACT_REPLAY = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_exact_certificate_replay_selected13_9696_9999_probe.json"
)
DEFAULT_WORKLIST = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_execution_worklist_selected13_9696_9999_probe.json"
)
DEFAULT_LINEAR_RANK = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_linear_rank_preflight_selected13_9696_9999_probe.json"
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_source_gap_preflight_selected13_9696_9999_probe.json"
)
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_source_gap_preflight_selected13_9696_9999_probe.h"
)

FAMILY_MASK_COUNT = 3
PRIMARY_BACKFILL_TRANSFERS = [9981, 9943]
TIER_CODES = {
    "same_row_key_exact_support": 1,
    "one_salt_neighbor_exact_support": 2,
    "support_span_only": 3,
}
EXPECTED_BACKFILL_ITEM_COUNT = 13
EXPECTED_TOTAL_MASK_COUNT = 39
EXPECTED_SAME_ROW_KEY_MASK_COUNT = 2
EXPECTED_ONE_SALT_NEIGHBOR_MASK_COUNT = 6
EXPECTED_SUPPORT_SPAN_ONLY_MASK_COUNT = 31
EXPECTED_PRIMARY_COUNTS = {
    9981: {
        "same_row_key_exact_support": 2,
        "one_salt_neighbor_exact_support": 0,
        "support_span_only": 1,
    },
    9943: {
        "same_row_key_exact_support": 0,
        "one_salt_neighbor_exact_support": 2,
        "support_span_only": 1,
    },
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


def digest_u64(raw: Any) -> int:
    text = str(raw or "")
    suffix = text.rsplit("_", 1)[-1]
    if suffix and all(ch in "0123456789abcdefABCDEF" for ch in suffix):
        return int(suffix[-16:], 16)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def exact_records(exact_replay: dict[str, Any]) -> list[dict[str, Any]]:
    return [record for record in exact_replay.get("records") or [] if isinstance(record, dict)]


def nonzero_form_masks(record: dict[str, Any]) -> list[int]:
    form_count = as_int(record.get("form_count"), len(record.get("form_masks") or []))
    return [as_int(mask) for mask in (record.get("form_masks") or [])[:form_count] if as_int(mask) != 0]


def work_items_by_index(worklist: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        as_int(item.get("work_item_index"), -1): item
        for item in worklist.get("work_items") or []
        if isinstance(item, dict)
    }


def materialized_row_keys(item: dict[str, Any]) -> list[str]:
    target = str(item.get("target") or "")
    return [f"{target}:uniform:256:salt{as_int(salt)}" for salt in item.get("salts") or []]


def exact_match_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("target"),
        record.get("selector"),
        as_int(record.get("top_k")),
        tuple(record.get("row_keys") or []),
        as_int(record.get("selected_support_mask")),
    )


def work_item_match_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item.get("target"),
        item.get("selector"),
        as_int(item.get("top_k")),
        tuple(materialized_row_keys(item)),
        as_int(item.get("selected_support_mask")),
    )


def one_salt_neighbor(record: dict[str, Any], item: dict[str, Any]) -> bool:
    record_salts = [as_int(salt) for salt in record.get("salts") or []]
    item_salts = [as_int(salt) for salt in item.get("salts") or []]
    common = sorted(set(record_salts) & set(item_salts))
    if len(common) != 1:
        return False
    shared = common[0]
    record_other = [salt for salt in record_salts if salt != shared]
    item_other = [salt for salt in item_salts if salt != shared]
    return len(record_other) == 1 and len(item_other) == 1 and abs(record_other[0] - item_other[0]) == 1


def salt_delta(record: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    record_salts = [as_int(salt) for salt in record.get("salts") or []]
    item_salts = [as_int(salt) for salt in item.get("salts") or []]
    common = sorted(set(record_salts) & set(item_salts))
    if len(common) != 1:
        return {
            "shared_salt": None,
            "record_other_salt": None,
            "backfill_other_salt": None,
            "other_salt_delta": None,
        }
    shared = common[0]
    record_other = [salt for salt in record_salts if salt != shared]
    item_other = [salt for salt in item_salts if salt != shared]
    if len(record_other) != 1 or len(item_other) != 1:
        return {
            "shared_salt": shared,
            "record_other_salt": None,
            "backfill_other_salt": None,
            "other_salt_delta": None,
        }
    return {
        "shared_salt": shared,
        "record_other_salt": record_other[0],
        "backfill_other_salt": item_other[0],
        "other_salt_delta": item_other[0] - record_other[0],
    }


def exact_witness_summary(record: dict[str, Any], item: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = {
        "certificate_hash": record.get("certificate_hash"),
        "certificate_hash_u64": as_int(record.get("certificate_hash_u64")) or digest_u64(record.get("certificate_hash")),
        "form_masks": nonzero_form_masks(record),
        "row_id": record.get("row_id"),
        "row_keys": record.get("row_keys") or [],
        "salts": record.get("salts") or [],
        "transfer_index": as_int(record.get("transfer_index"), -1),
    }
    if item is not None:
        summary["salt_delta"] = salt_delta(record, item)
    return summary


def witness_for_mask(
    mask: int,
    item: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    same = [
        record
        for record in records
        if exact_match_key(record) == work_item_match_key(item) and mask in nonzero_form_masks(record)
    ]
    if same:
        return {
            "family_mask": mask,
            "source_tier": "same_row_key_exact_support",
            "source_tier_code": TIER_CODES["same_row_key_exact_support"],
            "witnesses": [exact_witness_summary(record) for record in same],
        }

    neighbors = [
        record
        for record in records
        if record.get("target") == item.get("target")
        and record.get("selector") == item.get("selector")
        and as_int(record.get("top_k")) == as_int(item.get("top_k"))
        and as_int(record.get("selected_support_mask")) == as_int(item.get("selected_support_mask"))
        and mask in nonzero_form_masks(record)
        and one_salt_neighbor(record, item)
    ]
    if neighbors:
        return {
            "family_mask": mask,
            "source_tier": "one_salt_neighbor_exact_support",
            "source_tier_code": TIER_CODES["one_salt_neighbor_exact_support"],
            "witnesses": [exact_witness_summary(record, item) for record in neighbors],
        }

    span_witnesses = [
        record
        for record in records
        if record.get("target") == item.get("target")
        and record.get("selector") == item.get("selector")
        and as_int(record.get("top_k")) == as_int(item.get("top_k"))
        and mask in nonzero_form_masks(record)
    ]
    return {
        "family_mask": mask,
        "source_tier": "support_span_only",
        "source_tier_code": TIER_CODES["support_span_only"],
        "witnesses": [exact_witness_summary(record, item) for record in span_witnesses],
    }


def build_source_gap_items(
    linear_rank: dict[str, Any],
    worklist: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_index = work_items_by_index(worklist)
    out = []
    for backfill in linear_rank.get("backfill_mask_items") or []:
        item = by_index.get(as_int(backfill.get("work_item_index"), -1)) or {}
        family_masks = [as_int(mask) for mask in (backfill.get("family_masks") or [])[:FAMILY_MASK_COUNT]]
        while len(family_masks) < FAMILY_MASK_COUNT:
            family_masks.append(0)
        witnesses = [witness_for_mask(mask, item, records) for mask in family_masks]
        tier_counts = Counter(witness.get("source_tier") for witness in witnesses)
        source_gap_class = "support_span_only"
        if tier_counts.get("same_row_key_exact_support"):
            source_gap_class = "same_row_key_partial_carryover"
        elif tier_counts.get("one_salt_neighbor_exact_support"):
            source_gap_class = "one_salt_neighbor_partial_carryover"
        out.append(
            {
                "direct_status": item.get("direct_status"),
                "family_masks": family_masks,
                "is_primary": bool(backfill.get("is_primary")),
                "mask_witnesses": witnesses,
                "materialized_row_keys": materialized_row_keys(item),
                "row_check_hash": item.get("row_check_hash") or backfill.get("row_check_hash"),
                "row_check_hash_u64": as_int(item.get("row_check_hash_u64")) or digest_u64(item.get("row_check_hash")),
                "row_id": item.get("row_id") or backfill.get("row_id"),
                "salts": item.get("salts") or [],
                "source_gap_class": source_gap_class,
                "source_tier_counts": {tier: int(tier_counts.get(tier, 0)) for tier in TIER_CODES},
                "target": item.get("target"),
                "transfer_index": as_int(backfill.get("transfer_index"), -1),
                "work_item_index": as_int(backfill.get("work_item_index"), -1),
            }
        )
    return out


def validate_sources(
    exact_replay: dict[str, Any],
    worklist: dict[str, Any],
    linear_rank: dict[str, Any],
) -> list[dict[str, Any]]:
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
    if linear_rank.get("claim_status") != "FFE_SHARP_LANE_LINEAR_RANK_PREFLIGHT_READY":
        failures.append({"code": "linear_rank_not_ready", "claim_status": linear_rank.get("claim_status")})
    if linear_rank.get("failures"):
        failures.append({"code": "linear_rank_has_failures", "failures": linear_rank.get("failures")})
    if (linear_rank.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "linear_rank_summary_not_verified", "summary": linear_rank.get("summary")})
    return failures


def tier_count(items: list[dict[str, Any]], tier: str) -> int:
    return sum(as_int((item.get("source_tier_counts") or {}).get(tier)) for item in items)


def validate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(items) != EXPECTED_BACKFILL_ITEM_COUNT:
        failures.append({"code": "backfill_item_count_mismatch", "observed": len(items)})
    if sum(len(item.get("mask_witnesses") or []) for item in items) != EXPECTED_TOTAL_MASK_COUNT:
        failures.append({"code": "total_mask_count_mismatch"})
    if tier_count(items, "same_row_key_exact_support") != EXPECTED_SAME_ROW_KEY_MASK_COUNT:
        failures.append({"code": "same_row_key_mask_count_mismatch"})
    if tier_count(items, "one_salt_neighbor_exact_support") != EXPECTED_ONE_SALT_NEIGHBOR_MASK_COUNT:
        failures.append({"code": "one_salt_neighbor_mask_count_mismatch"})
    if tier_count(items, "support_span_only") != EXPECTED_SUPPORT_SPAN_ONLY_MASK_COUNT:
        failures.append({"code": "support_span_only_mask_count_mismatch"})

    primary_order = [as_int(item.get("transfer_index"), -1) for item in items if bool(item.get("is_primary"))]
    if primary_order != PRIMARY_BACKFILL_TRANSFERS:
        failures.append({"code": "primary_order_mismatch", "observed": primary_order})
    for item in items:
        transfer = as_int(item.get("transfer_index"), -1)
        mask_witnesses = item.get("mask_witnesses") or []
        if len(mask_witnesses) != FAMILY_MASK_COUNT:
            failures.append({"code": "family_mask_width_mismatch", "transfer_index": transfer})
        if item.get("direct_status") != "direct_certificate_missing":
            failures.append({"code": "backfill_not_marked_direct_missing", "transfer_index": transfer})
        tier_sum = sum(as_int((item.get("source_tier_counts") or {}).get(tier)) for tier in TIER_CODES)
        if tier_sum != FAMILY_MASK_COUNT:
            failures.append({"code": "tier_count_sum_mismatch", "transfer_index": transfer})
        if transfer in EXPECTED_PRIMARY_COUNTS:
            expected = EXPECTED_PRIMARY_COUNTS[transfer]
            observed = item.get("source_tier_counts") or {}
            for tier, count in expected.items():
                if as_int(observed.get(tier), -1) != count:
                    failures.append(
                        {
                            "code": "primary_tier_count_mismatch",
                            "expected": expected,
                            "observed": observed,
                            "transfer_index": transfer,
                        }
                    )
        for witness in mask_witnesses:
            tier = str(witness.get("source_tier") or "")
            if tier not in TIER_CODES:
                failures.append({"code": "unknown_source_tier", "transfer_index": transfer, "tier": tier})
            if as_int(witness.get("source_tier_code")) != TIER_CODES.get(tier, 0):
                failures.append({"code": "source_tier_code_mismatch", "transfer_index": transfer, "tier": tier})
            if not witness.get("witnesses"):
                failures.append({"code": "mask_has_no_exact_support_witness", "transfer_index": transfer})
    return failures


def c_u64_array(values: list[int]) -> str:
    if not values:
        return "{0ULL}"
    return "{" + ", ".join(f"{as_int(value)}ULL" for value in values) + "}"


def render_c_header(items: list[dict[str, Any]]) -> str:
    item_lines = []
    for item in items:
        tier_codes = [as_int(witness.get("source_tier_code")) for witness in item.get("mask_witnesses") or []]
        while len(tier_codes) < FAMILY_MASK_COUNT:
            tier_codes.append(0)
        family_masks = [as_int(mask) for mask in item.get("family_masks") or []]
        while len(family_masks) < FAMILY_MASK_COUNT:
            family_masks.append(0)
        counts = item.get("source_tier_counts") or {}
        item_lines.append(
            "  {"
            f"{as_int(item.get('work_item_index'))}ULL, "
            f"{as_int(item.get('transfer_index'))}ULL, "
            f"{as_int(item.get('row_check_hash_u64'))}ULL, "
            f"{c_u64_array(family_masks[:FAMILY_MASK_COUNT])}, "
            f"{c_u64_array(tier_codes[:FAMILY_MASK_COUNT])}, "
            f"{as_int(counts.get('same_row_key_exact_support'))}ULL, "
            f"{as_int(counts.get('one_salt_neighbor_exact_support'))}ULL, "
            f"{as_int(counts.get('support_span_only'))}ULL, "
            f"{1 if bool(item.get('is_primary')) else 0}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_FFE_SHARP_LANE_SOURCE_GAP_PREFLIGHT_H
#define LOW_TERM_TOTAL2_FFE_SHARP_LANE_SOURCE_GAP_PREFLIGHT_H

#include <stdint.h>

#define SELECTED13_SOURCE_GAP_BACKFILL_COUNT {len(items)}
#define SELECTED13_SOURCE_GAP_FAMILY_MASK_COUNT {FAMILY_MASK_COUNT}
#define SELECTED13_SOURCE_GAP_TOTAL_MASK_COUNT {sum(len(item.get("mask_witnesses") or []) for item in items)}
#define SELECTED13_SOURCE_GAP_SAME_ROW_KEY_MASK_COUNT {tier_count(items, "same_row_key_exact_support")}
#define SELECTED13_SOURCE_GAP_ONE_SALT_NEIGHBOR_MASK_COUNT {tier_count(items, "one_salt_neighbor_exact_support")}
#define SELECTED13_SOURCE_GAP_SUPPORT_SPAN_ONLY_MASK_COUNT {tier_count(items, "support_span_only")}

#define SELECTED13_SOURCE_TIER_SAME_ROW_KEY 1ULL
#define SELECTED13_SOURCE_TIER_ONE_SALT_NEIGHBOR 2ULL
#define SELECTED13_SOURCE_TIER_SUPPORT_SPAN_ONLY 3ULL

typedef struct {{
  uint64_t work_item_index;
  uint64_t transfer_index;
  uint64_t row_check_hash_u64;
  uint64_t family_masks[SELECTED13_SOURCE_GAP_FAMILY_MASK_COUNT];
  uint64_t source_tiers[SELECTED13_SOURCE_GAP_FAMILY_MASK_COUNT];
  uint64_t same_row_key_mask_count;
  uint64_t one_salt_neighbor_mask_count;
  uint64_t support_span_only_mask_count;
  uint64_t is_primary;
}} selected13_source_gap_backfill_item_t;

static const selected13_source_gap_backfill_item_t SELECTED13_SOURCE_GAP_BACKFILL_ITEMS[] = {{
{chr(10).join(item_lines)}
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
  uint64_t same_row_key_mask_count = 0;
  uint64_t one_salt_neighbor_mask_count = 0;
  uint64_t support_span_only_mask_count = 0;
  uint64_t total_mask_count = 0;
  uint64_t primary_count = 0;
  uint64_t primary_transfer0 = 0;
  uint64_t primary_transfer1 = 0;
  uint64_t primary0_same = 0;
  uint64_t primary0_neighbor = 0;
  uint64_t primary0_span_only = 0;
  uint64_t primary1_same = 0;
  uint64_t primary1_neighbor = 0;
  uint64_t primary1_span_only = 0;

  const size_t item_count =
      sizeof(SELECTED13_SOURCE_GAP_BACKFILL_ITEMS) / sizeof(SELECTED13_SOURCE_GAP_BACKFILL_ITEMS[0]);
  if (item_count != SELECTED13_SOURCE_GAP_BACKFILL_COUNT) failure_count++;

  for (size_t i = 0; i < item_count; i++) {{
    const selected13_source_gap_backfill_item_t *item = &SELECTED13_SOURCE_GAP_BACKFILL_ITEMS[i];
    uint64_t row_same = 0;
    uint64_t row_neighbor = 0;
    uint64_t row_span_only = 0;
    if (item->row_check_hash_u64 == 0) failure_count++;
    for (size_t j = 0; j < SELECTED13_SOURCE_GAP_FAMILY_MASK_COUNT; j++) {{
      if (item->family_masks[j] == 0) failure_count++;
      if (item->source_tiers[j] == SELECTED13_SOURCE_TIER_SAME_ROW_KEY) {{
        row_same++;
      }} else if (item->source_tiers[j] == SELECTED13_SOURCE_TIER_ONE_SALT_NEIGHBOR) {{
        row_neighbor++;
      }} else if (item->source_tiers[j] == SELECTED13_SOURCE_TIER_SUPPORT_SPAN_ONLY) {{
        row_span_only++;
      }} else {{
        failure_count++;
      }}
      total_mask_count++;
    }}
    if (row_same != item->same_row_key_mask_count) failure_count++;
    if (row_neighbor != item->one_salt_neighbor_mask_count) failure_count++;
    if (row_span_only != item->support_span_only_mask_count) failure_count++;
    if (row_same + row_neighbor + row_span_only != SELECTED13_SOURCE_GAP_FAMILY_MASK_COUNT) failure_count++;
    same_row_key_mask_count += row_same;
    one_salt_neighbor_mask_count += row_neighbor;
    support_span_only_mask_count += row_span_only;
    if (item->is_primary != 0) {{
      if (primary_count == 0) {{
        primary_transfer0 = item->transfer_index;
        primary0_same = row_same;
        primary0_neighbor = row_neighbor;
        primary0_span_only = row_span_only;
        if (item->transfer_index != {PRIMARY_BACKFILL_TRANSFERS[0]}ULL) failure_count++;
      }} else if (primary_count == 1) {{
        primary_transfer1 = item->transfer_index;
        primary1_same = row_same;
        primary1_neighbor = row_neighbor;
        primary1_span_only = row_span_only;
        if (item->transfer_index != {PRIMARY_BACKFILL_TRANSFERS[1]}ULL) failure_count++;
      }} else {{
        failure_count++;
      }}
      primary_count++;
    }}
  }}

  if (total_mask_count != SELECTED13_SOURCE_GAP_TOTAL_MASK_COUNT) failure_count++;
  if (same_row_key_mask_count != SELECTED13_SOURCE_GAP_SAME_ROW_KEY_MASK_COUNT) failure_count++;
  if (one_salt_neighbor_mask_count != SELECTED13_SOURCE_GAP_ONE_SALT_NEIGHBOR_MASK_COUNT) failure_count++;
  if (support_span_only_mask_count != SELECTED13_SOURCE_GAP_SUPPORT_SPAN_ONLY_MASK_COUNT) failure_count++;
  if (same_row_key_mask_count != {EXPECTED_SAME_ROW_KEY_MASK_COUNT}ULL) failure_count++;
  if (one_salt_neighbor_mask_count != {EXPECTED_ONE_SALT_NEIGHBOR_MASK_COUNT}ULL) failure_count++;
  if (support_span_only_mask_count != {EXPECTED_SUPPORT_SPAN_ONLY_MASK_COUNT}ULL) failure_count++;
  if (primary_count != 2) failure_count++;
  if (primary0_same != 2 || primary0_neighbor != 0 || primary0_span_only != 1) failure_count++;
  if (primary1_same != 0 || primary1_neighbor != 2 || primary1_span_only != 1) failure_count++;

  printf("{{");
  printf("\\\"item_count\\\":%llu,", (unsigned long long)item_count);
  printf("\\\"total_mask_count\\\":%llu,", (unsigned long long)total_mask_count);
  printf("\\\"same_row_key_mask_count\\\":%llu,", (unsigned long long)same_row_key_mask_count);
  printf("\\\"one_salt_neighbor_mask_count\\\":%llu,", (unsigned long long)one_salt_neighbor_mask_count);
  printf("\\\"support_span_only_mask_count\\\":%llu,", (unsigned long long)support_span_only_mask_count);
  printf("\\\"primary_count\\\":%llu,", (unsigned long long)primary_count);
  printf("\\\"primary_transfer0\\\":%llu,", (unsigned long long)primary_transfer0);
  printf("\\\"primary_transfer1\\\":%llu,", (unsigned long long)primary_transfer1);
  printf("\\\"primary0_same\\\":%llu,", (unsigned long long)primary0_same);
  printf("\\\"primary0_neighbor\\\":%llu,", (unsigned long long)primary0_neighbor);
  printf("\\\"primary0_span_only\\\":%llu,", (unsigned long long)primary0_span_only);
  printf("\\\"primary1_same\\\":%llu,", (unsigned long long)primary1_same);
  printf("\\\"primary1_neighbor\\\":%llu,", (unsigned long long)primary1_neighbor);
  printf("\\\"primary1_span_only\\\":%llu,", (unsigned long long)primary1_span_only);
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
    with tempfile.TemporaryDirectory(prefix="ecdlp_selected13_source_gap_", dir=str(temp_root)) as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "selected13_source_gap_preflight.c"
        exe_path = temp_path / "selected13_source_gap_preflight"
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


def compare_native(items: list[dict[str, Any]], native: dict[str, Any]) -> list[dict[str, Any]]:
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
    if as_int(summary.get("item_count"), -1) != len(items):
        failures.append({"code": "native_item_count_mismatch"})
    if as_int(summary.get("same_row_key_mask_count"), -1) != tier_count(items, "same_row_key_exact_support"):
        failures.append({"code": "native_same_row_key_count_mismatch"})
    if as_int(summary.get("one_salt_neighbor_mask_count"), -1) != tier_count(items, "one_salt_neighbor_exact_support"):
        failures.append({"code": "native_one_salt_neighbor_count_mismatch"})
    if as_int(summary.get("support_span_only_mask_count"), -1) != tier_count(items, "support_span_only"):
        failures.append({"code": "native_support_span_only_count_mismatch"})
    if as_int(summary.get("failure_count"), -1) != 0:
        failures.append({"code": "native_preflight_reported_failures", "native_summary": summary})
    return failures


def summarize(items: list[dict[str, Any]], native_summary: dict[str, Any]) -> dict[str, Any]:
    class_counts = Counter(str(item.get("source_gap_class")) for item in items)
    primary = [item for item in items if bool(item.get("is_primary"))]
    return {
        "backfill_item_count": len(items),
        "item_class_counts": dict(sorted(class_counts.items())),
        "native_preflight_failure_count": as_int(native_summary.get("failure_count"), -1),
        "native_preflight_verified": as_int(native_summary.get("failure_count"), -1) == 0,
        "one_salt_neighbor_mask_count": tier_count(items, "one_salt_neighbor_exact_support"),
        "primary_backfill_source_gaps": [
            {
                "source_gap_class": item.get("source_gap_class"),
                "source_tier_counts": item.get("source_tier_counts"),
                "transfer_index": item.get("transfer_index"),
            }
            for item in primary
        ],
        "primary_backfill_transfers": [as_int(item.get("transfer_index"), -1) for item in primary],
        "same_row_key_mask_count": tier_count(items, "same_row_key_exact_support"),
        "source_gap_interpretation": (
            "9981 has two same-row-key exact support witnesses but no exported direct certificate; "
            "9943 has two one-salt-neighbor exact support witnesses. Both still need actual "
            "FFE/summation-polynomial direct/rank export before any success claim."
        ),
        "support_span_only_mask_count": tier_count(items, "support_span_only"),
        "total_mask_count": sum(len(item.get("mask_witnesses") or []) for item in items),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-replay", type=Path, default=DEFAULT_EXACT_REPLAY)
    parser.add_argument("--worklist", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--linear-rank", type=Path, default=DEFAULT_LINEAR_RANK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--no-c-header", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exact_replay = load_json(args.exact_replay)
    worklist = load_json(args.worklist)
    linear_rank = load_json(args.linear_rank)
    records = exact_records(exact_replay)
    items = build_source_gap_items(linear_rank, worklist, records)
    failures = validate_sources(exact_replay, worklist, linear_rank)
    failures.extend(validate_items(items))

    native: dict[str, Any] = {"compiled": False, "executed": False, "native_summary": {}}
    if not args.no_c_header:
        args.c_header_out.parent.mkdir(parents=True, exist_ok=True)
        args.c_header_out.write_text(render_c_header(items))
        native = run_native_preflight(args.c_header_out, args.cc)
        failures.extend(compare_native(items, native))

    native_summary = native.get("native_summary") if isinstance(native.get("native_summary"), dict) else {}
    verified = not failures
    payload = {
        "artifacts": {
            "c_header": None if args.no_c_header else str(args.c_header_out),
            "exact_replay": str(args.exact_replay),
            "linear_rank": str(args.linear_rank),
            "worklist": str(args.worklist),
        },
        "claim_status": (
            "FFE_SHARP_LANE_SOURCE_GAP_PREFLIGHT_READY"
            if verified
            else "FFE_SHARP_LANE_SOURCE_GAP_PREFLIGHT_FAILED_CHECK"
        ),
        "created_at": now_iso(),
        "failures": failures,
        "honesty_boundary": [
            "This classifies source proximity for support masks only.",
            "Same-row-key and one-salt-neighbor witnesses are replay targets, not exported direct/rank certificates.",
            "It does not regenerate sidecar coefficients, evaluate summation polynomials, solve finite-field equations, solve ECDLP, or claim a Pollard-rho speedup.",
        ],
        "native_preflight": native,
        "parameters": {
            "expected_one_salt_neighbor_mask_count": EXPECTED_ONE_SALT_NEIGHBOR_MASK_COUNT,
            "expected_same_row_key_mask_count": EXPECTED_SAME_ROW_KEY_MASK_COUNT,
            "expected_support_span_only_mask_count": EXPECTED_SUPPORT_SPAN_ONLY_MASK_COUNT,
            "primary_backfill_transfers": PRIMARY_BACKFILL_TRANSFERS,
            "source_tier_codes": TIER_CODES,
        },
        "schema": SCHEMA,
        "source_gap_items": items,
        "summary": summarize(items, native_summary),
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
