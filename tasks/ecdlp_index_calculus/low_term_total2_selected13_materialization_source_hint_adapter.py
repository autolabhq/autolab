#!/usr/bin/env python3
"""Join selected13 materialization bridge packets to host support-scout rows.

The bridge packet binds the 15 selected13 materialization misses into direct-rank
worker rows, but the host scout artifacts still contain the row-level public
case reports.  This adapter checks exact packet-row materialization against
those scouts and extracts same-transfer direct-verified source hints for the
fresh FFE/summation-polynomial worker.

It does not export direct/rank rows, derive an ECDLP scalar, or claim a
Pollard-rho speedup.  Exact scout direct verification is kept as a promotion
candidate until a relation-derived export is produced by a verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_selected13_materialization_source_hint_adapter.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_PACKET = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_bridge_packet_111_full147_probe.json"
DEFAULT_WORKORDER = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_miss_workorder_111_full147_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_source_hint_adapter_111_full147_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_source_hint_adapter_111_full147_probe.h"

DIRECT_STATUS_CODES = {
    "support_report_missing": 1,
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


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def scaled_ops(value: Any) -> int:
    number = as_float(value)
    if number is None:
        return 0
    return round(number * 100_000_000)


def canonical_json(raw: Any) -> str:
    return json.dumps(raw, sort_keys=True, separators=(",", ":"))


def digest_u64(raw: Any) -> int:
    if raw is None:
        return 0
    return int(hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:16], 16)


def stable_hash_u64(raw: Any) -> int:
    return int(hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()[:16], 16)


def support_mask(terms: Any) -> int:
    mask = 0
    for term in terms or []:
        value = as_int(term, -1)
        if value >= 0:
            mask |= 1 << value
    return mask


def support_tuple(raw: Any) -> tuple[int, ...]:
    return tuple(sorted(as_int(item) for item in raw or []))


def row_key_tuple(raw: Any) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in raw or []))


def row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        as_int(row.get("transfer_index"), -1),
        row_key_tuple(row.get("row_keys")),
        str(row.get("selector") or ""),
        as_int(row.get("top_k")),
        support_tuple(row.get("selected_term_support")),
    )


def salts_from_row_keys(row_keys: Any) -> list[int]:
    salts = []
    for row_key in row_keys or []:
        text = str(row_key)
        if "salt" not in text:
            continue
        try:
            salts.append(int(text.rsplit("salt", 1)[1]))
        except ValueError:
            continue
    return sorted(salts)


def workorder_rows_by_request(workorder: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for item in workorder.get("work_items") or []:
        for row in item.get("candidate_rows") or []:
            if isinstance(row, dict) and row.get("row_request_id"):
                rows[str(row.get("row_request_id"))] = row
    return rows


def scout_paths_from_workorder(workorder: dict[str, Any]) -> list[Path]:
    paths = []
    for item in workorder.get("work_items") or []:
        for row in item.get("candidate_rows") or []:
            if isinstance(row, dict) and row.get("artifact"):
                paths.append(Path(str(row.get("artifact"))))
    return sorted(set(paths))


def scout_rows_by_identity(paths: list[Path]) -> tuple[dict[tuple[Any, ...], list[dict[str, Any]]], dict[int, list[dict[str, Any]]], list[str], dict[str, Any]]:
    exact: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    by_transfer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    missing = []
    summaries = {}
    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        payload = load_json(path)
        summaries[str(path)] = {
            "schema": payload.get("schema"),
            "summary": payload.get("summary"),
        }
        for row in payload.get("case_reports") or []:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item["artifact"] = str(path)
            exact[row_identity(item)].append(item)
            by_transfer[as_int(item.get("transfer_index"), -1)].append(item)
    return exact, by_transfer, missing, summaries


def source_hint_sort_key(packet_row: dict[str, Any], source_row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    packet_salts = set(salts_from_row_keys(packet_row.get("row_keys")))
    source_salts = set(salts_from_row_keys(source_row.get("row_keys")))
    packet_support = set(support_tuple(packet_row.get("selected_term_support")))
    source_support = set(support_tuple(source_row.get("selected_term_support")))
    packet_ops = as_float(packet_row.get("direct_ops_over_rho"))
    source_ops = as_float(source_row.get("direct_ops_over_rho"))
    ops_delta_scaled = scaled_ops(abs((packet_ops if packet_ops is not None else 9.0) - (source_ops if source_ops is not None else 9.0)))
    return (
        -len(packet_salts & source_salts),
        -len(packet_support & source_support),
        ops_delta_scaled,
        scaled_ops(source_ops),
        canonical_json(row_identity(source_row)),
    )


def build_source_hints(
    packet_row: dict[str, Any],
    same_transfer_rows: list[dict[str, Any]],
    max_hints: int,
) -> list[dict[str, Any]]:
    hints = []
    packet_identity = row_identity(packet_row)
    packet_salts = set(salts_from_row_keys(packet_row.get("row_keys")))
    packet_support = set(support_tuple(packet_row.get("selected_term_support")))
    for source in sorted(same_transfer_rows, key=lambda row: source_hint_sort_key(packet_row, row)):
        if not bool(source.get("direct_public_key_verified")):
            continue
        if row_identity(source) == packet_identity:
            continue
        source_salts = set(salts_from_row_keys(source.get("row_keys")))
        source_support = set(support_tuple(source.get("selected_term_support")))
        support_union = packet_support | source_support
        support_overlap = len(packet_support & source_support)
        hint = {
            "direct_ops_over_rho": as_float(source.get("direct_ops_over_rho")),
            "direct_ops_over_rho_scaled": scaled_ops(source.get("direct_ops_over_rho")),
            "direct_public_key_verified": True,
            "hint_hash_u64": stable_hash_u64(
                {
                    "packet_row_request_id": packet_row.get("row_request_id"),
                    "source_identity": row_identity(source),
                }
            ),
            "public_product_gate_selected": bool(source.get("public_product_gate_selected")),
            "row_keys": list(row_key_tuple(source.get("row_keys"))),
            "salt_overlap_count": len(packet_salts & source_salts),
            "selected_support_mask": support_mask(source.get("selected_term_support")),
            "selected_term_support": list(support_tuple(source.get("selected_term_support"))),
            "selector": source.get("selector"),
            "shared_product_public_key_verified": bool(source.get("shared_product_public_key_verified")),
            "source_artifact": source.get("artifact"),
            "source_row_hash_u64": stable_hash_u64(row_identity(source)),
            "support_jaccard_scaled_1e6": round((support_overlap / len(support_union)) * 1_000_000)
            if support_union
            else 0,
            "support_overlap_count": support_overlap,
            "target": source.get("target"),
            "top_k": as_int(source.get("top_k")),
            "transfer_index": as_int(source.get("transfer_index"), -1),
        }
        hints.append(hint)
        if len(hints) >= max_hints:
            break
    return hints


def build_rows(
    packet: dict[str, Any],
    workorder: dict[str, Any],
    max_hints: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    workorder_index = workorder_rows_by_request(workorder)
    scout_paths = scout_paths_from_workorder(workorder)
    scout_exact, scout_by_transfer, missing_scouts, scout_summaries = scout_rows_by_identity(scout_paths)
    failures: list[dict[str, Any]] = []
    rows = []
    hints_flat = []
    if missing_scouts:
        failures.append({"code": "support_scout_sources_missing", "paths": missing_scouts})

    for packet_row in packet.get("rows") or []:
        if not isinstance(packet_row, dict):
            continue
        row_request_id = str(packet_row.get("row_request_id") or "")
        workorder_row = workorder_index.get(row_request_id)
        if workorder_row is None:
            failures.append({"code": "workorder_row_missing", "row_request_id": row_request_id})
            workorder_row = {}
        exact_matches = scout_exact.get(row_identity(workorder_row), [])
        direct_exact = [row for row in exact_matches if bool(row.get("direct_public_key_verified"))]
        shared_exact = [row for row in exact_matches if bool(row.get("shared_product_public_key_verified"))]
        if not exact_matches:
            failures.append({"code": "exact_support_scout_row_missing", "row_request_id": row_request_id})
        hints = build_source_hints(workorder_row, scout_by_transfer.get(as_int(packet_row.get("transfer_index"), -1), []), max_hints)
        hint_start = len(hints_flat)
        hints_flat.extend(
            {
                **hint,
                "global_row_index": as_int(packet_row.get("global_row_index"), -1),
                "packet_index": as_int(packet_row.get("packet_index"), -1),
                "row_request_id": row_request_id,
                "row_request_id_u64": as_int(packet_row.get("row_request_id_u64")),
            }
            for hint in hints
        )
        exact = direct_exact[0] if direct_exact else (exact_matches[0] if exact_matches else {})
        packet_ops = as_float(packet_row.get("direct_ops_over_rho") or workorder_row.get("direct_ops_over_rho"))
        exact_ops = as_float(exact.get("direct_ops_over_rho"))
        direct_verified = bool(direct_exact)
        rows.append(
            {
                "below_rho_label": bool((exact_ops if exact_ops is not None else packet_ops if packet_ops is not None else 9.0) < 1.0),
                "bridge_worker_required": not direct_verified,
                "candidate_row_index": as_int(workorder_row.get("candidate_row_index")),
                "direct_ops_over_rho": exact_ops if exact_ops is not None else packet_ops,
                "direct_ops_over_rho_scaled": scaled_ops(exact_ops if exact_ops is not None else packet_ops),
                "direct_status": workorder_row.get("direct_status"),
                "direct_status_code": DIRECT_STATUS_CODES.get(str(workorder_row.get("direct_status") or ""), 0),
                "exact_direct_public_key_verified": direct_verified,
                "exact_match_count": len(exact_matches),
                "exact_shared_product_public_key_verified": bool(shared_exact),
                "global_row_index": as_int(packet_row.get("global_row_index"), -1),
                "hint_count": len(hints),
                "hint_start": hint_start,
                "is_best_manifest_row": bool(packet_row.get("is_best_manifest_row")),
                "matched_family_count": as_int(packet_row.get("matched_family_count")),
                "packet_index": as_int(packet_row.get("packet_index"), -1),
                "priority_hit_count": len(exact.get("priority_hits") or []),
                "relation_derived_ecdlp": False,
                "row_keys": workorder_row.get("row_keys") or [],
                "row_material_hash_u64": as_int(packet_row.get("row_material_hash_u64")),
                "row_request_id": row_request_id,
                "row_request_id_u64": as_int(packet_row.get("row_request_id_u64")),
                "selected_support_mask": as_int(packet_row.get("selected_support_mask")),
                "selector": workorder_row.get("selector"),
                "source_artifact": exact.get("artifact"),
                "target": workorder_row.get("target"),
                "top_k": as_int(workorder_row.get("top_k")),
                "transfer_index": as_int(packet_row.get("transfer_index"), -1),
                "worker_action": "promote_exact_scout_direct_verification"
                if direct_verified
                else "fresh_ffe_summation_polynomial_direct_rank_bridge_extension",
            }
        )

    source_meta = {
        "scout_paths": [str(path) for path in scout_paths],
        "scout_summaries": scout_summaries,
    }
    return rows, hints_flat, failures, source_meta


def summarize(rows: list[dict[str, Any]], hints: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    direct_transfers = sorted(
        {
            as_int(row.get("transfer_index"), -1)
            for row in rows
            if row.get("exact_direct_public_key_verified")
        }
    )
    hinted_need_rows = [
        row
        for row in rows
        if row.get("bridge_worker_required") and as_int(row.get("hint_count")) > 0
    ]
    return {
        "accepted_relation_export_count": 0,
        "below_rho_exact_direct_row_count": sum(
            1 for row in rows if row.get("exact_direct_public_key_verified") and row.get("below_rho_label")
        ),
        "bridge_worker_required_row_count": sum(1 for row in rows if row.get("bridge_worker_required")),
        "exact_direct_best_row_count": sum(
            1 for row in rows if row.get("exact_direct_public_key_verified") and row.get("is_best_manifest_row")
        ),
        "exact_direct_public_key_verified_row_count": sum(1 for row in rows if row.get("exact_direct_public_key_verified")),
        "exact_direct_transfer_count": len(direct_transfers),
        "exact_direct_transfers": direct_transfers,
        "exact_match_row_count": sum(1 for row in rows if as_int(row.get("exact_match_count")) > 0),
        "failure_count": len(failures),
        "hinted_bridge_worker_row_count": len(hinted_need_rows),
        "hinted_bridge_worker_transfer_count": len({as_int(row.get("transfer_index"), -1) for row in hinted_need_rows}),
        "max_hints_per_row": max([as_int(row.get("hint_count")) for row in rows] or [0]),
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "row_count": len(rows),
        "source_hint_count": len(hints),
        "status_counts": dict(sorted(Counter(str(row.get("direct_status")) for row in rows).items())),
        "verified": not failures,
        "worker_interpretation": (
            "Exact scout rows and same-transfer direct-verified source rows are now bound to the bridge packet. "
            "Only a fresh direct/rank verifier can turn these rows into relation-derived ECDLP exports."
        ),
    }


def claim_status(summary: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    if failures:
        return "SELECTED13_MATERIALIZATION_SOURCE_HINT_ADAPTER_FAILED"
    if as_int(summary.get("exact_direct_public_key_verified_row_count")):
        return "SELECTED13_MATERIALIZATION_SOURCE_HINTS_WITH_EXACT_DIRECT_PROMOTION"
    if as_int(summary.get("source_hint_count")):
        return "SELECTED13_MATERIALIZATION_SOURCE_HINTS_READY"
    return "SELECTED13_MATERIALIZATION_SOURCE_HINTS_NO_DIRECT_SOURCE"


def render_c_header(rows: list[dict[str, Any]], hints: list[dict[str, Any]]) -> str:
    row_lines = []
    for row in rows:
        row_lines.append(
            "  {"
            f"{as_int(row.get('packet_index'))}ULL, "
            f"{as_int(row.get('global_row_index'))}ULL, "
            f"{as_int(row.get('transfer_index'))}ULL, "
            f"{as_int(row.get('row_request_id_u64'))}ULL, "
            f"{as_int(row.get('row_material_hash_u64'))}ULL, "
            f"{as_int(row.get('selected_support_mask'))}ULL, "
            f"{as_int(row.get('direct_status_code'))}ULL, "
            f"{1 if row.get('is_best_manifest_row') else 0}ULL, "
            f"{1 if row.get('exact_direct_public_key_verified') else 0}ULL, "
            f"{1 if row.get('bridge_worker_required') else 0}ULL, "
            f"{as_int(row.get('exact_match_count'))}ULL, "
            f"{as_int(row.get('hint_start'))}ULL, "
            f"{as_int(row.get('hint_count'))}ULL, "
            f"{as_int(row.get('direct_ops_over_rho_scaled'))}ULL"
            "},"
        )
    hint_lines = []
    for hint in hints:
        hint_lines.append(
            "  {"
            f"{as_int(hint.get('row_request_id_u64'))}ULL, "
            f"{as_int(hint.get('hint_hash_u64'))}ULL, "
            f"{as_int(hint.get('source_row_hash_u64'))}ULL, "
            f"{as_int(hint.get('transfer_index'))}ULL, "
            f"{as_int(hint.get('selected_support_mask'))}ULL, "
            f"{as_int(hint.get('salt_overlap_count'))}ULL, "
            f"{as_int(hint.get('support_overlap_count'))}ULL, "
            f"{as_int(hint.get('support_jaccard_scaled_1e6'))}ULL, "
            f"{as_int(hint.get('direct_ops_over_rho_scaled'))}ULL, "
            f"{1 if hint.get('direct_public_key_verified') else 0}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_MATERIALIZATION_SOURCE_HINT_ADAPTER_H
#define LOW_TERM_TOTAL2_SELECTED13_MATERIALIZATION_SOURCE_HINT_ADAPTER_H

#include <stdint.h>

#define SELECTED13_MATERIALIZATION_SOURCE_HINT_ROW_COUNT {len(rows)}
#define SELECTED13_MATERIALIZATION_SOURCE_HINT_COUNT {len(hints)}
#define SELECTED13_MATERIALIZATION_EXACT_DIRECT_ROW_COUNT {sum(1 for row in rows if row.get('exact_direct_public_key_verified'))}
#define SELECTED13_MATERIALIZATION_BRIDGE_WORKER_REQUIRED_ROW_COUNT {sum(1 for row in rows if row.get('bridge_worker_required'))}
#define SELECTED13_MATERIALIZATION_RELATION_DERIVED_ECDLP 0
#define SELECTED13_MATERIALIZATION_RELATION_EXPORT_COUNT 0

typedef struct {{
  uint64_t packet_index;
  uint64_t global_row_index;
  uint64_t transfer_index;
  uint64_t row_request_id_u64;
  uint64_t row_material_hash_u64;
  uint64_t selected_support_mask;
  uint64_t direct_status_code;
  uint64_t is_best_manifest_row;
  uint64_t exact_direct_public_key_verified;
  uint64_t bridge_worker_required;
  uint64_t exact_match_count;
  uint64_t hint_start;
  uint64_t hint_count;
  uint64_t direct_ops_over_rho_scaled;
}} selected13_materialization_source_hint_row_t;

typedef struct {{
  uint64_t row_request_id_u64;
  uint64_t hint_hash_u64;
  uint64_t source_row_hash_u64;
  uint64_t transfer_index;
  uint64_t selected_support_mask;
  uint64_t salt_overlap_count;
  uint64_t support_overlap_count;
  uint64_t support_jaccard_scaled_1e6;
  uint64_t direct_ops_over_rho_scaled;
  uint64_t direct_public_key_verified;
}} selected13_materialization_source_hint_t;

static const selected13_materialization_source_hint_row_t SELECTED13_MATERIALIZATION_SOURCE_HINT_ROWS[] = {{
{chr(10).join(row_lines)}
}};

static const selected13_materialization_source_hint_t SELECTED13_MATERIALIZATION_SOURCE_HINTS[] = {{
{chr(10).join(hint_lines)}
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
  uint64_t row_count = sizeof(SELECTED13_MATERIALIZATION_SOURCE_HINT_ROWS) / sizeof(SELECTED13_MATERIALIZATION_SOURCE_HINT_ROWS[0]);
  uint64_t hint_count = sizeof(SELECTED13_MATERIALIZATION_SOURCE_HINTS) / sizeof(SELECTED13_MATERIALIZATION_SOURCE_HINTS[0]);
  uint64_t exact_direct = 0;
  uint64_t needs_bridge = 0;
  uint64_t hinted_needs = 0;
  uint64_t best_exact_direct = 0;
  uint64_t last_transfer = 0;
  uint64_t immediate_transfers = 0;

  if (row_count != SELECTED13_MATERIALIZATION_SOURCE_HINT_ROW_COUNT) failure_count++;
  if (hint_count != SELECTED13_MATERIALIZATION_SOURCE_HINT_COUNT) failure_count++;
  if (SELECTED13_MATERIALIZATION_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;
  if (SELECTED13_MATERIALIZATION_RELATION_EXPORT_COUNT != 0ULL) failure_count++;
  if (row_count == 0ULL) failure_count++;

  for (size_t i = 0; i < row_count; i++) {{
    const selected13_materialization_source_hint_row_t *row = &SELECTED13_MATERIALIZATION_SOURCE_HINT_ROWS[i];
    if (row->row_request_id_u64 == 0ULL || row->row_material_hash_u64 == 0ULL) failure_count++;
    if (row->selected_support_mask == 0ULL || row->direct_status_code == 0ULL) failure_count++;
    if (row->exact_match_count == 0ULL) failure_count++;
    if (row->hint_start + row->hint_count > hint_count) failure_count++;
    if (row->exact_direct_public_key_verified && row->bridge_worker_required) failure_count++;
    if (!row->exact_direct_public_key_verified && !row->bridge_worker_required) failure_count++;
    if (row->exact_direct_public_key_verified) {{
      exact_direct++;
      if (row->is_best_manifest_row) best_exact_direct++;
      if (row->transfer_index != last_transfer) immediate_transfers++;
    }}
    if (row->bridge_worker_required) {{
      needs_bridge++;
      if (row->hint_count != 0ULL) hinted_needs++;
    }}
    last_transfer = row->transfer_index;
  }}
  for (size_t i = 0; i < hint_count; i++) {{
    const selected13_materialization_source_hint_t *hint = &SELECTED13_MATERIALIZATION_SOURCE_HINTS[i];
    if (hint->row_request_id_u64 == 0ULL || hint->hint_hash_u64 == 0ULL || hint->source_row_hash_u64 == 0ULL) {{
      failure_count++;
    }}
    if (hint->selected_support_mask == 0ULL || hint->direct_public_key_verified == 0ULL) failure_count++;
    if (hint->salt_overlap_count == 0ULL || hint->support_overlap_count == 0ULL) failure_count++;
  }}
  if (exact_direct != SELECTED13_MATERIALIZATION_EXACT_DIRECT_ROW_COUNT) failure_count++;
  if (needs_bridge != SELECTED13_MATERIALIZATION_BRIDGE_WORKER_REQUIRED_ROW_COUNT) failure_count++;
  if (exact_direct + needs_bridge != row_count) failure_count++;

  printf("selected13_materialization_source_hint_preflight rows=%llu exact_direct=%llu needs_bridge=%llu source_hints=%llu hinted_needs=%llu immediate_transfers=%llu failures=%llu\\n",
         (unsigned long long)row_count,
         (unsigned long long)exact_direct,
         (unsigned long long)needs_bridge,
         (unsigned long long)hint_count,
         (unsigned long long)hinted_needs,
         (unsigned long long)immediate_transfers,
         (unsigned long long)failure_count);
  (void)best_exact_direct;
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_mat_source_hint_") as tmp:
        tmp_path = Path(tmp)
        c_path = tmp_path / "preflight.c"
        exe_path = tmp_path / "preflight"
        local_header = tmp_path / header_path.name
        c_path.write_text(source)
        local_header.write_text(header_path.read_text())
        compile_cmd = ["cc", "-std=c99", "-Wall", "-Wextra", "-O2", str(c_path), "-o", str(exe_path)]
        compile_run = subprocess.run(compile_cmd, text=True, capture_output=True, check=False)
        if compile_run.returncode != 0:
            return {
                "compile_command": compile_cmd,
                "compile_returncode": compile_run.returncode,
                "compile_stderr": compile_run.stderr,
                "verified": False,
            }
        preflight_run = subprocess.run([str(exe_path)], text=True, capture_output=True, check=False)
        return {
            "compile_command": compile_cmd,
            "compile_returncode": compile_run.returncode,
            "preflight_returncode": preflight_run.returncode,
            "preflight_stdout": preflight_run.stdout.strip(),
            "preflight_stderr": preflight_run.stderr.strip(),
            "verified": preflight_run.returncode == 0,
        }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = Path(args.packet)
    workorder_path = Path(args.workorder)
    packet = load_json(packet_path)
    workorder = load_json(workorder_path)
    failures: list[dict[str, Any]] = []
    if packet.get("claim_status") != "SELECTED13_MATERIALIZATION_BRIDGE_PACKET_READY":
        failures.append({"code": "bridge_packet_not_ready", "claim_status": packet.get("claim_status")})
    if workorder.get("claim_status") != "SELECTED13_MATERIALIZATION_MISS_WORKORDER_READY":
        failures.append({"code": "materialization_workorder_not_ready", "claim_status": workorder.get("claim_status")})
    rows, hints, row_failures, source_meta = build_rows(packet, workorder, as_int(args.max_source_hints))
    failures.extend(row_failures)
    summary = summarize(rows, hints, failures)
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(summary, failures),
        "parameters": {
            "max_source_hints": as_int(args.max_source_hints),
            "packet": str(packet_path),
            "workorder": str(workorder_path),
        },
        "source_meta": source_meta,
        "source_summary": {
            "bridge_packet": packet.get("summary"),
            "workorder": workorder.get("summary"),
        },
        "summary": summary,
        "rows": rows,
        "source_hints": hints,
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "direct_rank_bridge_extended": False,
            "packet_and_source_hint_only": True,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--workorder", type=Path, default=DEFAULT_WORKORDER)
    parser.add_argument("--max-source-hints", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["rows"], payload["source_hints"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["claim_status"] = "SELECTED13_MATERIALIZATION_SOURCE_HINT_ADAPTER_FAILED"
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(args.out), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
