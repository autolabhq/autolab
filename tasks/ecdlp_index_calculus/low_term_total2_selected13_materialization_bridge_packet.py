#!/usr/bin/env python3
"""Lower selected13 materialization-miss workorders into bridge packets.

The materialization workorder identifies support-missing validation rows for
the selected13 10056-10607 frontier.  This script turns that JSON workorder into
a typed packet/ABI for the next direct-rank bridge extension worker.

It does not evaluate summation polynomials, export direct/rank rows, derive an
ECDLP scalar, or claim a Pollard-rho speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_selected13_materialization_bridge_packet.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_WORKORDER = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_miss_workorder_111_full147_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_bridge_packet_111_full147_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_bridge_packet_111_full147_probe.h"


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


def canonical_json(raw: Any) -> str:
    return json.dumps(raw, sort_keys=True, separators=(",", ":"))


def digest_u64(raw: Any) -> int:
    return int(hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()[:16], 16)


def support_mask(terms: Any) -> int:
    mask = 0
    for term in terms or []:
        value = as_int(term, -1)
        if value >= 0:
            mask |= 1 << value
    return mask


def family_masks(row: dict[str, Any]) -> list[int]:
    return [support_mask(family) for family in row.get("matched_families") or []]


def packet_hash(packet: dict[str, Any]) -> int:
    return digest_u64(
        {
            "best_row_request_id": packet.get("best_row_request_id"),
            "row_request_ids": packet.get("row_request_ids"),
            "transfer_index": packet.get("transfer_index"),
        }
    )


def row_key_hash(row: dict[str, Any]) -> int:
    return digest_u64({"row_keys": row.get("row_keys") or [], "target": row.get("target")})


def lane_hash(packet_index: int, row: dict[str, Any], family_mask: int) -> int:
    return digest_u64(
        {
            "family_mask": family_mask,
            "packet_index": packet_index,
            "row_request_id": row.get("row_request_id"),
            "transfer_index": row.get("transfer_index"),
        }
    )


def build_packets(workorder: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    packets = []
    rows_out = []
    lanes = []
    global_row_offset = 0

    for packet_index, item in enumerate(workorder.get("work_items") or []):
        transfer = as_int(item.get("transfer_index"), -1)
        rows = [row for row in item.get("candidate_rows") or [] if isinstance(row, dict)]
        best_rows = [row for row in rows if row.get("is_best_manifest_row")]
        if len(best_rows) != 1:
            failures.append({"code": "packet_best_row_count_unexpected", "count": len(best_rows), "transfer_index": transfer})
        best = best_rows[0] if best_rows else (rows[0] if rows else {})
        best_local_index = rows.index(best) if best in rows else 0
        salts = [as_int(salt) for salt in best.get("salts") or []]
        while len(salts) < 2:
            salts.append(0)
        row_request_ids = [str(row.get("row_request_id") or "") for row in rows]
        packet = {
            "best_row_global_index": global_row_offset + best_local_index,
            "best_row_request_id": best.get("row_request_id"),
            "best_row_request_id_u64": as_int(best.get("row_request_id_u64")),
            "candidate_row_count": len(rows),
            "global_row_offset": global_row_offset,
            "packet_index": packet_index,
            "required_output": {
                "must_emit_direct_rank_bridge_rows": True,
                "must_match_row_material_hash": True,
                "must_set_transfer_index": transfer,
                "relation_derived_ecdlp_required_for_success_claim": True,
            },
            "row_key_hash_u64": row_key_hash(best) if best else 0,
            "row_request_ids": row_request_ids,
            "salt0": salts[0],
            "salt1": salts[1],
            "salt_gap": abs(salts[1] - salts[0]) if len(salts) >= 2 else 0,
            "target": best.get("target"),
            "target_hash_u64": digest_u64(best.get("target")),
            "transfer_index": transfer,
            "worker_action": "fresh_ffe_summation_polynomial_direct_rank_bridge_extension",
        }
        packet["packet_hash_u64"] = packet_hash(packet)
        packets.append(packet)

        for local_row_index, row in enumerate(rows):
            global_index = global_row_offset + local_row_index
            masks = family_masks(row)
            row_out = {
                "candidate_row_index": as_int(row.get("candidate_row_index")),
                "direct_ops_over_rho": row.get("direct_ops_over_rho"),
                "direct_status": row.get("direct_status"),
                "direct_status_code": as_int(row.get("direct_status_code")),
                "global_row_index": global_index,
                "is_best_manifest_row": bool(row.get("is_best_manifest_row")),
                "matched_family_count": as_int(row.get("matched_family_count")),
                "matched_family_masks": masks,
                "packet_index": packet_index,
                "requires_direct_rank_export": True,
                "row_keys": row.get("row_keys") or [],
                "row_key_hash_u64": row_key_hash(row),
                "row_material_hash": row.get("row_material_hash"),
                "row_material_hash_u64": int(str(row.get("row_material_hash") or "0")[:16], 16),
                "row_request_id": row.get("row_request_id"),
                "row_request_id_u64": as_int(row.get("row_request_id_u64")),
                "salts": [as_int(salt) for salt in row.get("salts") or []],
                "selected_support_mask": as_int(row.get("selected_support_mask")),
                "selected_term_support": [as_int(term) for term in row.get("selected_term_support") or []],
                "selector": row.get("selector"),
                "target": row.get("target"),
                "top_k": as_int(row.get("top_k")),
                "transfer_index": transfer,
                "worker_acceptance_gate": "fresh_direct_rank_export_and_relation_verifier",
            }
            rows_out.append(row_out)
            for family_index, family in enumerate(row.get("matched_families") or []):
                family_terms = [as_int(term) for term in family]
                mask = support_mask(family_terms)
                lanes.append(
                    {
                        "family_terms": family_terms,
                        "family_index": family_index,
                        "family_mask": mask,
                        "global_row_index": global_index,
                        "is_best_manifest_row": bool(row.get("is_best_manifest_row")),
                        "lane_hash_u64": lane_hash(packet_index, row, mask),
                        "packet_index": packet_index,
                        "row_request_id": row.get("row_request_id"),
                        "row_request_id_u64": as_int(row.get("row_request_id_u64")),
                        "transfer_index": transfer,
                        "worker_action": "fresh_ffe_summation_polynomial_family_probe",
                    }
                )
        global_row_offset += len(rows)
    return packets, rows_out, lanes, failures


def summarize(
    packets: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    lanes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    direct_status_counts = Counter(str(row.get("direct_status")) for row in rows)
    target_transfers = [as_int(packet.get("transfer_index")) for packet in packets]
    latest_bridge_end = as_int(source_summary.get("latest_bridge_end"))
    latest_transfer = max(target_transfers or [0])
    return {
        "accepted_relation_export_count": 0,
        "bridge_extension_required": latest_bridge_end < latest_transfer,
        "latest_bridge_end": latest_bridge_end,
        "latest_transfer": latest_transfer,
        "bridge_packet_count": len(packets),
        "bridge_row_count": len(rows),
        "direct_rank_export_required_count": sum(1 for row in rows if row.get("requires_direct_rank_export")),
        "direct_status_counts": dict(sorted(direct_status_counts.items())),
        "failure_count": len(failures),
        "full_family_lane_count": len(lanes),
        "packet_best_row_count": sum(1 for row in rows if row.get("is_best_manifest_row")),
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "target_transfers": target_transfers,
        "verified": not failures,
        "worker_interpretation": (
            "This packet is the typed ABI surface for a fresh FFE/summation-polynomial "
            "direct-rank bridge extension over selected13 materialization misses. "
            "It is not a relation export until a worker emits verifier-backed rows."
        ),
    }


def render_c_header(
    packets: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    lanes: list[dict[str, Any]],
    latest_bridge_end: int,
) -> str:
    packet_lines = []
    for packet in packets:
        packet_lines.append(
            "  {"
            f"{as_int(packet.get('packet_index'))}ULL, "
            f"{as_int(packet.get('packet_hash_u64'))}ULL, "
            f"{as_int(packet.get('transfer_index'))}ULL, "
            f"{as_int(packet.get('target_hash_u64'))}ULL, "
            f"{as_int(packet.get('row_key_hash_u64'))}ULL, "
            f"{as_int(packet.get('best_row_request_id_u64'))}ULL, "
            f"{as_int(packet.get('best_row_global_index'))}ULL, "
            f"{as_int(packet.get('salt0'))}ULL, "
            f"{as_int(packet.get('salt1'))}ULL, "
            f"{as_int(packet.get('salt_gap'))}ULL, "
            f"{as_int(packet.get('global_row_offset'))}ULL, "
            f"{as_int(packet.get('candidate_row_count'))}ULL"
            "},"
        )
    row_lines = []
    for row in rows:
        row_lines.append(
            "  {"
            f"{as_int(row.get('packet_index'))}ULL, "
            f"{as_int(row.get('global_row_index'))}ULL, "
            f"{as_int(row.get('row_request_id_u64'))}ULL, "
            f"{as_int(row.get('row_material_hash_u64'))}ULL, "
            f"{as_int(row.get('row_key_hash_u64'))}ULL, "
            f"{as_int(row.get('selected_support_mask'))}ULL, "
            f"{as_int(row.get('direct_status_code'))}ULL, "
            f"{as_int(row.get('matched_family_count'))}ULL, "
            f"{1 if row.get('is_best_manifest_row') else 0}ULL, "
            f"{1 if row.get('requires_direct_rank_export') else 0}ULL"
            "},"
        )
    lane_lines = []
    for lane in lanes:
        lane_lines.append(
            "  {"
            f"{as_int(lane.get('packet_index'))}ULL, "
            f"{as_int(lane.get('global_row_index'))}ULL, "
            f"{as_int(lane.get('row_request_id_u64'))}ULL, "
            f"{as_int(lane.get('family_index'))}ULL, "
            f"{as_int(lane.get('family_mask'))}ULL, "
            f"{as_int(lane.get('lane_hash_u64'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_MATERIALIZATION_BRIDGE_PACKET_H
#define LOW_TERM_TOTAL2_SELECTED13_MATERIALIZATION_BRIDGE_PACKET_H

#include <stdint.h>

#define SELECTED13_MATERIALIZATION_BRIDGE_PACKET_COUNT {len(packets)}
#define SELECTED13_MATERIALIZATION_BRIDGE_ROW_COUNT {len(rows)}
#define SELECTED13_MATERIALIZATION_BRIDGE_FAMILY_LANE_COUNT {len(lanes)}
#define SELECTED13_MATERIALIZATION_BRIDGE_LATEST_BRIDGE_END {latest_bridge_end}

typedef struct {{
  uint64_t packet_index;
  uint64_t packet_hash_u64;
  uint64_t transfer_index;
  uint64_t target_hash_u64;
  uint64_t row_key_hash_u64;
  uint64_t best_row_request_id_u64;
  uint64_t best_row_global_index;
  uint64_t salt0;
  uint64_t salt1;
  uint64_t salt_gap;
  uint64_t row_offset;
  uint64_t row_count;
}} selected13_materialization_bridge_packet_t;

typedef struct {{
  uint64_t packet_index;
  uint64_t global_row_index;
  uint64_t row_request_id_u64;
  uint64_t row_material_hash_u64;
  uint64_t row_key_hash_u64;
  uint64_t selected_support_mask;
  uint64_t direct_status_code;
  uint64_t matched_family_count;
  uint64_t is_best_manifest_row;
  uint64_t requires_direct_rank_export;
}} selected13_materialization_bridge_row_t;

typedef struct {{
  uint64_t packet_index;
  uint64_t global_row_index;
  uint64_t row_request_id_u64;
  uint64_t family_index;
  uint64_t family_mask;
  uint64_t lane_hash_u64;
}} selected13_materialization_bridge_family_lane_t;

static const selected13_materialization_bridge_packet_t SELECTED13_MATERIALIZATION_BRIDGE_PACKETS[] = {{
{chr(10).join(packet_lines)}
}};

static const selected13_materialization_bridge_row_t SELECTED13_MATERIALIZATION_BRIDGE_ROWS[] = {{
{chr(10).join(row_lines)}
}};

static const selected13_materialization_bridge_family_lane_t SELECTED13_MATERIALIZATION_BRIDGE_FAMILY_LANES[] = {{
{chr(10).join(lane_lines)}
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
  uint64_t packet_count = sizeof(SELECTED13_MATERIALIZATION_BRIDGE_PACKETS) / sizeof(SELECTED13_MATERIALIZATION_BRIDGE_PACKETS[0]);
  uint64_t row_count = sizeof(SELECTED13_MATERIALIZATION_BRIDGE_ROWS) / sizeof(SELECTED13_MATERIALIZATION_BRIDGE_ROWS[0]);
  uint64_t lane_count = sizeof(SELECTED13_MATERIALIZATION_BRIDGE_FAMILY_LANES) / sizeof(SELECTED13_MATERIALIZATION_BRIDGE_FAMILY_LANES[0]);
  uint64_t best_rows = 0;
  uint64_t direct_required = 0;
  uint64_t latest_transfer = 0;

  if (packet_count != SELECTED13_MATERIALIZATION_BRIDGE_PACKET_COUNT) failure_count++;
  if (row_count != SELECTED13_MATERIALIZATION_BRIDGE_ROW_COUNT) failure_count++;
  if (lane_count != SELECTED13_MATERIALIZATION_BRIDGE_FAMILY_LANE_COUNT) failure_count++;
  if (packet_count == 0ULL || row_count == 0ULL || lane_count == 0ULL) failure_count++;

  for (size_t i = 0; i < packet_count; i++) {{
    const selected13_materialization_bridge_packet_t *packet = &SELECTED13_MATERIALIZATION_BRIDGE_PACKETS[i];
    if (packet->packet_hash_u64 == 0ULL || packet->target_hash_u64 == 0ULL) failure_count++;
    if (packet->row_count != 3ULL) failure_count++;
    if (packet->row_offset + packet->row_count > row_count) failure_count++;
    if (packet->best_row_request_id_u64 == 0ULL || packet->best_row_global_index >= row_count) {{
      failure_count++;
    }} else {{
      const selected13_materialization_bridge_row_t *best_row = &SELECTED13_MATERIALIZATION_BRIDGE_ROWS[packet->best_row_global_index];
      if (!best_row->is_best_manifest_row || best_row->row_request_id_u64 != packet->best_row_request_id_u64) failure_count++;
    }}
    if (packet->transfer_index > latest_transfer) latest_transfer = packet->transfer_index;
  }}
  for (size_t i = 0; i < row_count; i++) {{
    const selected13_materialization_bridge_row_t *row = &SELECTED13_MATERIALIZATION_BRIDGE_ROWS[i];
    if (row->row_request_id_u64 == 0ULL || row->row_material_hash_u64 == 0ULL) failure_count++;
    if (row->selected_support_mask == 0ULL || row->direct_status_code == 0ULL) failure_count++;
    if (row->requires_direct_rank_export) direct_required++;
    if (row->is_best_manifest_row) best_rows++;
  }}
  for (size_t i = 0; i < lane_count; i++) {{
    const selected13_materialization_bridge_family_lane_t *lane = &SELECTED13_MATERIALIZATION_BRIDGE_FAMILY_LANES[i];
    if (lane->row_request_id_u64 == 0ULL || lane->family_mask == 0ULL || lane->lane_hash_u64 == 0ULL) failure_count++;
  }}
  if (best_rows != packet_count) failure_count++;
  if (direct_required != row_count) failure_count++;
  if (SELECTED13_MATERIALIZATION_BRIDGE_LATEST_BRIDGE_END >= latest_transfer) failure_count++;

  printf("selected13_materialization_bridge_packet_preflight packets=%llu rows=%llu lanes=%llu direct_required=%llu bridge_end=%llu failures=%llu\\n",
         (unsigned long long)packet_count,
         (unsigned long long)row_count,
         (unsigned long long)lane_count,
         (unsigned long long)direct_required,
         (unsigned long long)SELECTED13_MATERIALIZATION_BRIDGE_LATEST_BRIDGE_END,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_materialization_bridge_packet_") as tmp:
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
    workorder_path = Path(args.workorder)
    workorder = load_json(workorder_path)
    failures: list[dict[str, Any]] = []
    if workorder.get("claim_status") != "SELECTED13_MATERIALIZATION_MISS_WORKORDER_READY":
        failures.append({"code": "materialization_workorder_not_ready", "claim_status": workorder.get("claim_status")})
    packets, rows, lanes, packet_failures = build_packets(workorder)
    failures.extend(packet_failures)
    source_summary = workorder.get("summary") or {}
    target_transfers = [as_int(packet.get("transfer_index")) for packet in packets]
    latest_bridge_end = as_int(source_summary.get("latest_bridge_end"))
    if target_transfers and latest_bridge_end >= max(target_transfers):
        failures.append(
            {
                "code": "materialization_packet_not_beyond_bridge_frontier",
                "latest_bridge_end": latest_bridge_end,
                "latest_transfer": max(target_transfers),
            }
        )
    summary = summarize(packets, rows, lanes, failures, source_summary)
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": "SELECTED13_MATERIALIZATION_BRIDGE_PACKET_READY" if not failures else "SELECTED13_MATERIALIZATION_BRIDGE_PACKET_FAILED",
        "parameters": {"workorder": str(workorder_path)},
        "source_summary": source_summary,
        "summary": summary,
        "packets": packets,
        "rows": rows,
        "family_lanes": lanes,
        "failures": failures,
        "honesty_boundary": {
            "direct_rank_bridge_extended": False,
            "packet_only": True,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workorder", type=Path, default=DEFAULT_WORKORDER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(
        render_c_header(
            payload["packets"],
            payload["rows"],
            payload["family_lanes"],
            as_int(payload["summary"].get("latest_bridge_end")),
        )
    )
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["claim_status"] = "SELECTED13_MATERIALIZATION_BRIDGE_PACKET_FAILED"
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(args.out), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
