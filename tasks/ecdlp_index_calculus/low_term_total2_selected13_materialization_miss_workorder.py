#!/usr/bin/env python3
"""Build a direct/rank bridge workorder for selected13 materialization misses.

The 111/147 selected13 public policy leaves two different gaps: rule-selection
misses where the top-5 diagnostics already contain a below-rho leaf, and
materialization misses where the current top-5 diagnostic has no below-rho leaf.
This probe turns the materialization-miss list into a concrete row workorder
from the validation manifest.

It does not evaluate summation polynomials, export direct/rank rows, derive an
ECDLP secret, or claim a Pollard-rho speedup.  It is a hash-bound bridge
extension request for the next FFE/direct-rank worker pass.
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


SCHEMA = "ecdlp.low_term_total2_selected13_materialization_miss_workorder.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_VALIDATION_MANIFEST = (
    DEFAULT_STATE_DIR / "low_term_total2_public_lane_validation_manifest_selected13_nonadjacent_10056_10607_probe.json"
)
DEFAULT_FRONTIER_MINER = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_seeded_residual_rule_miner_111_all_insert_full147_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_miss_workorder_111_full147_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_materialization_miss_workorder_111_full147_probe.h"

STATUS_CODES = {
    "support_report_missing": 1,
    "direct_certificate_missing": 2,
}
FULL_FAMILY_COUNT = 3


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


def canonical_json(raw: Any) -> str:
    return json.dumps(raw, sort_keys=True, separators=(",", ":"))


def digest_hex(raw: Any) -> str:
    return hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()


def digest_u64(raw: Any) -> int:
    return int(digest_hex(raw)[:16], 16)


def support_mask(terms: Any) -> int:
    mask = 0
    for term in terms or []:
        value = as_int(term, -1)
        if value >= 0:
            mask |= 1 << value
    return mask


def row_key_tuple(raw: Any) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in (raw or [])))


def row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row_key_tuple(row.get("row_keys")),
        str(row.get("selector") or ""),
        as_int(row.get("top_k")),
        support_mask(row.get("selected_term_support")),
        tuple(as_int(item) for item in row.get("salts") or []),
    )


def materialization_misses(frontier_miner: dict[str, Any]) -> list[int]:
    summary = frontier_miner.get("summary") or {}
    return [as_int(item) for item in summary.get("residual_top5_materialization_misses") or []]


def rows_by_transfer(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(as_int(row.get("transfer_index"), -1), []).append(row)
    return out


def best_rows_by_transfer(queue: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out = {}
    for item in queue:
        transfer = as_int(item.get("transfer_index"), -1)
        best = item.get("best_row")
        if isinstance(best, dict):
            out[transfer] = best
    return out


def row_priority(row: dict[str, Any], best_identity: tuple[Any, ...] | None) -> tuple[int, int, int, float, str]:
    is_best = 1 if best_identity is not None and row_identity(row) == best_identity else 0
    full_family = 1 if len(row.get("matched_families") or []) >= FULL_FAMILY_COUNT else 0
    priority_hit_count = len(row.get("priority_hits") or [])
    ops = as_float(row.get("direct_ops_over_rho"))
    return (
        is_best,
        full_family,
        priority_hit_count,
        -(ops if ops is not None else 9.0),
        canonical_json(row_identity(row)),
    )


def row_contract(transfer: int, row_index: int, row: dict[str, Any], best_identity: tuple[Any, ...] | None) -> dict[str, Any]:
    material = {
        "artifact": row.get("artifact"),
        "row_keys": row.get("row_keys") or [],
        "selected_term_support": row.get("selected_term_support") or [],
        "selector": row.get("selector"),
        "target": row.get("target"),
        "top_k": as_int(row.get("top_k")),
        "transfer_index": transfer,
    }
    material_hash = digest_hex(material)
    direct_status = str(row.get("direct_status") or "")
    matched_families = row.get("matched_families") or []
    return {
        "artifact": row.get("artifact"),
        "candidate_row_index": row_index,
        "direct_ops_over_rho": as_float(row.get("direct_ops_over_rho")),
        "direct_status": direct_status,
        "direct_status_code": STATUS_CODES.get(direct_status, 0),
        "is_best_manifest_row": best_identity is not None and row_identity(row) == best_identity,
        "matched_family_count": len(matched_families),
        "matched_families": matched_families,
        "matched_rules": row.get("matched_rules") or [],
        "priority_hits": [as_int(item) for item in row.get("priority_hits") or []],
        "public_product_gate_selected": bool(row.get("public_product_gate_selected")),
        "range": row.get("range"),
        "row_keys": row.get("row_keys") or [],
        "row_material_hash": material_hash,
        "row_request_id": f"selected13_matmiss_{transfer}_{row_index}_{material_hash[:12]}",
        "row_request_id_u64": int(material_hash[:16], 16),
        "salts": [as_int(item) for item in row.get("salts") or []],
        "selected_support_mask": support_mask(row.get("selected_term_support")),
        "selected_term_support": [as_int(item) for item in row.get("selected_term_support") or []],
        "selector": row.get("selector"),
        "target": row.get("target"),
        "top_k": as_int(row.get("top_k")),
        "transfer_index": transfer,
        "worker_action": "extend_direct_rank_bridge_for_support_report_missing_row",
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.validation_manifest)
    frontier_path = Path(args.frontier_miner)
    manifest = load_json(manifest_path)
    frontier_miner = load_json(frontier_path)
    misses = materialization_misses(frontier_miner)
    if args.transfer_list:
        requested = {as_int(item.strip()) for item in args.transfer_list.split(",") if item.strip()}
        misses = [transfer for transfer in misses if transfer in requested]

    all_rows = [row for row in manifest.get("validation_rows") or [] if isinstance(row, dict)]
    rows_index = rows_by_transfer(all_rows)
    best_index = best_rows_by_transfer([item for item in manifest.get("direct_rank_transfer_queue") or [] if isinstance(item, dict)])

    failures: list[dict[str, Any]] = []
    work_items = []
    candidate_rows = []
    for transfer_order, transfer in enumerate(misses):
        rows = rows_index.get(transfer) or []
        best_identity = row_identity(best_index[transfer]) if transfer in best_index else None
        if not rows:
            failures.append({"code": "materialization_miss_rows_missing", "transfer_index": transfer})
        if best_identity is None:
            failures.append({"code": "materialization_miss_best_row_missing", "transfer_index": transfer})
        rows = sorted(rows, key=lambda row: row_priority(row, best_identity), reverse=True)
        contracts = [row_contract(transfer, index, row, best_identity) for index, row in enumerate(rows)]
        candidate_rows.extend(contracts)
        best_rows = [row for row in contracts if row.get("is_best_manifest_row")]
        if len(best_rows) != 1:
            failures.append(
                {
                    "code": "materialization_miss_best_row_count_unexpected",
                    "count": len(best_rows),
                    "transfer_index": transfer,
                }
            )
        work_items.append(
            {
                "bridge_extension_required": True,
                "candidate_row_count": len(contracts),
                "candidate_rows": contracts,
                "priority": transfer_order,
                "status": "awaiting_direct_rank_bridge_export",
                "transfer_index": transfer,
            }
        )

    direct_status_counts = Counter(str(row.get("direct_status")) for row in candidate_rows)
    latest_bridge_end = as_int((manifest.get("summary") or {}).get("latest_bridge_end"))
    latest_scanned_scout_end = as_int((manifest.get("summary") or {}).get("latest_scanned_scout_end"))
    result = {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": "SELECTED13_MATERIALIZATION_MISS_WORKORDER_READY" if not failures else "SELECTED13_MATERIALIZATION_MISS_WORKORDER_FAILED",
        "parameters": {
            "frontier_miner": str(frontier_path),
            "transfer_list": args.transfer_list,
            "validation_manifest": str(manifest_path),
        },
        "summary": {
            "accepted_relation_export_count": 0,
            "best_manifest_row_count": sum(1 for row in candidate_rows if row.get("is_best_manifest_row")),
            "bridge_extension_required": latest_bridge_end < max(misses or [0]),
            "candidate_row_count": len(candidate_rows),
            "direct_status_counts": dict(sorted(direct_status_counts.items())),
            "failure_count": len(failures),
            "full_family_candidate_row_count": sum(1 for row in candidate_rows if as_int(row.get("matched_family_count")) >= FULL_FAMILY_COUNT),
            "latest_bridge_end": latest_bridge_end,
            "latest_scanned_scout_end": latest_scanned_scout_end,
            "materialization_miss_count": len(misses),
            "materialization_miss_transfers": misses,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "verified": not failures,
            "worker_interpretation": (
                "These rows are concrete direct/rank bridge-extension requests for "
                "selected13 transfers whose current top-5 diagnostic has no below-rho "
                "leaf.  They remain work items until a fresh verifier exports "
                "relation_derived_ecdlp evidence."
            ),
        },
        "work_items": work_items,
        "candidate_rows": candidate_rows,
        "failures": failures,
        "honesty_boundary": {
            "direct_rank_bridge_extended": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "workorder_only": True,
        },
    }
    return result


def render_c_header(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    c_rows = []
    for row in rows:
        ops = as_float(row.get("direct_ops_over_rho"))
        ops_scaled = int(round((ops if ops is not None else 0.0) * 1_000_000))
        c_rows.append(
            "  {"
            f"{as_int(row.get('transfer_index'))}ULL, "
            f"{as_int(row.get('row_request_id_u64'))}ULL, "
            f"{as_int(row.get('candidate_row_index'))}ULL, "
            f"{1 if row.get('is_best_manifest_row') else 0}ULL, "
            f"{as_int(row.get('selected_support_mask'))}ULL, "
            f"{as_int(row.get('direct_status_code'))}ULL, "
            f"{as_int(row.get('matched_family_count'))}ULL, "
            f"{ops_scaled}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_MATERIALIZATION_MISS_WORKORDER_H
#define LOW_TERM_TOTAL2_SELECTED13_MATERIALIZATION_MISS_WORKORDER_H

#include <stdint.h>

#define SELECTED13_MATERIALIZATION_MISS_TRANSFER_COUNT {as_int(summary.get("materialization_miss_count"))}
#define SELECTED13_MATERIALIZATION_MISS_ROW_COUNT {len(rows)}
#define SELECTED13_MATERIALIZATION_MISS_BEST_ROW_COUNT {as_int(summary.get("best_manifest_row_count"))}
#define SELECTED13_MATERIALIZATION_MISS_LATEST_BRIDGE_END {as_int(summary.get("latest_bridge_end"))}

typedef struct {{
  uint64_t transfer_index;
  uint64_t row_request_id_u64;
  uint64_t candidate_row_index;
  uint64_t is_best_manifest_row;
  uint64_t selected_support_mask;
  uint64_t direct_status_code;
  uint64_t matched_family_count;
  uint64_t direct_ops_over_rho_scaled_1e6;
}} selected13_materialization_miss_row_t;

static const selected13_materialization_miss_row_t SELECTED13_MATERIALIZATION_MISS_ROWS[] = {{
{chr(10).join(c_rows)}
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
  uint64_t row_count = sizeof(SELECTED13_MATERIALIZATION_MISS_ROWS) / sizeof(SELECTED13_MATERIALIZATION_MISS_ROWS[0]);
  uint64_t best_rows = 0;
  uint64_t transfer_count = 0;
  uint64_t previous_transfer = 0ULL;

  if (row_count != SELECTED13_MATERIALIZATION_MISS_ROW_COUNT) failure_count++;
  if (row_count == 0ULL) failure_count++;
  for (size_t i = 0; i < row_count; i++) {{
    const selected13_materialization_miss_row_t *row = &SELECTED13_MATERIALIZATION_MISS_ROWS[i];
    if (row->row_request_id_u64 == 0ULL) failure_count++;
    if (row->selected_support_mask == 0ULL) failure_count++;
    if (row->direct_status_code == 0ULL) failure_count++;
    if (row->is_best_manifest_row) best_rows++;
    if (i == 0 || row->transfer_index != previous_transfer) transfer_count++;
    previous_transfer = row->transfer_index;
  }}
  if (transfer_count != SELECTED13_MATERIALIZATION_MISS_TRANSFER_COUNT) failure_count++;
  if (best_rows != SELECTED13_MATERIALIZATION_MISS_BEST_ROW_COUNT) failure_count++;
  if (best_rows != SELECTED13_MATERIALIZATION_MISS_TRANSFER_COUNT) failure_count++;
  if (SELECTED13_MATERIALIZATION_MISS_LATEST_BRIDGE_END >= previous_transfer) failure_count++;

  printf("selected13_materialization_miss_workorder_preflight transfers=%llu rows=%llu best_rows=%llu bridge_end=%llu failures=%llu\\n",
         (unsigned long long)transfer_count,
         (unsigned long long)row_count,
         (unsigned long long)best_rows,
         (unsigned long long)SELECTED13_MATERIALIZATION_MISS_LATEST_BRIDGE_END,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_materialization_workorder_preflight_") as tmp:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--frontier-miner", type=Path, default=DEFAULT_FRONTIER_MINER)
    parser.add_argument("--transfer-list", default="")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--c-header-out", type=Path, default=DEFAULT_C_HEADER_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["candidate_rows"], payload["summary"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["claim_status"] = "SELECTED13_MATERIALIZATION_MISS_WORKORDER_FAILED"
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(args.out), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
