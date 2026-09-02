#!/usr/bin/env python3
"""Materialize selected13 fresh-emission worker cases from host scout rows.

The fresh-emission packet binds row IDs, row hashes, negative replay controls,
and family lanes.  The selected-leaf support scouts still carry the public
case rows that a lower-level FFE/summation-polynomial worker would attempt.
This adapter joins those two layers for the priority selected13 targets.

It intentionally stops at the worker boundary.  A materialized case with
``direct_public_key_verified == false`` is not a relation export, not an ECDLP
recovery, and not evidence of a Pollard-rho speedup.  It is a concrete,
hash-bound obligation for a fresh direct/rank verifier.
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


SCHEMA = "ecdlp.low_term_total2_selected13_fresh_materialization_adapter.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SUPPORT_STATE_DIR = Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state")
DEFAULT_PACKET = (
    DEFAULT_STATE_DIR / "low_term_total2_ffe_sharp_lane_fresh_emission_packet_selected13_9696_9999_probe.json"
)
DEFAULT_WORKORDER = DEFAULT_STATE_DIR / "low_term_total2_selected13_workorder_9696_9999_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_fresh_materialization_adapter_9981_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_fresh_materialization_adapter_9981_probe.h"

RESULT_CODES = {
    "SOURCE_SECRET_REPLAY_ONLY": 1,
    "SOURCE_FORM_SYSTEM_INCONSISTENT": 2,
    "NO_UNIQUE_SECRET_FROM_REPLAY_FORMS": 3,
    "UNVERIFIED_NEW_LINEAR_DERIVATION": 4,
}
SOURCE_TIER_CODES = {
    "same_row_key_exact_support": 1,
    "one_salt_neighbor_exact_support": 2,
    "support_span_only": 3,
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


def digest_u64(raw: Any) -> int:
    if raw is None:
        return 0
    text = str(raw)
    suffix = text.rsplit("_", 1)[-1]
    if suffix and all(ch in "0123456789abcdefABCDEF" for ch in suffix):
        return int(suffix[-16:], 16)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def scaled_ops(value: Any) -> int:
    number = as_float(value)
    if number is None:
        return 0
    return round(number * 100_000_000)


def parse_targets(raw: str) -> list[int]:
    targets = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not targets:
        raise ValueError("At least one target transfer is required")
    return targets


def parse_paths(raw: str) -> list[Path]:
    return [Path(part.strip()) for part in raw.split(",") if part.strip()]


def support_tuple(raw: Any) -> tuple[int, ...]:
    return tuple(sorted(as_int(item) for item in (raw or [])))


def row_key_tuple(raw: Any) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in (raw or [])))


def families_key(raw: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(support_tuple(item) for item in (raw or [])))


def family_mask(terms: Any) -> int:
    mask = 0
    for term in terms or []:
        value = as_int(term)
        if value >= 0:
            mask |= 1 << value
    return mask


def workorder_rows_by_key(workorder: dict[str, Any]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in workorder.get("candidate_rows") or []:
        if not isinstance(row, dict):
            continue
        key = (
            as_int(row.get("transfer_index"), -1),
            row_key_tuple(row.get("row_keys")),
            str(row.get("selector") or ""),
            as_int(row.get("top_k")),
            support_tuple(row.get("selected_term_support")),
        )
        out.setdefault(key, []).append(row)
    return out


def scout_rows_by_key(paths: list[Path]) -> tuple[dict[tuple[Any, ...], list[dict[str, Any]]], list[str], dict[str, Any]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    missing: list[str] = []
    summaries: dict[str, Any] = {}
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
            key = (
                as_int(row.get("transfer_index"), -1),
                row_key_tuple(row.get("row_keys")),
                str(row.get("selector") or ""),
                as_int(row.get("top_k")),
                support_tuple(row.get("selected_term_support")),
            )
            item = dict(row)
            item["artifact"] = str(path)
            out.setdefault(key, []).append(item)
    return out, missing, summaries


def default_scout_path(support_state_dir: Path, range_label: str) -> Path:
    return (
        support_state_dir
        / f"low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_{range_label}_probe.json"
    )


def selected_packet_targets(packet: dict[str, Any], targets: list[int]) -> list[dict[str, Any]]:
    target_set = set(targets)
    selected = [
        target
        for target in packet.get("emission_targets") or []
        if isinstance(target, dict) and as_int(target.get("transfer_index"), -1) in target_set
    ]
    selected.sort(key=lambda item: targets.index(as_int(item.get("transfer_index"), -1)))
    return selected


def target_key(target: dict[str, Any]) -> tuple[Any, ...]:
    full = target.get("full_family_row") or {}
    public = target.get("public_first_pass") or {}
    return (
        as_int(target.get("transfer_index"), -1),
        row_key_tuple(public.get("row_keys")),
        str(full.get("selector") or ""),
        as_int(full.get("top_k")),
        support_tuple(full.get("selected_term_support")),
    )


def collect_default_scouts(
    packet_targets: list[dict[str, Any]],
    workorder_index: dict[tuple[Any, ...], list[dict[str, Any]]],
    support_state_dir: Path,
) -> list[Path]:
    paths = []
    for target in packet_targets:
        rows = workorder_index.get(target_key(target)) or []
        range_label = str((rows[0] if rows else {}).get("range") or "")
        if range_label:
            paths.append(default_scout_path(support_state_dir, range_label))
    return sorted(set(paths))


def lane_summary(lane: dict[str, Any]) -> dict[str, Any]:
    source_tier = str(lane.get("source_tier") or "")
    result = str(lane.get("replay_result_status") or "")
    return {
        "candidate_form_count": as_int(lane.get("candidate_form_count")),
        "family_mask": as_int(lane.get("family_mask")),
        "family_terms": [as_int(item) for item in lane.get("family_terms") or []],
        "fresh_source_solve_required": bool(lane.get("fresh_source_solve_required")),
        "lane_hash_u64": as_int(lane.get("lane_hash_u64")),
        "lane_index": as_int(lane.get("lane_index"), -1),
        "replay_result_code": RESULT_CODES.get(result, 0),
        "replay_result_status": result,
        "source_secret_count": as_int(lane.get("source_secret_count")),
        "source_tier": source_tier,
        "source_tier_code": SOURCE_TIER_CODES.get(source_tier, 0),
        "source_transfers": [as_int(item) for item in lane.get("source_transfers") or []],
        "worker_action": lane.get("worker_action"),
    }


def build_worker_case(
    target: dict[str, Any],
    workorder_rows: list[dict[str, Any]],
    scout_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transfer = as_int(target.get("transfer_index"), -1)
    full = target.get("full_family_row") or {}
    public = target.get("public_first_pass") or {}
    row_id = str(full.get("row_id") or target.get("backfill_row_id") or "")
    row_hash = str(full.get("row_check_hash") or target.get("backfill_row_check_hash") or "")
    lanes = [lane_summary(lane) for lane in target.get("family_lanes") or []]
    lane_masks = sorted(as_int(lane.get("family_mask")) for lane in lanes)
    lane_families = families_key([lane.get("family_terms") for lane in lanes])
    failures: list[dict[str, Any]] = []

    if len(workorder_rows) != 1:
        failures.append({"code": "workorder_exact_row_match_count_mismatch", "transfer_index": transfer, "count": len(workorder_rows)})
    if len(scout_rows) != 1:
        failures.append({"code": "support_scout_exact_row_match_count_mismatch", "transfer_index": transfer, "count": len(scout_rows)})

    workorder_row = workorder_rows[0] if workorder_rows else {}
    scout_row = scout_rows[0] if scout_rows else {}
    workorder_families = families_key(workorder_row.get("matched_families"))
    scout_direct_verified = bool(scout_row.get("direct_public_key_verified"))
    scout_shared_verified = bool(scout_row.get("shared_product_public_key_verified"))
    direct_ops = as_float(scout_row.get("direct_ops_over_rho") or workorder_row.get("direct_ops_over_rho"))
    below_rho = bool(direct_ops is not None and direct_ops < 1.0)

    if workorder_families != lane_families:
        failures.append(
            {
                "code": "workorder_family_lane_mismatch",
                "transfer_index": transfer,
                "workorder_families": [list(item) for item in workorder_families],
                "packet_families": [list(item) for item in lane_families],
            }
        )
    if str(target.get("full_gate_replay_result") or "") != "SOURCE_FORM_SYSTEM_INCONSISTENT":
        failures.append(
            {
                "code": "full_gate_negative_control_missing",
                "transfer_index": transfer,
                "result": target.get("full_gate_replay_result"),
            }
        )
    for lane in lanes:
        if lane.get("replay_result_status") != "SOURCE_SECRET_REPLAY_ONLY":
            failures.append(
                {
                    "code": "family_lane_negative_control_missing",
                    "transfer_index": transfer,
                    "family_mask": lane.get("family_mask"),
                    "result": lane.get("replay_result_status"),
                }
            )
    if str(full.get("direct_status") or "") != "direct_certificate_missing":
        failures.append({"code": "packet_full_family_not_direct_missing", "transfer_index": transfer})
    if str(workorder_row.get("direct_status") or "") != "direct_certificate_missing":
        failures.append({"code": "workorder_full_family_not_direct_missing", "transfer_index": transfer})
    if not row_id or not row_hash:
        failures.append({"code": "packet_row_identity_missing", "transfer_index": transfer})

    case_arg = (
        f"{public.get('target')}|{transfer}|{full.get('selector')}|{as_int(full.get('top_k'))}"
        if public
        else None
    )
    worker_case = {
        "accepted_relation_export": False,
        "backfill_row_check_hash": row_hash,
        "backfill_row_check_hash_u64": as_int(full.get("row_check_hash_u64")) or digest_u64(row_hash),
        "backfill_row_id": row_id,
        "backfill_row_id_u64": as_int(full.get("row_id_u64")) or digest_u64(row_id),
        "below_rho_label": below_rho,
        "candidate_case_arg": case_arg,
        "direct_ops_over_rho": direct_ops,
        "direct_ops_over_rho_scaled": scaled_ops(direct_ops),
        "direct_public_key_verified": scout_direct_verified,
        "direct_status": full.get("direct_status"),
        "family_lane_count": len(lanes),
        "family_lanes": lanes,
        "family_masks": lane_masks,
        "full_gate_replay_result": target.get("full_gate_replay_result"),
        "full_gate_replay_result_code": RESULT_CODES.get(str(target.get("full_gate_replay_result") or ""), 0),
        "materialized": not failures,
        "matched_families": [list(item) for item in lane_families],
        "needs_fresh_direct_verification": not scout_direct_verified,
        "observed_support_scout": {
            "artifact": scout_row.get("artifact"),
            "direct_public_key_verified": scout_direct_verified,
            "public_product_gate_selected": bool(scout_row.get("public_product_gate_selected")),
            "shared_product_public_key_verified": scout_shared_verified,
            "shared_product_relation_count": as_int(scout_row.get("shared_product_relation_count")),
        },
        "relation_derived_ecdlp": False,
        "required_output": {
            "must_match_backfill_row_check_hash": row_hash,
            "must_match_backfill_row_id": row_id,
            "must_not_use_copied_source_form_solve": True,
            "must_recompute_candidate_equality": True,
            "must_set_backfill_transfer_index": transfer,
            "must_verify_public_key": True,
            "must_write_direct_rank_export": True,
        },
        "row_keys": list(row_key_tuple(public.get("row_keys"))),
        "selected_term_support": list(support_tuple(full.get("selected_term_support"))),
        "selector": full.get("selector"),
        "shared_product_public_key_verified": scout_shared_verified,
        "target": public.get("target"),
        "top_k": as_int(full.get("top_k")),
        "transfer_index": transfer,
        "workorder": {
            "direct_status": workorder_row.get("direct_status"),
            "range": workorder_row.get("range"),
            "score": as_int(workorder_row.get("score")),
        },
        "worker_acceptance_gate": "fresh_ffe_summation_polynomial_direct_rank_export_only",
    }
    return worker_case, failures


def validate_sources(packet: dict[str, Any], workorder: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if packet.get("claim_status") != "FFE_SHARP_LANE_FRESH_EMISSION_PACKET_READY":
        failures.append({"code": "fresh_emission_packet_not_ready", "claim_status": packet.get("claim_status")})
    if packet.get("failures"):
        failures.append({"code": "fresh_emission_packet_has_failures", "failures": packet.get("failures")})
    if (packet.get("summary") or {}).get("accepted_backfill_export_count") != 0:
        failures.append({"code": "packet_claims_backfill_export", "summary": packet.get("summary")})
    if workorder.get("claim_status") != "SHARED_SUBCARRIER_HAS_DIRECT_MISSING_WORK_ORDERS":
        failures.append({"code": "selected13_workorder_not_direct_missing_queue", "claim_status": workorder.get("claim_status")})
    return failures


def claim_status(failures: list[dict[str, Any]], worker_cases: list[dict[str, Any]]) -> str:
    if failures:
        return "SELECTED13_FRESH_MATERIALIZATION_ADAPTER_FAILED"
    if all(bool(case.get("direct_public_key_verified")) for case in worker_cases):
        return "SELECTED13_FRESH_MATERIALIZATION_DIRECT_VERIFIED_READY"
    return "SELECTED13_FRESH_MATERIALIZATION_NEEDS_FRESH_DIRECT_VERIFICATION"


def render_c_header(worker_cases: list[dict[str, Any]]) -> str:
    target_rows = []
    lane_rows = []
    for case in worker_cases:
        target_rows.append(
            "  {"
            f"{as_int(case.get('transfer_index'))}ULL, "
            f"{as_int(case.get('backfill_row_id_u64'))}ULL, "
            f"{as_int(case.get('backfill_row_check_hash_u64'))}ULL, "
            f"{as_int(case.get('direct_ops_over_rho_scaled'))}ULL, "
            f"{1 if case.get('direct_public_key_verified') else 0}ULL, "
            f"{1 if case.get('shared_product_public_key_verified') else 0}ULL, "
            f"{as_int(case.get('full_gate_replay_result_code'))}ULL, "
            f"{1 if case.get('materialized') else 0}ULL, "
            f"{1 if case.get('needs_fresh_direct_verification') else 0}ULL, "
            f"{1 if case.get('below_rho_label') else 0}ULL"
            "},"
        )
        for lane in case.get("family_lanes") or []:
            lane_rows.append(
                "  {"
                f"{as_int(case.get('transfer_index'))}ULL, "
                f"{as_int(lane.get('family_mask'))}ULL, "
                f"{as_int(lane.get('source_tier_code'))}ULL, "
                f"{as_int(lane.get('source_secret_count'))}ULL, "
                f"{as_int(lane.get('candidate_form_count'))}ULL, "
                f"{as_int(lane.get('replay_result_code'))}ULL, "
                f"{1 if lane.get('fresh_source_solve_required') else 0}ULL, "
                f"{as_int(lane.get('lane_hash_u64'))}ULL"
                "},"
            )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_FRESH_MATERIALIZATION_ADAPTER_H
#define LOW_TERM_TOTAL2_SELECTED13_FRESH_MATERIALIZATION_ADAPTER_H

#include <stdint.h>

#define SELECTED13_FRESH_MATERIALIZATION_TARGET_COUNT {len(worker_cases)}
#define SELECTED13_FRESH_MATERIALIZATION_FAMILY_LANE_COUNT {sum(as_int(case.get('family_lane_count')) for case in worker_cases)}
#define SELECTED13_FRESH_MATERIALIZATION_DIRECT_VERIFIED_COUNT {sum(1 for case in worker_cases if case.get('direct_public_key_verified'))}
#define SELECTED13_FRESH_MATERIALIZATION_RELATION_EXPORT_COUNT 0
#define SELECTED13_FRESH_MATERIALIZATION_RELATION_DERIVED_ECDLP 0

typedef struct {{
  uint64_t transfer_index;
  uint64_t backfill_row_id_u64;
  uint64_t backfill_row_check_hash_u64;
  uint64_t direct_ops_over_rho_scaled;
  uint64_t direct_public_key_verified;
  uint64_t shared_product_public_key_verified;
  uint64_t full_gate_replay_result_code;
  uint64_t materialized;
  uint64_t needs_fresh_direct_verification;
  uint64_t below_rho_label;
}} selected13_fresh_materialization_target_t;

typedef struct {{
  uint64_t transfer_index;
  uint64_t family_mask;
  uint64_t source_tier_code;
  uint64_t source_secret_count;
  uint64_t candidate_form_count;
  uint64_t replay_result_code;
  uint64_t fresh_source_solve_required;
  uint64_t lane_hash_u64;
}} selected13_fresh_materialization_lane_t;

static const selected13_fresh_materialization_target_t SELECTED13_FRESH_MATERIALIZATION_TARGETS[] = {{
{chr(10).join(target_rows)}
}};

static const selected13_fresh_materialization_lane_t SELECTED13_FRESH_MATERIALIZATION_LANES[] = {{
{chr(10).join(lane_rows)}
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
  uint64_t target_count =
      sizeof(SELECTED13_FRESH_MATERIALIZATION_TARGETS) / sizeof(SELECTED13_FRESH_MATERIALIZATION_TARGETS[0]);
  uint64_t lane_count =
      sizeof(SELECTED13_FRESH_MATERIALIZATION_LANES) / sizeof(SELECTED13_FRESH_MATERIALIZATION_LANES[0]);
  uint64_t materialized_count = 0;
  uint64_t needs_fresh_count = 0;
  uint64_t below_rho_unverified_count = 0;
  uint64_t direct_verified_count = 0;

  if (target_count != SELECTED13_FRESH_MATERIALIZATION_TARGET_COUNT) failure_count++;
  if (lane_count != SELECTED13_FRESH_MATERIALIZATION_FAMILY_LANE_COUNT) failure_count++;
  if (SELECTED13_FRESH_MATERIALIZATION_RELATION_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_FRESH_MATERIALIZATION_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;
  if (target_count == 0ULL) failure_count++;

  for (size_t i = 0; i < target_count; i++) {{
    const selected13_fresh_materialization_target_t *target = &SELECTED13_FRESH_MATERIALIZATION_TARGETS[i];
    materialized_count += target->materialized;
    needs_fresh_count += target->needs_fresh_direct_verification;
    direct_verified_count += target->direct_public_key_verified;
    if (target->below_rho_label != 0ULL && target->direct_public_key_verified == 0ULL) {{
      below_rho_unverified_count++;
    }}
    if (target->backfill_row_id_u64 == 0ULL) failure_count++;
    if (target->backfill_row_check_hash_u64 == 0ULL) failure_count++;
    if (target->full_gate_replay_result_code != 2ULL) failure_count++;
  }}

  for (size_t i = 0; i < lane_count; i++) {{
    const selected13_fresh_materialization_lane_t *lane = &SELECTED13_FRESH_MATERIALIZATION_LANES[i];
    if (lane->family_mask == 0ULL) failure_count++;
    if (lane->candidate_form_count == 0ULL) failure_count++;
    if (lane->replay_result_code != 1ULL) failure_count++;
  }}

  if (materialized_count != target_count) failure_count++;
  if (direct_verified_count != SELECTED13_FRESH_MATERIALIZATION_DIRECT_VERIFIED_COUNT) failure_count++;

  printf("selected13_fresh_materialization_preflight targets=%llu lanes=%llu materialized=%llu needs_fresh_direct=%llu below_rho_unverified=%llu failures=%llu\\n",
         (unsigned long long)target_count,
         (unsigned long long)lane_count,
         (unsigned long long)materialized_count,
         (unsigned long long)needs_fresh_count,
         (unsigned long long)below_rho_unverified_count,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    with tempfile.TemporaryDirectory(prefix="selected13_materialization_preflight_") as tmp:
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
    targets = parse_targets(args.targets)
    packet_targets = selected_packet_targets(packet, targets)
    workorder_index = workorder_rows_by_key(workorder)
    if args.support_scouts:
        scout_paths = parse_paths(args.support_scouts)
    else:
        scout_paths = collect_default_scouts(packet_targets, workorder_index, Path(args.support_state_dir))
    scout_index, missing_scouts, scout_summaries = scout_rows_by_key(scout_paths)

    failures = validate_sources(packet, workorder)
    if len(packet_targets) != len(targets):
        failures.append({"code": "packet_target_count_mismatch", "requested": targets, "observed": len(packet_targets)})
    if missing_scouts:
        failures.append({"code": "support_scout_sources_missing", "paths": missing_scouts})

    worker_cases: list[dict[str, Any]] = []
    for target in packet_targets:
        key = target_key(target)
        case, case_failures = build_worker_case(target, workorder_index.get(key, []), scout_index.get(key, []))
        worker_cases.append(case)
        failures.extend(case_failures)

    direct_verified_count = sum(1 for case in worker_cases if case.get("direct_public_key_verified"))
    below_rho_unverified_count = sum(
        1 for case in worker_cases if case.get("below_rho_label") and not case.get("direct_public_key_verified")
    )
    lane_source_tiers = Counter(
        str(lane.get("source_tier"))
        for case in worker_cases
        for lane in case.get("family_lanes") or []
    )
    result = {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status(failures, worker_cases),
        "parameters": {
            "packet": str(packet_path),
            "support_scouts": [str(path) for path in scout_paths],
            "support_state_dir": str(args.support_state_dir),
            "targets": targets,
            "workorder": str(workorder_path),
        },
        "artifacts": {
            "fresh_emission_packet": str(packet_path),
            "selected13_workorder": str(workorder_path),
            "support_scout_summaries": scout_summaries,
        },
        "summary": {
            "accepted_relation_export_count": 0,
            "below_rho_unverified_count": below_rho_unverified_count,
            "direct_public_key_verified_count": direct_verified_count,
            "family_lane_count": sum(as_int(case.get("family_lane_count")) for case in worker_cases),
            "failure_count": len(failures),
            "lane_source_tier_counts": dict(sorted(lane_source_tiers.items())),
            "materialized_target_count": sum(1 for case in worker_cases if case.get("materialized")),
            "needs_fresh_direct_verification_count": sum(1 for case in worker_cases if case.get("needs_fresh_direct_verification")),
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "relation_export_ready_count": 0,
            "target_count": len(worker_cases),
            "verified": not failures,
            "worker_interpretation": (
                "Packet/workorder/support rows are hash-bound to a fresh worker case. "
                "Observed support-scout direct labels remain unverified, so the next "
                "step is a fresh FFE/summation-polynomial direct verifier, not a "
                "relation-derived ECDLP claim."
            ),
        },
        "worker_cases": worker_cases,
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "claim_requires_relation_derived_ecdlp": True,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "unverified_below_rho_labels_are_diagnostic_only": True,
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", default=str(DEFAULT_PACKET), help="Fresh-emission packet JSON")
    parser.add_argument("--workorder", default=str(DEFAULT_WORKORDER), help="Selected13 workorder JSON")
    parser.add_argument("--support-state-dir", default=str(DEFAULT_SUPPORT_STATE_DIR), help="Host support-scout state dir")
    parser.add_argument("--support-scouts", default="", help="Optional comma-separated support-scout JSON paths")
    parser.add_argument("--targets", default="9981", help="Comma-separated backfill transfer targets")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output adapter manifest")
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT), help="Output C preflight header")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    out_path = Path(args.out)
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["worker_cases"]))
    payload["artifacts"]["c_header"] = str(header_path)
    payload["native_preflight"] = run_native_preflight(header_path)
    if not payload["native_preflight"].get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": payload["native_preflight"]})
        payload["claim_status"] = claim_status(payload["failures"], payload["worker_cases"])
        payload["summary"]["failure_count"] = len(payload["failures"])
        payload["summary"]["verified"] = False
    write_json(out_path, payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": str(out_path), "summary": payload["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
