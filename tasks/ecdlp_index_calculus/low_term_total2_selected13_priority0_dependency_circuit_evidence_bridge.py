#!/usr/bin/env python3
"""Bridge public dependency-circuit evidence into the selected13 kernel ABI.

The priority-0 selected13 validator now requires group/lane evidence hashes and
target evidence digests before any external result can claim
`relation_derived_ecdlp`.  The live campaign also has a stronger public
dependency-circuit artifact: a relation-level circuit over 22050.cf1@11731 that
verifies against the public key and beats rho on the measured cost model.

This bridge compares that public circuit with the seven selected13 residual
term groups.  It emits a native-checkable work order that marks which groups are
already covered by public circuit evidence and which still need fresh
FFE/summation-polynomial synthesis.  It does not emit a validator-accepted
kernel result, solve ECDLP, or claim a Pollard-rho speedup.
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


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_dependency_circuit_evidence_bridge.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_KERNEL_CONTRACT = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_contract_10376_probe.json"
)
DEFAULT_RESULT_GATE = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_result_gate_10376_probe.json"
)
DEFAULT_VALIDATOR = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_result_validator_10376_probe.json"
)
DEFAULT_DEPENDENCY_CIRCUIT = (
    Path("/Volumes/Volume/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_dependency_circuit_probe.json")
)
DEFAULT_SLICE_QUADRATIC = (
    Path(
        "/Volumes/Volume/autolab/ecdlp_index_calculus_state/"
        "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_"
        "static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_slice_quadratic_root_probe.json"
    )
)
DEFAULT_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_dependency_circuit_evidence_bridge_10376_probe.json"
)
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_dependency_circuit_evidence_bridge_10376_probe.h"
)

EXPECTED_TRANSFER = 10376
EXPECTED_RESIDUAL_GROUP_COUNT = 7
EXPECTED_COVERED_GROUP_COUNT = 3
EXPECTED_UNCOVERED_GROUP_COUNT = 4
EXPECTED_CIRCUIT_TERM_COUNT = 4
EXPECTED_CIRCUIT_EVENT_COUNT = 4
EXPECTED_TARGET_ROW_COUNT = 2
EXPECTED_SLICE_ROW_MATCH_COUNT = 1


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


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_hash_u64(material: Any) -> int:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def pick_dependency_circuit(source: dict[str, Any], target: str) -> dict[str, Any]:
    candidates = [
        row
        for row in source.get("results") or []
        if isinstance(row, dict)
        and str(row.get("target")) == target
        and row.get("public_relation_key_verified") is True
        and row.get("public_relation_dependency_beats_rho") is True
    ]
    if not candidates:
        return {}
    return sorted(
        candidates,
        key=lambda row: (
            as_float(row.get("public_relation_dependency_ops_over_rho"), 10**18) or 10**18,
            as_int(row.get("public_relation_minimal_relation_count"), 10**18),
            as_int(row.get("public_relation_minimal_leaf_count"), 10**18),
        ),
    )[0]


def slice_rows_for_target(slice_source: dict[str, Any], row_keys: list[str]) -> list[dict[str, Any]]:
    wanted = set(str(row) for row in row_keys)
    matches = []
    for row in (slice_source.get("summary") or {}).get("best_preserving_slice_quadratic_surfaces") or []:
        if str(row.get("row_key")) in wanted:
            candidate = row.get("best_preserving_candidate") or {}
            matches.append(
                {
                    "axis": candidate.get("axis"),
                    "factor_count": len(candidate.get("factors") or []),
                    "factors": candidate.get("factors") or [],
                    "full_remainder_ffe_ops_over_rho": as_float(candidate.get("full_remainder_ffe_ops_over_rho")),
                    "preserves_selected_root_pairs": bool(candidate.get("preserves_selected_root_pairs")),
                    "row_key": row.get("row_key"),
                    "slice_quadratic_all_hit_beats_rho": bool(candidate.get("slice_quadratic_all_hit_beats_rho")),
                    "slice_quadratic_all_hit_ops_over_rho": as_float(
                        candidate.get("slice_quadratic_all_hit_ops_over_rho")
                    ),
                    "slice_quadratic_selected_hit_ops_over_rho": as_float(
                        candidate.get("slice_quadratic_selected_hit_ops_over_rho")
                    ),
                    "surface_id": row.get("surface_id"),
                }
            )
    return sorted(matches, key=lambda row: str(row.get("row_key")))


def group_bridge(group: dict[str, Any], circuit_terms: set[int], circuit_events: list[dict[str, Any]]) -> dict[str, Any]:
    term = as_int(group.get("term"), -1)
    supporting_events = [
        event
        for event in circuit_events
        if term in {as_int(value) for value in event.get("factor_support") or event.get("terms") or []}
    ]
    covered = term in circuit_terms
    bridge = {
        "covered_by_public_dependency_circuit": covered,
        "completion_token_ready": False,
        "dependency_event_count": len(supporting_events),
        "dependency_event_indices": [as_int(event.get("relation_index"), -1) for event in supporting_events],
        "dependency_leaf_indices": sorted({as_int(event.get("leaf_index"), -1) for event in supporting_events}),
        "expected_worker_evidence_kind": (
            "plain_hybrid_ffe_residual_synthesis"
            if group.get("term_group_class") == "plain_and_hybrid_residual"
            else "hybrid_ffe_residual_synthesis"
            if group.get("term_group_class") == "hybrid_residual"
            else "guarded_hybrid_ffe_residual_synthesis"
        ),
        "fresh_ffe_synthesis_still_required": True,
        "group_index": as_int(group.get("group_index"), -1),
        "hybrid_materialization_slot_count": as_int(group.get("hybrid_materialization_slot_count")),
        "lane_bitmap": as_int(group.get("lane_bitmap")),
        "partial_evidence_hash_u64": 0,
        "residual_relation_hash_required": True,
        "source_guard_slot_count": as_int(group.get("source_guard_slot_count")),
        "synthesis_required_slot_count": as_int(group.get("synthesis_required_slot_count")),
        "term": term,
        "term_group_class": group.get("term_group_class"),
        "term_group_hash_u64": as_int(group.get("term_group_hash_u64")),
        "term_mask": as_int(group.get("term_mask")),
    }
    if covered:
        bridge["partial_evidence_hash_u64"] = stable_hash_u64(
            {
                "supporting_events": supporting_events,
                "term_group_hash_u64": group.get("term_group_hash_u64"),
                "term": term,
            }
        )
    return bridge


def lane_bridge(lane: dict[str, Any], group_bridges: list[dict[str, Any]]) -> dict[str, Any]:
    lane_bitmap = 1 << as_int(lane.get("hint_local_index"), -1)
    participating = [group for group in group_bridges if as_int(group.get("lane_bitmap")) & lane_bitmap]
    covered = [group for group in participating if group.get("covered_by_public_dependency_circuit")]
    uncovered = [group for group in participating if not group.get("covered_by_public_dependency_circuit")]
    return {
        "completion_token_ready": False,
        "covered_group_count": len(covered),
        "covered_terms": [as_int(group.get("term")) for group in covered],
        "fresh_lane_evidence_still_required": True,
        "hint_local_index": as_int(lane.get("hint_local_index"), -1),
        "kernel_lane_hash_u64": as_int(lane.get("kernel_lane_hash_u64")),
        "lane_class": lane.get("lane_class"),
        "lane_evidence_hash_ready": False,
        "materialization_required": bool(lane.get("materialization_required")),
        "source_public_key_verified": bool(lane.get("source_public_key_verified")),
        "source_voter_hash_u64": as_int(lane.get("source_voter_hash_u64")),
        "source_voter_secret": as_int(lane.get("source_voter_secret"), -1),
        "synthesis_required_slot_count": as_int(lane.get("synthesis_required_slot_count")),
        "term_slot_count": as_int(lane.get("term_slot_count")),
        "uncovered_group_count": len(uncovered),
        "uncovered_terms": [as_int(group.get("term")) for group in uncovered],
    }


def build_bridge(
    kernel_contract: dict[str, Any],
    result_gate: dict[str, Any],
    validator: dict[str, Any],
    dependency_source: dict[str, Any],
    slice_source: dict[str, Any],
    dependency_path: Path,
    slice_path: Path,
) -> dict[str, Any]:
    target_slot = kernel_contract.get("target_slot") or {}
    target = str(target_slot.get("target") or "")
    circuit = pick_dependency_circuit(dependency_source, target)
    circuit_terms = {as_int(term) for term in circuit.get("dependency_term_support") or []}
    residual_groups = kernel_contract.get("fused_term_groups") or []
    group_bridges = [group_bridge(group, circuit_terms, circuit.get("dependency_events") or []) for group in residual_groups]
    covered_groups = [group for group in group_bridges if group.get("covered_by_public_dependency_circuit")]
    uncovered_groups = [group for group in group_bridges if not group.get("covered_by_public_dependency_circuit")]
    row_keys = [str(row) for row in target_slot.get("row_keys") or []]
    slice_matches = slice_rows_for_target(slice_source, row_keys)
    missing_slice_rows = sorted(set(row_keys) - {str(row.get("row_key")) for row in slice_matches})
    selected_terms = {as_int(term) for term in target_slot.get("selected_term_support") or []}
    residual_terms = {as_int(group.get("term")) for group in residual_groups}
    source_or_family_terms = sorted(selected_terms - residual_terms)
    source_or_family_circuit_terms = sorted(circuit_terms & set(source_or_family_terms))
    lane_bridges = [lane_bridge(lane, group_bridges) for lane in kernel_contract.get("kernel_lanes") or []]
    summary = {
        "accepted_relation_export_count": 0,
        "circuit_event_count": len(circuit.get("dependency_events") or []),
        "circuit_public_relation_key_verified": bool(circuit.get("public_relation_key_verified")),
        "circuit_relation_ops_over_rho": as_float(circuit.get("public_relation_dependency_ops_over_rho")),
        "circuit_relation_rho_win": bool(circuit.get("public_relation_dependency_beats_rho")),
        "circuit_term_count": len(circuit_terms),
        "covered_residual_group_count": len(covered_groups),
        "evidence_bridge_complete": False,
        "failure_count": 0,
        "fresh_ffe_group_count_remaining": len(uncovered_groups),
        "missing_slice_row_count": len(missing_slice_rows),
        "native_preflight_verified": False,
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "residual_group_count": len(residual_groups),
        "slice_quadratic_row_match_count": len(slice_matches),
        "source_or_family_circuit_term_count": len(source_or_family_circuit_terms),
        "uncovered_residual_group_count": len(uncovered_groups),
        "verified": True,
        "worker_interpretation": (
            "The public dependency circuit covers a strict subset of selected13 residual groups. "
            "It is useful evidence for the future worker, but four residual groups and one target "
            "row slice still need fresh FFE/summation evidence before the validator can accept."
        ),
    }
    failures = []
    if not circuit:
        failures.append({"code": "no_public_dependency_circuit_for_target", "target": target})
    if len(residual_groups) != EXPECTED_RESIDUAL_GROUP_COUNT:
        failures.append({"code": "residual_group_count_unexpected", "observed": len(residual_groups)})
    if len(covered_groups) != EXPECTED_COVERED_GROUP_COUNT:
        failures.append({"code": "covered_group_count_unexpected", "observed": len(covered_groups)})
    if len(uncovered_groups) != EXPECTED_UNCOVERED_GROUP_COUNT:
        failures.append({"code": "uncovered_group_count_unexpected", "observed": len(uncovered_groups)})
    if len(circuit_terms) != EXPECTED_CIRCUIT_TERM_COUNT:
        failures.append({"code": "circuit_term_count_unexpected", "observed": len(circuit_terms)})
    if len(circuit.get("dependency_events") or []) != EXPECTED_CIRCUIT_EVENT_COUNT:
        failures.append({"code": "circuit_event_count_unexpected", "observed": len(circuit.get("dependency_events") or [])})
    if len(row_keys) != EXPECTED_TARGET_ROW_COUNT:
        failures.append({"code": "target_row_count_unexpected", "observed": len(row_keys)})
    if len(slice_matches) != EXPECTED_SLICE_ROW_MATCH_COUNT:
        failures.append({"code": "slice_row_match_count_unexpected", "observed": len(slice_matches)})
    if (validator.get("summary") or {}).get("completion_evidence_bound") is not False:
        failures.append({"code": "validator_not_awaiting_evidence_bound_result", "summary": validator.get("summary")})
    summary["failure_count"] = len(failures)
    summary["verified"] = not failures
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_DEPENDENCY_CIRCUIT_EVIDENCE_BRIDGE_READY"
            if not failures
            else "SELECTED13_PRIORITY0_DEPENDENCY_CIRCUIT_EVIDENCE_BRIDGE_FAILED"
        ),
        "parameters": {
            "dependency_circuit": str(dependency_path),
            "kernel_contract": str(DEFAULT_KERNEL_CONTRACT),
            "result_gate": str(DEFAULT_RESULT_GATE),
            "slice_quadratic": str(slice_path),
            "validator": str(DEFAULT_VALIDATOR),
        },
        "packet_hash_u64": stable_hash_u64(
            {
                "circuit": circuit,
                "covered_terms": [group.get("term") for group in covered_groups],
                "target_slot": target_slot,
                "uncovered_terms": [group.get("term") for group in uncovered_groups],
            }
        ),
        "target_slot": target_slot,
        "source_dependency_circuit": {
            "dependency_events": circuit.get("dependency_events") or [],
            "dependency_factor_support": circuit.get("dependency_factor_support") or [],
            "dependency_term_shape_counts": circuit.get("dependency_term_shape_counts") or {},
            "dependency_term_support": sorted(circuit_terms),
            "generic_rho_steps": as_int(circuit.get("generic_rho_steps")),
            "public_relation_dependency_ops": as_int(circuit.get("public_relation_dependency_ops")),
            "public_relation_dependency_ops_over_rho": as_float(
                circuit.get("public_relation_dependency_ops_over_rho")
            ),
            "public_relation_key_verified": bool(circuit.get("public_relation_key_verified")),
            "target": circuit.get("target"),
        },
        "source_or_family_circuit_terms": source_or_family_circuit_terms,
        "group_evidence_bridge": group_bridges,
        "lane_evidence_bridge": lane_bridges,
        "slice_quadratic_bridge": {
            "matched_rows": slice_matches,
            "missing_rows": missing_slice_rows,
            "source_summary": slice_source.get("summary", {}),
        },
        "validator_bridge_status": {
            "completion_evidence_bound": bool((validator.get("summary") or {}).get("completion_evidence_bound")),
            "result_validator_claim_status": validator.get("claim_status"),
            "result_validator_packet_hash_u64": validator.get("packet_hash_u64"),
            "target_export_token": as_int((result_gate.get("target_export_completion_gate") or {}).get("completion_token_hash_u64")),
        },
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "dependency_circuit_is_partial_evidence_only": True,
            "external_kernel_result_emitted": False,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "summation_polynomial_evaluated_by_bridge": False,
        },
        "failures": failures,
        "summary": summary,
    }


def render_c_header(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    target = payload.get("target_slot") or {}
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_DEPENDENCY_CIRCUIT_EVIDENCE_BRIDGE_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_DEPENDENCY_CIRCUIT_EVIDENCE_BRIDGE_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_TRANSFER {as_int(target.get('transfer_index'))}ULL
#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_RESIDUAL_GROUP_COUNT {as_int(summary.get('residual_group_count'))}ULL
#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_COVERED_GROUP_COUNT {as_int(summary.get('covered_residual_group_count'))}ULL
#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_UNCOVERED_GROUP_COUNT {as_int(summary.get('uncovered_residual_group_count'))}ULL
#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_CIRCUIT_TERM_COUNT {as_int(summary.get('circuit_term_count'))}ULL
#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_CIRCUIT_EVENT_COUNT {as_int(summary.get('circuit_event_count'))}ULL
#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_PUBLIC_RELATION_VERIFIED {1 if summary.get('circuit_public_relation_key_verified') else 0}ULL
#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_CIRCUIT_RHO_WIN {1 if summary.get('circuit_relation_rho_win') else 0}ULL
#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_TARGET_ROW_COUNT {EXPECTED_TARGET_ROW_COUNT}ULL
#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_SLICE_ROW_MATCH_COUNT {as_int(summary.get('slice_quadratic_row_match_count'))}ULL
#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_MISSING_SLICE_ROW_COUNT {as_int(summary.get('missing_slice_row_count'))}ULL
#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_ACCEPTED_EXPORT_COUNT 0ULL
#define SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_RELATION_DERIVED_ECDLP 0ULL

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

int main(void) {{
  uint64_t failure_count = 0;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_TRANSFER != {EXPECTED_TRANSFER}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_RESIDUAL_GROUP_COUNT != {EXPECTED_RESIDUAL_GROUP_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_COVERED_GROUP_COUNT != {EXPECTED_COVERED_GROUP_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_UNCOVERED_GROUP_COUNT != {EXPECTED_UNCOVERED_GROUP_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_CIRCUIT_TERM_COUNT != {EXPECTED_CIRCUIT_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_CIRCUIT_EVENT_COUNT != {EXPECTED_CIRCUIT_EVENT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_PUBLIC_RELATION_VERIFIED != 1ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_CIRCUIT_RHO_WIN != 1ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_TARGET_ROW_COUNT != {EXPECTED_TARGET_ROW_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_SLICE_ROW_MATCH_COUNT != {EXPECTED_SLICE_ROW_MATCH_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_MISSING_SLICE_ROW_COUNT != 1ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;

  printf("selected13_priority0_dependency_circuit_evidence_bridge_preflight transfer=%llu circuit_terms=%llu circuit_events=%llu covered_groups=%llu uncovered_groups=%llu slice_rows=%llu missing_slice_rows=%llu public_relation_verified=%llu accepted=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_TRANSFER,
         (unsigned long long)SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_CIRCUIT_TERM_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_CIRCUIT_EVENT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_COVERED_GROUP_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_UNCOVERED_GROUP_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_SLICE_ROW_MATCH_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_MISSING_SLICE_ROW_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_PUBLIC_RELATION_VERIFIED,
         (unsigned long long)SELECTED13_PRIORITY0_DEPENDENCY_BRIDGE_ACCEPTED_EXPORT_COUNT,
         (unsigned long long)failure_count);
  return failure_count == 0ULL ? 0 : 1;
}}
"""


def run_native_preflight(header_path: Path, compiler: str) -> dict[str, Any]:
    source = render_preflight_c(header_path.name)
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    temp_root = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path(tempfile.gettempdir())
    env = os.environ.copy()
    env["TMPDIR"] = str(temp_root)
    with tempfile.TemporaryDirectory(prefix="selected13_dependency_bridge_", dir=str(temp_root)) as tmp:
        tmp_path = Path(tmp)
        c_path = tmp_path / "preflight.c"
        exe_path = tmp_path / "preflight"
        c_path.write_text(source)
        command = [
            compiler,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-O2",
            "-I",
            str(header_path.parent),
            str(c_path),
            "-o",
            str(exe_path),
        ]
        compile_run = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
        if compile_run.returncode != 0:
            return {
                "c_source_sha256": source_hash,
                "compile_command": command,
                "compile_returncode": compile_run.returncode,
                "compile_stderr": compile_run.stderr,
                "compile_stdout": compile_run.stdout,
                "compiled": False,
                "executed": False,
                "verified": False,
            }
        native_run = subprocess.run([str(exe_path)], capture_output=True, text=True, check=False, env=env)
        return {
            "c_source_sha256": source_hash,
            "compile_command": command,
            "compile_returncode": compile_run.returncode,
            "compile_stderr": compile_run.stderr,
            "compile_stdout": compile_run.stdout,
            "compiled": True,
            "executed": True,
            "preflight_returncode": native_run.returncode,
            "preflight_stderr": native_run.stderr.strip(),
            "preflight_stdout": native_run.stdout.strip(),
            "verified": native_run.returncode == 0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-contract", default=str(DEFAULT_KERNEL_CONTRACT))
    parser.add_argument("--result-gate", default=str(DEFAULT_RESULT_GATE))
    parser.add_argument("--validator", default=str(DEFAULT_VALIDATOR))
    parser.add_argument("--dependency-circuit", default=str(DEFAULT_DEPENDENCY_CIRCUIT))
    parser.add_argument("--slice-quadratic", default=str(DEFAULT_SLICE_QUADRATIC))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kernel_contract_path = Path(args.kernel_contract)
    result_gate_path = Path(args.result_gate)
    validator_path = Path(args.validator)
    dependency_path = Path(args.dependency_circuit)
    slice_path = Path(args.slice_quadratic)
    payload = build_bridge(
        load_json(kernel_contract_path),
        load_json(result_gate_path),
        load_json(validator_path),
        load_json(dependency_path),
        load_json(slice_path),
        dependency_path,
        slice_path,
    )
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload))
    payload["artifacts"] = {"c_header": str(header_path)}
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_DEPENDENCY_CIRCUIT_EVIDENCE_BRIDGE_FAILED"
    payload["summary"]["native_preflight_verified"] = bool(native_preflight.get("verified"))
    payload["summary"]["failure_count"] = len(payload["failures"])
    payload["summary"]["verified"] = not payload["failures"]
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
