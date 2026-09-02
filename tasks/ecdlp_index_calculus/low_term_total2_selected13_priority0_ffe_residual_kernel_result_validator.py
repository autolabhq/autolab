#!/usr/bin/env python3
"""Validate external results for the selected13 priority-0 residual kernel.

The result gate assigns completion tokens to the fused FFE/summation kernel
surface.  This validator is the fail-closed acceptance layer for a future
kernel result: every group token, lane token, target export token, row id,
expected scalar, and public-key check must line up before the result can set
`relation_derived_ecdlp`.

With no --kernel-result input, the script emits a ready validator artifact that
is explicitly awaiting external kernel output.  It does not evaluate summation
polynomials, emit a target direct/rank row, solve ECDLP, or claim a Pollard-rho
speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_ffe_residual_kernel_result_validator.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_RESULT_GATE = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_result_gate_10376_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_result_validator_10376_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_ffe_residual_kernel_result_validator_10376_probe.h"
DEFAULT_TASK_DIR = Path(os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus"))

EXPECTED_TRANSFER = 10376
EXPECTED_ROW_REQUEST_ID = "selected13_matmiss_10376_0_d352d0d05610"
EXPECTED_SECRET = 5859
EXPECTED_GROUP_COUNT = 7
EXPECTED_LANE_COUNT = 3
EXPECTED_TARGET_TOKEN = 4590949340060852637

GROUP_EVIDENCE_KINDS = {
    "guarded_hybrid_residual": "guarded_hybrid_ffe_residual_synthesis",
    "hybrid_residual": "hybrid_ffe_residual_synthesis",
    "plain_and_hybrid_residual": "plain_hybrid_ffe_residual_synthesis",
}
LANE_EVIDENCE_KINDS = {
    "shared_product_verified_coeff_materialization": "shared_product_coefficient_materialization",
    "source_coeff_guard_partial_residual_synthesis": "source_guard_residual_synthesis",
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


def stable_hash_u64(material: Any) -> int:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def resolve_path(raw: Any, default_base: Path) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == DEFAULT_STATE_DIR.name:
        return WORKTREE_ROOT / path
    return default_base / path


def load_verifier_module(task_dir: Path) -> Any:
    env_main = task_dir / "environment" / "main.py"
    data_path = task_dir / "environment" / "lmfdb_curves.json"
    if not env_main.is_file():
        raise FileNotFoundError(f"verifier environment missing: {env_main}")
    if not data_path.is_file():
        raise FileNotFoundError(f"LMFDB data missing: {data_path}")
    old_app_dir = os.environ.get("APP_DIR")
    old_lmfdb_data = os.environ.get("LMFDB_DATA")
    os.environ["APP_DIR"] = str(env_main.parent)
    os.environ["LMFDB_DATA"] = str(data_path)
    try:
        spec = importlib.util.spec_from_file_location("ecdlp_verifier_main_for_selected13_result_validator", env_main)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import verifier helpers from {env_main}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if old_app_dir is None:
            os.environ.pop("APP_DIR", None)
        else:
            os.environ["APP_DIR"] = old_app_dir
        if old_lmfdb_data is None:
            os.environ.pop("LMFDB_DATA", None)
        else:
            os.environ["LMFDB_DATA"] = old_lmfdb_data


def public_key_validation(kernel_contract: dict[str, Any], target_export: dict[str, Any], task_dir: Path) -> dict[str, Any]:
    public_raw = target_export.get("public_key", target_export.get("public"))
    secret = as_int(target_export.get("derived_secret", target_export.get("secret")), -1)
    context = kernel_contract.get("target_context") or {}
    result = {
        "coords_in_field": False,
        "expected_secret": EXPECTED_SECRET,
        "public_key_parse_ok": False,
        "public_key_present": public_raw is not None,
        "public_key_verified": False,
        "secret_matches_expected": secret == EXPECTED_SECRET,
        "subgroup_check": False,
        "verified_by": "mounted_verifier_mul_point",
    }
    if public_raw is None or secret != EXPECTED_SECRET or context.get("context_status") != "verified":
        return result
    verifier = load_verifier_module(task_dir)
    p = as_int(context.get("p"))
    ainvs = [as_int(value) for value in context.get("ainvs") or []]
    try:
        base = verifier.point_from_json(context.get("base"))
        public = verifier.point_from_json(public_raw)
    except (TypeError, ValueError) as exc:
        result["public_key_parse_error"] = str(exc)
        return result
    result["public_key_parse_ok"] = True
    if public is verifier.POINT_AT_INFINITY or len(ainvs) != 5 or p <= 1:
        return result
    x, y = public
    result["coords_in_field"] = 0 <= x < p and 0 <= y < p
    result["on_curve"] = bool(verifier.is_on_curve(public, ainvs, p))
    result["subgroup_check"] = bool(
        verifier.mul_point(as_int(context.get("base_order")), public, ainvs, p) is verifier.POINT_AT_INFINITY
    )
    expected_public = verifier.mul_point(secret % as_int(context.get("base_order")), base, ainvs, p)
    result["expected_public"] = verifier.point_to_json(expected_public)
    result["observed_public"] = verifier.point_to_json(public)
    result["public_key_verified"] = bool(
        result["coords_in_field"]
        and result["on_curve"]
        and result["subgroup_check"]
        and public == expected_public
    )
    return result


def token_set(items: list[dict[str, Any]], key: str = "completion_token_hash_u64") -> set[int]:
    return {as_int(item.get(key)) for item in items}


def stable_by_token(items: list[dict[str, Any]], key: str = "completion_token_hash_u64") -> dict[int, dict[str, Any]]:
    by_token = {}
    for item in items:
        token = as_int(item.get(key))
        if token:
            by_token[token] = item
    return by_token


def stable_by_hint(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_hint = {}
    for item in items:
        hint = as_int(item.get("hint_local_index"), -1)
        if hint >= 0:
            by_hint[hint] = item
    return by_hint


def nonzero_u64(value: Any) -> bool:
    observed = as_int(value)
    return 0 < observed < (1 << 64)


def expected_bool_for_count(count: int) -> bool:
    return count > 0


def compare_int_fields(
    observed: dict[str, Any],
    expected: dict[str, Any],
    fields: list[str],
    issue_code: str,
    issue_context: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    mismatches = []
    for field in fields:
        if as_int(observed.get(field), -1) != as_int(expected.get(field), -1):
            mismatches.append(
                {
                    "field": field,
                    "expected": as_int(expected.get(field), -1),
                    "observed": observed.get(field),
                }
            )
    if mismatches:
        issue = dict(issue_context)
        issue["code"] = issue_code
        issue["mismatches"] = mismatches
        issues.append(issue)


def normalize_group_completion(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": item.get("accepted") is True,
        "completion_token_hash_u64": as_int(item.get("completion_token_hash_u64")),
        "group_evidence_hash_u64": as_int(item.get("group_evidence_hash_u64")),
        "group_index": as_int(item.get("group_index"), -1),
        "hybrid_coefficients_materialized": item.get("hybrid_coefficients_materialized") is True,
        "hybrid_materialization_slot_count": as_int(item.get("hybrid_materialization_slot_count")),
        "lane_bitmap": as_int(item.get("lane_bitmap")),
        "lane_fanout_applied": item.get("lane_fanout_applied") is True,
        "residual_relation_hash_u64": as_int(item.get("residual_relation_hash_u64")),
        "residual_synthesis_evaluated": item.get("residual_synthesis_evaluated") is True,
        "source_guard_preserved": item.get("source_guard_preserved") is True,
        "source_guard_slot_count": as_int(item.get("source_guard_slot_count")),
        "synthesis_required_slot_count": as_int(item.get("synthesis_required_slot_count")),
        "term": as_int(item.get("term"), -1),
        "term_group_hash_u64": as_int(item.get("term_group_hash_u64")),
        "term_mask": as_int(item.get("term_mask")),
        "worker_evidence_kind": str(item.get("worker_evidence_kind") or ""),
    }


def normalize_lane_completion(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": item.get("accepted") is True,
        "coefficient_materialization_evaluated": item.get("coefficient_materialization_evaluated") is True,
        "completion_token_hash_u64": as_int(item.get("completion_token_hash_u64")),
        "hint_local_index": as_int(item.get("hint_local_index"), -1),
        "kernel_lane_hash_u64": as_int(item.get("kernel_lane_hash_u64")),
        "lane_evidence_hash_u64": as_int(item.get("lane_evidence_hash_u64")),
        "residual_synthesis_evaluated": item.get("residual_synthesis_evaluated") is True,
        "source_coefficient_guard_reused": item.get("source_coefficient_guard_reused") is True,
        "source_public_key_verified": item.get("source_public_key_verified") is True,
        "source_voter_hash_u64": as_int(item.get("source_voter_hash_u64")),
        "source_voter_secret": as_int(item.get("source_voter_secret"), -1),
        "synthesis_required_slot_count": as_int(item.get("synthesis_required_slot_count")),
        "term_slot_count": as_int(item.get("term_slot_count")),
        "worker_evidence_kind": str(item.get("worker_evidence_kind") or ""),
    }


def validate_completion_evidence(
    result_gate: dict[str, Any],
    kernel_contract: dict[str, Any],
    group_results: list[dict[str, Any]],
    lane_results: list[dict[str, Any]],
    target_export: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    group_gates_by_token = stable_by_token(result_gate.get("group_completion_gates") or [])
    lane_gates_by_token = stable_by_token(result_gate.get("lane_completion_gates") or [])
    lanes_by_hint = stable_by_hint(kernel_contract.get("kernel_lanes") or [])

    normalized_groups = []
    group_issue_count = 0
    for item in sorted(group_results, key=lambda raw: (as_int(raw.get("group_index"), -1), as_int(raw.get("completion_token_hash_u64")))):
        token = as_int(item.get("completion_token_hash_u64"))
        gate = group_gates_by_token.get(token)
        if gate is None:
            continue
        norm = normalize_group_completion(item)
        normalized_groups.append(norm)
        issues: list[dict[str, Any]] = []
        context = {
            "completion_token_hash_u64": token,
            "group_index": norm["group_index"],
            "term": norm["term"],
        }
        group_expected = {
            "group_index": gate.get("group_index"),
            "term": gate.get("term"),
            "term_mask": gate.get("term_mask"),
            "term_group_hash_u64": gate.get("term_group_hash_u64"),
            "lane_bitmap": gate.get("expected_lane_bitmap"),
            "synthesis_required_slot_count": gate.get("expected_synthesis_required_slot_count"),
            "hybrid_materialization_slot_count": gate.get("expected_hybrid_materialization_slot_count"),
            "source_guard_slot_count": gate.get("expected_source_guard_slot_count"),
        }
        compare_int_fields(
            item,
            group_expected,
            [
                "group_index",
                "term",
                "term_mask",
                "term_group_hash_u64",
                "lane_bitmap",
                "synthesis_required_slot_count",
                "hybrid_materialization_slot_count",
                "source_guard_slot_count",
            ],
            "group_completion_contract_field_mismatch",
            context,
            issues,
        )
        expected_kind = GROUP_EVIDENCE_KINDS.get(str(gate.get("term_group_class")), "")
        if item.get("worker_evidence_kind") != expected_kind:
            issues.append({**context, "code": "group_completion_evidence_kind_mismatch", "expected": expected_kind, "observed": item.get("worker_evidence_kind")})
        if not nonzero_u64(item.get("group_evidence_hash_u64")):
            issues.append({**context, "code": "group_completion_evidence_hash_missing"})
        if not nonzero_u64(item.get("residual_relation_hash_u64")):
            issues.append({**context, "code": "group_completion_residual_relation_hash_missing"})
        if item.get("residual_synthesis_evaluated") is not True:
            issues.append({**context, "code": "group_completion_residual_synthesis_not_evaluated"})
        if item.get("lane_fanout_applied") is not True:
            issues.append({**context, "code": "group_completion_lane_fanout_not_applied"})
        if expected_bool_for_count(as_int(gate.get("hybrid_materialization_slot_count"))) and item.get("hybrid_coefficients_materialized") is not True:
            issues.append({**context, "code": "group_completion_hybrid_coefficients_not_materialized"})
        if expected_bool_for_count(as_int(gate.get("source_guard_slot_count"))) and item.get("source_guard_preserved") is not True:
            issues.append({**context, "code": "group_completion_source_guard_not_preserved"})
        if issues:
            group_issue_count += len(issues)
            failures.extend(issues)

    normalized_lanes = []
    lane_issue_count = 0
    for item in sorted(lane_results, key=lambda raw: (as_int(raw.get("hint_local_index"), -1), as_int(raw.get("completion_token_hash_u64")))):
        token = as_int(item.get("completion_token_hash_u64"))
        gate = lane_gates_by_token.get(token)
        if gate is None:
            continue
        hint = as_int(gate.get("hint_local_index"), -1)
        lane_contract = lanes_by_hint.get(hint, {})
        norm = normalize_lane_completion(item)
        normalized_lanes.append(norm)
        issues = []
        context = {"completion_token_hash_u64": token, "hint_local_index": hint}
        compare_int_fields(
            item,
            {
                "hint_local_index": gate.get("hint_local_index"),
                "kernel_lane_hash_u64": gate.get("kernel_lane_hash_u64"),
                "source_voter_hash_u64": gate.get("source_voter_hash_u64"),
                "synthesis_required_slot_count": gate.get("expected_synthesis_required_slot_count"),
                "term_slot_count": gate.get("expected_term_slot_count"),
            },
            [
                "hint_local_index",
                "kernel_lane_hash_u64",
                "source_voter_hash_u64",
                "synthesis_required_slot_count",
                "term_slot_count",
            ],
            "lane_completion_contract_field_mismatch",
            context,
            issues,
        )
        expected_kind = LANE_EVIDENCE_KINDS.get(str(lane_contract.get("lane_class")), "")
        if item.get("worker_evidence_kind") != expected_kind:
            issues.append({**context, "code": "lane_completion_evidence_kind_mismatch", "expected": expected_kind, "observed": item.get("worker_evidence_kind")})
        if not nonzero_u64(item.get("lane_evidence_hash_u64")):
            issues.append({**context, "code": "lane_completion_evidence_hash_missing"})
        if item.get("source_public_key_verified") is not True:
            issues.append({**context, "code": "lane_completion_source_public_key_not_verified"})
        if as_int(item.get("source_voter_secret"), -1) != EXPECTED_SECRET:
            issues.append({**context, "code": "lane_completion_source_secret_mismatch", "observed": item.get("source_voter_secret")})
        if item.get("residual_synthesis_evaluated") is not True:
            issues.append({**context, "code": "lane_completion_residual_synthesis_not_evaluated"})
        if bool(lane_contract.get("materialization_required")):
            if item.get("coefficient_materialization_evaluated") is not True:
                issues.append({**context, "code": "lane_completion_coefficient_materialization_not_evaluated"})
        elif item.get("source_coefficient_guard_reused") is not True:
            issues.append({**context, "code": "lane_completion_source_guard_not_reused"})
        if issues:
            lane_issue_count += len(issues)
            failures.extend(issues)

    group_digest = stable_hash_u64(normalized_groups)
    lane_digest = stable_hash_u64(normalized_lanes)
    expected_kernel_packet = as_int(
        (result_gate.get("source_kernel_contract") or {}).get("packet_hash_u64"),
        as_int(kernel_contract.get("packet_hash_u64")),
    )
    expected_target_gate_packet = as_int((result_gate.get("source_target_gate") or {}).get("packet_hash_u64"))
    expected_result_gate_packet = as_int(result_gate.get("packet_hash_u64"))
    target_issues = []
    if as_int(target_export.get("kernel_contract_packet_hash_u64")) != expected_kernel_packet:
        target_issues.append(
            {
                "code": "target_export_kernel_contract_packet_mismatch",
                "expected": expected_kernel_packet,
                "observed": target_export.get("kernel_contract_packet_hash_u64"),
            }
        )
    if as_int(target_export.get("target_gate_packet_hash_u64")) != expected_target_gate_packet:
        target_issues.append(
            {
                "code": "target_export_target_gate_packet_mismatch",
                "expected": expected_target_gate_packet,
                "observed": target_export.get("target_gate_packet_hash_u64"),
            }
        )
    if as_int(target_export.get("result_gate_packet_hash_u64")) != expected_result_gate_packet:
        target_issues.append(
            {
                "code": "target_export_result_gate_packet_mismatch",
                "expected": expected_result_gate_packet,
                "observed": target_export.get("result_gate_packet_hash_u64"),
            }
        )
    if as_int(target_export.get("group_evidence_digest_u64")) != group_digest:
        target_issues.append(
            {
                "code": "target_export_group_evidence_digest_mismatch",
                "expected": group_digest,
                "observed": target_export.get("group_evidence_digest_u64"),
            }
        )
    if as_int(target_export.get("lane_evidence_digest_u64")) != lane_digest:
        target_issues.append(
            {
                "code": "target_export_lane_evidence_digest_mismatch",
                "expected": lane_digest,
                "observed": target_export.get("lane_evidence_digest_u64"),
            }
        )
    if target_export.get("worker_evidence_kind") != "fresh_ffe_direct_rank_export":
        target_issues.append(
            {
                "code": "target_export_evidence_kind_mismatch",
                "expected": "fresh_ffe_direct_rank_export",
                "observed": target_export.get("worker_evidence_kind"),
            }
        )
    if target_export.get("fresh_ffe_or_summation_polynomial_evaluated") is not True:
        target_issues.append({"code": "target_export_fresh_ffe_or_summation_not_evaluated"})
    if target_export.get("target_direct_rank_export_evaluated") is not True:
        target_issues.append({"code": "target_export_direct_rank_not_evaluated"})
    if not nonzero_u64(target_export.get("target_direct_rank_export_hash_u64")):
        target_issues.append({"code": "target_export_direct_rank_hash_missing"})
    if target_export.get("source_replay_only") is True:
        target_issues.append({"code": "target_export_source_replay_flag_set"})
    failures.extend(target_issues)
    evidence_bound = not failures
    return (
        {
            "evidence_bound": evidence_bound,
            "expected_kernel_contract_packet_hash_u64": expected_kernel_packet,
            "expected_result_gate_packet_hash_u64": expected_result_gate_packet,
            "expected_target_gate_packet_hash_u64": expected_target_gate_packet,
            "group_evidence_digest_u64": group_digest,
            "group_evidence_issue_count": group_issue_count,
            "lane_evidence_digest_u64": lane_digest,
            "lane_evidence_issue_count": lane_issue_count,
            "normalized_group_evidence_count": len(normalized_groups),
            "normalized_lane_evidence_count": len(normalized_lanes),
            "target_evidence_issue_count": len(target_issues),
        },
        failures,
    )


def validate_sources(result_gate: dict[str, Any], kernel_contract: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if result_gate.get("claim_status") != "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_GATE_READY":
        failures.append({"code": "result_gate_not_ready", "claim_status": result_gate.get("claim_status")})
    if result_gate.get("failures"):
        failures.append({"code": "result_gate_has_failures", "failures": result_gate.get("failures")})
    if (result_gate.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "result_gate_summary_not_verified", "summary": result_gate.get("summary")})
    if (result_gate.get("native_preflight") or {}).get("verified") is not True:
        failures.append({"code": "result_gate_native_preflight_not_verified"})
    if kernel_contract.get("claim_status") != "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_CONTRACT_READY":
        failures.append({"code": "kernel_contract_not_ready", "claim_status": kernel_contract.get("claim_status")})
    target = result_gate.get("target_slot") or {}
    if as_int(target.get("transfer_index"), -1) != EXPECTED_TRANSFER:
        failures.append({"code": "target_transfer_unexpected", "observed": target.get("transfer_index")})
    if target.get("row_request_id") != EXPECTED_ROW_REQUEST_ID:
        failures.append({"code": "target_row_request_unexpected", "observed": target.get("row_request_id")})
    target_export_gate = result_gate.get("target_export_completion_gate") or {}
    if as_int(target_export_gate.get("completion_token_hash_u64")) != EXPECTED_TARGET_TOKEN:
        failures.append({"code": "target_export_token_unexpected", "target_export_completion_gate": target_export_gate})
    return failures


def validate_kernel_result(
    result_gate: dict[str, Any],
    kernel_contract: dict[str, Any],
    kernel_result: dict[str, Any] | None,
    task_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures = validate_sources(result_gate, kernel_contract)
    group_gates = result_gate.get("group_completion_gates") or []
    lane_gates = result_gate.get("lane_completion_gates") or []
    expected_group_tokens = token_set(group_gates)
    expected_lane_tokens = token_set(lane_gates)
    expected_target_token = as_int((result_gate.get("target_export_completion_gate") or {}).get("completion_token_hash_u64"))
    if kernel_result is None:
        return (
            {
                "accepted_group_completion_count": 0,
                "accepted_lane_completion_count": 0,
                "all_group_tokens_bound": False,
                "all_lane_tokens_bound": False,
                "kernel_result_present": False,
                "public_key_validation": {"public_key_present": False, "public_key_verified": False},
                "relation_derived_ecdlp": False,
                "result_status": "awaiting_kernel_output",
                "target_export_accepted": False,
            },
            failures,
        )

    group_results = [item for item in kernel_result.get("group_completions") or [] if isinstance(item, dict)]
    lane_results = [item for item in kernel_result.get("lane_completions") or [] if isinstance(item, dict)]
    target_export = kernel_result.get("target_export") or {}
    group_tokens = token_set(group_results)
    lane_tokens = token_set(lane_results)
    missing_group_tokens = sorted(expected_group_tokens - group_tokens)
    extra_group_tokens = sorted(group_tokens - expected_group_tokens)
    missing_lane_tokens = sorted(expected_lane_tokens - lane_tokens)
    extra_lane_tokens = sorted(lane_tokens - expected_lane_tokens)

    if kernel_result.get("row_request_id") != EXPECTED_ROW_REQUEST_ID:
        failures.append({"code": "kernel_result_row_request_mismatch", "observed": kernel_result.get("row_request_id")})
    if missing_group_tokens or extra_group_tokens:
        failures.append(
            {
                "code": "group_completion_token_set_mismatch",
                "missing": missing_group_tokens,
                "extra": extra_group_tokens,
            }
        )
    if missing_lane_tokens or extra_lane_tokens:
        failures.append(
            {
                "code": "lane_completion_token_set_mismatch",
                "missing": missing_lane_tokens,
                "extra": extra_lane_tokens,
            }
        )
    accepted_group_count = sum(1 for item in group_results if item.get("accepted") is True)
    accepted_lane_count = sum(1 for item in lane_results if item.get("accepted") is True)
    if accepted_group_count != EXPECTED_GROUP_COUNT:
        failures.append({"code": "accepted_group_count_unexpected", "observed": accepted_group_count})
    if accepted_lane_count != EXPECTED_LANE_COUNT:
        failures.append({"code": "accepted_lane_count_unexpected", "observed": accepted_lane_count})
    if as_int(target_export.get("completion_token_hash_u64")) != expected_target_token:
        failures.append({"code": "target_export_token_mismatch", "target_export": target_export})
    public_validation = public_key_validation(kernel_contract, target_export, task_dir)
    if public_validation.get("public_key_verified") is not True:
        failures.append({"code": "target_public_key_not_verified", "public_key_validation": public_validation})
    if target_export.get("accepted") is not True:
        failures.append({"code": "target_export_not_accepted", "target_export": target_export})
    if target_export.get("relation_derived_ecdlp") is not True:
        failures.append({"code": "target_relation_not_derived", "target_export": target_export})
    if as_int(target_export.get("derived_secret")) != EXPECTED_SECRET:
        failures.append({"code": "target_export_secret_mismatch", "target_export": target_export})
    completion_evidence, evidence_failures = validate_completion_evidence(
        result_gate,
        kernel_contract,
        group_results,
        lane_results,
        target_export,
    )
    failures.extend(evidence_failures)

    relation_derived = (
        not missing_group_tokens
        and not extra_group_tokens
        and not missing_lane_tokens
        and not extra_lane_tokens
        and accepted_group_count == EXPECTED_GROUP_COUNT
        and accepted_lane_count == EXPECTED_LANE_COUNT
        and as_int(target_export.get("completion_token_hash_u64")) == expected_target_token
        and target_export.get("accepted") is True
        and target_export.get("relation_derived_ecdlp") is True
        and public_validation.get("public_key_verified") is True
        and completion_evidence.get("evidence_bound") is True
    )
    return (
        {
            "accepted_group_completion_count": accepted_group_count,
            "accepted_lane_completion_count": accepted_lane_count,
            "all_group_tokens_bound": not missing_group_tokens and not extra_group_tokens,
            "all_lane_tokens_bound": not missing_lane_tokens and not extra_lane_tokens,
            "kernel_result_present": True,
            "missing_group_completion_tokens": missing_group_tokens,
            "missing_lane_completion_tokens": missing_lane_tokens,
            "completion_evidence": completion_evidence,
            "public_key_validation": public_validation,
            "relation_derived_ecdlp": relation_derived,
            "result_status": "accepted" if relation_derived and not failures else "rejected",
            "target_export_accepted": target_export.get("accepted") is True,
        },
        failures,
    )


def expected_kernel_result_template(result_gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_request_id": EXPECTED_ROW_REQUEST_ID,
        "group_completions": [
            {
                "accepted": True,
                "completion_token_hash_u64": as_int(item.get("completion_token_hash_u64")),
                "group_evidence_hash_u64": "required_nonzero_u64",
                "group_index": as_int(item.get("group_index")),
                "hybrid_coefficients_materialized": as_int(item.get("expected_hybrid_materialization_slot_count")) > 0,
                "hybrid_materialization_slot_count": as_int(item.get("expected_hybrid_materialization_slot_count")),
                "lane_bitmap": as_int(item.get("expected_lane_bitmap")),
                "lane_fanout_applied": True,
                "residual_relation_hash_u64": "required_nonzero_u64",
                "residual_synthesis_evaluated": True,
                "source_guard_preserved": as_int(item.get("expected_source_guard_slot_count")) > 0,
                "source_guard_slot_count": as_int(item.get("expected_source_guard_slot_count")),
                "synthesis_required_slot_count": as_int(item.get("expected_synthesis_required_slot_count")),
                "term": as_int(item.get("term")),
                "term_group_hash_u64": as_int(item.get("term_group_hash_u64")),
                "term_mask": as_int(item.get("term_mask")),
                "worker_evidence_kind": GROUP_EVIDENCE_KINDS.get(str(item.get("term_group_class")), "required_group_evidence_kind"),
            }
            for item in result_gate.get("group_completion_gates") or []
        ],
        "lane_completions": [
            {
                "accepted": True,
                "coefficient_materialization_evaluated": item.get("expected_materialization_required") is True,
                "completion_token_hash_u64": as_int(item.get("completion_token_hash_u64")),
                "hint_local_index": as_int(item.get("hint_local_index")),
                "kernel_lane_hash_u64": as_int(item.get("kernel_lane_hash_u64")),
                "lane_evidence_hash_u64": "required_nonzero_u64",
                "residual_synthesis_evaluated": True,
                "source_coefficient_guard_reused": item.get("expected_materialization_required") is not True,
                "source_public_key_verified": item.get("expected_source_public_key_verified") is True,
                "source_voter_hash_u64": as_int(item.get("source_voter_hash_u64")),
                "source_voter_secret": as_int(item.get("expected_source_secret")),
                "synthesis_required_slot_count": as_int(item.get("expected_synthesis_required_slot_count")),
                "term_slot_count": as_int(item.get("expected_term_slot_count")),
                "worker_evidence_kind": LANE_EVIDENCE_KINDS.get(str(item.get("lane_class")), "required_lane_evidence_kind"),
            }
            for item in result_gate.get("lane_completion_gates") or []
        ],
        "target_export": {
            "accepted": True,
            "completion_token_hash_u64": EXPECTED_TARGET_TOKEN,
            "derived_secret": EXPECTED_SECRET,
            "fresh_ffe_or_summation_polynomial_evaluated": True,
            "group_evidence_digest_u64": "required_digest_u64",
            "kernel_contract_packet_hash_u64": as_int((result_gate.get("source_kernel_contract") or {}).get("packet_hash_u64")),
            "lane_evidence_digest_u64": "required_digest_u64",
            "public_key": ["required_x", "required_y"],
            "relation_derived_ecdlp": True,
            "result_gate_packet_hash_u64": as_int(result_gate.get("packet_hash_u64")),
            "source_replay_only": False,
            "target_direct_rank_export_evaluated": True,
            "target_direct_rank_export_hash_u64": "required_nonzero_u64",
            "target_gate_packet_hash_u64": as_int((result_gate.get("source_target_gate") or {}).get("packet_hash_u64")),
            "worker_evidence_kind": "fresh_ffe_direct_rank_export",
        },
    }


def render_c_header(result_gate: dict[str, Any], validation: dict[str, Any], failure_count: int) -> str:
    group_lines = []
    for item in result_gate.get("group_completion_gates") or []:
        group_lines.append(
            "  {"
            f"{as_int(item.get('group_index'))}ULL, "
            f"{as_int(item.get('completion_token_hash_u64'))}ULL, "
            f"{as_int(item.get('term_group_hash_u64'))}ULL, "
            f"{as_int(item.get('term'))}ULL, "
            f"{as_int(item.get('term_mask'))}ULL"
            "},"
        )
    lane_lines = []
    for item in result_gate.get("lane_completion_gates") or []:
        lane_lines.append(
            "  {"
            f"{as_int(item.get('hint_local_index'))}ULL, "
            f"{as_int(item.get('completion_token_hash_u64'))}ULL, "
            f"{as_int(item.get('kernel_lane_hash_u64'))}ULL, "
            f"{as_int(item.get('source_voter_hash_u64'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_VALIDATOR_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_VALIDATOR_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_TRANSFER {EXPECTED_TRANSFER}ULL
#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_EXPECTED_SECRET {EXPECTED_SECRET}ULL
#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_GROUP_COUNT {EXPECTED_GROUP_COUNT}ULL
#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_LANE_COUNT {EXPECTED_LANE_COUNT}ULL
#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_TARGET_TOKEN {EXPECTED_TARGET_TOKEN}ULL
#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_KERNEL_RESULT_PRESENT {1 if validation.get('kernel_result_present') else 0}ULL
#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_ACCEPTED_GROUPS {as_int(validation.get('accepted_group_completion_count'))}ULL
#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_ACCEPTED_LANES {as_int(validation.get('accepted_lane_completion_count'))}ULL
#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_TARGET_EXPORT_ACCEPTED {1 if validation.get('target_export_accepted') else 0}ULL
#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_PUBLIC_KEY_VERIFIED {1 if (validation.get('public_key_validation') or {}).get('public_key_verified') else 0}ULL
#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_COMPLETION_EVIDENCE_BOUND {1 if (validation.get('completion_evidence') or {}).get('evidence_bound') else 0}ULL
#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_RELATION_DERIVED_ECDLP {1 if validation.get('relation_derived_ecdlp') else 0}ULL
#define SELECTED13_PRIORITY0_RESULT_VALIDATOR_FAILURE_COUNT {failure_count}ULL

typedef struct {{
  uint64_t group_index;
  uint64_t completion_token_hash_u64;
  uint64_t term_group_hash_u64;
  uint64_t term;
  uint64_t term_mask;
}} selected13_priority0_result_validator_group_token_t;

typedef struct {{
  uint64_t hint_local_index;
  uint64_t completion_token_hash_u64;
  uint64_t kernel_lane_hash_u64;
  uint64_t source_voter_hash_u64;
}} selected13_priority0_result_validator_lane_token_t;

static const selected13_priority0_result_validator_group_token_t SELECTED13_PRIORITY0_RESULT_VALIDATOR_GROUP_TOKENS[] = {{
{chr(10).join(group_lines)}
}};

static const selected13_priority0_result_validator_lane_token_t SELECTED13_PRIORITY0_RESULT_VALIDATOR_LANE_TOKENS[] = {{
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
  const size_t group_count = sizeof(SELECTED13_PRIORITY0_RESULT_VALIDATOR_GROUP_TOKENS) / sizeof(SELECTED13_PRIORITY0_RESULT_VALIDATOR_GROUP_TOKENS[0]);
  const size_t lane_count = sizeof(SELECTED13_PRIORITY0_RESULT_VALIDATOR_LANE_TOKENS) / sizeof(SELECTED13_PRIORITY0_RESULT_VALIDATOR_LANE_TOKENS[0]);
  if (SELECTED13_PRIORITY0_RESULT_VALIDATOR_TRANSFER != {EXPECTED_TRANSFER}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESULT_VALIDATOR_EXPECTED_SECRET != {EXPECTED_SECRET}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESULT_VALIDATOR_TARGET_TOKEN != {EXPECTED_TARGET_TOKEN}ULL) failure_count++;
  if (group_count != SELECTED13_PRIORITY0_RESULT_VALIDATOR_GROUP_COUNT || group_count != {EXPECTED_GROUP_COUNT}ULL) failure_count++;
  if (lane_count != SELECTED13_PRIORITY0_RESULT_VALIDATOR_LANE_COUNT || lane_count != {EXPECTED_LANE_COUNT}ULL) failure_count++;
  for (size_t i = 0; i < group_count; i++) {{
    const selected13_priority0_result_validator_group_token_t *token = &SELECTED13_PRIORITY0_RESULT_VALIDATOR_GROUP_TOKENS[i];
    if (token->completion_token_hash_u64 == 0ULL || token->term_group_hash_u64 == 0ULL || token->term_mask == 0ULL) failure_count++;
  }}
  for (size_t i = 0; i < lane_count; i++) {{
    const selected13_priority0_result_validator_lane_token_t *token = &SELECTED13_PRIORITY0_RESULT_VALIDATOR_LANE_TOKENS[i];
    if (token->completion_token_hash_u64 == 0ULL || token->kernel_lane_hash_u64 == 0ULL || token->source_voter_hash_u64 == 0ULL) failure_count++;
  }}
  if (SELECTED13_PRIORITY0_RESULT_VALIDATOR_KERNEL_RESULT_PRESENT == 0ULL) {{
    if (SELECTED13_PRIORITY0_RESULT_VALIDATOR_ACCEPTED_GROUPS != 0ULL) failure_count++;
    if (SELECTED13_PRIORITY0_RESULT_VALIDATOR_ACCEPTED_LANES != 0ULL) failure_count++;
    if (SELECTED13_PRIORITY0_RESULT_VALIDATOR_TARGET_EXPORT_ACCEPTED != 0ULL) failure_count++;
    if (SELECTED13_PRIORITY0_RESULT_VALIDATOR_PUBLIC_KEY_VERIFIED != 0ULL) failure_count++;
    if (SELECTED13_PRIORITY0_RESULT_VALIDATOR_COMPLETION_EVIDENCE_BOUND != 0ULL) failure_count++;
    if (SELECTED13_PRIORITY0_RESULT_VALIDATOR_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;
  }}
  if (SELECTED13_PRIORITY0_RESULT_VALIDATOR_FAILURE_COUNT != 0ULL &&
      SELECTED13_PRIORITY0_RESULT_VALIDATOR_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_RESULT_VALIDATOR_RELATION_DERIVED_ECDLP != 0ULL &&
      SELECTED13_PRIORITY0_RESULT_VALIDATOR_COMPLETION_EVIDENCE_BOUND == 0ULL) failure_count++;

  printf("selected13_priority0_ffe_residual_kernel_result_validator_preflight transfer=%llu result_present=%llu groups=%llu lanes=%llu accepted_groups=%llu accepted_lanes=%llu target_export_accepted=%llu public_key_verified=%llu evidence_bound=%llu relation_derived=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_RESULT_VALIDATOR_TRANSFER,
         (unsigned long long)SELECTED13_PRIORITY0_RESULT_VALIDATOR_KERNEL_RESULT_PRESENT,
         (unsigned long long)group_count,
         (unsigned long long)lane_count,
         (unsigned long long)SELECTED13_PRIORITY0_RESULT_VALIDATOR_ACCEPTED_GROUPS,
         (unsigned long long)SELECTED13_PRIORITY0_RESULT_VALIDATOR_ACCEPTED_LANES,
         (unsigned long long)SELECTED13_PRIORITY0_RESULT_VALIDATOR_TARGET_EXPORT_ACCEPTED,
         (unsigned long long)SELECTED13_PRIORITY0_RESULT_VALIDATOR_PUBLIC_KEY_VERIFIED,
         (unsigned long long)SELECTED13_PRIORITY0_RESULT_VALIDATOR_COMPLETION_EVIDENCE_BOUND,
         (unsigned long long)SELECTED13_PRIORITY0_RESULT_VALIDATOR_RELATION_DERIVED_ECDLP,
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
    with tempfile.TemporaryDirectory(prefix="selected13_priority0_result_validator_", dir=str(temp_root)) as tmp:
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


def summarize(validation: dict[str, Any], failures: list[dict[str, Any]], native_preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_group_completion_count": as_int(validation.get("accepted_group_completion_count")),
        "accepted_lane_completion_count": as_int(validation.get("accepted_lane_completion_count")),
        "accepted_relation_export_count": 1 if validation.get("relation_derived_ecdlp") and not failures else 0,
        "completion_evidence_bound": bool((validation.get("completion_evidence") or {}).get("evidence_bound")),
        "completion_evidence_issue_count": as_int((validation.get("completion_evidence") or {}).get("group_evidence_issue_count"))
        + as_int((validation.get("completion_evidence") or {}).get("lane_evidence_issue_count"))
        + as_int((validation.get("completion_evidence") or {}).get("target_evidence_issue_count")),
        "failure_count": len(failures),
        "kernel_result_present": bool(validation.get("kernel_result_present")),
        "native_preflight_verified": bool(native_preflight.get("verified")),
        "pollard_rho_speedup_claimed": False,
        "public_key_verified": bool((validation.get("public_key_validation") or {}).get("public_key_verified")),
        "relation_derived_ecdlp": bool(validation.get("relation_derived_ecdlp")) and not failures,
        "result_status": validation.get("result_status"),
        "target_export_accepted": bool(validation.get("target_export_accepted")),
        "verified": not failures,
        "worker_interpretation": (
            "The validator is ready to accept an external kernel result. With no kernel result "
            "provided, it records all completion tokens as pending and keeps relation_derived_ecdlp false."
        )
        if not validation.get("kernel_result_present")
        else (
            "The validator accepted the external kernel result."
            if validation.get("relation_derived_ecdlp") and not failures
            else "The external kernel result was rejected by the fail-closed token/public-key checks."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-gate", default=str(DEFAULT_RESULT_GATE))
    parser.add_argument("--kernel-result", default="")
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_gate_path = Path(args.result_gate)
    result_gate = load_json(result_gate_path)
    kernel_path = resolve_path((result_gate.get("parameters") or {}).get("kernel_contract"), result_gate_path.parent)
    if kernel_path is None:
        raise FileNotFoundError("result gate does not identify its kernel_contract source")
    kernel_contract = load_json(kernel_path)
    kernel_result_path = Path(args.kernel_result) if args.kernel_result else None
    kernel_result = load_json(kernel_result_path) if kernel_result_path is not None else None
    validation, failures = validate_kernel_result(result_gate, kernel_contract, kernel_result, Path(args.task_dir))
    claim_status = (
        "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_VALIDATED"
        if validation.get("relation_derived_ecdlp") and not failures
        else "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_VALIDATOR_AWAITING_KERNEL_OUTPUT"
        if kernel_result is None and not failures
        else "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_VALIDATOR_FAILED"
    )
    payload = {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": claim_status,
        "parameters": {
            "kernel_contract": str(kernel_path),
            "kernel_result": str(kernel_result_path) if kernel_result_path is not None else None,
            "result_gate": str(result_gate_path),
            "task_dir": str(args.task_dir),
        },
        "packet_hash_u64": stable_hash_u64({"result_gate": result_gate.get("packet_hash_u64"), "validation": validation}),
        "target_slot": result_gate.get("target_slot"),
        "expected_kernel_result_template": expected_kernel_result_template(result_gate),
        "validation": validation,
        "honesty_boundary": {
            "accepted_relation_export_count": 1 if validation.get("relation_derived_ecdlp") and not failures else 0,
            "external_kernel_result_required": True,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": bool(validation.get("relation_derived_ecdlp")) and not failures,
            "summation_polynomial_evaluated_by_validator": False,
        },
        "failures": failures,
    }
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(result_gate, validation, len(failures)))
    payload["artifacts"] = {"c_header": str(header_path)}
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_VALIDATOR_FAILED"
    payload["summary"] = summarize(payload["validation"], payload["failures"], native_preflight)
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if payload["claim_status"] != "SELECTED13_PRIORITY0_FFE_RESIDUAL_KERNEL_RESULT_VALIDATOR_FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
