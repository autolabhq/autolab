#!/usr/bin/env python3
"""Emit the verifier gate for selected13 priority-0 target exports.

The residual synthesis worklist for transfer 10376 is now precise enough to
define the next acceptance boundary.  The worker still has to synthesize the
residual slots and emit a fresh target direct/rank row, but any candidate row
must be checked against the artifact-derived source consensus before it can set
`relation_derived_ecdlp`.

This is a verifier gate only.  It does not materialize coefficients, synthesize
residual terms, emit a target direct/rank row, solve ECDLP, or claim a
Pollard-rho speedup.
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


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_target_export_verifier_gate.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SYNTHESIS_WORKLIST = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_residual_synthesis_worklist_10376_probe.json"
)
DEFAULT_COEFF_GAP = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_residual_coeff_gap_manifest_10376_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_target_export_verifier_gate_10376_probe.json"
DEFAULT_C_HEADER_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_target_export_verifier_gate_10376_probe.h"

EXPECTED_TRANSFER = 10376
EXPECTED_ROW_REQUEST_ID = "selected13_matmiss_10376_0_d352d0d05610"
EXPECTED_SOURCE_CONSENSUS_SECRET = 5859
SOURCE_VOTER_CLASS_CODES = {
    "direct_certificate_source": 1,
    "shared_product_source": 2,
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


def term_slots(worklist: dict[str, Any]) -> list[dict[str, Any]]:
    return [slot for item in worklist.get("lane_work_items") or [] for slot in item.get("term_slots") or []]


def lane_by_hint(worklist: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {as_int(item.get("hint_local_index"), -1): item for item in worklist.get("lane_work_items") or []}


def validate_sources(worklist: dict[str, Any], coeff_gap: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if worklist.get("claim_status") != "SELECTED13_PRIORITY0_RESIDUAL_SYNTHESIS_WORKLIST_READY":
        failures.append({"code": "synthesis_worklist_not_ready", "claim_status": worklist.get("claim_status")})
    if worklist.get("failures"):
        failures.append({"code": "synthesis_worklist_has_failures", "failures": worklist.get("failures")})
    if (worklist.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "synthesis_worklist_summary_not_verified", "summary": worklist.get("summary")})
    if (worklist.get("native_preflight") or {}).get("verified") is not True:
        failures.append({"code": "synthesis_worklist_native_preflight_not_verified"})
    if coeff_gap.get("claim_status") != "SELECTED13_PRIORITY0_RESIDUAL_COEFF_GAP_MANIFEST_READY":
        failures.append({"code": "coeff_gap_not_ready", "claim_status": coeff_gap.get("claim_status")})
    if coeff_gap.get("failures"):
        failures.append({"code": "coeff_gap_has_failures", "failures": coeff_gap.get("failures")})
    if (coeff_gap.get("summary") or {}).get("verified") is not True:
        failures.append({"code": "coeff_gap_summary_not_verified", "summary": coeff_gap.get("summary")})
    target = worklist.get("target_slot") or {}
    if as_int(target.get("transfer_index"), -1) != EXPECTED_TRANSFER:
        failures.append({"code": "target_transfer_unexpected", "observed": target.get("transfer_index")})
    if target.get("row_request_id") != EXPECTED_ROW_REQUEST_ID:
        failures.append({"code": "target_row_request_unexpected", "observed": target.get("row_request_id")})
    if as_int(target.get("row_request_id_u64")) == 0:
        failures.append({"code": "target_row_request_u64_missing"})
    required = worklist.get("required_worker_outputs") or {}
    for field in (
        "must_materialize_shared_product_coefficients_for_hybrid_lanes",
        "must_synthesize_all_non_guard_residual_terms",
        "must_emit_target_direct_rank_export",
        "must_verify_public_key",
        "must_set_relation_derived_ecdlp",
    ):
        if required.get(field) is not True:
            failures.append({"code": "required_worker_output_missing", "field": field})
    summary = worklist.get("summary") or {}
    if as_int(summary.get("accepted_relation_export_count")) != 0:
        failures.append({"code": "synthesis_worklist_already_has_exports"})
    if summary.get("relation_derived_ecdlp") is not False:
        failures.append({"code": "synthesis_worklist_relation_derived_unexpected"})
    return failures


def direct_source_voters(coeff_gap: dict[str, Any], worklist: dict[str, Any]) -> list[dict[str, Any]]:
    lanes = lane_by_hint(worklist)
    out = []
    for lane in coeff_gap.get("residual_coeff_lanes") or []:
        secret = lane.get("direct_source_derived_secret")
        if secret is None:
            continue
        hint = as_int(lane.get("hint_local_index"), -1)
        work_lane = lanes.get(hint, {})
        voter = {
            "derived_secret": as_int(secret),
            "evidence_count": as_int(lane.get("coefficient_form_count")),
            "evidence_rank": as_int(lane.get("coefficient_rank_mod_order")),
            "hint_local_index": hint,
            "materialization_required": False,
            "source_public_key_verified": bool(lane.get("direct_public_key_verified")),
            "source_row_hash_u64": as_int(lane.get("source_row_hash_u64")),
            "source_selector": lane.get("source_selector"),
            "source_top_k": as_int(lane.get("source_top_k")),
            "source_voter_class": "direct_certificate_source",
            "source_voter_class_code": SOURCE_VOTER_CLASS_CODES["direct_certificate_source"],
            "term_slot_count": as_int(work_lane.get("term_slot_count")),
            "voter_interpretation": "direct certificate coefficient source, used as a guard rather than a target export",
        }
        voter["source_voter_hash_u64"] = stable_hash_u64(voter)
        out.append(voter)
    return out


def shared_product_source_voters(worklist: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for lane in worklist.get("lane_work_items") or []:
        if lane.get("lane_class") != "shared_product_verified_coeff_materialization":
            continue
        voter = {
            "derived_secret": as_int(lane.get("shared_product_union_derived_secret")),
            "evidence_count": as_int(lane.get("shared_product_union_relation_count")),
            "evidence_rank": as_int(lane.get("shared_product_union_rank")),
            "hint_local_index": as_int(lane.get("hint_local_index"), -1),
            "materialization_required": True,
            "source_public_key_verified": bool(lane.get("shared_product_union_public_key_verified")),
            "source_row_hash_u64": as_int(lane.get("source_row_hash_u64")),
            "source_selector": lane.get("source_selector"),
            "source_top_k": as_int(lane.get("source_top_k")),
            "source_voter_class": "shared_product_source",
            "source_voter_class_code": SOURCE_VOTER_CLASS_CODES["shared_product_source"],
            "term_slot_count": as_int(lane.get("term_slot_count")),
            "voter_interpretation": "shared-product verified source relation, coefficient materialization still required",
        }
        voter["source_voter_hash_u64"] = stable_hash_u64(voter)
        out.append(voter)
    return out


def build_source_voters(worklist: dict[str, Any], coeff_gap: dict[str, Any]) -> list[dict[str, Any]]:
    voters = direct_source_voters(coeff_gap, worklist)
    voters.extend(shared_product_source_voters(worklist))
    return sorted(voters, key=lambda item: (as_int(item.get("hint_local_index"), -1), item.get("source_voter_class", "")))


def source_consensus(voters: list[dict[str, Any]]) -> dict[str, Any]:
    secrets = [as_int(voter.get("derived_secret")) for voter in voters]
    unique_secrets = sorted(set(secrets))
    expected_secret = unique_secrets[0] if len(unique_secrets) == 1 else 0
    classes = Counter(str(voter.get("source_voter_class")) for voter in voters)
    return {
        "all_source_voters_match_expected_secret": len(unique_secrets) == 1
        and expected_secret == EXPECTED_SOURCE_CONSENSUS_SECRET,
        "expected_secret": expected_secret,
        "expected_secret_source": "artifact_consensus_from_verified_source_voters",
        "secret_histogram": dict(sorted(Counter(secrets).items())),
        "source_voter_class_counts": dict(sorted(classes.items())),
        "source_voter_count": len(voters),
        "unique_secrets": unique_secrets,
        "verified_source_voter_count": sum(1 for voter in voters if voter.get("source_public_key_verified")),
    }


def validate_gate(
    worklist: dict[str, Any],
    coeff_gap: dict[str, Any],
    voters: list[dict[str, Any]],
    consensus: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = validate_sources(worklist, coeff_gap)
    summary = worklist.get("summary") or {}
    slots = term_slots(worklist)
    classes = Counter(str(voter.get("source_voter_class")) for voter in voters)
    if len(voters) != 3:
        failures.append({"code": "source_voter_count_unexpected", "observed": len(voters)})
    if classes.get("direct_certificate_source", 0) != 1:
        failures.append({"code": "direct_source_voter_count_unexpected", "classes": dict(classes)})
    if classes.get("shared_product_source", 0) != 2:
        failures.append({"code": "shared_product_source_voter_count_unexpected", "classes": dict(classes)})
    if consensus.get("all_source_voters_match_expected_secret") is not True:
        failures.append({"code": "source_consensus_secret_mismatch", "consensus": consensus})
    if as_int(consensus.get("expected_secret")) != EXPECTED_SOURCE_CONSENSUS_SECRET:
        failures.append({"code": "source_consensus_secret_unexpected", "consensus": consensus})
    if as_int(consensus.get("verified_source_voter_count")) != len(voters):
        failures.append({"code": "source_voter_not_public_key_verified", "consensus": consensus})
    if as_int(summary.get("term_slot_count")) != 20 or len(slots) != 20:
        failures.append({"code": "term_slot_count_unexpected", "summary": summary, "observed": len(slots)})
    if as_int(summary.get("synthesis_required_term_count")) != 19:
        failures.append({"code": "synthesis_required_term_count_unexpected", "summary": summary})
    if as_int(summary.get("shared_product_verified_lane_count")) != 2:
        failures.append({"code": "shared_verified_lane_count_unexpected", "summary": summary})
    for voter in voters:
        if as_int(voter.get("source_voter_hash_u64")) == 0:
            failures.append({"code": "source_voter_hash_missing", "hint": voter.get("hint_local_index")})
        if as_int(voter.get("source_row_hash_u64")) == 0:
            failures.append({"code": "source_row_hash_missing", "hint": voter.get("hint_local_index")})
        if as_int(voter.get("evidence_rank")) != 2 or as_int(voter.get("evidence_count")) != 2:
            failures.append({"code": "source_evidence_shape_unexpected", "voter": voter})
        if voter.get("source_voter_class") == "shared_product_source" and voter.get("materialization_required") is not True:
            failures.append({"code": "shared_product_voter_must_require_materialization", "voter": voter})
    return failures


def render_c_header(voters: list[dict[str, Any]], worklist: dict[str, Any], consensus: dict[str, Any]) -> str:
    summary = worklist.get("summary") or {}
    target = worklist.get("target_slot") or {}
    voter_lines = []
    for voter in voters:
        voter_lines.append(
            "  {"
            f"{as_int(voter.get('hint_local_index'))}ULL, "
            f"{as_int(voter.get('source_voter_class_code'))}ULL, "
            f"{as_int(voter.get('source_voter_hash_u64'))}ULL, "
            f"{as_int(voter.get('source_row_hash_u64'))}ULL, "
            f"{1 if voter.get('source_public_key_verified') else 0}ULL, "
            f"{as_int(voter.get('derived_secret'))}ULL, "
            f"{as_int(voter.get('evidence_rank'))}ULL, "
            f"{as_int(voter.get('evidence_count'))}ULL, "
            f"{1 if voter.get('materialization_required') else 0}ULL, "
            f"{as_int(voter.get('term_slot_count'))}ULL"
            "},"
        )
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_TARGET_EXPORT_VERIFIER_GATE_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_TARGET_EXPORT_VERIFIER_GATE_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_TARGET_GATE_TRANSFER {as_int(target.get('transfer_index'))}ULL
#define SELECTED13_PRIORITY0_TARGET_GATE_ROW_REQUEST_U64 {as_int(target.get('row_request_id_u64'))}ULL
#define SELECTED13_PRIORITY0_TARGET_GATE_EXPECTED_SECRET {as_int(consensus.get('expected_secret'))}ULL
#define SELECTED13_PRIORITY0_TARGET_GATE_SOURCE_VOTER_COUNT {len(voters)}ULL
#define SELECTED13_PRIORITY0_TARGET_GATE_TERM_SLOT_COUNT {as_int(summary.get('term_slot_count'))}ULL
#define SELECTED13_PRIORITY0_TARGET_GATE_SYNTHESIS_REQUIRED_TERM_COUNT {as_int(summary.get('synthesis_required_term_count'))}ULL
#define SELECTED13_PRIORITY0_TARGET_GATE_HYBRID_COEFF_LANE_COUNT {as_int(summary.get('shared_product_verified_lane_count'))}ULL
#define SELECTED13_PRIORITY0_TARGET_GATE_ACCEPTED_EXPORT_COUNT 0ULL
#define SELECTED13_PRIORITY0_TARGET_GATE_RELATION_DERIVED_ECDLP 0ULL
#define SELECTED13_PRIORITY0_TARGET_GATE_POLLARD_RHO_SPEEDUP_CLAIMED 0ULL

#define SELECTED13_TARGET_GATE_VOTER_DIRECT_CERTIFICATE 1ULL
#define SELECTED13_TARGET_GATE_VOTER_SHARED_PRODUCT 2ULL

typedef struct {{
  uint64_t hint_local_index;
  uint64_t source_voter_class_code;
  uint64_t source_voter_hash_u64;
  uint64_t source_row_hash_u64;
  uint64_t source_public_key_verified;
  uint64_t derived_secret;
  uint64_t evidence_rank;
  uint64_t evidence_count;
  uint64_t materialization_required;
  uint64_t term_slot_count;
}} selected13_priority0_target_gate_source_voter_t;

static const selected13_priority0_target_gate_source_voter_t SELECTED13_PRIORITY0_TARGET_GATE_SOURCE_VOTERS[] = {{
{chr(10).join(voter_lines)}
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
  uint64_t direct_voter_count = 0;
  uint64_t shared_product_voter_count = 0;
  uint64_t verified_voter_count = 0;
  uint64_t matching_secret_count = 0;
  uint64_t materialization_required_count = 0;
  uint64_t term_slot_sum = 0;

  const size_t voter_count = sizeof(SELECTED13_PRIORITY0_TARGET_GATE_SOURCE_VOTERS) / sizeof(SELECTED13_PRIORITY0_TARGET_GATE_SOURCE_VOTERS[0]);
  if (SELECTED13_PRIORITY0_TARGET_GATE_TRANSFER != {EXPECTED_TRANSFER}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TARGET_GATE_ROW_REQUEST_U64 == 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TARGET_GATE_EXPECTED_SECRET != {EXPECTED_SOURCE_CONSENSUS_SECRET}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TARGET_GATE_SOURCE_VOTER_COUNT != 3ULL) failure_count++;
  if (voter_count != SELECTED13_PRIORITY0_TARGET_GATE_SOURCE_VOTER_COUNT) failure_count++;
  if (SELECTED13_PRIORITY0_TARGET_GATE_TERM_SLOT_COUNT != 20ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TARGET_GATE_SYNTHESIS_REQUIRED_TERM_COUNT != 19ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TARGET_GATE_HYBRID_COEFF_LANE_COUNT != 2ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TARGET_GATE_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TARGET_GATE_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_TARGET_GATE_POLLARD_RHO_SPEEDUP_CLAIMED != 0ULL) failure_count++;

  for (size_t i = 0; i < voter_count; i++) {{
    const selected13_priority0_target_gate_source_voter_t *voter = &SELECTED13_PRIORITY0_TARGET_GATE_SOURCE_VOTERS[i];
    if (voter->source_voter_hash_u64 == 0ULL || voter->source_row_hash_u64 == 0ULL) failure_count++;
    if (voter->source_public_key_verified) verified_voter_count++;
    if (voter->derived_secret == SELECTED13_PRIORITY0_TARGET_GATE_EXPECTED_SECRET) matching_secret_count++;
    if (voter->evidence_rank != 2ULL || voter->evidence_count != 2ULL) failure_count++;
    if (voter->term_slot_count == 0ULL) failure_count++;
    term_slot_sum += voter->term_slot_count;
    if (voter->materialization_required) materialization_required_count++;
    if (voter->source_voter_class_code == SELECTED13_TARGET_GATE_VOTER_DIRECT_CERTIFICATE) {{
      direct_voter_count++;
      if (voter->materialization_required != 0ULL) failure_count++;
    }} else if (voter->source_voter_class_code == SELECTED13_TARGET_GATE_VOTER_SHARED_PRODUCT) {{
      shared_product_voter_count++;
      if (voter->materialization_required == 0ULL) failure_count++;
    }} else {{
      failure_count++;
    }}
  }}
  if (direct_voter_count != 1ULL) failure_count++;
  if (shared_product_voter_count != 2ULL) failure_count++;
  if (verified_voter_count != 3ULL) failure_count++;
  if (matching_secret_count != 3ULL) failure_count++;
  if (materialization_required_count != 2ULL) failure_count++;
  if (term_slot_sum != SELECTED13_PRIORITY0_TARGET_GATE_TERM_SLOT_COUNT) failure_count++;

  printf("selected13_priority0_target_export_verifier_gate_preflight transfer=%llu expected_secret=%llu source_voters=%llu terms=%llu synthesis_terms=%llu hybrid_coeff_lanes=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_TARGET_GATE_TRANSFER,
         (unsigned long long)SELECTED13_PRIORITY0_TARGET_GATE_EXPECTED_SECRET,
         (unsigned long long)voter_count,
         (unsigned long long)SELECTED13_PRIORITY0_TARGET_GATE_TERM_SLOT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_TARGET_GATE_SYNTHESIS_REQUIRED_TERM_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_TARGET_GATE_HYBRID_COEFF_LANE_COUNT,
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
    with tempfile.TemporaryDirectory(prefix="selected13_priority0_target_gate_", dir=str(temp_root)) as tmp:
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


def summarize(
    voters: list[dict[str, Any]],
    consensus: dict[str, Any],
    worklist: dict[str, Any],
    failures: list[dict[str, Any]],
    native_preflight: dict[str, Any],
) -> dict[str, Any]:
    summary = worklist.get("summary") or {}
    return {
        "accepted_relation_export_count": 0,
        "expected_secret": as_int(consensus.get("expected_secret")),
        "failure_count": len(failures),
        "hybrid_coefficient_materialization_lane_count": as_int(summary.get("shared_product_verified_lane_count")),
        "native_preflight_verified": bool(native_preflight.get("verified")),
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "source_consensus_verified": bool(consensus.get("all_source_voters_match_expected_secret"))
        and as_int(consensus.get("verified_source_voter_count")) == len(voters),
        "source_voter_count": len(voters),
        "synthesis_required_term_count": as_int(summary.get("synthesis_required_term_count")),
        "term_slot_count": as_int(summary.get("term_slot_count")),
        "verified": not failures,
        "worker_interpretation": (
            "The gate binds a future 10376 target direct/rank export to the verified source "
            "consensus secret 5859, the exact row request id, all 20 synthesis slots, public-key "
            "verification, and relation_derived_ecdlp=true. It does not accept a source replay as "
            "a target export."
        ),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    synthesis_path = Path(args.synthesis_worklist)
    coeff_gap_path = Path(args.coeff_gap)
    worklist = load_json(synthesis_path)
    coeff_gap = load_json(coeff_gap_path)
    voters = build_source_voters(worklist, coeff_gap)
    consensus = source_consensus(voters)
    failures = validate_gate(worklist, coeff_gap, voters, consensus)
    target = worklist.get("target_slot") or {}
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_TARGET_EXPORT_VERIFIER_GATE_READY"
            if not failures
            else "SELECTED13_PRIORITY0_TARGET_EXPORT_VERIFIER_GATE_FAILED"
        ),
        "parameters": {
            "coeff_gap": str(coeff_gap_path),
            "synthesis_worklist": str(synthesis_path),
        },
        "packet_hash_u64": stable_hash_u64({"target": target, "source_voters": voters, "consensus": consensus}),
        "target_slot": target,
        "source_consensus": consensus,
        "source_voters": voters,
        "target_export_acceptance_gate": {
            "accepted_relation_export_count": 0,
            "expected_secret": as_int(consensus.get("expected_secret")),
            "expected_secret_source": consensus.get("expected_secret_source"),
            "must_complete_synthesis_required_terms": as_int((worklist.get("summary") or {}).get("synthesis_required_term_count")),
            "must_complete_term_slot_count": as_int((worklist.get("summary") or {}).get("term_slot_count")),
            "must_emit_target_direct_rank_export": True,
            "must_materialize_hybrid_coeff_lane_count": as_int(
                (worklist.get("summary") or {}).get("shared_product_verified_lane_count")
            ),
            "must_match_expected_secret": as_int(consensus.get("expected_secret")),
            "must_match_row_request_id": target.get("row_request_id"),
            "must_reject_source_hint_replay_as_target_export": True,
            "must_set_relation_derived_ecdlp": True,
            "must_verify_public_key": True,
            "public_key_validation_scope": (
                "The mounted LMFDB curve record does not carry a static precomputed challenge public key; "
                "the target public key is replay-artifact-specific. Future target exports must therefore "
                "run the verifier on their emitted public key while matching the artifact-derived source "
                "consensus scalar recorded here."
            ),
            "relation_derived_ecdlp": False,
            "target_row_request_id_u64": as_int(target.get("row_request_id_u64")),
        },
        "failures": failures,
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "source_voters_are_not_target_row_exports": True,
            "target_export_not_emitted_by_this_gate": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis-worklist", default=str(DEFAULT_SYNTHESIS_WORKLIST))
    parser.add_argument("--coeff-gap", default=str(DEFAULT_COEFF_GAP))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    worklist = load_json(Path(args.synthesis_worklist))
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload["source_voters"], worklist, payload["source_consensus"]))
    payload["artifacts"] = {"c_header": str(header_path)}
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_TARGET_EXPORT_VERIFIER_GATE_FAILED"
    payload["summary"] = summarize(
        payload["source_voters"],
        payload["source_consensus"],
        worklist,
        payload["failures"],
        native_preflight,
    )
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
