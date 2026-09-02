#!/usr/bin/env python3
"""Build a lift worklist from the 10376 target-pair direct certificate.

The direct-certificate bridge identifies one public-key-verified target-row-pair
certificate for transfer 10376 at `mode_low_term_support_total5/top_k=12`.
The selected13 target-export gate requires the same row pair at `top_k=16`,
whose selected support adds six terms.  This worklist classifies those missing
terms by available evidence so the next worker can focus on the two still-open
FFE/summation-polynomial terms 8 and 12.
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


SCHEMA = "ecdlp.low_term_total2_selected13_priority0_direct_certificate_lift_worklist.v1"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_TARGET_GATE = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_target_export_verifier_gate_10376_probe.json"
DEFAULT_DIRECT_BRIDGE = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_direct_certificate_bridge_10376_probe.json"
DEFAULT_DEPENDENCY_BRIDGE = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_dependency_circuit_evidence_bridge_10376_probe.json"
)
DEFAULT_GAP_SCANNER = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_gap_closure_scanner_10376_probe.json"
DEFAULT_TERM812_SCOUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_term812_route_scout_10376_probe.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_direct_certificate_lift_worklist_10376_probe.json"
DEFAULT_C_HEADER_OUT = (
    DEFAULT_STATE_DIR / "low_term_total2_selected13_priority0_direct_certificate_lift_worklist_10376_probe.h"
)

EXPECTED_TRANSFER = 10376
EXPECTED_SOURCE_CERT_COUNT = 1
EXPECTED_SOURCE_SUPPORT_COUNT = 9
EXPECTED_TARGET_SUPPORT_COUNT = 15
EXPECTED_LIFT_TERM_COUNT = 6
EXPECTED_DEPENDENCY_BACKED_TERM_COUNT = 2
EXPECTED_ANALOGUE_BACKED_TERM_COUNT = 2
EXPECTED_OPEN_TERM_COUNT = 2
EXPECTED_OPEN_TERMS = [8, 12]
EXPECTED_LEAF8_SURFACE_HIT_COUNT = 4
EXPECTED_LEAF12_SEED_COUNT = 0


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


def term_list(value: Any) -> list[int]:
    return sorted({as_int(term, -1) for term in value or [] if as_int(term, -1) >= 0})


def pick_target_pair_certificate(direct_bridge: dict[str, Any]) -> dict[str, Any]:
    certs = (direct_bridge.get("direct_certificate_bridge") or {}).get("target_pair_certificates") or []
    public_certs = [
        cert
        for cert in certs
        if cert.get("public_key_verified")
        and cert.get("secret_matches_expected")
        and cert.get("below_rho")
        and not cert.get("exact_target_slot_match")
    ]
    if not public_certs:
        return {}
    return sorted(
        public_certs,
        key=lambda cert: (
            as_float(cert.get("direct_ops_over_rho"), 10**18) or 10**18,
            as_int(cert.get("top_k"), 10**18),
        ),
    )[0]


def evidence_class_for_term(
    term: int,
    dependency_terms: set[int],
    analogue_terms: set[int],
    open_terms: set[int],
    term812_summary: dict[str, Any],
) -> tuple[str, str, int]:
    if term in open_terms:
        if term == 8 and as_int(term812_summary.get("leaf8_surface_gate_unique_hit_count")) > 0:
            return (
                "open_leaf_route_seed_only",
                "Use same-target leaf-8 hit streams as generation seeds, then require explicit residual-term-8 evidence.",
                1,
            )
        if term == 12 and as_int(term812_summary.get("leaf12_seed_count")) == 0:
            return (
                "open_no_same_target_route_seed",
                "Generate fresh same-target FFE/summation-polynomial residual-term-12 evidence.",
                0,
            )
        return ("open_residual_synthesis_required", "Generate fresh same-target residual-term evidence.", 1)
    if term in analogue_terms:
        return (
            "same_target_analogue_replay_required",
            "Replay same-target analogue evidence onto the transfer-10376 target slice and bind it to the row request.",
            2,
        )
    if term in dependency_terms:
        return (
            "dependency_circuit_partial_evidence",
            "Materialize dependency-circuit evidence into the selected13 kernel completion hash.",
            3,
        )
    return ("unclassified_missing_term", "Inspect source support before worker execution.", 4)


def build_worklist(
    target_gate: dict[str, Any],
    direct_bridge: dict[str, Any],
    dependency_bridge: dict[str, Any],
    gap_scanner: dict[str, Any],
    term812_scout: dict[str, Any],
) -> dict[str, Any]:
    target_slot = target_gate.get("target_slot") or {}
    target_support = term_list(target_slot.get("selected_term_support"))
    source_cert = pick_target_pair_certificate(direct_bridge)
    source_support = term_list(source_cert.get("selected_term_support"))
    lift_terms = sorted(set(target_support) - set(source_support))
    dependency_terms = {
        as_int(group.get("term"), -1)
        for group in (dependency_bridge.get("group_evidence_bridge") or [])
        if group.get("covered_by_public_dependency_circuit")
    }
    analogue_terms = set(term_list((gap_scanner.get("summary") or {}).get("same_target_new_terms")))
    open_terms = set(term_list((gap_scanner.get("summary") or {}).get("remaining_same_target_terms")))
    term812_summary = term812_scout.get("summary") or {}
    work_items = []
    for term in lift_terms:
        evidence_class, next_action, priority = evidence_class_for_term(
            term, dependency_terms, analogue_terms, open_terms, term812_summary
        )
        work_items.append(
            {
                "already_in_source_certificate": term in source_support,
                "covered_by_dependency_bridge": term in dependency_terms,
                "evidence_class": evidence_class,
                "next_action": next_action,
                "priority": priority,
                "same_target_analogue_available": term in analogue_terms,
                "same_target_residual_term_open": term in open_terms,
                "term": term,
            }
        )
    work_items.sort(key=lambda item: (as_int(item.get("priority"), 99), as_int(item.get("term"), 99)))
    dependency_backed = [item for item in work_items if item.get("covered_by_dependency_bridge")]
    analogue_backed = [item for item in work_items if item.get("same_target_analogue_available")]
    open_items = [item for item in work_items if item.get("same_target_residual_term_open")]
    failures = []
    summary = {
        "accepted_relation_export_count": 0,
        "analogue_backed_lift_term_count": len(analogue_backed),
        "dependency_backed_lift_term_count": len(dependency_backed),
        "exact_target_slot_direct_verified_count": as_int(
            (direct_bridge.get("summary") or {}).get("exact_target_slot_direct_verified_count")
        ),
        "failure_count": 0,
        "leaf12_seed_count": as_int(term812_summary.get("leaf12_seed_count")),
        "leaf8_surface_gate_unique_hit_count": as_int(term812_summary.get("leaf8_surface_gate_unique_hit_count")),
        "lift_term_count": len(lift_terms),
        "native_preflight_verified": False,
        "open_lift_term_count": len(open_items),
        "open_lift_terms": sorted(as_int(item.get("term"), -1) for item in open_items),
        "pollard_rho_speedup_claimed": False,
        "relation_derived_ecdlp": False,
        "source_certificate_count": 1 if source_cert else 0,
        "source_certificate_direct_ops_over_rho": as_float(source_cert.get("direct_ops_over_rho")),
        "source_certificate_top_k": as_int(source_cert.get("top_k"), -1),
        "source_support_count": len(source_support),
        "target_support_count": len(target_support),
        "target_transfer": as_int(target_slot.get("transfer_index"), -1),
        "verified": True,
        "worker_interpretation": (
            "The top_k=12 target-pair certificate can serve as a positive control for a top_k=16 lift. "
            "Terms 2 and 13 have dependency-circuit partial evidence, terms 3 and 9 have same-target "
            "analogue evidence, and terms 8 and 12 still require fresh residual synthesis."
        ),
    }
    checks = [
        ("transfer_unexpected", summary["target_transfer"] == EXPECTED_TRANSFER, summary["target_transfer"]),
        (
            "source_certificate_count_unexpected",
            summary["source_certificate_count"] == EXPECTED_SOURCE_CERT_COUNT,
            summary["source_certificate_count"],
        ),
        (
            "source_support_count_unexpected",
            summary["source_support_count"] == EXPECTED_SOURCE_SUPPORT_COUNT,
            summary["source_support_count"],
        ),
        (
            "target_support_count_unexpected",
            summary["target_support_count"] == EXPECTED_TARGET_SUPPORT_COUNT,
            summary["target_support_count"],
        ),
        ("lift_term_count_unexpected", summary["lift_term_count"] == EXPECTED_LIFT_TERM_COUNT, summary["lift_term_count"]),
        (
            "dependency_backed_lift_term_count_unexpected",
            summary["dependency_backed_lift_term_count"] == EXPECTED_DEPENDENCY_BACKED_TERM_COUNT,
            summary["dependency_backed_lift_term_count"],
        ),
        (
            "analogue_backed_lift_term_count_unexpected",
            summary["analogue_backed_lift_term_count"] == EXPECTED_ANALOGUE_BACKED_TERM_COUNT,
            summary["analogue_backed_lift_term_count"],
        ),
        (
            "open_lift_term_count_unexpected",
            summary["open_lift_term_count"] == EXPECTED_OPEN_TERM_COUNT,
            summary["open_lift_term_count"],
        ),
        (
            "open_lift_terms_unexpected",
            summary["open_lift_terms"] == EXPECTED_OPEN_TERMS,
            summary["open_lift_terms"],
        ),
        (
            "leaf8_surface_gate_unique_hit_count_unexpected",
            summary["leaf8_surface_gate_unique_hit_count"] == EXPECTED_LEAF8_SURFACE_HIT_COUNT,
            summary["leaf8_surface_gate_unique_hit_count"],
        ),
        (
            "leaf12_seed_count_unexpected",
            summary["leaf12_seed_count"] == EXPECTED_LEAF12_SEED_COUNT,
            summary["leaf12_seed_count"],
        ),
        (
            "exact_target_slot_direct_verified_nonzero",
            summary["exact_target_slot_direct_verified_count"] == 0,
            summary["exact_target_slot_direct_verified_count"],
        ),
    ]
    for code, ok, observed in checks:
        if not ok:
            failures.append({"code": code, "observed": observed})
    summary["failure_count"] = len(failures)
    summary["verified"] = not failures
    return {
        "schema": SCHEMA,
        "created_at": now_iso(),
        "claim_status": (
            "SELECTED13_PRIORITY0_DIRECT_CERTIFICATE_LIFT_WORKLIST_READY"
            if not failures
            else "SELECTED13_PRIORITY0_DIRECT_CERTIFICATE_LIFT_WORKLIST_FAILED"
        ),
        "packet_hash_u64": stable_hash_u64(
            {
                "lift_terms": lift_terms,
                "source_certificate": source_cert,
                "target_slot": target_slot,
                "work_items": work_items,
            }
        ),
        "target_slot": target_slot,
        "source_certificate": source_cert,
        "source_support_terms": source_support,
        "target_support_terms": target_support,
        "lift_terms": lift_terms,
        "lift_work_items": work_items,
        "evidence_sources": {
            "dependency_bridge_terms": sorted(dependency_terms),
            "gap_scanner_same_target_new_terms": sorted(analogue_terms),
            "gap_scanner_remaining_same_target_terms": sorted(open_terms),
            "term812_summary": term812_summary,
        },
        "honesty_boundary": {
            "accepted_relation_export_count": 0,
            "direct_certificate_is_lift_source_only": True,
            "exact_target_slot_direct_verified": False,
            "general_ecdlp_algorithm_claimed": False,
            "pollard_rho_speedup_claimed": False,
            "relation_derived_ecdlp": False,
            "validator_acceptance_ready": False,
        },
        "failures": failures,
        "summary": summary,
    }


def render_c_header(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    return f"""#ifndef LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_DIRECT_CERTIFICATE_LIFT_WORKLIST_H
#define LOW_TERM_TOTAL2_SELECTED13_PRIORITY0_DIRECT_CERTIFICATE_LIFT_WORKLIST_H

#include <stdint.h>

#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_TRANSFER {as_int(summary.get('target_transfer'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_SOURCE_CERT_COUNT {as_int(summary.get('source_certificate_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_SOURCE_SUPPORT_COUNT {as_int(summary.get('source_support_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_TARGET_SUPPORT_COUNT {as_int(summary.get('target_support_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_TERM_COUNT {as_int(summary.get('lift_term_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_DEPENDENCY_TERMS {as_int(summary.get('dependency_backed_lift_term_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_ANALOGUE_TERMS {as_int(summary.get('analogue_backed_lift_term_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_OPEN_TERMS {as_int(summary.get('open_lift_term_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_LEAF8_SURFACE_HITS {as_int(summary.get('leaf8_surface_gate_unique_hit_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_LEAF12_SEEDS {as_int(summary.get('leaf12_seed_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_EXACT_SLOT_DIRECT_VERIFIED {as_int(summary.get('exact_target_slot_direct_verified_count'))}ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_ACCEPTED_EXPORT_COUNT 0ULL
#define SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_RELATION_DERIVED_ECDLP 0ULL

#endif
"""


def render_preflight_c(header_basename: str) -> str:
    return f"""#include <stdint.h>
#include <stdio.h>

#include "{header_basename}"

int main(void) {{
  uint64_t failure_count = 0;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_TRANSFER != {EXPECTED_TRANSFER}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_SOURCE_CERT_COUNT != {EXPECTED_SOURCE_CERT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_SOURCE_SUPPORT_COUNT != {EXPECTED_SOURCE_SUPPORT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_TARGET_SUPPORT_COUNT != {EXPECTED_TARGET_SUPPORT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_TERM_COUNT != {EXPECTED_LIFT_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_DEPENDENCY_TERMS != {EXPECTED_DEPENDENCY_BACKED_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_ANALOGUE_TERMS != {EXPECTED_ANALOGUE_BACKED_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_OPEN_TERMS != {EXPECTED_OPEN_TERM_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_LEAF8_SURFACE_HITS != {EXPECTED_LEAF8_SURFACE_HIT_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_LEAF12_SEEDS != {EXPECTED_LEAF12_SEED_COUNT}ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_EXACT_SLOT_DIRECT_VERIFIED != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_ACCEPTED_EXPORT_COUNT != 0ULL) failure_count++;
  if (SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_RELATION_DERIVED_ECDLP != 0ULL) failure_count++;

  printf("selected13_priority0_direct_certificate_lift_worklist_preflight transfer=%llu source_certs=%llu source_terms=%llu target_terms=%llu lift_terms=%llu dependency_terms=%llu analogue_terms=%llu open_terms=%llu leaf8_surface_hits=%llu leaf12_seeds=%llu accepted=%llu failures=%llu\\n",
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_TRANSFER,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_SOURCE_CERT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_SOURCE_SUPPORT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_TARGET_SUPPORT_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_TERM_COUNT,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_DEPENDENCY_TERMS,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_ANALOGUE_TERMS,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_OPEN_TERMS,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_LEAF8_SURFACE_HITS,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_LEAF12_SEEDS,
         (unsigned long long)SELECTED13_PRIORITY0_DIRECT_CERT_LIFT_ACCEPTED_EXPORT_COUNT,
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
    with tempfile.TemporaryDirectory(prefix="selected13_direct_cert_lift_", dir=str(temp_root)) as tmp:
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
    parser.add_argument("--target-gate", default=str(DEFAULT_TARGET_GATE))
    parser.add_argument("--direct-bridge", default=str(DEFAULT_DIRECT_BRIDGE))
    parser.add_argument("--dependency-bridge", default=str(DEFAULT_DEPENDENCY_BRIDGE))
    parser.add_argument("--gap-scanner", default=str(DEFAULT_GAP_SCANNER))
    parser.add_argument("--term812-scout", default=str(DEFAULT_TERM812_SCOUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--c-header-out", default=str(DEFAULT_C_HEADER_OUT))
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_worklist(
        load_json(Path(args.target_gate)),
        load_json(Path(args.direct_bridge)),
        load_json(Path(args.dependency_bridge)),
        load_json(Path(args.gap_scanner)),
        load_json(Path(args.term812_scout)),
    )
    header_path = Path(args.c_header_out)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(render_c_header(payload))
    payload["artifacts"] = {"c_header": str(header_path)}
    native_preflight = run_native_preflight(header_path, args.cc)
    payload["native_preflight"] = native_preflight
    if not native_preflight.get("verified"):
        payload["failures"].append({"code": "native_preflight_failed", "native_preflight": native_preflight})
        payload["claim_status"] = "SELECTED13_PRIORITY0_DIRECT_CERTIFICATE_LIFT_WORKLIST_FAILED"
    payload["summary"]["native_preflight_verified"] = bool(native_preflight.get("verified"))
    payload["summary"]["failure_count"] = len(payload["failures"])
    payload["summary"]["verified"] = not payload["failures"]
    write_json(Path(args.out), payload)
    print(json.dumps({"claim_status": payload["claim_status"], "out": args.out, "summary": payload["summary"]}, sort_keys=True))
    return 0 if not payload["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
