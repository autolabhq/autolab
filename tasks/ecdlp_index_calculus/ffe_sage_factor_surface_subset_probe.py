#!/usr/bin/env sage --python
"""Sage-factor only selected FFE surfaces from a signature source.

This is a narrow follow-up to the external-signature audit.  The live Sage
factor probe can materialize and factor every surface from a signature source;
for broad public-leaf banks that can be too expensive.  This wrapper reuses the
live evaluator but filters to an explicit set of surface IDs before invoking
finite-field factorization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


CAMPAIGN_TASK_DIR = Path(
    os.environ.get("ECDLP_TASK_DIR", "/Volumes/Volume/autolab/tasks/ecdlp_index_calculus")
).resolve()
if str(CAMPAIGN_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_TASK_DIR))

import frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_probe as sage_factor_probe


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_SIGNATURE_SOURCE = DEFAULT_STATE_DIR / "low_term_total2_signature_wide_external_only.json"
DEFAULT_AUDIT_SOURCE = DEFAULT_STATE_DIR / "ffe_external_signature_source_audit.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_sage_factor_wide_external_subset.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def pending_surface_ids(audit_source: dict[str, Any]) -> set[str]:
    summary = audit_source.get("summary") or {}
    return {
        str(surface_id)
        for surface_id in summary.get("external_pending_sage_factor_surfaces") or []
        if surface_id
    }


def explicit_surface_ids(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-source", type=Path, default=DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--audit-source", type=Path, default=DEFAULT_AUDIT_SOURCE)
    parser.add_argument("--surface-ids")
    parser.add_argument("--state-dir", type=Path, default=sage_factor_probe.DEFAULT_STATE_DIR)
    parser.add_argument("--bank-source", type=Path, default=sage_factor_probe.cross_surface_probe.compress_probe.DEFAULT_BANK_SOURCE)
    parser.add_argument("--config-source", type=Path, default=sage_factor_probe.cross_surface_probe.compress_probe.DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--direct-source", type=Path, default=sage_factor_probe.cross_surface_probe.compress_probe.DEFAULT_DIRECT_SOURCE)
    parser.add_argument("--transfer-source", type=Path, default=sage_factor_probe.cross_surface_probe.compress_probe.DEFAULT_TRANSFER_SOURCE)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--row-pool", type=int, default=512)
    parser.add_argument("--row-count", type=int, default=128)
    parser.add_argument("--scout-limit", type=int, default=192)
    parser.add_argument("--scout-mode", default="s3_coeff_spread")
    parser.add_argument("--scout-order", default="eval_cover_hits_high")
    parser.add_argument("--selected-limit", type=int, default=64)
    parser.add_argument("--factor-base-size", type=int, default=16)
    parser.add_argument("--max-relations", type=int, default=96)
    parser.add_argument("--min-distinct-indices", type=int, default=4)
    parser.add_argument("--min-unsigned-distinct-indices", type=int, default=2)
    parser.add_argument("--allow-combined-coefficients", dest="require_unit_coefficients", action="store_false")
    parser.set_defaults(require_unit_coefficients=True)
    parser.add_argument("--row-factor", type=int, default=512)
    parser.add_argument("--product-factor", type=int, default=4096)
    parser.add_argument("--seed", default="ecdlp-frontier-signed-dual-sieve-v1")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    requested_surface_ids = explicit_surface_ids(args.surface_ids)
    if not requested_surface_ids:
        requested_surface_ids = pending_surface_ids(load_json(args.audit_source))
    if not requested_surface_ids:
        raise SystemExit("no surface ids provided and audit source has no pending surface ids")

    signature = load_json(args.signature_source)
    positive_cases = [case for case in signature.get("positive_cases") or [] if isinstance(case, dict)]
    if args.max_cases and args.max_cases > 0:
        positive_cases = positive_cases[: args.max_cases]

    bank = load_json(args.bank_source)
    config_source = load_json(args.config_source)
    direct_source = load_json(args.direct_source)
    transfer_source = load_json(args.transfer_source)
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    radius = int(args.radius if args.radius is not None else (params or {}).get("radius") or 4)
    bank_rows = {
        sage_factor_probe.cross_surface_probe.compress_probe.row_key(row): row
        for row in bank.get("bank_rows") or []
        if isinstance(row, dict) and sage_factor_probe.cross_surface_probe.compress_probe.row_key(row)
    }
    specs_by_target = sage_factor_probe.cross_surface_probe.leaf_trim_probe.specs_by_target_and_key(
        sage_factor_probe.cross_surface_probe.salt_neighborhood_probe.witness_specs(
            direct_source,
            bank_rows,
            radius,
        )
    )
    verifier = sage_factor_probe.cross_surface_probe.relation_probe.load_verifier_module()
    records = verifier.load_records()
    surface_records, case_results = sage_factor_probe.cross_surface_probe.materialize_surface_records(
        verifier,
        records,
        config_source,
        specs_by_target,
        positive_cases,
        args,
    )
    selected_records = [
        record for record in surface_records if str(record.get("surface_id")) in requested_surface_ids
    ]
    missing = sorted(requested_surface_ids - {str(record.get("surface_id")) for record in selected_records})
    surfaces = [sage_factor_probe.evaluate_surface(surface_record) for surface_record in selected_records]
    output = {
        "schema": "ecdlp_low_term_total2_ffe_sage_factor_surface_subset_probe_v1",
        "method": "sage_backed_factorization_for_selected_external_surfaces",
        "parameters": {
            "signature_source": str(args.signature_source),
            "audit_source": str(args.audit_source),
            "requested_surface_ids": sorted(requested_surface_ids),
            "radius": radius,
            "seed": args.seed,
        },
        "summary": {
            **sage_factor_probe.summarize(surfaces, case_results),
            "requested_surface_count": len(requested_surface_ids),
            "selected_surface_count": len(selected_records),
            "missing_requested_surface_count": len(missing),
            "missing_requested_surface_ids": missing,
        },
        "surfaces": surfaces,
        "cases": case_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
