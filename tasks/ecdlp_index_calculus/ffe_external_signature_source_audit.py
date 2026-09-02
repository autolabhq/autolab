#!/usr/bin/env python3
"""Audit external low-term total-2 signature cases before FFE factorization.

The FFE quotient route currently closes the default 19 positive signature cases.
This probe compares that default bank to a broader signature file collected
from all public-leaf policy artifacts.  It does not factor new resultants.
Instead it pre-registers which external cases/surfaces are new and which are
already covered by the existing Sage-factor bank.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_DEFAULT_SIGNATURE_SOURCE = Path(
    "/Volumes/Volume/autolab/ecdlp_index_calculus_state/"
    "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_signature_probe.json"
)
DEFAULT_ALL_SIGNATURE_SOURCE = DEFAULT_STATE_DIR / "low_term_total2_signature_all_public_leaf_sources.json"
DEFAULT_SAGE_FACTOR_SOURCE = Path(
    "/Volumes/Volume/autolab/ecdlp_index_calculus_state/"
    "frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_probe.json"
)
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_external_signature_source_audit.json"
SEED = "ecdlp-frontier-signed-dual-sieve-v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def case_key(case: dict[str, Any]) -> tuple[Any, ...]:
    row_leaf_signature = tuple(
        (
            str(item.get("row_key")),
            tuple(int(leaf) for leaf in item.get("leaf_indices") or []),
        )
        for item in case.get("row_leaf_keys") or []
        if isinstance(item, dict)
    )
    return (
        str(case.get("target")),
        int(case.get("transfer_index") or 0),
        int(case.get("top_k") or 0),
        str(case.get("policy")),
        str(case.get("leaf_selector")),
        row_leaf_signature,
    )


def source_label(source: str) -> str:
    return Path(str(source)).name


def inferred_surface_id(case: dict[str, Any], row_key: str) -> str:
    target = str(case.get("target"))
    transfer_index = int(case.get("transfer_index") or 0)
    challenge_seed = f"{SEED}:shared-transfer:{transfer_index}:{target}"
    return f"{target}|{row_key}|{challenge_seed}"


def case_surfaces(case: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in case.get("row_leaf_keys") or []:
        if not isinstance(item, dict):
            continue
        row_key = str(item.get("row_key"))
        rows.append(
            {
                "row_key": row_key,
                "surface_id": inferred_surface_id(case, row_key),
                "leaf_indices": [int(leaf) for leaf in item.get("leaf_indices") or []],
            }
        )
    return rows


def summarize_cases(default_cases: list[dict[str, Any]], all_cases: list[dict[str, Any]], sage_source: dict[str, Any]) -> dict[str, Any]:
    default_keys = {case_key(case) for case in default_cases}
    default_surface_ids = {
        surface["surface_id"]
        for case in default_cases
        for surface in case_surfaces(case)
    }
    factored_surface_ids = {
        str(surface.get("surface_id"))
        for surface in sage_source.get("surfaces") or []
        if isinstance(surface, dict)
    }
    external_cases = [case for case in all_cases if case_key(case) not in default_keys]
    external_rows = []
    source_counts: Counter[str] = Counter()
    transfer_counts: Counter[int] = Counter()
    missing_surface_ids: set[str] = set()
    already_factored_surface_ids: set[str] = set()
    for case in external_cases:
        surfaces = case_surfaces(case)
        for surface in surfaces:
            if surface["surface_id"] in factored_surface_ids:
                already_factored_surface_ids.add(surface["surface_id"])
            else:
                missing_surface_ids.add(surface["surface_id"])
        source_counts[source_label(str(case.get("source")))] += 1
        transfer_counts[int(case.get("transfer_index") or 0)] += 1
        external_rows.append(
            {
                "target": case.get("target"),
                "transfer_index": int(case.get("transfer_index") or 0),
                "top_k": int(case.get("top_k") or 0),
                "ops_over_rho": case.get("ops_over_rho"),
                "policy": case.get("policy"),
                "row_selector": case.get("row_selector"),
                "leaf_selector": case.get("leaf_selector"),
                "source": source_label(str(case.get("source"))),
                "surfaces": [
                    {
                        **surface,
                        "already_in_default_signature_bank": surface["surface_id"] in default_surface_ids,
                        "already_factored": surface["surface_id"] in factored_surface_ids,
                    }
                    for surface in surfaces
                ],
            }
        )
    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in external_rows:
        rows_by_source[str(row["source"])].append(row)
    return {
        "default_case_count": len(default_cases),
        "all_case_count": len(all_cases),
        "external_case_count": len(external_cases),
        "default_surface_count": len(default_surface_ids),
        "factored_surface_count": len(factored_surface_ids),
        "external_surface_count": len({surface["surface_id"] for row in external_rows for surface in row["surfaces"]}),
        "external_already_factored_surface_count": len(already_factored_surface_ids),
        "external_pending_sage_factor_surface_count": len(missing_surface_ids),
        "external_case_source_counts": dict(source_counts.most_common()),
        "external_case_transfer_counts": dict(sorted(transfer_counts.items())),
        "external_pending_sage_factor_surfaces": sorted(missing_surface_ids),
        "external_already_factored_surfaces": sorted(already_factored_surface_ids),
        "external_cases_by_source": {
            source: sorted(
                rows,
                key=lambda row: (
                    int(row["transfer_index"]),
                    str(row["target"]),
                    float(row.get("ops_over_rho") or 10**9),
                ),
            )
            for source, rows in sorted(rows_by_source.items())
        },
        "interpretation": (
            "External cases are verifier-backed below-rho signatures found in "
            "public-leaf artifacts outside the default FFE signature source. "
            "Rows marked pending_sage_factor require true finite-field resultant "
            "factorization before the public FFE route can be scored."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-signature-source", type=Path, default=DEFAULT_DEFAULT_SIGNATURE_SOURCE)
    parser.add_argument("--all-signature-source", type=Path, default=DEFAULT_ALL_SIGNATURE_SOURCE)
    parser.add_argument("--sage-factor-source", type=Path, default=DEFAULT_SAGE_FACTOR_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    default_source = load_json(args.default_signature_source)
    all_source = load_json(args.all_signature_source)
    sage_source = load_json(args.sage_factor_source)
    default_cases = [case for case in default_source.get("positive_cases") or [] if isinstance(case, dict)]
    all_cases = [case for case in all_source.get("positive_cases") or [] if isinstance(case, dict)]
    output = {
        "schema": "ecdlp_low_term_total2_ffe_external_signature_source_audit_v1",
        "method": "compare_default_signature_bank_to_all_public_leaf_sources",
        "parameters": {
            "default_signature_source": str(args.default_signature_source),
            "all_signature_source": str(args.all_signature_source),
            "sage_factor_source": str(args.sage_factor_source),
        },
        "summary": summarize_cases(default_cases, all_cases, sage_source),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
