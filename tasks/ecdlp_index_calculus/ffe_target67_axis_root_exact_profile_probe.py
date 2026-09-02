#!/usr/bin/env sage --python
"""Run axis-root line lifts on arbitrary target-67 exact-profile artifacts.

The first axis-root audit was tied to the historical target-67 line-stage
ledger, which is useful for recovery accounting but awkward for fresh
near-miss diagnostics.  This probe takes exact-profile Sage artifacts directly,
rematerializes each row/leaf profile, and reports the pre-factor candidate
lines emitted by c-axis roots plus selected-leaf line lifts.
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

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import ffe_target67_axis_root_line_lift_audit as axis_lift


DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_target67_axis_root_exact_profile_probe.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else WORKTREE_ROOT / path


def compact_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 8)
    except (TypeError, ValueError):
        return None


def best_candidate(surface: dict[str, Any]) -> dict[str, Any]:
    candidate = surface.get("best_preserving_candidate")
    if isinstance(candidate, dict):
        return candidate
    candidates = [
        item for item in surface.get("preserving_candidates") or [] if isinstance(item, dict)
    ]
    if not candidates:
        return {}
    return min(
        candidates,
        key=lambda item: (
            float(item.get("factor_root_scan_ops_over_rho") or 10**9),
            int(item.get("factor_monomials") or 10**9),
            str(item.get("candidate_name") or ""),
        ),
    )


def exact_sources(args: argparse.Namespace) -> list[Path]:
    if args.exact_source:
        return [resolve_path(source) for source in args.exact_source]
    return [resolve_path(DEFAULT_STATE_DIR / "ffe_sage_factor_exact_profiles_target67_transfer514_targetcap3_512_519.json")]


def generic_rho(surface: dict[str, Any]) -> int:
    candidate = best_candidate(surface)
    value = candidate.get("generic_rho_steps")
    if value is not None:
        return int(value)
    p = int(surface.get("p") or 0)
    if p == 9803:
        return 125
    return 125


def source_case(surface: dict[str, Any]) -> dict[str, Any]:
    cases = [case for case in surface.get("source_cases") or [] if isinstance(case, dict)]
    return cases[0] if cases else {}


def record_for_surface(
    exact_source: Path,
    exact_artifact: dict[str, Any],
    surface: dict[str, Any],
    args: argparse.Namespace,
    verifier: Any,
    verifier_records: list[dict[str, Any]],
    config_source: dict[str, Any],
    specs_by_target: dict[str, dict[str, dict[str, Any]]],
    live_args: argparse.Namespace,
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]],
) -> dict[str, Any]:
    signature_source = resolve_path(
        str((exact_artifact.get("parameters") or {}).get("signature_source") or "")
    )
    entry = {
        "source": exact_source,
        "artifact": exact_artifact,
        "signature_source": signature_source,
        "surface": surface,
    }
    surface_record = axis_lift.materialize_surface_record(
        entry,
        verifier,
        verifier_records,
        config_source,
        specs_by_target,
        live_args,
        context_cache,
    )
    lift = axis_lift.lift_candidate_lines(surface_record, int(args.max_axis_roots))
    rho = generic_rho(surface)
    candidate = best_candidate(surface)
    source = source_case(surface)
    line_lift_ops = int(lift["axis_root_count"]) + int(lift["candidate_line_test_count"])
    return {
        "exact_source": str(exact_source.relative_to(WORKTREE_ROOT)),
        "surface_id": surface.get("surface_id"),
        "surface_profile_id": surface.get("surface_profile_id"),
        "target": surface.get("target"),
        "row_key": surface.get("row_key"),
        "transfer_index": int((surface.get("exact_profile") or {}).get("transfer_index") or source.get("transfer_index") or 0),
        "top_k": int((surface.get("exact_profile") or {}).get("top_k") or source.get("top_k") or 0),
        "policy": (surface.get("exact_profile") or {}).get("policy") or source.get("policy"),
        "leaf_selector": (surface.get("exact_profile") or {}).get("leaf_selector") or source.get("leaf_selector"),
        "selected_leaf_indices": [int(leaf) for leaf in surface.get("selected_leaf_indices") or []],
        "source_ops_over_rho": compact_float(source.get("source_ops_over_rho")),
        "selected_root_pair_count": int(candidate.get("selected_root_pair_count") or 0),
        "selected_missed_leaves": int(candidate.get("selected_missed_leaves") or 0),
        "preserving_factor_total_degree": candidate.get("factor_total_degree"),
        "preserving_factor_root_scan_ops_over_rho": compact_float(
            candidate.get("factor_root_scan_ops_over_rho")
        ),
        "preserving_factor_surface_ffe_ops_over_rho": compact_float(
            candidate.get("surface_ffe_ops_over_rho")
        ),
        "full_remainder_ffe_ops_over_rho": compact_float(
            candidate.get("full_remainder_ffe_ops_over_rho")
        ),
        "axis_root_count": int(lift["axis_root_count"]),
        "axis_roots_truncated": bool(lift["axis_roots_truncated"]),
        "selected_leaf_point_count": int(lift["selected_leaf_point_count"]),
        "candidate_line_test_count": int(lift["candidate_line_test_count"]),
        "line_lift_ops": line_lift_ops,
        "line_lift_ops_over_rho": round(line_lift_ops / rho, 8),
        "candidate_line_count": len(lift["candidate_lines"]),
        "candidate_lines": lift["candidate_lines"],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected_positive = [record for record in records if record["selected_root_pair_count"] > 0]
    counts = [int(record["candidate_line_count"]) for record in records]
    positive_counts = [int(record["candidate_line_count"]) for record in selected_positive]
    return {
        "surface_count": len(records),
        "selected_root_positive_surface_count": len(selected_positive),
        "candidate_line_count_values": sorted(counts),
        "selected_root_positive_candidate_line_count_values": sorted(positive_counts),
        "singleton_candidate_surface_count": sum(1 for count in counts if count == 1),
        "selected_root_positive_singleton_candidate_count": sum(
            1 for count in positive_counts if count == 1
        ),
        "line_lift_ops_over_rho_values": sorted(
            record["line_lift_ops_over_rho"] for record in records
        ),
        "min_line_lift_ops_over_rho": min(
            [record["line_lift_ops_over_rho"] for record in records] or [None]
        ),
        "max_line_lift_ops_over_rho": max(
            [record["line_lift_ops_over_rho"] for record in records] or [None]
        ),
        "interpretation": (
            "This is a fresh exact-profile diagnostic for the axis-root lift. "
            "It reports candidate-line enumeration before full bivariate factor "
            "selection, but it does not by itself prove an ECDLP speedup."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source", type=Path, action="append")
    parser.add_argument("--live-state-dir", type=Path, default=axis_lift.DEFAULT_LIVE_STATE_DIR)
    parser.add_argument("--radius", type=int)
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
    parser.add_argument("--max-axis-roots", type=int, default=128)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    live_args = axis_lift.live_arg_namespace(args)
    transfer_source = load_json(live_args.transfer_source)
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    radius = int(live_args.radius if live_args.radius is not None else (params or {}).get("radius") or 4)
    config_source = load_json(live_args.config_source)
    specs_by_target = axis_lift.exact_probe.build_specs_by_target(live_args, radius)
    verifier = axis_lift.exact_probe.sage_factor_probe.cross_surface_probe.relation_probe.load_verifier_module()
    verifier_records = verifier.load_records()
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}

    records = []
    sources = exact_sources(args)
    for exact_source in sources:
        artifact = load_json(exact_source)
        for surface in artifact.get("surfaces") or []:
            records.append(
                record_for_surface(
                    exact_source,
                    artifact,
                    surface,
                    args,
                    verifier,
                    verifier_records,
                    config_source,
                    specs_by_target,
                    live_args,
                    context_cache,
                )
            )

    output = {
        "schema": "ecdlp_target67_axis_root_exact_profile_probe_v1",
        "method": "axis_root_line_lift_for_fresh_exact_profiles",
        "parameters": {
            "exact_sources": [str(source) for source in sources],
            "live_state_dir": str(args.live_state_dir),
            "radius": radius,
            "max_axis_roots": int(args.max_axis_roots),
        },
        "summary": summarize(records),
        "records": records,
        "non_claims": [
            "This is a diagnostic over exact-profile artifacts, not a public row-selector win.",
            "It uses rematerialized exact profiles to expose resultant structure.",
            "A below-rho line-lift proxy must still be combined with public branch reach and replay.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
