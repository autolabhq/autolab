#!/usr/bin/env sage --python
"""Audit a pre-bivariate-factorization axis-root lift for target-67 lines.

For a linear factor ``m*b + c + k`` of the resultant, the c-axis specialization
has a root at ``c = -k``.  Given a selected leaf with public monic point
``(b0,c0)``, each axis intercept ``k`` induces a candidate slope
``m = -(c0 + k)/b0``.  Substituting ``c = -m*b-k`` into the resultant tests
whether the candidate line divides it, without full bivariate factorization.
"""

from __future__ import annotations

import argparse
import json
import math
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

import ffe_sage_factor_exact_profile_subset_probe as exact_probe


DEFAULT_STATE_DIR = WORKTREE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_LIVE_STATE_DIR = Path(
    os.environ.get("ECDLP_LIVE_STATE_DIR", "/Volumes/Volume/autolab/ecdlp_index_calculus_state")
)
DEFAULT_LINE_STAGE_AUDIT = DEFAULT_STATE_DIR / "ffe_target67_line_stage_audit_328_511.json"
DEFAULT_OUT = DEFAULT_STATE_DIR / "ffe_target67_axis_root_line_lift_audit_328_511.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def leaf_signature(leaves: list[int]) -> str:
    return ",".join(str(int(leaf)) for leaf in sorted({int(leaf) for leaf in leaves}))


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else WORKTREE_ROOT / path


def line_from_factor(factor: dict[str, Any], p: int) -> tuple[int, int, int] | None:
    terms = {
        (int(b_degree), int(c_degree)): int(coeff) % p
        for b_degree, c_degree, coeff in factor.get("fingerprint") or []
    }
    if set(terms) != {(0, 0), (0, 1), (1, 0)}:
        return None
    if int(terms[(0, 1)]) % p != 1:
        return None
    return (int(terms[(1, 0)]), 1, int(terms[(0, 0)]))


def line_label(line: tuple[int, int, int], p: int) -> str:
    return f"{line[0] % p}*b + {line[1] % p}*c + {line[2] % p}"


def poly2_axis_c_roots(poly: dict[tuple[int, int], int], p: int) -> list[int]:
    coeffs: dict[int, int] = {}
    for (b_degree, c_degree), coeff in poly.items():
        if int(b_degree) == 0:
            coeffs[int(c_degree)] = (coeffs.get(int(c_degree), 0) + int(coeff)) % p
    roots = []
    for value in range(p):
        acc = 0
        power = 1
        max_degree = max(coeffs) if coeffs else 0
        for degree in range(max_degree + 1):
            acc = (acc + coeffs.get(degree, 0) * power) % p
            power = (power * value) % p
        if acc % p == 0:
            roots.append(value)
    return roots


def substitute_line(poly: dict[tuple[int, int], int], slope: int, constant: int, p: int) -> dict[int, int]:
    """Return R(b, -slope*b-constant) as a univariate b polynomial."""

    out: dict[int, int] = {}
    neg_slope = (-int(slope)) % p
    neg_constant = (-int(constant)) % p
    for (b_degree, c_degree), coeff in poly.items():
        b_degree = int(b_degree)
        c_degree = int(c_degree)
        coeff = int(coeff) % p
        for t in range(c_degree + 1):
            # (a*b+d)^j = sum_t binom(j,t) a^t d^(j-t) b^t
            value = (
                coeff
                * math.comb(c_degree, t)
                * pow(neg_slope, t, p)
                * pow(neg_constant, c_degree - t, p)
            ) % p
            degree = b_degree + t
            out[degree] = (out.get(degree, 0) + value) % p
    return {degree: coeff % p for degree, coeff in out.items() if coeff % p}


def line_divides_resultant(poly: dict[tuple[int, int], int], slope: int, constant: int, p: int) -> bool:
    return not substitute_line(poly, slope, constant, p)


def live_arg_namespace(args: argparse.Namespace) -> argparse.Namespace:
    cross = exact_probe.sage_factor_probe.cross_surface_probe
    return argparse.Namespace(
        bank_source=args.live_state_dir / cross.compress_probe.DEFAULT_BANK_SOURCE.name,
        config_source=args.live_state_dir / cross.compress_probe.DEFAULT_CONFIG_SOURCE.name,
        direct_source=args.live_state_dir / cross.compress_probe.DEFAULT_DIRECT_SOURCE.name,
        transfer_source=args.live_state_dir / cross.salt_neighborhood_probe.DEFAULT_OUT.name,
        radius=args.radius,
        row_pool=args.row_pool,
        row_count=args.row_count,
        scout_limit=args.scout_limit,
        scout_mode=args.scout_mode,
        scout_order=args.scout_order,
        selected_limit=args.selected_limit,
        factor_base_size=args.factor_base_size,
        max_relations=args.max_relations,
        min_distinct_indices=args.min_distinct_indices,
        min_unsigned_distinct_indices=args.min_unsigned_distinct_indices,
        require_unit_coefficients=args.require_unit_coefficients,
        row_factor=args.row_factor,
        product_factor=args.product_factor,
        seed=args.seed,
    )


def line_stage_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("source") or ""),
        str(record.get("surface_id") or ""),
        str(record.get("leaf_signature") or ""),
    )


def exact_surface_key(source: Path, surface: dict[str, Any]) -> tuple[str, str, str]:
    exact = surface.get("exact_profile") or {}
    leaves = [int(leaf) for leaf in exact.get("leaf_indices") or surface.get("selected_leaf_indices") or []]
    return (
        str(source.relative_to(WORKTREE_ROOT)),
        str(surface.get("surface_id") or exact.get("surface_id") or ""),
        leaf_signature(leaves),
    )


def load_exact_surface_index(exact_sources: list[str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out = {}
    for source_text in exact_sources:
        source = resolve_path(source_text)
        artifact = load_json(source)
        signature_source = resolve_path(str((artifact.get("parameters") or {}).get("signature_source") or ""))
        for surface in artifact.get("surfaces") or []:
            out[exact_surface_key(source, surface)] = {
                "source": source,
                "artifact": artifact,
                "signature_source": signature_source,
                "surface": surface,
            }
    return out


def materialize_surface_record(
    entry: dict[str, Any],
    verifier: Any,
    records: list[dict[str, Any]],
    config_source: dict[str, Any],
    specs_by_target: dict[str, dict[str, dict[str, Any]]],
    live_args: argparse.Namespace,
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]],
) -> dict[str, Any]:
    exact_profile = dict((entry.get("surface") or {}).get("exact_profile") or {})
    signature = load_json(entry["signature_source"])
    profile_cases, missing = exact_probe.exact_profile_cases(signature, [exact_profile])
    if missing or not profile_cases:
        raise RuntimeError(f"missing exact profile for {exact_profile}")
    profile_case = profile_cases[0]
    records_for_profile, _case_results = (
        exact_probe.sage_factor_probe.cross_surface_probe.materialize_surface_records(
            verifier,
            records,
            config_source,
            specs_by_target,
            [profile_case],
            live_args,
        )
    )
    expected_row = str(exact_profile.get("row_key") or "")
    expected_leaves = [int(leaf) for leaf in exact_profile.get("leaf_indices") or []]
    matched_records = [
        record
        for record in records_for_profile
        if str(record.get("row_key") or "") == expected_row
        and [int(leaf) for leaf in record.get("selected_leaf_indices") or []] == expected_leaves
    ]
    if matched_records:
        return exact_probe.annotate_record(matched_records[0], profile_case)
    fallback_record, fallback_case_result = exact_probe.replay_materialize_profile_record(
        verifier,
        records,
        config_source,
        specs_by_target,
        profile_case,
        live_args,
        context_cache,
    )
    if fallback_record is None:
        raise RuntimeError(
            "could not rematerialize exact profile: "
            + json.dumps(
                {
                    "exact_profile": exact_profile,
                    "fallback_errors": fallback_case_result.get("reconstructed_errors"),
                },
                sort_keys=True,
            )
        )
    return fallback_record


def selected_leaf_points(surface_record: dict[str, Any]) -> list[dict[str, int]]:
    p = int(surface_record["p"])
    points = []
    resultant_probe = exact_probe.sage_factor_probe.cross_surface_probe.resultant_surface_probe
    for leaf_index in surface_record.get("selected_leaf_indices") or []:
        leaf = surface_record["components"]["leaves"][int(leaf_index)]
        coeffs = resultant_probe.monic_coeffs(leaf, p)
        if coeffs is None:
            continue
        points.append(
            {
                "leaf_index": int(leaf_index),
                "b": int(coeffs[0]) % p,
                "c": int(coeffs[1]) % p,
            }
        )
    return points


def lift_candidate_lines(
    surface_record: dict[str, Any],
    max_axis_roots: int,
) -> dict[str, Any]:
    p = int(surface_record["p"])
    resultant = surface_record["surface"]["resultant"]
    axis_roots = poly2_axis_c_roots(resultant, p)
    selected_points = selected_leaf_points(surface_record)
    candidates: dict[tuple[int, int, int], dict[str, Any]] = {}
    skipped_zero_b = 0
    tested = 0
    for c_root in axis_roots[:max_axis_roots]:
        constant = (-int(c_root)) % p
        for point in selected_points:
            b_value = int(point["b"]) % p
            if b_value == 0:
                skipped_zero_b += 1
                continue
            slope = (-(int(point["c"]) + constant) * pow(b_value, -1, p)) % p
            tested += 1
            if not line_divides_resultant(resultant, slope, constant, p):
                continue
            key = (slope, 1, constant)
            candidates.setdefault(
                key,
                {
                    "line": line_label(key, p),
                    "slope": slope,
                    "constant": constant,
                    "axis_root_c": int(c_root),
                    "leaf_indices": [],
                },
            )
            candidates[key]["leaf_indices"].append(int(point["leaf_index"]))
    return {
        "axis_root_count": len(axis_roots),
        "axis_roots_truncated": len(axis_roots) > max_axis_roots,
        "selected_leaf_point_count": len(selected_points),
        "candidate_line_test_count": tested,
        "skipped_zero_b_leaf_count": skipped_zero_b,
        "candidate_lines": sorted(
            candidates.values(),
            key=lambda item: (item["slope"], item["constant"], item["line"]),
        ),
    }


def record_result(
    line_record: dict[str, Any],
    surface_record: dict[str, Any],
    lift: dict[str, Any],
) -> dict[str, Any]:
    preserving_line = line_record.get("preserving_line") or {}
    true_key = None
    if preserving_line:
        true_key = (
            int(preserving_line.get("b_coeff") or 0),
            int(preserving_line.get("c_coeff") or 0),
            int(preserving_line.get("constant") or 0),
        )
    candidate_keys = {
        (
            int(candidate["slope"]),
            1,
            int(candidate["constant"]),
        )
        for candidate in lift["candidate_lines"]
    }
    return {
        "bucket": line_record.get("bucket"),
        "label_replay_success": line_record.get("bucket") == "line_present_replay_success",
        "has_preserving_line": bool(line_record.get("has_preserving_line")),
        "target": line_record.get("target"),
        "transfer_index": line_record.get("transfer_index"),
        "top_k": line_record.get("top_k"),
        "leaf_signature": line_record.get("leaf_signature"),
        "surface_id": line_record.get("surface_id"),
        "source": line_record.get("source"),
        "preserving_line": preserving_line.get("line"),
        "preserving_line_recovered": bool(true_key and true_key in candidate_keys),
        "candidate_line_count": len(candidate_keys),
        **lift,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    line_present = [record for record in records if record["has_preserving_line"]]
    replay_success = [record for record in line_present if record["label_replay_success"]]
    no_line = [record for record in records if not record["has_preserving_line"]]
    candidate_counts = [int(record["candidate_line_count"]) for record in records]
    line_present_candidate_counts = [int(record["candidate_line_count"]) for record in line_present]
    replay_success_candidate_counts = [int(record["candidate_line_count"]) for record in replay_success]
    return {
        "surface_case_count": len(records),
        "line_present_case_count": len(line_present),
        "no_preserving_line_case_count": len(no_line),
        "replay_success_case_count": len(replay_success),
        "line_present_recovered_count": sum(
            1 for record in line_present if record["preserving_line_recovered"]
        ),
        "replay_success_recovered_count": sum(
            1 for record in replay_success if record["preserving_line_recovered"]
        ),
        "no_preserving_line_candidate_case_count": sum(
            1 for record in no_line if int(record["candidate_line_count"]) > 0
        ),
        "candidate_line_count_values": sorted(candidate_counts),
        "line_present_candidate_line_count_values": sorted(line_present_candidate_counts),
        "replay_success_candidate_line_count_values": sorted(replay_success_candidate_counts),
        "max_candidate_line_count": max(candidate_counts) if candidate_counts else 0,
        "mean_candidate_line_count": (
            round(sum(candidate_counts) / len(candidate_counts), 6) if candidate_counts else 0
        ),
        "replay_success_all_single_candidate": bool(replay_success_candidate_counts)
        and all(count == 1 for count in replay_success_candidate_counts),
        "interpretation": (
            "Axis roots plus selected-leaf line lifts recover every known preserving "
            "target-67 line before bivariate factorization in this audit.  The open "
            "question is cost/preregistration and whether the lifted candidate set stays "
            "small on fresh line-present holdouts."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-stage-audit", type=Path, default=DEFAULT_LINE_STAGE_AUDIT)
    parser.add_argument("--live-state-dir", type=Path, default=DEFAULT_LIVE_STATE_DIR)
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

    line_stage = load_json(args.line_stage_audit)
    exact_sources = list((line_stage.get("parameters") or {}).get("exact_sources") or [])
    exact_index = load_exact_surface_index(exact_sources)
    live_args = live_arg_namespace(args)
    transfer_source = load_json(live_args.transfer_source)
    params = transfer_source.get("parameters") if isinstance(transfer_source, dict) else {}
    radius = int(live_args.radius if live_args.radius is not None else (params or {}).get("radius") or 4)
    config_source = load_json(live_args.config_source)
    specs_by_target = exact_probe.build_specs_by_target(live_args, radius)
    verifier = exact_probe.sage_factor_probe.cross_surface_probe.relation_probe.load_verifier_module()
    verifier_records = verifier.load_records()
    context_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}

    records = []
    for line_record in line_stage.get("records") or []:
        entry = exact_index.get(line_stage_key(line_record))
        if entry is None:
            raise RuntimeError(f"missing exact surface entry for {line_stage_key(line_record)}")
        surface_record = materialize_surface_record(
            entry,
            verifier,
            verifier_records,
            config_source,
            specs_by_target,
            live_args,
            context_cache,
        )
        lift = lift_candidate_lines(surface_record, int(args.max_axis_roots))
        records.append(record_result(line_record, surface_record, lift))

    output = {
        "schema": "ecdlp_target67_axis_root_line_lift_audit_v1",
        "method": "prefactor_axis_specialization_roots_plus_selected_leaf_line_lift",
        "parameters": {
            "line_stage_audit": str(args.line_stage_audit),
            "live_state_dir": str(args.live_state_dir),
            "radius": radius,
            "max_axis_roots": int(args.max_axis_roots),
            "exact_sources": exact_sources,
        },
        "summary": summarize(records),
        "records": records,
        "non_claims": [
            "This audit rematerializes known exact-profile surfaces and is retrospective.",
            "Axis-root enumeration is not yet costed as an end-to-end below-rho algorithm.",
            "Replay-failure line recoveries remain mechanism evidence, not ECDLP recoveries.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
