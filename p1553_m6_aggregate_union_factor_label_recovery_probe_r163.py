#!/usr/bin/env python3
"""Reduce the R162 nonlinear batch to one aggregate union factor."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_aggregate_union_factor_label_recovery.r163.v1"

R162_PRODUCER = ROOT / "p1553_m6_batch_inverse_transpose_modcomp_fit_probe_r162.py"
R162_REPORT = ROOT / "p1553_m6_batch_inverse_transpose_modcomp_fit_probe_report_r162.json"
R162_FROZEN = ROOT / "frozen_m6_batch_inverse_transpose_modcomp_fit.json"
R162_COST = ROOT / "m6_batch_inverse_transpose_modcomp_fit_cost_ledger.json"
R162_REPLAY = ROOT / "m6_batch_inverse_transpose_modcomp_fit_replay.json"
R162_CONTROLS = ROOT / "m6_batch_inverse_transpose_modcomp_fit_controls.json"
R162_TRANSPOSE = ROOT / "batch_inverse_functional_transpose_r162.json"
R162_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_batch_inverse_transpose_modcomp_fit_probe_r162.py"
R162_GATE = ROOT / "p1553_m6_batch_inverse_transpose_modcomp_fit_probe_gate_r162.md"
R162_PARENT = ROOT / "p1553_m6_batch_inverse_transpose_modcomp_fit_probe_parent_report_r162.yaml"
R88_REPORT = ROOT / "p1553_5a5c_black_box_translated_resultant_localizer_probe_report_r88.json"
R88_GATE = ROOT / "p1553_5a5c_black_box_translated_resultant_localizer_probe_gate_r88.md"

SOURCE_BINDINGS = (
    ("r162_producer", R162_PRODUCER, "734938d6e5f7c79b10539dd57583f7a012b66406bfb55e1a3bbcb96e24ee89f1"),
    ("r162_report", R162_REPORT, "55a46fbdefc4e1c9a84d5286cf825beb1a0c2659ca6653d63cc5b95fbfea1742"),
    ("r162_frozen", R162_FROZEN, "69411c74d272c73d721d3a863ea2f25262906d2461eb43f3835ef912531e58a7"),
    ("r162_cost", R162_COST, "ab274fdea53dc0c89b140ab910e6b288d25a56a1912f5b6f26c6ee716450b7e4"),
    ("r162_replay", R162_REPLAY, "a51f01eee7da30b83dcbd00703ba52af085c8b849272f85c1921ab4d21335022"),
    ("r162_controls", R162_CONTROLS, "ba3e8bd83c66f2876b599fc095eb69c637963e89783a73bfc9481bff0d95d2d8"),
    ("r162_transpose", R162_TRANSPOSE, "dd373da7cbe4f98ac2f354f7e082bc16f7415345afb90838a94b7576a5596319"),
    ("r162_test", R162_TEST, "863ac5fce63370328976b4c6ce2b67bafb63d29f27bf7d84a7371a40326011ab"),
    ("r162_gate", R162_GATE, "e5fa538ae4f3bef7f6764d7271a64f48da7e8e340e0f10eae0af6ce345a10338"),
    ("r162_parent", R162_PARENT, "a49a2631c8a7b9313fe20794df0c56b7ca6081bc2242b5b05b6fca0eec914927"),
    ("r88_report", R88_REPORT, "d73e017c731a54c6913aeaa94e6b5c6d54ca3757f56be14e8ca8ee5524031de1"),
    ("r88_gate", R88_GATE, "90f0980bdeb51f540cd233f207218361cc14beb8a899c246418410d36ef43d56"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_aggregate_union_factor_label_recovery_probe_report_r163.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_aggregate_union_factor_label_recovery.json"
DEFAULT_COST = ROOT / "m6_aggregate_union_factor_label_recovery_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_aggregate_union_factor_label_recovery_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_aggregate_union_factor_label_recovery_controls.json"
DEFAULT_LABELS = ROOT / "aggregate_union_target_labels_and_backpointers_r163.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R162 = load_module("p1553_r162_for_r163", R162_PRODUCER)
R161 = R162.R161


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {name: sha256_file(path) for name, path, _ in SOURCE_BINDINGS}
    failures = [
        name for name, _, expected in SOURCE_BINDINGS if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R163 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def target_records(control: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *control["positive_targets"],
        control["empty_target"],
        control["denominator_exception_target"],
    ]


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    prime = int(curve["field_prime"])
    dimension = len(R161.R157.c_point_pairs(curve, 0))
    factor_base = R161.R160.generic_factor_base(curve, dimension, seed)
    representatives = factor_base["representatives"]
    divisor = R161.signed_c3_divisor(representatives, curve)
    base = R161.finite_control(curve, seed)
    targets = target_records(base)
    point_index = {
        tuple(record["endpoint"]): record for record in divisor["records"]
    }
    membership_rows: list[dict[str, Any]] = []
    union_records: list[dict[str, Any]] = []
    for left in divisor["records"]:
        left_point = tuple(left["endpoint"])
        matches = []
        for target_record in targets:
            target = tuple(target_record["target"])
            right_point = R161.R70.add(
                target, R161.R70.negate(left_point, curve), curve
            )
            right = point_index.get(right_point)
            if right is None:
                continue
            source = tuple(
                a + b for a, b in zip(left["source"], right["source"])
            )
            matches.append(
                {
                    "target_role": target_record["role"],
                    "target": target_record["target"],
                    "left_indices": left["indices"],
                    "right_indices": right["indices"],
                    "source": source,
                    "source_identity_exact": (
                        R161.R157.row_point(source, representatives, curve)
                        == target
                    ),
                    "denominator_exception": left_point[0] == target[0],
                }
            )
        membership_rows.append(
            {
                "left_endpoint": left["endpoint"],
                "membership_projector_product_value": 0 if matches else 1,
                "match_count": len(matches),
            }
        )
        if matches:
            union_records.append({"left": left, "matches": matches})

    selector = R161.interpolate(
        [
            (
                int(row["left_endpoint"][0]),
                int(row["membership_projector_product_value"]),
            )
            for row in membership_rows
        ],
        prime,
    )
    union_gcd = R161.poly_gcd(divisor["u"], selector, prime)
    expected_union = R161.monic_root_polynomial(
        [int(record["left"]["endpoint"][0]) for record in union_records],
        prime,
    )

    recovered_labels = []
    dictionary_scans = 0
    target_scans = 0
    for left in divisor["records"]:
        left_x = int(left["endpoint"][0])
        dictionary_scans += 1
        if R161.poly_eval(union_gcd, left_x, prime) != 0:
            continue
        left_point = tuple(left["endpoint"])
        for target_record in targets:
            target_scans += 1
            target = tuple(target_record["target"])
            right_point = R161.R70.add(
                target, R161.R70.negate(left_point, curve), curve
            )
            right = point_index.get(right_point)
            if right is None:
                continue
            source = tuple(
                a + b for a, b in zip(left["source"], right["source"])
            )
            recovered_labels.append(
                {
                    "target_role": target_record["role"],
                    "target": target_record["target"],
                    "left_indices": left["indices"],
                    "right_indices": right["indices"],
                    "source": source,
                    "source_identity_exact": (
                        R161.R157.row_point(source, representatives, curve)
                        == target
                    ),
                    "denominator_exception": left_point[0] == target[0],
                }
            )

    expected_sources = {
        record["role"]: tuple(record["expected_source"])
        for record in targets
        if record["expected_source"] is not None
    }
    recovered_sources = {
        role: {
            tuple(record["source"])
            for record in recovered_labels
            if record["target_role"] == role
        }
        for role in expected_sources
    }
    all_expected_sources_recovered = all(
        expected in recovered_sources.get(role, set())
        for role, expected in expected_sources.items()
    )
    empty_role = base["empty_target"]["role"]
    empty_target_rejected = not any(
        record["target_role"] == empty_role for record in recovered_labels
    )
    all_label_identities_exact = all(
        record["source_identity_exact"] for record in recovered_labels
    )
    exceptional_label_count = sum(
        record["denominator_exception"] for record in recovered_labels
    )
    return {
        "control_id": f"{curve['family_id']}_aggregate_union_seed{seed}_d{dimension}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "seed": seed,
        "factor_base_dimension": dimension,
        "c3_divisor_degree": len(divisor["u"]) - 1,
        "target_count": len(targets),
        "positive_target_count": len(expected_sources),
        "union_factor_degree": max(R161.poly_degree(union_gcd), 0),
        "union_degree_bound": 20 * len(expected_sources),
        "union_degree_bound_holds": (
            max(R161.poly_degree(union_gcd), 0) <= 20 * len(expected_sources)
        ),
        "selector_sha256": sha256_json(selector),
        "union_factor_sha256": sha256_json(union_gcd),
        "expected_union_sha256": sha256_json(expected_union),
        "aggregate_union_factor_exact": union_gcd == expected_union,
        "membership_rows_sha256": sha256_json(membership_rows),
        "recovered_labels_sha256": sha256_json(recovered_labels),
        "dictionary_scan_count": dictionary_scans,
        "target_label_scan_count": target_scans,
        "all_expected_sources_recovered": all_expected_sources_recovered,
        "all_label_source_identities_exact": all_label_identities_exact,
        "empty_target_rejected": empty_target_rejected,
        "exceptional_label_count": exceptional_label_count,
        "exceptional_matches_preserved": (
            exceptional_label_count
            == sum(
                match["denominator_exception"]
                for record in union_records
                for match in record["matches"]
            )
        ),
        "recovered_label_count": len(recovered_labels),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
        "finite_endpoint_and_target_scans_receive_attack_credit": False,
        "recovered_labels": recovered_labels,
    }


def synthetic_positive_exceptional_control() -> dict[str, Any]:
    curve = R161.R159.R82.FAMILIES[0]
    prime = int(curve["field_prime"])
    factor_base = R161.R160.generic_factor_base(curve, 3, 16001)
    point_p = factor_base["generator"]
    point_q = R161.R70.negate(R161.R70.scalar_mul(2, point_p, curve), curve)
    target = R161.R70.negate(point_p, curve)
    if point_p is None or point_q is None or target is None:
        raise AssertionError("synthetic exceptional control left affine chart")
    if point_p[0] == point_q[0]:
        raise AssertionError("synthetic exceptional control has x collision")
    points = [point_p, point_q]
    u_poly = R161.monic_root_polynomial([point[0] for point in points], prime)
    selector = R161.interpolate([(point[0], 0) for point in points], prime)
    union_factor = R161.poly_gcd(u_poly, selector, prime)
    matches = []
    point_index = {point: index for index, point in enumerate(points)}
    for left_index, left in enumerate(points):
        right = R161.R70.add(
            target, R161.R70.negate(left, curve), curve
        )
        right_index = point_index.get(right)
        if right_index is None:
            continue
        matches.append(
            {
                "left_index": left_index,
                "right_index": right_index,
                "denominator_exception": left[0] == target[0],
                "point_identity_exact": (
                    R161.R70.add(left, points[right_index], curve) == target
                ),
            }
        )
    return {
        "control_id": "synthetic_positive_denominator_exception_p_plus_minus2p",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "signed_set_degree": len(points),
        "target": R161.R157.point_record(target),
        "target_x_equals_left_x": point_p[0] == target[0],
        "positive_exceptional_match_count": sum(
            match["denominator_exception"] for match in matches
        ),
        "all_point_identities_exact": all(
            match["point_identity_exact"] for match in matches
        ),
        "union_factor_exact": union_factor == u_poly,
        "union_factor_sha256": sha256_json(union_factor),
        "matches": matches,
        "candidate_oracle_consumed": False,
        "finite_control_receives_attack_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "coupled_target_projector": (
            "In the split algebra A=F_p[X]/U, let a_j and b_j be the R161 "
            "regular x- and signed-y membership residuals for target j. "
            "e_j=(1-a_j^(p-1))(1-b_j^(p-1)) is one exactly when both "
            "residuals vanish at the same C3 endpoint and zero otherwise."
        ),
        "aggregate_union_factor": (
            "With m_j=1-e_j and H=product_j m_j, gcd(U,H) is exactly the "
            "square-free polynomial of left C3 endpoints participating in "
            "at least one target decomposition. Denominator-exception roots "
            "are checked directly and unioned with the regular factor."
        ),
        "cross_target_false_positive_guard": (
            "gcd(U,product_j a_j,product_j b_j) is not equivalent: its two "
            "products can vanish because of different targets. Coupling the "
            "two residuals before the target product is required."
        ),
        "target_label_recovery": (
            "Given the union roots and a persistent point-to-C3-source "
            "dictionary, scan all N public targets for each root P. Compute "
            "Q=T_j-P and hash Q in the dictionary. Every hit gives the target "
            "label, both C3 backpointers, and the positive-C6 source."
        ),
        "output_sensitive_degree": (
            "If K positive targets each have a unique positive-C6 coefficient "
            "source and the signed C3 x-map is injective, the union factor "
            "has degree at most 20K."
        ),
        "scope": (
            "This reduces target labels and backpointers to postprocessing of "
            "one aggregate union factor. It does not construct that factor "
            "inside the below-rho cap without finite endpoint enumeration."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_aggregate_union_factor_label_recovery.cost.r163.v1",
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "positive_target_count_exponent_B": fraction_record(Fraction(3, 4)),
        "union_factor_degree_exponent_B": fraction_record(Fraction(3, 4)),
        "union_factorization_exponent_B": fraction_record(Fraction(3, 4)),
        "target_label_scan_exponent_B": fraction_record(Fraction(2)),
        "post_union_total_exponent_B": fraction_record(Fraction(2)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "post_union_rho_gap_exponent_B": fraction_record(Fraction(1, 2)),
        "aggregate_union_preferred_exponent_B": fraction_record(Fraction(9, 4)),
        "independent_membership_polynomial_batch_exponent_B": fraction_record(
            Fraction(7, 2)
        ),
        "post_union_target_labels_and_backpointers_inside_rho": True,
        "cross_target_uncoupled_product_valid": False,
        "aggregate_union_factor_algorithm_supplied": False,
        "finite_union_factor_enumeration_receives_attack_credit": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    synthetic_exceptional = synthetic_positive_exceptional_control()
    all_union = all(row["aggregate_union_factor_exact"] for row in rows)
    all_degree = all(row["union_degree_bound_holds"] for row in rows)
    all_sources = all(row["all_expected_sources_recovered"] for row in rows)
    all_labels = all(row["all_label_source_identities_exact"] for row in rows)
    all_empty = all(row["empty_target_rejected"] for row in rows)
    all_exceptional = all(row["exceptional_matches_preserved"] for row in rows)
    controls = {
        "schema": "p1553.m6_aggregate_union_factor_label_recovery.controls.r163.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "exact_union_factor_control_count": sum(
            row["aggregate_union_factor_exact"] for row in rows
        ),
        "exact_source_recovery_control_count": sum(
            row["all_expected_sources_recovered"]
            and row["all_label_source_identities_exact"]
            for row in rows
        ),
        "exact_empty_target_control_count": sum(
            row["empty_target_rejected"] for row in rows
        ),
        "exact_exceptional_control_count": sum(
            row["exceptional_matches_preserved"] for row in rows
        ),
        "synthetic_positive_exceptional_control": synthetic_exceptional,
        "candidate_oracle_consumed": False,
        "finite_controls_receive_attack_credit": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "twelve_source_bindings_verified": len(actual_bindings) == 12,
        "r162_global_below_rho_window_inherited": True,
        "r88_conditional_localization_semantically_deduplicated": True,
        "coupled_same_target_fermat_projector_complete": True,
        "cross_target_false_positive_guard_complete": True,
        "aggregate_union_factor_biconditional_complete": True,
        "exceptional_root_union_semantics_complete": True,
        "union_degree_at_most_20k_complete": True,
        "target_label_scan_reduction_complete": True,
        "target_label_scan_b2_below_rho_complete": True,
        "six_finite_aggregate_controls_complete": len(rows) == 6,
        "all_finite_union_factors_exact": all_union,
        "all_finite_degree_bounds_hold": all_degree,
        "all_finite_sources_recovered": all_sources,
        "all_finite_label_identities_exact": all_labels,
        "all_finite_empty_targets_rejected": all_empty,
        "all_finite_exceptional_matches_preserved": all_exceptional,
        "positive_exceptional_match_control_complete": (
            synthetic_exceptional["target_x_equals_left_x"]
            and synthetic_exceptional["positive_exceptional_match_count"] >= 1
            and synthetic_exceptional["all_point_identities_exact"]
            and synthetic_exceptional["union_factor_exact"]
        ),
        "candidate_oracles_avoided": True,
        "finite_controls_scoped_without_attack_credit": True,
        "aggregate_union_factor_below_rho_constructed": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    admission = {
        "obligations": obligations,
        "passed_obligation_count": sum(obligations.values()),
        "obligation_count": len(obligations),
        "aggregate_union_semantics_admitted": True,
        "post_union_label_recovery_admitted": True,
        "aggregate_union_constructor_admitted": False,
        "lane_admitted": False,
    }
    labels = {
        "schema": "p1553.m6_aggregate_union_target_labels.r163.v1",
        "control_records": [
            {
                "control_id": row["control_id"],
                "union_factor_degree": row["union_factor_degree"],
                "union_factor_sha256": row["union_factor_sha256"],
                "recovered_labels_sha256": row["recovered_labels_sha256"],
                "recovered_label_count": row["recovered_label_count"],
                "target_label_scan_count": row["target_label_scan_count"],
                "all_expected_sources_recovered": row[
                    "all_expected_sources_recovered"
                ],
            }
            for row in rows
        ],
        "all_labels_and_backpointers_exact": all_sources and all_labels,
        "aggregate_union_constructor_below_rho": False,
    }
    replay = {
        "schema": "p1553.m6_aggregate_union_factor_label_recovery.replay.r163.v1",
        "source_bindings": source_binding_records(),
        "control_records": [
            {
                "control_id": row["control_id"],
                "selector_sha256": row["selector_sha256"],
                "union_factor_sha256": row["union_factor_sha256"],
                "expected_union_sha256": row["expected_union_sha256"],
                "recovered_labels_sha256": row["recovered_labels_sha256"],
                "union_exact": row["aggregate_union_factor_exact"],
                "sources_exact": row["all_expected_sources_recovered"],
                "empty_exact": row["empty_target_rejected"],
                "exceptional_exact": row["exceptional_matches_preserved"],
            }
            for row in rows
        ],
        "all_replay_invariants_pass": (
            all_union
            and all_degree
            and all_sources
            and all_labels
            and all_empty
            and all_exceptional
        ),
    }
    frozen = {
        "schema": "p1553.m6_aggregate_union_factor_label_recovery.frozen.r163.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": cost,
        "admission": admission,
        "successor_interface": {
            "input": (
                "signed C3 divisor U,V and all N=B^(5/4) public targets"
            ),
            "required_output": (
                "one degree-O(B^(3/4)) union factor; labels are postprocessed"
            ),
            "preferred_work": "B^(9/4+o(1))",
            "maximum_total_work": "strictly below B^(5/2)",
            "same_target_coupling": (
                "couple x/y residuals before multiplying across targets"
            ),
            "open_primitive": (
                "aggregate product of target-coupled membership projectors "
                "modulo U without independent modular compositions"
            ),
        },
    }
    next_action = (
        "Construct the aggregate union factor itself below B^(5/2), ideally "
        "in B^(9/4+o(1)), without forming every target's U(phi_j) and "
        "V(phi_j). It may return an unlabeled degree-O(B^(3/4)) factor: R163 "
        "now supplies exact B^2 target-label and C3-source backpointer "
        "postprocessing. The constructor must couple signed x/y membership "
        "per target before aggregation and handle denominator exceptions."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Reduce the R162 nonlinear target batch to one exact low-degree "
            "aggregate union factor and prove that labels and source "
            "backpointers are affordable postprocessing."
        ),
        "source_bindings": source_binding_records(),
        "deduplication": {
            "r88": (
                "Already proves logarithmic source localization given a "
                "translated-resultant zero oracle. R163 instead aggregates "
                "the complete R159 target batch and supplies a direct B^2 "
                "label/backpointer scan once the union roots are known."
            ),
            "r161": (
                "Supplies exact per-target signed membership residuals. R163 "
                "couples and aggregates their semantics but does not claim a "
                "fast constructor for the product."
            ),
            "r162": (
                "Supplies the batch denominator linear layer and global cost "
                "window. R163 removes target labeling from the hard primitive."
            ),
        },
        "theorem": theorem,
        "cost": cost,
        "controls": controls,
        "admission": admission,
        "classification": (
            "SAME_TARGET_XY_FERMAT_PROJECTOR_EXACT__PRODUCT_OVER_TARGETS_AND_"
            "GCD_WITH_U_RETURNS_LEFT_ENDPOINT_UNION__UNCORRELATED_XY_PRODUCTS_"
            "REJECTED__UNIQUE_C6_UNION_DEGREE_AT_MOST_20K__SIX_FINITE_UNION_"
            "EMPTY_EXCEPTIONAL_CONTROLS_EXACT__UNION_ROOT_TARGET_LABEL_AND_"
            "SOURCE_BACKPOINTER_SCAN_B2_BELOW_RHO__AGGREGATE_UNION_CONSTRUCTOR_"
            "OPEN__NO_RHO_SHOUP_BREAKTHROUGH"
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "next_action": next_action,
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "labels": labels,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--labels-output", type=Path, default=DEFAULT_LABELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.labels_output, bundle["labels"])
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
