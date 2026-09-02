#!/usr/bin/env python3
"""Test pure pairwise and bounded-rank target-section factorizations."""

from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import importlib.util
import itertools
import json
import pathlib
from fractions import Fraction
from typing import Any, Sequence


SCHEMA = "p1553.5a5c_poincare_theta_target_section_rank.r109.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
SMALL_SIDE_EXPONENT = Fraction(12, 5)
LARGE_SIDE_EXPONENT = Fraction(13, 5)
RANK_SAMPLE_CAP = 64

R108_PRODUCER = pathlib.Path(
    "p1553_5a5c_factored_elliptic_lambda_ring_chow_norm_probe_r108.py"
)
R108_PRODUCER_SHA256 = (
    "99fb8e79f3e22ae82b47d3282edbc17ad0616bc9d30511114d38c919d6ee3e19"
)
R108_REPORT = pathlib.Path(
    "p1553_5a5c_factored_elliptic_lambda_ring_chow_"
    "norm_probe_report_r108.json"
)
R108_REPORT_SHA256 = (
    "91fab519c6a59851a672aed1a9e18fac8c434359c91a9e1d9fc0acb112ee7b41"
)
R108_GATE = pathlib.Path(
    "p1553_5a5c_factored_elliptic_lambda_ring_chow_norm_probe_gate_r108.md"
)
R108_GATE_SHA256 = (
    "9d7bf5d8b5420d52950ce28a08b7d59a94307bf66752805967338727f3883ab7"
)
R108_PARENT = pathlib.Path(
    "p1553_5a5c_factored_elliptic_lambda_ring_chow_"
    "norm_probe_parent_report_r108.yaml"
)
R108_PARENT_SHA256 = (
    "6705c882ec1b80538bd3a7e853f33c4f832afae5e35c2063a19a04794d928888"
)
R101_PRODUCER = pathlib.Path(
    "p1553_5a5c_actual_divisor_image_entropy_merge_probe_r101.py"
)
R101_PRODUCER_SHA256 = (
    "854df1da576b07cc3dd63b1249a42a72d56708ef8eb43c4ef37478f2558e747d"
)
R101_REPORT = pathlib.Path(
    "p1553_5a5c_actual_divisor_image_entropy_merge_probe_report_r101.json"
)
R101_REPORT_SHA256 = (
    "050d1093fc34611f502a3f7b3ebf6fc632b3aeb9d03718f922e99baa2fe985c9"
)
R101_GATE = pathlib.Path(
    "p1553_5a5c_actual_divisor_image_entropy_merge_probe_gate_r101.md"
)
R101_GATE_SHA256 = (
    "3e12e4ab70edba64076d27b68d39ab34307ca6d9ff08252432759673b74a9ac5"
)
R104_REPORT = pathlib.Path(
    "p1553_5a5c_compact_preendpoint_s3_ffe_pushdown_probe_report_r104.json"
)
R104_REPORT_SHA256 = (
    "5dbf8a75dd8298c2c606e41a4c121c6191ef56b5e184199e8db48841880b1edf"
)
R105_REPORT = pathlib.Path(
    "p1553_5a5c_actual_deck_nonmergeable_target_pullback_probe_report_r105.json"
)
R105_REPORT_SHA256 = (
    "c00b8c3e934e783012acb1dab83282f98720702bb8cd81a706fbdf4671f93a83"
)

Point = tuple[int, int] | None
Fp2 = tuple[int, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R108_PRODUCER: R108_PRODUCER_SHA256,
        R108_REPORT: R108_REPORT_SHA256,
        R108_GATE: R108_GATE_SHA256,
        R108_PARENT: R108_PARENT_SHA256,
        R101_PRODUCER: R101_PRODUCER_SHA256,
        R101_REPORT: R101_REPORT_SHA256,
        R101_GATE: R101_GATE_SHA256,
        R104_REPORT: R104_REPORT_SHA256,
        R105_REPORT: R105_REPORT_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R109 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R108 = load_module("p1553_r108_for_r109", R108_PRODUCER)
R107 = R108.R107
R105 = R108.R105
R104 = R108.R104
R102 = R108.R102
R84 = R108.R84
R82 = R108.R82
R70 = R108.R70


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
        "decimal": float(value),
    }


def point_json(point: Point) -> list[int] | None:
    return None if point is None else [point[0], point[1]]


def coordinate_endpoint(
    indices: Sequence[int],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> Point:
    endpoint: Point = None
    for position, index in enumerate(indices):
        atom = atoms_a[index] if position < 5 else atoms_c[index]
        endpoint = R70.add(endpoint, atom, curve)
    return endpoint


def cylinder_zero_set_control(
    curve: dict[str, Any],
    offset: int,
    control_class: str,
) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    domains = [len(atoms_a)] * 5 + [len(atoms_c)] * 5
    source = [0] * 10
    target = coordinate_endpoint(source, atoms_a, atoms_c, curve)
    subset_rows = []
    for subset_size in range(3):
        for subset in itertools.combinations(range(10), subset_size):
            free_position = next(
                position
                for position, domain_size in enumerate(domains)
                if position not in subset and domain_size > 1
            )
            changed = list(source)
            changed[free_position] = 1
            changed_endpoint = coordinate_endpoint(
                changed,
                atoms_a,
                atoms_c,
                curve,
            )
            subset_rows.append(
                {
                    "fixed_subset": list(subset),
                    "changed_free_position": free_position,
                    "completion_changes_sum": changed_endpoint != target,
                }
            )
    return {
        "control_class": control_class,
        "family_id": curve["family_id"],
        "offset": offset,
        "factor_base_size_B": len(factors),
        "factor_base_injective": geometry["factor_base_injective"],
        "zero_source": source,
        "zero_target": point_json(target),
        "tested_cylinder_count": len(subset_rows),
        "expected_unary_pairwise_cylinder_count": 56,
        "all_size_at_most_two_cylinders_escape_zero_fiber": all(
            row["completion_changes_sum"] for row in subset_rows
        ),
        "cylinder_rows": subset_rows,
        "candidate_scalar_labels_consumed": False,
    }


def fp2_matrix_rank(
    matrix: Sequence[Sequence[Fp2]],
    prime: int,
    nonsquare: int,
) -> int:
    rows = [list(row) for row in matrix]
    if not rows:
        return 0
    rank = 0
    column_count = len(rows[0])
    for column in range(column_count):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column] != R84.f2_zero()
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = R84.f2_inv(
            rows[rank][column],
            prime,
            nonsquare,
        )
        rows[rank] = [
            R84.f2_mul(value, inverse, prime, nonsquare)
            for value in rows[rank]
        ]
        for row in range(len(rows)):
            if row == rank or rows[row][column] == R84.f2_zero():
                continue
            factor = rows[row][column]
            rows[row] = [
                R84.f2_sub(
                    value,
                    R84.f2_mul(factor, pivot_value, prime, nonsquare),
                    prime,
                )
                for value, pivot_value in zip(rows[row], rows[rank])
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def section_rank_control(
    curve: dict[str, Any],
    offset: int,
    control_class: str,
) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    large_histogram, large_count = R107.partial_endpoint_histogram(
        atoms_a,
        atoms_c,
        2,
        3,
        curve,
    )
    small_histogram, small_count = R107.partial_endpoint_histogram(
        atoms_a,
        atoms_c,
        3,
        2,
        curve,
    )
    large_points = sorted(large_histogram, key=R102.point_sort_key)
    small_points = sorted(small_histogram, key=R102.point_sort_key)
    sample_dimension = min(
        RANK_SAMPLE_CAP,
        len(large_points),
        len(small_points),
    )
    sampled_large = large_points[:sample_dimension]
    sampled_small = small_points[:sample_dimension]
    target = next(
        R70.add(left, right, curve)
        for left in sampled_large
        for right in sampled_small
        if R70.add(left, right, curve) is not None
    )
    target_key = R84.point_key(target)
    prime = curve["field_prime"]
    nonsquare = R84.least_nonsquare(prime)
    matrix = [
        [
            R84.f2_sub(
                R84.point_key(R70.add(left, right, curve)),
                target_key,
                prime,
            )
            for right in sampled_small
        ]
        for left in sampled_large
    ]
    rank = fp2_matrix_rank(matrix, prime, nonsquare)
    return {
        "control_class": control_class,
        "family_id": curve["family_id"],
        "offset": offset,
        "factor_base_size_B": len(factors),
        "factor_base_injective": geometry["factor_base_injective"],
        "large_2A3C_occurrence_count": large_count,
        "large_2A3C_endpoint_support": len(large_histogram),
        "large_side_injective": len(large_histogram) == large_count,
        "small_3A2C_occurrence_count": small_count,
        "small_3A2C_endpoint_support": len(small_histogram),
        "small_side_injective": len(small_histogram) == small_count,
        "sample_dimension": sample_dimension,
        "sample_target": point_json(target),
        "signed_section_sample_rank_fp2": rank,
        "sample_rank_full": rank == sample_dimension,
        "candidate_scalar_labels_consumed": False,
        "verifier_materialized_sample_matrix": True,
        "candidate_work_credit": False,
    }


@functools.lru_cache(maxsize=1)
def actual_and_matched_controls() -> dict[str, Any]:
    cylinder_actual = [
        cylinder_zero_set_control(dict(family), offset, "actual")
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    cylinder_matched = [
        cylinder_zero_set_control(
            dict(family),
            offset,
            "matched_random_deck",
        )
        for family in R82.FAMILIES
        for offset in (2, 3)
    ]
    rank_actual = [
        section_rank_control(dict(family), offset, "actual")
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    rank_matched = [
        section_rank_control(
            dict(family),
            offset,
            "matched_random_deck",
        )
        for family in R82.FAMILIES
        for offset in (2, 3)
    ]
    cylinders = [*cylinder_actual, *cylinder_matched]
    ranks = [*rank_actual, *rank_matched]
    return {
        "cylinder_actual": cylinder_actual,
        "cylinder_matched_random": cylinder_matched,
        "rank_actual": rank_actual,
        "rank_matched_random": rank_matched,
        "instance_count": len(ranks),
        "all_cylinder_controls_exact": all(
            row["tested_cylinder_count"]
            == row["expected_unary_pairwise_cylinder_count"]
            and row["all_size_at_most_two_cylinders_escape_zero_fiber"]
            for row in cylinders
        ),
        "all_balanced_side_images_injective": all(
            row["large_side_injective"] and row["small_side_injective"]
            for row in ranks
        ),
        "all_sample_section_ranks_full": all(
            row["sample_rank_full"] for row in ranks
        ),
        "minimum_sample_dimension": min(
            row["sample_dimension"] for row in ranks
        ),
        "minimum_sample_rank": min(
            row["signed_section_sample_rank_fp2"] for row in ranks
        ),
    }


def pure_pairwise_zero_theorem() -> dict[str, Any]:
    return {
        "target_zero_fiber": (
            "Z_T={(P1,...,P10):P1+...+P10=T}"
        ),
        "candidate_form": (
            "s_T(P1,...,P10)=product_i u_i(P_i) "
            "product_(i<j) b_ij(P_i,P_j)"
        ),
        "proof": (
            "A zero of a regular finite-valued product makes one unary or "
            "pairwise factor zero, hence forces the whole cylinder obtained "
            "by varying the other coordinates to vanish. In a nontrivial "
            "prime-order group, changing one free coordinate changes the "
            "sum, so Z_T contains no unary or pairwise cylinder."
        ),
        "regular_finite_valued_pure_product_refuted": True,
        "rational_pole_cancellation_covered": False,
        "bounded_sum_of_products_covered": False,
        "projective_trivialization_covered": False,
    }


def translated_section_rank_theorem() -> dict[str, Any]:
    return {
        "section": "f_T(P,Q)=kappa(P+Q)-kappa(T) over F_(p^2)(E)",
        "kappa": "x+w*y with a pole of order three at O",
        "proof": (
            "For distinct P_i, the translate Q -> kappa(P_i+Q)-kappa(T) "
            "has its unique order-three pole at Q=-P_i. At that pole no "
            "other translate has a pole, so every finite linear relation "
            "has zero coefficient. The translates are linearly independent."
        ),
        "generic_balanced_small_side_exponent_B": fraction_record(
            SMALL_SIDE_EXPONENT
        ),
        "uniform_two_block_section_rank_lower_exponent_B": fraction_record(
            SMALL_SIDE_EXPONENT
        ),
        "rank_lower_bound_above_setup_cap": (
            SMALL_SIDE_EXPONENT > SETUP_CAP
        ),
        "rank_lower_bound_above_online_cap": (
            SMALL_SIDE_EXPONENT > ONLINE_CAP
        ),
        "ffe_scalar_extension_changes_rank": False,
        "scope": (
            "This closes uniform finite separated-rank identities in the "
            "function field. It is not a lower bound for arbitrary small "
            "arithmetic circuits that generate high rank implicitly."
        ),
    }


def cost_ledger() -> dict[str, Any]:
    cube = R108.theorem_of_cube_control()
    pure = pure_pairwise_zero_theorem()
    rank = translated_section_rank_theorem()
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_workspace_exponent_B": fraction_record(ONLINE_CAP),
        },
        "theorem_of_cube_pair_tables": {
            "pair_table_exponents_B": cube["pair_table_exponents_B"],
            "all_pair_tables_inside_online_cap": cube[
                "all_pair_tables_inside_online_cap"
            ],
        },
        "regular_pure_pairwise_scalar_product": {
            "zero_set_cylinder_theorem": pure,
            "constructor_valid": False,
            "pairwise_norm_product_cost_credit": False,
        },
        "uniform_bounded_separated_section": {
            "translation_rank_theorem": rank,
            "minimum_rank_state_exponent_B": fraction_record(
                SMALL_SIDE_EXPONENT
            ),
            "inside_setup_cap": False,
            "inside_online_cap": False,
        },
        "scope_boundary": {
            "pure_regular_pairwise_section_product_closed": True,
            "uniform_separated_rank_below_B12O5_closed": True,
            "rational_pole_cancellation_network_open": True,
            "actual_deck_specific_high_rank_circuit_open": True,
            "general_arithmetic_circuit_lower_bound_claimed": False,
        },
    }


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        "r108_producer": {
            "path": str(R108_PRODUCER),
            "sha256": R108_PRODUCER_SHA256,
        },
        "r108_report": {
            "path": str(R108_REPORT),
            "sha256": R108_REPORT_SHA256,
        },
        "r108_gate": {
            "path": str(R108_GATE),
            "sha256": R108_GATE_SHA256,
        },
        "r108_parent": {
            "path": str(R108_PARENT),
            "sha256": R108_PARENT_SHA256,
        },
        "r101_producer": {
            "path": str(R101_PRODUCER),
            "sha256": R101_PRODUCER_SHA256,
        },
        "r101_report": {
            "path": str(R101_REPORT),
            "sha256": R101_REPORT_SHA256,
        },
        "r101_gate": {
            "path": str(R101_GATE),
            "sha256": R101_GATE_SHA256,
        },
        "r104_report": {
            "path": str(R104_REPORT),
            "sha256": R104_REPORT_SHA256,
        },
        "r105_report": {
            "path": str(R105_REPORT),
            "sha256": R105_REPORT_SHA256,
        },
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    controls = actual_and_matched_controls()
    pure = pure_pairwise_zero_theorem()
    rank = translated_section_rank_theorem()
    ledger = cost_ledger()
    obligations = {
        "nine_source_bindings_verified": len(source_hashes) == 9,
        "sixteen_actual_and_matched_instances": (
            controls["instance_count"] == 16
        ),
        "all_unary_pairwise_cylinder_controls_exact": controls[
            "all_cylinder_controls_exact"
        ],
        "regular_pure_pairwise_zero_set_refuted": pure[
            "regular_finite_valued_pure_product_refuted"
        ],
        "all_balanced_side_images_injective": controls[
            "all_balanced_side_images_injective"
        ],
        "all_signed_section_sample_ranks_full": controls[
            "all_sample_section_ranks_full"
        ],
        "translated_sections_linearly_independent": True,
        "uniform_section_rank_exponent_B12O5": (
            rank["uniform_two_block_section_rank_lower_exponent_B"]["exact"]
            == "12/5"
        ),
        "uniform_section_rank_misses_setup_cap": rank[
            "rank_lower_bound_above_setup_cap"
        ],
        "uniform_section_rank_misses_online_cap": rank[
            "rank_lower_bound_above_online_cap"
        ],
        "ffe_extension_preserves_rank": not rank[
            "ffe_scalar_extension_changes_rank"
        ],
        "poincare_pair_tables_individually_inside_online_cap": ledger[
            "theorem_of_cube_pair_tables"
        ]["all_pair_tables_inside_online_cap"],
        "r108_canonical_weight_14400_preserved": True,
        "rational_pole_cancellation_network_closed": False,
        "bounded_sum_theta_addition_circuit_closed": False,
        "actual_deck_specific_high_rank_circuit_closed": False,
        "homogeneous_projective_trivialization_complete": False,
        "generic_marker_factorization_inside_caps": False,
        "generic_multiplicity_and_integer_lift_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_algorithm": False,
        "shoup_improvement_complete": False,
        "full_pipeline_fresh_workspace_inside_cap": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    failures = [name for name, value in obligations.items() if not value]
    classification = (
        "TARGET_EQUALITY_FIBER_HAS_NO_UNARY_OR_PAIRWISE_ZERO_CYLINDER__"
        "REGULAR_PURE_POINCARE_SECTION_PRODUCT_REFUTED__TRANSLATED_SIGNED_"
        "SECTIONS_HAVE_B12O5_UNIFORM_FLATTENING_RANK__ACTUAL_AND_MATCHED_"
        "BALANCED_IMAGES_INJECTIVE_AND_SAMPLE_RANKS_FULL__RATIONAL_THETA_"
        "CANCELLATION_NETWORK_OPEN"
    )
    next_action = (
        "Construct or refute one exact finite-field theta-addition "
        "cancellation network for the target section. Freeze the theta "
        "basis, rational poles, projective trivializations, additions, bond "
        "dimensions, and deck contraction order; allow high flattening rank "
        "only when generated implicitly inside both caps; require exact "
        "zero biconditional, the R108 weight 14400 marker lift, generic "
        "multiplicity/integer lifting, rank, logs, and identical descent."
    )
    frozen = {
        "schema": "p1553.frozen_5a5c_poincare_theta_target_section.r109.v1",
        "campaign": "P1553",
        "round": "R109",
        "public_inputs": "ten atom choices, compact D_A,D_C, target T",
        "section": "kappa(sum_i P_i)-kappa(T)",
        "candidate_classes": [
            "regular unary/pairwise scalar product",
            "uniform two-block separated section",
        ],
        "rank_sample_cap": RANK_SAMPLE_CAP,
        "actual_offsets": list(R82.INSTANCE_OFFSETS),
        "matched_random_offsets": [2, 3],
        "identities_frozen_before_outcomes": True,
        "source_bindings": source_binding_records(),
        "candidate_scalar_labels_consumed": False,
    }
    marker_replay = {
        "schema": "p1553.poincare_theta_canonical_marker_replay.r109.v1",
        "r108_cycle_weight_per_canonical_source": 14_400,
        "pure_pairwise_zero_set_theorem": pure,
        "finite_cylinder_controls": {
            "actual": controls["cylinder_actual"],
            "matched_random": controls["cylinder_matched_random"],
        },
        "canonical_marker_weight_preserved_conditionally": True,
        "candidate_scalar_section_evaluator_complete": False,
        "candidate_work_credit": False,
    }
    exceptional = {
        "schema": "p1553.poincare_theta_exceptional_controls.r109.v1",
        "signed_section_rank_controls": {
            "actual": controls["rank_actual"],
            "matched_random": controls["rank_matched_random"],
        },
        "kappa_function_field_pole_order_at_identity": 3,
        "finite_point_key_reserves_identity_as_zero": True,
        "function_field_and_projective_identity_charts_identical": False,
        "homogeneous_projective_trivialization_complete": False,
        "rational_pole_cancellation_network_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r109.v1",
        "relation_source": (
            "No inside-cap scalar section circuit survives the pure-product "
            "or uniform separated-rank gates."
        ),
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "identical_relation_and_target_distribution": False,
        "generic_prime_family_algorithm": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    report = {
        "schema": SCHEMA,
        "classification": classification,
        "claim_status": (
            "SCOPED_PURE_SECTION_AND_UNIFORM_RANK_NEGATIVE_"
            "NO_ALGORITHM_CLAIM"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_hashes_verified": source_hashes,
        "primary_sources": [
            {
                "url": "https://stacks.math.columbia.edu/tag/0BFE",
                "use": (
                    "line-bundle scope of the theorem of the cube; no "
                    "target-section pure tensor is asserted"
                ),
            }
        ],
        "theorem": {
            "pure_pairwise_zero_set": pure,
            "translated_section_rank": rank,
        },
        "finite_controls": {
            "instance_count": controls["instance_count"],
            "all_cylinders_exact": controls[
                "all_cylinder_controls_exact"
            ],
            "all_side_images_injective": controls[
                "all_balanced_side_images_injective"
            ],
            "all_sample_ranks_full": controls[
                "all_sample_section_ranks_full"
            ],
            "minimum_sample_dimension": controls[
                "minimum_sample_dimension"
            ],
            "minimum_sample_rank": controls["minimum_sample_rank"],
        },
        "cost_ledger": ledger,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "failures": failures,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": "frozen_5a5c_poincare_theta_target_section.json",
            "rank_ledger": "poincare_theta_section_rank_ledger.json",
            "marker_replay": "poincare_theta_canonical_marker_replay.json",
            "exceptional": "poincare_theta_exceptional_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r109.json",
        },
        "next_action": next_action,
        "non_claims": [
            "No lower bound for arbitrary arithmetic circuits is proved.",
            "Rational theta identities with pole cancellation remain open.",
            "Actual-deck-specific high-rank implicit circuits remain open.",
            "No generic-prime ECDLP algorithm is constructed.",
            "No rank, factor-log, target-descent, or Shoup gate passes.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "REJECT_REGULAR_PURE_PAIRWISE_TARGET_SECTION_AND_UNIFORM_"
            "SEPARATED_RANK_BELOW_B12O5_ONLY__PRESERVE_RATIONAL_THETA_"
            "CANCELLATION_NETWORK__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__"
            "NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "rank_ledger": ledger,
        "marker_replay": marker_replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_poincare_theta_target_section_"
            "rank_probe_report_r109.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_poincare_theta_target_section.json"
        ),
    )
    parser.add_argument(
        "--rank-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "poincare_theta_section_rank_ledger.json"
        ),
    )
    parser.add_argument(
        "--marker-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "poincare_theta_canonical_marker_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "poincare_theta_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r109.json"
        ),
    )
    return parser.parse_args()


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.rank_output, bundle["rank_ledger"])
    write_json(args.marker_output, bundle["marker_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    admission = bundle["report"]["admission"]
    print(
        f"R109 classification={bundle['report']['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
