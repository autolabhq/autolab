#!/usr/bin/env python3
"""Verify the R58 public subgroup projection and charge its split-class rate."""

import json
import math
import hashlib
from pathlib import Path

import p1553_full_curve_primitive_product_pencil_search_r58 as r58


REPORT_PATH = "p1553_full_curve_primitive_product_pencil_search_report_r58.json"

ROOT_PATH = Path(__file__).resolve().parent


DEPENDENCY_HASHES = {
    "p1553_full_curve_primitive_product_pencil_search_r58.py": "d4fda31b41d93ef58b393476a6e4a64535947e305e132fe671dcac7cf9c141f3",
    "p1553_primitive_degree_nine_rank_search_r44.py": "b9bbad90f2f5076ea0cb8346d94514a7e8f82993daa07c374d050e334be2687b",
    "p1553_low_boundary_pencil_search_r38.py": "a750f12e3aa8840f63e4b32ff1f783f00086f0d2c4244bb2c6a70614488a5de8",
}


def file_sha256(path: str) -> str:
    return hashlib.sha256((ROOT_PATH / path).read_bytes()).hexdigest()


def run():
    with open(REPORT_PATH, encoding="utf-8") as input_file:
        r58_report = json.load(input_file)
    points = r58.curve_points()
    subgroup_points = {
        r58.r44.r38.scalar_mul(scalar, r58.r44.r38.GENERATOR): scalar
        for scalar in range(r58.r44.r38.SUBGROUP_ORDER)
    }
    projector = r58_report["public_cofactor_projector"]
    projected_points = [
        r58.scalar_mul_integer(projector, point) for point in points
    ]
    image = set(projected_points)
    kernel = {
        point
        for point, projected in zip(points, projected_points)
        if projected is None
    }

    homomorphism_checks = 0
    for left in points:
        for right in points:
            homomorphism_checks += 1
            if r58.scalar_mul_integer(projector, r58.r44.r38.add(left, right)) != (
                r58.r44.r38.add(
                    r58.scalar_mul_integer(projector, left),
                    r58.scalar_mul_integer(projector, right),
                )
            ):
                raise AssertionError("cofactor projection is not a homomorphism")

    witness = r58_report["witness"]
    third_records = witness["third_point_records"]
    projected_labels = [
        subgroup_points[
            r58.scalar_mul_integer(
                projector,
                points[record["curve_point_index"]],
            )
        ]
        for record in third_records
    ]
    label_by_index = {
        record["curve_point_index"]: label
        for record, label in zip(third_records, projected_labels)
    }
    projected_factor_labels = [
        [label_by_index[index] for index in block]
        for block in witness["third_factor_point_indices"]
    ]
    first_labels = set(witness["first_scalar_block"])
    second_labels = set(witness["second_scalar_block"])
    third_labels = set(projected_labels)

    degree = 9
    factor_count = 3
    factor_degree = 3
    ordered_class_partitions = math.factorial(degree) // (
        math.factorial(factor_degree) ** factor_count
    )
    heuristic_success_probability = (
        (r58.r44.r38.PRIME - 1)
        * ordered_class_partitions
        / (
            math.factorial(degree)
            * r58.r44.r38.SUBGROUP_ORDER ** (factor_count - 1)
        )
    )
    heuristic_pencils_per_success = 1 / heuristic_success_probability
    observed_pencils_per_success = r58_report["search"][
        "accepted_coprime_pencils"
    ]

    degree_three_factor_model = []
    for factors in range(2, 7):
        model_degree = factor_degree * factors
        partitions = math.factorial(model_degree) // (
            math.factorial(factor_degree) ** factors
        )
        probability = (
            (r58.r44.r38.PRIME - 1)
            * partitions
            / (
                math.factorial(model_degree)
                * r58.r44.r38.SUBGROUP_ORDER ** (factors - 1)
            )
        )
        degree_three_factor_model.append(
            {
                "factor_count": factors,
                "map_degree": model_degree,
                "ordered_class_partitions": partitions,
                "heuristic_success_probability_per_pencil": probability,
                "heuristic_pencils_per_success": 1 / probability,
                "asymptotic_probability_for_p_near_N": f"1/(6^{factors}*N^{factors-2})",
            }
        )

    checks = {
        "r58_report_passes": r58_report["pass"] is True,
        "projector_is_104": projector == 104,
        "projector_congruences_hold": projector % 103 == 1
        and projector % 2 == 0,
        "projection_image_is_exact_prime_subgroup": image == set(subgroup_points),
        "projection_kernel_is_identity_and_two_torsion": kernel
        == {None, tuple(r58_report["two_torsion_point"])},
        "projection_is_idempotent_on_all_curve_points": all(
            r58.scalar_mul_integer(projector, projected) == projected
            for projected in projected_points
        ),
        "r58_source_matches_pinned_sha256": file_sha256(
            "p1553_full_curve_primitive_product_pencil_search_r58.py"
        )
        == DEPENDENCY_HASHES[
            "p1553_full_curve_primitive_product_pencil_search_r58.py"
        ],
        "r44_source_matches_pinned_sha256": file_sha256(
            "p1553_primitive_degree_nine_rank_search_r44.py"
        )
        == DEPENDENCY_HASHES[
            "p1553_primitive_degree_nine_rank_search_r44.py"
        ],
        "r38_source_matches_pinned_sha256": file_sha256(
            "p1553_low_boundary_pencil_search_r38.py"
        )
        == DEPENDENCY_HASHES["p1553_low_boundary_pencil_search_r38.py"],
        "projection_homomorphism_checked_on_all_pairs": homomorphism_checks
        == len(points) ** 2,
        "third_projected_labels_are_distinct": len(third_labels) == 9,
        "third_projected_factor_sums_match_classes": [
            sum(labels) % r58.r44.r38.SUBGROUP_ORDER
            for labels in projected_factor_labels
        ]
        == list(r58.CLASS_SUM_SCALARS),
        "third_projected_total_sum_matches": sum(projected_labels)
        % r58.r44.r38.SUBGROUP_ORDER
        == r58.TOTAL_SUM_SCALAR,
        "projection_collisions_with_generators_are_56_and_49": first_labels
        & third_labels
        == {56}
        and second_labels & third_labels == {49},
        "ordered_class_partition_count_is_1680": ordered_class_partitions
        == 1680,
        "r58_observation_matches_fixed_degree_model_within_factor_two": 0.5
        <= observed_pencils_per_success / heuristic_pencils_per_success
        <= 2.0,
    }
    if not all(checks.values()):
        raise AssertionError(f"R59 projection/cost gate failed: {checks}")

    return {
        "schema": "p1553.public_cofactor_projection_cost_gate.r59.v1",
        "classification": [
            "toy",
            "exact-projection",
            "heuristic-fixed-degree-cost-model",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r58.r44.r38.PRIME,
        "curve_order": len(points),
        "prime_subgroup_order": r58.r44.r38.SUBGROUP_ORDER,
        "cofactor": 2,
        "projector": {
            "scalar": projector,
            "image_size": len(image),
            "kernel": [
                None if point is None else list(point)
                for point in sorted(
                    kernel,
                    key=lambda point: (-1, -1) if point is None else point,
                )
            ],
            "all_group_pairs_checked": homomorphism_checks,
        },
        "projected_witness": {
            "third_projected_scalar_labels_for_verification_only": projected_labels,
            "third_projected_factor_labels_for_verification_only": projected_factor_labels,
            "first_third_label_intersection": sorted(first_labels & third_labels),
            "second_third_label_intersection": sorted(second_labels & third_labels),
            "distinct_projected_atoms": len(third_labels),
            "public_point_recovery_requires_dlp": False,
            "scalar_log_recovery_requires_linear_algebra_or_dlp": True,
        },
        "fixed_degree_rate_model": {
            "status": "heuristic",
            "assumptions": [
                "degree-d pencil fibers have random permutation splitting statistics",
                "ordered class-sum constraints behave as independent uniform subgroup equations",
                "field size p and prime subgroup order N are the same asymptotic scale",
            ],
            "degree": degree,
            "degree_three_factors": factor_count,
            "split_fiber_probability": f"1/{math.factorial(degree)}",
            "ordered_class_partitions": ordered_class_partitions,
            "class_sum_constraints": factor_count - 1,
            "heuristic_success_probability_per_pencil": heuristic_success_probability,
            "heuristic_pencils_per_success": heuristic_pencils_per_success,
            "observed_first_success_accepted_pencil": observed_pencils_per_success,
            "observed_to_heuristic_ratio": observed_pencils_per_success
            / heuristic_pencils_per_success,
            "general_degree_three_factor_model": degree_three_factor_model,
        },
        "checks": checks,
        "result": {
            "third_fiber_subgroup_atoms_publicly_recovered": True,
            "third_fiber_scalar_logs_recovered": False,
            "fixed_degree_nine_relation_cost_below_rho": False,
            "degree_six_two_factor_rate_candidate": True,
            "asymptotic_pencil_family": False,
            "fresh_target_action": False,
            "r10_queried_coefficients": False,
            "independent_rank": False,
            "factor_base_logs": False,
            "scalar_blind_descent": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "limits": [
            "the public projection is exact, but the scalar labels shown are toy verification data obtained from exhaustive lookup",
            "projection merges cofactor cosets and creates repeated subgroup atoms that must retain multiplicity",
            "the split-fiber and class-sum rate calculation is a random-fiber heuristic, not a theorem",
            "the degree-nine fixed-factor model predicts linear work and does not beat rho",
            "no target locator, R10 output, rank campaign, factor-base logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
