#!/usr/bin/env python3
"""Generate smooth trisecants by decomposing multiplication-kernel matrices."""

import argparse
import hashlib
import itertools
import json
import pathlib

import p1553_exact_public_base_trisecant_scan_r65 as r65


BASE_SIZE = 24
PINNED_DEPENDENCY_SHA256 = {
    "p1553_catalog_pencil_fiber_scan_r46.py": "3101a43126a2c1511b15ad6aef1bdc7a0144719ecb31bc9708d10128413e2572",
    "p1553_coprime_product_trisecant_probe_r54.py": "c42ce441526d9adc95617eccac15a5b09e29a044e40f253ea0f8123a6a7b1ccc",
    "p1553_degree_six_primitive_product_pencil_search_r60.py": "5a9b2f6987fc945e3664347b6c929fc0fc4b973b615e0277545bc4ec5091e5f8",
    "p1553_degree_six_product_hypersurface_locator_r61.py": "b8331ae9534fde907df8cffda074d2443f219324174fee6f49756cd40351bbfb",
    "p1553_degree_six_scan_free_factor_lift_r62.py": "15ca1a87202d069b79325b7a1743762d221f0726e793020910fb74c3633d900a",
    "p1553_exact_public_base_trisecant_scan_r65.py": "148c8c0b39c5a1849a99e7509e5212364f777d6e38899f55affe6611b6ee3ff8",
    "p1553_fixed_class_multiplication_kernel_scan_r56.py": "40698d92012a4ca4e79f5ac8da58f76967a14e2dd20ba234e1b61c273aedc210",
    "p1553_full_curve_primitive_product_pencil_search_r58.py": "d4fda31b41d93ef58b393476a6e4a64535947e305e132fe671dcac7cf9c141f3",
    "p1553_low_boundary_pencil_search_r38.py": "a750f12e3aa8840f63e4b32ff1f783f00086f0d2c4244bb2c6a70614488a5de8",
    "p1553_primitive_degree_nine_rank_search_r44.py": "b9bbad90f2f5076ea0cb8346d94514a7e8f82993daa07c374d050e334be2687b",
    "p1553_public_factor_base_closure_probe_r63.py": "7f5d061a539d62d2b3d362fed8252fead5b8e14dbc9250f5543e686f3a8d89cb",
}


def matrix_multiply(left, right):
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(3))
            % r65.r63.PRIME
            for column in range(3)
        ]
        for row in range(3)
    ]


def matrix_rank(matrix):
    return r65.r63.r62.r61.matrix_rank_mod(matrix)


def multiplication_kernel_basis(multiplication_matrix):
    echelon, pivots = r65.r63.r62.r61.echelon_mod(multiplication_matrix)
    free_columns = [column for column in range(9) if column not in pivots]
    if len(pivots) != 6 or len(free_columns) != 3:
        raise AssertionError("unexpected multiplication-kernel dimension")
    basis = []
    for free_column in free_columns:
        vector = [0] * 9
        vector[free_column] = 1
        for row, pivot in reversed(list(enumerate(pivots))):
            vector[pivot] = -sum(
                int(echelon[row, column]) * vector[column]
                for column in range(pivot + 1, 9)
            ) % r65.r63.PRIME
        basis.append(tuple(vector))
    if any(
        any(
            sum(row[column] * vector[column] for column in range(9))
            % r65.r63.PRIME
            for row in multiplication_matrix
        )
        for vector in basis
    ):
        raise AssertionError("kernel basis failed multiplication-map replay")
    return basis


def tensor_matrix(vector):
    return [list(vector[3 * row : 3 * row + 3]) for row in range(3)]


def normalized_coordinates(coordinates):
    return r65.r63.r62.r61.r60.r58.normalized(tuple(coordinates))


def factor_section_catalog(base, class_sum, factor_basis):
    records = r65.r63.public_factor_catalog(base, class_sum, factor_basis)
    by_coordinates = {}
    for record in records:
        key = normalized_coordinates(record["coordinates"])
        if key in by_coordinates:
            raise AssertionError("one factor section has two lifted divisors")
        by_coordinates[key] = record
    return records, by_coordinates


def kernel_combination(kernel_matrices, coordinates):
    return [
        [
            sum(
                coordinates[index] * kernel_matrices[index][row][column]
                for index in range(3)
            )
            % r65.r63.PRIME
            for column in range(3)
        ]
        for row in range(3)
    ]


def product_record(left, right, multiplication_matrix):
    if left["mask"] & right["mask"]:
        return None
    section = r65.r63.r62.r61.product_output(
        multiplication_matrix,
        left["coordinates"],
        right["coordinates"],
    )
    if not any(section):
        return None
    return {
        "section": normalized_coordinates(section),
        "mask": left["mask"] | right["mask"],
        "factors": [
            {
                "base_indices": list(left["base_indices"]),
                "verification_scalars": list(left["verification_scalars"]),
                "torsion_lift_pattern": list(left["torsion_lift_pattern"]),
                "line": list(left["line"]),
            },
            {
                "base_indices": list(right["base_indices"]),
                "verification_scalars": list(right["verification_scalars"]),
                "torsion_lift_pattern": list(right["torsion_lift_pattern"]),
                "line": list(right["line"]),
            },
        ],
    }


def run():
    r61_path = pathlib.Path(
        "p1553_degree_six_product_hypersurface_locator_report_r61.json"
    )
    r61_bytes = r61_path.read_bytes()
    r61_report = json.loads(r61_bytes)
    r65_path = pathlib.Path("p1553_exact_public_base_trisecant_scan_report_r65.json")
    r65_bytes = r65_path.read_bytes()
    r65_report = json.loads(r65_bytes)
    r65_script_bytes = pathlib.Path(
        "p1553_exact_public_base_trisecant_scan_r65.py"
    ).read_bytes()
    dependency_source_sha256 = {
        path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
        for path in PINNED_DEPENDENCY_SHA256
    }
    multiplication_matrix = r61_report["multiplication_map"]["multiplication_matrix"]
    kernel_basis = multiplication_kernel_basis(multiplication_matrix)
    kernel_matrices = [tensor_matrix(vector) for vector in kernel_basis]
    factor_bases = r65.r63.r62.factor_basis_lines(r61_report)
    public_base = r65.r63.subgroup_points_in_public_order()[:BASE_SIZE]
    left_records, left_lookup = factor_section_catalog(
        public_base,
        r65.r63.CLASS_SUM_SCALARS[0],
        factor_bases[0],
    )
    right_records, right_lookup = factor_section_catalog(
        public_base,
        r65.r63.CLASS_SUM_SCALARS[1],
        factor_bases[1],
    )

    left_triples = 0
    pairwise_disjoint_left_triples = 0
    invertible_left_triples = 0
    singular_first_row_maps = 0
    candidate_kernel_coordinates = 0
    all_three_right_factors_smooth = 0
    pairwise_disjoint_right_factor_triples = 0
    internally_simple_relation_candidates = 0
    primitive_relation_candidates = 0
    generated_lines = {}
    smooth_right_hit_histogram_per_X = {}
    disjoint_right_hit_histogram_per_X = {}
    maximum_smooth_right_hits_for_one_X = 0
    maximum_disjoint_right_hits_for_one_X = 0

    for left_indices in itertools.combinations(range(len(left_records)), 3):
        left_triples += 1
        left_factors = [left_records[index] for index in left_indices]
        if any(
            left_factors[first]["mask"] & left_factors[second]["mask"]
            for first, second in itertools.combinations(range(3), 2)
        ):
            continue
        pairwise_disjoint_left_triples += 1
        left_matrix = [
            [left_factors[column]["coordinates"][row] for column in range(3)]
            for row in range(3)
        ]
        if matrix_rank(left_matrix) != 3:
            continue
        invertible_left_triples += 1
        left_inverse = r65.r63.r62.r61.r56.matrix_inverse_mod(left_matrix)
        transformed_kernel = [
            matrix_multiply(left_inverse, kernel_matrix)
            for kernel_matrix in kernel_matrices
        ]
        first_row_map = [
            [transformed_kernel[basis][0][coordinate] for basis in range(3)]
            for coordinate in range(3)
        ]
        if matrix_rank(first_row_map) != 3:
            singular_first_row_maps += 1
            continue
        first_row_inverse = r65.r63.r62.r61.r56.matrix_inverse_mod(first_row_map)
        smooth_right_hits_for_X = 0
        disjoint_right_hits_for_X = 0

        for first_right in right_records:
            candidate_kernel_coordinates += 1
            kernel_coordinates = r65.r63.r62.r61.r56.matrix_vector(
                first_row_inverse,
                first_right["coordinates"],
            )
            kernel_matrix = kernel_combination(kernel_matrices, kernel_coordinates)
            right_rows = matrix_multiply(left_inverse, kernel_matrix)
            right_keys = []
            failed = False
            for row in right_rows:
                if not any(row):
                    failed = True
                    break
                key = normalized_coordinates(row)
                if key not in right_lookup:
                    failed = True
                    break
                right_keys.append(key)
            if failed:
                continue
            all_three_right_factors_smooth += 1
            smooth_right_hits_for_X += 1
            right_factors = [right_lookup[key] for key in right_keys]
            if any(
                right_factors[first]["mask"] & right_factors[second]["mask"]
                for first, second in itertools.combinations(range(3), 2)
            ):
                continue
            pairwise_disjoint_right_factor_triples += 1
            disjoint_right_hits_for_X += 1
            products = [
                product_record(left, right, multiplication_matrix)
                for left, right in zip(left_factors, right_factors)
            ]
            if any(product is None for product in products):
                continue
            if len({product["section"] for product in products}) != 3:
                continue
            internally_simple_relation_candidates += 1
            if any(
                products[first]["mask"] & products[second]["mask"]
                for first, second in itertools.combinations(range(3), 2)
            ):
                continue
            primitive_relation_candidates += 1
            unnormalized_sections = [
                r65.r63.r62.r61.product_output(
                    multiplication_matrix,
                    left["coordinates"],
                    right_row,
                )
                for left, right_row in zip(left_factors, right_rows)
            ]
            if any(
                sum(section[coordinate] for section in unnormalized_sections)
                % r65.r63.PRIME
                for coordinate in range(6)
            ):
                raise AssertionError("kernel decomposition did not produce a section relation")
            normalized_relation_coefficients = []
            for section in unnormalized_sections:
                first_nonzero = next(value for value in section if value)
                normalized_relation_coefficients.append(first_nonzero)
            if any(
                sum(
                    coefficient * product["section"][coordinate]
                    for coefficient, product in zip(
                        normalized_relation_coefficients,
                        products,
                    )
                )
                % r65.r63.PRIME
                for coordinate in range(6)
            ):
                raise AssertionError("normalized relation coefficients failed replay")
            key = r65.plucker_key(products[0]["section"], products[1]["section"])
            if any(
                r65.plucker_key(products[0]["section"], product["section"]) != key
                for product in products[1:]
            ):
                raise AssertionError("generated products are not collinear")
            generated_lines.setdefault(
                key,
                {
                    "kernel_coordinates": list(kernel_coordinates),
                    "kernel_matrix": kernel_matrix,
                    "kernel_matrix_rank": matrix_rank(kernel_matrix),
                    "normalized_section_relation_coefficients": normalized_relation_coefficients,
                    "products": products,
                },
            )
        smooth_right_hit_histogram_per_X[smooth_right_hits_for_X] = (
            smooth_right_hit_histogram_per_X.get(smooth_right_hits_for_X, 0) + 1
        )
        disjoint_right_hit_histogram_per_X[disjoint_right_hits_for_X] = (
            disjoint_right_hit_histogram_per_X.get(disjoint_right_hits_for_X, 0) + 1
        )
        maximum_smooth_right_hits_for_one_X = max(
            maximum_smooth_right_hits_for_one_X,
            smooth_right_hits_for_X,
        )
        maximum_disjoint_right_hits_for_one_X = max(
            maximum_disjoint_right_hits_for_one_X,
            disjoint_right_hits_for_X,
        )

    r65_witness = r65_report["exact_incidence_scan"]["first_primitive_witness"]
    r65_key = r65.plucker_key(
        r65_witness["sections"][0],
        r65_witness["sections"][1],
    )
    checks = {
        "direct_r61_report_digest_is_pinned": hashlib.sha256(r61_bytes).hexdigest()
        == "f91213021bd350b77cf1ac7d5cb8f2f9ae63b1265a5081d4ed58a3da07aa1f64",
        "direct_r65_report_digest_is_pinned": hashlib.sha256(r65_bytes).hexdigest()
        == "30d686228a271cbdf4a61e2bf4b7d553bc297d7e6203955863b15b7ff98fdb8a",
        "imported_r65_implementation_digest_is_pinned": hashlib.sha256(
            r65_script_bytes
        ).hexdigest()
        == "148c8c0b39c5a1849a99e7509e5212364f777d6e38899f55affe6611b6ee3ff8",
        "all_loaded_dependency_source_digests_are_pinned": (
            dependency_source_sha256 == PINNED_DEPENDENCY_SHA256
        ),
        "factor_section_counts_match_r65": [len(left_records), len(right_records)]
        == [56, 84],
        "kernel_dimension_is_three": len(kernel_basis) == 3,
        "all_generated_kernel_matrices_are_full_rank": all(
            record["kernel_matrix_rank"] == 3 for record in generated_lines.values()
        ),
        "generated_primitive_line_set_matches_r65": set(generated_lines) == {r65_key},
    }
    if not all(checks.values()):
        raise AssertionError(f"R66 kernel relation generator failed: {checks}")

    left_section_count = len(left_records)
    right_section_count = len(right_records)
    projective_factor_space_size = r65.r63.PRIME**2 + r65.r63.PRIME + 1
    expected_hits_per_invertible_left_triple = (
        right_section_count**3 / projective_factor_space_size**2
    )
    return {
        "schema": "p1553.kernel_decomposition_relation_generator.r66.v1",
        "classification": [
            "toy",
            "exact-kernel-decomposition-generator",
            "heuristic-cost-model",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r65.r63.PRIME,
        "prime_subgroup_order": r65.r63.SUBGROUP_ORDER,
        "public_base_size": BASE_SIZE,
        "smooth_factor_section_counts": [left_section_count, right_section_count],
        "dependency_source_sha256": dependency_source_sha256,
        "multiplication_kernel": {
            "dimension": len(kernel_basis),
            "basis_vectors": [list(vector) for vector in kernel_basis],
            "basis_matrices": kernel_matrices,
        },
        "generation_scan": {
            "left_factor_triples": left_triples,
            "pairwise_disjoint_left_factor_triples": pairwise_disjoint_left_triples,
            "invertible_left_factor_triples": invertible_left_triples,
            "singular_first_row_maps": singular_first_row_maps,
            "candidate_first_right_factors": candidate_kernel_coordinates,
            "all_three_right_factors_smooth": all_three_right_factors_smooth,
            "pairwise_disjoint_right_factor_triples": pairwise_disjoint_right_factor_triples,
            "internally_simple_relation_candidates": internally_simple_relation_candidates,
            "primitive_relation_candidate_visits": primitive_relation_candidates,
            "distinct_primitive_relation_lines": len(generated_lines),
            "smooth_right_hit_histogram_per_X": {
                str(hits): smooth_right_hit_histogram_per_X[hits]
                for hits in sorted(smooth_right_hit_histogram_per_X)
            },
            "disjoint_right_hit_histogram_per_X": {
                str(hits): disjoint_right_hit_histogram_per_X[hits]
                for hits in sorted(disjoint_right_hit_histogram_per_X)
            },
            "maximum_smooth_right_hits_for_one_X": maximum_smooth_right_hits_for_one_X,
            "maximum_disjoint_right_hits_for_one_X": maximum_disjoint_right_hits_for_one_X,
            "generated_lines": [
                {
                    "plucker_key_hex": key.hex(),
                    **record,
                }
                for key, record in sorted(generated_lines.items())
            ],
        },
        "cost_model": {
            "construction": "choose three smooth left sections X; enumerate one smooth right section y1; solve a 3x3 linear map for K; the remaining y2,y3 are forced",
            "candidate_work_per_invertible_X": right_section_count,
            "uniform_factor_section_expected_hits_per_X": expected_hits_per_invertible_left_triple,
            "generic_smooth_factor_section_scale": "S about B^3/N for a random fixed-sum point base when p is comparable to N",
            "generic_expected_X_trials_per_relation": "about |P^2(F_p)|^2/S^3",
            "generic_expected_candidate_tests_per_relation": "about |P^2(F_p)|^2/S^2, asymptotically (N/B)^6 when p is comparable to N",
            "relation_collection_for_B_rows": "about N^6/B^5 candidate tests under the uniform model; no exponent gain over R63",
            "full_toy_scan_cost": "C(S_left,3)*S_right candidate tests before rank and disjointness filters",
            "full_scan_asymptotic": "Theta(S_left^3*S_right); for balanced catalogs this is Theta(S^4), the same order as all pairs of M about S^2 smooth products",
            "toy_candidate_count_ratio_to_R65_pairs": candidate_kernel_coordinates
            / r65_report["exact_incidence_scan"]["section_pairs"],
            "base_and_factor_catalog_construction_charged": False,
            "scalar_labelled_subgroup_table_materialization": "literal builder uses Theta(N log N) group operations from N independent double-and-add multiplications plus O(N log N) sorting and creates an invertible scalar-labelled lookup table; not an inverse DLP, but already too expensive for a sub-rho claim",
            "scalar_labelled_subgroup_table_cost_charged_as_blocking": True,
            "relation_rank_charged": False,
            "projected_scalar_logs_charged": False,
            "linear_algebra_charged": False,
            "target_descent_charged": False,
        },
        "checks": checks,
        "result": {
            "conditional_precomputed_catalog_kernel_decomposition_rule": True,
            "unconditional_public_kernel_decomposition_generation_rule": False,
            "subquadratic_in_full_smooth_product_pair_scan": False,
            "toy_constant_factor_candidate_reduction": True,
            "unique_r65_primitive_line_recovered": True,
            "asymptotic_exponent_improvement_under_uniform_model": False,
            "asymptotic_relation_algorithm": False,
            "fresh_target_action": False,
            "r10_queried_coefficients": False,
            "independent_rank": False,
            "factor_base_logs": False,
            "scalar_blind_descent": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "limits": [
            "the exact generator is tested on one B=24 public base and one toy curve",
            "the executable depends on a literal Theta(N log N)-group-work scalar-labelled subgroup table plus O(N log N) sorting and is public only after those toy catalogs are precomputed",
            "the generator is cheaper than all smooth-product pairs but its uniform smoothness model recovers the same N^6/B^5 relation-collection exponent",
            "base construction, relation rank, projected logs, linear algebra, and fresh-target descent remain uncharged",
            "no Shoup-bound improvement or ECDLP breakthrough is claimed",
        ],
        "pass": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    arguments = parser.parse_args()
    payload = json.dumps(run(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        pathlib.Path(arguments.output).write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
