#!/usr/bin/env python3
"""Verify that the R67 FFE relation layers conserve fixed-sum row information."""

import argparse
import hashlib
import json
import pathlib

import p1553_scalar_blind_rank_descent_audit_r67 as r67


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
    "p1553_scalar_blind_rank_descent_audit_r67.py": "60cf8dc258f2ea097a8581bd95b8002829be492a512cf336c270abda6876129e",
}


def incidence_vector(indices, column_count):
    vector = [0] * column_count
    for index in indices:
        vector[index] += 1
    return vector


def subtract(left, right):
    modulus = r67.r65.r63.SUBGROUP_ORDER
    return [(a - b) % modulus for a, b in zip(left, right)]


def add(left, right):
    modulus = r67.r65.r63.SUBGROUP_ORDER
    return [(a + b) % modulus for a, b in zip(left, right)]


def analyze_base(base, multiplication_matrix, factor_bases):
    catalogs = [
        r67.scalar_blind_factor_catalog(base, class_sum, basis)[0]
        for class_sum, basis in zip(
            r67.r65.r63.CLASS_SUM_SCALARS,
            factor_bases,
        )
    ]
    catalog_triples = [
        sorted({tuple(record["base_indices"]) for record in catalog})
        for catalog in catalogs
    ]
    catalog_vectors = [
        [incidence_vector(triple, len(base)) for triple in triples]
        for triples in catalog_triples
    ]
    fixed_sum_rows = [row for rows in catalog_vectors for row in rows]
    fixed_sum_rank = len(r67.row_reduce(fixed_sum_rows, len(base))[1])

    difference_rows = []
    for rows in catalog_vectors:
        reference = rows[0]
        difference_rows.extend(subtract(row, reference) for row in rows[1:])
    difference_rank = len(r67.row_reduce(difference_rows, len(base))[1])

    products, admissible_factor_pairs = r67.scalar_blind_smooth_products(
        catalogs,
        multiplication_matrix,
    )
    product_rows = []
    decomposition_rows = []
    for product in products:
        factors = product["representative_factors"]
        left = incidence_vector(factors[0]["base_indices"], len(base))
        right = incidence_vector(factors[1]["base_indices"], len(base))
        factor_sum = add(left, right)
        collapsed = r67.collapsed_divisor_vector(product["mask"], len(base))
        if factor_sum != collapsed:
            raise AssertionError("projected factor rows do not equal product divisor")
        product_rows.append(collapsed)
        decomposition_rows.append((left, right))

    quotient_rows = [subtract(row, product_rows[0]) for row in product_rows[1:]]
    reference_left, reference_right = decomposition_rows[0]
    decomposed_quotient_rows = [
        add(
            subtract(left, reference_left),
            subtract(right, reference_right),
        )
        for left, right in decomposition_rows[1:]
    ]
    quotient_rank = len(r67.row_reduce(quotient_rows, len(base))[1])
    combined_rank = len(
        r67.row_reduce(difference_rows + quotient_rows, len(base))[1]
    )

    return {
        "base_size": len(base),
        "catalog_triple_counts": [len(rows) for rows in catalog_vectors],
        "fixed_sum_rows": len(fixed_sum_rows),
        "fixed_sum_coefficient_rank": fixed_sum_rank,
        "homogeneous_catalog_difference_rows": len(difference_rows),
        "homogeneous_catalog_difference_rank": difference_rank,
        "admissible_factor_pairs": admissible_factor_pairs,
        "distinct_product_sections": len(products),
        "direct_product_quotient_rows": len(quotient_rows),
        "direct_product_quotient_rank": quotient_rank,
        "difference_plus_quotient_rank": combined_rank,
        "all_product_divisors_equal_factor_row_sums": True,
        "all_quotient_rows_equal_two_catalog_row_differences": (
            quotient_rows == decomposed_quotient_rows
        ),
        "quotient_rows_add_no_homogeneous_rank": combined_rank == difference_rank,
    }


def exponent_rows():
    rows = []
    for beta in (0.2, 0.25, 1 / 3, 0.4, 0.5):
        rows.append(
            {
                "factor_base_exponent_beta": beta,
                "uniform_fixed_sum_rows_exponent": 3 * beta - 1,
                "rows_required_exponent": beta,
                "pair_complement_precomputation_exponent": 2 * beta,
                "online_target_lookup_exponent": beta,
                "constant_success_target_pair_exponent": 2 * beta - 1,
                "enough_uniform_rows": 3 * beta - 1 >= beta,
                "constant_expected_target_pairs": 2 * beta - 1 >= 0,
            }
        )
    return rows


def run():
    r61_path = pathlib.Path(
        "p1553_degree_six_product_hypersurface_locator_report_r61.json"
    )
    r61_bytes = r61_path.read_bytes()
    r61_report = json.loads(r61_bytes)
    r67_path = pathlib.Path("p1553_scalar_blind_rank_descent_audit_report_r67.json")
    r67_bytes = r67_path.read_bytes()
    r67_report = json.loads(r67_bytes)
    dependency_source_sha256 = {
        path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
        for path in PINNED_DEPENDENCY_SHA256
    }
    factor_bases = r67.r65.r63.r62.factor_basis_lines(r61_report)
    multiplication_matrix = r61_report["multiplication_map"]["multiplication_matrix"]
    core_base, _ = r67.public_factor_base()
    target = tuple(r67_report["target_input"]["point"])
    expanded_base = core_base + [
        {
            "point": target,
            "source": "caller_supplied_target",
            "atom_index": len(core_base),
        }
    ]
    core = analyze_base(core_base, multiplication_matrix, factor_bases)
    expanded = analyze_base(expanded_base, multiplication_matrix, factor_bases)

    checks = {
        "direct_r61_report_digest_is_pinned": hashlib.sha256(r61_bytes).hexdigest()
        == "f91213021bd350b77cf1ac7d5cb8f2f9ae63b1265a5081d4ed58a3da07aa1f64",
        "direct_r67_report_digest_is_pinned": hashlib.sha256(r67_bytes).hexdigest()
        == "fc47d3d509597fe5b7a0928c5fcffc4cd6c60679ea7efc7d531f4b24b28f143b",
        "all_loaded_dependency_source_digests_are_pinned": (
            dependency_source_sha256 == PINNED_DEPENDENCY_SHA256
        ),
        "core_product_quotients_decompose_into_catalog_differences": core[
            "all_quotient_rows_equal_two_catalog_row_differences"
        ],
        "expanded_product_quotients_decompose_into_catalog_differences": expanded[
            "all_quotient_rows_equal_two_catalog_row_differences"
        ],
        "core_product_quotients_add_no_rank": core[
            "quotient_rows_add_no_homogeneous_rank"
        ],
        "expanded_product_quotients_add_no_rank": expanded[
            "quotient_rows_add_no_homogeneous_rank"
        ],
        "core_counts_match_r67": core["distinct_product_sections"]
        == r67_report["core_precomputation"]["distinct_smooth_product_sections"],
        "expanded_counts_match_r67": expanded["distinct_product_sections"]
        == r67_report["target_extension"]["distinct_smooth_product_sections"],
        "uniform_rank_and_target_threshold_is_beta_one_half": all(
            row["enough_uniform_rows"] == (row["factor_base_exponent_beta"] >= 0.5)
            and row["constant_expected_target_pairs"]
            == (row["factor_base_exponent_beta"] >= 0.5)
            for row in exponent_rows()
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"R68 information-conservation gate failed: {checks}")

    return {
        "schema": "p1553.ffe_fixed_sum_information_conservation.r68.v1",
        "classification": [
            "exact-row-span-theorem-control",
            "toy-replay",
            "model-bound-cost-gate",
            "novelty-unverified",
        ],
        "field_prime": r67.r65.r63.PRIME,
        "prime_subgroup_order": r67.r65.r63.SUBGROUP_ORDER,
        "general_identity": {
            "factor_rows": "u and w are incidence rows of fixed-sum factor triples",
            "product_divisor_row": "v=u+w",
            "product_quotient_row": "v-v0=(u-u0)+(w-w0)",
            "pencil_incidence_consequence": "any relation between catalog product sections is already in the homogeneous span of fixed-sum factor-row differences",
            "new_factor_consequence": "a newly factored pencil output contributes information only through its newly discovered fixed-sum factor rows",
        },
        "core_replay": core,
        "target_expanded_replay": expanded,
        "uniform_cost_model": {
            "constant_public_class_count": "K=Theta(1)",
            "expected_fixed_sum_rows": "Theta(K*B^3/N)",
            "necessary_rank_density": "Theta(K*B^3/N)>=B requires B=Omega(sqrt(N))",
            "pair_complement_precomputation": "Theta(K*B^2)=Omega(N) at the rank threshold",
            "expected_target_pairs": "Theta(K*B^2/N)",
            "online_target_lookup": "Theta(K*B)=Omega(sqrt(N)) for constant expected success",
            "exponent_rows": exponent_rows(),
            "structured_base_escape": "requires prospectively public superuniform additive energy, full rank, and subquadratic source enumeration; this operation is already owned by IDEAs 027, 340, and 389",
        },
        "semantic_dedup": {
            "new_idea_created": False,
            "closest_owners": [
                "ECDLP-IDEA-027 bounded-defect Freiman chart",
                "ECDLP-IDEA-340 BSG energy source chart",
                "ECDLP-IDEA-389 Plunnecke magnification source graph",
            ],
            "disposition": "retain R68 as an IDEA-195 information-conservation receipt; do not create a duplicate structured-base hypothesis",
        },
        "dependency_source_sha256": dependency_source_sha256,
        "checks": checks,
        "result": {
            "ffe_product_relations_add_information_beyond_fixed_sum_rows": False,
            "pencil_factorization_can_only_help_by_finding_new_fixed_sum_rows": True,
            "uniform_model_beats_rho": False,
            "structured_base_candidate_is_semantically_new": False,
            "fresh_target_action_better_than_rho": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "limits": [
            "the row-span identity is exact, but the rank and count replay is one size-103 toy",
            "the uniform density model is not a lower bound for every structured coordinate base",
            "structured additive-energy escapes are ledger-owned and require a public DLP-free chart and source enumerator",
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
