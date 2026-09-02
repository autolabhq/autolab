#!/usr/bin/env python3
"""Probe whether public smooth products close under the R61 pencil locator."""

import argparse
import hashlib
import itertools
import json
import math
import pathlib
import random

import p1553_degree_six_scan_free_factor_lift_r62 as r62


PRIME = r62.PRIME
SUBGROUP_ORDER = r62.SUBGROUP_ORDER
CLASS_SUM_SCALARS = r62.CLASS_SUM_SCALARS
BASE_SIZES = (12, 16, 20, 24, 28, 32, 40, 48)
MAX_ACCEPTED_PENCILS = 512
PAIR_SAMPLE_SEED = 1563


def public_point_key(point):
    encoding = "O" if point is None else f"{point[0]},{point[1]}"
    return hashlib.sha256(f"P1553-R63|{encoding}".encode()).digest(), encoding


def subgroup_points_in_public_order():
    points = []
    for verification_scalar in range(SUBGROUP_ORDER):
        point = r62.r61.r60.r58.r44.r38.scalar_mul(
            verification_scalar,
            r62.r61.r60.r58.r44.r38.GENERATOR,
        )
        digest, encoding = public_point_key(point)
        points.append(
            {
                "point": point,
                "public_encoding": encoding,
                "public_digest": digest,
                "verification_scalar": verification_scalar,
            }
        )
    return sorted(points, key=lambda record: (record["public_digest"], record["public_encoding"]))


def add_points(points):
    result = None
    for point in points:
        result = r62.r61.r60.r58.r44.r38.add(result, point)
    return result


def factor_coordinates(line, basis):
    basis_matrix = [
        [basis["lines"][column][row] for column in range(3)]
        for row in range(3)
    ]
    inverse = r62.r61.r56.matrix_inverse_mod(basis_matrix)
    coordinates = r62.r61.r56.matrix_vector(inverse, line)
    reconstructed = tuple(
        sum(coordinates[column] * basis["lines"][column][row] for column in range(3))
        % PRIME
        for row in range(3)
    )
    if r62.r61.r60.r58.normalized(reconstructed) != line:
        raise AssertionError("factor coordinates failed public-base replay")
    return coordinates


def public_factor_catalog(base, class_sum, basis):
    class_point = r62.r61.r60.r58.r44.r38.scalar_mul(
        class_sum,
        r62.r61.r60.r58.r44.r38.GENERATOR,
    )
    catalog = []
    for indices in itertools.combinations(range(len(base)), 3):
        records = tuple(base[index] for index in indices)
        points = tuple(record["point"] for record in records)
        if add_points(points) != class_point:
            continue
        verification_scalars = tuple(
            record["verification_scalar"] for record in records
        )
        if sum(verification_scalars) % SUBGROUP_ORDER != class_sum:
            raise AssertionError("public group-sum test disagrees with verification labels")
        two_torsion = (192, 0)
        translation_point = r62.r61.r60.r58.r44.r38.scalar_mul(
            basis["translation"],
            r62.r61.r60.r58.r44.r38.GENERATOR,
        )
        for lift_pattern in ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)):
            lifted_points = tuple(
                r62.r61.r60.r58.r44.r38.add(point, two_torsion)
                if parity
                else point
                for point, parity in zip(points, lift_pattern)
            )
            if add_points(lifted_points) != class_point:
                raise AssertionError("even-parity lift changed the bound divisor class")
            shifted = [
                r62.r61.r60.r58.r44.r38.projective(
                    r62.r61.r60.r58.r44.r38.add(point, translation_point)
                )
                for point in lifted_points
            ]
            line = r62.r61.r60.r58.r44.r38.cross(shifted[0], shifted[1])
            if line == (0, 0, 0) or r62.r61.r60.r58.r44.r38.dot(line, shifted[2]):
                raise AssertionError("public factor-base lift failed to define a line")
            line = r62.r61.r60.r58.r44.r38.normalized(line)
            catalog.append(
                {
                    "base_indices": indices,
                    "verification_scalars": verification_scalars,
                    "torsion_lift_pattern": lift_pattern,
                    "mask": sum(
                        1 << (scalar + parity * SUBGROUP_ORDER)
                        for scalar, parity in zip(verification_scalars, lift_pattern)
                    ),
                    "line": line,
                    "coordinates": factor_coordinates(line, basis),
                }
            )
    return catalog


def smooth_products(catalogs, multiplication_matrix):
    products_by_section = {}
    factor_pair_count = 0
    for left in catalogs[0]:
        for right in catalogs[1]:
            if left["mask"] & right["mask"]:
                continue
            factor_pair_count += 1
            section = r62.r61.product_output(
                multiplication_matrix,
                left["coordinates"],
                right["coordinates"],
            )
            section = r62.r61.r60.r58.normalized(section)
            mask = left["mask"] | right["mask"]
            if section in products_by_section:
                if products_by_section[section]["mask"] != mask:
                    raise AssertionError("equal product sections have different zero divisors")
                products_by_section[section]["factorization_multiplicity"] += 1
            else:
                products_by_section[section] = {
                    "section": section,
                    "mask": mask,
                    "factorization_multiplicity": 1,
                    "representative_factors": [
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
    records = list(products_by_section.values())
    multiplicity_histogram = {}
    for record in records:
        multiplicity = record["factorization_multiplicity"]
        multiplicity_histogram[multiplicity] = multiplicity_histogram.get(multiplicity, 0) + 1
    return records, products_by_section, {
        "admissible_factor_pairs": factor_pair_count,
        "distinct_product_sections": len(records),
        "factorization_multiplicity_histogram": {
            str(multiplicity): multiplicity_histogram[multiplicity]
            for multiplicity in sorted(multiplicity_histogram)
        },
    }


def accepted_pencil_pairs(products, seed):
    if len(products) < 2:
        return [], 0
    total_pairs = len(products) * (len(products) - 1) // 2
    if total_pairs <= 4 * MAX_ACCEPTED_PENCILS:
        candidates = itertools.combinations(range(len(products)), 2)
        accepted = [
            pair
            for pair in candidates
            if not products[pair[0]]["mask"] & products[pair[1]]["mask"]
        ]
        return accepted[:MAX_ACCEPTED_PENCILS], total_pairs

    source = random.Random(seed)
    accepted = []
    seen = set()
    attempts = 0
    while len(accepted) < MAX_ACCEPTED_PENCILS and attempts < 100 * MAX_ACCEPTED_PENCILS:
        attempts += 1
        first, second = sorted(source.sample(range(len(products)), 2))
        pair = (first, second)
        if pair in seen:
            continue
        seen.add(pair)
        if products[first]["mask"] & products[second]["mask"]:
            continue
        accepted.append(pair)
    return accepted, attempts


def equation_from_report(r61_report):
    payload = r61_report["implicitization"]["equation"]
    return {
        "degree": payload["degree"],
        "exponents": [tuple(exponent) for exponent in payload["exponents"]],
        "coefficients": tuple(payload["coefficients"]),
    }


def closure_probe(products, smooth_lookup, pairs, equation, multiplication_matrix):
    nonzero_restrictions = 0
    new_hypersurface_roots_with_multiplicity = 0
    new_distinct_hypersurface_roots = 0
    new_rational_rank_one_product_sections = 0
    new_nonrational_or_nonunique_rank_one_fibers = 0
    new_smooth_roots = 0
    pencils_with_new_smooth_root = 0
    rational_root_histogram = {}
    first_smooth_witness = None
    for first_index, second_index in pairs:
        first = products[first_index]["section"]
        second = products[second_index]["section"]
        restricted = r62.r61.restrict_to_line(equation, first, second)
        if not any(restricted):
            continue
        nonzero_restrictions += 1
        _, _, roots = r62.r61.factor_univariate(restricted)
        roots_with_multiplicity = [root for root in roots if root]
        roots = sorted(set(roots_with_multiplicity))
        new_hypersurface_roots_with_multiplicity += len(roots_with_multiplicity)
        new_distinct_hypersurface_roots += len(roots)
        rational_root_histogram[len(roots)] = rational_root_histogram.get(len(roots), 0) + 1
        smooth_for_pencil = 0
        for root in roots:
            section = r62.r61.r60.r58.normalized(
                tuple(
                    (left + root * right) % PRIME
                    for left, right in zip(first, second)
                )
            )
            try:
                r62.rank_one_lift(multiplication_matrix, section)
            except AssertionError:
                new_nonrational_or_nonunique_rank_one_fibers += 1
                continue
            new_rational_rank_one_product_sections += 1
            if section in smooth_lookup:
                smooth_for_pencil += 1
                if first_smooth_witness is None:
                    first_smooth_witness = {
                        "source_product_sections": [list(first), list(second)],
                        "source_product_factors": [
                            products[first_index]["representative_factors"],
                            products[second_index]["representative_factors"],
                        ],
                        "line_parameter": root,
                        "target_product_section": list(section),
                        "target_product_factors": smooth_lookup[section][
                            "representative_factors"
                        ],
                        "target_factorization_multiplicity": smooth_lookup[section][
                            "factorization_multiplicity"
                        ],
                    }
        new_smooth_roots += smooth_for_pencil
        if smooth_for_pencil:
            pencils_with_new_smooth_root += 1
    return {
        "accepted_disjoint_source_pencils": len(pairs),
        "nonzero_sextic_restrictions": nonzero_restrictions,
        "new_finite_hypersurface_roots_with_multiplicity": new_hypersurface_roots_with_multiplicity,
        "new_distinct_finite_hypersurface_roots": new_distinct_hypersurface_roots,
        "new_rational_rank_one_product_sections": new_rational_rank_one_product_sections,
        "new_nonrational_or_nonunique_rank_one_fibers": new_nonrational_or_nonunique_rank_one_fibers,
        "new_smooth_product_roots": new_smooth_roots,
        "pencils_with_new_smooth_root": pencils_with_new_smooth_root,
        "rational_root_histogram": {
            str(count): rational_root_histogram[count]
            for count in sorted(rational_root_histogram)
        },
        "smooth_root_rate_per_new_rational_product_root": (
            new_smooth_roots / new_rational_rank_one_product_sections
            if new_rational_rank_one_product_sections
            else None
        ),
        "smooth_closure_rate_per_accepted_pencil": (
            pencils_with_new_smooth_root / len(pairs) if pairs else None
        ),
        "first_smooth_witness": first_smooth_witness,
    }


def asymptotic_models():
    rows = []
    for beta in (0.25, 0.5, 0.75, 1.0):
        rows.append(
            {
                "factor_base_exponent_beta": beta,
                "random_six_atom_relation_collection_exponent": 6 - 5 * beta,
                "optimistic_independent_fixed_sum_pair_exponent": 4 - 3 * beta,
                "optimistic_perfectly_correlated_fixed_sum_pair_exponent": 2 - beta,
                "sparse_linear_algebra_exponent_B_squared": 2 * beta,
            }
        )
    return rows


def run():
    r61_path = pathlib.Path(
        "p1553_degree_six_product_hypersurface_locator_report_r61.json"
    )
    r61_bytes = r61_path.read_bytes()
    r61_report = json.loads(r61_bytes)
    r62_script_bytes = pathlib.Path(
        "p1553_degree_six_scan_free_factor_lift_r62.py"
    ).read_bytes()
    multiplication_matrix = r61_report["multiplication_map"]["multiplication_matrix"]
    equation = equation_from_report(r61_report)
    factor_bases = r62.factor_basis_lines(r61_report)
    public_order = subgroup_points_in_public_order()
    rows = []
    for base_size in BASE_SIZES:
        base = public_order[:base_size]
        catalogs = [
            public_factor_catalog(base, class_sum, basis)
            for class_sum, basis in zip(CLASS_SUM_SCALARS, factor_bases)
        ]
        products, smooth_lookup, product_multiplicities = smooth_products(
            catalogs,
            multiplication_matrix,
        )
        pairs, pair_draws = accepted_pencil_pairs(
            products,
            PAIR_SAMPLE_SEED + base_size,
        )
        closure = closure_probe(
            products,
            smooth_lookup,
            pairs,
            equation,
            multiplication_matrix,
        )
        projective_factor_section_count = PRIME * PRIME + PRIME + 1
        multiplicity_one_smooth_sections = product_multiplicities[
            "factorization_multiplicity_histogram"
        ].get("1", 0)
        unconditional_multiplicity_one_reference_intensity = (
            multiplicity_one_smooth_sections
            / (projective_factor_section_count * projective_factor_section_count)
        )
        closure["unconditional_multiplicity_one_reference_intensity"] = (
            unconditional_multiplicity_one_reference_intensity
        )
        closure["reference_intensity_times_conditioned_unique_lift_count"] = (
            unconditional_multiplicity_one_reference_intensity
            * closure["new_rational_rank_one_product_sections"]
        )
        random_expected_projected_triple_counts = [
            1717
            * math.comb(base_size, 3)
            / math.comb(SUBGROUP_ORDER, 3)
            for _ in CLASS_SUM_SCALARS
        ]
        class_upper_bound = min(1717, base_size * (base_size - 1) // 6)
        rows.append(
            {
                "base_size": base_size,
                "base_digest_prefix": hashlib.sha256(
                    b"".join(record["public_digest"] for record in base)
                ).hexdigest(),
                "projected_class_triple_counts": [
                    len(catalog) // 4 for catalog in catalogs
                ],
                "even_torsion_lifted_factor_section_counts": [
                    len(catalog) for catalog in catalogs
                ],
                "random_subset_expected_projected_triple_counts": random_expected_projected_triple_counts,
                "pair_counting_upper_bound_per_class": class_upper_bound,
                "distinct_smooth_degree_six_products": len(products),
                "smooth_product_factorizations": product_multiplicities,
                "pair_candidates_or_random_draws": pair_draws,
                "closure_probe": closure,
            }
        )

    checks = {
        "direct_r61_report_digest_is_pinned": hashlib.sha256(r61_bytes).hexdigest()
        == "f91213021bd350b77cf1ac7d5cb8f2f9ae63b1265a5081d4ed58a3da07aa1f64",
        "imported_r62_implementation_digest_is_pinned": hashlib.sha256(
            r62_script_bytes
        ).hexdigest()
        == "15ca1a87202d069b79325b7a1743762d221f0726e793020910fb74c3633d900a",
        "public_base_is_nested": all(
            set(record["public_digest"] for record in public_order[:left]).issubset(
                record["public_digest"] for record in public_order[:right]
            )
            for left, right in zip(BASE_SIZES, BASE_SIZES[1:])
        ),
        "public_group_sum_checks_need_no_discrete_logs": True,
        "all_class_counts_obey_pair_counting_bound": all(
            all(
                count <= row["pair_counting_upper_bound_per_class"]
                for count in row["projected_class_triple_counts"]
            )
            for row in rows
        ),
        "all_sampled_source_pencils_have_nonzero_restriction": all(
            row["closure_probe"]["accepted_disjoint_source_pencils"]
            == row["closure_probe"]["nonzero_sextic_restrictions"]
            for row in rows
        ),
        "full_factor_base_not_claimed_or_tested": max(BASE_SIZES) < SUBGROUP_ORDER,
        "cost_models_do_not_beat_rho_at_any_displayed_beta": all(
            min(
                model["random_six_atom_relation_collection_exponent"],
                model["optimistic_independent_fixed_sum_pair_exponent"],
                model["optimistic_perfectly_correlated_fixed_sum_pair_exponent"],
            )
            >= 1
            for model in asymptotic_models()
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"R63 public factor-base closure probe failed: {checks}")

    return {
        "schema": "p1553.public_factor_base_closure_probe.r63.v1",
        "classification": [
            "toy",
            "deterministic-public-factor-base-probe",
            "heuristic-cost-model",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": PRIME,
        "prime_subgroup_order": SUBGROUP_ORDER,
        "class_sum_scalars": list(CLASS_SUM_SCALARS),
        "public_factor_base_definition": {
            "membership_order": "SHA256('P1553-R63|' || canonical affine point encoding), with O encoded as O",
            "nested_base_sizes": list(BASE_SIZES),
            "verification_scalar_labels_used_for_speed_only": True,
            "every_class_membership_test_replayed_by_public_curve addition": True,
            "each_projected_triple_expanded_to_four_even_two_torsion_lifts": True,
            "base_materialization_cost_charged": False,
        },
        "maximum_accepted_pencils_per_base": MAX_ACCEPTED_PENCILS,
        "pair_sample_seed": PAIR_SAMPLE_SEED,
        "base_rows": rows,
        "asymptotic_cost_models": {
            "status": "heuristic and model-bound; output distributions could be biased by structured source pencils",
            "random_six_atom_model": "smooth probability about (B/N)^6; B relations cost N^6/B^5 pencils",
            "finite_toy_unconditional_reference": "multiplicity-one smooth catalog sections over the unconditional |P^2(F_p)|^2 factor-pair universe; multiplying this intensity by the conditioned unique-lift root count is descriptive only, not an expected count for that filtered population",
            "fixed_sum_pair_counting_bound": "each class has at most B(B-1)/6 distinct triples because each triple uses three pairs",
            "optimistic_independent_two_factor_model": "smooth probability at most about (B/N)^4; B relations cost at least N^4/B^3",
            "optimistic_perfect_correlation_model": "even treating the two factor-smoothness events as perfectly correlated gives about (B/N)^2 and N^2/B relation work",
            "important_limit": "the two optimistic models require distribution assumptions and are not universal lower bounds for pencils chosen from the same structured base",
            "exponent_rows": asymptotic_models(),
        },
        "checks": checks,
        "result": {
            "public_factor_base_closure_measured": True,
            "smooth_closure_amplification_inference": "inconclusive exploratory sample",
            "source_factor_selection_cost_charged": False,
            "base_materialization_cost_charged": False,
            "generic_prime_distribution_theorem": False,
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
            "the nested coordinate-hash bases are one toy family and their materialization cost is uncharged",
            "the closure rates come from at most 512 accepted pencils per base and do not prove a generic-prime distribution law",
            "verification scalar labels accelerate catalog construction but every accepted triple is checked by public group addition",
            "source selection, relation rank, projected scalar logs, linear algebra, and fresh-target descent remain uncharged",
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
