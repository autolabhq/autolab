#!/usr/bin/env python3
"""Exhaust representatives of every maximal fixed-class product orbit."""

import collections
import hashlib
import itertools
import json

import p1553_fixed_class_multiplication_kernel_scan_r56 as r56


REFLECTION_OFFSET_SUM = 3 * 80


def all_factors_by_residue():
    factors = collections.defaultdict(list)
    for window in range(r56.r54.WINDOW_COUNT):
        points = range(
            r56.r54.WINDOW_SIZE * window,
            r56.r54.WINDOW_SIZE * (window + 1),
        )
        for block in itertools.combinations(points, r56.r54.FACTOR_SIZE):
            mask = sum(1 << point for point in block)
            factors[sum(block) % r56.r54.r44.r38.SUBGROUP_ORDER].append(mask)
    return factors


def product_masks(first, second, third):
    masks = set()
    for first_mask in first:
        for second_mask in second:
            if first_mask & second_mask:
                continue
            first_two = first_mask | second_mask
            for third_mask in third:
                if first_two & third_mask:
                    continue
                masks.add(first_two | third_mask)
    return masks


def maximal_class_triples(factors):
    residues = sorted(factors)
    maximum = -1
    triples = []
    candidate_count = 0
    for triple in itertools.combinations_with_replacement(residues, 3):
        if sum(triple) % r56.r54.r44.r38.SUBGROUP_ORDER != r56.TOTAL_CLASS_RESIDUE:
            continue
        products = product_masks(*(factors[residue] for residue in triple))
        if not products:
            continue
        candidate_count += 1
        product_count = len(products)
        if product_count > maximum:
            maximum = product_count
            triples = [triple]
        elif product_count == maximum:
            triples.append(triple)
    return candidate_count, maximum, triples


def reflected_triple(triple):
    return tuple(
        sorted(
            (REFLECTION_OFFSET_SUM - residue)
            % r56.r54.r44.r38.SUBGROUP_ORDER
            for residue in triple
        )
    )


def orbit_key(triple):
    reflected = reflected_triple(triple)
    return min(tuple(sorted(triple)), reflected)


def classify_representative(triple):
    catalogs = r56.factor_catalogs(triple)
    multiplication = r56.build_multiplication_data(catalogs)
    line_scan = r56.exhaustive_line_scan(multiplication["products"])
    base_degrees = collections.Counter(
        line["common_base_divisor_degree"] for line in line_scan["line_records"]
    )
    line_sizes = collections.Counter(
        line["catalog_points_on_line"] for line in line_scan["line_records"]
    )
    varying_factors = collections.Counter()
    primitive_lines = 0
    for line in line_scan["line_records"]:
        varying = sum(
            len({indices[position] for indices in line["factor_indices"]}) > 1
            for position in range(3)
        )
        line["varying_factor_count"] = varying
        varying_factors[varying] += 1
        primitive_lines += line["primitive_coprime"]
    digest = hashlib.sha256(
        json.dumps(
            line_scan["line_records"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "class_residues": list(triple),
        "reflected_class_residues": list(reflected_triple(triple)),
        "factor_counts": [len(catalog) for catalog in catalogs],
        "product_sections": len(multiplication["products"]),
        "multiplication_rank": multiplication["multiplication_rank"],
        "multiplication_kernel_dimension": 27
        - multiplication["multiplication_rank"],
        "trivialization_ratio_support": multiplication[
            "trivialization_ratio_support"
        ],
        "section_pairs": line_scan["section_pairs"],
        "exact_catalog_trisecant_lines": line_scan[
            "exact_catalog_trisecant_lines"
        ],
        "line_size_histogram": {
            str(size): line_sizes[size] for size in sorted(line_sizes)
        },
        "common_base_divisor_degree_histogram": {
            str(degree): base_degrees[degree] for degree in sorted(base_degrees)
        },
        "varying_factor_count_histogram": {
            str(count): varying_factors[count] for count in sorted(varying_factors)
        },
        "primitive_coprime_trisecant_lines": primitive_lines,
        "line_classification_sha256": digest,
    }


def run():
    factors = all_factors_by_residue()
    candidate_count, maximum_products, maximal_triples = maximal_class_triples(
        factors
    )
    orbit_members = collections.defaultdict(list)
    for triple in maximal_triples:
        orbit_members[orbit_key(triple)].append(triple)
    representatives = sorted(orbit_members)
    classifications = [
        classify_representative(representative)
        for representative in representatives
    ]

    checks = {
        "factor_residue_count_is_97": len(factors) == 97,
        "maximum_product_count_is_3136": maximum_products == 3_136,
        "maximal_unordered_class_triples_match": maximal_triples
        == [
            (16, 42, 96),
            (17, 41, 96),
            (17, 42, 95),
            (18, 41, 95),
        ],
        "maximal_class_triples_form_three_reflection_orbits": len(
            representatives
        )
        == 3,
        "every_representative_has_3136_products": all(
            item["product_sections"] == 3_136 for item in classifications
        ),
        "every_multiplication_matrix_has_rank_nine_kernel_eighteen": all(
            item["multiplication_rank"] == 9
            and item["multiplication_kernel_dimension"] == 18
            for item in classifications
        ),
        "every_representative_pair_space_exhausted": all(
            item["section_pairs"] == 3_136 * 3_135 // 2
            for item in classifications
        ),
        "every_trisecant_varies_exactly_one_factor": all(
            item["varying_factor_count_histogram"]
            == {"1": item["exact_catalog_trisecant_lines"]}
            for item in classifications
        ),
        "no_primitive_coprime_trisecant_in_any_representative": all(
            item["primitive_coprime_trisecant_lines"] == 0
            for item in classifications
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"R57 maximal class orbit scan failed: {checks}")

    return {
        "schema": "p1553.maximal_class_orbit_scan.r57.v1",
        "classification": [
            "toy",
            "exact-maximal-class-enumeration",
            "exhaustive-orbit-representative-product-lines",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r56.r54.PRIME,
        "subgroup_order": r56.r54.r44.r38.SUBGROUP_ORDER,
        "factor_residues": len(factors),
        "nonempty_unordered_class_triples": candidate_count,
        "maximum_product_sections": maximum_products,
        "maximal_unordered_class_triples": [
            list(triple) for triple in maximal_triples
        ],
        "reflection": {
            "point_map": "offset -> 80-offset",
            "class_map": "residue -> 240-residue mod 103",
            "orbit_representatives": [
                list(representative) for representative in representatives
            ],
            "orbit_members": {
                "_".join(map(str, representative)): [
                    list(member) for member in sorted(orbit_members[representative])
                ]
                for representative in representatives
            },
        },
        "representative_classifications": classifications,
        "checks": checks,
        "result": {
            "primitive_coprime_trisecant_found": False,
            "maximal_product_class_orbits_exhausted": True,
            "lower_product_class_triples_exhausted": False,
            "asymptotic_product_line_theorem": False,
            "asymptotic_interval_pencil_family": False,
            "fresh_target_action": False,
            "r10_queried_coefficients": False,
            "independent_rank": False,
            "factor_base_logs": False,
            "scalar_blind_descent": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "limits": [
            "the orbit closeout covers only the four maximal 3136-product class triples in the frozen toy",
            "lower-product class triples and factors outside the within-window catalog remain open",
            "the reflection reduction uses the frozen path involution and does not imply an asymptotic theorem",
            "no target locator, R10 output, rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
