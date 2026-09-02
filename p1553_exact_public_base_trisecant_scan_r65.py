#!/usr/bin/env python3
"""Exhaust all smooth-product trisecants for the tractable R63 B=24 base."""

import argparse
import hashlib
import itertools
import json
import pathlib

import numpy as np

import p1553_public_factor_base_closure_probe_r63 as r63


BASE_SIZE = 24
FINGERPRINT_SEED_1 = 0xA24BAED4963EE407
FINGERPRINT_SEED_2 = 0x9FB21C651E98DF25


def fingerprint_coefficients(seed, count):
    values = []
    state = seed
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = state
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
        value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
        values.append(value ^ (value >> 31))
    return np.asarray(values, dtype=np.uint64)


def plucker_key(first, second):
    coordinates = [
        (first[left] * second[right] - first[right] * second[left]) % r63.PRIME
        for left, right in itertools.combinations(range(6), 2)
    ]
    first_nonzero = next(value for value in coordinates if value)
    inverse = pow(first_nonzero, -1, r63.PRIME)
    return bytes(value * inverse % r63.PRIME for value in coordinates)


def candidate_fingerprint_ranges(sorted_first, sorted_second):
    count = len(sorted_first)
    if not count:
        return []
    changes = np.ones(count, dtype=bool)
    changes[1:] = (sorted_first[1:] != sorted_first[:-1]) | (
        sorted_second[1:] != sorted_second[:-1]
    )
    starts = np.flatnonzero(changes)
    ends = np.append(starts[1:], count)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= 2
    ]


def exhaustive_line_scan(products):
    sections = np.asarray([record["section"] for record in products], dtype=np.int64)
    mask_words = np.asarray(
        [
            [
                (record["mask"] >> (64 * word)) & ((1 << 64) - 1)
                for word in range(4)
            ]
            for record in products
        ],
        dtype=np.uint64,
    )
    count = len(products)
    pair_count = count * (count - 1) // 2
    hashes_first = np.empty(pair_count, dtype=np.uint64)
    hashes_second = np.empty(pair_count, dtype=np.uint64)
    packed_pairs = np.empty(pair_count, dtype=np.uint32)
    coordinate_pairs = tuple(itertools.combinations(range(6), 2))
    left_coordinates = np.asarray([left for left, _ in coordinate_pairs])
    right_coordinates = np.asarray([right for _, right in coordinate_pairs])
    inverses = np.asarray(
        [0] + [pow(value, -1, r63.PRIME) for value in range(1, r63.PRIME)],
        dtype=np.int64,
    )
    coefficients_first = fingerprint_coefficients(
        FINGERPRINT_SEED_1,
        len(coordinate_pairs),
    )
    coefficients_second = fingerprint_coefficients(
        FINGERPRINT_SEED_2,
        len(coordinate_pairs),
    )

    cursor = 0
    disjoint_section_pairs = 0
    for first in range(count - 1):
        tail = sections[first + 1 :]
        coordinates = (
            sections[first, left_coordinates] * tail[:, right_coordinates]
            - sections[first, right_coordinates] * tail[:, left_coordinates]
        ) % r63.PRIME
        first_nonzero_positions = np.argmax(coordinates != 0, axis=1)
        first_nonzero_values = coordinates[
            np.arange(len(tail)),
            first_nonzero_positions,
        ]
        if np.any(first_nonzero_values == 0):
            raise AssertionError("distinct normalized sections became proportional")
        coordinates = coordinates * inverses[first_nonzero_values, None] % r63.PRIME
        coordinates_u64 = coordinates.astype(np.uint64)
        size = len(tail)
        disjoint_section_pairs += int(
            np.count_nonzero(
                ~np.any(mask_words[first] & mask_words[first + 1 :], axis=1)
            )
        )
        hashes_first[cursor : cursor + size] = np.sum(
            coordinates_u64 * coefficients_first,
            axis=1,
            dtype=np.uint64,
        )
        hashes_second[cursor : cursor + size] = np.sum(
            coordinates_u64 * coefficients_second,
            axis=1,
            dtype=np.uint64,
        )
        packed_pairs[cursor : cursor + size] = (
            np.uint32(first << 16)
            | np.arange(first + 1, count, dtype=np.uint32)
        )
        cursor += size
    if cursor != pair_count:
        raise AssertionError("pair fingerprint arrays were not filled")

    order = np.lexsort((hashes_second, hashes_first))
    ranges = candidate_fingerprint_ranges(hashes_first[order], hashes_second[order])
    exact_lines = {}
    fingerprint_collision_pair_groups = 0
    for start, end in ranges:
        local_keys = {}
        for packed in packed_pairs[order[start:end]]:
            first = int(packed >> np.uint32(16))
            second = int(packed & np.uint32(0xFFFF))
            key = plucker_key(products[first]["section"], products[second]["section"])
            local_keys.setdefault(key, set()).update((first, second))
        if len(local_keys) > 1:
            fingerprint_collision_pair_groups += 1
        for key, indices in local_keys.items():
            if len(indices) >= 3:
                exact_lines.setdefault(key, set()).update(indices)

    records = []
    primitive_lines = 0
    primitive_product_points = 0
    primitive_endpoint_pairs = 0
    first_primitive_witness = None
    line_size_histogram = {}
    base_degree_histogram = {}
    for key, indices in sorted(exact_lines.items(), key=lambda item: item[0]):
        sorted_indices = sorted(indices)
        common_mask = products[sorted_indices[0]]["mask"]
        for index in sorted_indices[1:]:
            common_mask &= products[index]["mask"]
        pair_overlaps = [
            (products[first]["mask"] & products[second]["mask"]).bit_count()
            for first, second in itertools.combinations(sorted_indices, 2)
        ]
        base_degree = common_mask.bit_count()
        if any(overlap != base_degree for overlap in pair_overlaps):
            raise AssertionError("pair overlap does not equal the pencil base divisor")
        primitive = base_degree == 0
        line_size = len(sorted_indices)
        line_size_histogram[line_size] = line_size_histogram.get(line_size, 0) + 1
        base_degree_histogram[base_degree] = base_degree_histogram.get(base_degree, 0) + 1
        if primitive:
            primitive_lines += 1
            primitive_product_points += line_size
            primitive_endpoint_pairs += line_size * (line_size - 1) // 2
        record = {
            "product_indices": sorted_indices,
            "catalog_points_on_line": line_size,
            "common_base_divisor_degree": base_degree,
            "primitive_pairwise_disjoint": primitive,
            "factorization_multiplicities": [
                products[index]["factorization_multiplicity"]
                for index in sorted_indices
            ],
        }
        records.append(record)
        if primitive and first_primitive_witness is None:
            first_primitive_witness = {
                **record,
                "sections": [list(products[index]["section"]) for index in sorted_indices],
                "representative_factors": [
                    products[index]["representative_factors"]
                    for index in sorted_indices
                ],
            }

    return {
        "smooth_product_sections": count,
        "section_pairs": pair_count,
        "disjoint_section_pairs": disjoint_section_pairs,
        "fingerprint_candidate_ranges": len(ranges),
        "fingerprint_collision_pair_groups": fingerprint_collision_pair_groups,
        "exact_smooth_trisecant_lines": len(records),
        "primitive_pairwise_disjoint_trisecant_lines": primitive_lines,
        "primitive_product_point_incidences": primitive_product_points,
        "primitive_disjoint_endpoint_pair_incidences": primitive_endpoint_pairs,
        "exact_smooth_closure_rate_per_disjoint_endpoint_pair": (
            primitive_endpoint_pairs / disjoint_section_pairs
            if disjoint_section_pairs
            else None
        ),
        "line_size_histogram": {
            str(size): line_size_histogram[size] for size in sorted(line_size_histogram)
        },
        "base_degree_histogram": {
            str(degree): base_degree_histogram[degree]
            for degree in sorted(base_degree_histogram)
        },
        "first_primitive_witness": first_primitive_witness,
        "line_records": records,
    }


def run():
    r61_path = pathlib.Path(
        "p1553_degree_six_product_hypersurface_locator_report_r61.json"
    )
    r61_bytes = r61_path.read_bytes()
    r61_report = json.loads(r61_bytes)
    r63_script_bytes = pathlib.Path(
        "p1553_public_factor_base_closure_probe_r63.py"
    ).read_bytes()
    r62_script_bytes = pathlib.Path(
        "p1553_degree_six_scan_free_factor_lift_r62.py"
    ).read_bytes()
    multiplication_matrix = r61_report["multiplication_map"]["multiplication_matrix"]
    factor_bases = r63.r62.factor_basis_lines(r61_report)
    public_base = r63.subgroup_points_in_public_order()[:BASE_SIZE]
    catalogs = [
        r63.public_factor_catalog(public_base, class_sum, basis)
        for class_sum, basis in zip(r63.CLASS_SUM_SCALARS, factor_bases)
    ]
    products, _, multiplicities = r63.smooth_products(catalogs, multiplication_matrix)
    unique_lift_counts_by_catalog_multiplicity = {}
    for product in products:
        multiplicity = str(product["factorization_multiplicity"])
        counts = unique_lift_counts_by_catalog_multiplicity.setdefault(
            multiplicity,
            {"accepted_unique_rational_lift": 0, "rejected_nonunique_fiber": 0},
        )
        try:
            r63.r62.rank_one_lift(multiplication_matrix, product["section"])
        except AssertionError:
            counts["rejected_nonunique_fiber"] += 1
        else:
            counts["accepted_unique_rational_lift"] += 1
    scan = exhaustive_line_scan(products)
    projective_factor_section_count = r63.PRIME * r63.PRIME + r63.PRIME + 1
    unique_rational_lift_smooth_sections = sum(
        counts["accepted_unique_rational_lift"]
        for counts in unique_lift_counts_by_catalog_multiplicity.values()
    )
    unconditional_unique_lift_reference_probability = (
        unique_rational_lift_smooth_sections
        / (projective_factor_section_count * projective_factor_section_count)
    )
    conditional_rate = scan["exact_smooth_closure_rate_per_disjoint_endpoint_pair"]
    checks = {
        "direct_r61_report_digest_is_pinned": hashlib.sha256(r61_bytes).hexdigest()
        == "f91213021bd350b77cf1ac7d5cb8f2f9ae63b1265a5081d4ed58a3da07aa1f64",
        "imported_r63_implementation_digest_is_pinned": hashlib.sha256(
            r63_script_bytes
        ).hexdigest()
        == "7f5d061a539d62d2b3d362fed8252fead5b8e14dbc9250f5543e686f3a8d89cb",
        "transitively_used_r62_implementation_digest_is_pinned": hashlib.sha256(
            r62_script_bytes
        ).hexdigest()
        == "15ca1a87202d069b79325b7a1743762d221f0726e793020910fb74c3633d900a",
        "base_size_is_24": len(public_base) == BASE_SIZE,
        "projected_class_triple_counts_match_r63": [
            len(catalog) // 4 for catalog in catalogs
        ]
        == [14, 21],
        "lifted_factor_section_counts_match_r63": [len(catalog) for catalog in catalogs]
        == [56, 84],
        "smooth_product_count_matches_r63": len(products) == 3748,
        "unique_rational_lift_smooth_section_count_is_3656": (
            unique_rational_lift_smooth_sections == 3656
        ),
        "all_section_pairs_were_fingerprinted": scan["section_pairs"]
        == len(products) * (len(products) - 1) // 2,
        "all_fingerprint_candidates_received_exact_key_verification": True,
        "pair_overlap_equals_fixed_base_degree_on_every_line": True,
    }
    if not all(checks.values()):
        raise AssertionError(f"R65 exact public-base trisecant scan failed: {checks}")

    return {
        "schema": "p1553.exact_public_base_trisecant_scan.r65.v1",
        "classification": [
            "toy",
            "exact-exhaustive-incidence-scan",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r63.PRIME,
        "prime_subgroup_order": r63.SUBGROUP_ORDER,
        "public_base_definition": "R63 coordinate-hash prefix",
        "public_base_size": BASE_SIZE,
        "projected_class_triple_counts": [len(catalog) // 4 for catalog in catalogs],
        "even_torsion_lifted_factor_section_counts": [
            len(catalog) for catalog in catalogs
        ],
        "smooth_product_factorizations": multiplicities,
        "unique_lift_counts_by_catalog_multiplicity": (
            unique_lift_counts_by_catalog_multiplicity
        ),
        "exact_incidence_scan": scan,
        "conditional_baseline_comparison": {
            "unconditional_unique_rational_lift_smooth_reference_probability": (
                unconditional_unique_lift_reference_probability
            ),
            "exact_conditional_closure_probability": conditional_rate,
            "exact_to_unconditional_reference_ratio": conditional_rate
            / unconditional_unique_lift_reference_probability,
            "unconditional_reference_to_exact_ratio": (
                unconditional_unique_lift_reference_probability / conditional_rate
            ),
            "uniform_disjoint_endpoint_sample_mean_for_512": 512 * conditional_rate,
            "scope": "exact finite B=24 conditional rate versus an unconditional independent-factor-pair reference; the ratio is descriptive, not a matched conditional null or generic-prime theorem",
        },
        "checks": checks,
        "cost_boundary": {
            "exact_pair_count": scan["section_pairs"],
            "time_complexity_in_smooth_product_count_M": "O(M^2 log M) from lexicographic sorting of C(M,2) fingerprint records",
            "memory_complexity_in_smooth_product_count_M": "Theta(M^2) for two fingerprints and packed pairs",
            "fingerprints_used_only_as_candidate_filter": True,
            "all_candidate_lines_rechecked_with_full_15_coordinate_plucker_key": True,
            "generic_prime_extrapolation": False,
            "inherited_r61_preprocessing": "current toy multiplication-map and sextic setup includes uncharged Theta(N^3) triple enumeration",
            "base_materialization_charged": False,
            "relation_rank_charged": False,
            "projected_scalar_logs_charged": False,
            "linear_algebra_charged": False,
            "target_descent_charged": False,
        },
        "result": {
            "exact_conditional_smooth_trisecant_count_available": True,
            "generic_prime_incidence_theorem": False,
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
            "the exact count is one B=24 public base on one toy curve",
            "quadratic pair materialization is not scalable and supplies no generic-prime incidence law",
            "the inherited R61 multiplication-map and sextic preprocessing includes uncharged Theta(N^3) triple enumeration",
            "base setup, rank, projected logs, linear algebra, and fresh-target descent remain uncharged",
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
