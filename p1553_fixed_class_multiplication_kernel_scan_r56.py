#!/usr/bin/env python3
"""Build one fixed-class multiplication tensor and exhaust product trisecants."""

import collections
import hashlib
import itertools
import json

import numpy as np

import p1553_catalog_pencil_fiber_scan_r46 as r46
import p1553_coprime_product_trisecant_probe_r54 as r54


CLASS_RESIDUES = (96, 42, 16)
TOTAL_CLASS_RESIDUE = r54.PRODUCT_OFFSET_SUM % r54.r44.r38.SUBGROUP_ORDER
FINGERPRINT_SEED_1 = 0x9E3779B97F4A7C15
FINGERPRINT_SEED_2 = 0xD1B54A32D192ED03


def factor_catalogs(class_residues=CLASS_RESIDUES):
    catalogs = {residue: [] for residue in class_residues}
    inverse_degree = pow(r54.FACTOR_SIZE, -1, r54.r44.r38.SUBGROUP_ORDER)
    for window in range(r54.WINDOW_COUNT):
        points = range(r54.WINDOW_SIZE * window, r54.WINDOW_SIZE * (window + 1))
        for block in itertools.combinations(points, r54.FACTOR_SIZE):
            residue = sum(block) % r54.r44.r38.SUBGROUP_ORDER
            if residue not in catalogs:
                continue
            scalar_block = tuple(
                (r54.PATH_START + r54.PATH_STEP * offset)
                % r54.r44.r38.SUBGROUP_ORDER
                for offset in block
            )
            abel_sum = sum(scalar_block) % r54.r44.r38.SUBGROUP_ORDER
            translation = (-abel_sum * inverse_degree) % r54.r44.r38.SUBGROUP_ORDER
            line = r54.r44.r38.fiber_line(scalar_block, translation)
            if line is None:
                raise AssertionError("fixed-class factor did not define a line section")
            catalogs[residue].append(
                {
                    "block": block,
                    "mask": sum(1 << point for point in block),
                    "line": line,
                    "translation": translation,
                    "abel_sum_scalar": abel_sum,
                }
            )
    return [catalogs[residue] for residue in class_residues]


def matrix_inverse_mod(matrix):
    size = len(matrix)
    augmented = [
        [value % r54.PRIME for value in row]
        + [int(row_index == column) for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            row
            for row in range(column, size)
            if augmented[row][column] % r54.PRIME
        )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        inverse = pow(augmented[column][column] % r54.PRIME, -1, r54.PRIME)
        augmented[column] = [
            value * inverse % r54.PRIME for value in augmented[column]
        ]
        for row in range(size):
            if row == column:
                continue
            scale = augmented[row][column] % r54.PRIME
            if not scale:
                continue
            augmented[row] = [
                (left - scale * right) % r54.PRIME
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [row[size:] for row in augmented]


def matrix_vector(matrix, vector):
    return tuple(
        sum(left * right for left, right in zip(row, vector)) % r54.PRIME
        for row in matrix
    )


def normalized(vector):
    first_nonzero = next(value for value in vector if value % r54.PRIME)
    inverse = pow(first_nonzero % r54.PRIME, -1, r54.PRIME)
    return tuple(value * inverse % r54.PRIME for value in vector)


def choose_factor_basis(catalog):
    for indices in itertools.combinations(range(len(catalog)), 3):
        lines = [catalog[index]["line"] for index in indices]
        if r54.r44.section_rank(lines) == 3:
            basis_matrix = [
                [lines[column][row] for column in range(3)]
                for row in range(3)
            ]
            inverse = matrix_inverse_mod(basis_matrix)
            coordinates = [
                matrix_vector(inverse, factor["line"]) for factor in catalog
            ]
            if not all(
                normalized(
                    tuple(
                        sum(
                            coordinate[column] * lines[column][row]
                            for column in range(3)
                        )
                        % r54.PRIME
                        for row in range(3)
                    )
                )
                == factor["line"]
                for factor, coordinate in zip(catalog, coordinates)
            ):
                raise AssertionError("factor basis coordinates failed replay")
            return indices, lines, coordinates
    raise AssertionError("fixed factor class did not span its degree-three space")


def product_value(factors, scalar):
    value = 1
    for factor in factors:
        point = r54.r44.r38.projective(
            r54.r44.r38.scalar_mul(
                (scalar + factor["translation"])
                % r54.r44.r38.SUBGROUP_ORDER,
                r54.r44.r38.GENERATOR,
            )
        )
        value = value * r54.r44.r38.dot(factor["line"], point) % r54.PRIME
    return value


def normalized_product_section(factors):
    block = tuple(sorted(point for factor in factors for point in factor["block"]))
    section, total_translation = r54.r44.block_section(
        block,
        r54.PATH_START,
        r54.PATH_STEP,
    )
    return tuple(section), total_translation, block


def product_trivialization_ratio(factors):
    section, total_translation, _ = normalized_product_section(factors)
    ratios = {}
    for scalar in range(r54.r44.r38.SUBGROUP_ORDER):
        product = product_value(factors, scalar)
        section_value = r54.r44.r38.dot(
            section,
            r54.r44.SECTION_EMBEDDINGS[
                (scalar + total_translation) % r54.r44.r38.SUBGROUP_ORDER
            ],
        )
        if product == 0 and section_value == 0:
            continue
        if product == 0 or section_value == 0:
            raise AssertionError("reference product and section zero sets differ")
        ratios[scalar] = product * pow(section_value, -1, r54.PRIME) % r54.PRIME
    if len(ratios) < r54.r44.r38.SUBGROUP_ORDER - r54.PRODUCT_SIZE:
        raise AssertionError("reference trivialization ratio has too little support")
    return ratios


def scaled_product_section(factors, trivialization_ratio):
    section, total_translation, block = normalized_product_section(factors)
    scale = None
    for scalar in range(r54.r44.r38.SUBGROUP_ORDER):
        ratio = trivialization_ratio.get(scalar)
        if ratio is None:
            continue
        product = product_value(factors, scalar)
        section_value = r54.r44.r38.dot(
            section,
            r54.r44.SECTION_EMBEDDINGS[
                (scalar + total_translation) % r54.r44.r38.SUBGROUP_ORDER
            ],
        )
        if product == 0 and section_value == 0:
            continue
        if product == 0 or section_value == 0:
            raise AssertionError("product and degree-nine section zero sets differ")
        candidate = (
            product
            * pow(section_value, -1, r54.PRIME)
            * pow(ratio, -1, r54.PRIME)
            % r54.PRIME
        )
        if scale is None:
            scale = candidate
        elif scale != candidate:
            raise AssertionError("product scaling is not constant")
    if scale is None:
        raise AssertionError("product section vanished everywhere")
    return (
        tuple(scale * value % r54.PRIME for value in section),
        tuple(section),
        block,
    )


def build_multiplication_data(catalogs):
    basis_data = [choose_factor_basis(catalog) for catalog in catalogs]
    basis_indices = [data[0] for data in basis_data]
    factor_coordinates = [data[2] for data in basis_data]

    reference_factors = tuple(
        catalogs[index][basis_indices[index][0]] for index in range(3)
    )
    trivialization_ratio = product_trivialization_ratio(reference_factors)

    multiplication_columns = []
    for first, second, third in itertools.product(range(3), repeat=3):
        factors = (
            catalogs[0][basis_indices[0][first]],
            catalogs[1][basis_indices[1][second]],
            catalogs[2][basis_indices[2][third]],
        )
        scaled_section, _, _ = scaled_product_section(
            factors,
            trivialization_ratio,
        )
        multiplication_columns.append(scaled_section)

    products = []
    section_index = {}
    for indices in itertools.product(
        range(len(catalogs[0])),
        range(len(catalogs[1])),
        range(len(catalogs[2])),
    ):
        factors = tuple(catalog[index] for catalog, index in zip(catalogs, indices))
        if any(
            factors[left]["mask"] & factors[right]["mask"]
            for left, right in itertools.combinations(range(3), 2)
        ):
            continue
        actual_scaled, actual_normalized, block = scaled_product_section(
            factors,
            trivialization_ratio,
        )
        tensor_coordinates = [
            factor_coordinates[0][indices[0]][first]
            * factor_coordinates[1][indices[1]][second]
            * factor_coordinates[2][indices[2]][third]
            % r54.PRIME
            for first, second, third in itertools.product(range(3), repeat=3)
        ]
        reconstructed = tuple(
            sum(
                tensor_coordinates[column] * multiplication_columns[column][row]
                for column in range(27)
            )
            % r54.PRIME
            for row in range(9)
        )
        if normalized(reconstructed) != actual_normalized:
            raise AssertionError("multiplication tensor did not reconstruct product")
        section_bytes = bytes(actual_normalized)
        if section_bytes in section_index:
            raise AssertionError("fixed-class product divisor collision")
        section_index[section_bytes] = len(products)
        products.append(
            {
                "factor_indices": indices,
                "block": block,
                "mask": sum(1 << point for point in block),
                "section": section_bytes,
                "scaled_section": actual_scaled,
            }
        )

    return {
        "basis_indices": basis_indices,
        "factor_coordinates": factor_coordinates,
        "multiplication_columns": multiplication_columns,
        "multiplication_rank": r54.r44.section_rank(multiplication_columns),
        "trivialization_ratio_support": len(trivialization_ratio),
        "products": products,
        "section_index": section_index,
    }


def fingerprint_coefficients(seed):
    values = []
    state = seed
    for _ in range(36):
        state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = state
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
        value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
        values.append(value ^ (value >> 31))
    return np.asarray(values, dtype=np.uint64)


def exhaustive_line_scan(products):
    sections = np.asarray([list(product["section"]) for product in products], dtype=np.int64)
    count = len(products)
    pair_count = count * (count - 1) // 2
    hashes_first = np.empty(pair_count, dtype=np.uint64)
    hashes_second = np.empty(pair_count, dtype=np.uint64)
    packed_pairs = np.empty(pair_count, dtype=np.uint32)
    coordinate_pairs = tuple(itertools.combinations(range(9), 2))
    left_coordinates = np.asarray([left for left, _ in coordinate_pairs])
    right_coordinates = np.asarray([right for _, right in coordinate_pairs])
    inverses = np.asarray(
        [0] + [pow(value, -1, r54.PRIME) for value in range(1, r54.PRIME)],
        dtype=np.int64,
    )
    coeff_first = fingerprint_coefficients(FINGERPRINT_SEED_1)
    coeff_second = fingerprint_coefficients(FINGERPRINT_SEED_2)

    cursor = 0
    for first in range(count - 1):
        tail = sections[first + 1 :]
        coordinates = (
            sections[first, left_coordinates] * tail[:, right_coordinates]
            - sections[first, right_coordinates] * tail[:, left_coordinates]
        ) % r54.PRIME
        first_nonzero_positions = np.argmax(coordinates != 0, axis=1)
        first_nonzero_values = coordinates[
            np.arange(len(tail)), first_nonzero_positions
        ]
        coordinates = (
            coordinates * inverses[first_nonzero_values, np.newaxis]
        ) % r54.PRIME
        coordinates_u64 = coordinates.astype(np.uint64)
        size = len(tail)
        hashes_first[cursor : cursor + size] = np.sum(
            coordinates_u64 * coeff_first,
            axis=1,
            dtype=np.uint64,
        )
        hashes_second[cursor : cursor + size] = np.sum(
            coordinates_u64 * coeff_second,
            axis=1,
            dtype=np.uint64,
        )
        packed_pairs[cursor : cursor + size] = (
            np.uint32(first << 16)
            | np.arange(first + 1, count, dtype=np.uint32)
        )
        cursor += size
    if cursor != pair_count:
        raise AssertionError("fixed-class pair array was not filled")

    order = np.lexsort((hashes_second, hashes_first))
    sorted_first = hashes_first[order]
    sorted_second = hashes_second[order]
    equal_previous = np.zeros(pair_count, dtype=bool)
    equal_previous[1:] = (
        (sorted_first[1:] == sorted_first[:-1])
        & (sorted_second[1:] == sorted_second[:-1])
    )
    duplicate_positions = np.flatnonzero(equal_previous)

    candidate_ranges = []
    for position in duplicate_positions:
        start = position - 1
        while start > 0 and equal_previous[start]:
            start -= 1
        end = position + 1
        while end < pair_count and equal_previous[end]:
            end += 1
        candidate_ranges.append((start, end))
    candidate_ranges = sorted(set(candidate_ranges))

    exact_lines = {}
    for start, end in candidate_ranges:
        for packed in packed_pairs[order[start:end]]:
            first = int(packed >> np.uint32(16))
            second = int(packed & np.uint32(0xFFFF))
            key = r46.plucker_key(
                products[first]["section"],
                products[second]["section"],
            )
            exact_lines.setdefault(key, set()).update((first, second))
    exact_lines = {
        key: indices for key, indices in exact_lines.items() if len(indices) >= 3
    }

    line_records = []
    for key, indices in sorted(exact_lines.items(), key=lambda item: item[0]):
        sorted_indices = sorted(indices)
        common_mask = products[sorted_indices[0]]["mask"]
        for index in sorted_indices[1:]:
            common_mask &= products[index]["mask"]
        pair_overlaps = [
            (
                products[first]["mask"] & products[second]["mask"]
            ).bit_count()
            for first, second in itertools.combinations(sorted_indices, 2)
        ]
        line_records.append(
            {
                "product_indices": sorted_indices,
                "factor_indices": [
                    list(products[index]["factor_indices"])
                    for index in sorted_indices
                ],
                "blocks": [
                    list(products[index]["block"]) for index in sorted_indices
                ],
                "catalog_points_on_line": len(sorted_indices),
                "common_base_divisor_degree": common_mask.bit_count(),
                "pair_overlap_histogram": {
                    str(overlap): pair_overlaps.count(overlap)
                    for overlap in sorted(set(pair_overlaps))
                },
                "primitive_coprime": common_mask == 0
                and all(overlap == 0 for overlap in pair_overlaps),
            }
        )

    return {
        "section_pairs": pair_count,
        "fingerprint_collision_ranges": len(candidate_ranges),
        "exact_catalog_trisecant_lines": len(line_records),
        "line_records": line_records,
    }


def run():
    catalogs = factor_catalogs()
    multiplication = build_multiplication_data(catalogs)
    line_scan = exhaustive_line_scan(multiplication["products"])
    primitive_lines = [
        line
        for line in line_scan["line_records"]
        if line["primitive_coprime"]
    ]
    base_degree_histogram = collections.Counter(
        line["common_base_divisor_degree"]
        for line in line_scan["line_records"]
    )
    line_size_histogram = collections.Counter(
        line["catalog_points_on_line"] for line in line_scan["line_records"]
    )
    varying_factor_count_histogram = collections.Counter()
    for line in line_scan["line_records"]:
        varying_factors = sum(
            len({indices[position] for indices in line["factor_indices"]}) > 1
            for position in range(3)
        )
        line["varying_factor_count"] = varying_factors
        varying_factor_count_histogram[varying_factors] += 1
    line_classification_sha256 = hashlib.sha256(
        json.dumps(
            line_scan["line_records"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    representative_lines = []
    for degree in sorted(base_degree_histogram):
        representative_lines.extend(
            [
                line
                for line in line_scan["line_records"]
                if line["common_base_divisor_degree"] == degree
            ][:3]
        )
    checks = {
        "class_residues_sum_to_product_class": sum(CLASS_RESIDUES)
        % r54.r44.r38.SUBGROUP_ORDER
        == TOTAL_CLASS_RESIDUE,
        "factor_catalog_sizes_are_14_14_16": [len(catalog) for catalog in catalogs]
        == [14, 14, 16],
        "each_factor_class_spans_dimension_three": all(
            r54.r44.section_rank([factor["line"] for factor in catalog]) == 3
            for catalog in catalogs
        ),
        "multiplication_matrix_rank_is_nine": multiplication[
            "multiplication_rank"
        ]
        == 9,
        "multiplication_kernel_dimension_is_eighteen": 27
        - multiplication["multiplication_rank"]
        == 18,
        "all_3136_cartesian_products_reconstructed": len(
            multiplication["products"]
        )
        == 3_136,
        "all_product_sections_distinct": len(multiplication["section_index"])
        == 3_136,
        "all_section_pairs_scanned": line_scan["section_pairs"]
        == 3_136 * 3_135 // 2,
        "every_catalog_trisecant_varies_exactly_one_factor": varying_factor_count_histogram
        == {1: line_scan["exact_catalog_trisecant_lines"]},
    }
    if not all(checks.values()):
        raise AssertionError(f"R56 fixed-class multiplication scan failed: {checks}")

    return {
        "schema": "p1553.fixed_class_multiplication_kernel_scan.r56.v1",
        "classification": [
            "toy",
            "exact-multiplication-matrix",
            "exhaustive-fixed-class-product-lines",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r54.PRIME,
        "subgroup_order": r54.r44.r38.SUBGROUP_ORDER,
        "class_residues": list(CLASS_RESIDUES),
        "total_class_residue": TOTAL_CLASS_RESIDUE,
        "factor_catalogs": [
            {
                "class_residue": residue,
                "factor_count": len(catalog),
                "abel_sum_scalar": catalog[0]["abel_sum_scalar"],
                "translation": catalog[0]["translation"],
                "basis_indices": list(multiplication["basis_indices"][index]),
            }
            for index, (residue, catalog) in enumerate(
                zip(CLASS_RESIDUES, catalogs)
            )
        ],
        "multiplication": {
            "domain_dimension": 27,
            "codomain_dimension": 9,
            "matrix_rank": multiplication["multiplication_rank"],
            "kernel_dimension": 27 - multiplication["multiplication_rank"],
            "product_sections_reconstructed": len(multiplication["products"]),
            "trivialization_ratio_support": multiplication[
                "trivialization_ratio_support"
            ],
        },
        "line_scan": {
            "section_pairs": line_scan["section_pairs"],
            "fingerprint_collision_ranges": line_scan[
                "fingerprint_collision_ranges"
            ],
            "exact_catalog_trisecant_lines": line_scan[
                "exact_catalog_trisecant_lines"
            ],
            "common_base_divisor_degree_histogram": {
                str(degree): base_degree_histogram[degree]
                for degree in sorted(base_degree_histogram)
            },
            "catalog_points_on_line_histogram": {
                str(size): line_size_histogram[size]
                for size in sorted(line_size_histogram)
            },
            "varying_factor_count_histogram": {
                str(count): varying_factor_count_histogram[count]
                for count in sorted(varying_factor_count_histogram)
            },
            "line_classification_sha256": line_classification_sha256,
            "representative_lines": representative_lines,
            "primitive_coprime_trisecant_lines": len(primitive_lines),
        },
        "checks": checks,
        "result": {
            "primitive_coprime_trisecant_found": bool(primitive_lines),
            "fixed_class_product_pair_space_exhausted": True,
            "other_class_triples_exhausted": False,
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
            "the exact multiplication and line scan cover only class residues 96,42,16 in one frozen toy",
            "other Picard-class triples and factors outside the within-window catalog remain open",
            "the scan classifies finite split catalog points, not all rational points of the projected Segre variety",
            "no asymptotic theorem, target locator, R10 output, rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
