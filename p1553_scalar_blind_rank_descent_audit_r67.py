#!/usr/bin/env python3
"""Audit exact relation rank and scalar-blind target descent on the R65 toy."""

import argparse
import hashlib
import itertools
import json
import math
import pathlib

import p1553_exact_public_base_trisecant_scan_r65 as r65


BASE_SIZE = 24
DEFAULT_TARGET = (137, 171)
HASH_DOMAIN = "P1553-R67"
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


def raw_scalar_mul(scalar, point):
    result = None
    addend = point
    while scalar:
        if scalar & 1:
            result = r65.r63.r62.r61.r60.r58.r44.r38.add(result, addend)
        addend = r65.r63.r62.r61.r60.r58.r44.r38.add(addend, addend)
        scalar >>= 1
    return result


def modular_square_root(value):
    prime = r65.r63.PRIME
    value %= prime
    if value == 0:
        return 0
    if pow(value, (prime - 1) // 2, prime) != 1:
        return None
    odd_part = prime - 1
    two_adic_order = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        two_adic_order += 1
    if two_adic_order == 1:
        root = pow(value, (prime + 1) // 4, prime)
        return min(root, (-root) % prime)
    nonresidue = 2
    while pow(nonresidue, (prime - 1) // 2, prime) != prime - 1:
        nonresidue += 1
    coefficient = pow(nonresidue, odd_part, prime)
    root = pow(value, (odd_part + 1) // 2, prime)
    residue = pow(value, odd_part, prime)
    order = two_adic_order
    while residue != 1:
        exponent = 1
        probe = residue * residue % prime
        while probe != 1:
            probe = probe * probe % prime
            exponent += 1
        update = pow(coefficient, 1 << (order - exponent - 1), prime)
        root = root * update % prime
        residue = residue * update * update % prime
        coefficient = update * update % prime
        order = exponent
    if root * root % prime != value:
        raise AssertionError("Tonelli-Shanks replay failed")
    return min(root, (-root) % prime)


def public_hash_to_subgroup(counter):
    curve = r65.r63.r62.r61.r60.r58.r44.r38
    digest = hashlib.sha256(f"{HASH_DOMAIN}|{counter}".encode()).digest()
    x_coordinate = int.from_bytes(digest, "big") % r65.r63.PRIME
    right_hand_side = (
        x_coordinate**3 + curve.CURVE_A * x_coordinate + curve.CURVE_B
    ) % r65.r63.PRIME
    y_coordinate = modular_square_root(right_hand_side)
    if y_coordinate is None:
        return None
    if digest[0] & 1:
        y_coordinate = (-y_coordinate) % r65.r63.PRIME
    point = raw_scalar_mul(2, (x_coordinate, y_coordinate))
    if raw_scalar_mul(r65.r63.SUBGROUP_ORDER, point) is not None:
        raise AssertionError("cofactor-cleared point is outside the prime subgroup")
    return point


def public_factor_base():
    generator = r65.r63.r62.r61.r60.r58.r44.r38.GENERATOR
    records = [
        {"point": None, "source": "anchor_identity"},
        {"point": generator, "source": "anchor_generator"},
    ]
    seen = {None, generator}
    counter = 0
    attempted_counters = 0
    quadratic_residue_counters = 0
    duplicate_or_identity_outputs = 0
    while len(records) < BASE_SIZE:
        point = public_hash_to_subgroup(counter)
        attempted_counters += 1
        counter += 1
        if point is None:
            continue
        quadratic_residue_counters += 1
        if point in seen:
            duplicate_or_identity_outputs += 1
            continue
        seen.add(point)
        records.append(
            {
                "point": point,
                "source": f"sha256_counter_{counter - 1}",
            }
        )
    for index, record in enumerate(records):
        record["atom_index"] = index
    return records, {
        "attempted_counters": attempted_counters,
        "quadratic_residue_counters": quadratic_residue_counters,
        "duplicate_or_identity_outputs": duplicate_or_identity_outputs,
        "accepted_hash_points": BASE_SIZE - 2,
    }


def scalar_blind_factor_catalog(base, class_sum, factor_basis):
    curve = r65.r63.r62.r61.r60.r58.r44.r38
    class_point = raw_scalar_mul(class_sum, curve.GENERATOR)
    translation_point = raw_scalar_mul(factor_basis["translation"], curve.GENERATOR)
    two_torsion = (192, 0)
    catalog = []
    group_sum_tests = 0
    for indices in itertools.combinations(range(len(base)), 3):
        points = tuple(base[index]["point"] for index in indices)
        group_sum_tests += 1
        if r65.r63.add_points(points) != class_point:
            continue
        for lift_pattern in ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)):
            lifted_points = tuple(
                curve.add(point, two_torsion) if parity else point
                for point, parity in zip(points, lift_pattern)
            )
            if r65.r63.add_points(lifted_points) != class_point:
                raise AssertionError("even torsion lift changed the divisor class")
            shifted = [
                curve.projective(curve.add(point, translation_point))
                for point in lifted_points
            ]
            line = curve.cross(shifted[0], shifted[1])
            if line == (0, 0, 0) or curve.dot(line, shifted[2]):
                raise AssertionError("scalar-blind divisor failed line replay")
            line = curve.normalized(line)
            catalog.append(
                {
                    "base_indices": indices,
                    "torsion_lift_pattern": lift_pattern,
                    "mask": sum(
                        1 << (index + parity * len(base))
                        for index, parity in zip(indices, lift_pattern)
                    ),
                    "line": line,
                    "coordinates": r65.r63.factor_coordinates(line, factor_basis),
                }
            )
    return catalog, group_sum_tests


def fixed_sum_triples_by_pair_lookup(base, class_sum):
    curve = r65.r63.r62.r61.r60.r58.r44.r38
    class_point = raw_scalar_mul(class_sum, curve.GENERATOR)
    point_to_index = {record["point"]: index for index, record in enumerate(base)}
    triples = set()
    pair_lookups = 0
    for first, second in itertools.combinations(range(len(base)), 2):
        pair_lookups += 1
        complement = curve.add(
            curve.add(class_point, curve.negate(base[first]["point"])),
            curve.negate(base[second]["point"]),
        )
        third = point_to_index.get(complement)
        if third is None or third in (first, second):
            continue
        triple = tuple(sorted((first, second, third)))
        if r65.r63.add_points(base[index]["point"] for index in triple) != class_point:
            raise AssertionError("pair-complement triple failed public group replay")
        triples.add(triple)
    return sorted(triples), pair_lookups


def target_triples_by_linear_lookup(core_base, target, class_sum):
    curve = r65.r63.r62.r61.r60.r58.r44.r38
    class_point = raw_scalar_mul(class_sum, curve.GENERATOR)
    desired_pair_sum = curve.add(class_point, curve.negate(target))
    point_to_index = {
        record["point"]: index for index, record in enumerate(core_base)
    }
    pairs = set()
    hash_lookups = 0
    for first, record in enumerate(core_base):
        hash_lookups += 1
        second_point = curve.add(desired_pair_sum, curve.negate(record["point"]))
        second = point_to_index.get(second_point)
        if second is None or second == first:
            continue
        pair = tuple(sorted((first, second)))
        if r65.r63.add_points(
            (core_base[pair[0]]["point"], core_base[pair[1]]["point"], target)
        ) != class_point:
            raise AssertionError("target pair-complement lookup failed group replay")
        pairs.add(pair)
    return sorted(pairs), hash_lookups


def scalar_blind_smooth_products(catalogs, multiplication_matrix):
    curve = r65.r63.r62.r61.r60.r58
    products_by_section = {}
    admissible_factor_pairs = 0
    for left in catalogs[0]:
        for right in catalogs[1]:
            if left["mask"] & right["mask"]:
                continue
            admissible_factor_pairs += 1
            section = r65.r63.r62.r61.product_output(
                multiplication_matrix,
                left["coordinates"],
                right["coordinates"],
            )
            section = curve.normalized(section)
            mask = left["mask"] | right["mask"]
            if section in products_by_section:
                if products_by_section[section]["mask"] != mask:
                    raise AssertionError("equal sections have different zero divisors")
                products_by_section[section]["factorization_multiplicity"] += 1
            else:
                products_by_section[section] = {
                    "section": section,
                    "mask": mask,
                    "factorization_multiplicity": 1,
                    "representative_factors": [
                        {
                            "base_indices": list(left["base_indices"]),
                            "torsion_lift_pattern": list(
                                left["torsion_lift_pattern"]
                            ),
                            "line": list(left["line"]),
                        },
                        {
                            "base_indices": list(right["base_indices"]),
                            "torsion_lift_pattern": list(
                                right["torsion_lift_pattern"]
                            ),
                            "line": list(right["line"]),
                        },
                    ],
                }
    return list(products_by_section.values()), admissible_factor_pairs


def collapsed_divisor_vector(mask, base_size):
    return [
        ((mask >> index) & 1) + ((mask >> (index + base_size)) & 1)
        for index in range(base_size)
    ]


def relation_rows(products, line_records, base_size):
    modulus = r65.r63.SUBGROUP_ORDER
    divisors = [
        collapsed_divisor_vector(product["mask"], base_size)
        for product in products
    ]
    rows = []
    rows_by_base_degree = {}
    for line in line_records:
        indices = line["product_indices"]
        reference = divisors[indices[0]]
        for product_index in indices[1:]:
            row = [
                (value - reference[column]) % modulus
                for column, value in enumerate(divisors[product_index])
            ]
            if sum(row) % modulus:
                raise AssertionError("relation row changed divisor degree")
            rows.append(row)
            rows_by_base_degree.setdefault(
                line["common_base_divisor_degree"], []
            ).append(row)
    return rows, rows_by_base_degree


def row_reduce(rows, column_count):
    modulus = r65.r63.SUBGROUP_ORDER
    matrix = [
        [value % modulus for value in row]
        for row in rows
        if any(value % modulus for value in row)
    ]
    pivot_columns = []
    pivot_row = 0
    for column in range(column_count):
        source = next(
            (
                row
                for row in range(pivot_row, len(matrix))
                if matrix[row][column]
            ),
            None,
        )
        if source is None:
            continue
        matrix[pivot_row], matrix[source] = matrix[source], matrix[pivot_row]
        inverse = pow(matrix[pivot_row][column], -1, modulus)
        matrix[pivot_row] = [
            value * inverse % modulus for value in matrix[pivot_row]
        ]
        for row in range(len(matrix)):
            if row == pivot_row or not matrix[row][column]:
                continue
            multiplier = matrix[row][column]
            matrix[row] = [
                (value - multiplier * pivot) % modulus
                for value, pivot in zip(matrix[row], matrix[pivot_row])
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return matrix[:pivot_row], pivot_columns


def solve_unique_system(rows, right_hand_sides, column_count):
    modulus = r65.r63.SUBGROUP_ORDER
    matrix = [
        [value % modulus for value in row] + [right_hand_side % modulus]
        for row, right_hand_side in zip(rows, right_hand_sides)
    ]
    pivot_columns = []
    pivot_row = 0
    for column in range(column_count):
        source = next(
            (
                row
                for row in range(pivot_row, len(matrix))
                if matrix[row][column]
            ),
            None,
        )
        if source is None:
            continue
        matrix[pivot_row], matrix[source] = matrix[source], matrix[pivot_row]
        inverse = pow(matrix[pivot_row][column], -1, modulus)
        matrix[pivot_row] = [
            value * inverse % modulus for value in matrix[pivot_row]
        ]
        for row in range(len(matrix)):
            if row == pivot_row or not matrix[row][column]:
                continue
            multiplier = matrix[row][column]
            matrix[row] = [
                (value - multiplier * pivot) % modulus
                for value, pivot in zip(matrix[row], matrix[pivot_row])
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    if any(
        not any(row[:column_count]) and row[column_count]
        for row in matrix
    ):
        raise AssertionError("fixed-sum relation system is inconsistent")
    if len(pivot_columns) != column_count:
        return len(pivot_columns), None
    solution = [0] * column_count
    for row, pivot_column in enumerate(pivot_columns):
        solution[pivot_column] = matrix[row][column_count]
    return len(pivot_columns), solution


def direct_product_relation_rows(products, base_size):
    modulus = r65.r63.SUBGROUP_ORDER
    divisors = [
        collapsed_divisor_vector(product["mask"], base_size)
        for product in products
    ]
    reference = divisors[0]
    return [
        [
            (value - reference[column]) % modulus
            for column, value in enumerate(divisor)
        ]
        for divisor in divisors[1:]
    ]


def nullspace_basis(reduced_rows, pivot_columns, column_count):
    modulus = r65.r63.SUBGROUP_ORDER
    free_columns = [
        column for column in range(column_count) if column not in pivot_columns
    ]
    basis = []
    for free_column in free_columns:
        vector = [0] * column_count
        vector[free_column] = 1
        for row, pivot_column in enumerate(pivot_columns):
            vector[pivot_column] = -reduced_rows[row][free_column] % modulus
        basis.append(vector)
    return basis


def oriented_logs_from_anchors(nullspace):
    if len(nullspace) != 2:
        return None
    modulus = r65.r63.SUBGROUP_ORDER
    first, second = nullspace
    determinant = (first[0] * second[1] - first[1] * second[0]) % modulus
    if determinant == 0:
        return None
    determinant_inverse = pow(determinant, -1, modulus)
    first_coefficient = -second[0] * determinant_inverse % modulus
    second_coefficient = first[0] * determinant_inverse % modulus
    logs = [
        (first_coefficient * left + second_coefficient * right) % modulus
        for left, right in zip(first, second)
    ]
    if logs[:2] != [0, 1]:
        raise AssertionError("identity/generator anchors failed orientation")
    return logs


def relation_rank_summary(rows, rows_by_base_degree, column_count):
    reduced, pivots = row_reduce(rows, column_count)
    rank_by_degree = {}
    cumulative = []
    cumulative_rank_by_degree = {}
    for degree in sorted(rows_by_base_degree):
        degree_rows = rows_by_base_degree[degree]
        rank_by_degree[str(degree)] = len(
            row_reduce(degree_rows, column_count)[1]
        )
        cumulative.extend(degree_rows)
        cumulative_rank_by_degree[str(degree)] = len(
            row_reduce(cumulative, column_count)[1]
        )
    nullspace = nullspace_basis(reduced, pivots, column_count)
    return {
        "relation_rows": len(rows),
        "rank": len(pivots),
        "nullity": column_count - len(pivots),
        "rows_by_common_lifted_base_degree": {
            str(degree): len(rows_by_base_degree[degree])
            for degree in sorted(rows_by_base_degree)
        },
        "rank_by_common_lifted_base_degree": rank_by_degree,
        "cumulative_rank_through_common_lifted_base_degree": (
            cumulative_rank_by_degree
        ),
        "nullspace_basis": nullspace,
    }, nullspace


def scan_base(base, multiplication_matrix, factor_bases):
    catalogs_and_costs = [
        scalar_blind_factor_catalog(base, class_sum, basis)
        for class_sum, basis in zip(r65.r63.CLASS_SUM_SCALARS, factor_bases)
    ]
    catalogs = [item[0] for item in catalogs_and_costs]
    catalog_triples = [
        sorted({tuple(record["base_indices"]) for record in catalog})
        for catalog in catalogs
    ]
    lookup_results = [
        fixed_sum_triples_by_pair_lookup(base, class_sum)
        for class_sum in r65.r63.CLASS_SUM_SCALARS
    ]
    lookup_triples = [result[0] for result in lookup_results]
    if catalog_triples != lookup_triples:
        raise AssertionError("factor catalogs disagree with pair-complement lookup")
    fixed_sum_rows = []
    fixed_sum_right_hand_sides = []
    for class_sum, triples in zip(r65.r63.CLASS_SUM_SCALARS, lookup_triples):
        for triple in triples:
            row = [0] * len(base)
            for index in triple:
                row[index] = 1
            fixed_sum_rows.append(row)
            fixed_sum_right_hand_sides.append(class_sum)
    fixed_sum_rank, fixed_sum_solution = solve_unique_system(
        fixed_sum_rows,
        fixed_sum_right_hand_sides,
        len(base),
    )
    products, admissible_factor_pairs = scalar_blind_smooth_products(
        catalogs,
        multiplication_matrix,
    )
    direct_rows = direct_product_relation_rows(products, len(base))
    direct_reduced, direct_pivots = row_reduce(direct_rows, len(base))
    direct_nullspace = nullspace_basis(
        direct_reduced,
        direct_pivots,
        len(base),
    )
    scan = r65.exhaustive_line_scan(products)
    rows, rows_by_base_degree = relation_rows(
        products,
        scan["line_records"],
        len(base),
    )
    rank_summary, nullspace = relation_rank_summary(
        rows,
        rows_by_base_degree,
        len(base),
    )
    return {
        "projected_class_triple_counts": [len(catalog) // 4 for catalog in catalogs],
        "fixed_sum_pair_complement_lookup": {
            "triple_counts": [len(triples) for triples in lookup_triples],
            "pair_hash_lookups": [result[1] for result in lookup_results],
            "relation_rows": len(fixed_sum_rows),
            "coefficient_matrix_rank": fixed_sum_rank,
            "unique_log_solution_available": fixed_sum_solution is not None,
        },
        "lifted_factor_section_counts": [len(catalog) for catalog in catalogs],
        "public_group_sum_tests": [item[1] for item in catalogs_and_costs],
        "admissible_factor_pairs": admissible_factor_pairs,
        "distinct_smooth_product_sections": len(products),
        "direct_product_quotient_relations": {
            "relation_rows": len(direct_rows),
            "rank": len(direct_pivots),
            "nullity": len(base) - len(direct_pivots),
            "incidence_scan_required": False,
        },
        "exact_product_section_pairs": scan["section_pairs"],
        "exact_trisecant_lines": scan["exact_smooth_trisecant_lines"],
        "primitive_trisecant_lines": scan[
            "primitive_pairwise_disjoint_trisecant_lines"
        ],
        "line_size_histogram": scan["line_size_histogram"],
        "common_lifted_base_degree_histogram": scan["base_degree_histogram"],
        "rank": rank_summary,
    }, rows, nullspace, fixed_sum_solution, direct_rows, direct_nullspace, lookup_triples


def encode_point(point):
    return "O" if point is None else f"{point[0]},{point[1]}"


def run(target=DEFAULT_TARGET):
    if target is not None:
        target = tuple(target)
    curve = r65.r63.r62.r61.r60.r58.r44.r38
    if target in (None, curve.GENERATOR):
        raise ValueError("target must be distinct from the two anchors")
    if raw_scalar_mul(r65.r63.SUBGROUP_ORDER, target) is not None:
        raise ValueError("target is outside the prime-order subgroup")

    r61_path = pathlib.Path(
        "p1553_degree_six_product_hypersurface_locator_report_r61.json"
    )
    r61_bytes = r61_path.read_bytes()
    r61_report = json.loads(r61_bytes)
    dependency_source_sha256 = {
        path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
        for path in PINNED_DEPENDENCY_SHA256
    }
    factor_bases = r65.r63.r62.factor_basis_lines(r61_report)
    multiplication_matrix = r61_report["multiplication_map"]["multiplication_matrix"]
    core_base, hash_cost = public_factor_base()
    if target in {record["point"] for record in core_base}:
        raise ValueError("target is already in the target-independent core base")
    expanded_base = core_base + [
        {
            "point": target,
            "source": "caller_supplied_target",
            "atom_index": len(core_base),
        }
    ]

    (
        core_scan,
        core_incidence_rows,
        core_incidence_nullspace,
        core_fixed_sum_logs,
        core_direct_rows,
        core_direct_nullspace,
        core_triples,
    ) = scan_base(
        core_base,
        multiplication_matrix,
        factor_bases,
    )
    (
        target_scan,
        target_incidence_rows,
        target_incidence_nullspace,
        target_fixed_sum_logs,
        target_direct_rows,
        target_direct_nullspace,
        expanded_triples,
    ) = scan_base(
        expanded_base,
        multiplication_matrix,
        factor_bases,
    )
    core_logs = core_fixed_sum_logs
    expanded_logs = target_fixed_sum_logs
    target_column = len(expanded_base) - 1
    target_involving_incidence_rows = [
        row
        for row in target_incidence_rows
        if row[target_column] % r65.r63.SUBGROUP_ORDER
    ]
    target_involving_direct_rows = [
        row
        for row in target_direct_rows
        if row[target_column] % r65.r63.SUBGROUP_ORDER
    ]
    target_lookup_results = [
        target_triples_by_linear_lookup(core_base, target, class_sum)
        for class_sum in r65.r63.CLASS_SUM_SCALARS
    ]
    target_lookup_triples = [
        [tuple(pair) + (target_column,) for pair in result[0]]
        for result in target_lookup_results
    ]
    expanded_target_triples = [
        [triple for triple in triples if target_column in triple]
        for triples in expanded_triples
    ]
    target_log_candidates = []
    target_relation_rows = []
    for class_sum, triples in zip(
        r65.r63.CLASS_SUM_SCALARS,
        target_lookup_triples,
    ):
        for triple in triples:
            row = [0] * len(expanded_base)
            for index in triple:
                row[index] = 1
            target_relation_rows.append(row)
            target_log_candidates.append(
                (
                    class_sum
                    - sum(core_logs[index] for index in triple if index != target_column)
                )
                % r65.r63.SUBGROUP_ORDER
            )
    core_relation_rows = []
    for triples in core_triples:
        for triple in triples:
            row = [0] * len(expanded_base)
            for index in triple:
                row[index] = 1
            core_relation_rows.append(row)
    core_plus_target_rank = len(
        row_reduce(
            core_relation_rows + target_relation_rows,
            len(expanded_base),
        )[1]
    )
    target_log = expanded_logs[target_column] if expanded_logs is not None else None
    direct_core_logs = oriented_logs_from_anchors(core_direct_nullspace)
    direct_expanded_logs = oriented_logs_from_anchors(target_direct_nullspace)
    incidence_core_logs = oriented_logs_from_anchors(core_incidence_nullspace)
    incidence_expanded_logs = oriented_logs_from_anchors(
        target_incidence_nullspace
    )

    core_log_verification = (
        core_logs is not None
        and all(
            raw_scalar_mul(logarithm, curve.GENERATOR) == record["point"]
            for logarithm, record in zip(core_logs, core_base)
        )
    )
    target_log_verification = (
        target_log is not None
        and raw_scalar_mul(target_log, curve.GENERATOR) == target
    )
    core_logs_stable_after_target = (
        core_logs is not None
        and expanded_logs is not None
        and core_logs == expanded_logs[: len(core_base)]
    )

    checks = {
        "direct_r61_report_digest_is_pinned": hashlib.sha256(r61_bytes).hexdigest()
        == "f91213021bd350b77cf1ac7d5cb8f2f9ae63b1265a5081d4ed58a3da07aa1f64",
        "all_loaded_dependency_source_digests_are_pinned": (
            dependency_source_sha256 == PINNED_DEPENDENCY_SHA256
        ),
        "factor_base_generation_consumes_no_scalar_labels": True,
        "target_is_absent_from_target_independent_core_base": True,
        "pair_lookup_matches_factor_catalog_triples": True,
        "linear_target_lookup_matches_expanded_catalog": target_lookup_triples
        == expanded_target_triples,
        "core_fixed_sum_matrix_has_full_column_rank": core_scan[
            "fixed_sum_pair_complement_lookup"
        ]["coefficient_matrix_rank"]
        == len(core_base),
        "expanded_fixed_sum_matrix_has_full_column_rank": target_scan[
            "fixed_sum_pair_complement_lookup"
        ]["coefficient_matrix_rank"]
        == len(expanded_base),
        "core_rank_reaches_dimension_minus_two": core_scan["rank"]["rank"]
        == len(core_base) - 2,
        "expanded_rank_reaches_dimension_minus_two": target_scan["rank"]["rank"]
        == len(expanded_base) - 2,
        "direct_product_quotient_rank_matches_incidence_rank": core_scan[
            "direct_product_quotient_relations"
        ]["rank"]
        == core_scan["rank"]["rank"]
        and target_scan["direct_product_quotient_relations"]["rank"]
        == target_scan["rank"]["rank"],
        "core_incidence_nullity_is_exactly_two": len(core_incidence_nullspace)
        == 2,
        "expanded_incidence_nullity_is_exactly_two": len(
            target_incidence_nullspace
        )
        == 2,
        "all_three_relation_layers_recover_identical_logs": (
            core_logs == direct_core_logs == incidence_core_logs
            and expanded_logs == direct_expanded_logs == incidence_expanded_logs
        ),
        "all_linear_target_relations_recover_one_log": set(target_log_candidates)
        == {target_log},
        "core_logs_verify_by_public_scalar_multiplication": core_log_verification,
        "target_log_verifies_by_public_scalar_multiplication": target_log_verification,
        "core_logs_are_stable_after_target_extension": core_logs_stable_after_target,
        "core_plus_linear_target_rows_reach_expanded_rank": core_plus_target_rank
        == len(expanded_base),
    }
    if not all(checks.values()):
        raise AssertionError(f"R67 scalar-blind rank/descent audit failed: {checks}")

    return {
        "schema": "p1553.scalar_blind_rank_descent_audit.r67.v1",
        "classification": [
            "toy",
            "exact-relation-rank",
            "scalar-blind-target-action",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r65.r63.PRIME,
        "prime_subgroup_order": r65.r63.SUBGROUP_ORDER,
        "target_input": {
            "point": list(target),
            "encoding": encode_point(target),
            "default_fixture": target == DEFAULT_TARGET,
            "consumed_scalar_label": False,
        },
        "factor_base": {
            "definition": "identity and generator anchors followed by SHA256 counter-to-x, Tonelli-Shanks, and cofactor-two clearing",
            "target_independent": True,
            "points": [
                {
                    "index": record["atom_index"],
                    "encoding": encode_point(record["point"]),
                    "source": record["source"],
                }
                for record in core_base
            ],
            "hash_to_curve_cost": hash_cost,
            "scalar_labels_materialized": False,
        },
        "core_precomputation": core_scan,
        "target_extension": {
            **target_scan,
            "linear_target_pair_lookup": {
                "pair_counts": [len(result[0]) for result in target_lookup_results],
                "core_point_hash_lookups": [
                    result[1] for result in target_lookup_results
                ],
                "target_log_candidates": target_log_candidates,
                "core_plus_target_relation_rank": core_plus_target_rank,
            },
            "target_involving_direct_quotient_rows": len(
                target_involving_direct_rows
            ),
            "target_involving_direct_rows_rank": len(
                row_reduce(target_involving_direct_rows, len(expanded_base))[1]
            ),
            "target_involving_incidence_rows": len(
                target_involving_incidence_rows
            ),
            "target_involving_incidence_rows_rank": len(
                row_reduce(target_involving_incidence_rows, len(expanded_base))[1]
            ),
        },
        "linear_algebra": {
            "modulus": r65.r63.SUBGROUP_ORDER,
            "fixed_sum_right_hand_sides": list(r65.r63.CLASS_SUM_SCALARS),
            "core_logs_recovered_from_fixed_sum_system": core_logs,
            "target_log_recovered_from_fixed_sum_system": target_log,
            "direct_quotient_and_incidence_nullspaces_used_as_controls": True,
            "dense_toy_elimination_complexity": "O(R*B^2) field operations for R relation rows and B columns",
            "generic_sparse_linear_algebra_model": "about B^2 field operations for a full-rank sparse B-column system absent additional matrix structure",
        },
        "cost_boundary": {
            "factor_base_setup": "O(B polylog(p)) expected public field/group operations; no Theta(N) table and no scalar labels",
            "fixed_sum_catalog_enumeration": "Theta(B^2) pair-complement hash lookups; the toy uses exactly 2*C(B,2)",
            "uniform_fixed_sum_relation_count": "about K*B^3/N for K public class sums; B independent rows require B^2 on the order of N when K is constant",
            "factor_base_scale_for_rank": "B at least on the order of sqrt(N) under the uniform model",
            "relation_precomputation_at_required_scale": "Theta(B^2)=Theta(N), already above rho",
            "online_target_lookup": "Theta(K*B) group/hash operations by solving P_i+P_j=C-Q with one complement lookup per P_i",
            "uniform_target_pair_count": "about K*B^2/N; constant success requires B on the order of sqrt(N)",
            "online_target_cost_at_required_scale": "Theta(sqrt(N)), rho matching rather than improving",
            "smooth_product_materialization": "not needed for logs; retained only as an FFE control",
            "exact_incidence_scan": "not needed for logs; audited at O(M^2 log M) time and Theta(M^2) memory",
            "target_descent_amortized": True,
            "target_extension_repeats_relation_collection": False,
            "exact_toy_target_hash_lookups": sum(
                result[1] for result in target_lookup_results
            ),
            "relation_generation_sub_rho": False,
            "linear_algebra_sub_rho": False,
        },
        "dependency_source_sha256": dependency_source_sha256,
        "checks": checks,
        "result": {
            "exact_core_relation_rank_available": True,
            "factor_base_logs_recovered_on_toy": True,
            "fresh_public_target_action": True,
            "scalar_blind_target_log_recovered_on_toy": True,
            "fixed_sum_equations_solve_before_ffe": True,
            "ffe_layers_add_independent_log_information": False,
            "primitive_only_relations_full_rank": core_scan["rank"][
                "rank_by_common_lifted_base_degree"
            ].get("0")
            == len(core_base) - 2,
            "quadratic_all_pairs_relation_scan_required": False,
            "rho_matching_online_descent_under_uniform_model": True,
            "asymptotic_relation_algorithm": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "limits": [
            "the complete solve is one prime-order subgroup of size 103 on one finite toy curve",
            "fixed-sum triple equations solve the toy before any product, sextic, or trisecant computation",
            "under the uniform model, enough fixed-sum rows require B about sqrt(N), Theta(N) precomputation, and Theta(sqrt(N)) online target lookup",
            "the generic sparse-linear-algebra cost at B about sqrt(N) is about N field operations",
            "the exact FFE incidence scans are retained as redundant controls and are not part of the claimed target action",
            "no generic-prime distribution or rank theorem is proved",
            "no Shoup-bound improvement or ECDLP breakthrough is claimed",
        ],
        "pass": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-x", type=int, default=DEFAULT_TARGET[0])
    parser.add_argument("--target-y", type=int, default=DEFAULT_TARGET[1])
    parser.add_argument("--output")
    arguments = parser.parse_args()
    payload = json.dumps(
        run((arguments.target_x, arguments.target_y)),
        indent=2,
        sort_keys=True,
    ) + "\n"
    if arguments.output:
        pathlib.Path(arguments.output).write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
