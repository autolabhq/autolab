#!/usr/bin/env python3
"""Cost exact non-character resultant grammars for the canonical 5A+5C norm."""

from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any, Iterable, Iterator, Sequence


SCHEMA = "p1553.5a5c_noncharacter_algebraic_target_norm_resultant.r107.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
FULL_A_COUNT = 5
FULL_C_COUNT = 5
A_ATOM_EXPONENT = Fraction(2, 5)
C_ATOM_EXPONENT = Fraction(3, 5)
FULL_SOURCE_EXPONENT = Fraction(5)
BALANCED_LEFT = (2, 3)
BALANCED_RIGHT = (3, 2)
MARKER_DIMENSION = 10

R106_PRODUCER = pathlib.Path(
    "p1553_5a5c_scalar_target_norm_count_circuit_probe_r106.py"
)
R106_PRODUCER_SHA256 = (
    "fdff39f3b49b0d01690ae5a480c17bfd701adcc528bc2701aa6ae9756bda6652"
)
R106_REPORT = pathlib.Path(
    "p1553_5a5c_scalar_target_norm_count_circuit_probe_report_r106.json"
)
R106_REPORT_SHA256 = (
    "08a84327617ec660b189e5a42f22eaa79f817e1a6aa3167724263595f1fa192d"
)
R106_GATE = pathlib.Path(
    "p1553_5a5c_scalar_target_norm_count_circuit_probe_gate_r106.md"
)
R106_GATE_SHA256 = (
    "dcc7fe476ed09c0ad559f115a38bdefc31d5209ec72fd5bca817c58dcba6e95c"
)
R106_PARENT = pathlib.Path(
    "p1553_5a5c_scalar_target_norm_count_circuit_probe_parent_report_r106.yaml"
)
R106_PARENT_SHA256 = (
    "ac4c9c0062951d690a1917882662f0b0d641cc19a5aadf4d73f5e4d16e05253f"
)
R105_PRODUCER = pathlib.Path(
    "p1553_5a5c_actual_deck_nonmergeable_target_pullback_probe_r105.py"
)
R105_PRODUCER_SHA256 = (
    "d7215def9fa82b01ec72ea43f3d297fd1343d2ceae9f5d364a351326c9f9c6b8"
)
R105_REPORT = pathlib.Path(
    "p1553_5a5c_actual_deck_nonmergeable_target_pullback_probe_report_r105.json"
)
R105_REPORT_SHA256 = (
    "c00b8c3e934e783012acb1dab83282f98720702bb8cd81a706fbdf4671f93a83"
)
R105_GATE = pathlib.Path(
    "p1553_5a5c_actual_deck_nonmergeable_target_pullback_probe_gate_r105.md"
)
R105_GATE_SHA256 = (
    "e349dc695975b3d5ce923ed334c61c6329884bbc30228f7bf52fb1c8413ea3ef"
)
R105_PARENT = pathlib.Path(
    "p1553_5a5c_actual_deck_nonmergeable_target_pullback_probe_parent_report_r105.yaml"
)
R105_PARENT_SHA256 = (
    "34ed83a9fbb47048734f9a14f0311fc272761b747da9b12fa66e110ab8daa3ad"
)
R102_REPORT = pathlib.Path(
    "p1553_5a5c_two_sided_implicit_join_probe_report_r102.json"
)
R102_REPORT_SHA256 = (
    "5f16489b5f0535df6c9728edfed5f2f83641a4b9c30fd2609db52a73f46d40e7"
)
R102_GATE = pathlib.Path(
    "p1553_5a5c_two_sided_implicit_join_probe_gate_r102.md"
)
R102_GATE_SHA256 = (
    "1af82597762525110baf31cec7b4e7b96693a730270f7e1bd6556906e5c5f56c"
)
R84_REPORT = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_probe_report_r84.json"
)
R84_REPORT_SHA256 = (
    "c9b1c5fb0f58f2c5118562623fd5dfff5d55d7238b513892d4178a67af5ccf0b"
)
R84_GATE = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_probe_gate_r84.md"
)
R84_GATE_SHA256 = (
    "4e23024a1a5971a52d6664be678fd095f506814297e61b0f8992076e643e3661"
)

Point = tuple[int, int] | None
Source = tuple[tuple[int, ...], tuple[int, ...]]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R106_PRODUCER: R106_PRODUCER_SHA256,
        R106_REPORT: R106_REPORT_SHA256,
        R106_GATE: R106_GATE_SHA256,
        R106_PARENT: R106_PARENT_SHA256,
        R105_PRODUCER: R105_PRODUCER_SHA256,
        R105_REPORT: R105_REPORT_SHA256,
        R105_GATE: R105_GATE_SHA256,
        R105_PARENT: R105_PARENT_SHA256,
        R102_REPORT: R102_REPORT_SHA256,
        R102_GATE: R102_GATE_SHA256,
        R84_REPORT: R84_REPORT_SHA256,
        R84_GATE: R84_GATE_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R107 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R106 = load_module("p1553_r106_for_r107", R106_PRODUCER)
R105 = R106.R105
R104 = R105.R104
R102 = R105.R102
R84 = R105.R84
R82 = R105.R82
R70 = R105.R70


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


def source_json(source: Source) -> list[list[int]]:
    return [list(source[0]), list(source[1])]


def source_exponent(count_a: int, count_c: int) -> Fraction:
    return count_a * A_ATOM_EXPONENT + count_c * C_ATOM_EXPONENT


def root_partition_ledger() -> dict[str, Any]:
    rows = []
    for left_a in range(FULL_A_COUNT + 1):
        for left_c in range(FULL_C_COUNT + 1):
            if (left_a, left_c) in (
                (0, 0),
                (FULL_A_COUNT, FULL_C_COUNT),
            ):
                continue
            right_a = FULL_A_COUNT - left_a
            right_c = FULL_C_COUNT - left_c
            left_exponent = source_exponent(left_a, left_c)
            right_exponent = source_exponent(right_a, right_c)
            interface_exponent = max(left_exponent, right_exponent)
            rows.append(
                {
                    "left_counts": {"a": left_a, "c": left_c},
                    "right_counts": {"a": right_a, "c": right_c},
                    "left_degree_exponent_B": fraction_record(left_exponent),
                    "right_degree_exponent_B": fraction_record(
                        right_exponent
                    ),
                    "sylvester_or_subresultant_interface_exponent_B": (
                        fraction_record(interface_exponent)
                    ),
                    "inside_setup_cap": interface_exponent <= SETUP_CAP,
                    "inside_online_cap": interface_exponent <= ONLINE_CAP,
                }
            )
    minimum = min(
        Fraction(
            row[
                "sylvester_or_subresultant_interface_exponent_B"
            ]["exact"]
        )
        for row in rows
    )
    minimizers = [
        row
        for row in rows
        if Fraction(
            row[
                "sylvester_or_subresultant_interface_exponent_B"
            ]["exact"]
        )
        == minimum
    ]
    return {
        "partition_count": len(rows),
        "partitions": rows,
        "minimum_interface_exponent_B": fraction_record(minimum),
        "minimum_interface_above_setup_cap": minimum > SETUP_CAP,
        "minimum_interface_above_online_cap": minimum > ONLINE_CAP,
        "minimizer_count": len(minimizers),
        "minimizers": minimizers,
        "balanced_mixed_split_present": any(
            row["left_counts"] == {"a": 2, "c": 3}
            and row["right_counts"] == {"a": 3, "c": 2}
            for row in minimizers
        ),
    }


def partial_sources(
    size_a: int,
    size_c: int,
    count_a: int,
    count_c: int,
) -> Iterator[Source]:
    for indices_a in itertools.combinations_with_replacement(
        range(size_a),
        count_a,
    ):
        for indices_c in itertools.combinations_with_replacement(
            range(size_c),
            count_c,
        ):
            yield indices_a, indices_c


def partial_endpoint_histogram(
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    count_a: int,
    count_c: int,
    curve: dict[str, Any],
) -> tuple[collections.Counter[Point], int]:
    histogram: collections.Counter[Point] = collections.Counter()
    occurrence_count = 0
    for source in partial_sources(
        len(atoms_a),
        len(atoms_c),
        count_a,
        count_c,
    ):
        endpoint = R102.source_endpoint(
            source,
            atoms_a,
            atoms_c,
            curve,
        )
        histogram[endpoint] += 1
        occurrence_count += 1
    expected = (
        math.comb(len(atoms_a) + count_a - 1, count_a)
        * math.comb(len(atoms_c) + count_c - 1, count_c)
    )
    if occurrence_count != expected or sum(histogram.values()) != expected:
        raise AssertionError("partial source count drifted")
    return histogram, occurrence_count


def bounded_submultiset_count(
    indices: Sequence[int],
    selected_size: int,
) -> int:
    multiplicities = collections.Counter(indices)
    coefficients = [1] + [0] * selected_size
    for multiplicity in multiplicities.values():
        updated = [0] * (selected_size + 1)
        for current_size, count in enumerate(coefficients):
            for take in range(
                min(multiplicity, selected_size - current_size) + 1
            ):
                updated[current_size + take] += count
        coefficients = updated
    return coefficients[selected_size]


def balanced_partition_weight(source: Source) -> int:
    return (
        bounded_submultiset_count(source[0], BALANCED_LEFT[0])
        * bounded_submultiset_count(source[1], BALANCED_LEFT[1])
    )


def split_convolution_count(
    target: Point,
    left: collections.Counter[Point],
    right: collections.Counter[Point],
    curve: dict[str, Any],
) -> int:
    count = 0
    for endpoint, multiplicity in left.items():
        complement = R70.add(
            target,
            R70.negate(endpoint, curve),
            curve,
        )
        count += multiplicity * right.get(complement, 0)
    return count


def query_record(
    label: str,
    target: Point,
    source_map: dict[Point, list[Source]],
    left_histogram: collections.Counter[Point],
    right_histogram: collections.Counter[Point],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> dict[str, Any]:
    sources = sorted(source_map.get(target, ()))
    weights = [balanced_partition_weight(source) for source in sources]
    weighted_sum = sum(weights)
    split_count = split_convolution_count(
        target,
        left_histogram,
        right_histogram,
        curve,
    )
    markers = [
        R105.marker_vector(source, curve["field_prime"])
        for source in sources
    ]
    recovered = [
        R105.source_from_marker(
            marker,
            len(atoms_a),
            len(atoms_c),
            curve["field_prime"],
        )
        for marker in markers
    ]
    return {
        "label": label,
        "target": point_json(target),
        "canonical_integer_source_count": len(sources),
        "balanced_resultant_vanishing_order": split_count,
        "sum_of_source_partition_weights": weighted_sum,
        "partition_weight_identity_exact": split_count == weighted_sum,
        "partition_weights": weights,
        "uniform_partition_weight_within_target_fiber": (
            len(set(weights)) <= 1
        ),
        "canonical_sources": [source_json(source) for source in sources],
        "marker_factor_multiplicities": [
            {
                "marker": list(marker),
                "partition_multiplicity": weight,
            }
            for marker, weight in zip(markers, weights)
        ],
        "markers_recover_canonical_sources": recovered == sources,
        "all_sources_replay_target": all(
            R102.source_endpoint(source, atoms_a, atoms_c, curve) == target
            for source in recovered
        ),
        "returned_bottom": not sources,
        "candidate_scalar_labels_consumed": False,
        "verifier_enumerated_sources": True,
        "candidate_work_credit": False,
    }


def actual_instance(
    curve: dict[str, Any],
    offset: int,
    control_class: str,
) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    canonical_rows = R105.source_factor_rows(atoms_a, atoms_c, curve)
    source_map: dict[Point, list[Source]] = collections.defaultdict(list)
    for source, endpoint, _key, _marker in canonical_rows:
        source_map[endpoint].append(source)
    left_histogram, left_count = partial_endpoint_histogram(
        atoms_a,
        atoms_c,
        *BALANCED_LEFT,
        curve,
    )
    right_histogram, right_count = partial_endpoint_histogram(
        atoms_a,
        atoms_c,
        *BALANCED_RIGHT,
        curve,
    )
    unique_rows = [
        (
            balanced_partition_weight(sources[0]),
            target,
        )
        for target, sources in source_map.items()
        if len(sources) == 1
    ]
    if not unique_rows:
        raise AssertionError("actual deck has no unique target")
    unique_min_target = min(
        unique_rows,
        key=lambda row: (row[0], R102.point_sort_key(row[1])),
    )[1]
    unique_max_target = max(
        unique_rows,
        key=lambda row: (row[0], R102.point_sort_key(row[1])),
    )[1]
    maximum_target = max(
        source_map,
        key=lambda target: (
            len(source_map[target]),
            tuple(-value for value in R102.point_sort_key(target)),
        ),
    )
    blind = R102.blind_target(
        source_map,
        curve,
        f"R107|{control_class}|{offset}|blind",
    )
    queries = [
        query_record(
            "unique_min_partition_weight",
            unique_min_target,
            source_map,
            left_histogram,
            right_histogram,
            atoms_a,
            atoms_c,
            curve,
        ),
        query_record(
            "unique_max_partition_weight",
            unique_max_target,
            source_map,
            left_histogram,
            right_histogram,
            atoms_a,
            atoms_c,
            curve,
        ),
        query_record(
            "maximum_multiplicity",
            maximum_target,
            source_map,
            left_histogram,
            right_histogram,
            atoms_a,
            atoms_c,
            curve,
        ),
        query_record(
            "blind",
            blind,
            source_map,
            left_histogram,
            right_histogram,
            atoms_a,
            atoms_c,
            curve,
        ),
        query_record(
            "projective_identity",
            None,
            source_map,
            left_histogram,
            right_histogram,
            atoms_a,
            atoms_c,
            curve,
        ),
    ]
    all_weights = {
        balanced_partition_weight(source)
        for source, _endpoint, _key, _marker in canonical_rows
    }
    return {
        "control_class": control_class,
        "family_id": curve["family_id"],
        "offset": offset,
        "field_prime": curve["field_prime"],
        "subgroup_order": curve["subgroup_order"],
        "factor_base_size_B": len(factors),
        "factor_base_injective": geometry["factor_base_injective"],
        "atom_a_size": len(atoms_a),
        "atom_c_size": len(atoms_c),
        "canonical_source_count": len(canonical_rows),
        "canonical_target_support_size": len(source_map),
        "maximum_canonical_target_multiplicity": max(
            len(sources) for sources in source_map.values()
        ),
        "balanced_left_source_count": left_count,
        "balanced_right_source_count": right_count,
        "finite_sylvester_dimension": left_count + right_count,
        "balanced_left_endpoint_support": len(left_histogram),
        "balanced_right_endpoint_support": len(right_histogram),
        "global_partition_weight_minimum": min(all_weights),
        "global_partition_weight_maximum": max(all_weights),
        "global_partition_weight_value_count": len(all_weights),
        "global_partition_weights_nonconstant": len(all_weights) > 1,
        "query_controls": queries,
        "all_partition_weight_identities_exact": all(
            query["partition_weight_identity_exact"] for query in queries
        ),
        "all_markers_recover_sources": all(
            query["markers_recover_canonical_sources"] for query in queries
        ),
        "all_recovered_sources_replay": all(
            query["all_sources_replay_target"] for query in queries
        ),
        "maximum_fiber_has_nonuniform_partition_weights": False,
        "candidate_scalar_labels_consumed": False,
    }


def normalize_instance_flags(instance: dict[str, Any]) -> dict[str, Any]:
    maximum_query = next(
        query
        for query in instance["query_controls"]
        if query["label"] == "maximum_multiplicity"
    )
    instance["maximum_fiber_has_nonuniform_partition_weights"] = (
        maximum_query["canonical_integer_source_count"] > 1
        and not maximum_query[
            "uniform_partition_weight_within_target_fiber"
        ]
    )
    return instance


@functools.lru_cache(maxsize=1)
def actual_and_matched_controls() -> dict[str, Any]:
    actual = [
        normalize_instance_flags(
            actual_instance(dict(family), offset, "actual")
        )
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    matched = [
        normalize_instance_flags(
            actual_instance(dict(family), offset, "matched_random_deck")
        )
        for family in R82.FAMILIES
        for offset in (2, 3)
    ]
    rows = [*actual, *matched]
    double_queries = [
        {
            "control_class": row["control_class"],
            "family_id": row["family_id"],
            "offset": row["offset"],
            "canonical_count": query["canonical_integer_source_count"],
            "partition_weights": query["partition_weights"],
            "balanced_resultant_vanishing_order": query[
                "balanced_resultant_vanishing_order"
            ],
        }
        for row in rows
        for query in row["query_controls"]
        if query["label"] == "maximum_multiplicity"
        and query["canonical_integer_source_count"] > 1
    ]
    return {
        "actual": actual,
        "matched_random_decks": matched,
        "actual_count": len(actual),
        "matched_random_deck_count": len(matched),
        "query_count": sum(len(row["query_controls"]) for row in rows),
        "all_partition_weight_identities_exact": all(
            row["all_partition_weight_identities_exact"] for row in rows
        ),
        "all_markers_recover_sources": all(
            row["all_markers_recover_sources"] for row in rows
        ),
        "all_recovered_sources_replay": all(
            row["all_recovered_sources_replay"] for row in rows
        ),
        "all_global_partition_weights_nonconstant": all(
            row["global_partition_weights_nonconstant"] for row in rows
        ),
        "double_fiber_queries": double_queries,
        "double_fiber_query_count": len(double_queries),
        "nonuniform_double_fiber_query_count": sum(
            len(set(row["partition_weights"])) > 1
            for row in double_queries
        ),
        "actual_nonuniform_double_fiber_present": any(
            row["control_class"] == "actual"
            and len(set(row["partition_weights"])) > 1
            for row in double_queries
        ),
    }


def cyclic_resultant_jet_control() -> dict[str, Any]:
    modulus = 11
    left = collections.Counter({1: 2, 3: 1, 7: 1})
    right = collections.Counter({2: 1, 5: 2})
    target = 6
    order = 0
    leading = 1
    for left_root, left_multiplicity in left.items():
        for right_root, right_multiplicity in right.items():
            multiplicity = left_multiplicity * right_multiplicity
            difference = (left_root + right_root - target) % modulus
            if difference == 0:
                order += multiplicity
            else:
                leading = (
                    leading * pow(difference, multiplicity, modulus)
                ) % modulus
    convolution_count = sum(
        multiplicity * right.get((target - root) % modulus, 0)
        for root, multiplicity in left.items()
    )
    return {
        "group": "Z/11Z",
        "left_root_multiplicities": dict(sorted(left.items())),
        "right_root_multiplicities": dict(sorted(right.items())),
        "target": target,
        "resultant_identity": (
            "Res(prod_l(X-l),prod_r(X-(T-r)))="
            "prod_(l,r)(l+r-T)"
        ),
        "lowest_nonzero_jet_order": order,
        "direct_convolution_count": convolution_count,
        "order_equals_convolution_count": order == convolution_count,
        "first_nonzero_coefficient_mod_11": leading,
        "first_nonzero_coefficient_nonzero": leading != 0,
        "duplicate_roots_present": any(value > 1 for value in left.values())
        or any(value > 1 for value in right.values()),
    }


def collapsed_deck_multiplicity_control() -> dict[str, Any]:
    size_a = 3
    size_c = 5
    canonical_count = (
        math.comb(size_a + FULL_A_COUNT - 1, FULL_A_COUNT)
        * math.comb(size_c + FULL_C_COUNT - 1, FULL_C_COUNT)
    )
    left_count = (
        math.comb(size_a + BALANCED_LEFT[0] - 1, BALANCED_LEFT[0])
        * math.comb(size_c + BALANCED_LEFT[1] - 1, BALANCED_LEFT[1])
    )
    right_count = (
        math.comb(size_a + BALANCED_RIGHT[0] - 1, BALANCED_RIGHT[0])
        * math.comb(size_c + BALANCED_RIGHT[1] - 1, BALANCED_RIGHT[1])
    )
    weighted_count = 0
    for source in partial_sources(
        size_a,
        size_c,
        FULL_A_COUNT,
        FULL_C_COUNT,
    ):
        weighted_count += balanced_partition_weight(source)
    return {
        "control": "all formal atom occurrences map to one endpoint",
        "factor_base_injectivity_required": False,
        "canonical_target_fiber_multiplicity": canonical_count,
        "balanced_left_occurrence_count": left_count,
        "balanced_right_occurrence_count": right_count,
        "balanced_resultant_vanishing_order": left_count * right_count,
        "sum_of_partition_weights": weighted_count,
        "weighted_identity_exact": weighted_count == left_count * right_count,
        "fixed_order_two_jet_insufficient": canonical_count > 2,
        "asymptotic_canonical_multiplicity_exponent_B": fraction_record(
            FULL_SOURCE_EXPONENT
        ),
        "purpose": (
            "A generic exact-count theorem cannot infer a constant jet "
            "order from the low multiplicities of the frozen actual decks."
        ),
    }


def projective_controls() -> dict[str, Any]:
    curve = dict(R82.FAMILIES[0])
    atoms_a, atoms_c, _factors, _geometry = R82.compact_factor_base(
        curve,
        0,
    )
    point = atoms_a[0]
    negative = R70.negate(point, curve)
    inverse_sum = R70.add(point, negative, curve)
    doubled = R70.add(point, point, curve)
    return {
        "signed_fp2_key_injective_on_control": (
            R84.point_key(point) != R84.point_key(negative)
        ),
        "identity_key_reserved": R84.point_key(None) == (0, 0),
        "inverse_pair_reaches_projective_identity": inverse_sum is None,
        "tangent_doubling_affine_on_control": doubled is not None,
        "identity_target_in_actual_queries": True,
        "group_verifier_projective_branches_exact": True,
        "candidate_homogeneous_resultant_charts_supplied": False,
        "candidate_work_credit": False,
    }


def algebraic_grammar_ledger() -> dict[str, Any]:
    partitions = root_partition_ledger()
    minimum = Fraction(
        partitions["minimum_interface_exponent_B"]["exact"]
    )
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_workspace_exponent_B": fraction_record(ONLINE_CAP),
        },
        "frozen_grammar": {
            "allowed": [
                "explicit endpoint Chow or coefficient vectors",
                "Sylvester and subresultant sequences",
                "quotient-free transposed cofactors",
                "sparse Macaulay determinant or determinant quotient",
                "constant-order target and marker truncation",
            ],
            "forbidden_unit_cost_oracles": [
                "resultant",
                "determinant",
                "gcd",
                "root",
                "source",
            ],
            "target_may_be_injected_at_any_gate": True,
            "character_or_dlp_coordinates_forbidden": True,
        },
        "all_binary_root_partitions": partitions,
        "best_binary_root_interface": {
            "degree_exponent_B": fraction_record(minimum),
            "minimum_is_B13O5": minimum == Fraction(13, 5),
            "inside_setup_cap": minimum <= SETUP_CAP,
            "inside_online_cap": minimum <= ONLINE_CAP,
            "reason": (
                "The 5A+5C source weight is five. Discrete 2/5 and 3/5 "
                "leaf weights make 12/5 versus 13/5 the best root split."
            ),
        },
        "explicit_side_coefficient_realization": {
            "minimum_coefficient_body_exponent_B": fraction_record(minimum),
            "inside_setup_cap": False,
            "inside_online_cap": False,
        },
        "sylvester_subresultant_realization": {
            "minimum_matrix_or_cofactor_dimension_exponent_B": (
                fraction_record(minimum)
            ),
            "constant_order_truncated_resultant_work_exponent_B": (
                fraction_record(minimum)
            ),
            "inside_setup_cap": False,
            "inside_online_cap": False,
            "literature_scope": (
                "Moroz-Schost gives soft-O(kd) for order-k truncated "
                "resultants represented with degree-d polynomial bodies. "
                "This charges the frozen standard representation; it is "
                "not an arithmetic-circuit lower bound."
            ),
        },
        "sparse_macaulay_realization": {
            "mixed_source_factor_exponent_B": fraction_record(
                FULL_SOURCE_EXPONENT
            ),
            "explicit_matrix_body_inside_caps": False,
            "literature_scope": (
                "Sparse Macaulay formulas express the resultant as a "
                "quotient of determinants; no determinant is unit cost."
            ),
        },
        "canonical_weight_correction": {
            "unrestricted_resultant_counts_split_representations": True,
            "canonical_source_weight_is_source_dependent": True,
            "constant_scalar_rescaling_exact": False,
            "r105_markers_can_identify_low_multiplicity_factors": True,
            "generic_factorization_or_count_correction_inside_caps": False,
        },
        "marker_channels": {
            "channel_count": MARKER_DIMENSION + 1,
            "channel_exponent_B": fraction_record(Fraction(0)),
            "changes_dominant_interface_exponent": False,
        },
        "scope_boundary": {
            "standard_explicit_resultant_grammar_closed": True,
            "arbitrary_factored_algebraic_or_rational_circuit_closed": False,
            "general_arithmetic_circuit_lower_bound_claimed": False,
            "elliptic_lambda_ring_or_chow_recurrence_open": True,
        },
    }


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        "r106_producer": {
            "path": str(R106_PRODUCER),
            "sha256": R106_PRODUCER_SHA256,
        },
        "r106_report": {
            "path": str(R106_REPORT),
            "sha256": R106_REPORT_SHA256,
        },
        "r106_gate": {
            "path": str(R106_GATE),
            "sha256": R106_GATE_SHA256,
        },
        "r106_parent": {
            "path": str(R106_PARENT),
            "sha256": R106_PARENT_SHA256,
        },
        "r105_producer": {
            "path": str(R105_PRODUCER),
            "sha256": R105_PRODUCER_SHA256,
        },
        "r105_report": {
            "path": str(R105_REPORT),
            "sha256": R105_REPORT_SHA256,
        },
        "r105_gate": {
            "path": str(R105_GATE),
            "sha256": R105_GATE_SHA256,
        },
        "r105_parent": {
            "path": str(R105_PARENT),
            "sha256": R105_PARENT_SHA256,
        },
        "r102_report": {
            "path": str(R102_REPORT),
            "sha256": R102_REPORT_SHA256,
        },
        "r102_gate": {
            "path": str(R102_GATE),
            "sha256": R102_GATE_SHA256,
        },
        "r84_report": {
            "path": str(R84_REPORT),
            "sha256": R84_REPORT_SHA256,
        },
        "r84_gate": {
            "path": str(R84_GATE),
            "sha256": R84_GATE_SHA256,
        },
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    controls = actual_and_matched_controls()
    cyclic = cyclic_resultant_jet_control()
    collapsed = collapsed_deck_multiplicity_control()
    projective = projective_controls()
    ledger = algebraic_grammar_ledger()
    obligations = {
        "twelve_source_bindings_verified": len(source_hashes) == 12,
        "thirty_four_binary_root_partitions_exhausted": (
            ledger["all_binary_root_partitions"]["partition_count"] == 34
        ),
        "best_root_interface_exponent_B13O5": ledger[
            "best_binary_root_interface"
        ]["minimum_is_B13O5"],
        "best_root_interface_misses_setup_cap": not ledger[
            "best_binary_root_interface"
        ]["inside_setup_cap"],
        "best_root_interface_misses_online_cap": not ledger[
            "best_binary_root_interface"
        ]["inside_online_cap"],
        "standard_truncated_resultant_body_over_cap": not ledger[
            "sylvester_subresultant_realization"
        ]["inside_online_cap"],
        "explicit_macaulay_body_over_cap": not ledger[
            "sparse_macaulay_realization"
        ]["explicit_matrix_body_inside_caps"],
        "all_actual_and_matched_weight_identities_exact": controls[
            "all_partition_weight_identities_exact"
        ],
        "all_actual_and_matched_markers_recover_sources": controls[
            "all_markers_recover_sources"
        ],
        "all_actual_and_matched_sources_replay": controls[
            "all_recovered_sources_replay"
        ],
        "actual_nonuniform_double_fiber_present": controls[
            "actual_nonuniform_double_fiber_present"
        ],
        "cyclic_resultant_jet_identity_exact": cyclic[
            "order_equals_convolution_count"
        ],
        "collapsed_deck_generic_multiplicity_control": collapsed[
            "fixed_order_two_jet_insufficient"
        ],
        "signed_and_identity_group_verifier_controls": (
            projective["signed_fp2_key_injective_on_control"]
            and projective["inverse_pair_reaches_projective_identity"]
            and projective["tangent_doubling_affine_on_control"]
        ),
        "standard_explicit_resultant_grammar_closed": ledger[
            "scope_boundary"
        ]["standard_explicit_resultant_grammar_closed"],
        "arbitrary_factored_algebraic_circuit_closed": ledger[
            "scope_boundary"
        ]["arbitrary_factored_algebraic_or_rational_circuit_closed"],
        "canonical_scalar_norm_inside_caps": False,
        "candidate_homogeneous_projective_charts_complete": projective[
            "candidate_homogeneous_resultant_charts_supplied"
        ],
        "generic_multiplicity_bound_complete": False,
        "asymptotic_integer_lift_complete": False,
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
        "BALANCED_NONCHARACTER_ROOT_RESULTANT_REQUIRES_B13O5_INTERFACE__"
        "UNRESTRICTED_RESULTANT_COUNTS_SOURCE_DEPENDENT_PARTITION_WEIGHTS__"
        "ACTUAL_DOUBLE_FIBERS_HAVE_NONUNIFORM_WEIGHTS__STANDARD_TRUNCATED_"
        "SUBRESULTANT_AND_MACAULAY_GRAMMARS_OVER_CAP__FACTORED_ELLIPTIC_"
        "LAMBDA_RING_CHOW_CIRCUIT_OPEN"
    )
    next_action = (
        "Construct or refute one factored elliptic lambda-ring/Chow-form "
        "recurrence for the canonical, not partition-weighted, 5A+5C target "
        "norm. It must consume compact D_A,D_C and T without explicit side "
        "coefficients, Sylvester/subresultant vectors, or Macaulay bodies; "
        "support the eleven R105 marker channels, homogeneous projective "
        "branches, generic multiplicity and integer lifting; and fit both "
        "caps before rank, factor logs, and identical descent."
    )
    frozen = {
        "schema": "p1553.frozen_5a5c_noncharacter_algebraic_target_norm.r107.v1",
        "campaign": "P1553",
        "round": "R107",
        "public_inputs": "D_A=sum_i[A_i], D_C=sum_j[C_j], target T",
        "source_arity": {"a": FULL_A_COUNT, "c": FULL_C_COUNT},
        "atom_exponents_B": {
            "a": fraction_record(A_ATOM_EXPONENT),
            "c": fraction_record(C_ATOM_EXPONENT),
        },
        "balanced_root_split": {
            "left": {"a": 2, "c": 3},
            "right": {"a": 3, "c": 2},
        },
        "grammar": ledger["frozen_grammar"],
        "target_and_partitions_frozen_before_outcomes": True,
        "actual_offsets": list(R82.INSTANCE_OFFSETS),
        "matched_random_offsets": [2, 3],
        "query_labels": [
            "unique_min_partition_weight",
            "unique_max_partition_weight",
            "maximum_multiplicity",
            "blind",
            "projective_identity",
        ],
        "source_bindings": source_binding_records(),
        "scalar_labels_consumed_by_candidate": False,
    }
    marker_replay = {
        "schema": "p1553.noncharacter_norm_marker_jet_replay.r107.v1",
        "identity": (
            "ord_T Res(D_L,D_R(T-*)) = sum_(s:endpoint(s)=T) w_split(s)"
        ),
        "canonical_weight_formula": (
            "w_split(s)=#[bounded 2-submultisets of s_A] * "
            "#[bounded 3-submultisets of s_C]"
        ),
        "controls": controls,
        "all_weighted_marker_factor_multiplicities_exact": controls[
            "all_partition_weight_identities_exact"
        ],
        "constant_scalar_normalization_refuted": controls[
            "all_global_partition_weights_nonconstant"
        ],
        "actual_nonuniform_double_fiber_present": controls[
            "actual_nonuniform_double_fiber_present"
        ],
        "candidate_work_credit": False,
    }
    exceptional = {
        "schema": "p1553.noncharacter_norm_exceptional_controls.r107.v1",
        "cyclic_resultant_jet": cyclic,
        "collapsed_deck_multiplicity": collapsed,
        "signed_projective": projective,
        "candidate_homogeneous_projective_charts_complete": False,
        "generic_multiplicity_and_integer_lift_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r107.v1",
        "relation_source": (
            "No candidate relation generator survives the scalar constructor "
            "and canonical-weight gates."
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
            "SCOPED_STANDARD_RESULTANT_GRAMMAR_NEGATIVE_NO_ALGORITHM_CLAIM"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_hashes_verified": source_hashes,
        "primary_sources": [
            {
                "url": "https://arxiv.org/abs/1609.04259",
                "use": (
                    "soft-O(kd) truncated resultant algorithm for explicit "
                    "degree-d polynomial bodies"
                ),
            },
            {
                "url": "https://arxiv.org/abs/math/0107181",
                "use": (
                    "sparse Macaulay resultant formulas as determinant "
                    "quotients"
                ),
            },
        ],
        "theorem": {
            "all_root_partitions": ledger[
                "all_binary_root_partitions"
            ],
            "best_interface": ledger["best_binary_root_interface"],
            "partition_weight_identity": marker_replay["identity"],
            "scope": ledger["scope_boundary"],
        },
        "finite_controls": {
            "actual_count": controls["actual_count"],
            "matched_random_deck_count": controls[
                "matched_random_deck_count"
            ],
            "query_count": controls["query_count"],
            "double_fiber_queries": controls["double_fiber_queries"],
            "cyclic_resultant_jet": cyclic,
            "collapsed_deck_multiplicity": collapsed,
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
            "frozen": "frozen_5a5c_noncharacter_algebraic_target_norm.json",
            "circuit_ledger": "noncharacter_norm_resultant_gate_ledger.json",
            "marker_replay": "noncharacter_norm_marker_jet_replay.json",
            "exceptional": "noncharacter_norm_exceptional_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r107.json",
        },
        "next_action": next_action,
        "non_claims": [
            "No general arithmetic-circuit lower bound is proved.",
            "No factored elliptic lambda-ring or Chow-form circuit is ruled out.",
            "No generic-prime ECDLP algorithm is constructed.",
            "No rank, factor-log, or fresh-target descent gate passes.",
            "No Pollard-rho or Shoup-bound improvement is claimed.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "REJECT_STANDARD_EXPLICIT_NONCHARACTER_RESULTANT_GRAMMARS_ONLY__"
            "B13O5_MINIMUM_ROOT_INTERFACE__CANONICAL_PARTITION_WEIGHTS_"
            "NONUNIFORM__FACTORED_LAMBDA_RING_CHOW_CIRCUIT_OPEN__NO_RANK__"
            "NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "circuit_ledger": ledger,
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
            "p1553_5a5c_noncharacter_algebraic_target_norm_"
            "resultant_probe_report_r107.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_noncharacter_algebraic_target_norm.json"
        ),
    )
    parser.add_argument(
        "--circuit-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "noncharacter_norm_resultant_gate_ledger.json"
        ),
    )
    parser.add_argument(
        "--marker-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "noncharacter_norm_marker_jet_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "noncharacter_norm_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r107.json"
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
    write_json(args.circuit_output, bundle["circuit_ledger"])
    write_json(args.marker_output, bundle["marker_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    admission = bundle["report"]["admission"]
    print(
        f"R107 classification={bundle['report']['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
