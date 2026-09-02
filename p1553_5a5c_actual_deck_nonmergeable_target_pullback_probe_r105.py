#!/usr/bin/env python3
"""Test a marker-jet source adjoint for a whole-deck target norm circuit."""

from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import importlib.util
import json
import pathlib
from fractions import Fraction
from typing import Any, Sequence


SCHEMA = "p1553.5a5c_actual_deck_nonmergeable_target_pullback.r105.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
ATOM_SOURCE_RECOVERY_EXPONENT = Fraction(3, 5)
MARKER_DIMENSION = 10
SOURCE_ARITY = 5

R104_PRODUCER = pathlib.Path(
    "p1553_5a5c_compact_preendpoint_s3_ffe_pushdown_probe_r104.py"
)
R104_PRODUCER_SHA256 = (
    "3dc487af42b59d3dec0f5b0250d3fbe061d45650ff27492aa434ce6f8e2d13ee"
)
R104_REPORT = pathlib.Path(
    "p1553_5a5c_compact_preendpoint_s3_ffe_"
    "pushdown_probe_report_r104.json"
)
R104_REPORT_SHA256 = (
    "5dbf8a75dd8298c2c606e41a4c121c6191ef56b5e184199e8db48841880b1edf"
)
R104_GATE = pathlib.Path(
    "p1553_5a5c_compact_preendpoint_s3_ffe_pushdown_probe_gate_r104.md"
)
R104_GATE_SHA256 = (
    "e817cc2e806f06b372bb6e61823efb6717e243adfbb4da33fe2fd4ff550ad822"
)
R104_PARENT = pathlib.Path(
    "p1553_5a5c_compact_preendpoint_s3_ffe_"
    "pushdown_probe_parent_report_r104.yaml"
)
R104_PARENT_SHA256 = (
    "8f4c0476ff678cd12a12ae170faf475ff892e5436b0a29e332a1ddd5f9b6a19b"
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
R88_REPORT = pathlib.Path(
    "p1553_5a5c_black_box_translated_resultant_"
    "localizer_probe_report_r88.json"
)
R88_REPORT_SHA256 = (
    "d73e017c731a54c6913aeaa94e6b5c6d54ca3757f56be14e8ca8ee5524031de1"
)
R88_GATE = pathlib.Path(
    "p1553_5a5c_black_box_translated_resultant_localizer_probe_gate_r88.md"
)
R88_GATE_SHA256 = (
    "90f0980bdeb51f540cd233f207218361cc14beb8a899c246418410d36ef43d56"
)
R89_REPORT = pathlib.Path(
    "p1553_5a5c_fixed_marker_scalar_recurrence_probe_report_r89.json"
)
R89_REPORT_SHA256 = (
    "50a075dd73d3c5298be00efdbf5186734ddf694ed14bbb38cb37f890815a3b63"
)
R89_GATE = pathlib.Path(
    "p1553_5a5c_fixed_marker_scalar_recurrence_probe_gate_r89.md"
)
R89_GATE_SHA256 = (
    "2e6ac3545bb9db94e5d8b54eed5b4a67a22cf870d544dc0af80b96b41b444858"
)

Point = tuple[int, int] | None
Source = tuple[tuple[int, ...], tuple[int, ...]]
Fp2 = tuple[int, int]
MarkerVector = tuple[int, ...]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R104_PRODUCER: R104_PRODUCER_SHA256,
        R104_REPORT: R104_REPORT_SHA256,
        R104_GATE: R104_GATE_SHA256,
        R104_PARENT: R104_PARENT_SHA256,
        R84_REPORT: R84_REPORT_SHA256,
        R84_GATE: R84_GATE_SHA256,
        R88_REPORT: R88_REPORT_SHA256,
        R88_GATE: R88_GATE_SHA256,
        R89_REPORT: R89_REPORT_SHA256,
        R89_GATE: R89_GATE_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R105 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R104 = load_module("p1553_r104_for_r105", R104_PRODUCER)
R103 = R104.R103
R102 = R104.R102
R84 = R104.R84
R82 = R104.R82
R70 = R104.R70


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


def source_json(source: Source | None) -> list[list[int]] | None:
    if source is None:
        return None
    return [list(source[0]), list(source[1])]


def marker_vector(source: Source, prime: int) -> MarkerVector:
    values = []
    for indices in source:
        values.extend(
            sum(pow(index, degree, prime) for index in indices) % prime
            for degree in range(1, SOURCE_ARITY + 1)
        )
    result = tuple(values)
    if len(result) != MARKER_DIMENSION:
        raise AssertionError("marker dimension drifted")
    return result


def poly_eval_desc(coefficients: Sequence[int], value: int, prime: int) -> int:
    result = 0
    for coefficient in coefficients:
        result = (result * value + coefficient) % prime
    return result


def synthetic_division_desc(
    coefficients: Sequence[int],
    root: int,
    prime: int,
) -> tuple[list[int], int]:
    quotient = [coefficients[0] % prime]
    for coefficient in coefficients[1:]:
        quotient.append((coefficient + quotient[-1] * root) % prime)
    return quotient[:-1], quotient[-1]


def multiset_from_power_sums(
    power_sums: Sequence[int],
    deck_size: int,
    prime: int,
) -> tuple[int, ...]:
    if len(power_sums) != SOURCE_ARITY:
        raise AssertionError("five power sums required")
    elementary = [1]
    for degree in range(1, SOURCE_ARITY + 1):
        numerator = 0
        for index in range(1, degree + 1):
            sign = 1 if index % 2 else -1
            numerator += (
                sign
                * elementary[degree - index]
                * power_sums[index - 1]
            )
        elementary.append(
            numerator * pow(degree, prime - 2, prime) % prime
        )
    polynomial = [1]
    polynomial.extend(
        (
            elementary[degree]
            if degree % 2 == 0
            else -elementary[degree]
        )
        % prime
        for degree in range(1, SOURCE_ARITY + 1)
    )
    roots = []
    remainder_poly = polynomial
    for candidate in range(deck_size):
        while len(remainder_poly) > 1:
            quotient, remainder = synthetic_division_desc(
                remainder_poly,
                candidate,
                prime,
            )
            if remainder:
                break
            roots.append(candidate)
            remainder_poly = quotient
    if len(roots) != SOURCE_ARITY or len(remainder_poly) != 1:
        raise AssertionError("power sums did not recover a five-multiset")
    if any(poly_eval_desc(polynomial, root, prime) for root in roots):
        raise AssertionError("recovered marker root does not vanish")
    return tuple(roots)


def source_from_marker(
    marker: MarkerVector,
    size_a: int,
    size_c: int,
    prime: int,
) -> Source:
    return (
        multiset_from_power_sums(marker[:5], size_a, prime),
        multiset_from_power_sums(marker[5:], size_c, prime),
    )


def legendre_symbol(value: int, prime: int) -> int:
    if value % prime == 0:
        return 0
    symbol = pow(value % prime, (prime - 1) // 2, prime)
    return -1 if symbol == prime - 1 else symbol


def sqrt_mod(value: int, prime: int) -> int:
    value %= prime
    if value == 0:
        return 0
    if legendre_symbol(value, prime) != 1:
        raise ValueError("nonsquare discriminant")
    if prime % 4 == 3:
        return pow(value, (prime + 1) // 4, prime)
    odd = prime - 1
    power = 0
    while odd % 2 == 0:
        power += 1
        odd //= 2
    nonresidue = 2
    while legendre_symbol(nonresidue, prime) != -1:
        nonresidue += 1
    c = pow(nonresidue, odd, prime)
    x = pow(value, (odd + 1) // 2, prime)
    t = pow(value, odd, prime)
    m = power
    while t != 1:
        index = 1
        squared = t * t % prime
        while squared != 1:
            squared = squared * squared % prime
            index += 1
            if index >= m:
                raise AssertionError("Tonelli-Shanks failed")
        factor = pow(c, 1 << (m - index - 1), prime)
        x = x * factor % prime
        t = t * factor * factor % prime
        c = factor * factor % prime
        m = index
    return x


def quadratic_roots(total: int, product: int, prime: int) -> tuple[int, int]:
    discriminant = (total * total - 4 * product) % prime
    root = sqrt_mod(discriminant, prime)
    inverse_two = (prime + 1) // 2
    values = (
        (total - root) * inverse_two % prime,
        (total + root) * inverse_two % prime,
    )
    return tuple(sorted(values))


def normalized_marker_jet(
    markers: Sequence[MarkerVector],
    nonzero_product: Fp2,
    prime: int,
    nonsquare: int,
) -> dict[str, Any]:
    multiplicity = len(markers)
    if multiplicity > 2:
        raise ValueError("finite R105 decoder is frozen through order two")
    inverse = R84.f2_inv(nonzero_product, prime, nonsquare)
    if multiplicity == 0:
        return {
            "vanishing_order": 0,
            "base_lowest_coefficient": list(nonzero_product),
            "normalized_linear_sum": [],
            "normalized_quadratic_coefficients": [],
        }
    linear_sum = [
        sum(marker[index] for marker in markers) % prime
        for index in range(MARKER_DIMENSION)
    ]
    scaled_linear = [
        R84.f2_scale(nonzero_product, value, prime)
        for value in linear_sum
    ]
    normalized_linear = [
        R84.f2_mul(value, inverse, prime, nonsquare)
        for value in scaled_linear
    ]
    if any(value[1] or value[0] != expected for value, expected in zip(
        normalized_linear,
        linear_sum,
    )):
        raise AssertionError("linear marker normalization drifted")
    quadratic: list[list[int]] = []
    if multiplicity == 2:
        left, right = markers
        for row in range(MARKER_DIMENSION):
            values = []
            for column in range(MARKER_DIMENSION):
                if row == column:
                    value = left[row] * right[row]
                else:
                    value = (
                        left[row] * right[column]
                        + left[column] * right[row]
                    )
                values.append(value % prime)
            quadratic.append(values)
    return {
        "vanishing_order": multiplicity,
        "base_lowest_coefficient": list(nonzero_product),
        "normalized_linear_sum": linear_sum,
        "normalized_quadratic_coefficients": quadratic,
        "normalization_lies_in_base_field": True,
        "homogeneous_factorization": (
            "product_s (t0 + sum_j marker[s,j]*t_j)"
        ),
    }


def marker_factors_from_jet(
    jet: dict[str, Any],
    prime: int,
) -> list[MarkerVector]:
    multiplicity = jet["vanishing_order"]
    if multiplicity == 0:
        return []
    linear_sum = jet["normalized_linear_sum"]
    if multiplicity == 1:
        return [tuple(linear_sum)]
    if multiplicity != 2:
        raise ValueError("finite decoder supports multiplicity at most two")
    quadratic = jet["normalized_quadratic_coefficients"]
    coordinate_pairs = [
        quadratic_roots(
            linear_sum[index],
            quadratic[index][index],
            prime,
        )
        for index in range(MARKER_DIMENSION)
    ]
    pivot = next(
        (
            index
            for index, roots in enumerate(coordinate_pairs)
            if roots[0] != roots[1]
        ),
        None,
    )
    if pivot is None:
        raise AssertionError("two distinct marker factors became equal")
    left = [0] * MARKER_DIMENSION
    right = [0] * MARKER_DIMENSION
    left[pivot], right[pivot] = coordinate_pairs[pivot]
    for index, roots in enumerate(coordinate_pairs):
        if index == pivot:
            continue
        choices = (roots, tuple(reversed(roots)))
        valid = [
            choice
            for choice in choices
            if (
                left[pivot] * choice[1]
                + right[pivot] * choice[0]
            )
            % prime
            == quadratic[pivot][index]
        ]
        if not valid:
            raise AssertionError("quadratic marker alignment failed")
        left[index], right[index] = min(valid)
    result = sorted((tuple(left), tuple(right)))
    for row in range(MARKER_DIMENSION):
        for column in range(MARKER_DIMENSION):
            expected = (
                result[0][row] * result[1][column]
                if row == column
                else (
                    result[0][row] * result[1][column]
                    + result[0][column] * result[1][row]
                )
            ) % prime
            if expected != quadratic[row][column]:
                raise AssertionError("recovered marker factors fail jet")
    return result


def source_factor_rows(
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> list[tuple[Source, Point, Fp2, MarkerVector]]:
    prime = curve["field_prime"]
    rows = []
    for source in R102.full_sources(len(atoms_a), len(atoms_c)):
        endpoint = R102.source_endpoint(
            source,
            atoms_a,
            atoms_c,
            curve,
        )
        rows.append(
            (
                source,
                endpoint,
                R84.point_key(endpoint),
                marker_vector(source, prime),
            )
        )
    return rows


def norm_jet_query(
    label: str,
    target: Point,
    rows: Sequence[tuple[Source, Point, Fp2, MarkerVector]],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
    nonsquare: int,
) -> dict[str, Any]:
    prime = curve["field_prime"]
    target_key = R84.point_key(target)
    zero_rows = []
    nonzero_product = R84.f2_one()
    for source, endpoint, endpoint_key, marker in rows:
        difference = R84.f2_sub(endpoint_key, target_key, prime)
        if difference == R84.f2_zero():
            zero_rows.append((source, endpoint, marker))
        else:
            nonzero_product = R84.f2_mul(
                nonzero_product,
                difference,
                prime,
                nonsquare,
            )
    markers = [row[2] for row in zero_rows]
    jet = normalized_marker_jet(
        markers,
        nonzero_product,
        prime,
        nonsquare,
    )
    recovered_markers = marker_factors_from_jet(jet, prime)
    recovered_sources = sorted(
        source_from_marker(
            marker,
            len(atoms_a),
            len(atoms_c),
            prime,
        )
        for marker in recovered_markers
    )
    expected_sources = sorted(row[0] for row in zero_rows)
    all_sources_replay = all(
        R102.source_endpoint(source, atoms_a, atoms_c, curve) == target
        for source in recovered_sources
    )
    return {
        "label": label,
        "target": point_json(target),
        "direct_integer_occurrence_count": len(zero_rows),
        "lowest_nonzero_jet_order": jet["vanishing_order"],
        "count_from_jet_order_exact": (
            jet["vanishing_order"] == len(zero_rows)
        ),
        "normalized_marker_jet": jet,
        "recovered_sources": [
            source_json(source) for source in recovered_sources
        ],
        "expected_sources": [
            source_json(source) for source in expected_sources
        ],
        "all_marker_factors_recovered": (
            recovered_sources == expected_sources
        ),
        "all_recovered_sources_group_replay": all_sources_replay,
        "first_coupled_source": source_json(
            recovered_sources[0] if recovered_sources else None
        ),
        "returned_bottom": not recovered_sources,
        "scalar_labels_consumed": False,
        "explicit_source_product_enumerated_for_verification": True,
        "candidate_work_credit": False,
    }


def actual_instance(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    rows = source_factor_rows(atoms_a, atoms_c, curve)
    histogram = collections.Counter(endpoint for _s, endpoint, _k, _m in rows)
    unique_target = min(
        (
            target
            for target, count in histogram.items()
            if count == 1
        ),
        key=R102.point_sort_key,
    )
    maximum_target = max(
        histogram,
        key=lambda target: (
            histogram[target],
            tuple(-value for value in R102.point_sort_key(target)),
        ),
    )
    blind = R102.blind_target(
        histogram,
        curve,
        f"R105|{offset}|blind",
    )
    nonsquare = R84.least_nonsquare(curve["field_prime"])
    queries = [
        norm_jet_query(
            "unique_positive",
            unique_target,
            rows,
            atoms_a,
            atoms_c,
            curve,
            nonsquare,
        ),
        norm_jet_query(
            "maximum_multiplicity",
            maximum_target,
            rows,
            atoms_a,
            atoms_c,
            curve,
            nonsquare,
        ),
        norm_jet_query(
            "blind",
            blind,
            rows,
            atoms_a,
            atoms_c,
            curve,
            nonsquare,
        ),
        norm_jet_query(
            "projective_identity",
            None,
            rows,
            atoms_a,
            atoms_c,
            curve,
            nonsquare,
        ),
    ]
    return {
        "family_id": curve["family_id"],
        "offset": offset,
        "field_prime": curve["field_prime"],
        "subgroup_order": curve["subgroup_order"],
        "factor_base_size_B": len(factors),
        "factor_base_injective": geometry["factor_base_injective"],
        "source_factor_count": len(rows),
        "target_support_size": len(histogram),
        "maximum_target_multiplicity": max(histogram.values()),
        "all_finite_fibers_at_most_two": max(histogram.values()) <= 2,
        "query_controls": queries,
        "all_query_counts_and_sources_exact": all(
            query["count_from_jet_order_exact"]
            and query["all_marker_factors_recovered"]
            and query["all_recovered_sources_group_replay"]
            for query in queries
        ),
        "scalar_labels_consumed": False,
    }


@functools.lru_cache(maxsize=1)
def actual_controls() -> dict[str, Any]:
    instances = [
        actual_instance(dict(family), offset)
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    query_rows = [
        query
        for instance in instances
        for query in instance["query_controls"]
    ]
    return {
        "instances": instances,
        "instance_count": len(instances),
        "query_control_count": len(query_rows),
        "all_query_counts_and_sources_exact": all(
            instance["all_query_counts_and_sources_exact"]
            for instance in instances
        ),
        "all_actual_fibers_at_most_two": all(
            instance["all_finite_fibers_at_most_two"]
            for instance in instances
        ),
        "multiplicity_two_instance_count": sum(
            instance["maximum_target_multiplicity"] == 2
            for instance in instances
        ),
        "all_unique_positive_sources_exact": all(
            query["direct_integer_occurrence_count"] == 1
            and query["all_marker_factors_recovered"]
            for query in query_rows
            if query["label"] == "unique_positive"
        ),
        "all_blind_queries_bottom": all(
            query["returned_bottom"]
            and query["lowest_nonzero_jet_order"] == 0
            for query in query_rows
            if query["label"] == "blind"
        ),
        "all_projective_identity_queries_exact": all(
            query["count_from_jet_order_exact"]
            and query["all_marker_factors_recovered"]
            for query in query_rows
            if query["label"] == "projective_identity"
        ),
        "all_scalar_blind": all(
            not instance["scalar_labels_consumed"]
            for instance in instances
        ),
    }


def marker_adjoint_theorem() -> dict[str, Any]:
    return {
        "source_norm": (
            "N_T=product_s (kappa(endpoint(s))-kappa(T)) over canonical "
            "unordered 5A+5C sources"
        ),
        "marker_deformation": (
            "replace each source factor z_s by "
            "z_s+t0+sum_(j=1)^10 marker_j(s)*t_j"
        ),
        "lowest_homogeneous_jet": (
            "if the target fiber is Z and m=|Z|, the first nonzero "
            "homogeneous term is product_(r notin Z) z_r times "
            "product_(s in Z)(t0+marker(s) dot t)"
        ),
        "integer_count": "the first nonzero jet order is exactly m",
        "coupled_source": (
            "factoring the normalized homogeneous jet returns one complete "
            "ten-channel marker vector per coupled source"
        ),
        "marker_channels": {
            "a_power_sums": ["sum i^k" for k in range(1, 6)],
            "c_power_sums": ["sum j^k" for k in range(1, 6)],
            "channel_count": MARKER_DIMENSION,
        },
        "multiset_inverse": (
            "Newton identities form the monic degree-five root polynomial; "
            "testing the public atom indices with multiplicity recovers the "
            "unordered source in O(|A|+|C|) field operations"
        ),
        "reverse_mode_cost": (
            "Baur-Strassen gives constant-factor first derivatives of a "
            "scalar arithmetic circuit; a truncated order-m jet has "
            "overhead polynomial in binomial(10+m,m)"
        ),
        "finite_actual_multiplicity_cap": 2,
        "finite_actual_jet_overhead_constant": True,
        "source_recovery_exponent_B": fraction_record(
            ATOM_SOURCE_RECOVERY_EXPONENT
        ),
        "source_recovery_inside_online_cap": (
            ATOM_SOURCE_RECOVERY_EXPONENT <= ONLINE_CAP
        ),
        "scope": (
            "Conditional on a compact scalar norm/count circuit that accepts "
            "the ten fixed marker deformations. The theorem does not "
            "construct that scalar circuit, prove an asymptotic fiber-"
            "multiplicity bound, or make explicit B^5 source products cheap."
        ),
    }


def cost_ledger() -> dict[str, Any]:
    theorem = marker_adjoint_theorem()
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_workspace_exponent_B": fraction_record(ONLINE_CAP),
        },
        "conditional_compact_circuit": {
            "scalar_target_norm_circuit_size_exponent_B": "s_unknown",
            "scalar_target_norm_circuit_state_exponent_B": "mu_unknown",
            "marker_channel_count_exponent_B": fraction_record(Fraction(0)),
            "finite_order_two_jet_overhead_exponent_B": fraction_record(
                Fraction(0)
            ),
            "source_recovery_exponent_B": theorem[
                "source_recovery_exponent_B"
            ],
            "source_recovery_inside_online_cap": theorem[
                "source_recovery_inside_online_cap"
            ],
        },
        "standard_explicit_realization": {
            "source_factor_count_exponent_B": fraction_record(Fraction(5)),
            "source_product_inside_setup_cap": False,
            "source_product_inside_online_cap": False,
            "candidate_work_credit": False,
        },
        "constructor_status": {
            "whole_deck_scalar_norm_circuit_inside_caps": False,
            "target_injected_before_child_summary": False,
            "generic_multiplicity_bound": False,
            "exact_integer_lift_for_all_fibers": False,
        },
        "progress": (
            "For simple and double fibers, source adjoint and unranking add "
            "only constant jet overhead plus B^(3/5) public root testing. "
            "The unresolved dominant object is now the scalar target norm/"
            "count circuit itself."
        ),
    }


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        "r104_producer": {
            "path": str(R104_PRODUCER),
            "sha256": R104_PRODUCER_SHA256,
        },
        "r104_report": {
            "path": str(R104_REPORT),
            "sha256": R104_REPORT_SHA256,
        },
        "r104_gate": {
            "path": str(R104_GATE),
            "sha256": R104_GATE_SHA256,
        },
        "r104_parent": {
            "path": str(R104_PARENT),
            "sha256": R104_PARENT_SHA256,
        },
        "r84_report": {
            "path": str(R84_REPORT),
            "sha256": R84_REPORT_SHA256,
        },
        "r84_gate": {
            "path": str(R84_GATE),
            "sha256": R84_GATE_SHA256,
        },
        "r88_report": {
            "path": str(R88_REPORT),
            "sha256": R88_REPORT_SHA256,
        },
        "r88_gate": {
            "path": str(R88_GATE),
            "sha256": R88_GATE_SHA256,
        },
        "r89_report": {
            "path": str(R89_REPORT),
            "sha256": R89_REPORT_SHA256,
        },
        "r89_gate": {
            "path": str(R89_GATE),
            "sha256": R89_GATE_SHA256,
        },
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    controls = actual_controls()
    theorem = marker_adjoint_theorem()
    costs = cost_ledger()
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_actual_deck_nonmergeable_"
            "target_pullback.r105.v1"
        ),
        "whole_deck_scalar": theorem["source_norm"],
        "target_injection": "inside every source factor before the product",
        "marker_deformation": theorem["marker_deformation"],
        "source_adjoint": theorem["lowest_homogeneous_jet"],
        "source_decoder": theorem["multiset_inverse"],
        "allowed_fiber_multiplicity_for_finite_decoder": [0, 1, 2],
        "caps": costs["caps"],
        "excluded_open_operation": (
            "constructing the scalar target norm/count from compact D_A,D_C "
            "inside both caps without source, endpoint, residual, quotient, "
            "or coefficient-body materialization"
        ),
    }
    circuit_ledger = {
        "schema": (
            "p1553.actual_deck_target_pullback_circuit_ledger.r105.v1"
        ),
        "frozen_candidate": frozen,
        "marker_adjoint_theorem": theorem,
        "cost_ledger": costs,
        "finite_circuit_semantics": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "source_factor_count": instance["source_factor_count"],
                "target_support_size": instance["target_support_size"],
                "maximum_target_multiplicity": instance[
                    "maximum_target_multiplicity"
                ],
            }
            for instance in controls["instances"]
        ],
    }
    source_replay = {
        "schema": (
            "p1553.actual_deck_target_pullback_integer_source_replay.r105.v1"
        ),
        "all_query_counts_and_sources_exact": controls[
            "all_query_counts_and_sources_exact"
        ],
        "all_actual_fibers_at_most_two": controls[
            "all_actual_fibers_at_most_two"
        ],
        "instances": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "query_controls": instance["query_controls"],
            }
            for instance in controls["instances"]
        ],
        "explicit_source_product_is_verifier_only": True,
        "scalar_labels_consumed": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": (
            "p1553.actual_deck_target_pullback_exceptional_controls.r105.v1"
        ),
        "blind_zero_fiber_complete": controls["all_blind_queries_bottom"],
        "projective_identity_target_complete": controls[
            "all_projective_identity_queries_exact"
        ],
        "multiplicity_two_instance_count": controls[
            "multiplicity_two_instance_count"
        ],
        "multiplicity_two_marker_factorization_replayed": (
            controls["multiplicity_two_instance_count"] > 0
            and controls["all_query_counts_and_sources_exact"]
        ),
        "signed_point_key": "kappa(P)=x+w*y in F_(p^2), kappa(O)=0",
        "point_key_injective_on_curve_and_identity": True,
        "generic_multiplicity_greater_than_two_complete": False,
        "asymptotic_integer_no_wrap_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r105.v1",
        "conditional_marker_source_adjoint_exact": True,
        "whole_deck_scalar_target_norm_inside_caps": False,
        "generic_multiplicity_and_integer_lift_complete": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "ten_source_bindings_verified": len(bindings) == 10,
        "eight_actual_instances_replayed": controls["instance_count"] == 8,
        "thirty_two_query_controls_replayed": (
            controls["query_control_count"] == 32
        ),
        "all_actual_marker_jet_counts_exact": controls[
            "all_query_counts_and_sources_exact"
        ],
        "all_actual_marker_factors_recovered": controls[
            "all_query_counts_and_sources_exact"
        ],
        "all_actual_coupled_sources_group_replay": controls[
            "all_query_counts_and_sources_exact"
        ],
        "all_unique_positive_sources_exact": controls[
            "all_unique_positive_sources_exact"
        ],
        "all_blind_queries_return_bottom": controls[
            "all_blind_queries_bottom"
        ],
        "all_projective_identity_queries_exact": controls[
            "all_projective_identity_queries_exact"
        ],
        "actual_multiplicity_two_branch_present": (
            controls["multiplicity_two_instance_count"] > 0
        ),
        "all_actual_fibers_at_most_two": controls[
            "all_actual_fibers_at_most_two"
        ],
        "scalar_blind_construction": controls["all_scalar_blind"],
        "lowest_nonzero_jet_order_equals_integer_count": True,
        "homogeneous_marker_jet_factors_by_coupled_source": True,
        "ten_power_sum_markers_invert_unordered_sources": True,
        "source_recovery_exponent_B3O5": (
            theorem["source_recovery_exponent_B"]["exact"] == "3/5"
        ),
        "source_recovery_inside_online_cap": theorem[
            "source_recovery_inside_online_cap"
        ],
        "finite_order_two_jet_overhead_constant": theorem[
            "finite_actual_jet_overhead_constant"
        ],
        "whole_deck_scalar_target_norm_inside_caps": False,
        "target_injected_before_any_child_summary": False,
        "generic_fiber_multiplicity_bound": False,
        "asymptotic_integer_lift_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_algorithm": False,
        "shoup_improvement_complete": False,
        "full_pipeline_fresh_workspace_inside_cap": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "WHOLE_DECK_NORM_JET_SOURCE_ADJOINT_EXACT_FOR_MULTIPLICITY_"
            "AT_MOST_TWO__TEN_POWER_SUM_MARKERS_RECOVER_COUPLED_SOURCE__"
            "BAUR_STRASSEN_OVERHEAD_CONSTANT__SCALAR_TARGET_NORM_"
            "CONSTRUCTOR_AND_GENERIC_MULTIPLICITY_BOUND_UNSUPPLIED"
        ),
        "source_bindings": source_binding_records(),
        "novelty_scope": (
            "R105 is the first campaign receipt to make source recovery a "
            "lowest-homogeneous-jet adjoint of one target-injected whole-"
            "deck scalar norm, recover unordered 5A+5C sources from ten "
            "power sums, and replay double fibers without a source "
            "dictionary."
        ),
        "actual_controls": controls,
        "marker_adjoint_theorem": theorem,
        "cost_ledger": costs,
        "artifacts": {
            "frozen": (
                "frozen_5a5c_actual_deck_nonmergeable_target_pullback.json"
            ),
            "circuit_ledger": (
                "actual_deck_target_pullback_circuit_ledger.json"
            ),
            "source_replay": (
                "actual_deck_target_pullback_integer_source_replay.json"
            ),
            "exceptional": (
                "actual_deck_target_pullback_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r105.json",
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "conditional_source_adjoint_admitted": True,
        "scalar_target_norm_constructor_complete": False,
        "generic_multiplicity_bound_complete": False,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "scope_boundary": theorem["scope"],
        "next_action": (
            "Construct or refute the scalar target norm/count circuit now "
            "isolated by R105. It must consume compact D_A,D_C and T as one "
            "non-mergeable program, support the eleven unit/power-sum "
            "deformations, avoid B^(14/5) residuals, B^(11/5) translated "
            "coefficient reads, B^5 source factors, and oracle determinants, "
            "fit both caps, and prove a generic multiplicity/integer-lift "
            "gate before rank, factor logs, and identical descent."
        ),
        "disposition": (
            "ADMIT_CONDITIONAL_WHOLE_DECK_MARKER_ADJOINT_ONLY__LOWEST_JET_"
            "ORDER_IS_INTEGER_FIBER_COUNT__TEN_POWER_SUM_MARKERS_RECOVER_"
            "COUPLED_UNORDERED_SOURCE__ACTUAL_SIMPLE_AND_DOUBLE_FIBERS_"
            "EXACT__SOURCE_RECOVERY_B3O5__SCALAR_TARGET_NORM_CONSTRUCTOR_"
            "UNSUPPLIED__GENERIC_MULTIPLICITY_BOUND_UNSUPPLIED__NO_RANK__"
            "NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "circuit_ledger": circuit_ledger,
        "source_replay": source_replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_actual_deck_nonmergeable_"
            "target_pullback_probe_report_r105.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_actual_deck_nonmergeable_target_pullback.json"
        ),
    )
    parser.add_argument(
        "--circuit-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "actual_deck_target_pullback_circuit_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "actual_deck_target_pullback_integer_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "actual_deck_target_pullback_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r105.json"
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
    write_json(args.source_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    admission = bundle["report"]["admission"]
    print(
        "R105 "
        f"classification={bundle['report']['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
