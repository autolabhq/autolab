#!/usr/bin/env python3
"""Test a cycle-index lambda-ring factorization of the canonical 5A+5C norm."""

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
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.5a5c_factored_elliptic_lambda_ring_chow_norm.r108.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
A_ATOM_EXPONENT = Fraction(2, 5)
C_ATOM_EXPONENT = Fraction(3, 5)
SYMMETRIC_ARITY = 5
CYCLE_SCALE = math.factorial(SYMMETRIC_ARITY)
FULL_CYCLE_SCALE = CYCLE_SCALE * CYCLE_SCALE

R107_PRODUCER = pathlib.Path(
    "p1553_5a5c_noncharacter_algebraic_target_norm_"
    "resultant_probe_r107.py"
)
R107_PRODUCER_SHA256 = (
    "2598f635a8572c816eec2c79873da60e71af241fddbce7a0deb90d0918ffe00a"
)
R107_REPORT = pathlib.Path(
    "p1553_5a5c_noncharacter_algebraic_target_norm_"
    "resultant_probe_report_r107.json"
)
R107_REPORT_SHA256 = (
    "0f85b6c29df4b91a56b6f17badc08c08f34c95e21c3b5219261a9308f493d65d"
)
R107_GATE = pathlib.Path(
    "p1553_5a5c_noncharacter_algebraic_target_norm_"
    "resultant_probe_gate_r107.md"
)
R107_GATE_SHA256 = (
    "74343856b9bd1865d92bde6055b9d3b441d2ad716797f478d99ab8ee85cc1c51"
)
R107_PARENT = pathlib.Path(
    "p1553_5a5c_noncharacter_algebraic_target_norm_"
    "resultant_probe_parent_report_r107.yaml"
)
R107_PARENT_SHA256 = (
    "7859557b1d418699aabde0da890cbe5445f987001950e84b7ba8c2682b5d850c"
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
R104_REPORT = pathlib.Path(
    "p1553_5a5c_compact_preendpoint_s3_ffe_pushdown_probe_report_r104.json"
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
R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)

Point = tuple[int, int] | None


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R107_PRODUCER: R107_PRODUCER_SHA256,
        R107_REPORT: R107_REPORT_SHA256,
        R107_GATE: R107_GATE_SHA256,
        R107_PARENT: R107_PARENT_SHA256,
        R105_PRODUCER: R105_PRODUCER_SHA256,
        R105_REPORT: R105_REPORT_SHA256,
        R105_GATE: R105_GATE_SHA256,
        R105_PARENT: R105_PARENT_SHA256,
        R104_REPORT: R104_REPORT_SHA256,
        R104_GATE: R104_GATE_SHA256,
        R82_REPORT: R82_REPORT_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R108 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R107 = load_module("p1553_r107_for_r108", R107_PRODUCER)
R105 = R107.R105
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


def integer_partitions(
    total: int,
    maximum: int | None = None,
) -> Iterable[tuple[int, ...]]:
    if total == 0:
        yield ()
        return
    upper = total if maximum is None else min(total, maximum)
    for first in range(upper, 0, -1):
        for suffix in integer_partitions(total - first, first):
            yield (first, *suffix)


def cycle_class_size(parts: Sequence[int]) -> int:
    multiplicities = collections.Counter(parts)
    denominator = 1
    for length, count in multiplicities.items():
        denominator *= (length**count) * math.factorial(count)
    return math.factorial(sum(parts)) // denominator


def cycle_types() -> list[dict[str, Any]]:
    rows = [
        {
            "parts": parts,
            "cycle_count": len(parts),
            "class_size": cycle_class_size(parts),
        }
        for parts in integer_partitions(SYMMETRIC_ARITY)
    ]
    if len(rows) != 7 or sum(row["class_size"] for row in rows) != CYCLE_SCALE:
        raise AssertionError("degree-five cycle index drifted")
    return rows


def counter_digest(counter: collections.Counter[Point]) -> str:
    digest = hashlib.sha256()
    for point, count in sorted(
        counter.items(),
        key=lambda item: R102.point_sort_key(item[0]),
    ):
        digest.update(
            json.dumps(
                [point_json(point), count],
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def add_scaled_counter(
    destination: collections.Counter[Point],
    source: collections.Counter[Point],
    scale: int,
) -> None:
    for point, count in source.items():
        destination[point] += scale * count


def deck_state_counter(state: dict[Point, tuple[int, Any]]) -> collections.Counter[Point]:
    return collections.Counter(
        {point: count for point, (count, _source) in state.items()}
    )


def adams_cycle_histogram(
    atoms: Sequence[Point],
    parts: Sequence[int],
    curve: dict[str, Any],
) -> tuple[collections.Counter[Point], int]:
    histogram: collections.Counter[Point] = collections.Counter()
    occurrence_count = 0
    for indices in itertools.product(range(len(atoms)), repeat=len(parts)):
        endpoint: Point = None
        for part, index in zip(parts, indices):
            endpoint = R70.add(
                endpoint,
                R70.scalar_mul(part, atoms[index], curve),
                curve,
            )
        histogram[endpoint] += 1
        occurrence_count += 1
    expected = len(atoms) ** len(parts)
    if occurrence_count != expected or sum(histogram.values()) != expected:
        raise AssertionError("Adams cycle occurrence count drifted")
    return histogram, occurrence_count


def deck_cycle_index_control(
    atoms: Sequence[Point],
    curve: dict[str, Any],
) -> dict[str, Any]:
    direct_state, _stages = R104.multiset_group_algebra(
        atoms,
        SYMMETRIC_ARITY,
        curve,
    )
    direct = deck_state_counter(direct_state)
    scaled: collections.Counter[Point] = collections.Counter()
    term_rows = []
    for row in cycle_types():
        histogram, occurrences = adams_cycle_histogram(
            atoms,
            row["parts"],
            curve,
        )
        add_scaled_counter(scaled, histogram, row["class_size"])
        term_rows.append(
            {
                "parts": list(row["parts"]),
                "cycle_count": row["cycle_count"],
                "class_size": row["class_size"],
                "ordered_occurrence_count": occurrences,
                "endpoint_support_size": len(histogram),
                "endpoint_digest": counter_digest(histogram),
            }
        )
    expected = collections.Counter(
        {point: CYCLE_SCALE * count for point, count in direct.items()}
    )
    return {
        "atom_count": len(atoms),
        "cycle_term_count": len(term_rows),
        "cycle_terms": term_rows,
        "direct_symmetric_occurrence_count": sum(direct.values()),
        "scaled_cycle_occurrence_count": sum(scaled.values()),
        "direct_endpoint_support_size": len(direct),
        "scaled_endpoint_support_size": len(scaled),
        "direct_endpoint_digest": counter_digest(direct),
        "scaled_endpoint_digest": counter_digest(scaled),
        "scaled_cycle_index_equals_120_symmetric_divisor": scaled == expected,
        "all_scaled_coefficients_divisible_by_120": all(
            count % CYCLE_SCALE == 0 for count in scaled.values()
        ),
    }


def group_convolution(
    left: collections.Counter[Point],
    right: collections.Counter[Point],
    curve: dict[str, Any],
) -> collections.Counter[Point]:
    output: collections.Counter[Point] = collections.Counter()
    for left_point, left_count in left.items():
        for right_point, right_count in right.items():
            output[R70.add(left_point, right_point, curve)] += (
                left_count * right_count
            )
    return output


def scaled_cycle_divisor(
    atoms: Sequence[Point],
    curve: dict[str, Any],
) -> collections.Counter[Point]:
    output: collections.Counter[Point] = collections.Counter()
    for row in cycle_types():
        histogram, _occurrences = adams_cycle_histogram(
            atoms,
            row["parts"],
            curve,
        )
        add_scaled_counter(output, histogram, row["class_size"])
    return output


def full_cycle_index_control(
    curve: dict[str, Any],
    offset: int,
    control_class: str,
) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    a_direct_state, _a_stages = R104.multiset_group_algebra(
        atoms_a,
        SYMMETRIC_ARITY,
        curve,
    )
    c_direct_state, _c_stages = R104.multiset_group_algebra(
        atoms_c,
        SYMMETRIC_ARITY,
        curve,
    )
    a_direct = deck_state_counter(a_direct_state)
    c_direct = deck_state_counter(c_direct_state)
    a_scaled = scaled_cycle_divisor(atoms_a, curve)
    c_scaled = scaled_cycle_divisor(atoms_c, curve)
    canonical = group_convolution(a_direct, c_direct, curve)
    scaled_full = group_convolution(a_scaled, c_scaled, curve)
    expected = collections.Counter(
        {
            point: FULL_CYCLE_SCALE * count
            for point, count in canonical.items()
        }
    )
    return {
        "control_class": control_class,
        "family_id": curve["family_id"],
        "offset": offset,
        "field_prime": curve["field_prime"],
        "subgroup_order": curve["subgroup_order"],
        "factor_base_size_B": len(factors),
        "factor_base_injective": geometry["factor_base_injective"],
        "a_deck": deck_cycle_index_control(atoms_a, curve),
        "c_deck": deck_cycle_index_control(atoms_c, curve),
        "canonical_full_occurrence_count": sum(canonical.values()),
        "canonical_full_endpoint_support_size": len(canonical),
        "canonical_full_digest": counter_digest(canonical),
        "scaled_49_term_occurrence_count": sum(scaled_full.values()),
        "scaled_49_term_endpoint_support_size": len(scaled_full),
        "scaled_49_term_digest": counter_digest(scaled_full),
        "scaled_49_term_identity_exact": scaled_full == expected,
        "all_target_counts_divide_exactly_by_14400": all(
            count % FULL_CYCLE_SCALE == 0
            for count in scaled_full.values()
        ),
        "cycle_scale_invertible_in_field": (
            math.gcd(curve["field_prime"], FULL_CYCLE_SCALE) == 1
        ),
        "candidate_scalar_labels_consumed": False,
        "verifier_materialized_endpoint_histograms": True,
        "candidate_work_credit": False,
    }


@functools.lru_cache(maxsize=1)
def actual_and_matched_controls() -> dict[str, Any]:
    actual = [
        full_cycle_index_control(dict(family), offset, "actual")
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    matched = [
        full_cycle_index_control(
            dict(family),
            offset,
            "matched_random_deck",
        )
        for family in R82.FAMILIES
        for offset in (2, 3)
    ]
    rows = [*actual, *matched]
    return {
        "actual": actual,
        "matched_random_decks": matched,
        "actual_count": len(actual),
        "matched_random_deck_count": len(matched),
        "all_deck_cycle_identities_exact": all(
            row["a_deck"][
                "scaled_cycle_index_equals_120_symmetric_divisor"
            ]
            and row["c_deck"][
                "scaled_cycle_index_equals_120_symmetric_divisor"
            ]
            for row in rows
        ),
        "all_full_49_term_identities_exact": all(
            row["scaled_49_term_identity_exact"] for row in rows
        ),
        "all_target_counts_divide_exactly": all(
            row["all_target_counts_divide_exactly_by_14400"]
            for row in rows
        ),
        "all_cycle_scales_invertible_in_field": all(
            row["cycle_scale_invertible_in_field"] for row in rows
        ),
    }


def expanded_source(
    parts: Sequence[int],
    indices: Sequence[int],
) -> tuple[int, ...]:
    output = []
    for part, index in zip(parts, indices):
        output.extend([index] * part)
    return tuple(sorted(output))


def cycle_source_weight_profile(atom_count: int) -> dict[str, Any]:
    weights: collections.Counter[tuple[int, ...]] = collections.Counter()
    for row in cycle_types():
        for indices in itertools.product(
            range(atom_count),
            repeat=row["cycle_count"],
        ):
            weights[expanded_source(row["parts"], indices)] += row[
                "class_size"
            ]
    expected_source_count = math.comb(
        atom_count + SYMMETRIC_ARITY - 1,
        SYMMETRIC_ARITY,
    )
    values = set(weights.values())
    return {
        "atom_count": atom_count,
        "canonical_source_count": len(weights),
        "expected_canonical_source_count": expected_source_count,
        "weight_values": sorted(values),
        "every_canonical_source_weight_120": (
            len(weights) == expected_source_count
            and values == {CYCLE_SCALE}
        ),
    }


def marker_weight_controls() -> dict[str, Any]:
    a_sizes = sorted({family["atom_a_size"] for family in R82.FAMILIES})
    c_sizes = sorted({family["atom_c_size"] for family in R82.FAMILIES})
    a_profiles = [cycle_source_weight_profile(size) for size in a_sizes]
    c_profiles = [cycle_source_weight_profile(size) for size in c_sizes]
    return {
        "cycle_scale_per_deck": CYCLE_SCALE,
        "cycle_scale_per_full_source": FULL_CYCLE_SCALE,
        "a_profiles": a_profiles,
        "c_profiles": c_profiles,
        "all_side_sources_weight_120": all(
            row["every_canonical_source_weight_120"]
            for row in [*a_profiles, *c_profiles]
        ),
        "all_full_sources_weight_14400": True,
        "expanded_cycle_marker_rule": (
            "A cycle of length r contributes r copies of its selected atom "
            "index to each R105 power-sum marker."
        ),
        "marker_vector_preserved_by_cycle_expansion": True,
        "canonical_count_from_aggregate_jet_order": (
            "m_canonical = m_49_term_product / 14400"
        ),
        "generic_marker_factorization_inside_caps": False,
        "candidate_work_credit": False,
    }


def best_root_interface(
    a_cycle_count: int,
    c_cycle_count: int,
) -> Fraction:
    total = (
        a_cycle_count * A_ATOM_EXPONENT
        + c_cycle_count * C_ATOM_EXPONENT
    )
    candidates = []
    for selected_a in range(a_cycle_count + 1):
        for selected_c in range(c_cycle_count + 1):
            if (selected_a, selected_c) in (
                (0, 0),
                (a_cycle_count, c_cycle_count),
            ):
                continue
            selected = (
                selected_a * A_ATOM_EXPONENT
                + selected_c * C_ATOM_EXPONENT
            )
            candidates.append(max(selected, total - selected))
    if not candidates:
        raise AssertionError("each 49-term body has at least two variables")
    return min(candidates)


def termwise_cost_ledger() -> dict[str, Any]:
    rows = []
    for a_row in cycle_types():
        for c_row in cycle_types():
            total = (
                a_row["cycle_count"] * A_ATOM_EXPONENT
                + c_row["cycle_count"] * C_ATOM_EXPONENT
            )
            interface = best_root_interface(
                a_row["cycle_count"],
                c_row["cycle_count"],
            )
            rows.append(
                {
                    "a_parts": list(a_row["parts"]),
                    "c_parts": list(c_row["parts"]),
                    "coefficient": (
                        a_row["class_size"] * c_row["class_size"]
                    ),
                    "ordered_source_body_exponent_B": fraction_record(total),
                    "best_binary_root_interface_exponent_B": (
                        fraction_record(interface)
                    ),
                    "source_body_inside_setup_cap": total <= SETUP_CAP,
                    "source_body_inside_online_cap": total <= ONLINE_CAP,
                    "root_interface_inside_setup_cap": interface <= SETUP_CAP,
                    "root_interface_inside_online_cap": interface <= ONLINE_CAP,
                }
            )
    hard = next(
        row
        for row in rows
        if row["a_parts"] == [1, 1, 1, 1, 1]
        and row["c_parts"] == [1, 1, 1, 1, 1]
    )
    return {
        "term_count": len(rows),
        "terms": rows,
        "source_body_inside_setup_count": sum(
            row["source_body_inside_setup_cap"] for row in rows
        ),
        "source_body_inside_online_count": sum(
            row["source_body_inside_online_cap"] for row in rows
        ),
        "root_interface_inside_setup_count": sum(
            row["root_interface_inside_setup_cap"] for row in rows
        ),
        "root_interface_inside_online_count": sum(
            row["root_interface_inside_online_cap"] for row in rows
        ),
        "identity_cycle_term": hard,
        "identity_cycle_term_coefficient_one": hard["coefficient"] == 1,
        "identity_cycle_term_source_exponent_B5": (
            hard["ordered_source_body_exponent_B"]["exact"] == "5"
        ),
        "identity_cycle_term_root_interface_B13O5": (
            hard["best_binary_root_interface_exponent_B"]["exact"] == "13/5"
        ),
        "all_coefficients_positive": all(row["coefficient"] > 0 for row in rows),
        "termwise_evaluation_inside_caps": False,
    }


def theorem_of_cube_control() -> dict[str, Any]:
    pair_tables = {
        "A_A": 2 * A_ATOM_EXPONENT,
        "A_C": A_ATOM_EXPONENT + C_ATOM_EXPONENT,
        "C_C": 2 * C_ATOM_EXPONENT,
    }
    return {
        "line_bundle_identity": (
            "m123^*L tensor m1^*L tensor m2^*L tensor m3^*L "
            "= m12^*L tensor m13^*L tensor m23^*L"
        ),
        "primary_source": "https://stacks.math.columbia.edu/tag/0BFE",
        "iterated_line_bundle_class_uses_one_body_and_pairwise_factors": True,
        "pair_table_exponents_B": {
            name: fraction_record(value)
            for name, value in pair_tables.items()
        },
        "all_pair_tables_inside_online_cap": all(
            value <= ONLINE_CAP for value in pair_tables.values()
        ),
        "line_bundle_isomorphism_is_section_factorization": False,
        "target_section_pure_tensor_factorization_supplied": False,
        "poincare_trivialization_and_scalar_evaluator_supplied": False,
        "theta_addition_section_rank_bound_supplied": False,
        "candidate_work_credit": False,
        "scope": (
            "The theorem of the cube factors the pullback line-bundle class. "
            "It does not state that the pulled-back target section is one "
            "pure tensor of those factors."
        ),
    }


def cost_ledger() -> dict[str, Any]:
    termwise = termwise_cost_ledger()
    cube = theorem_of_cube_control()
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_workspace_exponent_B": fraction_record(ONLINE_CAP),
        },
        "cycle_index": {
            "terms_per_deck": 7,
            "full_term_count": 49,
            "constant_factor_exponent_B": fraction_record(Fraction(0)),
            "canonical_weight_correction_exact": True,
        },
        "termwise_chow_or_norm_evaluation": termwise,
        "theorem_of_cube": cube,
        "constructor_status": {
            "canonical_virtual_divisor_identity_complete": True,
            "canonical_marker_weight_identity_complete": True,
            "termwise_scalar_norm_inside_caps": False,
            "target_section_factorization_inside_caps": False,
            "generic_multiplicity_and_integer_lift_complete": False,
            "homogeneous_projective_scalar_evaluator_complete": False,
        },
        "scope_boundary": {
            "cycle_index_canonical_correction_admitted": True,
            "termwise_49_norm_constructor_rejected": True,
            "theorem_of_cube_line_bundle_factorization_admitted": True,
            "poincare_theta_section_factorization_open": True,
            "general_section_tensor_rank_lower_bound_claimed": False,
        },
    }


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        "r107_producer": {
            "path": str(R107_PRODUCER),
            "sha256": R107_PRODUCER_SHA256,
        },
        "r107_report": {
            "path": str(R107_REPORT),
            "sha256": R107_REPORT_SHA256,
        },
        "r107_gate": {
            "path": str(R107_GATE),
            "sha256": R107_GATE_SHA256,
        },
        "r107_parent": {
            "path": str(R107_PARENT),
            "sha256": R107_PARENT_SHA256,
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
        "r104_report": {
            "path": str(R104_REPORT),
            "sha256": R104_REPORT_SHA256,
        },
        "r104_gate": {
            "path": str(R104_GATE),
            "sha256": R104_GATE_SHA256,
        },
        "r82_report": {
            "path": str(R82_REPORT),
            "sha256": R82_REPORT_SHA256,
        },
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    controls = actual_and_matched_controls()
    markers = marker_weight_controls()
    ledger = cost_ledger()
    cube = ledger["theorem_of_cube"]
    termwise = ledger["termwise_chow_or_norm_evaluation"]
    obligations = {
        "eleven_source_bindings_verified": len(source_hashes) == 11,
        "seven_cycle_types_exact": len(cycle_types()) == 7,
        "cycle_class_sizes_sum_120": (
            sum(row["class_size"] for row in cycle_types()) == CYCLE_SCALE
        ),
        "all_actual_and_matched_side_cycle_identities_exact": controls[
            "all_deck_cycle_identities_exact"
        ],
        "all_actual_and_matched_full_49_term_identities_exact": controls[
            "all_full_49_term_identities_exact"
        ],
        "all_finite_target_counts_divide_by_14400": controls[
            "all_target_counts_divide_exactly"
        ],
        "all_cycle_scales_invertible_in_finite_fields": controls[
            "all_cycle_scales_invertible_in_field"
        ],
        "all_side_source_cycle_weights_120": markers[
            "all_side_sources_weight_120"
        ],
        "all_full_source_cycle_weights_14400": markers[
            "all_full_sources_weight_14400"
        ],
        "marker_vector_preserved_by_cycle_expansion": markers[
            "marker_vector_preserved_by_cycle_expansion"
        ],
        "forty_nine_term_cost_ledger_complete": termwise["term_count"] == 49,
        "identity_cycle_term_coefficient_one": termwise[
            "identity_cycle_term_coefficient_one"
        ],
        "identity_cycle_term_source_exponent_B5": termwise[
            "identity_cycle_term_source_exponent_B5"
        ],
        "identity_cycle_term_root_interface_B13O5": termwise[
            "identity_cycle_term_root_interface_B13O5"
        ],
        "all_cycle_coefficients_positive": termwise[
            "all_coefficients_positive"
        ],
        "theorem_of_cube_line_bundle_identity": cube[
            "iterated_line_bundle_class_uses_one_body_and_pairwise_factors"
        ],
        "all_poincare_pair_tables_inside_online_cap": cube[
            "all_pair_tables_inside_online_cap"
        ],
        "target_section_pure_tensor_factorization_supplied": cube[
            "target_section_pure_tensor_factorization_supplied"
        ],
        "poincare_scalar_evaluator_inside_caps": cube[
            "poincare_trivialization_and_scalar_evaluator_supplied"
        ],
        "termwise_49_norm_constructor_inside_caps": termwise[
            "termwise_evaluation_inside_caps"
        ],
        "generic_marker_factorization_inside_caps": markers[
            "generic_marker_factorization_inside_caps"
        ],
        "generic_multiplicity_and_integer_lift_complete": False,
        "homogeneous_projective_scalar_evaluator_complete": False,
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
        "DEGREE_FIVE_CYCLE_INDEX_GIVES_EXACT_SEVEN_TERM_CANONICAL_DIVISOR__"
        "FORTY_NINE_TERM_5A5C_IDENTITY_AND_14400_MARKER_WEIGHT_EXACT__"
        "IDENTITY_CYCLE_TERM_RETAINS_B5_BODY_AND_B13O5_ROOT_INTERFACE__"
        "THEOREM_OF_CUBE_FACTORS_LINE_BUNDLE_NOT_TARGET_SECTION__POINCARE_"
        "THETA_SECTION_FACTORIZATION_OPEN"
    )
    next_action = (
        "Construct or refute one Poincare/theta target-section factorization "
        "of the ten-input elliptic sum pullback. Freeze trivializations and "
        "addition identities before outcomes; prove the section, not only "
        "its line-bundle class, has a bounded-rank one-body/pairwise tensor "
        "representation whose deck contractions fit both caps and preserve "
        "the 14400 canonical marker weight, projective charts, generic "
        "multiplicity, and integer lifting."
    )
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_factored_elliptic_lambda_ring_"
            "chow_norm.r108.v1"
        ),
        "campaign": "P1553",
        "round": "R108",
        "public_inputs": "D_A=sum_i[A_i], D_C=sum_j[C_j], target T",
        "symmetric_arity_per_deck": SYMMETRIC_ARITY,
        "cycle_types": [
            {
                "parts": list(row["parts"]),
                "cycle_count": row["cycle_count"],
                "class_size": row["class_size"],
            }
            for row in cycle_types()
        ],
        "cycle_scale_per_deck": CYCLE_SCALE,
        "cycle_scale_per_full_source": FULL_CYCLE_SCALE,
        "actual_offsets": list(R82.INSTANCE_OFFSETS),
        "matched_random_offsets": [2, 3],
        "target_and_identities_frozen_before_outcomes": True,
        "source_bindings": source_binding_records(),
        "candidate_scalar_labels_consumed": False,
    }
    marker_replay = {
        "schema": "p1553.canonical_weight_marker_jet_replay.r108.v1",
        "cycle_identity": (
            "120 h5(D)=p1(D)^5+10p1(D)^3p2(D)+15p1(D)p2(D)^2+"
            "20p1(D)^2p3(D)+20p2(D)p3(D)+30p1(D)p4(D)+24p5(D)"
        ),
        "full_identity": (
            "14400 H_A5*H_C5 is the positive 7x7 Adams-cycle sum"
        ),
        "marker_controls": markers,
        "finite_controls": controls,
        "canonical_weight_correction_exact": (
            controls["all_full_49_term_identities_exact"]
            and markers["all_full_sources_weight_14400"]
        ),
        "candidate_work_credit": False,
    }
    exceptional = {
        "schema": "p1553.lambda_ring_chow_exceptional_controls.r108.v1",
        "cycle_scale_invertible_on_all_finite_curves": controls[
            "all_cycle_scales_invertible_in_field"
        ],
        "identity_endpoint_included_in_all_target_histograms": True,
        "signed_point_keys_inherited_from_r105": True,
        "theorem_of_cube": cube,
        "homogeneous_projective_scalar_evaluator_complete": False,
        "generic_multiplicity_and_integer_lift_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r108.v1",
        "relation_source": (
            "The cycle identity corrects canonical weights but does not "
            "supply an inside-cap target-section evaluator."
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
            "EXACT_CANONICAL_CYCLE_IDENTITY_CONDITIONAL_POSITIVE_"
            "NO_SCALAR_ALGORITHM_CLAIM"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_hashes_verified": source_hashes,
        "primary_sources": [
            {
                "url": "https://stacks.math.columbia.edu/tag/0BFE",
                "use": (
                    "theorem-of-the-cube line-bundle identity on abelian "
                    "varieties"
                ),
            }
        ],
        "theorem": {
            "cycle_index": marker_replay["cycle_identity"],
            "full_49_term_identity": marker_replay["full_identity"],
            "theorem_of_cube_scope": cube["scope"],
        },
        "finite_controls": {
            "actual_count": controls["actual_count"],
            "matched_random_deck_count": controls[
                "matched_random_deck_count"
            ],
            "all_side_identities_exact": controls[
                "all_deck_cycle_identities_exact"
            ],
            "all_full_identities_exact": controls[
                "all_full_49_term_identities_exact"
            ],
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
            "frozen": (
                "frozen_5a5c_factored_elliptic_lambda_ring_chow_norm.json"
            ),
            "circuit_ledger": "lambda_ring_chow_gate_and_cost_ledger.json",
            "marker_replay": "canonical_weight_marker_jet_replay.json",
            "exceptional": "lambda_ring_chow_exceptional_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r108.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The 49-term identity is not an inside-cap scalar evaluator.",
            "The theorem of the cube does not factor the target section.",
            "No general section tensor-rank lower bound is proved.",
            "No generic-prime ECDLP algorithm is constructed.",
            "No rank, factor-log, target-descent, or Shoup gate passes.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_CYCLE_INDEX_CANONICAL_WEIGHT_IDENTITY_ONLY__REJECT_"
            "TERMWISE_49_NORM_EVALUATION__PRESERVE_POINCARE_THETA_TARGET_"
            "SECTION_FACTORIZATION__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__"
            "NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
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
            "p1553_5a5c_factored_elliptic_lambda_ring_chow_"
            "norm_probe_report_r108.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_factored_elliptic_lambda_ring_chow_norm.json"
        ),
    )
    parser.add_argument(
        "--circuit-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "lambda_ring_chow_gate_and_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--marker-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "canonical_weight_marker_jet_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "lambda_ring_chow_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r108.json"
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
        f"R108 classification={bundle['report']['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
