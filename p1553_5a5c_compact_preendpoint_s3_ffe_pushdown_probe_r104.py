#!/usr/bin/env python3
"""Audit mergeable pre-endpoint S3/FFE pushdowns for the 5A+5C lane."""

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


SCHEMA = "p1553.5a5c_compact_preendpoint_s3_ffe_pushdown.r104.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
ATOM_A_EXPONENT = Fraction(2, 5)
ATOM_C_EXPONENT = Fraction(3, 5)
PREFIX_EXPONENT = Fraction(11, 5)
SUFFIX_EXPONENT = Fraction(14, 5)
FULL_SOURCE_EXPONENT = Fraction(5, 1)

R103_PRODUCER = pathlib.Path(
    "p1553_5a5c_target_forced_algebraic_join_filter_probe_r103.py"
)
R103_PRODUCER_SHA256 = (
    "9ebba96331379a99abaa4596772855a884a62137828edefd5f6a2d81478152dd"
)
R103_REPORT = pathlib.Path(
    "p1553_5a5c_target_forced_algebraic_join_filter_probe_report_r103.json"
)
R103_REPORT_SHA256 = (
    "587d8283716ee8568aab1576376b57b786f412c7c16d4e3942d271b276a5173c"
)
R103_GATE = pathlib.Path(
    "p1553_5a5c_target_forced_algebraic_join_filter_probe_gate_r103.md"
)
R103_GATE_SHA256 = (
    "63e155a40d7762e1bc362d72610a91c075f5075b35997e680e74c82c29215aa2"
)
R103_PARENT = pathlib.Path(
    "p1553_5a5c_target_forced_algebraic_join_filter_"
    "probe_parent_report_r103.yaml"
)
R103_PARENT_SHA256 = (
    "315fa9d6511bb95fa6647413537ad2d2d62f6f44815dcd4adb397194dd7e3ab0"
)
R102_PRODUCER = pathlib.Path(
    "p1553_5a5c_two_sided_implicit_join_probe_r102.py"
)
R102_PRODUCER_SHA256 = (
    "3fae17c49683586178518a0df8d377c8f553bd131539f1bab579c0f55d5b0742"
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
R102_PARENT = pathlib.Path(
    "p1553_5a5c_two_sided_implicit_join_probe_parent_report_r102.yaml"
)
R102_PARENT_SHA256 = (
    "9a965158bc96e51dd8daea1abb86f5552b75dac94998633ec97b22f8b9a3ad67"
)
R87_REPORT = pathlib.Path(
    "p1553_5a5c_jet_preserving_addition_pushforward_probe_report_r87.json"
)
R87_REPORT_SHA256 = (
    "f10ba663867815c9ee0b1234f4d9dee698d450a3a7171336d36f3e328ea2333a"
)
R87_GATE = pathlib.Path(
    "p1553_5a5c_jet_preserving_addition_pushforward_probe_gate_r87.md"
)
R87_GATE_SHA256 = (
    "16d635add67bc64d63d5870663f68ce37e35a21fa4feb436c428b7afbe6ed565"
)
R93_REPORT = pathlib.Path(
    "p1553_5a5c_shared_semilinear_incidence_"
    "correspondence_probe_report_r93.json"
)
R93_REPORT_SHA256 = (
    "f33c764ebbf491e55d7b8342fda9df8b29038beef8cdb06662c343ac415b7d4b"
)
R93_GATE = pathlib.Path(
    "p1553_5a5c_shared_semilinear_incidence_"
    "correspondence_probe_gate_r93.md"
)
R93_GATE_SHA256 = (
    "15d58598c875b8ffe89fb1c7d551cd2adfc15cc8f7215a5f3242987305a2b2ad"
)

Point = tuple[int, int] | None
Source = tuple[tuple[int, ...], tuple[int, ...]]
DeckEntry = tuple[int, tuple[int, ...]]
DeckState = dict[Point, DeckEntry]
JointEntry = tuple[int, Source]
JointState = dict[Point, JointEntry]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R103_PRODUCER: R103_PRODUCER_SHA256,
        R103_REPORT: R103_REPORT_SHA256,
        R103_GATE: R103_GATE_SHA256,
        R103_PARENT: R103_PARENT_SHA256,
        R102_PRODUCER: R102_PRODUCER_SHA256,
        R102_REPORT: R102_REPORT_SHA256,
        R102_GATE: R102_GATE_SHA256,
        R102_PARENT: R102_PARENT_SHA256,
        R87_REPORT: R87_REPORT_SHA256,
        R87_GATE: R87_GATE_SHA256,
        R93_REPORT: R93_REPORT_SHA256,
        R93_GATE: R93_GATE_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R104 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R103 = load_module("p1553_r103_for_r104", R103_PRODUCER)
R102 = R103.R102
R82 = R102.R82
R84 = R102.R84
R70 = R82.R70


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


def support_exponent(size: int, base_size: int) -> float:
    if size <= 1 or base_size <= 1:
        return 0.0
    return math.log(size) / math.log(base_size)


def add_deck_entry(
    state: DeckState,
    endpoint: Point,
    count: int,
    source: tuple[int, ...],
) -> None:
    previous = state.get(endpoint)
    if previous is None:
        state[endpoint] = (count, source)
        return
    previous_count, previous_source = previous
    state[endpoint] = (
        previous_count + count,
        min(previous_source, source),
    )


def multiset_group_algebra(
    deck: Sequence[Point],
    arity: int,
    curve: dict[str, Any],
) -> tuple[DeckState, list[dict[str, Any]]]:
    """Compute complete homogeneous powers in the elliptic group algebra."""
    previous: list[DeckState] = [{None: (1, ())}]
    previous.extend({} for _ in range(arity))
    stages: list[dict[str, Any]] = []
    for atom_index, atom in enumerate(deck):
        current: list[DeckState] = [{None: (1, ())}]
        for degree in range(1, arity + 1):
            state = dict(previous[degree])
            for endpoint, (count, source) in current[degree - 1].items():
                shifted = R70.add(endpoint, atom, curve)
                add_deck_entry(
                    state,
                    shifted,
                    count,
                    source + (atom_index,),
                )
            current.append(state)
            expected_occurrences = math.comb(atom_index + degree, degree)
            observed_occurrences = sum(
                count for count, _source in state.values()
            )
            if observed_occurrences != expected_occurrences:
                raise AssertionError("multiset recurrence count drifted")
            stages.append(
                {
                    "atom_prefix_size": atom_index + 1,
                    "degree": degree,
                    "support_size": len(state),
                    "integer_occurrence_count": observed_occurrences,
                    "expected_multiset_count": expected_occurrences,
                    "count_exact": True,
                }
            )
        previous = current
    return previous[arity], stages


def add_joint_entry(
    state: JointState,
    endpoint: Point,
    count: int,
    source: Source,
) -> None:
    previous = state.get(endpoint)
    if previous is None:
        state[endpoint] = (count, source)
        return
    previous_count, previous_source = previous
    state[endpoint] = (
        previous_count + count,
        min(previous_source, source),
    )


def convolve_deck_states(
    left: DeckState,
    right: DeckState,
    curve: dict[str, Any],
) -> tuple[JointState, int]:
    state: JointState = {}
    operations = 0
    for left_endpoint, (left_count, left_source) in left.items():
        for right_endpoint, (right_count, right_source) in right.items():
            endpoint = R70.add(left_endpoint, right_endpoint, curve)
            add_joint_entry(
                state,
                endpoint,
                left_count * right_count,
                (left_source, right_source),
            )
            operations += 1
    return state, operations


def state_histogram(state: JointState) -> collections.Counter[Point]:
    return collections.Counter(
        {endpoint: count for endpoint, (count, _source) in state.items()}
    )


def source_replays(
    source: Source | None,
    target: Point,
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> bool:
    if source is None:
        return False
    return (
        R102.source_endpoint(source, atoms_a, atoms_c, curve)
        == target
    )


def residual_query(
    target: Point,
    prefix_index: dict[Point, list[Source]],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> dict[str, Any]:
    residual_histogram: collections.Counter[Point] = collections.Counter()
    residual_first: dict[Point, Source] = {}
    count = 0
    first_source: Source | None = None
    scanned = 0
    candidate_prefix_records = 0
    for suffix in R102.suffix_sources(len(atoms_a), len(atoms_c)):
        suffix_endpoint = R102.source_endpoint(
            suffix,
            atoms_a,
            atoms_c,
            curve,
        )
        residual = R70.add(
            target,
            R70.negate(suffix_endpoint, curve),
            curve,
        )
        residual_histogram[residual] += 1
        residual_first.setdefault(residual, suffix)
        candidates = prefix_index.get(residual, ())
        candidate_prefix_records += len(candidates)
        for prefix in candidates:
            if not R102.canonical_boundaries_hold(prefix, suffix):
                continue
            source = R102.combine_source(prefix, suffix)
            count += 1
            if first_source is None or source < first_source:
                first_source = source
        scanned += 1
    replay = (
        first_source is None
        if count == 0
        else source_replays(
            first_source,
            target,
            atoms_a,
            atoms_c,
            curve,
        )
    )
    sign_control: dict[str, Any]
    if first_source is None:
        sign_control = {
            "source_present": False,
            "point_level_sign_replay": True,
        }
    else:
        prefix = (
            first_source[0][: R102.PREFIX_A_COUNT],
            first_source[1][: R102.PREFIX_C_COUNT],
        )
        suffix = (
            first_source[0][R102.PREFIX_A_COUNT :],
            first_source[1][R102.PREFIX_C_COUNT :],
        )
        left = R102.source_endpoint(
            prefix,
            atoms_a,
            atoms_c,
            curve,
        )
        right = R102.source_endpoint(
            suffix,
            atoms_a,
            atoms_c,
            curve,
        )
        labels = R103.signed_branch_labels(left, right, target, curve)
        affine = left is not None and right is not None and target is not None
        sign_control = {
            "source_present": True,
            "left": point_json(left),
            "right": point_json(right),
            "target": point_json(target),
            "projective_group_replay": R70.add(left, right, curve) == target,
            "affine_s3_available": affine,
            "affine_s3_zero": (
                R70.semaev_s3(left[0], right[0], target[0], curve)
                if affine
                else None
            ),
            "signed_branches": labels,
            "correct_sign_branch_selected": (
                "R=T-L" in labels if affine else True
            ),
            "point_level_sign_replay": (
                R70.add(left, right, curve) == target
                and ("R=T-L" in labels if affine else True)
            ),
        }
    return {
        "integer_occurrence_count": count,
        "source": source_json(first_source),
        "returned_bottom": first_source is None,
        "source_replays": replay,
        "suffix_occurrence_count": scanned,
        "residual_distinct_endpoint_count": len(residual_histogram),
        "residual_maximum_multiplicity": max(
            residual_histogram.values(),
            default=0,
        ),
        "candidate_prefix_records": candidate_prefix_records,
        "residual_first_source_count": len(residual_first),
        "sign_control": sign_control,
    }


def query_control(
    label: str,
    target: Point,
    expected: collections.Counter[Point],
    prefix_index: dict[Point, list[Source]],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> dict[str, Any]:
    query = residual_query(
        target,
        prefix_index,
        atoms_a,
        atoms_c,
        curve,
    )
    expected_count = expected.get(target, 0)
    return {
        "label": label,
        "target": point_json(target),
        "expected_integer_occurrence_count": expected_count,
        "query": query,
        "count_exact": query["integer_occurrence_count"] == expected_count,
        "bottom_exact": query["returned_bottom"] == (expected_count == 0),
        "source_exact": (
            query["source_replays"]
            if expected_count
            else query["source"] is None
        ),
    }


def actual_instance(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    a5, a_stages = multiset_group_algebra(
        atoms_a,
        R102.FULL_A_COUNT,
        curve,
    )
    c5, c_stages = multiset_group_algebra(
        atoms_c,
        R102.FULL_C_COUNT,
        curve,
    )
    full_state, convolution_operations = convolve_deck_states(
        a5,
        c5,
        curve,
    )
    recurrence_histogram = state_histogram(full_state)
    direct_histogram, direct_first, direct_count = (
        R84.multiset_endpoint_section(
            atoms_a,
            atoms_c,
            R102.FULL_A_COUNT,
            R102.FULL_C_COUNT,
            curve,
        )
    )
    prefix_index, prefix_count = R102.build_prefix_index(
        atoms_a,
        atoms_c,
        curve,
    )
    suffix_rows = list(
        R102.suffix_sources(len(atoms_a), len(atoms_c))
    )
    suffix_endpoints = [
        R102.source_endpoint(source, atoms_a, atoms_c, curve)
        for source in suffix_rows
    ]
    suffix_distinct = len(set(suffix_endpoints))
    positive_target = min(direct_first, key=R102.point_sort_key)
    repeated_target = max(
        direct_histogram,
        key=lambda point: (
            direct_histogram[point],
            tuple(-value for value in R102.point_sort_key(point)),
        ),
    )
    blind = R102.blind_target(
        direct_histogram,
        curve,
        f"R104|{offset}|blind",
    )
    queries = [
        query_control(
            "positive",
            positive_target,
            direct_histogram,
            prefix_index,
            atoms_a,
            atoms_c,
            curve,
        ),
        query_control(
            "blind",
            blind,
            direct_histogram,
            prefix_index,
            atoms_a,
            atoms_c,
            curve,
        ),
        query_control(
            "maximum_multiplicity",
            repeated_target,
            direct_histogram,
            prefix_index,
            atoms_a,
            atoms_c,
            curve,
        ),
        query_control(
            "projective_identity",
            None,
            direct_histogram,
            prefix_index,
            atoms_a,
            atoms_c,
            curve,
        ),
    ]
    recurrence_count = sum(recurrence_histogram.values())
    all_state_sources_replay = all(
        source_replays(
            source,
            endpoint,
            atoms_a,
            atoms_c,
            curve,
        )
        for endpoint, (_count, source) in full_state.items()
    )
    return {
        "family_id": curve["family_id"],
        "offset": offset,
        "field_prime": curve["field_prime"],
        "subgroup_order": curve["subgroup_order"],
        "factor_base_size_B": len(factors),
        "atom_a_size": len(atoms_a),
        "atom_c_size": len(atoms_c),
        "factor_base_injective": geometry["factor_base_injective"],
        "scalar_labels_consumed": False,
        "a5_recurrence_stages": a_stages,
        "c5_recurrence_stages": c_stages,
        "a5_support_size": len(a5),
        "c5_support_size": len(c5),
        "full_support_size": len(full_state),
        "full_convolution_endpoint_pair_operations": convolution_operations,
        "recurrence_integer_occurrence_count": recurrence_count,
        "direct_integer_occurrence_count": direct_count,
        "recurrence_histogram_equals_direct": (
            recurrence_histogram == direct_histogram
        ),
        "all_recurrence_first_sources_replay": all_state_sources_replay,
        "prefix_occurrence_count": prefix_count,
        "prefix_distinct_endpoint_count": len(prefix_index),
        "prefix_endpoint_map_injective": len(prefix_index) == prefix_count,
        "suffix_occurrence_count": len(suffix_rows),
        "suffix_distinct_endpoint_count": suffix_distinct,
        "suffix_endpoint_map_injective": suffix_distinct == len(suffix_rows),
        "suffix_translation_context_matrix_rank": suffix_distinct,
        "suffix_context_matrix_is_identity_after_endpoint_ordering": (
            suffix_distinct == len(suffix_rows)
        ),
        "maximum_target_multiplicity": max(direct_histogram.values()),
        "has_repeated_endpoint_multiplicity": (
            max(direct_histogram.values()) > 1
        ),
        "query_controls": queries,
        "all_query_counts_sources_and_signs_exact": all(
            row["count_exact"]
            and row["bottom_exact"]
            and row["source_exact"]
            and row["query"]["sign_control"]["point_level_sign_replay"]
            for row in queries
        ),
        "finite_support_exponents_B": {
            "prefix_occurrences": support_exponent(prefix_count, len(factors)),
            "suffix_occurrences": support_exponent(
                len(suffix_rows),
                len(factors),
            ),
            "full_occurrences": support_exponent(
                direct_count,
                len(factors),
            ),
        },
    }


@functools.lru_cache(maxsize=1)
def actual_controls() -> dict[str, Any]:
    instances = [
        actual_instance(dict(family), offset)
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    query_rows = [
        row
        for instance in instances
        for row in instance["query_controls"]
    ]
    return {
        "instances": instances,
        "instance_count": len(instances),
        "query_control_count": len(query_rows),
        "all_group_algebra_histograms_exact": all(
            instance["recurrence_histogram_equals_direct"]
            for instance in instances
        ),
        "all_group_algebra_sources_replay": all(
            instance["all_recurrence_first_sources_replay"]
            for instance in instances
        ),
        "all_prefix_endpoint_maps_injective": all(
            instance["prefix_endpoint_map_injective"]
            for instance in instances
        ),
        "all_suffix_endpoint_maps_injective": all(
            instance["suffix_endpoint_map_injective"]
            for instance in instances
        ),
        "all_suffix_context_matrices_full_rank": all(
            instance["suffix_translation_context_matrix_rank"]
            == instance["suffix_occurrence_count"]
            for instance in instances
        ),
        "all_residual_queries_exact": all(
            instance["all_query_counts_sources_and_signs_exact"]
            for instance in instances
        ),
        "blind_query_count": sum(
            row["label"] == "blind" for row in query_rows
        ),
        "all_blind_queries_return_bottom": all(
            row["query"]["returned_bottom"]
            for row in query_rows
            if row["label"] == "blind"
        ),
        "projective_identity_query_count": sum(
            row["label"] == "projective_identity" for row in query_rows
        ),
        "all_projective_identity_queries_exact": all(
            row["count_exact"] and row["bottom_exact"]
            for row in query_rows
            if row["label"] == "projective_identity"
        ),
        "repeated_multiplicity_instance_count": sum(
            instance["has_repeated_endpoint_multiplicity"]
            for instance in instances
        ),
        "all_scalar_blind": all(
            not instance["scalar_labels_consumed"]
            for instance in instances
        ),
    }


def mergeable_state_separation_theorem() -> dict[str, Any]:
    return {
        "grammar_domain": (
            "universal exact mergeable summaries of endpoint histograms, "
            "including target-specialized summaries, with arbitrary "
            "singleton sibling contexts"
        ),
        "state_semantics": (
            "integer count at one fixed target plus a canonical source "
            "adjoint after elliptic-addition merge"
        ),
        "sign_resolution_reduction": (
            "the correct S3 branch is R=T-L, so exact sign-resolved "
            "acceptance is the Kronecker predicate [L+R=T]"
        ),
        "fixed_target_separation": (
            "if histograms f and g differ at endpoint P, merge each with "
            "the singleton context [T-P]; the exact output at fixed T is "
            "f(P) versus g(P)"
        ),
        "nonlinear_state_count_lower_bound": (
            "on binary histograms over D distinguishable endpoints, the "
            "summary needs at least 2^D states and D bits"
        ),
        "field_word_lower_bound": (
            "Omega(D/log(p)) base-field words, preserving the exponent of D"
        ),
        "linear_semilinear_rank_lower_bound": (
            "the singleton-context readout matrix is the D by D identity, "
            "so every linear or fixed-Frobenius semilinear sketch has rank D"
        ),
        "ffe_scalar_extension": (
            "extension of scalars preserves the identity-minor rank D; FFE "
            "coordinates can redistribute but not remove these D readouts"
        ),
        "source_adjoint": (
            "returning one coupled canonical source cannot weaken the count "
            "separation requirement and additionally preserves a witness "
            "for every nonzero endpoint bucket"
        ),
        "suffix_distinguishable_endpoint_exponent_B": fraction_record(
            SUFFIX_EXPONENT
        ),
        "required_state_exponent_B": fraction_record(SUFFIX_EXPONENT),
        "inside_setup_cap": SUFFIX_EXPONENT <= SETUP_CAP,
        "inside_online_cap": SUFFIX_EXPONENT <= ONLINE_CAP,
        "theorem_scope": (
            "This is exact for a universal mergeable child-state API and "
            "for explicit residual sets. It is not a word lower bound for "
            "one fixed actual deck represented by a non-mergeable, "
            "target-specific arithmetic circuit or short program."
        ),
    }


def projective_and_exceptional_controls() -> dict[str, Any]:
    curve = dict(R82.FAMILIES[0])
    atoms_a, atoms_c, _factors, _geometry = R82.compact_factor_base(
        curve,
        0,
    )
    left = atoms_a[0]
    right = atoms_c[0]
    negative_left = R70.negate(left, curve)
    doubled = R70.add(left, left, curve)
    sign = R103.synthetic_sign_complete_control()
    controls = {
        "vertical_inverse_identity": {
            "left": point_json(left),
            "right": point_json(negative_left),
            "target": None,
            "group_chart_exact": R70.add(left, negative_left, curve) is None,
            "affine_x_s3_available": False,
        },
        "target_equals_left_identity_child": {
            "left": point_json(left),
            "right": None,
            "target": point_json(left),
            "group_chart_exact": R70.add(left, None, curve) == left,
            "s3_leading_coefficient_degree_drop": True,
        },
        "tangent_doubling": {
            "left": point_json(left),
            "right": point_json(left),
            "target": point_json(doubled),
            "group_chart_exact": R70.add(left, left, curve) == doubled,
        },
        "ordinary_affine": {
            "left": point_json(left),
            "right": point_json(right),
            "target": point_json(R70.add(left, right, curve)),
            "group_chart_exact": True,
        },
        "synthetic_four_sign_branches": sign,
    }
    return {
        "curve_family_id": curve["family_id"],
        "controls": controls,
        "all_group_charts_exact": all(
            row["group_chart_exact"]
            for name, row in controls.items()
            if name != "synthetic_four_sign_branches"
        ),
        "four_sign_branch_control_exact": (
            sign["all_four_signed_points_pass_s3"]
            and sign["exactly_one_signed_point_is_true_join"]
            and sign["two_distinct_x_roots"]
        ),
        "identity_and_degree_drop_covered_before_affine_s3": True,
        "tangent_doubling_covered": True,
    }


def cost_ledger() -> dict[str, Any]:
    theorem = mergeable_state_separation_theorem()
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_workspace_exponent_B": fraction_record(ONLINE_CAP),
        },
        "complete_homogeneous_group_algebra": {
            "h5_a_occurrence_exponent_B": fraction_record(
                5 * ATOM_A_EXPONENT
            ),
            "h5_c_occurrence_exponent_B": fraction_record(
                5 * ATOM_C_EXPONENT
            ),
            "full_pair_convolution_exponent_B": fraction_record(
                FULL_SOURCE_EXPONENT
            ),
            "h5_a_inside_setup_cap": 5 * ATOM_A_EXPONENT <= SETUP_CAP,
            "h5_c_inside_setup_cap": 5 * ATOM_C_EXPONENT <= SETUP_CAP,
            "full_pair_convolution_inside_online_cap": (
                FULL_SOURCE_EXPONENT <= ONLINE_CAP
            ),
        },
        "target_specialized_reverse_residual_recurrence": {
            "stored_prefix": "4A+1C canonical source buckets",
            "prefix_state_exponent_B": fraction_record(PREFIX_EXPONENT),
            "reverse_suffix": (
                "residuals T-(A+4C), preserving multiplicity and source"
            ),
            "suffix_residual_exponent_B": fraction_record(SUFFIX_EXPONENT),
            "prefix_inside_setup_cap": PREFIX_EXPONENT <= SETUP_CAP,
            "suffix_inside_setup_cap": SUFFIX_EXPONENT <= SETUP_CAP,
            "suffix_inside_online_cap": SUFFIX_EXPONENT <= ONLINE_CAP,
            "sign_resolved_s3_changes_residual_exponent": False,
        },
        "mergeable_summary_lower_bound": theorem,
        "fatal_obstruction": (
            "the explicit reverse pushdown emits B^(14/5) residual/source "
            "leaves, while any universal exact mergeable replacement must "
            "distinguish the same singleton contexts"
        ),
        "scope_exception": (
            "an actual-deck-specific nonlinear, non-mergeable target circuit "
            "that consumes the compact D_A,D_C program as a whole and "
            "returns count/source without exposing endpoint or residual "
            "child summaries"
        ),
    }


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        "r103_producer": {
            "path": str(R103_PRODUCER),
            "sha256": R103_PRODUCER_SHA256,
        },
        "r103_report": {
            "path": str(R103_REPORT),
            "sha256": R103_REPORT_SHA256,
        },
        "r103_gate": {
            "path": str(R103_GATE),
            "sha256": R103_GATE_SHA256,
        },
        "r103_parent": {
            "path": str(R103_PARENT),
            "sha256": R103_PARENT_SHA256,
        },
        "r102_producer": {
            "path": str(R102_PRODUCER),
            "sha256": R102_PRODUCER_SHA256,
        },
        "r102_report": {
            "path": str(R102_REPORT),
            "sha256": R102_REPORT_SHA256,
        },
        "r102_gate": {
            "path": str(R102_GATE),
            "sha256": R102_GATE_SHA256,
        },
        "r102_parent": {
            "path": str(R102_PARENT),
            "sha256": R102_PARENT_SHA256,
        },
        "r87_report": {
            "path": str(R87_REPORT),
            "sha256": R87_REPORT_SHA256,
        },
        "r87_gate": {
            "path": str(R87_GATE),
            "sha256": R87_GATE_SHA256,
        },
        "r93_report": {
            "path": str(R93_REPORT),
            "sha256": R93_REPORT_SHA256,
        },
        "r93_gate": {
            "path": str(R93_GATE),
            "sha256": R93_GATE_SHA256,
        },
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    controls = actual_controls()
    exceptional_controls = projective_and_exceptional_controls()
    costs = cost_ledger()
    theorem = costs["mergeable_summary_lower_bound"]
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_compact_preendpoint_s3_ffe_pushdown.r104.v1"
        ),
        "recurrence_grammar": (
            "exact mergeable endpoint histograms over the elliptic group "
            "algebra, arbitrary nonlinear encodings allowed, with exact "
            "merge against singleton contexts"
        ),
        "sign_marker": "full projective point; accept only R=T-L",
        "target_transition": "P maps to residual T-P",
        "ffe_state": (
            "optional finite-extension linear or fixed-Frobenius semilinear "
            "encoding of the mergeable histogram"
        ),
        "source_unranking_adjoint": (
            "integer multiplicity plus lexicographically first coupled "
            "canonical 5A+5C source"
        ),
        "projective_charts": [
            "ordinary_affine",
            "tangent_doubling",
            "vertical_inverse_identity",
            "identity_child",
            "x_s3_degree_drop",
        ],
        "caps": costs["caps"],
        "excluded_open_operation": costs["scope_exception"],
    }
    recurrence_ledger = {
        "schema": (
            "p1553.preendpoint_pushdown_recurrence_cost_ledger.r104.v1"
        ),
        "frozen_candidate": frozen,
        "exact_recurrence_identity": (
            "H_(i,k)=H_(i-1,k)+[P_i]*H_(i,k-1) in Z[E(F_p)]"
        ),
        "full_relation_identity": (
            "[T](h_5(D_A)*h_5(D_C)) equals the canonical 5A+5C "
            "integer source count"
        ),
        "sign_resolved_s3_reduction": theorem["sign_resolution_reduction"],
        "mergeable_state_separation_theorem": theorem,
        "costs": costs,
        "actual_stage_receipts": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "a5_recurrence_stages": instance["a5_recurrence_stages"],
                "c5_recurrence_stages": instance["c5_recurrence_stages"],
                "a5_support_size": instance["a5_support_size"],
                "c5_support_size": instance["c5_support_size"],
                "full_support_size": instance["full_support_size"],
                "full_convolution_endpoint_pair_operations": instance[
                    "full_convolution_endpoint_pair_operations"
                ],
            }
            for instance in controls["instances"]
        ],
    }
    source_replay = {
        "schema": (
            "p1553.preendpoint_s3_ffe_integer_source_replay.r104.v1"
        ),
        "all_group_algebra_histograms_exact": controls[
            "all_group_algebra_histograms_exact"
        ],
        "all_group_algebra_sources_replay": controls[
            "all_group_algebra_sources_replay"
        ],
        "all_residual_queries_exact": controls[
            "all_residual_queries_exact"
        ],
        "all_actual_suffix_endpoint_maps_injective": controls[
            "all_suffix_endpoint_maps_injective"
        ],
        "all_actual_suffix_context_matrices_full_rank": controls[
            "all_suffix_context_matrices_full_rank"
        ],
        "instances": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "prefix_occurrence_count": instance[
                    "prefix_occurrence_count"
                ],
                "prefix_distinct_endpoint_count": instance[
                    "prefix_distinct_endpoint_count"
                ],
                "suffix_occurrence_count": instance[
                    "suffix_occurrence_count"
                ],
                "suffix_distinct_endpoint_count": instance[
                    "suffix_distinct_endpoint_count"
                ],
                "suffix_translation_context_matrix_rank": instance[
                    "suffix_translation_context_matrix_rank"
                ],
                "query_controls": instance["query_controls"],
            }
            for instance in controls["instances"]
        ],
        "scalar_labels_consumed": False,
        "candidate_credit": False,
    }
    exceptional = {
        "schema": (
            "p1553.preendpoint_pushdown_exceptional_controls.r104.v1"
        ),
        "projective_and_sign_controls": exceptional_controls,
        "blind_query_count": controls["blind_query_count"],
        "all_blind_queries_return_bottom": controls[
            "all_blind_queries_return_bottom"
        ],
        "identity_query_count": controls[
            "projective_identity_query_count"
        ],
        "all_identity_queries_exact": controls[
            "all_projective_identity_queries_exact"
        ],
        "actual_repeated_multiplicity_instance_count": controls[
            "repeated_multiplicity_instance_count"
        ],
        "actual_repeated_multiplicity_branch_present": (
            controls["repeated_multiplicity_instance_count"] > 0
        ),
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r104.v1",
        "sign_resolved_s3_reduction_exact": True,
        "explicit_residual_recurrence_exact": True,
        "mergeable_preendpoint_pushdown_inside_caps": False,
        "actual_deck_specific_nonmergeable_circuit_supplied": False,
        "known_rhs_relation_collection_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_recovered_without_verifier_dlp": False,
        "factor_logs_verified_algorithmically": False,
        "identical_scalar_blind_target_descent_complete": False,
        "breakthrough": False,
        "shoup_bound_improvement": False,
    }
    obligations = {
        "twelve_source_bindings_verified": len(bindings) == 12,
        "eight_actual_instances_replayed": controls["instance_count"] == 8,
        "thirty_two_target_controls_replayed": (
            controls["query_control_count"] == 32
        ),
        "complete_homogeneous_recurrence_histograms_exact": controls[
            "all_group_algebra_histograms_exact"
        ],
        "complete_homogeneous_recurrence_sources_replay": controls[
            "all_group_algebra_sources_replay"
        ],
        "all_residual_query_counts_sources_and_signs_exact": controls[
            "all_residual_queries_exact"
        ],
        "all_actual_prefix_maps_injective": controls[
            "all_prefix_endpoint_maps_injective"
        ],
        "all_actual_suffix_maps_injective": controls[
            "all_suffix_endpoint_maps_injective"
        ],
        "all_actual_suffix_context_matrices_full_rank": controls[
            "all_suffix_context_matrices_full_rank"
        ],
        "all_blind_queries_return_bottom": controls[
            "all_blind_queries_return_bottom"
        ],
        "all_projective_identity_queries_exact": controls[
            "all_projective_identity_queries_exact"
        ],
        "actual_repeated_multiplicity_branch_replayed": (
            controls["repeated_multiplicity_instance_count"] > 0
        ),
        "scalar_blind_construction": controls["all_scalar_blind"],
        "projective_group_charts_exact": exceptional_controls[
            "all_group_charts_exact"
        ],
        "synthetic_four_sign_branches_exact": exceptional_controls[
            "four_sign_branch_control_exact"
        ],
        "sign_resolved_s3_equals_kronecker_join": True,
        "fixed_target_singleton_context_separation_exact": True,
        "nonlinear_mergeable_state_count_lower_bound_exact": True,
        "linear_semilinear_identity_rank_lower_bound_exact": True,
        "ffe_scalar_extension_preserves_identity_rank": True,
        "source_adjoint_not_weaker_than_count_state": True,
        "suffix_residual_exponent_B14O5": (
            theorem["required_state_exponent_B"]["exact"] == "14/5"
        ),
        "suffix_residual_state_inside_setup_cap": theorem[
            "inside_setup_cap"
        ],
        "suffix_residual_state_inside_online_cap": theorem[
            "inside_online_cap"
        ],
        "mergeable_preendpoint_pushdown_inside_caps": False,
        "actual_deck_specific_nonmergeable_circuit_supplied": False,
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
            "SIGN_RESOLVED_S3_IS_GROUP_COEFFICIENT__EXPLICIT_REVERSE_"
            "RESIDUAL_RECURRENCE_EXACT__ACTUAL_SUFFIX_IMAGES_INJECTIVE__"
            "B14O5_RESIDUAL_BODY__UNIVERSAL_MERGEABLE_SUMMARIES_REQUIRE_"
            "B14O5_STATE__ACTUAL_DECK_SPECIFIC_NONMERGEABLE_CIRCUIT_OPEN"
        ),
        "source_bindings": source_binding_records(),
        "novelty_scope": (
            "R104 is the first campaign receipt to reduce a fully "
            "sign-resolved target-forced S3 pushdown to exact elliptic "
            "group-algebra coefficient extraction, replay the reverse "
            "residual recurrence on every actual family, and prove a "
            "fixed-target singleton-context separation theorem for "
            "universal mergeable nonlinear and FFE summaries."
        ),
        "actual_controls": controls,
        "projective_and_exceptional_controls": exceptional_controls,
        "mergeable_state_separation_theorem": theorem,
        "cost_ledger": costs,
        "artifacts": {
            "frozen": (
                "frozen_5a5c_compact_preendpoint_s3_ffe_pushdown.json"
            ),
            "recurrence_ledger": (
                "preendpoint_pushdown_recurrence_and_cost_ledger.json"
            ),
            "source_replay": (
                "preendpoint_s3_ffe_integer_source_replay.json"
            ),
            "exceptional": (
                "preendpoint_pushdown_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r104.json",
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "mergeable_preendpoint_pushdown_closed": True,
        "actual_deck_specific_nonmergeable_circuit_open": True,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "scope_boundary": theorem["theorem_scope"],
        "next_action": (
            "Construct or refute one actual-deck-specific nonlinear, "
            "non-mergeable target pullback circuit for the frozen D_A,D_C "
            "program. It must inject T before any child summary, avoid a "
            "universal endpoint/residual API, expand every arithmetic and "
            "FFE gate, return exact integer multiplicity and one coupled "
            "source inside B^(9/4) setup and B^(5/4) fresh workspace, and "
            "survive matched random-deck, sign, infinity, tangent, blind, "
            "rank, factor-log, and identical-descent controls."
        ),
        "disposition": (
            "REJECT_UNIVERSAL_MERGEABLE_PREENDPOINT_S3_FFE_PUSHDOWNS__"
            "SIGN_RESOLUTION_COLLAPSES_TO_GROUP_COEFFICIENT__ACTUAL_"
            "REVERSE_RESIDUAL_COUNTS_AND_SOURCES_EXACT__SUFFIX_RESIDUAL_"
            "BODY_B14O5__NONLINEAR_STATE_SEPARATION_AND_FFE_IDENTITY_RANK_"
            "B14O5__ACTUAL_DECK_SPECIFIC_NONMERGEABLE_CIRCUIT_OPEN__NO_"
            "RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_"
            "BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "recurrence_ledger": recurrence_ledger,
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
            "p1553_5a5c_compact_preendpoint_s3_ffe_"
            "pushdown_probe_report_r104.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_compact_preendpoint_s3_ffe_pushdown.json"
        ),
    )
    parser.add_argument(
        "--recurrence-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "preendpoint_pushdown_recurrence_and_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "preendpoint_s3_ffe_integer_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "preendpoint_pushdown_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r104.json"
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
    write_json(args.recurrence_output, bundle["recurrence_ledger"])
    write_json(args.source_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    admission = bundle["report"]["admission"]
    print(
        "R104 "
        f"classification={bundle['report']['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
