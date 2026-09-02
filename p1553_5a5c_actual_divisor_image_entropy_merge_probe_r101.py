#!/usr/bin/env python3
"""Construct exact local 2A+3C and 3A+2C source oracles after R100."""

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


SCHEMA = "p1553.5a5c_actual_divisor_image_entropy_merge.r101.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
ATOM_INPUT_EXPONENT = Fraction(3, 5)
RIGHT_STATE_EXPONENT = Fraction(6, 5)
RIGHT_QUERY_EXPONENT = Fraction(6, 5)
LEFT_STATE_EXPONENT = Fraction(9, 5)
LEFT_QUERY_EXPONENT = Fraction(4, 5)

R100_PRODUCER = pathlib.Path(
    "p1553_5a5c_succinct_aggregate_digit_trie_probe_r100.py"
)
R100_PRODUCER_SHA256 = (
    "168f462c50aa89be7e513d21fe9f34f6080bf04922c192306997805029b88bd9"
)
R100_REPORT = pathlib.Path(
    "p1553_5a5c_succinct_aggregate_digit_"
    "trie_probe_report_r100.json"
)
R100_REPORT_SHA256 = (
    "fc7281ba084824963da550dfb8f034baa4f6d133305e18af11684e2be61154dd"
)
R100_GATE = pathlib.Path(
    "p1553_5a5c_succinct_aggregate_digit_trie_probe_gate_r100.md"
)
R100_GATE_SHA256 = (
    "653ad0f51c9ec6f88a89769ca757f45ae70432cb947148d447eda968286a2e50"
)
R99_REPORT = pathlib.Path(
    "p1553_5a5c_multiedge_digitized_equality_"
    "projector_probe_report_r99.json"
)
R99_REPORT_SHA256 = (
    "dac6fbf38357e640bb174860df16117fdd3461717cb3b2ceac4e76ec7f77707c"
)
R99_GATE = pathlib.Path(
    "p1553_5a5c_multiedge_digitized_equality_projector_probe_gate_r99.md"
)
R99_GATE_SHA256 = (
    "a7e69ad7cf661f09c679039c5ed9874940eca5432d7ca91dc980f97005bfe560"
)
R84_PRODUCER = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_probe_r84.py"
)
R84_PRODUCER_SHA256 = (
    "80cd0887fd24bfc37f0568f03a8af4c98eb148e030a3112576c1a0940f4ad99d"
)
R84_REPORT = pathlib.Path(
    "p1553_5a5c_marked_resultant_source_section_"
    "probe_report_r84.json"
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
R82_PRODUCER = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_r82.py"
)
R82_PRODUCER_SHA256 = (
    "7380bff3175625016affee4703b0b0f2867a28113f72eef2d90614ed57ffef07"
)
R82_REPORT = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_report_r82.json"
)
R82_REPORT_SHA256 = (
    "ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832"
)
R82_GATE = pathlib.Path(
    "p1553_cartesian_sum_compact_divisor_probe_gate_r82.md"
)
R82_GATE_SHA256 = (
    "7c34e1d905c95a756689d4ec0ea92c6bd47808bcb3407d858ce08cccf75fd55e"
)
P1513_HANDOFF = pathlib.Path(
    "/Volumes/Volume/autolab/research/"
    "p1513_idea121_direct_ku_handoff_v3_20260717.md"
)
P1513_HANDOFF_SHA256 = (
    "27c8f1f15fd0c3b81ebe2008aa96db12417c3f6612c5c151212206dcba388dcc"
)

Point = tuple[int, int] | None
Source = tuple[tuple[int, ...], tuple[int, ...]]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R100_PRODUCER: R100_PRODUCER_SHA256,
        R100_REPORT: R100_REPORT_SHA256,
        R100_GATE: R100_GATE_SHA256,
        R99_REPORT: R99_REPORT_SHA256,
        R99_GATE: R99_GATE_SHA256,
        R84_PRODUCER: R84_PRODUCER_SHA256,
        R84_REPORT: R84_REPORT_SHA256,
        R84_GATE: R84_GATE_SHA256,
        R82_PRODUCER: R82_PRODUCER_SHA256,
        R82_REPORT: R82_REPORT_SHA256,
        R82_GATE: R82_GATE_SHA256,
        P1513_HANDOFF: P1513_HANDOFF_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R101 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R82 = load_module("p1553_r82_for_r101", R82_PRODUCER)
R84 = load_module("p1553_r84_for_r101", R84_PRODUCER)


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
        "decimal": float(value),
    }


def multiset_endpoint_map(
    points: Sequence[Point],
    arity: int,
    curve: dict[str, Any],
) -> tuple[
    collections.Counter[Point],
    dict[Point, tuple[int, ...]],
    int,
]:
    histogram: collections.Counter[Point] = collections.Counter()
    first: dict[Point, tuple[int, ...]] = {}
    occurrence_count = 0
    for indices in itertools.combinations_with_replacement(
        range(len(points)),
        arity,
    ):
        endpoint = R82.add_many(
            (points[index] for index in indices),
            curve,
        )
        histogram[endpoint] += 1
        first.setdefault(endpoint, indices)
        occurrence_count += 1
    expected = math.comb(len(points) + arity - 1, arity)
    if occurrence_count != expected or sum(histogram.values()) != expected:
        raise AssertionError("multiset endpoint count drifted")
    return histogram, first, occurrence_count


def query_side_index(
    target: Point,
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    count_a: int,
    count_c: int,
    c_histogram: collections.Counter[Point],
    c_first: dict[Point, tuple[int, ...]],
    curve: dict[str, Any],
) -> dict[str, Any]:
    count = 0
    first_source: Source | None = None
    scanned_a_multisets = 0
    for a_indices in itertools.combinations_with_replacement(
        range(len(atoms_a)),
        count_a,
    ):
        a_endpoint = R82.add_many(
            (atoms_a[index] for index in a_indices),
            curve,
        )
        complement = R82.R70.add(
            target,
            R82.R70.negate(a_endpoint, curve),
            curve,
        )
        multiplicity = c_histogram.get(complement, 0)
        count += multiplicity
        if multiplicity and first_source is None:
            first_source = (a_indices, c_first[complement])
        scanned_a_multisets += 1
    source_replays = (
        first_source is None
        if count == 0
        else R84.source_endpoint(
            first_source,
            atoms_a,
            atoms_c,
            curve,
        )
        == target
    )
    return {
        "integer_occurrence_count": count,
        "source": (
            None
            if first_source is None
            else [list(first_source[0]), list(first_source[1])]
        ),
        "returned_bottom": first_source is None,
        "source_replays": source_replays,
        "scanned_a_multisets": scanned_a_multisets,
        "expected_scan_count": math.comb(
            len(atoms_a) + count_a - 1,
            count_a,
        ),
    }


def blind_target(
    support: Iterable[Point],
    curve: dict[str, Any],
    salt: str,
) -> Point:
    support_set = set(support)
    return next(
        point
        for point in R82.hash_point_candidates(curve, salt)
        if point not in support_set
    )


def analyze_side(
    side_name: str,
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    count_a: int,
    count_c: int,
    curve: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    c_histogram, c_first, c_occurrences = multiset_endpoint_map(
        atoms_c,
        count_c,
        curve,
    )
    direct_histogram, direct_first, direct_occurrences = (
        R84.multiset_endpoint_section(
            atoms_a,
            atoms_c,
            count_a,
            count_c,
            curve,
        )
    )
    all_counts_exact = True
    all_sources_replay = True
    all_scan_counts_exact = True
    for target, expected_count in direct_histogram.items():
        query = query_side_index(
            target,
            atoms_a,
            atoms_c,
            count_a,
            count_c,
            c_histogram,
            c_first,
            curve,
        )
        all_counts_exact &= (
            query["integer_occurrence_count"] == expected_count
        )
        all_sources_replay &= query["source_replays"]
        all_scan_counts_exact &= (
            query["scanned_a_multisets"]
            == query["expected_scan_count"]
        )
    blind = blind_target(
        direct_histogram,
        curve,
        f"R101|{side_name}|{offset}|blind",
    )
    blind_query = query_side_index(
        blind,
        atoms_a,
        atoms_c,
        count_a,
        count_c,
        c_histogram,
        c_first,
        curve,
    )
    identity_query = query_side_index(
        None,
        atoms_a,
        atoms_c,
        count_a,
        count_c,
        c_histogram,
        c_first,
        curve,
    )
    repeated_source: Source = (
        tuple([0] * count_a),
        tuple([0] * count_c),
    )
    repeated_target = R84.source_endpoint(
        repeated_source,
        atoms_a,
        atoms_c,
        curve,
    )
    repeated_query = query_side_index(
        repeated_target,
        atoms_a,
        atoms_c,
        count_a,
        count_c,
        c_histogram,
        c_first,
        curve,
    )
    return {
        "side": side_name,
        "source_arity": {
            "atom_a_count": count_a,
            "atom_c_count": count_c,
        },
        "input_atom_count": len(atoms_a) + len(atoms_c),
        "stored_c_multiset_occurrences": c_occurrences,
        "stored_c_distinct_endpoints": len(c_histogram),
        "scanned_a_multisets_per_query": math.comb(
            len(atoms_a) + count_a - 1,
            count_a,
        ),
        "direct_side_occurrence_count": direct_occurrences,
        "direct_side_distinct_endpoint_count": len(direct_histogram),
        "direct_source_map_count": len(direct_first),
        "all_attained_target_counts_exact": all_counts_exact,
        "all_attained_target_sources_replay": all_sources_replay,
        "all_query_scan_counts_exact": all_scan_counts_exact,
        "blind_target_query": blind_query,
        "blind_target_exact_bottom": (
            blind_query["integer_occurrence_count"] == 0
            and blind_query["returned_bottom"]
        ),
        "identity_target_expected_count": direct_histogram.get(None, 0),
        "identity_target_query": identity_query,
        "identity_target_exact": (
            identity_query["integer_occurrence_count"]
            == direct_histogram.get(None, 0)
        ),
        "repeated_atom_target_query": repeated_query,
        "repeated_atom_target_positive_and_replays": (
            repeated_query["integer_occurrence_count"] >= 1
            and repeated_query["source_replays"]
        ),
        "scalar_labels_consumed": False,
        "endpoint_or_source_leaves_persisted": False,
        "persistent_summary": (
            "weighted count and one source for each count_c*C endpoint"
        ),
    }


def cyclic_multiset_control() -> dict[str, Any]:
    modulus = 11
    atoms_a = [1, 1, 4]
    atoms_c = [2, 5, 9]
    count_a = 3
    count_c = 2
    c_histogram: collections.Counter[int] = collections.Counter()
    c_first: dict[int, tuple[int, ...]] = {}
    for indices in itertools.combinations_with_replacement(
        range(len(atoms_c)),
        count_c,
    ):
        endpoint = sum(atoms_c[index] for index in indices) % modulus
        c_histogram[endpoint] += 1
        c_first.setdefault(endpoint, indices)
    direct: collections.Counter[int] = collections.Counter()
    direct_first: dict[int, Source] = {}
    for a_indices in itertools.combinations_with_replacement(
        range(len(atoms_a)),
        count_a,
    ):
        for c_indices in itertools.combinations_with_replacement(
            range(len(atoms_c)),
            count_c,
        ):
            endpoint = (
                sum(atoms_a[index] for index in a_indices)
                + sum(atoms_c[index] for index in c_indices)
            ) % modulus
            direct[endpoint] += 1
            direct_first.setdefault(endpoint, (a_indices, c_indices))
    queries = {}
    for target in range(modulus):
        count = 0
        source: Source | None = None
        for a_indices in itertools.combinations_with_replacement(
            range(len(atoms_a)),
            count_a,
        ):
            a_endpoint = (
                sum(atoms_a[index] for index in a_indices) % modulus
            )
            complement = (target - a_endpoint) % modulus
            count += c_histogram.get(complement, 0)
            if source is None and complement in c_first:
                source = (a_indices, c_first[complement])
        queries[target] = {
            "count": count,
            "expected": direct.get(target, 0),
            "source_replays": (
                source is None
                if not count
                else (
                    sum(atoms_a[index] for index in source[0])
                    + sum(atoms_c[index] for index in source[1])
                )
                % modulus
                == target
            ),
        }
    maximum = max(direct.values())
    return {
        "group": "Z/11Z",
        "duplicate_atom_values_present": len(set(atoms_a)) < len(atoms_a),
        "all_target_counts_exact": all(
            row["count"] == row["expected"] for row in queries.values()
        ),
        "all_positive_sources_replay": all(
            row["source_replays"] for row in queries.values()
        ),
        "maximum_endpoint_occurrence_multiplicity": maximum,
        "multiplicity_above_one": maximum > 1,
        "identity_target_count": direct.get(0, 0),
        "identity_target_positive": direct.get(0, 0) > 0,
    }


@functools.lru_cache(maxsize=1)
def actual_image_controls() -> dict[str, Any]:
    instances = []
    for family in R82.FAMILIES:
        for offset in R82.INSTANCE_OFFSETS:
            curve = dict(family)
            atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
                curve,
                offset,
            )
            left = analyze_side(
                "2A+3C",
                atoms_a,
                atoms_c,
                2,
                3,
                curve,
                offset,
            )
            right = analyze_side(
                "3A+2C",
                atoms_a,
                atoms_c,
                3,
                2,
                curve,
                offset,
            )
            instances.append(
                {
                    "family_id": curve["family_id"],
                    "offset": offset,
                    "field_prime": curve["field_prime"],
                    "factor_base_size_B": len(factors),
                    "atom_a_size": len(atoms_a),
                    "atom_c_size": len(atoms_c),
                    "factor_base_injective": geometry[
                        "factor_base_injective"
                    ],
                    "left": left,
                    "right": right,
                }
            )
    cyclic = cyclic_multiset_control()
    sides = [
        instance[side]
        for instance in instances
        for side in ("left", "right")
    ]
    return {
        "instances": instances,
        "instance_count": len(instances),
        "side_query_count": len(sides),
        "all_attained_counts_exact": all(
            side["all_attained_target_counts_exact"] for side in sides
        ),
        "all_attained_sources_replay": all(
            side["all_attained_target_sources_replay"] for side in sides
        ),
        "all_blind_targets_return_bottom": all(
            side["blind_target_exact_bottom"] for side in sides
        ),
        "all_identity_target_queries_exact": all(
            side["identity_target_exact"] for side in sides
        ),
        "all_repeated_atom_targets_replay": all(
            side["repeated_atom_target_positive_and_replays"]
            for side in sides
        ),
        "all_scalar_blind": all(
            not side["scalar_labels_consumed"] for side in sides
        ),
        "cyclic_multiplicity_control": cyclic,
    }


def asymptotic_cost_control() -> dict[str, Any]:
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_exponent_B": fraction_record(ONLINE_CAP),
        },
        "public_atom_inputs": {
            "state_exponent_B": fraction_record(ATOM_INPUT_EXPONENT),
            "inside_setup_cap": ATOM_INPUT_EXPONENT <= SETUP_CAP,
        },
        "right_3A_plus_2C": {
            "stored_summary": "weighted 2C endpoint dictionary",
            "stored_state_exponent_B": fraction_record(
                RIGHT_STATE_EXPONENT
            ),
            "fresh_query": "stream all unordered 3A multisets",
            "fresh_work_exponent_B": fraction_record(
                RIGHT_QUERY_EXPONENT
            ),
            "inside_setup_cap": RIGHT_STATE_EXPONENT <= SETUP_CAP,
            "inside_online_cap": RIGHT_QUERY_EXPONENT <= ONLINE_CAP,
        },
        "left_2A_plus_3C": {
            "stored_summary": "weighted 3C endpoint dictionary",
            "stored_state_exponent_B": fraction_record(
                LEFT_STATE_EXPONENT
            ),
            "fresh_query": "stream all unordered 2A multisets",
            "fresh_work_exponent_B": fraction_record(
                LEFT_QUERY_EXPONENT
            ),
            "inside_setup_cap": LEFT_STATE_EXPONENT <= SETUP_CAP,
            "inside_online_cap": LEFT_QUERY_EXPONENT <= ONLINE_CAP,
        },
        "full_two_sided_join": {
            "problem": (
                "find ell in 2A+3C with T-ell in 3A+2C"
            ),
            "supplied": False,
            "enumerating_left_endpoints_exponent_B": fraction_record(
                Fraction(13, 5)
            ),
            "enumerating_right_endpoints_exponent_B": fraction_record(
                Fraction(12, 5)
            ),
            "inside_online_cap": False,
        },
        "scope": (
            "Constructs exact local membership/count/source oracles for both "
            "actual R84 side images from compact atom decks. It does not "
            "solve the implicit two-sided intersection required for a fresh "
            "5A+5C target."
        ),
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    controls = actual_image_controls()
    costs = asymptotic_cost_control()
    cyclic = controls["cyclic_multiplicity_control"]
    frozen = {
        "schema": (
            "p1553.frozen_5a5c_actual_divisor_image_entropy_merge.r101.v1"
        ),
        "public_inputs": "D_A=sum[A_i], D_C=sum[C_j]",
        "right_summary": "weighted endpoint dictionary for 2C",
        "right_query": "stream unordered 3A and complement-lookup",
        "left_summary": "weighted endpoint dictionary for 3C",
        "left_query": "stream unordered 2A and complement-lookup",
        "caps": costs["caps"],
        "excluded_unsolved_operation": (
            "implicit intersection of 2A+3C with T-(3A+2C)"
        ),
    }
    entropy_ledger = {
        "schema": (
            "p1553.actual_divisor_image_parameter_entropy_ledger.r101.v1"
        ),
        "actual_image_controls": controls,
        "asymptotic_cost": costs,
        "local_side_oracles_inside_caps": True,
        "full_two_sided_join_inside_caps": False,
    }
    source_replay = {
        "schema": (
            "p1553.actual_divisor_leaf_free_merge_source_replay.r101.v1"
        ),
        "actual_instances": controls["instances"],
        "all_attained_counts_exact": controls[
            "all_attained_counts_exact"
        ],
        "all_attained_sources_replay": controls[
            "all_attained_sources_replay"
        ],
        "full_five_a_five_c_joint_source_complete": False,
        "candidate_credit": "local_side_oracles_only",
    }
    exceptional = {
        "schema": (
            "p1553.actual_divisor_image_exceptional_controls.r101.v1"
        ),
        "blind_bottom_complete": controls[
            "all_blind_targets_return_bottom"
        ],
        "identity_target_complete": controls[
            "all_identity_target_queries_exact"
        ],
        "repeated_atom_tangent_paths_complete": controls[
            "all_repeated_atom_targets_replay"
        ],
        "synthetic_collision_multiplicity_complete": (
            cyclic["all_target_counts_exact"]
            and cyclic["multiplicity_above_one"]
        ),
        "proper_subsum_complete": False,
        "full_join_multiplicity_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r101.v1",
        "local_side_oracles_inside_caps": True,
        "full_two_sided_join_inside_caps": False,
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
        "eight_actual_instances_replayed": controls[
            "instance_count"
        ]
        == 8,
        "both_sides_per_instance_replayed": controls[
            "side_query_count"
        ]
        == 16,
        "all_attained_target_counts_exact": controls[
            "all_attained_counts_exact"
        ],
        "all_attained_target_sources_replay": controls[
            "all_attained_sources_replay"
        ],
        "all_blind_targets_return_bottom": controls[
            "all_blind_targets_return_bottom"
        ],
        "all_identity_target_queries_exact": controls[
            "all_identity_target_queries_exact"
        ],
        "all_repeated_atom_tangent_paths_replay": controls[
            "all_repeated_atom_targets_replay"
        ],
        "synthetic_collision_multiplicity_exact": exceptional[
            "synthetic_collision_multiplicity_complete"
        ],
        "public_atom_input_exponent_B3O5": (
            costs["public_atom_inputs"]["state_exponent_B"]["exact"]
            == "3/5"
        ),
        "right_state_exponent_B6O5": (
            costs["right_3A_plus_2C"][
                "stored_state_exponent_B"
            ]["exact"]
            == "6/5"
        ),
        "right_query_exponent_B6O5": (
            costs["right_3A_plus_2C"][
                "fresh_work_exponent_B"
            ]["exact"]
            == "6/5"
        ),
        "left_state_exponent_B9O5": (
            costs["left_2A_plus_3C"][
                "stored_state_exponent_B"
            ]["exact"]
            == "9/5"
        ),
        "left_query_exponent_B4O5": (
            costs["left_2A_plus_3C"][
                "fresh_work_exponent_B"
            ]["exact"]
            == "4/5"
        ),
        "right_oracle_inside_both_caps": (
            costs["right_3A_plus_2C"]["inside_setup_cap"]
            and costs["right_3A_plus_2C"]["inside_online_cap"]
        ),
        "left_oracle_inside_both_caps": (
            costs["left_2A_plus_3C"]["inside_setup_cap"]
            and costs["left_2A_plus_3C"]["inside_online_cap"]
        ),
        "scalar_blind_local_construction": controls[
            "all_scalar_blind"
        ],
        "actual_image_short_generator_supplied": True,
        "local_leaf_free_merge_law_supplied": True,
        "full_two_sided_join_inside_caps": False,
        "full_five_a_five_c_integer_count_complete": False,
        "full_five_a_five_c_joint_source_complete": False,
        "proper_subsum_complete": False,
        "full_join_multiplicity_complete": False,
        "known_rhs_rank_without_verifier_dlp": False,
        "factor_logs_without_verifier_dlp": False,
        "identical_fresh_target_descent": False,
        "generic_prime_family_algorithm": False,
        "shoup_improvement_complete": False,
        "full_pipeline_fresh_workspace_inside_cap": False,
        "all_exceptional_branches_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, passed in obligations.items() if not passed]
    report = {
        "schema": SCHEMA,
        "classification": (
            "ACTUAL_3A2C_AND_2A3C_LOCAL_COUNT_SOURCE_ORACLES_PASS_CAPS__"
            "WEIGHTED_C_SUMMARIES_LEAF_FREE__FULL_TWO_SIDED_JOIN_OPEN"
        ),
        "source_bindings": {
            "r100_producer": {
                "path": str(R100_PRODUCER),
                "sha256": R100_PRODUCER_SHA256,
            },
            "r100_report": {
                "path": str(R100_REPORT),
                "sha256": R100_REPORT_SHA256,
            },
            "r100_gate": {
                "path": str(R100_GATE),
                "sha256": R100_GATE_SHA256,
            },
            "r99_report": {
                "path": str(R99_REPORT),
                "sha256": R99_REPORT_SHA256,
            },
            "r99_gate": {
                "path": str(R99_GATE),
                "sha256": R99_GATE_SHA256,
            },
            "r84_producer": {
                "path": str(R84_PRODUCER),
                "sha256": R84_PRODUCER_SHA256,
            },
            "r84_report": {
                "path": str(R84_REPORT),
                "sha256": R84_REPORT_SHA256,
            },
            "r84_gate": {
                "path": str(R84_GATE),
                "sha256": R84_GATE_SHA256,
            },
            "r82_producer": {
                "path": str(R82_PRODUCER),
                "sha256": R82_PRODUCER_SHA256,
            },
            "r82_report": {
                "path": str(R82_REPORT),
                "sha256": R82_REPORT_SHA256,
            },
            "r82_gate": {
                "path": str(R82_GATE),
                "sha256": R82_GATE_SHA256,
            },
            "p1513_handoff": {
                "path": str(P1513_HANDOFF),
                "sha256": P1513_HANDOFF_SHA256,
            },
        },
        "novelty_scope": (
            "R101 is the first campaign receipt to construct exact local "
            "count-and-source oracles for both actual R84 side images from "
            "weighted lower-arity C summaries, without side endpoint leaves."
        ),
        "actual_image_controls": controls,
        "asymptotic_cost_control": costs,
        "side_artifacts": {
            "frozen": (
                "frozen_5a5c_actual_divisor_image_entropy_merge.json"
            ),
            "entropy_ledger": (
                "actual_divisor_image_parameter_entropy_ledger.json"
            ),
            "source_replay": (
                "actual_divisor_leaf_free_merge_source_replay.json"
            ),
            "exceptional": (
                "actual_divisor_image_exceptional_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r101.json",
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "local_side_oracles_admitted": True,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "scope_boundary": (
            "R101 is a real local positive for the structured R84 images, "
            "not an entropy obstruction. It constructs cap-sized exact "
            "membership/count/source oracles for each side separately. It "
            "does not find an intersection between 2A+3C and T-(3A+2C) "
            "without enumerating an over-cap side."
        ),
        "next_action": (
            "Construct or refute one two-sided implicit intersection "
            "algorithm combining the passing 2A+3C and 3A+2C local oracles. "
            "Freeze its shared state and target transition before outcomes; "
            "require B^(9/4) setup/state, B^(5/4) fresh work/workspace, exact "
            "5A+5C count and one jointly coupled source, all exceptional "
            "branches, known-RHS rank, factor logs, and identical descent, "
            "without enumerating either B^(13/5) or B^(12/5) side."
        ),
        "disposition": (
            "ACCEPT_LOCAL_ACTUAL_DIVISOR_IMAGE_ORACLES_ONLY__WEIGHTED_2C_"
            "STATE_B6O5_AND_3A_QUERY_B6O5__WEIGHTED_3C_STATE_B9O5_AND_2A_"
            "QUERY_B4O5__EIGHT_ACTUAL_INSTANCES__EXACT_COUNTS_SOURCES_BLIND_"
            "IDENTITY_REPEATED_ATOM_AND_SYNTHETIC_MULTIPLICITY_CONTROLS__"
            "FULL_TWO_SIDED_IMPLICIT_JOIN_OPEN__NO_KNOWN_RHS_RANK__NO_"
            "FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "entropy_ledger": entropy_ledger,
        "source_replay": source_replay,
        "exceptional": exceptional,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_5a5c_actual_divisor_image_entropy_"
            "merge_probe_report_r101.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_actual_divisor_image_entropy_merge.json"
        ),
    )
    parser.add_argument(
        "--entropy-ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "actual_divisor_image_parameter_entropy_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "actual_divisor_leaf_free_merge_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "actual_divisor_image_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r101.json"
        ),
    )
    return parser.parse_args()


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.entropy_ledger_output, bundle["entropy_ledger"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
