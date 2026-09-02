#!/usr/bin/env python3
"""Test a canonical two-sided join and density-one filtering after R101."""

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


SCHEMA = "p1553.5a5c_two_sided_implicit_join.r102.v1"
SETUP_CAP = Fraction(9, 4)
ONLINE_CAP = Fraction(5, 4)
ATOM_A_EXPONENT = Fraction(2, 5)
ATOM_C_EXPONENT = Fraction(3, 5)
GROUP_EXPONENT = Fraction(5, 1)
PREFIX_A_COUNT = 4
PREFIX_C_COUNT = 1
SUFFIX_A_COUNT = 1
SUFFIX_C_COUNT = 4
FULL_A_COUNT = 5
FULL_C_COUNT = 5
PREFIX_STATE_EXPONENT = Fraction(11, 5)
SUFFIX_QUERY_EXPONENT = Fraction(14, 5)
FILTER_DEFICIT_EXPONENT = SUFFIX_QUERY_EXPONENT - ONLINE_CAP
PARTITION_MODULI = (2, 4, 8, 16)
POSITIVE_QUERY_SAMPLE_COUNT = 16

R101_PRODUCER = pathlib.Path(
    "p1553_5a5c_actual_divisor_image_entropy_merge_probe_r101.py"
)
R101_PRODUCER_SHA256 = (
    "854df1da576b07cc3dd63b1249a42a72d56708ef8eb43c4ef37478f2558e747d"
)
R101_REPORT = pathlib.Path(
    "p1553_5a5c_actual_divisor_image_entropy_"
    "merge_probe_report_r101.json"
)
R101_REPORT_SHA256 = (
    "050d1093fc34611f502a3f7b3ebf6fc632b3aeb9d03718f922e99baa2fe985c9"
)
R101_GATE = pathlib.Path(
    "p1553_5a5c_actual_divisor_image_entropy_merge_probe_gate_r101.md"
)
R101_GATE_SHA256 = (
    "3e12e4ab70edba64076d27b68d39ab34307ca6d9ff08252432759673b74a9ac5"
)
R101_PARENT = pathlib.Path(
    "p1553_5a5c_actual_divisor_image_entropy_"
    "merge_probe_parent_report_r101.yaml"
)
R101_PARENT_SHA256 = (
    "14beac87b86d3664fd439c82663e444aa379035d292c60873523f0d6c9c67d6d"
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
P1513_HANDOFF = pathlib.Path(
    "/Volumes/Volume/autolab/research/"
    "p1513_idea121_direct_ku_handoff_v3_20260717.md"
)
P1513_HANDOFF_SHA256 = (
    "27c8f1f15fd0c3b81ebe2008aa96db12417c3f6612c5c151212206dcba388dcc"
)

Point = tuple[int, int] | None
Source = tuple[tuple[int, ...], tuple[int, ...]]
PrefixIndex = dict[Point, list[Source]]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_bindings() -> dict[str, str]:
    expected = {
        R101_PRODUCER: R101_PRODUCER_SHA256,
        R101_REPORT: R101_REPORT_SHA256,
        R101_GATE: R101_GATE_SHA256,
        R101_PARENT: R101_PARENT_SHA256,
        R100_REPORT: R100_REPORT_SHA256,
        R100_GATE: R100_GATE_SHA256,
        R84_PRODUCER: R84_PRODUCER_SHA256,
        R84_REPORT: R84_REPORT_SHA256,
        R84_GATE: R84_GATE_SHA256,
        R82_PRODUCER: R82_PRODUCER_SHA256,
        R82_REPORT: R82_REPORT_SHA256,
        P1513_HANDOFF: P1513_HANDOFF_SHA256,
    }
    actual = {str(path): sha256_file(path) for path in expected}
    failures = [
        str(path)
        for path, expected_hash in expected.items()
        if actual[str(path)] != expected_hash
    ]
    if failures:
        raise AssertionError(f"R102 source binding mismatch: {failures}")
    return actual


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R101 = load_module("p1553_r101_for_r102", R101_PRODUCER)
R82 = R101.R82
R84 = R101.R84


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
        "decimal": float(value),
    }


def source_json(source: Source | None) -> list[list[int]] | None:
    if source is None:
        return None
    return [list(source[0]), list(source[1])]


def source_endpoint(
    source: Source,
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> Point:
    return R84.source_endpoint(source, atoms_a, atoms_c, curve)


def full_sources(size_a: int, size_c: int) -> Iterator[Source]:
    for indices_a in itertools.combinations_with_replacement(
        range(size_a),
        FULL_A_COUNT,
    ):
        for indices_c in itertools.combinations_with_replacement(
            range(size_c),
            FULL_C_COUNT,
        ):
            yield indices_a, indices_c


def build_prefix_index(
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> tuple[PrefixIndex, int]:
    index: PrefixIndex = collections.defaultdict(list)
    record_count = 0
    for indices_a in itertools.combinations_with_replacement(
        range(len(atoms_a)),
        PREFIX_A_COUNT,
    ):
        for indices_c in itertools.combinations_with_replacement(
            range(len(atoms_c)),
            PREFIX_C_COUNT,
        ):
            source = (indices_a, indices_c)
            index[source_endpoint(source, atoms_a, atoms_c, curve)].append(
                source
            )
            record_count += 1
    expected = (
        math.comb(len(atoms_a) + PREFIX_A_COUNT - 1, PREFIX_A_COUNT)
        * math.comb(len(atoms_c) + PREFIX_C_COUNT - 1, PREFIX_C_COUNT)
    )
    if record_count != expected:
        raise AssertionError("canonical prefix state count drifted")
    return dict(index), record_count


def suffix_sources(size_a: int, size_c: int) -> Iterator[Source]:
    for indices_a in itertools.combinations_with_replacement(
        range(size_a),
        SUFFIX_A_COUNT,
    ):
        for indices_c in itertools.combinations_with_replacement(
            range(size_c),
            SUFFIX_C_COUNT,
        ):
            yield indices_a, indices_c


def canonical_boundaries_hold(prefix: Source, suffix: Source) -> bool:
    return (
        prefix[0][-1] <= suffix[0][0]
        and prefix[1][-1] <= suffix[1][0]
    )


def combine_source(prefix: Source, suffix: Source) -> Source:
    if not canonical_boundaries_hold(prefix, suffix):
        raise AssertionError("noncanonical prefix/suffix combination")
    return prefix[0] + suffix[0], prefix[1] + suffix[1]


def canonical_query(
    target: Point,
    index: PrefixIndex,
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> dict[str, Any]:
    count = 0
    first_source: Source | None = None
    scanned_suffixes = 0
    candidate_prefix_records = 0
    for suffix in suffix_sources(len(atoms_a), len(atoms_c)):
        suffix_endpoint = source_endpoint(
            suffix,
            atoms_a,
            atoms_c,
            curve,
        )
        complement = R82.R70.add(
            target,
            R82.R70.negate(suffix_endpoint, curve),
            curve,
        )
        candidates = index.get(complement, ())
        candidate_prefix_records += len(candidates)
        for prefix in candidates:
            if not canonical_boundaries_hold(prefix, suffix):
                continue
            source = combine_source(prefix, suffix)
            count += 1
            if first_source is None:
                first_source = source
        scanned_suffixes += 1
    expected_scans = (
        math.comb(len(atoms_a) + SUFFIX_A_COUNT - 1, SUFFIX_A_COUNT)
        * math.comb(len(atoms_c) + SUFFIX_C_COUNT - 1, SUFFIX_C_COUNT)
    )
    source_replays = (
        first_source is None
        if count == 0
        else source_endpoint(first_source, atoms_a, atoms_c, curve) == target
    )
    return {
        "integer_occurrence_count": count,
        "source": source_json(first_source),
        "returned_bottom": first_source is None,
        "source_replays": source_replays,
        "scanned_suffixes": scanned_suffixes,
        "expected_suffix_scans": expected_scans,
        "candidate_prefix_records": candidate_prefix_records,
    }


def canonical_all_target_audit(
    index: PrefixIndex,
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
) -> tuple[collections.Counter[Point], dict[Point, Source], int]:
    histogram: collections.Counter[Point] = collections.Counter()
    first: dict[Point, Source] = {}
    source_count = 0
    for suffix in suffix_sources(len(atoms_a), len(atoms_c)):
        suffix_endpoint = source_endpoint(
            suffix,
            atoms_a,
            atoms_c,
            curve,
        )
        for prefix_sources in index.values():
            for prefix in prefix_sources:
                if not canonical_boundaries_hold(prefix, suffix):
                    continue
                source = combine_source(prefix, suffix)
                target = R82.R70.add(
                    source_endpoint(prefix, atoms_a, atoms_c, curve),
                    suffix_endpoint,
                    curve,
                )
                histogram[target] += 1
                first.setdefault(target, source)
                source_count += 1
    expected = (
        math.comb(len(atoms_a) + FULL_A_COUNT - 1, FULL_A_COUNT)
        * math.comb(len(atoms_c) + FULL_C_COUNT - 1, FULL_C_COUNT)
    )
    if source_count != expected:
        raise AssertionError("canonical full-source count drifted")
    return histogram, first, source_count


def source_partition_class(
    family_id: str,
    offset: int,
    source: Source,
) -> int:
    encoded = (
        f"R102|{family_id}|{offset}|"
        f"{','.join(map(str, source[0]))}|"
        f"{','.join(map(str, source[1]))}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def partition_controls(
    direct_first: dict[Point, Source],
    direct_histogram: collections.Counter[Point],
    atoms_a: Sequence[Point],
    atoms_c: Sequence[Point],
    curve: dict[str, Any],
    offset: int,
) -> list[dict[str, Any]]:
    class_counts = {
        modulus: [0] * modulus for modulus in PARTITION_MODULI
    }
    target_masks = {
        modulus: collections.defaultdict(int) for modulus in PARTITION_MODULI
    }
    incidence_count = 0
    for source in full_sources(len(atoms_a), len(atoms_c)):
        target = source_endpoint(source, atoms_a, atoms_c, curve)
        digest = source_partition_class(curve["family_id"], offset, source)
        for modulus in PARTITION_MODULI:
            label = digest % modulus
            class_counts[modulus][label] += 1
            target_masks[modulus][target] |= 1 << label
        incidence_count += 1
    if incidence_count != sum(direct_histogram.values()):
        raise AssertionError("partition incidence count drifted")
    positive_target_count = len(direct_first)
    rows = []
    for modulus in PARTITION_MODULI:
        counts = class_counts[modulus]
        masks = target_masks[modulus]
        coverage = [
            sum(bool(mask & (1 << label)) for mask in masks.values())
            for label in range(modulus)
        ]
        rows.append(
            {
                "partition_modulus": modulus,
                "source_incidence_count": incidence_count,
                "class_source_incidence_counts": counts,
                "class_source_incidence_sum_exact": (
                    sum(counts) == incidence_count
                ),
                "class_positive_target_coverage_counts": coverage,
                "maximum_single_class_source_weighted_fraction": (
                    max(counts) / incidence_count
                ),
                "maximum_single_class_positive_target_coverage_fraction": (
                    max(coverage) / positive_target_count
                ),
                "all_positive_targets_covered_by_class_union": (
                    set(masks) == set(direct_first)
                    and all(mask != 0 for mask in masks.values())
                ),
                "all_classes_required_for_partition_guarantee": modulus,
                "ideal_per_class_work_reduction_factor": modulus,
                "guaranteed_repetition_work_product": modulus / modulus,
            }
        )
    return rows


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


def point_sort_key(point: Point) -> tuple[int, int, int]:
    return (-1, 0, 0) if point is None else (0, point[0], point[1])


def analyze_instance(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    atoms_a, atoms_c, factors, geometry = R82.compact_factor_base(
        curve,
        offset,
    )
    index, prefix_count = build_prefix_index(atoms_a, atoms_c, curve)
    canonical_histogram, canonical_first, canonical_count = (
        canonical_all_target_audit(index, atoms_a, atoms_c, curve)
    )
    direct_histogram, direct_first, direct_count = (
        R84.multiset_endpoint_section(
            atoms_a,
            atoms_c,
            FULL_A_COUNT,
            FULL_C_COUNT,
            curve,
        )
    )
    histograms_exact = canonical_histogram == direct_histogram
    source_keys_exact = set(canonical_first) == set(direct_first)
    all_first_sources_replay = all(
        source_endpoint(source, atoms_a, atoms_c, curve) == target
        for target, source in canonical_first.items()
    )
    positive_targets = sorted(direct_first, key=point_sort_key)
    sample_targets = positive_targets[:POSITIVE_QUERY_SAMPLE_COUNT]
    repeated_source: Source = (
        tuple([0] * FULL_A_COUNT),
        tuple([0] * FULL_C_COUNT),
    )
    repeated_target = source_endpoint(
        repeated_source,
        atoms_a,
        atoms_c,
        curve,
    )
    for target in (repeated_target, None):
        if target not in sample_targets:
            sample_targets.append(target)
    query_receipts = []
    for target in sample_targets:
        query = canonical_query(
            target,
            index,
            atoms_a,
            atoms_c,
            curve,
        )
        query_receipts.append(
            {
                "target": R82.point_json(target),
                "expected_count": direct_histogram.get(target, 0),
                "query": query,
                "count_exact": (
                    query["integer_occurrence_count"]
                    == direct_histogram.get(target, 0)
                ),
            }
        )
    blind = blind_target(
        direct_histogram,
        curve,
        f"R102|{offset}|blind",
    )
    blind_query = canonical_query(
        blind,
        index,
        atoms_a,
        atoms_c,
        curve,
    )
    partitions = partition_controls(
        direct_first,
        direct_histogram,
        atoms_a,
        atoms_c,
        curve,
        offset,
    )
    ordered_source_count = len(factors) ** FULL_A_COUNT
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
        "prefix_index_record_count": prefix_count,
        "prefix_index_distinct_endpoint_count": len(index),
        "suffix_scan_count_per_query": (
            math.comb(
                len(atoms_a) + SUFFIX_A_COUNT - 1,
                SUFFIX_A_COUNT,
            )
            * math.comb(
                len(atoms_c) + SUFFIX_C_COUNT - 1,
                SUFFIX_C_COUNT,
            )
        ),
        "canonical_full_source_count": canonical_count,
        "direct_full_source_count": direct_count,
        "canonical_histogram_equals_direct": histograms_exact,
        "canonical_source_keys_equal_direct": source_keys_exact,
        "all_canonical_first_sources_replay": all_first_sources_replay,
        "positive_target_count": len(direct_first),
        "maximum_target_multiplicity": max(direct_histogram.values()),
        "all_sampled_queries_exact": all(
            row["count_exact"] and row["query"]["source_replays"]
            for row in query_receipts
        ),
        "sample_query_receipts": query_receipts,
        "blind_query": blind_query,
        "blind_query_exact_bottom": (
            blind_query["integer_occurrence_count"] == 0
            and blind_query["returned_bottom"]
        ),
        "identity_expected_count": direct_histogram.get(None, 0),
        "identity_query_exact": all(
            row["count_exact"]
            for row in query_receipts
            if row["target"] is None
        ),
        "repeated_target_expected_count": direct_histogram[repeated_target],
        "repeated_target_query_exact": all(
            row["count_exact"] and row["query"]["source_replays"]
            for row in query_receipts
            if row["target"] == R82.point_json(repeated_target)
        ),
        "ordered_factor_source_count": ordered_source_count,
        "ordered_source_density_vs_subgroup": (
            ordered_source_count / curve["subgroup_order"]
        ),
        "partition_controls": partitions,
    }


def cyclic_collision_control() -> dict[str, Any]:
    modulus = 11
    atoms_a = (1, 1, 4)
    atoms_c = (2, 5, 9)

    def endpoint(source: Source) -> int:
        return (
            sum(atoms_a[index] for index in source[0])
            + sum(atoms_c[index] for index in source[1])
        ) % modulus

    prefix_index: dict[int, list[Source]] = collections.defaultdict(list)
    for indices_a in itertools.combinations_with_replacement(
        range(len(atoms_a)),
        PREFIX_A_COUNT,
    ):
        for indices_c in itertools.combinations_with_replacement(
            range(len(atoms_c)),
            PREFIX_C_COUNT,
        ):
            source = (indices_a, indices_c)
            prefix_index[endpoint(source)].append(source)
    direct: collections.Counter[int] = collections.Counter()
    for source in full_sources(len(atoms_a), len(atoms_c)):
        direct[endpoint(source)] += 1
    queries = {}
    for target in range(modulus):
        count = 0
        first: Source | None = None
        for suffix in suffix_sources(len(atoms_a), len(atoms_c)):
            complement = (target - endpoint(suffix)) % modulus
            for prefix in prefix_index.get(complement, ()):
                if not canonical_boundaries_hold(prefix, suffix):
                    continue
                count += 1
                if first is None:
                    first = combine_source(prefix, suffix)
        queries[target] = {
            "count": count,
            "expected": direct[target],
            "source_replays": first is not None and endpoint(first) == target,
        }
    return {
        "group": "Z/11Z",
        "duplicate_atom_values_present": len(set(atoms_a)) < len(atoms_a),
        "all_target_counts_exact": all(
            row["count"] == row["expected"] for row in queries.values()
        ),
        "all_positive_sources_replay": all(
            row["source_replays"] for row in queries.values()
        ),
        "identity_target_count": direct[0],
        "identity_target_positive": direct[0] > 0,
        "maximum_target_multiplicity": max(direct.values()),
        "multiplicity_above_one": max(direct.values()) > 1,
    }


def split_exponent_ledger() -> dict[str, Any]:
    rows = []
    for stored_a in range(FULL_A_COUNT + 1):
        for stored_c in range(FULL_C_COUNT + 1):
            if (stored_a, stored_c) in (
                (0, 0),
                (FULL_A_COUNT, FULL_C_COUNT),
            ):
                continue
            state = (
                stored_a * ATOM_A_EXPONENT
                + stored_c * ATOM_C_EXPONENT
            )
            query = GROUP_EXPONENT - state
            rows.append(
                {
                    "stored_a_count": stored_a,
                    "stored_c_count": stored_c,
                    "state_exponent_B": fraction_record(state),
                    "complement_query_exponent_B": fraction_record(query),
                    "inside_setup_cap": state <= SETUP_CAP,
                    "inside_online_cap": query <= ONLINE_CAP,
                }
            )
    eligible = [row for row in rows if row["inside_setup_cap"]]
    best_query = min(
        Fraction(row["complement_query_exponent_B"]["exact"])
        for row in eligible
    )
    best = [
        row
        for row in eligible
        if Fraction(row["complement_query_exponent_B"]["exact"])
        == best_query
    ]
    return {
        "all_nontrivial_atom_count_splits": rows,
        "eligible_split_count": len(eligible),
        "best_setup_eligible_splits": best,
        "best_state_exponent_B": fraction_record(
            GROUP_EXPONENT - best_query
        ),
        "best_query_exponent_B": fraction_record(best_query),
        "best_query_inside_online_cap": best_query <= ONLINE_CAP,
    }


def asymptotic_control() -> dict[str, Any]:
    split = split_exponent_ledger()
    return {
        "caps": {
            "setup_state_exponent_B": fraction_record(SETUP_CAP),
            "fresh_work_workspace_exponent_B": fraction_record(ONLINE_CAP),
        },
        "canonical_prefix_join": {
            "stored_side": "4A+1C canonical prefixes",
            "queried_side": "1A+4C canonical suffixes",
            "state_exponent_B": fraction_record(PREFIX_STATE_EXPONENT),
            "query_exponent_B": fraction_record(SUFFIX_QUERY_EXPONENT),
            "inside_setup_cap": PREFIX_STATE_EXPONENT <= SETUP_CAP,
            "inside_online_cap": SUFFIX_QUERY_EXPONENT <= ONLINE_CAP,
            "exact_integer_count_and_source": True,
        },
        "exhaustive_split_ledger": split,
        "representation_density": {
            "ordered_factor_source_exponent_B": fraction_record(
                GROUP_EXPONENT
            ),
            "subgroup_order_exponent_B": fraction_record(GROUP_EXPONENT),
            "representation_surplus_exponent_B": fraction_record(
                Fraction(0)
            ),
            "permutation_and_pairing_slack": (
                "bounded arity supplies constants only, not B^delta"
            ),
        },
        "disjoint_filter_conservation": {
            "required_ideal_pruning_exponent_B": fraction_record(
                FILTER_DEFICIT_EXPONENT
            ),
            "retained_representation_exponent_B": fraction_record(
                -FILTER_DEFICIT_EXPONENT
            ),
            "constant_success_repetition_exponent_B": fraction_record(
                FILTER_DEFICIT_EXPONENT
            ),
            "restored_total_work_exponent_B": fraction_record(
                ONLINE_CAP + FILTER_DEFICIT_EXPONENT
            ),
            "identity": (
                "5/4 + 31/20 = 14/5; ideal partition pruning is paid "
                "back by the classes or repetitions needed at density one"
            ),
            "scope": (
                "Applies to extra disjoint or independent filters that thin "
                "the true source set. It does not cover a target-forced "
                "algebraic identity satisfied by every true join."
            ),
        },
    }


@functools.lru_cache(maxsize=1)
def actual_controls() -> dict[str, Any]:
    instances = [
        analyze_instance(dict(family), offset)
        for family in R82.FAMILIES
        for offset in R82.INSTANCE_OFFSETS
    ]
    partitions = [
        row
        for instance in instances
        for row in instance["partition_controls"]
    ]
    cyclic = cyclic_collision_control()
    return {
        "instances": instances,
        "instance_count": len(instances),
        "all_canonical_histograms_exact": all(
            instance["canonical_histogram_equals_direct"]
            for instance in instances
        ),
        "all_canonical_source_keys_exact": all(
            instance["canonical_source_keys_equal_direct"]
            for instance in instances
        ),
        "all_first_sources_replay": all(
            instance["all_canonical_first_sources_replay"]
            for instance in instances
        ),
        "all_sampled_queries_exact": all(
            instance["all_sampled_queries_exact"]
            for instance in instances
        ),
        "all_blind_queries_bottom": all(
            instance["blind_query_exact_bottom"]
            for instance in instances
        ),
        "all_identity_queries_exact": all(
            instance["identity_query_exact"] for instance in instances
        ),
        "all_repeated_targets_exact": all(
            instance["repeated_target_query_exact"]
            for instance in instances
        ),
        "all_scalar_blind": all(
            not instance["scalar_labels_consumed"]
            for instance in instances
        ),
        "all_partition_incidence_sums_exact": all(
            row["class_source_incidence_sum_exact"] for row in partitions
        ),
        "all_partition_unions_complete": all(
            row["all_positive_targets_covered_by_class_union"]
            for row in partitions
        ),
        "cyclic_collision_control": cyclic,
    }


@functools.lru_cache(maxsize=1)
def build_bundle() -> dict[str, dict[str, Any]]:
    bindings = verify_source_bindings()
    controls = actual_controls()
    costs = asymptotic_control()
    split = costs["exhaustive_split_ledger"]
    cyclic = controls["cyclic_collision_control"]
    frozen = {
        "schema": "p1553.frozen_5a5c_two_sided_implicit_join.r102.v1",
        "public_inputs": "D_A=sum[A_i], D_C=sum[C_j], fresh target T",
        "canonical_state": "4A+1C prefix endpoint dictionary with sources",
        "canonical_query": (
            "stream 1A+4C suffixes, complement-lookup, enforce sorted "
            "boundary indices"
        ),
        "partition_filters": list(PARTITION_MODULI),
        "caps": costs["caps"],
        "excluded_open_operation": (
            "target-forced algebraic join filter that keeps every true "
            "source while avoiding B^(14/5) suffix enumeration"
        ),
    }
    state_ledger = {
        "schema": (
            "p1553.two_sided_join_state_and_transition_ledger.r102.v1"
        ),
        "asymptotic_control": costs,
        "actual_state_and_query_counts": [
            {
                "family_id": instance["family_id"],
                "offset": instance["offset"],
                "prefix_index_record_count": instance[
                    "prefix_index_record_count"
                ],
                "suffix_scan_count_per_query": instance[
                    "suffix_scan_count_per_query"
                ],
                "ordered_source_density_vs_subgroup": instance[
                    "ordered_source_density_vs_subgroup"
                ],
            }
            for instance in controls["instances"]
        ],
        "best_split_query_inside_online_cap": split[
            "best_query_inside_online_cap"
        ],
    }
    source_replay = {
        "schema": "p1553.two_sided_join_integer_source_replay.r102.v1",
        "instances": controls["instances"],
        "all_target_histograms_exact": controls[
            "all_canonical_histograms_exact"
        ],
        "all_first_sources_replay": controls["all_first_sources_replay"],
        "all_sampled_fresh_queries_exact": controls[
            "all_sampled_queries_exact"
        ],
        "full_query_inside_caps": False,
    }
    exceptional = {
        "schema": "p1553.two_sided_join_exceptional_controls.r102.v1",
        "blind_bottom_complete": controls["all_blind_queries_bottom"],
        "identity_target_complete": controls["all_identity_queries_exact"],
        "repeated_atom_target_complete": controls[
            "all_repeated_targets_exact"
        ],
        "synthetic_joint_collision_multiplicity_complete": (
            cyclic["all_target_counts_exact"]
            and cyclic["all_positive_sources_replay"]
            and cyclic["multiplicity_above_one"]
        ),
        "partition_incidence_conservation_complete": controls[
            "all_partition_incidence_sums_exact"
        ],
        "partition_union_coverage_complete": controls[
            "all_partition_unions_complete"
        ],
        "projective_infinity_complete": False,
        "proper_subsum_complete": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_identical_descent.r102.v1",
        "exact_overcap_two_sided_source_baseline": True,
        "target_forced_algebraic_filter_supplied": False,
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
        "canonical_all_target_histograms_exact": controls[
            "all_canonical_histograms_exact"
        ],
        "canonical_all_target_source_keys_exact": controls[
            "all_canonical_source_keys_exact"
        ],
        "canonical_all_first_sources_replay": controls[
            "all_first_sources_replay"
        ],
        "sampled_fresh_query_counts_and_sources_exact": controls[
            "all_sampled_queries_exact"
        ],
        "blind_queries_return_bottom": controls[
            "all_blind_queries_bottom"
        ],
        "identity_queries_exact": controls["all_identity_queries_exact"],
        "repeated_atom_targets_exact": controls[
            "all_repeated_targets_exact"
        ],
        "synthetic_joint_collision_multiplicity_exact": exceptional[
            "synthetic_joint_collision_multiplicity_complete"
        ],
        "scalar_blind_construction": controls["all_scalar_blind"],
        "prefix_state_exponent_B11O5": (
            costs["canonical_prefix_join"]["state_exponent_B"]["exact"]
            == "11/5"
        ),
        "prefix_state_inside_setup_cap": costs[
            "canonical_prefix_join"
        ]["inside_setup_cap"],
        "suffix_query_exponent_B14O5": (
            costs["canonical_prefix_join"]["query_exponent_B"]["exact"]
            == "14/5"
        ),
        "suffix_query_outside_online_cap": not costs[
            "canonical_prefix_join"
        ]["inside_online_cap"],
        "all_atom_count_splits_exhausted": len(
            split["all_nontrivial_atom_count_splits"]
        )
        == 34,
        "best_setup_eligible_split_is_B11O5_B14O5": (
            split["best_state_exponent_B"]["exact"] == "11/5"
            and split["best_query_exponent_B"]["exact"] == "14/5"
        ),
        "ordered_source_and_group_exponents_both_B5": (
            costs["representation_density"][
                "ordered_factor_source_exponent_B"
            ]["exact"]
            == "5"
            and costs["representation_density"][
                "subgroup_order_exponent_B"
            ]["exact"]
            == "5"
        ),
        "representation_surplus_exponent_zero": (
            costs["representation_density"][
                "representation_surplus_exponent_B"
            ]["exact"]
            == "0"
        ),
        "partition_source_incidence_conserved": controls[
            "all_partition_incidence_sums_exact"
        ],
        "partition_class_union_covers_all_targets": controls[
            "all_partition_unions_complete"
        ],
        "required_filter_deficit_exponent_B31O20": (
            costs["disjoint_filter_conservation"][
                "required_ideal_pruning_exponent_B"
            ]["exact"]
            == "31/20"
        ),
        "filter_repetition_restores_B14O5": (
            costs["disjoint_filter_conservation"][
                "restored_total_work_exponent_B"
            ]["exact"]
            == "14/5"
        ),
        "exact_integer_count_and_joint_source_baseline": True,
        "full_two_sided_join_inside_caps": False,
        "target_forced_algebraic_filter_supplied": False,
        "projective_and_proper_subsum_complete": False,
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
            "CANONICAL_4A1C_PREFIX_JOIN_EXACT_WITH_B11O5_STATE__"
            "B14O5_FRESH_QUERY_OVER_CAP__DENSITY_ONE_DISJOINT_FILTER_"
            "PRUNING_PAID_BACK_BY_REPETITIONS__TARGET_FORCED_ALGEBRAIC_"
            "JOIN_OPEN"
        ),
        "source_bindings": {
            "r101_producer": {
                "path": str(R101_PRODUCER),
                "sha256": R101_PRODUCER_SHA256,
            },
            "r101_report": {
                "path": str(R101_REPORT),
                "sha256": R101_REPORT_SHA256,
            },
            "r101_gate": {
                "path": str(R101_GATE),
                "sha256": R101_GATE_SHA256,
            },
            "r101_parent": {
                "path": str(R101_PARENT),
                "sha256": R101_PARENT_SHA256,
            },
            "r100_report": {
                "path": str(R100_REPORT),
                "sha256": R100_REPORT_SHA256,
            },
            "r100_gate": {
                "path": str(R100_GATE),
                "sha256": R100_GATE_SHA256,
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
            "p1513_handoff": {
                "path": str(P1513_HANDOFF),
                "sha256": P1513_HANDOFF_SHA256,
            },
        },
        "novelty_scope": (
            "R102 freezes the cap-optimal canonical atom-count split and an "
            "exact integer/source query, then separates density-one filter "
            "conservation from target-forced algebraic identities."
        ),
        "actual_controls": controls,
        "asymptotic_control": costs,
        "artifacts": {
            "frozen": "frozen_5a5c_two_sided_implicit_join.json",
            "state_ledger": (
                "two_sided_join_state_and_transition_ledger.json"
            ),
            "source_replay": (
                "two_sided_join_integer_source_replay.json"
            ),
            "exceptional": "two_sided_join_exceptional_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r102.json",
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "lane_admitted": not failures,
            "failures": failures,
        },
        "exact_overcap_join_baseline_admitted": True,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "scope_boundary": (
            "R102 closes direct canonical meet-in-the-middle splits and "
            "extra disjoint/Wagner-style filters with no polynomial "
            "representation surplus. It is not a lower bound on a "
            "target-forced Semaev, FFE, or other algebraic identity that "
            "every true join satisfies."
        ),
        "next_action": (
            "Construct or refute one target-forced algebraic join filter "
            "for L+R=T. Freeze a public rational invariant and its Semaev/"
            "FFE relation before outcomes; require that every true source "
            "survives without B^delta thinning, that compact D_A,D_C state "
            "builds below B^(9/4), and that a fresh target returns exact "
            "integer count and one coupled source below B^(5/4), with "
            "false positives, infinity, multiplicity, rank, factor logs, "
            "and identical descent fully charged."
        ),
        "disposition": (
            "ADMIT_EXACT_OVER_CAP_CANONICAL_JOIN_ONLY__4A1C_PREFIX_STATE_"
            "B11O5__1A4C_QUERY_B14O5__ALL_ACTUAL_HISTOGRAMS_AND_SOURCES_"
            "EXACT__HASH_PARTITION_INCIDENCE_CONSERVED__SOURCE_DENSITY_"
            "EXPONENT_ZERO__B31O20_IDEAL_PRUNING_REQUIRES_B31O20_RETRIES__"
            "TARGET_FORCED_ALGEBRAIC_FILTER_OPEN__NO_RANK__NO_FACTOR_LOGS__"
            "NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "state_ledger": state_ledger,
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
            "p1553_5a5c_two_sided_implicit_join_probe_report_r102.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_5a5c_two_sided_implicit_join.json"
        ),
    )
    parser.add_argument(
        "--state-ledger-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "two_sided_join_state_and_transition_ledger.json"
        ),
    )
    parser.add_argument(
        "--source-replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "two_sided_join_integer_source_replay.json"
        ),
    )
    parser.add_argument(
        "--exceptional-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "two_sided_join_exceptional_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r102.json"
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
    write_json(args.state_ledger_output, bundle["state_ledger"])
    write_json(args.source_replay_output, bundle["source_replay"])
    write_json(args.exceptional_output, bundle["exceptional"])
    write_json(args.logs_output, bundle["logs_descent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
