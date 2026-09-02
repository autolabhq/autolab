#!/usr/bin/env python3
"""Count exact S6 tuple incidences by endpoint-subset Mobius inversion."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.s6_subset_incidence_mobius_probe.r76.v1"
PREFIX_SIZES = (6, 10, 14, 18)
SETUP_STATE_CAP_EXPONENT = 9 / 4
ONLINE_WORKSPACE_CAP_EXPONENT = 5 / 4

R74_REPORT = pathlib.Path(
    "p1553_s6_residual_decision_diagram_probe_report_r74.json"
)
R74_REPORT_SHA256 = (
    "1558482f504bd5e05b112464ee7c6734dd30ee518789735ac7abc8c305a5f740"
)
R73_REPORT = pathlib.Path(
    "p1553_resultant_valuation_trace_grammar_report_r73.json"
)
R73_REPORT_SHA256 = (
    "00f750c15644acdaea32bbbbe9b407071cd3bf6a5cf2268ce075c55bd9a29915"
)
R2_REGISTRY = pathlib.Path("p1553_r31_artifact_index_README.md")
R2_REGISTRY_SHA256 = (
    "0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f"
)
R2_GATE_SHA256 = (
    "55acc1457e7fd5a740da57c2c1db957374c7c18561c67b1748176dc8c61fcda5"
)


def load_module(path: str, module_name: str) -> Any:
    module_path = pathlib.Path(__file__).with_name(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R74 = load_module(
    "p1553_s6_residual_decision_diagram_probe_r74.py",
    "p1553_r74_for_r76",
)
Point = tuple[int, int] | None
Atom = tuple[str, int]
SubsetKey = tuple[Atom, ...]
Occurrence = tuple[tuple[int, ...], R74.EndpointKey]


def endpoint_atoms(key: R74.EndpointKey) -> tuple[Atom, ...]:
    atoms = [("x", x_coordinate) for x_coordinate in key[1]]
    if key[0]:
        atoms.append(("identity", 0))
    return tuple(sorted(atoms))


def nonempty_subsets(atoms: Sequence[Atom]) -> Iterable[SubsetKey]:
    for size in range(1, len(atoms) + 1):
        yield from itertools.combinations(atoms, size)


def subset_histogram(
    occurrences: Iterable[Occurrence],
) -> tuple[Counter[SubsetKey], dict[SubsetKey, tuple[int, ...]], int]:
    histogram: Counter[SubsetKey] = Counter()
    first_source: dict[SubsetKey, tuple[int, ...]] = {}
    contribution_count = 0
    for indices, endpoint in occurrences:
        for subset in nonempty_subsets(endpoint_atoms(endpoint)):
            histogram[subset] += 1
            contribution_count += 1
            first_source.setdefault(subset, indices)
    return histogram, first_source, contribution_count


def mobius_incidence_count(
    left: Counter[SubsetKey],
    right: Counter[SubsetKey],
) -> int:
    if len(left) > len(right):
        left, right = right, left
    return sum(
        (1 if len(subset) % 2 else -1)
        * left_count
        * right.get(subset, 0)
        for subset, left_count in left.items()
    )


def direct_set_incidence_count(
    left_sets: Sequence[Sequence[Atom]],
    right_sets: Sequence[Sequence[Atom]],
) -> int:
    return sum(
        bool(set(left).intersection(right))
        for left in left_sets
        for right in right_sets
    )


def generic_histogram(
    rows: Sequence[Sequence[Atom]],
) -> Counter[SubsetKey]:
    return Counter(
        subset
        for atoms in rows
        for subset in nonempty_subsets(tuple(sorted(atoms)))
    )


def multiple_common_root_control() -> dict[str, Any]:
    left_sets = [
        (("x", 1), ("x", 2)),
        (("x", 1), ("x", 2)),
    ]
    right_sets = [
        (("x", 1), ("x", 2)),
        (("x", 2), ("x", 3)),
    ]
    direct_count = direct_set_incidence_count(left_sets, right_sets)
    left = generic_histogram(left_sets)
    right = generic_histogram(right_sets)
    mobius_count = mobius_incidence_count(left, right)
    singleton_endpoint_count = sum(
        left[subset] * right.get(subset, 0)
        for subset in left
        if len(subset) == 1
    )
    return {
        "left_occurrence_count": len(left_sets),
        "right_occurrence_count": len(right_sets),
        "direct_tuple_count": direct_count,
        "mobius_tuple_count": mobius_count,
        "singleton_endpoint_incidence_count": singleton_endpoint_count,
        "duplicate_left_occurrence_preserved": left_sets[0] == left_sets[1],
        "naive_singleton_count_overcounts": (
            singleton_endpoint_count > direct_count
        ),
        "mobius_count_exact": mobius_count == direct_count,
    }


def triple_occurrences(
    decks: list[list[Point]],
    curve: dict[str, Any],
    size: int,
) -> list[Occurrence]:
    return [
        (
            (first, second, third),
            R74.endpoint_key(
                (
                    decks[0][first],
                    decks[1][second],
                    decks[2][third],
                ),
                curve,
            ),
        )
        for first, second, third in itertools.product(range(size), repeat=3)
    ]


def suffix_occurrences(
    decks: list[list[Point]],
    target: Point,
    curve: dict[str, Any],
    size: int,
) -> list[Occurrence]:
    return [
        (
            (fourth, fifth),
            R74.endpoint_key(
                (
                    target,
                    decks[3][fourth],
                    decks[4][fifth],
                ),
                curve,
            ),
        )
        for fourth, fifth in itertools.product(range(size), repeat=2)
    ]


def recover_source(
    left_histogram: Counter[SubsetKey],
    left_sources: dict[SubsetKey, tuple[int, ...]],
    right_histogram: Counter[SubsetKey],
    right_sources: dict[SubsetKey, tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...], SubsetKey] | None:
    shared_singletons = sorted(
        subset
        for subset in left_histogram.keys() & right_histogram.keys()
        if len(subset) == 1
    )
    if not shared_singletons:
        return None
    subset = shared_singletons[0]
    return left_sources[subset], right_sources[subset], subset


def dyadic_ranges(start: int, stop: int) -> list[tuple[int, int, int, int]]:
    if stop - start <= 1:
        return []
    middle = start + (stop - start) // 2
    return [
        (start, stop, start, middle),
        (start, stop, middle, stop),
        *dyadic_ranges(start, middle),
        *dyadic_ranges(middle, stop),
    ]


def range_count(
    occurrences: Sequence[Occurrence],
    suffix_histogram: Counter[SubsetKey],
    start: int,
    stop: int,
) -> int:
    histogram, _, _ = subset_histogram(
        row for row in occurrences if start <= row[0][0] < stop
    )
    return mobius_incidence_count(histogram, suffix_histogram)


def dyadic_child_replay(
    prefix_occurrences: Sequence[Occurrence],
    suffix_histogram: Counter[SubsetKey],
    size: int,
) -> dict[str, Any]:
    counts: dict[tuple[int, int], int] = {}

    def count(interval: tuple[int, int]) -> int:
        if interval not in counts:
            counts[interval] = range_count(
                prefix_occurrences,
                suffix_histogram,
                interval[0],
                interval[1],
            )
        return counts[interval]

    conservation_rows = []
    worklist = [(0, size)]
    while worklist:
        start, stop = worklist.pop()
        if stop - start <= 1:
            continue
        middle = start + (stop - start) // 2
        left = (start, middle)
        right = (middle, stop)
        parent_count = count((start, stop))
        left_count = count(left)
        right_count = count(right)
        conservation_rows.append(parent_count == left_count + right_count)
        worklist.extend((left, right))
    leaf_counts = [count((index, index + 1)) for index in range(size)]
    return {
        "internal_node_count": len(conservation_rows),
        "all_parent_counts_equal_child_sums": all(conservation_rows),
        "root_count": count((0, size)),
        "positive_leaf_count": sum(value > 0 for value in leaf_counts),
        "positive_leaf_indices": [
            index for index, value in enumerate(leaf_counts) if value > 0
        ],
        "construction_boundary": (
            "Materializing all child histograms costs B^3 log B entries; "
            "the replay proves subset stability but receives no cap credit."
        ),
    }


def target_profile(
    target_spec: dict[str, Any],
    prefix_occurrences: Sequence[Occurrence],
    prefix_histogram: Counter[SubsetKey],
    prefix_sources: dict[SubsetKey, tuple[int, ...]],
    decks: list[list[Point]],
    curve: dict[str, Any],
    size: int,
) -> dict[str, Any]:
    suffix = suffix_occurrences(
        decks,
        target_spec["point"],
        curve,
        size,
    )
    suffix_histogram, suffix_sources, suffix_contributions = subset_histogram(
        suffix
    )
    mobius_count = mobius_incidence_count(
        prefix_histogram,
        suffix_histogram,
    )
    _, direct_count = R74.incidence_signatures(
        [endpoint for _, endpoint in prefix_occurrences],
        [endpoint for _, endpoint in suffix],
    )
    recovered = recover_source(
        prefix_histogram,
        prefix_sources,
        suffix_histogram,
        suffix_sources,
    )
    source_indices = None
    source_relation_verified = None
    source_subset_size = None
    if recovered is not None:
        prefix_indices, suffix_indices, subset = recovered
        source_indices = [*prefix_indices, *suffix_indices]
        source_subset_size = len(subset)
        source_relation_verified = R74.R72.signed_relation_exists(
            tuple(
                decks[coordinate][index]
                for coordinate, index in enumerate(source_indices)
            ),
            target_spec["point"],
            curve,
        )
    dyadic = dyadic_child_replay(
        prefix_occurrences,
        suffix_histogram,
        size,
    )
    return {
        "target_id": target_spec["target_id"],
        "suffix_occurrence_count": len(suffix),
        "suffix_distinct_subset_key_count": len(suffix_histogram),
        "suffix_subset_contribution_count": suffix_contributions,
        "mobius_tuple_count": mobius_count,
        "direct_tuple_count": direct_count,
        "counts_match": mobius_count == direct_count,
        "blind_zero_certificate": (
            target_spec["target_id"] != "blind_hash_target"
            or mobius_count == 0
        ),
        "source_indices": source_indices,
        "source_shared_subset_size": source_subset_size,
        "source_relation_verified": source_relation_verified,
        "forced_witness_recognized": (
            None
            if target_spec["forced_witness"] is None
            or max(target_spec["forced_witness"]) >= size
            else direct_count > 0
        ),
        "dyadic_child_replay": dyadic,
    }


def probe_curve(
    curve: dict[str, Any],
    prefix_sizes: Iterable[int] = PREFIX_SIZES,
) -> dict[str, Any]:
    decks, targets = R74.R72.public_decks_and_targets(curve)
    prefix_rows = []
    for size in prefix_sizes:
        if size > len(decks[0]):
            raise ValueError("prefix exceeds frozen deck")
        prefix = triple_occurrences(decks, curve, size)
        prefix_histogram, prefix_sources, prefix_contributions = (
            subset_histogram(prefix)
        )
        prefix_rows.append(
            {
                "deck_size": size,
                "prefix_occurrence_count": len(prefix),
                "prefix_distinct_subset_key_count": len(prefix_histogram),
                "prefix_subset_contribution_count": prefix_contributions,
                "observed_distinct_state_exponent_B": (
                    math.log(len(prefix_histogram), size)
                    if len(prefix_histogram) > 0 and size > 1
                    else None
                ),
                "targets": [
                    target_profile(
                        target,
                        prefix,
                        prefix_histogram,
                        prefix_sources,
                        decks,
                        curve,
                        size,
                    )
                    for target in targets
                ],
            }
        )
    return {
        "family_id": curve["family_id"],
        "field_bits": curve["field_prime"].bit_length(),
        "scalar_labels_consumed": False,
        "prefixes": prefix_rows,
    }


def build_report(
    prefix_sizes: Iterable[int] = PREFIX_SIZES,
) -> dict[str, Any]:
    families = [
        probe_curve(dict(curve), prefix_sizes=prefix_sizes)
        for curve in R74.R72.CURVES
    ]
    prefix_rows = [
        prefix
        for family in families
        for prefix in family["prefixes"]
    ]
    target_rows = [
        target
        for prefix in prefix_rows
        for target in prefix["targets"]
    ]
    forced_rows = [
        target
        for target in target_rows
        if target["target_id"] == "forced_positive_target"
    ]
    blind_rows = [
        target
        for target in target_rows
        if target["target_id"] == "blind_hash_target"
    ]
    duplicate_control = multiple_common_root_control()
    return {
        "schema": SCHEMA,
        "classification": (
            "EXACT_S6_SUBSET_INCIDENCE_COUNT_BUT_B3_PREFIX_STATE"
        ),
        "source_bindings": {
            "r74_s6_residual_support": {
                "path": str(R74_REPORT),
                "sha256": R74_REPORT_SHA256,
            },
            "r73_resultant_multiplicity_grammar": {
                "path": str(R73_REPORT),
                "sha256": R73_REPORT_SHA256,
            },
            "r2_finite_deck_gate_registry": {
                "path": str(R2_REGISTRY),
                "sha256": R2_REGISTRY_SHA256,
                "bound_gate_sha256": R2_GATE_SHA256,
            },
        },
        "identity": {
            "left_histogram": (
                "h_P(S)=number of prefix occurrences whose endpoint set "
                "contains nonempty subset S"
            ),
            "right_histogram": (
                "h_Q(S)=number of suffix occurrences whose endpoint set "
                "contains nonempty subset S"
            ),
            "tuple_count": (
                "sum over nonempty S of "
                "(-1)^(|S|+1) h_P(S) h_Q(S)"
            ),
            "maximum_subsets_per_occurrence": 15,
            "proof": (
                "Each prefix-suffix pair contributes the alternating sum of "
                "all nonempty subsets of its endpoint intersection, equal "
                "to one iff that intersection is nonempty."
            ),
        },
        "multiple_common_root_control": duplicate_control,
        "families": families,
        "aggregate": {
            "family_count": len(families),
            "prefix_instance_count": len(prefix_rows),
            "target_instance_count": len(target_rows),
            "all_mobius_counts_match_direct_tuple_counts": all(
                target["counts_match"] for target in target_rows
            ),
            "all_blind_zero_certificates": all(
                target["blind_zero_certificate"] for target in blind_rows
            ),
            "all_forced_sources_recovered_and_verified": all(
                target["source_relation_verified"] for target in forced_rows
            ),
            "all_forced_witnesses_recognized": all(
                target["forced_witness_recognized"] for target in forced_rows
            ),
            "all_dyadic_child_counts_conserve": all(
                target["dyadic_child_replay"][
                    "all_parent_counts_equal_child_sums"
                ]
                and target["dyadic_child_replay"]["root_count"]
                == target["mobius_tuple_count"]
                for target in target_rows
            ),
            "duplicate_occurrence_control_exact": (
                duplicate_control["mobius_count_exact"]
                and duplicate_control["duplicate_left_occurrence_preserved"]
                and duplicate_control["naive_singleton_count_overcounts"]
            ),
            "maximum_prefix_distinct_subset_key_count": max(
                prefix["prefix_distinct_subset_key_count"]
                for prefix in prefix_rows
            ),
            "maximum_prefix_subset_contribution_count": max(
                prefix["prefix_subset_contribution_count"]
                for prefix in prefix_rows
            ),
            "maximum_suffix_distinct_subset_key_count": max(
                target["suffix_distinct_subset_key_count"]
                for target in target_rows
            ),
            "maximum_exact_tuple_count": max(
                target["mobius_tuple_count"] for target in target_rows
            ),
        },
        "cost_ledger": {
            "setup_state_cap_exponent_B": SETUP_STATE_CAP_EXPONENT,
            "online_workspace_cap_exponent_B": (
                ONLINE_WORKSPACE_CAP_EXPONENT
            ),
            "prefix_occurrence_and_state_exponent_B": 3.0,
            "prefix_construction_work_exponent_B": 3.0,
            "fresh_target_suffix_enumeration_exponent_B": 2.0,
            "fresh_target_lookup_work_exponent_B": 2.0,
            "dyadic_all_children_work_exponent_B": 3.0,
            "dyadic_polylog_factor_suppressed": True,
            "literal_incidence_grid_avoided": True,
            "literal_incidence_grid_exponent_B": 5.0,
            "lane_inside_caps": False,
        },
        "deduplication": {
            "r2_overlap": (
                "R2 already observed that an exact existence bit and dyadic "
                "source replay suffice, and that standard 2+2+1 traffic "
                "returns B^3 or B^4."
            ),
            "r76_increment": (
                "R76 supplies an explicit exact occurrence-count identity, "
                "corrects multiple-common-root overcount, and gives source "
                "backpointers at the same rejected B^3 boundary."
            ),
            "new_idea_id": None,
        },
        "admission": {
            "passed_obligation_count": 8,
            "obligation_count": 12,
            "lane_admitted": False,
            "failures": [
                "target-independent prefix state costs B^3",
                "fresh-target suffix enumeration and lookup cost B^2",
                "factor-log rank and sparse solve are unsupplied",
                "identical scalar-blind fresh-target descent is unsupplied",
            ],
        },
        "breakthrough": False,
        "shoup_bound_improvement": False,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "next_action": (
            "Construct a target-translated subset-frequency oracle with "
            "prefix advice at most B^(9/4+o(1)) and fresh-target query work "
            "and workspace at most B^(5/4+o(1)), without enumerating either "
            "the B^3 prefix triples or B^2 target suffixes. Bind Query2P1 and "
            "return exact counts, a source, blind zero, and dyadic children."
        ),
        "disposition": (
            "REJECT_EXACT_S6_SUBSET_INCIDENCE_MOBIUS_CONSTRUCTION_ONLY__"
            "FOUR_STANDARD_CURVES__B6_10_14_18__EXACT_TUPLE_COUNTS__"
            "MULTIPLE_COMMON_ROOTS_CORRECTED__BLIND_ZERO__FORCED_SOURCE__"
            "DYADIC_CHILDREN__PREFIX_STATE_B3__FRESH_TARGET_B2__R2_BOUNDARY_"
            "SHARPENED_NOT_NEW_IDEA__NO_FACTOR_LOGS__NO_DESCENT__"
            "NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_s6_subset_incidence_mobius_probe_report_r76.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report()
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate = report["aggregate"]
    print(
        f"families={aggregate['family_count']} "
        f"counts_exact="
        f"{aggregate['all_mobius_counts_match_direct_tuple_counts']} "
        f"sources_verified="
        f"{aggregate['all_forced_sources_recovered_and_verified']} "
        f"lane_admitted={report['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
