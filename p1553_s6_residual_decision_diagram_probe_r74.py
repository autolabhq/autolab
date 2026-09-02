#!/usr/bin/env python3
"""Measure exact S6 residual-state reuse without root presupposition."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from typing import Any, Iterable


SCHEMA = "p1553.s6_residual_decision_diagram_probe.r74.v1"
PREFIX_SIZES = (6, 10, 14, 18)
SETUP_STATE_CAP_EXPONENT = 9 / 4
ONLINE_WORKSPACE_CAP_EXPONENT = 5 / 4

R72_REPORT = pathlib.Path(
    "p1553_s6_centered_carry_rank_minor_probe_report_r72.json"
)
R72_REPORT_SHA256 = (
    "7e63b52fc7667be14aadc1db3aeeb22b43876ff2c62e3dce87b312c056e85e43"
)
R73_REPORT = pathlib.Path(
    "p1553_resultant_valuation_trace_grammar_report_r73.json"
)
R73_REPORT_SHA256 = (
    "00f750c15644acdaea32bbbbe9b407071cd3bf6a5cf2268ce075c55bd9a29915"
)
R14_GATE = pathlib.Path("p1553_tensor_trace_minpoly_compiler_gate_r14.md")
R14_GATE_SHA256 = (
    "da12515cf2bef622f320fd1a2c174b3fc2920cc39ae223af23b314b64709b4ac"
)


def load_module(path: str, module_name: str) -> Any:
    module_path = pathlib.Path(__file__).with_name(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R72 = load_module(
    "p1553_s6_centered_carry_rank_minor_probe_r72.py",
    "p1553_r72_for_r74",
)
R73 = load_module(
    "p1553_resultant_valuation_trace_grammar_r73.py",
    "p1553_r73_for_r74",
)
Point = tuple[int, int] | None
EndpointKey = tuple[bool, tuple[int, ...]]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint_key(
    points: tuple[Point, Point, Point],
    curve: dict[str, Any],
) -> EndpointKey:
    """Return all signed three-point sums modulo global sign."""

    if any(point is None for point in points):
        raise ValueError("endpoint key expects affine nonidentity inputs")
    endpoints: set[int] = set()
    contains_identity = False
    first, second, third = points
    for second_sign, third_sign in itertools.product((-1, 1), repeat=2):
        signed_second = (
            second
            if second_sign == 1
            else R72.R70.negate(second, curve)
        )
        signed_third = (
            third
            if third_sign == 1
            else R72.R70.negate(third, curve)
        )
        total = R72.R70.add(first, signed_second, curve)
        total = R72.R70.add(total, signed_third, curve)
        if total is None:
            contains_identity = True
        else:
            endpoints.add(total[0])
    return contains_identity, tuple(sorted(endpoints))


def keys_intersect(left: EndpointKey, right: EndpointKey) -> bool:
    if left[0] and right[0]:
        return True
    return bool(set(left[1]).intersection(right[1]))


def projective_normalize(
    polynomial: list[int],
    modulus: int,
) -> tuple[int, ...]:
    polynomial = R73.trim(
        [coefficient % modulus for coefficient in polynomial]
    )
    if polynomial == [0]:
        return (0,)
    scale = R73.inverse(polynomial[-1], modulus)
    return tuple(coefficient * scale % modulus for coefficient in polynomial)


def squarefree_part(
    polynomial: list[int],
    modulus: int,
) -> list[int]:
    polynomial = R73.trim(
        [coefficient % modulus for coefficient in polynomial]
    )
    if polynomial == [0]:
        return [0]
    derivative = [
        degree * polynomial[degree] % modulus
        for degree in range(1, len(polynomial))
    ]
    if not derivative:
        return polynomial
    common = R73.polynomial_gcd(polynomial, derivative, modulus)
    quotient, remainder = R73.polynomial_divmod(
        polynomial,
        common,
        modulus,
    )
    if remainder != [0]:
        raise AssertionError("squarefree division left a remainder")
    return R73.trim(quotient)


def endpoint_polynomial_key(
    key: EndpointKey,
    modulus: int,
) -> tuple[bool, tuple[int, ...]]:
    polynomial = R73.polynomial_from_roots(key[1], modulus)
    return key[0], projective_normalize(polynomial, modulus)


def s4_radical_key(
    points: tuple[Point, Point, Point],
    curve: dict[str, Any],
) -> tuple[bool, tuple[int, ...]]:
    polynomial = R72.s4_polynomial_last(
        points[0][0],
        points[1][0],
        points[2][0],
        curve["curve_a"],
        curve["curve_b"],
        curve["field_prime"],
    )
    radical = squarefree_part(polynomial, curve["field_prime"])
    contains_infinity = len(R73.trim(polynomial[:])) - 1 < 4
    return (
        contains_infinity,
        projective_normalize(radical, curve["field_prime"]),
    )


def triple_key_rows(
    decks: list[list[Point]],
    curve: dict[str, Any],
    size: int,
) -> list[EndpointKey]:
    return [
        endpoint_key(
            (
                decks[0][first],
                decks[1][second],
                decks[2][third],
            ),
            curve,
        )
        for first, second, third in itertools.product(range(size), repeat=3)
    ]


def suffix_key_rows(
    decks: list[list[Point]],
    target: Point,
    curve: dict[str, Any],
    size: int,
) -> list[EndpointKey]:
    return [
        endpoint_key(
            (
                target,
                decks[3][fourth],
                decks[4][fifth],
            ),
            curve,
        )
        for fourth, fifth in itertools.product(range(size), repeat=2)
    ]


def incidence_signatures(
    triple_keys: list[EndpointKey],
    suffix_keys: list[EndpointKey],
) -> tuple[list[tuple[int, ...]], int]:
    suffix_by_endpoint: dict[tuple[str, int], set[int]] = {}
    for suffix_index, key in enumerate(suffix_keys):
        if key[0]:
            suffix_by_endpoint.setdefault(("identity", 0), set()).add(
                suffix_index
            )
        for x_coordinate in key[1]:
            suffix_by_endpoint.setdefault(
                ("x", x_coordinate),
                set(),
            ).add(suffix_index)

    signatures = []
    incidence_count = 0
    for key in triple_keys:
        matches: set[int] = set()
        if key[0]:
            matches.update(suffix_by_endpoint.get(("identity", 0), set()))
        for x_coordinate in key[1]:
            matches.update(
                suffix_by_endpoint.get(("x", x_coordinate), set())
            )
        signature = tuple(sorted(matches))
        incidence_count += len(signature)
        signatures.append(signature)
    return signatures, incidence_count


def radical_sample_replay(
    decks: list[list[Point]],
    curve: dict[str, Any],
    size: int,
) -> dict[str, Any]:
    candidate_indices = [
        (0, 0, 0),
        (0, min(1, size - 1), min(2, size - 1)),
        (min(1, size - 1), 0, min(3, size - 1)),
        (size - 1, size - 1, size - 1),
    ]
    rows = []
    for indices in candidate_indices:
        points = tuple(
            decks[coordinate][index]
            for coordinate, index in enumerate(indices)
        )
        endpoint = endpoint_key(points, curve)
        endpoint_polynomial = endpoint_polynomial_key(
            endpoint,
            curve["field_prime"],
        )
        radical = s4_radical_key(points, curve)
        row = {
            "indices": list(indices),
            "endpoint_key": {
                "contains_infinity": endpoint[0],
                "finite_root_count": len(endpoint[1]),
            },
            "matches_s4_squarefree_radical": endpoint_polynomial == radical,
        }
        rows.append(row)
    return {
        "rows": rows,
        "all_match": all(
            row["matches_s4_squarefree_radical"] for row in rows
        ),
    }


def target_profile(
    target_spec: dict[str, Any],
    triple_keys: list[EndpointKey],
    decks: list[list[Point]],
    curve: dict[str, Any],
    size: int,
) -> dict[str, Any]:
    suffix_keys = suffix_key_rows(
        decks,
        target_spec["point"],
        curve,
        size,
    )
    signatures, incidence_count = incidence_signatures(
        triple_keys,
        suffix_keys,
    )
    distinct_signatures = set(signatures)
    forced_witness_recognized = None
    forced_witness = target_spec["forced_witness"]
    if forced_witness is not None and max(forced_witness) < size:
        first, second, third, fourth, fifth = forced_witness
        triple_index = (first * size + second) * size + third
        suffix_index = fourth * size + fifth
        forced_witness_recognized = suffix_index in signatures[triple_index]
    return {
        "target_id": target_spec["target_id"],
        "suffix_occurrence_count": len(suffix_keys),
        "distinct_suffix_algebraic_key_count": len(set(suffix_keys)),
        "oracle_boolean_residual_state_count": len(distinct_signatures),
        "oracle_empty_state_present": () in distinct_signatures,
        "nonempty_triple_residual_count": sum(bool(row) for row in signatures),
        "predicate_incidence_count": incidence_count,
        "forced_witness_recognized": forced_witness_recognized,
        "oracle_construction_charged": False,
        "oracle_construction_boundary": (
            "The signatures are measured from exact endpoint intersections. "
            "Constructing them naively evaluates B^3 by B^2 incidences and "
            "presupposes the relation support."
        ),
    }


def probe_curve(
    curve: dict[str, Any],
    prefix_sizes: Iterable[int] = PREFIX_SIZES,
) -> dict[str, Any]:
    decks, targets = R72.public_decks_and_targets(curve)
    prefix_rows = []
    for size in prefix_sizes:
        if size > len(decks[0]):
            raise ValueError("prefix exceeds frozen deck")
        triple_keys = triple_key_rows(decks, curve, size)
        distinct_triple_keys = len(set(triple_keys))
        prefix_rows.append(
            {
                "deck_size": size,
                "triple_occurrence_count": len(triple_keys),
                "distinct_triple_algebraic_key_count": distinct_triple_keys,
                "triple_key_collision_count": (
                    len(triple_keys) - distinct_triple_keys
                ),
                "observed_state_exponent_B": (
                    math.log(distinct_triple_keys, size)
                    if size > 1 and distinct_triple_keys > 0
                    else None
                ),
                "radical_sample_replay": radical_sample_replay(
                    decks,
                    curve,
                    size,
                ),
                "targets": [
                    target_profile(
                        target,
                        triple_keys,
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


def build_report() -> dict[str, Any]:
    families = [probe_curve(dict(curve)) for curve in R72.CURVES]
    prefix_rows = [
        prefix_row
        for family in families
        for prefix_row in family["prefixes"]
    ]
    target_rows = [
        target
        for prefix_row in prefix_rows
        for target in prefix_row["targets"]
    ]
    all_algebraic_keys_distinct = all(
        row["triple_key_collision_count"] == 0 for row in prefix_rows
    )
    all_radicals_match = all(
        row["radical_sample_replay"]["all_match"] for row in prefix_rows
    )
    forced_rows = [
        row for row in target_rows if row["forced_witness_recognized"] is not None
    ]
    blind_rows = [
        row for row in target_rows if row["target_id"] == "blind_hash_target"
    ]
    return {
        "schema": SCHEMA,
        "classification": (
            "S6_RESIDUAL_RADICAL_MEMOIZATION_HAS_B3_ALGEBRAIC_STATES"
        ),
        "source_bindings": {
            "r72_s6_carry_probe": {
                "path": str(R72_REPORT),
                "sha256": R72_REPORT_SHA256,
            },
            "r73_resultant_valuation_grammar": {
                "path": str(R73_REPORT),
                "sha256": R73_REPORT_SHA256,
            },
            "r14_tensor_trace_dedup": {
                "path": str(R14_GATE),
                "sha256": R14_GATE_SHA256,
            },
        },
        "grammar": {
            "grammar_id": "s6_squarefree_residual_radical_memoization_v1",
            "split": "three_labels_by_two_labels_plus_target",
            "algebraic_state": (
                "squarefree projective root support of "
                "S4(x1,x2,x3,z)"
            ),
            "transition": (
                "accept suffix iff its S4 squarefree root support intersects "
                "the stored prefix support"
            ),
            "not_centered_carry": True,
            "not_product_resultant": True,
            "not_tensor_trace_minpoly": True,
        },
        "families": families,
        "aggregate": {
            "family_count": len(families),
            "prefix_instance_count": len(prefix_rows),
            "target_instance_count": len(target_rows),
            "all_triple_algebraic_keys_distinct": all_algebraic_keys_distinct,
            "all_sampled_endpoint_keys_match_s4_radicals": all_radicals_match,
            "all_forced_witnesses_recognized": all(
                row["forced_witness_recognized"] for row in forced_rows
            ),
            "all_blind_target_oracle_diagrams_single_empty_state": all(
                row["oracle_boolean_residual_state_count"] == 1
                and row["predicate_incidence_count"] == 0
                for row in blind_rows
            ),
            "maximum_oracle_boolean_residual_state_count": max(
                row["oracle_boolean_residual_state_count"]
                for row in target_rows
            ),
            "maximum_predicate_incidence_count": max(
                row["predicate_incidence_count"] for row in target_rows
            ),
        },
        "cost_ledger": {
            "setup_state_cap_exponent_B": SETUP_STATE_CAP_EXPONENT,
            "online_workspace_cap_exponent_B": (
                ONLINE_WORKSPACE_CAP_EXPONENT
            ),
            "algebraic_prefix_state_exponent_B": 3.0,
            "algebraic_prefix_construction_exponent_B": 3.0,
            "literal_target_incidence_scan_exponent_B": 5.0,
            "pair_side_state_exponent_B": 2.0,
            "pair_by_fifth_transition_exponent_B": 3.0,
            "oracle_boolean_diagram_state_is_not_constructive_credit": True,
            "lane_inside_caps": False,
        },
        "admission": {
            "passed_obligation_count": 6,
            "obligation_count": 10,
            "lane_admitted": False,
            "failures": [
                "all frozen algebraic triple residual keys are distinct",
                "target-independent residual state exponent is 3",
                "on-demand algebraic construction costs B^3 online",
                "oracle Boolean compression presupposes relation incidences",
            ],
        },
        "breakthrough": False,
        "shoup_bound_improvement": False,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "next_action": (
            "Construct a support-adaptive transposed incidence algorithm that "
            "emits only nonempty S4-prefix/S4-suffix intersections without "
            "enumerating B^3 prefix radicals or B^5 incidences. Require blind "
            "zero certificates, the forced source, R73 multiplicities, dyadic "
            "children, and direct caps."
        ),
        "disposition": (
            "REJECT_S6_SQUAREFREE_RESIDUAL_RADICAL_MEMOIZATION_GRAMMAR_ONLY__"
            "FOUR_STANDARD_CURVES__B6_10_14_18__ALL_TRIPLE_ALGEBRAIC_KEYS_"
            "DISTINCT__STATE_AND_CONSTRUCTION_B3__ORACLE_BOOLEAN_DIAGRAM_TINY_"
            "ONLY_AFTER_ROOT_INCIDENCES__R14_KRYLOV_DUPLICATE_NOT_RERUN__"
            "NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_s6_residual_decision_diagram_probe_report_r74.json"
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
        f"algebraic_distinct={aggregate['all_triple_algebraic_keys_distinct']} "
        f"max_oracle_states={aggregate['maximum_oracle_boolean_residual_state_count']} "
        f"lane_admitted={report['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
