#!/usr/bin/env python3
"""Measure exact translation-orbit rank for the S6 pair-query kernel."""

from __future__ import annotations

import argparse
from functools import cache
import importlib.util
import json
import pathlib
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.target_translated_frequency_orbit_probe.r77.v1"
PREFIX_SIZES = (2, 3, 4, 6, 8)
SETUP_STATE_CAP_EXPONENT = 9 / 4
ONLINE_WORKSPACE_CAP_EXPONENT = 5 / 4

R70_REPORT = pathlib.Path(
    "p1553_multiplicative_x_s3_closure_screen_report_r70.json"
)
R70_REPORT_SHA256 = (
    "9e58c6178eb18b7535c59e117ad942465d6c7853890291dec2c7c0abdb4ffd89"
)
R76_REPORT = pathlib.Path(
    "p1553_s6_subset_incidence_mobius_probe_report_r76.json"
)
R76_REPORT_SHA256 = (
    "de41d1618bc71c46f700bfead0ed72c5ac0b29a89da3b5c32534a15314ef4c93"
)
R3_REGISTRY = pathlib.Path("p1553_r31_artifact_index_README.md")
R3_REGISTRY_SHA256 = (
    "0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f"
)
R3_GATE_SHA256 = (
    "b2ee5934e295ab1f0d6b43452898e520d0cb18e718a8f5865694b25909b0df5e"
)


def load_module(path: str, module_name: str) -> Any:
    module_path = pathlib.Path(__file__).with_name(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R70 = load_module(
    "p1553_multiplicative_x_s3_closure_screen_r70.py",
    "p1553_r70_for_r77",
)
Point = tuple[int, int] | None


@cache
def fourier_field(group_order: int, minimum_prime: int) -> tuple[int, int]:
    multiplier = max(1, (minimum_prime - 1 + group_order - 1) // group_order)
    while True:
        candidate = multiplier * group_order + 1
        if R70.is_prime(candidate):
            generator = R70.primitive_root(candidate)
            root = pow(generator, multiplier, candidate)
            if (
                pow(root, group_order, candidate) == 1
                and root != 1
            ):
                return candidate, root
        multiplier += 1


def scalar_label_table(
    generator: Point,
    curve: dict[str, Any],
) -> dict[Point, int]:
    if generator is None:
        raise ValueError("generator must be nonidentity")
    order = curve["subgroup_order"]
    labels: dict[Point, int] = {None: 0}
    point = None
    for scalar in range(1, order):
        point = R70.add(point, generator, curve)
        if point is None or point in labels:
            raise AssertionError("generator orbit repeated before prime order")
        labels[point] = scalar
    if R70.add(point, generator, curve) is not None:
        raise AssertionError("generator orbit did not close at subgroup order")
    return labels


def symmetric_indicator(
    points: Sequence[Point],
    size: int,
    labels: dict[Point, int],
    group_order: int,
) -> list[int]:
    indicator = [0] * group_order
    for point in points[:size]:
        if point is None:
            raise ValueError("deck contains identity")
        scalar = labels[point]
        indicator[scalar] += 1
        indicator[-scalar % group_order] += 1
    return indicator


def cyclic_convolution(left: Sequence[int], right: Sequence[int]) -> list[int]:
    if len(left) != len(right):
        raise ValueError("cyclic vectors must have equal length")
    order = len(left)
    output = [0] * order
    left_support = [(index, value) for index, value in enumerate(left) if value]
    right_support = [
        (index, value) for index, value in enumerate(right) if value
    ]
    for left_index, left_value in left_support:
        for right_index, right_value in right_support:
            output[(left_index + right_index) % order] += (
                left_value * right_value
            )
    return output


def convolve_all(vectors: Sequence[Sequence[int]]) -> list[int]:
    if not vectors:
        raise ValueError("at least one vector is required")
    result = [1, *([0] * (len(vectors[0]) - 1))]
    for vector in vectors:
        result = cyclic_convolution(result, vector)
    return result


def dft(
    vector: Sequence[int],
    root: int,
    modulus: int,
) -> list[int]:
    order = len(vector)
    powers = [pow(root, exponent, modulus) for exponent in range(order)]
    support = [(index, value % modulus) for index, value in enumerate(vector) if value]
    return [
        sum(
            value * powers[(frequency * index) % order]
            for index, value in support
        )
        % modulus
        for frequency in range(order)
    ]


def pointwise_product(
    vectors: Sequence[Sequence[int]],
    modulus: int,
) -> list[int]:
    if not vectors:
        raise ValueError("at least one vector is required")
    return [
        product
        for values in zip(*vectors)
        for product in [
            _modular_product(values, modulus)
        ]
    ]


def _modular_product(values: Iterable[int], modulus: int) -> int:
    product = 1
    for value in values:
        product = product * value % modulus
    return product


def circulant_rows(vector: Sequence[int]) -> list[list[int]]:
    order = len(vector)
    return [
        [vector[(column - row) % order] for column in range(order)]
        for row in range(order)
    ]


def small_circulant_rank_control() -> dict[str, Any]:
    order = 7
    modulus, root = fourier_field(order, 100)
    vector = [1, 2, 0, 1, 0, 0, 0]
    spectrum = dft(vector, root, modulus)
    _, pivots = R70.row_reduce(
        circulant_rows(vector),
        order,
        modulus,
    )
    spectral_support = sum(value != 0 for value in spectrum)
    return {
        "group_order": order,
        "fourier_field_prime": modulus,
        "circulant_matrix_rank": len(pivots),
        "nonzero_fourier_mode_count": spectral_support,
        "rank_equals_fourier_support": len(pivots) == spectral_support,
    }


def source_decks(
    curve: dict[str, Any],
    source_id: str,
) -> list[list[Point]]:
    if source_id == "multiplicative_x_reused":
        deck, _ = R70.multiplicative_x_base(curve)
        return [deck[:] for _ in range(5)]
    if source_id == "independent_hash_decks":
        return [
            R70.hash_control_base(curve, 7700 + deck_index)
            for deck_index in range(5)
        ]
    raise ValueError(f"unknown source {source_id}")


def source_profile(
    curve: dict[str, Any],
    source_id: str,
    prefix_sizes: Iterable[int],
) -> dict[str, Any]:
    decks = source_decks(curve, source_id)
    generator = next(point for point in decks[0] if point is not None)
    labels = scalar_label_table(generator, curve)
    order = curve["subgroup_order"]
    maximum_size = max(prefix_sizes)
    maximum_branch_count = (2 * maximum_size) ** 5
    modulus, root = fourier_field(order, maximum_branch_count + 1)
    rows = []
    for size in prefix_sizes:
        indicators = [
            symmetric_indicator(deck, size, labels, order)
            for deck in decks
        ]
        spectra = [dft(vector, root, modulus) for vector in indicators]
        pair_spectrum = pointwise_product(spectra[3:], modulus)
        full_spectrum = pointwise_product(spectra, modulus)
        full_correlation = convolve_all(indicators)
        direct_spectrum = dft(full_correlation, root, modulus)
        pair_support = sum(value != 0 for value in pair_spectrum)
        full_support = sum(value != 0 for value in full_spectrum)
        rows.append(
            {
                "deck_size": size,
                "oriented_occurrence_count": (2 * size) ** 5,
                "target_zero_count": sum(
                    count == 0 for count in full_correlation
                ),
                "target_positive_count": sum(
                    count > 0 for count in full_correlation
                ),
                "maximum_target_branch_multiplicity": max(full_correlation),
                "pair_query_translation_orbit_rank": pair_support,
                "pair_query_translation_orbit_full_rank": (
                    pair_support == order
                ),
                "fivefold_correlation_fourier_support": full_support,
                "fivefold_correlation_full_fourier_support": (
                    full_support == order
                ),
                "convolution_theorem_exact": (
                    direct_spectrum == full_spectrum
                ),
                "branch_mass_conserved": (
                    sum(full_correlation) == (2 * size) ** 5
                ),
            }
        )
    return {
        "source_id": source_id,
        "scalar_labels_consumed_for_diagnostic_fourier_control": True,
        "scalar_labels_available_to_fresh_target_algorithm": False,
        "fourier_field_prime": modulus,
        "fourier_root_order": order,
        "prefixes": rows,
    }


def probe_curve(
    curve: dict[str, Any],
    prefix_sizes: Iterable[int] = PREFIX_SIZES,
) -> dict[str, Any]:
    prefix_sizes = tuple(prefix_sizes)
    return {
        "family_id": curve["family_id"],
        "field_prime": curve["field_prime"],
        "subgroup_order": curve["subgroup_order"],
        "sources": [
            source_profile(curve, source_id, prefix_sizes)
            for source_id in (
                "multiplicative_x_reused",
                "independent_hash_decks",
            )
        ],
    }


def build_report(
    prefix_sizes: Iterable[int] = PREFIX_SIZES,
) -> dict[str, Any]:
    families = [
        probe_curve(dict(curve), prefix_sizes=prefix_sizes)
        for curve in R70.CURVES
    ]
    rows = [
        row
        for family in families
        for source in family["sources"]
        for row in source["prefixes"]
    ]
    rank_control = small_circulant_rank_control()
    return {
        "schema": SCHEMA,
        "classification": (
            "TARGET_TRANSLATED_LINEAR_FREQUENCY_ORACLE_HAS_FULL_GROUP_ORBIT"
        ),
        "source_bindings": {
            "r70_multiplicative_x_control": {
                "path": str(R70_REPORT),
                "sha256": R70_REPORT_SHA256,
            },
            "r76_subset_incidence_control": {
                "path": str(R76_REPORT),
                "sha256": R76_REPORT_SHA256,
            },
            "r3_query2p1_registry": {
                "path": str(R3_REGISTRY),
                "sha256": R3_REGISTRY_SHA256,
                "bound_gate_sha256": R3_GATE_SHA256,
            },
        },
        "reduction": {
            "oriented_endpoint_indicator": (
                "f_i(P)=occurrence multiplicity of P or -P in deck i"
            ),
            "pair_query_kernel": "g=f_4*f_5 in the prime-order group algebra",
            "target_query": (
                "the signed five-list branch count at target R is "
                "(f_1*f_2*f_3*f_4*f_5)(-R)"
            ),
            "linear_orbit_rank_lemma": (
                "The rank of all translates of g equals the number of "
                "nonzero group-character Fourier coefficients of g."
            ),
            "relation_to_r76": (
                "On a support-separated chart this is the singleton part of "
                "R76. Higher endpoint subsets remain required on degenerate "
                "multiple-common-root strata."
            ),
        },
        "small_circulant_rank_control": rank_control,
        "families": families,
        "aggregate": {
            "family_count": len(families),
            "source_count": sum(len(family["sources"]) for family in families),
            "instance_count": len(rows),
            "all_convolution_theorem_checks_exact": all(
                row["convolution_theorem_exact"] for row in rows
            ),
            "all_branch_masses_conserved": all(
                row["branch_mass_conserved"] for row in rows
            ),
            "all_pair_query_translation_orbits_full_rank": all(
                row["pair_query_translation_orbit_full_rank"] for row in rows
            ),
            "all_fivefold_correlations_have_full_fourier_support": all(
                row["fivefold_correlation_full_fourier_support"]
                for row in rows
            ),
            "blind_zero_instance_count": sum(
                row["target_zero_count"] for row in rows
            ),
            "positive_target_instance_count": sum(
                row["target_positive_count"] for row in rows
            ),
            "small_rank_control_passed": (
                rank_control["rank_equals_fourier_support"]
            ),
        },
        "cost_ledger": {
            "setup_state_cap_exponent_B": SETUP_STATE_CAP_EXPONENT,
            "online_workspace_cap_exponent_B": (
                ONLINE_WORKSPACE_CAP_EXPONENT
            ),
            "universal_linear_translation_sketch_dimension": (
                "ambient prime subgroup order q on every frozen instance"
            ),
            "nontrivial_character_evaluation_on_fresh_target": (
                "requires an injective scalar-additive representation and is "
                "not supplied"
            ),
            "r3_scalar_additive_map_boundary_applies": True,
            "linear_frequency_lane_inside_caps": False,
        },
        "deduplication": {
            "r3_overlap": (
                "R3 already rejects scalar-additive fresh-target indexing and "
                "standard shifted pair-divisor representations."
            ),
            "r77_increment": (
                "R77 gives an executable exact orbit-rank control: even after "
                "granting scalar labels, the universal linear pair-query "
                "translation family has no missing character mode."
            ),
            "new_idea_id": None,
        },
        "admission": {
            "passed_obligation_count": 5,
            "obligation_count": 9,
            "lane_admitted": False,
            "failures": [
                "linear query advice has full ambient group dimension",
                "fresh-target character coordinates are unsupplied",
                "R76 higher-subset degeneracy corrections are omitted",
                "factor logs and identical target descent are unsupplied",
            ],
        },
        "breakthrough": False,
        "shoup_bound_improvement": False,
        "factor_log_solve_complete": False,
        "fresh_target_descent_complete": False,
        "scope_boundary": (
            "This rejects universal exact linear shift-equivariant sketches "
            "for the oriented pair kernel. It is not a lower bound against "
            "nonlinear, target-specialized, implicit resultant, randomized "
            "Las Vegas, or support-restricted data structures."
        ),
        "next_action": (
            "Freeze one nonlinear target-specialized nested-resultant scalar "
            "functional for the actual S4 deck factors. It must avoid group "
            "characters, B^3 prefix values, and B^2 suffix materialization, "
            "while replaying R76 counts, multiple-root correction, blind zero, "
            "one source, dyadic children, and direct caps."
        ),
        "disposition": (
            "REJECT_UNIVERSAL_LINEAR_TARGET_TRANSLATION_FREQUENCY_SKETCH_ONLY__"
            "FOUR_PRIME_ORDER_TOY_GROUPS__MULTIPLICATIVE_AND_HASH_DECKS__"
            "B2_3_4_6_8__FULL_PAIR_QUERY_CHARACTER_SUPPORT__EXACT_CONVOLUTION__"
            "SCALAR_LABELS_GRANTED_ONLY_AS_DIAGNOSTIC__R3_BOUNDARY_SHARPENED__"
            "NONLINEAR_TARGET_SPECIALIZATION_OPEN__NO_FACTOR_LOGS__NO_DESCENT__"
            "NO_SHOUP_CLAIM__NO_BREAKTHROUGH"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_target_translated_frequency_orbit_probe_report_r77.json"
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
        f"pair_orbits_full="
        f"{aggregate['all_pair_query_translation_orbits_full_rank']} "
        f"convolution_exact="
        f"{aggregate['all_convolution_theorem_checks_exact']} "
        f"lane_admitted={report['admission']['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
