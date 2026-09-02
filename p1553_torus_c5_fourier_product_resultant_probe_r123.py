#!/usr/bin/env python3
"""Audit Fourier and product-resultant circuits for torus C5 membership."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from collections import Counter
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.torus_c5_fourier_product_resultant.r123.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
C_ATOM_EXPONENT = Fraction(3, 4)
C2_EXPONENT = 2 * C_ATOM_EXPONENT
C3_EXPONENT = 3 * C_ATOM_EXPONENT
C5_EXPONENT = 5 * C_ATOM_EXPONENT
GROUP_ORDER_EXPONENT = Fraction(5)

R122_PRODUCER = pathlib.Path(
    "p1553_torus_c5_explicit_split_global_rebalance_probe_r122.py"
)
R122_PRODUCER_SHA256 = (
    "053aa87da1e5443965b6141b34c2bc7a71b358d8c7bfd01658f1235726091185"
)
R122_REPORT = pathlib.Path(
    "p1553_torus_c5_explicit_split_global_"
    "rebalance_probe_report_r122.json"
)
R122_REPORT_SHA256 = (
    "e7725ca09099cd1945af08959d381679e3ea905a6b6fd7f5f7085116f0ab949a"
)
R122_FROZEN = pathlib.Path(
    "frozen_torus_c5_explicit_split_global_rebalance.json"
)
R122_FROZEN_SHA256 = (
    "1f74ae158c4ef1070b1d87fe59a6d815a6b09c558d857c6f229cd5e665937de4"
)
R122_COST = pathlib.Path(
    "torus_c5_explicit_split_global_rebalance_cost_ledger.json"
)
R122_COST_SHA256 = (
    "18027809e82ddac7c6d848b638da79b231ff5fc1dbf425766c033bc829cc6720"
)
R122_REPLAY = pathlib.Path(
    "torus_c5_explicit_split_global_rebalance_replay.json"
)
R122_REPLAY_SHA256 = (
    "7e7267fc053e064b654075fcdda408880562e8db03a30e70ee2bdbbbe646f7a3"
)
R122_CONTROLS = pathlib.Path(
    "torus_c5_explicit_split_global_rebalance_controls.json"
)
R122_CONTROLS_SHA256 = (
    "d055d78dcfc6deb9427195bf567ebe0d4c61d3a4e55f5b6468e498fbe3b99a4f"
)
R122_LOGS = pathlib.Path("factor_logs_and_identical_descent_r122.json")
R122_LOGS_SHA256 = (
    "d42ba1a0b0eb9365451eb4fcd95612d93becaab3dedf2ce470f377bf99096ae1"
)
R122_GATE = pathlib.Path(
    "p1553_torus_c5_explicit_split_global_rebalance_probe_gate_r122.md"
)
R122_GATE_SHA256 = (
    "337f0f24c3e6b5ee03f85990c8f8d621157951da75b01f66511917428266008d"
)
R122_PARENT = pathlib.Path(
    "p1553_torus_c5_explicit_split_global_"
    "rebalance_probe_parent_report_r122.yaml"
)
R122_PARENT_SHA256 = (
    "14732beb737e155da02ff823d88df1806ae3893228741ddacf6015f07717b95c"
)
R121_PRODUCER = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_moment_torus_probe_r121.py"
)
R121_PRODUCER_SHA256 = (
    "b590354da291830d9a3372d5d144f1552dee04b2f3e8368258213637c022a6f9"
)
R121_REPORT = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_"
    "moment_torus_probe_report_r121.json"
)
R121_REPORT_SHA256 = (
    "356560169e475900452e83c68a8726420151d57f44735194cb9aaae23a06e54e"
)
R121_GATE = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_moment_torus_probe_gate_r121.md"
)
R121_GATE_SHA256 = (
    "9266e3655a3f4176280834ec91c897cc7d30274382df6e197628af37bae71309"
)
R77_PRODUCER = pathlib.Path(
    "p1553_target_translated_frequency_orbit_probe_r77.py"
)
R77_PRODUCER_SHA256 = (
    "b5e863090aa036504d3c9fd7d47878e370c2c812f7f4e75b9b1d8d0ad972b335"
)
R77_REPORT = pathlib.Path(
    "p1553_target_translated_frequency_orbit_probe_report_r77.json"
)
R77_REPORT_SHA256 = (
    "73f66184fa53a2d43397a915c4249a41c5687cbcd1d5d16cc3e4ff47cf254787"
)
R77_GATE = pathlib.Path(
    "p1553_target_translated_frequency_orbit_probe_gate_r77.md"
)
R77_GATE_SHA256 = (
    "45324f816cc159032cb6c2ac97c0a2f52a618c409e276616521b4e894cb46b55"
)
R90_GATE = pathlib.Path(
    "p1553_5a5c_nonlocal_moment_hankel_translation_probe_gate_r90.md"
)
R90_GATE_SHA256 = (
    "b1618f6a354b995db01fbbc7aeeb69df6ebb5248b5c558bc4c72d87ce523897b"
)
R118_GATE = pathlib.Path(
    "p1553_m6_nonlinear_value_sensitive_c6_source_locator_probe_gate_r118.md"
)
R118_GATE_SHA256 = (
    "9ed7f59fb94dd1ecff54d8844068502183a7928c64e991c9a98b270ab8645b9d"
)
MOROZ_SCHOST_PDF = pathlib.Path(
    "references/moroz_schost_truncated_resultant_1609.04259v1.pdf"
)
MOROZ_SCHOST_PDF_SHA256 = (
    "160c68cfbb413ca27352a064cbf2d27f7ad4ed6a210c3d6ead2770e00204b709"
)
BHARGAVA_MULTIPOINT_PDF = pathlib.Path(
    "references/bhargava_ghosh_guo_kumar_umans_"
    "multipoint_2205.00342v1.pdf"
)
BHARGAVA_MULTIPOINT_PDF_SHA256 = (
    "14eddc304a7dd8995ebc1e24171571fd9dc0f1f837ca35a7f9e2e6fb21bfafa8"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R122 = load_module("p1553_r122_for_r123", R122_PRODUCER)
R121 = R122.R121
R82 = R121.R82
Field = R121.Field

Fp2 = tuple[int, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r122_producer", R122_PRODUCER, R122_PRODUCER_SHA256),
        ("r122_report", R122_REPORT, R122_REPORT_SHA256),
        ("r122_frozen", R122_FROZEN, R122_FROZEN_SHA256),
        ("r122_cost", R122_COST, R122_COST_SHA256),
        ("r122_replay", R122_REPLAY, R122_REPLAY_SHA256),
        ("r122_controls", R122_CONTROLS, R122_CONTROLS_SHA256),
        ("r122_logs", R122_LOGS, R122_LOGS_SHA256),
        ("r122_gate", R122_GATE, R122_GATE_SHA256),
        ("r122_parent", R122_PARENT, R122_PARENT_SHA256),
        ("r121_producer", R121_PRODUCER, R121_PRODUCER_SHA256),
        ("r121_report", R121_REPORT, R121_REPORT_SHA256),
        ("r121_gate", R121_GATE, R121_GATE_SHA256),
        ("r77_producer", R77_PRODUCER, R77_PRODUCER_SHA256),
        ("r77_report", R77_REPORT, R77_REPORT_SHA256),
        ("r77_gate", R77_GATE, R77_GATE_SHA256),
        ("r90_gate", R90_GATE, R90_GATE_SHA256),
        ("r118_gate", R118_GATE, R118_GATE_SHA256),
        ("moroz_schost_pdf", MOROZ_SCHOST_PDF, MOROZ_SCHOST_PDF_SHA256),
        (
            "bhargava_multipoint_pdf",
            BHARGAVA_MULTIPOINT_PDF,
            BHARGAVA_MULTIPOINT_PDF_SHA256,
        ),
    )
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in rows
    }


def verify_source_bindings() -> dict[str, str]:
    bindings = source_binding_records()
    actual = {
        name: sha256_file(pathlib.Path(binding["path"]))
        for name, binding in bindings.items()
    }
    failures = [
        name
        for name, binding in bindings.items()
        if actual[name] != binding["sha256"]
    ]
    if failures:
        raise AssertionError(f"R123 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def field_sum(field: Field, values: Iterable[Fp2]) -> Fp2:
    result = field.zero
    for value in values:
        result = field.add(result, value)
    return result


def canonical_weight(source: tuple[int, ...], prime: int) -> int:
    denominator = 1
    for multiplicity in Counter(source).values():
        denominator *= math.factorial(multiplicity)
    return math.factorial(len(source)) // denominator % prime


def r82_fourier_control(
    curve: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    field, _, deck = R121.pairing_deck(curve, offset)
    sources = tuple(
        itertools.combinations_with_replacement(range(len(deck)), 5)
    )
    products = tuple(
        field.product(deck[index] for index in source)
        for source in sources
    )
    if len(set(products)) != len(products):
        raise AssertionError("R123 controls require injective C5 support")
    weights = tuple(canonical_weight(source, field.p) for source in sources)
    moment_count = 2 * len(products)
    direct_ordered_moments = tuple(
        field.pow(
            field_sum(
                field,
                (field.pow(value, exponent) for value in deck),
            ),
            5,
        )
        for exponent in range(moment_count)
    )
    weighted_support_moments = tuple(
        field_sum(
            field,
            (
                field.mul(
                    field.elt(weight),
                    field.pow(product, exponent),
                )
                for weight, product in zip(weights, products)
            ),
        )
        for exponent in range(moment_count)
    )
    order, recurrence = R121.berlekamp_massey(
        direct_ordered_moments,
        field,
    )
    annihilator = R121.root_annihilator(products, field)
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": curve["subgroup_order"],
        "deck_size": len(deck),
        "ordered_c5_source_count": len(deck) ** 5,
        "canonical_c5_source_count": len(sources),
        "distinct_product_count": len(set(products)),
        "all_ordered_multiplicity_weights_nonzero": all(weights),
        "moment_prefix_length": moment_count,
        "ordered_moment_sha256": sha256_json(
            [field.json(value) for value in direct_ordered_moments]
        ),
        "weighted_support_moment_sha256": sha256_json(
            [field.json(value) for value in weighted_support_moments]
        ),
        "ordered_fourier_moment_identity_exact": (
            direct_ordered_moments == weighted_support_moments
        ),
        "berlekamp_massey_order": order,
        "berlekamp_massey_order_equals_distinct_support": (
            order == len(products)
        ),
        "recurrence_equals_support_annihilator": recurrence == annihilator,
        "candidate_scalar_labels_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def element_of_order(prime: int, order: int) -> int:
    if (prime - 1) % order:
        raise ValueError("order must divide p-1")
    exponent = (prime - 1) // order
    for candidate in range(2, prime):
        value = pow(candidate, exponent, prime)
        if value != 1 and pow(value, order, prime) == 1:
            return value
    raise AssertionError("no subgroup generator found")


def synthetic_fourier_resultant_control() -> dict[str, Any]:
    prime = 353
    order = 11
    generator = element_of_order(prime, order)
    deck = (1, generator)
    subgroup = tuple(pow(generator, exponent, prime) for exponent in range(order))
    ordered_products = tuple(
        math.prod(values) % prime
        for values in itertools.product(deck, repeat=5)
    )
    direct_counts = {
        target: sum(product == target for product in ordered_products)
        for target in subgroup
    }
    inverse_order = pow(order, prime - 2, prime)
    fourier_counts = {}
    for target in subgroup:
        total = 0
        for mode in range(order):
            deck_transform = sum(
                pow(value, mode, prime) for value in deck
            ) % prime
            total += (
                pow(deck_transform, 5, prime)
                * pow(pow(target, prime - 2, prime), mode, prime)
            )
        fourier_counts[target] = total * inverse_order % prime

    pair_roots = tuple(
        left * right % prime for left, right in itertools.product(deck, repeat=2)
    )
    triple_roots = tuple(
        left * middle % prime * right % prime
        for left, middle, right in itertools.product(deck, repeat=3)
    )
    resultant_values = {}
    direct_product_polynomial_values = {}
    for target in subgroup:
        resultant = 1
        for pair in pair_roots:
            for triple in triple_roots:
                resultant = (
                    resultant * (target - pair * triple)
                ) % prime
        direct = 1
        for product in ordered_products:
            direct = direct * (target - product) % prime
        resultant_values[target] = resultant
        direct_product_polynomial_values[target] = direct
    exact_fourier = all(
        fourier_counts[target] == direct_counts[target]
        for target in subgroup
    )
    exact_membership = all(
        (fourier_counts[target] != 0) == (direct_counts[target] != 0)
        for target in subgroup
    )
    exact_resultant = resultant_values == direct_product_polynomial_values
    exact_resultant_membership = all(
        (resultant_values[target] == 0) == (direct_counts[target] != 0)
        for target in subgroup
    )
    return {
        "field_prime": prime,
        "subgroup_order": order,
        "deck_size": len(deck),
        "subgroup_generator": generator,
        "ordered_c5_source_count": len(ordered_products),
        "distinct_c5_product_count": len(set(ordered_products)),
        "field_characteristic_exceeds_total_ordered_source_count": (
            prime > len(ordered_products)
        ),
        "full_q_mode_fourier_counts_equal_direct_counts": exact_fourier,
        "fourier_nonzero_iff_membership": exact_membership,
        "p2_root_count_with_multiplicity": len(pair_roots),
        "p3_root_count_with_multiplicity": len(triple_roots),
        "p2_p3_product_resultant_equals_ordered_p5_evaluation": (
            exact_resultant
        ),
        "resultant_zero_iff_membership": exact_resultant_membership,
        "positive_target_count": sum(
            count != 0 for count in direct_counts.values()
        ),
        "empty_target_count": sum(
            count == 0 for count in direct_counts.values()
        ),
        "candidate_discrete_logs_consumed": False,
        "verifier_exponents_used_only_to_enumerate_subgroup": True,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    actual = [
        r82_fourier_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    synthetic = synthetic_fourier_resultant_control()
    return {
        "schema": (
            "p1553.torus_c5_fourier_product_resultant_controls.r123.v1"
        ),
        "r82_ordered_fourier_controls": actual,
        "r82_control_count": len(actual),
        "all_r82_ordered_fourier_moment_identities_exact": all(
            row["ordered_fourier_moment_identity_exact"] for row in actual
        ),
        "all_r82_bm_orders_equal_distinct_support": all(
            row["berlekamp_massey_order_equals_distinct_support"]
            and row["recurrence_equals_support_annihilator"]
            for row in actual
        ),
        "synthetic_full_fourier_and_resultant_control": synthetic,
        "synthetic_fourier_and_resultant_semantics_exact": (
            synthetic["full_q_mode_fourier_counts_equal_direct_counts"]
            and synthetic["fourier_nonzero_iff_membership"]
            and synthetic[
                "p2_p3_product_resultant_equals_ordered_p5_evaluation"
            ]
            and synthetic["resultant_zero_iff_membership"]
        ),
        "candidate_scalar_labels_or_discrete_logs_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_fourier_product_resultant_cost_ledger.r123.v1"
        ),
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
        },
        "fourier_identity": {
            "ordered_count": (
                "c5(y)=q^(-1)*sum_(j=0..q-1)"
                "(sum_i z_i^j)^5*y^(-j)"
            ),
            "characters_computable_without_dlp": "chi_j(z)=z^j",
            "integer_nonzero_semantics_requires": (
                "characteristic exceeds the charged ordered multiplicity "
                "or a noncancellation proof"
            ),
        },
        "product_resultant_identity": {
            "deck_polynomial": "P1(X)=product_i(X-z_i)",
            "ordered_product_polynomial": (
                "P_(a+b)(Y)=Res_X(P_a(X),X^deg(P_b)*P_b(Y/X))"
            ),
            "membership": "P5(y)=0",
            "source_recovery_from_zero_requires_gcd_roots_and_backpointers": (
                True
            ),
        },
        "routes": [
            {
                "route_id": "full_multiplicative_fourier_inversion",
                "mode_count_exponent_B": fraction_record(
                    GROUP_ORDER_EXPONENT
                ),
                "inside_setup_or_query_cap": False,
            },
            {
                "route_id": "sparse_fourier_prony_bm_reconstruction",
                "linear_complexity_exponent_B": fraction_record(
                    C5_EXPONENT
                ),
                "required_mode_prefix_exponent_B": fraction_record(
                    C5_EXPONENT
                ),
                "inside_setup_cap": False,
                "scope": "linear recurrence / Prony / BM grammar",
            },
            {
                "route_id": "ordered_p3_product_polynomial",
                "degree_and_coefficient_exponent_B": fraction_record(
                    C3_EXPONENT
                ),
                "inside_setup_cap": True,
            },
            {
                "route_id": "target_scaled_p2_p3_fast_resultant",
                "p2_degree_exponent_B": fraction_record(C2_EXPONENT),
                "p3_degree_exponent_B": fraction_record(C3_EXPONENT),
                "optimistic_query_exponent_B": fraction_record(C3_EXPONENT),
                "inside_polylog_query_cap": False,
                "source_recovery_additional": True,
            },
            {
                "route_id": "symbolic_ordered_p5_product_resultant",
                "degree_and_output_exponent_B": fraction_record(C5_EXPONENT),
                "inside_setup_cap": False,
            },
            {
                "route_id": "full_c5_grid_multipoint_evaluation",
                "represented_point_count_exponent_B": fraction_record(
                    C5_EXPONENT
                ),
                "inside_setup_cap": False,
            },
            {
                "route_id": (
                    "target_specialized_nonrepresented_fourier_resultant_"
                    "torus_circuit"
                ),
                "covered_by_scoped_costs": False,
                "exact_circuit_constructed": False,
                "general_lower_bound_claimed": False,
                "status": "open",
            },
        ],
        "semantic_dedup": {
            "nearby_lanes": [
                "R77 additive target-frequency orbit",
                "R90 additive moment/Hankel translation",
                "R118 represented S6 quotient/resultant routes",
                "R121 multiplicative complete-homogeneous moments",
                "R122 all-arity explicit endpoint splits",
            ],
            "r123_distinct_scope": (
                "multiplicative characters computable directly on pairing "
                "images and the ordered product-composition resultant"
            ),
            "new_idea_id_claimed": False,
        },
        "candidate_work_credit": False,
    }


def source_replay(controls: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_fourier_product_resultant_replay.r123.v1"
        ),
        "r122_nonoccurrence_interface_inherited": True,
        "all_r82_ordered_fourier_moment_identities_exact": controls[
            "all_r82_ordered_fourier_moment_identities_exact"
        ],
        "synthetic_full_fourier_and_resultant_semantics_exact": controls[
            "synthetic_fourier_and_resultant_semantics_exact"
        ],
        "inside_cap_target_specialized_membership_constructed": False,
        "inside_cap_five_source_recovery_constructed": False,
        "candidate_scalar_labels_or_discrete_logs_consumed": False,
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R122_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R122 nonclaim boundary drifted")
    controls = finite_controls()
    cost = cost_ledger()
    replay = source_replay(controls)
    obligations = {
        "nineteen_source_bindings_verified": len(source_hashes) == 19,
        "r122_nonoccurrence_torus_interface_inherited": (
            inherited["admission"][
                "explicit_split_rebalance_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "all_eight_r82_ordered_fourier_controls_complete": (
            controls["r82_control_count"] == 8
        ),
        "all_r82_ordered_fourier_moment_identities_exact": controls[
            "all_r82_ordered_fourier_moment_identities_exact"
        ],
        "all_r82_bm_orders_equal_distinct_support": controls[
            "all_r82_bm_orders_equal_distinct_support"
        ],
        "synthetic_full_fourier_semantics_exact": controls[
            "synthetic_full_fourier_and_resultant_control"
        ]["full_q_mode_fourier_counts_equal_direct_counts"],
        "synthetic_product_resultant_semantics_exact": controls[
            "synthetic_full_fourier_and_resultant_control"
        ]["p2_p3_product_resultant_equals_ordered_p5_evaluation"],
        "positive_and_empty_membership_controls_exact": controls[
            "synthetic_fourier_and_resultant_semantics_exact"
        ],
        "fourier_mode_and_recurrence_costs_charged": (
            not cost["candidate_work_credit"]
        ),
        "represented_product_resultant_costs_charged": (
            not cost["candidate_work_credit"]
        ),
        "semantic_dedup_complete": (
            not cost["semantic_dedup"]["new_idea_id_claimed"]
        ),
        "candidate_scalar_labels_and_dlogs_not_consumed": (
            not controls[
                "candidate_scalar_labels_or_discrete_logs_consumed"
            ]
        ),
        "inside_cap_target_specialized_membership_complete": False,
        "inside_cap_five_source_recovery_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Construct or refute one target-specialized nonrepresented torus C5 "
        "circuit that evaluates membership and returns five projective "
        "sources without q Fourier modes, B^(15/4) Prony/BM state, a "
        "B^(9/4) target-scaled P2|P3 resultant, symbolic P5 output, or "
        "explicit endpoint splits. A valid route must expose its arithmetic "
        "circuit or data structure, exact empty certificate, noncancellation "
        "argument, reverse source adjoint, B^(9/4+o(1)) state, "
        "polylogarithmic query, no field DLP, and complete pairing, rank, "
        "logs, identical descent, memory, field-operation, and bit costs."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_fourier_product_resultant.r123.v1"
        ),
        "source_bindings": source_binding_records(),
        "required_interface": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
            "exact_empty_rejection_required": True,
            "five_projective_backpointers_required": True,
            "field_discrete_logarithms_allowed": False,
        },
        "closed_scoped_grammars": [
            "full q-mode multiplicative Fourier inversion",
            "Prony/Berlekamp-Massey reconstruction of full C5 support",
            "represented target-scaled P2|P3 fast resultant",
            "symbolic ordered P5 product resultant",
            "full C5-grid multipoint evaluation",
        ],
        "preserved_interface": (
            "target-specialized nonrepresented Fourier/resultant torus "
            "circuit outside all closed mode, degree, and endpoint bodies"
        ),
        "general_arithmetic_circuit_or_data_structure_lower_bound_claimed": (
            False
        ),
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r123.v1",
        "r122_global_explicit_split_audit_complete": True,
        "r123_fourier_product_resultant_audit_complete": True,
        "inside_cap_target_specialized_source_index_complete": False,
        "relation_independence_theorem_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_log_solve_complete": False,
        "factor_log_verification_complete": False,
        "fresh_target_descent_complete": False,
        "identical_algorithm_used_for_relation_and_descent": False,
        "full_source_to_target_cost_complete": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_MULTIPLICATIVE_FOURIER_AND_PRODUCT_RESULTANT_CONTROLS_"
            "WITH_SCOPED_REPRESENTATION_NEGATIVES_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": (
            "MULTIPLICATIVE_FOURIER_CHARACTERS_Z_TO_ZJ_REQUIRE_NO_DLP__"
            "FULL_INVERSION_Q_MODES_B5__ORDERED_MOMENT_BM_ORDER_EQUALS_"
            "INJECTIVE_C5_SUPPORT_B15O4__ORDERED_P2P3_PRODUCT_RESULTANT_"
            "EXACT__P3_SETUP_B9O4_AND_TARGET_RESULTANT_QUERY_B9O4__"
            "SYMBOLIC_P5_B15O4__TARGET_SPECIALIZED_NONREPRESENTED_FOURIER_"
            "RESULTANT_TORUS_CIRCUIT_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_"
            "BREAKTHROUGH"
        ),
        "source_bindings": source_binding_records(),
        "fourier_identity": cost["fourier_identity"],
        "product_resultant_identity": cost["product_resultant_identity"],
        "finite_evidence": {
            "r82_control_count": controls["r82_control_count"],
            "all_r82_ordered_fourier_moment_identities_exact": controls[
                "all_r82_ordered_fourier_moment_identities_exact"
            ],
            "all_r82_bm_orders_equal_distinct_support": controls[
                "all_r82_bm_orders_equal_distinct_support"
            ],
            "synthetic_fourier_and_resultant_semantics_exact": controls[
                "synthetic_fourier_and_resultant_semantics_exact"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "fourier_and_product_resultant_semantics_admitted": True,
            "scoped_representation_negatives_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": "frozen_torus_c5_fourier_product_resultant.json",
            "cost": "torus_c5_fourier_product_resultant_cost_ledger.json",
            "source_replay": (
                "torus_c5_fourier_product_resultant_replay.json"
            ),
            "controls": "torus_c5_fourier_product_resultant_controls.json",
            "logs_descent": "factor_logs_and_identical_descent_r123.json",
        },
        "next_action": next_action,
        "non_claims": [
            "A compact Fourier identity is not a compact inverse transform.",
            "A compact product resultant identity is not a cheap evaluation.",
            "The BM result is scoped to linear recurrence/Prony grammars.",
            "No general arithmetic-circuit or data-structure lower bound is claimed.",
            "No pairing-image discrete logarithm receives oracle credit.",
            "No known-RHS rank, factor logs, or identical descent is supplied.",
            "No generic-prime ECDLP or Shoup improvement is claimed.",
        ],
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_EXACT_MULTIPLICATIVE_FOURIER_ORDERED_MOMENT_AND_P2P3_"
            "PRODUCT_RESULTANT_SEMANTICS_ONLY__REJECT_FULL_Q_MODE_PRONY_BM_"
            "REPRESENTED_TARGET_RESULTANT_SYMBOLIC_P5_AND_GRID_ROUTES_AT_"
            "FROZEN_CAPS__PRESERVE_TARGET_SPECIALIZED_NONREPRESENTED_"
            "FOURIER_RESULTANT_TORUS_CIRCUIT__NO_LOCATOR__NO_RANK__NO_"
            "LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH"
        ),
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "logs_descent": logs_descent,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "p1553_torus_c5_fourier_product_"
            "resultant_probe_report_r123.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_fourier_product_resultant.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_fourier_product_resultant_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_fourier_product_resultant_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_fourier_product_resultant_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "factor_logs_and_identical_descent_r123.json"
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
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.logs_output, bundle["logs_descent"])
    report = bundle["report"]
    admission = report["admission"]
    print(
        f"R123 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
