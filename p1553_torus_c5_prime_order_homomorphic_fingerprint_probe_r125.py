#!/usr/bin/env python3
"""Audit product-preserving fingerprints for prime-order torus C5."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any, Iterable


SCHEMA = "p1553.torus_c5_prime_order_homomorphic_fingerprint.r125.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
C_ATOM_EXPONENT = Fraction(3, 4)
C2_EXPONENT = 2 * C_ATOM_EXPONENT
C3_EXPONENT = 3 * C_ATOM_EXPONENT
GROUP_ORDER_EXPONENT = Fraction(5)

R124_PRODUCER = pathlib.Path(
    "p1553_torus_c5_linear_sketch_circulant_probe_r124.py"
)
R124_PRODUCER_SHA256 = (
    "c3aabfe9d2ce8553c3facdc7fcf2f8b669b09ec48a6144772068f264e4ab012d"
)
R124_REPORT = pathlib.Path(
    "p1553_torus_c5_linear_sketch_circulant_probe_report_r124.json"
)
R124_REPORT_SHA256 = (
    "99df2561ba0a07506f36c67d7152c299c6954272508eca1035ea3cdbe061dc6d"
)
R124_FROZEN = pathlib.Path(
    "frozen_torus_c5_linear_sketch_circulant.json"
)
R124_FROZEN_SHA256 = (
    "129aac9bf7868c3305ef38d2af4443d2c6e8227f75f54b8b1aa09a1b4a323334"
)
R124_COST = pathlib.Path(
    "torus_c5_linear_sketch_circulant_cost_ledger.json"
)
R124_COST_SHA256 = (
    "b8b5e44c76b7abb85b9f32acf08a1ebc00ce6a5729710975f7eb3362d40b906f"
)
R124_REPLAY = pathlib.Path(
    "torus_c5_linear_sketch_circulant_replay.json"
)
R124_REPLAY_SHA256 = (
    "162d813653f83c782ec69334d6876f1bf6bf2cab9a4d1a9d87e1ff768110e094"
)
R124_CONTROLS = pathlib.Path(
    "torus_c5_linear_sketch_circulant_controls.json"
)
R124_CONTROLS_SHA256 = (
    "87011644cd14261d3f0afd45d367a18a39b484cdfc9b06a91de1e4812e07dfc2"
)
R124_LOGS = pathlib.Path("factor_logs_and_identical_descent_r124.json")
R124_LOGS_SHA256 = (
    "dc170b78fc8460a65709fb3d3537d56571159f68e72c8b9b24c225b3c15b0e7a"
)
R124_GATE = pathlib.Path(
    "p1553_torus_c5_linear_sketch_circulant_probe_gate_r124.md"
)
R124_GATE_SHA256 = (
    "30a3ce3036bab96731430a662cde3ded8e63b9fe1326942a5091148b3e939938"
)
R124_PARENT = pathlib.Path(
    "p1553_torus_c5_linear_sketch_circulant_probe_parent_report_r124.yaml"
)
R124_PARENT_SHA256 = (
    "eb31ae1bccf81431d27bcef8d6fe27ac3269f077bcf115eecab61526bc7a752a"
)
R121_GATE = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_moment_torus_probe_gate_r121.md"
)
R121_GATE_SHA256 = (
    "9266e3655a3f4176280834ec91c897cc7d30274382df6e197628af37bae71309"
)
R120_GATE = pathlib.Path(
    "p1553_m6_suboutput_implicit_c5_character_pairing_probe_gate_r120.md"
)
R120_GATE_SHA256 = (
    "ab8efbe571f310a510f943e45f9fb65bac34fd37b2ee4d04337cdf3a5c9c8da1"
)
R77_GATE = pathlib.Path(
    "p1553_target_translated_frequency_orbit_probe_gate_r77.md"
)
R77_GATE_SHA256 = (
    "45324f816cc159032cb6c2ac97c0a2f52a618c409e276616521b4e894cb46b55"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R124 = load_module("p1553_r124_for_r125", R124_PRODUCER)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r124_producer", R124_PRODUCER, R124_PRODUCER_SHA256),
        ("r124_report", R124_REPORT, R124_REPORT_SHA256),
        ("r124_frozen", R124_FROZEN, R124_FROZEN_SHA256),
        ("r124_cost", R124_COST, R124_COST_SHA256),
        ("r124_replay", R124_REPLAY, R124_REPLAY_SHA256),
        ("r124_controls", R124_CONTROLS, R124_CONTROLS_SHA256),
        ("r124_logs", R124_LOGS, R124_LOGS_SHA256),
        ("r124_gate", R124_GATE, R124_GATE_SHA256),
        ("r124_parent", R124_PARENT, R124_PARENT_SHA256),
        ("r121_gate", R121_GATE, R121_GATE_SHA256),
        ("r120_gate", R120_GATE, R120_GATE_SHA256),
        ("r77_gate", R77_GATE, R77_GATE_SHA256),
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
        raise AssertionError(f"R125 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def five_sum_support(
    order: int,
    deck_support: Iterable[int],
) -> tuple[int, ...]:
    support = tuple(deck_support)
    return tuple(
        sorted(
            {
                sum(source) % order
                for source in itertools.product(support, repeat=5)
            }
        )
    )


def power_map_image_size(order: int, exponent: int) -> int:
    return order // math.gcd(order, exponent)


def power_fingerprint_control(
    order: int,
    deck_support: tuple[int, ...],
) -> dict[str, Any]:
    c5_support = five_sum_support(order, deck_support)
    c5_set = set(c5_support)
    maps = []
    for exponent in range(order):
        image_size = power_map_image_size(order, exponent)
        mapped_support = {
            exponent * value % order for value in c5_support
        }
        predicted = tuple(
            exponent * target % order in mapped_support
            for target in range(order)
        )
        actual = tuple(target in c5_set for target in range(order))
        maps.append(
            {
                "exponent": exponent,
                "kernel_size": math.gcd(order, exponent),
                "image_size": image_size,
                "is_trivial": image_size == 1,
                "is_injective": image_size == order,
                "membership_exact_for_this_deck": predicted == actual,
                "false_positive_count": sum(
                    prediction and not truth
                    for prediction, truth in zip(predicted, actual)
                ),
                "false_negative_count": sum(
                    truth and not prediction
                    for prediction, truth in zip(predicted, actual)
                ),
            }
        )
    return {
        "group_order": order,
        "deck_support": list(deck_support),
        "deck_size": len(deck_support),
        "c5_support_size": len(c5_support),
        "positive_target_count": len(c5_support),
        "empty_target_count": order - len(c5_support),
        "has_positive_and_empty_targets": (
            0 < len(c5_support) < order
        ),
        "maps": maps,
        "intermediate_image_size_count": sum(
            1 < row["image_size"] < order for row in maps
        ),
        "all_nontrivial_maps_injective": all(
            row["is_injective"] for row in maps if not row["is_trivial"]
        ),
        "all_injective_maps_exact": all(
            row["membership_exact_for_this_deck"]
            for row in maps
            if row["is_injective"]
        ),
        "all_trivial_maps_fail_empty_targets": all(
            not row["membership_exact_for_this_deck"]
            and row["false_positive_count"] > 0
            and row["false_negative_count"] == 0
            for row in maps
            if row["is_trivial"]
        ),
        "candidate_discrete_logs_consumed": False,
        "exponent_labels_used_for_verifier_only": True,
        "finite_control_receives_asymptotic_credit": False,
        "map_table_sha256": sha256_json(maps),
    }


def tuple_fingerprint_control(
    order: int,
    deck_support: tuple[int, ...],
) -> dict[str, Any]:
    c5_support = set(five_sum_support(order, deck_support))
    families = ((0,), (1,), (2,), (0, 3), (0, 0), (2, 5))
    rows = []
    for exponents in families:
        images = {
            tuple(exponent * value % order for exponent in exponents)
            for value in range(order)
        }
        mapped_support = {
            tuple(exponent * value % order for exponent in exponents)
            for value in c5_support
        }
        predicted = tuple(
            tuple(
                exponent * target % order for exponent in exponents
            )
            in mapped_support
            for target in range(order)
        )
        actual = tuple(target in c5_support for target in range(order))
        rows.append(
            {
                "exponents": list(exponents),
                "image_size": len(images),
                "all_components_trivial": all(
                    exponent % order == 0 for exponent in exponents
                ),
                "membership_exact_for_this_deck": predicted == actual,
            }
        )
    return {
        "group_order": order,
        "families": rows,
        "all_tuple_images_are_one_or_q": all(
            row["image_size"] in (1, order) for row in rows
        ),
        "all_nontrivial_tuples_injective_and_exact": all(
            row["image_size"] == order
            and row["membership_exact_for_this_deck"]
            for row in rows
            if not row["all_components_trivial"]
        ),
        "all_trivial_tuples_inexact": all(
            not row["membership_exact_for_this_deck"]
            for row in rows
            if row["all_components_trivial"]
        ),
    }


def finite_controls() -> dict[str, Any]:
    prime_cases = (
        (5, (1,)),
        (7, (0, 1)),
        (11, (0, 1)),
        (13, (0, 1)),
    )
    composite_cases = (
        (6, (1,)),
        (8, (1,)),
        (12, (1,)),
    )
    prime_controls = [
        power_fingerprint_control(order, support)
        for order, support in prime_cases
    ]
    composite_controls = [
        power_fingerprint_control(order, support)
        for order, support in composite_cases
    ]
    tuple_control = tuple_fingerprint_control(11, (0, 1))
    return {
        "schema": (
            "p1553.torus_c5_prime_order_homomorphic_fingerprint_"
            "controls.r125.v1"
        ),
        "prime_controls": prime_controls,
        "composite_controls": composite_controls,
        "tuple_control": tuple_control,
        "prime_control_count": len(prime_controls),
        "composite_control_count": len(composite_controls),
        "all_prime_controls_have_positive_and_empty_targets": all(
            row["has_positive_and_empty_targets"]
            for row in prime_controls
        ),
        "all_prime_nontrivial_power_maps_injective_and_exact": all(
            row["all_nontrivial_maps_injective"]
            and row["all_injective_maps_exact"]
            for row in prime_controls
        ),
        "all_prime_trivial_power_maps_fail_empty_targets": all(
            row["all_trivial_maps_fail_empty_targets"]
            for row in prime_controls
        ),
        "all_prime_intermediate_image_counts_zero": all(
            row["intermediate_image_size_count"] == 0
            for row in prime_controls
        ),
        "all_composite_controls_have_intermediate_images": all(
            row["intermediate_image_size_count"] > 0
            for row in composite_controls
        ),
        "tuple_dichotomy_exact": (
            tuple_control["all_tuple_images_are_one_or_q"]
            and tuple_control[
                "all_nontrivial_tuples_injective_and_exact"
            ]
            and tuple_control["all_trivial_tuples_inexact"]
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "model": (
            "Product-preserving fingerprints phi_i:mu_q->H_i are group "
            "homomorphisms used to map C2, C3, and the arbitrary target "
            "before a compressed convolution or membership test."
        ),
        "prime_order_kernel_dichotomy": (
            "For prime q, ker(phi) is either mu_q or {1}. Hence phi is "
            "trivial or injective."
        ),
        "power_map_realization": (
            "Every endomorphism of mu_q is z->z^k. It is directly "
            "computable without field DLP; k=0 mod q is constant and every "
            "other k is a permutation."
        ),
        "multiple_map_dichotomy": (
            "The product fingerprint (phi_1,...,phi_t) has kernel equal "
            "to the intersection of component kernels. It is trivial when "
            "all components are trivial and injective otherwise."
        ),
        "exact_membership_consequence": (
            "A trivial fingerprint maps every target to the occupied image "
            "whenever C5 is nonempty, so it gives false positives on empty "
            "targets. An exact pure homomorphic fingerprint must therefore "
            "be injective and have image cardinality at least q."
        ),
        "minimum_exact_image_cardinality": "q",
        "image_exponent_B": fraction_record(GROUP_ORDER_EXPONENT),
        "inside_setup_cap_for_full_image_table": False,
        "scope_limits": [
            "pure product-preserving group homomorphisms",
            "single or finite tuples of homomorphisms",
            "exact arbitrary-target membership with positive and empty targets",
        ],
        "not_covered": [
            "nonhomomorphic fingerprints with separately proved correction data",
            "adaptive or target-dependent probes",
            "nonlinear rational-Krylov circuits",
            "deck-specific perfect hashing unrelated to group multiplication",
            "general arithmetic circuits or data structures",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_prime_order_homomorphic_fingerprint_"
            "cost_ledger.r125.v1"
        ),
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
        },
        "theorem": theorem,
        "routes": [
            {
                "route_id": "trivial_homomorphic_fingerprint",
                "image_exponent_B": fraction_record(Fraction(0)),
                "exact_empty_rejection": False,
                "admitted": False,
            },
            {
                "route_id": "nontrivial_prime_order_homomorphic_fingerprint",
                "image_exponent_B": fraction_record(
                    GROUP_ORDER_EXPONENT
                ),
                "injective": True,
                "inside_full_table_setup_cap": False,
            },
            {
                "route_id": "injective_homomorphic_C3_hash_C2_scan",
                "setup_exponent_B": fraction_record(C3_EXPONENT),
                "query_exponent_B": fraction_record(C2_EXPONENT),
                "inside_setup_cap": True,
                "inside_polylog_query_cap": False,
            },
            {
                "route_id": "finite_tuple_of_homomorphic_fingerprints",
                "minimum_exact_combined_image_exponent_B": fraction_record(
                    GROUP_ORDER_EXPONENT
                ),
                "inside_full_table_setup_cap": False,
            },
            {
                "route_id": "nonhomomorphic_or_adaptive_fingerprint",
                "scoped_lower_bound_proved": False,
                "exact_structure_constructed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R124_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R124 nonclaim boundary drifted")
    controls = finite_controls()
    theorem = theorem_record()
    cost = cost_ledger(theorem)
    obligations = {
        "twelve_source_bindings_verified": len(source_hashes) == 12,
        "r124_coupled_nonlinear_interface_inherited": (
            inherited["admission"][
                "scoped_universal_linear_sketch_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "four_prime_controls_complete": (
            controls["prime_control_count"] == 4
        ),
        "prime_controls_have_positive_and_empty_targets": controls[
            "all_prime_controls_have_positive_and_empty_targets"
        ],
        "prime_nontrivial_power_maps_injective_and_exact": controls[
            "all_prime_nontrivial_power_maps_injective_and_exact"
        ],
        "prime_trivial_power_maps_fail_empty_targets": controls[
            "all_prime_trivial_power_maps_fail_empty_targets"
        ],
        "prime_intermediate_images_absent": controls[
            "all_prime_intermediate_image_counts_zero"
        ],
        "composite_specificity_controls_have_quotients": controls[
            "all_composite_controls_have_intermediate_images"
        ],
        "multiple_homomorphism_dichotomy_exact": controls[
            "tuple_dichotomy_exact"
        ],
        "prime_order_kernel_theorem_explicit": (
            "trivial or injective"
            in theorem["prime_order_kernel_dichotomy"]
        ),
        "B5_exact_image_cost_charged": (
            theorem["image_exponent_B"]
            == fraction_record(GROUP_ORDER_EXPONENT)
            and not theorem["inside_setup_cap_for_full_image_table"]
        ),
        "candidate_dlogs_not_consumed": (
            not controls["candidate_discrete_logs_consumed"]
        ),
        "inside_cap_nonhomomorphic_membership_complete": False,
        "inside_cap_five_source_recovery_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    failures = [name for name, value in obligations.items() if not value]
    next_action = (
        "Freeze one nonhomomorphic or adaptive target fingerprint for the "
        "coupled pair (u^(*2),u^(*3)). Require explicit correction data for "
        "every product-law failure, deterministic exact empty semantics, "
        "five projective source backpointers, B^(9/4+o(1)) total setup, "
        "polylogarithmic arbitrary-target work, no field DLP, and complete "
        "rank, logs, identical descent, memory, field-operation, and bit "
        "costs."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_prime_order_homomorphic_"
            "fingerprint.r125.v1"
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
        "closed_scoped_grammar": (
            "single or finite tuples of pure product-preserving "
            "homomorphisms from the prime-order pairing image"
        ),
        "preserved_interface": (
            "nonhomomorphic or adaptive target fingerprint with fully "
            "charged product-law correction data"
        ),
        "general_data_structure_or_arithmetic_circuit_lower_bound_claimed": (
            False
        ),
    }
    replay = {
        "schema": (
            "p1553.torus_c5_prime_order_homomorphic_fingerprint_"
            "replay.r125.v1"
        ),
        "prime_kernel_dichotomy_controls_exact": (
            controls["all_prime_intermediate_image_counts_zero"]
        ),
        "trivial_maps_fail_empty_controls": controls[
            "all_prime_trivial_power_maps_fail_empty_targets"
        ],
        "injective_maps_preserve_membership_controls": controls[
            "all_prime_nontrivial_power_maps_injective_and_exact"
        ],
        "composite_order_specificity_controls_exact": controls[
            "all_composite_controls_have_intermediate_images"
        ],
        "inside_cap_nonhomomorphic_membership_constructed": False,
        "inside_cap_five_source_recovery_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r125.v1",
        "r124_linear_sketch_audit_complete": True,
        "r125_prime_order_homomorphic_fingerprint_audit_complete": True,
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
    classification = (
        "PRIME_ORDER_TORUS_HOMOMORPHIC_FINGERPRINT_KERNEL_TRIVIAL_OR_"
        "INJECTIVE__TRIVIAL_MAP_FAILS_EMPTY_TARGETS__NONTRIVIAL_POWER_MAP_"
        "IS_DLP_FREE_PERMUTATION_WITH_IMAGE_Q_B5__FINITE_TUPLES_RETAIN_"
        "DICHOTOMY__COMPOSITE_CONTROLS_HAVE_PROPER_QUOTIENTS__"
        "NONHOMOMORPHIC_ADAPTIVE_FINGERPRINT_OPEN__NO_SOURCE_RANK_LOGS_"
        "DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_PRIME_ORDER_HOMOMORPHIC_FINGERPRINT_DICHOTOMY_AND_"
            "SCOPED_NEGATIVE_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "prime_control_count": controls["prime_control_count"],
            "composite_control_count": controls[
                "composite_control_count"
            ],
            "prime_power_map_dichotomy_exact": (
                controls[
                    "all_prime_nontrivial_power_maps_injective_and_exact"
                ]
                and controls[
                    "all_prime_trivial_power_maps_fail_empty_targets"
                ]
            ),
            "multiple_homomorphism_dichotomy_exact": controls[
                "tuple_dichotomy_exact"
            ],
            "composite_specificity_exact": controls[
                "all_composite_controls_have_intermediate_images"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "prime_order_homomorphism_dichotomy_admitted": True,
            "scoped_homomorphic_fingerprint_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_prime_order_homomorphic_fingerprint.json"
            ),
            "cost": (
                "torus_c5_prime_order_homomorphic_fingerprint_"
                "cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_prime_order_homomorphic_fingerprint_replay.json"
            ),
            "controls": (
                "torus_c5_prime_order_homomorphic_fingerprint_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r125.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The theorem covers only product-preserving homomorphisms.",
            "It does not cover nonhomomorphic fingerprints plus corrections.",
            "Finite controls receive no asymptotic credit.",
            "No five-source locator, rank, logs, or descent is supplied.",
            "No generic-prime ECDLP or Shoup improvement is claimed.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_PRIME_ORDER_HOMOMORPHIC_FINGERPRINT_TRIVIAL_OR_INJECTIVE_"
            "DICHOTOMY_ONLY__REJECT_TRIVIAL_MAP_FOR_EMPTY_TARGETS_AND_"
            "NONTRIVIAL_MAP_AT_Q_B5_IMAGE__PRESERVE_NONHOMOMORPHIC_ADAPTIVE_"
            "FINGERPRINT_WITH_CHARGED_CORRECTIONS__NO_LOCATOR__NO_RANK__NO_"
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
            "p1553_torus_c5_prime_order_homomorphic_"
            "fingerprint_probe_report_r125.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_prime_order_homomorphic_fingerprint.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_prime_order_homomorphic_fingerprint_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_prime_order_homomorphic_fingerprint_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_prime_order_homomorphic_fingerprint_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r125.json"),
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
        f"R125 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
