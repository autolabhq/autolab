#!/usr/bin/env python3
"""Audit bucket-resultant routing tradeoffs for torus C5 correction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pathlib
from collections import defaultdict
from fractions import Fraction
from typing import Any


SCHEMA = "p1553.torus_c5_bucket_resultant_routing_tradeoff.r127.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
C2_EXPONENT = Fraction(3, 2)
C3_EXPONENT = Fraction(9, 4)
C5_EXPONENT = Fraction(15, 4)

R126_PRODUCER = pathlib.Path(
    "p1553_torus_c5_explicit_hash_correction_support_probe_r126.py"
)
R126_PRODUCER_SHA256 = (
    "6540a8847cc02206879e217a00b342fafedbe87a31ecac80e24702c3b5a91f64"
)
R126_REPORT = pathlib.Path(
    "p1553_torus_c5_explicit_hash_correction_"
    "support_probe_report_r126.json"
)
R126_REPORT_SHA256 = (
    "d7af982e0a35bded47611f2e2d50d2a602a26b6e86b8af94f5781963593c5dee"
)
R126_FROZEN = pathlib.Path(
    "frozen_torus_c5_explicit_hash_correction_support.json"
)
R126_FROZEN_SHA256 = (
    "0c5455ab77affc5250ef21184b73035db93d519447deb5911657e6d3ba2fdcbd"
)
R126_COST = pathlib.Path(
    "torus_c5_explicit_hash_correction_support_cost_ledger.json"
)
R126_COST_SHA256 = (
    "b4a63f593f162eb47d9ce0bb27db10f336993e8e7b1679d0690bf9ab9b2e3de5"
)
R126_REPLAY = pathlib.Path(
    "torus_c5_explicit_hash_correction_support_replay.json"
)
R126_REPLAY_SHA256 = (
    "af4e1d8ad163de0c9919dee4a29510e307747b95ae553b413dbd4d9aea1cbc23"
)
R126_CONTROLS = pathlib.Path(
    "torus_c5_explicit_hash_correction_support_controls.json"
)
R126_CONTROLS_SHA256 = (
    "f1dc25662a9707ab08f1b90f05c280610bb7659a8af2d1565182c8159bcfb132"
)
R126_LOGS = pathlib.Path("factor_logs_and_identical_descent_r126.json")
R126_LOGS_SHA256 = (
    "de11144ba29fa6728ddd30843fc5de128c974cc4be25a67ae964dcffd4210da2"
)
R126_GATE = pathlib.Path(
    "p1553_torus_c5_explicit_hash_correction_support_probe_gate_r126.md"
)
R126_GATE_SHA256 = (
    "31de425cd111ddc6d06d33b4e81818c964932be4d18a80fb154164c62e8bab5b"
)
R126_PARENT = pathlib.Path(
    "p1553_torus_c5_explicit_hash_correction_"
    "support_probe_parent_report_r126.yaml"
)
R126_PARENT_SHA256 = (
    "3dd1924eac95dd48cd2ab78a5a6edec5887c960a86c6770a9a34b8850ea3dff7"
)
R123_GATE = pathlib.Path(
    "p1553_torus_c5_fourier_product_resultant_probe_gate_r123.md"
)
R123_GATE_SHA256 = (
    "1b0711d92aeee1a1a4bb2208cdbc772d8a07e1dff14b423803e75f1bd859733f"
)
R121_GATE = pathlib.Path(
    "p1553_m6_small_k_multiplicative_c5_moment_torus_probe_gate_r121.md"
)
R121_GATE_SHA256 = (
    "9266e3655a3f4176280834ec91c897cc7d30274382df6e197628af37bae71309"
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


R126 = load_module("p1553_r126_for_r127", R126_PRODUCER)
R121 = R126.R121
R82 = R121.R82
Field = R121.Field
Fp2 = tuple[int, int]


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
        ("r126_producer", R126_PRODUCER, R126_PRODUCER_SHA256),
        ("r126_report", R126_REPORT, R126_REPORT_SHA256),
        ("r126_frozen", R126_FROZEN, R126_FROZEN_SHA256),
        ("r126_cost", R126_COST, R126_COST_SHA256),
        ("r126_replay", R126_REPLAY, R126_REPLAY_SHA256),
        ("r126_controls", R126_CONTROLS, R126_CONTROLS_SHA256),
        ("r126_logs", R126_LOGS, R126_LOGS_SHA256),
        ("r126_gate", R126_GATE, R126_GATE_SHA256),
        ("r126_parent", R126_PARENT, R126_PARENT_SHA256),
        ("r123_gate", R123_GATE, R123_GATE_SHA256),
        ("r121_gate", R121_GATE, R121_GATE_SHA256),
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
        raise AssertionError(f"R127 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def find_empty_subgroup_target(
    field: Field,
    deck: tuple[Fp2, ...],
    c5_products: set[Fp2],
) -> Fp2:
    generator = next(value for value in deck if value != field.one)
    value = field.one
    for _ in range(len(c5_products) + 2):
        if value not in c5_products:
            return value
        value = field.mul(value, generator)
    raise AssertionError("failed to find finite empty target")


def bucket_resultant_control(
    curve: dict[str, Any],
    offset: int,
    hash_modulus: int,
    salt: int,
) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    deck = tuple(deck_values)
    c2 = R126.source_products(field, deck, 2)
    c3 = R126.source_products(field, deck, 3)
    c5 = R126.source_products(field, deck, 5)
    c5_products = {product for _, product in c5}
    c2_buckets: dict[int, list[tuple[tuple[int, ...], Fp2]]] = (
        defaultdict(list)
    )
    c3_buckets: dict[int, list[tuple[tuple[int, ...], Fp2]]] = (
        defaultdict(list)
    )
    for row in c2:
        c2_buckets[
            R126.coordinate_hash(row[1], hash_modulus, salt)
        ].append(row)
    for row in c3:
        c3_buckets[
            R126.coordinate_hash(row[1], hash_modulus, salt)
        ].append(row)

    pair_products: dict[
        tuple[int, int],
        dict[Fp2, tuple[tuple[int, ...], tuple[int, ...]]],
    ] = {}
    target_bucket_pairs: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for left_bucket, left_rows in c2_buckets.items():
        for right_bucket, right_rows in c3_buckets.items():
            bucket_pair = (left_bucket, right_bucket)
            products: dict[
                Fp2,
                tuple[tuple[int, ...], tuple[int, ...]],
            ] = {}
            for left_source, left_value in left_rows:
                for right_source, right_value in right_rows:
                    product = field.mul(left_value, right_value)
                    products.setdefault(
                        product,
                        (left_source, right_source),
                    )
                    target_bucket_pairs[
                        R126.coordinate_hash(
                            product,
                            hash_modulus,
                            salt,
                        )
                    ].add(bucket_pair)
            pair_products[bucket_pair] = products

    empty_target = find_empty_subgroup_target(field, deck, c5_products)
    queries = tuple(sorted(c5_products)) + (empty_target,)
    all_pair_exact = True
    routed_exact = True
    source_replay_exact = True
    routed_pair_counts = []
    routed_degree_work = []
    for target in queries:
        direct = target in c5_products
        all_pair_hits = [
            (bucket_pair, products[target])
            for bucket_pair, products in pair_products.items()
            if target in products
        ]
        all_pair_exact &= bool(all_pair_hits) == direct
        target_bucket = R126.coordinate_hash(
            target,
            hash_modulus,
            salt,
        )
        routed_pairs = target_bucket_pairs[target_bucket]
        routed_hits = [
            (bucket_pair, pair_products[bucket_pair][target])
            for bucket_pair in routed_pairs
            if target in pair_products[bucket_pair]
        ]
        routed_exact &= bool(routed_hits) == direct
        routed_pair_counts.append(len(routed_pairs))
        routed_degree_work.append(
            sum(
                max(
                    len(c2_buckets[bucket_pair[0]]),
                    len(c3_buckets[bucket_pair[1]]),
                )
                for bucket_pair in routed_pairs
            )
        )
        for _, sources in routed_hits:
            merged = tuple(sorted(sources[0] + sources[1]))
            source_replay_exact &= (
                field.product(deck[index] for index in merged) == target
            )
    all_pair_degree_work = sum(
        max(
            len(c2_buckets[bucket_pair[0]]),
            len(c3_buckets[bucket_pair[1]]),
        )
        for bucket_pair in pair_products
    )
    symbolic_output_degree = sum(
        len(c2_buckets[bucket_pair[0]])
        * len(c3_buckets[bucket_pair[1]])
        for bucket_pair in pair_products
    )
    tensor_edges = sum(
        len(pairs) for pairs in target_bucket_pairs.values()
    )
    return {
        "control_id": (
            f"{curve['family_id']}_offset{offset}_"
            f"m{hash_modulus}_s{salt}"
        ),
        "field_prime": field.p,
        "subgroup_order": curve["subgroup_order"],
        "deck_size": len(deck),
        "hash_modulus": hash_modulus,
        "hash_salt": salt,
        "candidate_discrete_logs_consumed": False,
        "occupied_c2_bucket_count": len(c2_buckets),
        "occupied_c3_bucket_count": len(c3_buckets),
        "occupied_bucket_pair_count": len(pair_products),
        "target_bucket_routing_tensor_edge_count": tensor_edges,
        "maximum_target_bucket_routed_pair_count": max(
            len(pairs) for pairs in target_bucket_pairs.values()
        ),
        "minimum_target_bucket_routed_pair_count": min(
            len(pairs) for pairs in target_bucket_pairs.values()
        ),
        "all_pair_resultant_membership_exact": all_pair_exact,
        "target_bucket_routed_resultant_membership_exact": routed_exact,
        "all_positive_routed_sources_replay": source_replay_exact,
        "all_pair_optimistic_degree_work": all_pair_degree_work,
        "maximum_routed_optimistic_degree_work": max(
            routed_degree_work
        ),
        "minimum_routed_optimistic_degree_work": min(
            routed_degree_work
        ),
        "symbolic_bucket_resultant_total_output_degree": (
            symbolic_output_degree
        ),
        "symbolic_output_degree_equals_c2_times_c3": (
            symbolic_output_degree == len(c2) * len(c3)
        ),
        "positive_query_count": len(c5_products),
        "empty_query_count": 1,
        "routing_profile_sha256": sha256_json(
            {
                str(target_bucket): sorted([list(pair) for pair in pairs])
                for target_bucket, pairs in sorted(
                    target_bucket_pairs.items()
                )
            }
        ),
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        bucket_resultant_control(
            curve,
            offset,
            hash_modulus,
            salt,
        )
        for curve in R82.FAMILIES
        for offset in (0, 1)
        for hash_modulus, salt in ((2, 1), (3, 2), (5, 3))
    ]
    return {
        "schema": (
            "p1553.torus_c5_bucket_resultant_routing_tradeoff_"
            "controls.r127.v1"
        ),
        "controls": controls,
        "control_count": len(controls),
        "all_pair_resultant_membership_exact": all(
            row["all_pair_resultant_membership_exact"]
            for row in controls
        ),
        "all_target_bucket_routed_membership_exact": all(
            row["target_bucket_routed_resultant_membership_exact"]
            for row in controls
        ),
        "all_positive_routed_sources_replay": all(
            row["all_positive_routed_sources_replay"]
            for row in controls
        ),
        "all_symbolic_degrees_equal_c2_times_c3": all(
            row["symbolic_output_degree_equals_c2_times_c3"]
            for row in controls
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
    }


def tradeoff_record() -> dict[str, Any]:
    return {
        "balanced_bucket_parameter": "H=B^gamma",
        "balanced_degrees": {
            "C2_per_bucket": "B^(3/2-gamma)",
            "C3_per_bucket": "B^(9/4-gamma)",
        },
        "optimistic_one_pair_resultant_work": "B^(9/4-gamma)",
        "all_H2_pairs_query_work": "B^(9/4+gamma)",
        "R_routed_pairs_parameter": "R=B^rho",
        "R_pair_query_work": "B^(rho+9/4-gamma)",
        "quotient_or_latin_slice_R_equals_H": {
            "rho": "gamma",
            "query_work": "B^(9/4)",
            "inside_polylog_query_cap": False,
        },
        "polylog_query_necessary_inequality": "rho<=gamma-9/4",
        "nonnegative_rho_and_nonempty_C3_bucket_consequence": (
            "With 0<=gamma<=9/4 and rho>=0, polylog query requires "
            "gamma=9/4 and rho=0."
        ),
        "extreme_survivor": (
            "H=B^(9/4) cap-tight C3 singleton buckets plus an implicit "
            "O(1)-pair arbitrary-target router and source locator."
        ),
        "extreme_survivor_constructed": False,
        "represented_symbolic_bucket_resultants": {
            "total_output_degree": "B^(15/4)",
            "inside_setup_cap": False,
        },
        "represented_dense_target_bucket_tensor": {
            "state_exponent_B": "3*gamma",
            "inside_setup_requires": "gamma<=3/4",
            "does_not_supply_polylog_resultant_query": True,
        },
        "scope_limits": [
            "balanced bucket polynomial/resultant grammars",
            "optimistic quasi-linear work in the larger represented degree",
            "explicit target-bucket routing tensors",
            "independent evaluation of routed bucket-pair resultants",
        ],
        "not_covered": [
            "shared transposed evaluation across bucket resultants",
            "implicit O(1)-pair routing at gamma=9/4",
            "adaptive cell-probe structures",
            "nonlinear non-resultant correction certificates",
            "general arithmetic circuits or data structures",
        ],
    }


def cost_ledger(tradeoff: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_bucket_resultant_routing_tradeoff_"
            "cost_ledger.r127.v1"
        ),
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
        },
        "tradeoff": tradeoff,
        "routes": [
            {
                "route_id": "all_bucket_pair_resultants",
                "query_exponent_B": "9/4+gamma",
                "inside_polylog_query_cap": False,
            },
            {
                "route_id": "quotient_style_H_routed_pair_resultants",
                "query_exponent_B": "9/4",
                "inside_polylog_query_cap": False,
            },
            {
                "route_id": "represented_symbolic_bucket_resultants",
                "setup_output_exponent_B": fraction_record(C5_EXPONENT),
                "inside_setup_cap": False,
            },
            {
                "route_id": "extreme_implicit_singleton_C3_router",
                "bucket_exponent_gamma": fraction_record(C3_EXPONENT),
                "routed_pair_exponent_rho": fraction_record(Fraction(0)),
                "inside_setup_and_query_caps_if_constructed": True,
                "exact_router_constructed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R126_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R126 nonclaim boundary drifted")
    controls = finite_controls()
    tradeoff = tradeoff_record()
    cost = cost_ledger(tradeoff)
    obligations = {
        "thirteen_source_bindings_verified": len(source_hashes) == 13,
        "r126_implicit_correction_interface_inherited": (
            inherited["admission"][
                "scoped_explicit_correction_support_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "twenty_four_bucket_resultant_controls_complete": (
            controls["control_count"] == 24
        ),
        "all_pair_resultant_membership_exact": controls[
            "all_pair_resultant_membership_exact"
        ],
        "target_bucket_routed_membership_exact": controls[
            "all_target_bucket_routed_membership_exact"
        ],
        "all_positive_routed_sources_replay": controls[
            "all_positive_routed_sources_replay"
        ],
        "symbolic_degrees_equal_c2_times_c3": controls[
            "all_symbolic_degrees_equal_c2_times_c3"
        ],
        "candidate_dlogs_not_consumed": (
            not controls["candidate_discrete_logs_consumed"]
        ),
        "all_pair_tradeoff_charged": (
            tradeoff["all_H2_pairs_query_work"] == "B^(9/4+gamma)"
        ),
        "quotient_style_tradeoff_charged": (
            tradeoff["quotient_or_latin_slice_R_equals_H"][
                "query_work"
            ]
            == "B^(9/4)"
        ),
        "polylog_necessary_inequality_explicit": (
            tradeoff["polylog_query_necessary_inequality"]
            == "rho<=gamma-9/4"
        ),
        "extreme_cap_tight_survivor_preserved": (
            "H=B^(9/4)" in tradeoff["extreme_survivor"]
            and not tradeoff["extreme_survivor_constructed"]
        ),
        "inside_cap_extreme_router_complete": False,
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
        "Construct or refute the isolated extreme survivor: a "
        "B^(9/4+o(1))-state C3 singleton-bucket index with an implicit "
        "O(1)-pair arbitrary-target router that returns the matching C2 "
        "and C3 sources without field DLP. It may share transposed work "
        "across bucket resultants, but must expose every routing cell, exact "
        "empty certificate, source adjoint, rank, logs, identical descent, "
        "memory, field operation, and bit cost."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_bucket_resultant_routing_"
            "tradeoff.r127.v1"
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
            "independent evaluation of all H^2 bucket-pair resultants",
            "quotient-style H-pair target routing",
            "represented symbolic bucket resultants",
            "represented dense target-bucket routing tensors",
        ],
        "preserved_interface": tradeoff["extreme_survivor"],
        "general_data_structure_or_arithmetic_circuit_lower_bound_claimed": (
            False
        ),
    }
    replay = {
        "schema": (
            "p1553.torus_c5_bucket_resultant_routing_tradeoff_"
            "replay.r127.v1"
        ),
        "actual_control_count": controls["control_count"],
        "all_pair_semantics_exact": controls[
            "all_pair_resultant_membership_exact"
        ],
        "target_bucket_routing_semantics_exact": controls[
            "all_target_bucket_routed_membership_exact"
        ],
        "source_backpointers_exact": controls[
            "all_positive_routed_sources_replay"
        ],
        "inside_cap_extreme_router_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r127.v1",
        "r126_explicit_correction_audit_complete": True,
        "r127_bucket_resultant_tradeoff_audit_complete": True,
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
        "BALANCED_H_BGAMMA_BUCKET_RESULTANT_PAIR_WORK_B9O4_MINUS_GAMMA__"
        "ALL_H2_QUERY_B9O4_PLUS_GAMMA__QUOTIENT_H_PAIR_QUERY_B9O4__"
        "POLYLOG_REQUIRES_GAMMA9O4_RHO0_EXTREME__SYMBOLIC_OUTPUT_B15O4__"
        "ACTUAL_COORDINATE_HASH_ROUTING_AND_SOURCE_CONTROLS_EXACT__"
        "CAP_TIGHT_C3_SINGLETON_IMPLICIT_O1_ROUTER_OPEN__NO_RANK_LOGS_"
        "DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "EXACT_BUCKET_ROUTING_CONTROLS_AND_SCOPED_RESULTANT_TRADEOFF_"
            "ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "tradeoff": tradeoff,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "all_pair_resultant_membership_exact": controls[
                "all_pair_resultant_membership_exact"
            ],
            "target_bucket_routed_membership_exact": controls[
                "all_target_bucket_routed_membership_exact"
            ],
            "all_positive_routed_sources_replay": controls[
                "all_positive_routed_sources_replay"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "failures": failures,
            "bucket_resultant_routing_semantics_admitted": True,
            "scoped_resultant_tradeoff_negative_admitted": True,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_bucket_resultant_routing_tradeoff.json"
            ),
            "cost": (
                "torus_c5_bucket_resultant_routing_tradeoff_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_bucket_resultant_routing_tradeoff_replay.json"
            ),
            "controls": (
                "torus_c5_bucket_resultant_routing_tradeoff_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r127.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The tradeoff covers independent represented resultants.",
            "It does not cover shared transposed or adaptive evaluation.",
            "The cap-tight singleton-C3 router is not constructed.",
            "Finite controls receive no asymptotic credit.",
            "No complete rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_ACTUAL_BUCKET_RESULTANT_ROUTING_AND_SOURCE_SEMANTICS_"
            "ONLY__REJECT_ALL_PAIR_QUOTIENT_STYLE_SYMBOLIC_AND_DENSE_TENSOR_"
            "GRAMMARS_AT_FROZEN_CAPS__PRESERVE_GAMMA9O4_RHO0_IMPLICIT_"
            "SINGLETON_C3_ROUTER__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__"
            "NO_SHOUP__NO_BREAKTHROUGH"
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
            "p1553_torus_c5_bucket_resultant_routing_"
            "tradeoff_probe_report_r127.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_bucket_resultant_routing_tradeoff.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_bucket_resultant_routing_tradeoff_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_bucket_resultant_routing_tradeoff_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_bucket_resultant_routing_tradeoff_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r127.json"),
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
        f"R127 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
