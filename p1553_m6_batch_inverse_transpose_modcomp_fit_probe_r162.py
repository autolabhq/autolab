#!/usr/bin/env python3
"""Probe the shareable linear layer in the R161 signed-divisor target batch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_batch_inverse_transpose_modcomp_fit.r162.v1"

R161_PRODUCER = ROOT / "p1553_m6_signed_c3_divisor_translation_gcd_probe_r161.py"
R161_REPORT = ROOT / "p1553_m6_signed_c3_divisor_translation_gcd_probe_report_r161.json"
R161_FROZEN = ROOT / "frozen_m6_signed_c3_divisor_translation_gcd.json"
R161_COST = ROOT / "m6_signed_c3_divisor_translation_gcd_cost_ledger.json"
R161_REPLAY = ROOT / "m6_signed_c3_divisor_translation_gcd_replay.json"
R161_CONTROLS = ROOT / "m6_signed_c3_divisor_translation_gcd_controls.json"
R161_LOGS = ROOT / "factor_logs_and_identical_descent_r161.json"
R161_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_signed_c3_divisor_translation_gcd_probe_r161.py"
R161_GATE = ROOT / "p1553_m6_signed_c3_divisor_translation_gcd_probe_gate_r161.md"
R161_PARENT = ROOT / "p1553_m6_signed_c3_divisor_translation_gcd_probe_parent_report_r161.yaml"
PRECOMPUTATION_PAPER = ROOT / "references/neiger_rosenkilde_solomatov_modcomp_precomputation_2020.pdf"
TWO_RELATION_PAPER = ROOT / "references/neiger_salvy_schost_villard_two_relation_modcomp_2026.pdf"

SOURCE_BINDINGS = (
    ("r161_producer", R161_PRODUCER, "da25fe5203e243746879ecfc699da912b299ea0781f72df222cdb322bba00e80"),
    ("r161_report", R161_REPORT, "7e73325212e9d8c3cf42ae46cce0a5c1bdebea9784b9bf5c30761c006cb8bedd"),
    ("r161_frozen", R161_FROZEN, "8c9c3981cbf9f1529547c61cc9673e778079beba86a306eef1ac54979d2520b7"),
    ("r161_cost", R161_COST, "511aeedfbb72a268e6bef87178d156cc700eccbfc49c4b7a8441e473d0461b4b"),
    ("r161_replay", R161_REPLAY, "96dee1fb3ab5423a2f5463827135e6d70b7c124c7b5171a49b3a048eb28b08c3"),
    ("r161_controls", R161_CONTROLS, "6baa61c223de901691e08771da7f8af860a21195565d40be87c9662209cfa955"),
    ("r161_logs", R161_LOGS, "70d9dc60c96d4b54f107904f31e4a6800b943fb060b2d4e4b894587da21f69d1"),
    ("r161_test", R161_TEST, "661edbc66d4dda3179a9829e59a2ddef5e816f11497a1a334513a1ae46676a38"),
    ("r161_gate", R161_GATE, "0ca8660b9554c39e2c90a8f29cee05554b9fb3adce0c4c2a1038e739b3888bcd"),
    ("r161_parent", R161_PARENT, "67fc0a45c5493011d0d14188150c08402aaf78539c966a8840ca11e723134429"),
    ("modcomp_precomputation_2020", PRECOMPUTATION_PAPER, "9fdce743a5183f544df2e2d640e30e6eb2cf2c647b6d89b04aea5faa21cdb488"),
    ("two_relation_modcomp_2026", TWO_RELATION_PAPER, "bfa0a9fb8f3ec6cd1d2aa95907a03df131d6a4ffb3abac56983bfee42c235866"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_batch_inverse_transpose_modcomp_fit_probe_report_r162.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_batch_inverse_transpose_modcomp_fit.json"
DEFAULT_COST = ROOT / "m6_batch_inverse_transpose_modcomp_fit_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_batch_inverse_transpose_modcomp_fit_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_batch_inverse_transpose_modcomp_fit_controls.json"
DEFAULT_TRANSPOSE = ROOT / "batch_inverse_functional_transpose_r162.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R161 = load_module("p1553_r161_for_r162", R161_PRODUCER)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {name: sha256_file(path) for name, path, _ in SOURCE_BINDINGS}
    failures = [
        name for name, _, expected in SOURCE_BINDINGS if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R162 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def build_product_tree(points: list[int], prime: int) -> dict[str, Any]:
    def build(start: int, stop: int) -> dict[str, Any]:
        if stop - start == 1:
            return {
                "poly": [(-points[start]) % prime, 1],
                "index": start,
            }
        middle = (start + stop) // 2
        left = build(start, middle)
        right = build(middle, stop)
        return {
            "poly": R161.poly_mul(left["poly"], right["poly"], prime),
            "left": left,
            "right": right,
        }

    if not points:
        raise ValueError("multipoint tree requires at least one point")
    return build(0, len(points))


def multipoint_evaluate(poly: list[int], points: list[int], prime: int) -> list[int]:
    tree = build_product_tree(points, prime)
    values = [0] * len(points)

    def descend(remainder: list[int], node: dict[str, Any]) -> None:
        if "index" in node:
            values[int(node["index"])] = int(remainder[0]) % prime
            return
        left = node["left"]
        right = node["right"]
        descend(R161.poly_mod(remainder, left["poly"], prime), left)
        descend(R161.poly_mod(remainder, right["poly"], prime), right)

    descend(R161.poly_mod(poly, tree["poly"], prime), tree)
    return values


def denominator_quotient(
    u_value: int, u_poly: list[int], prime: int
) -> dict[str, Any]:
    scalar = R161.poly_eval(u_poly, u_value, prime)
    denominator = [u_value % prime, (-1) % prime]
    numerator = R161.poly_sub([scalar], u_poly, prime)
    quotient, remainder = R161.poly_divmod(numerator, denominator, prime)
    quotient_identity_exact = remainder == [0] and R161.poly_sub(
        R161.poly_mul(denominator, quotient, prime), numerator, prime
    ) == [0]
    if scalar == 0:
        return {
            "u_value": u_value,
            "u_at_target": 0,
            "exceptional": True,
            "quotient_identity_exact": quotient_identity_exact,
            "quotient": quotient,
            "inverse": None,
            "inverse_identity_exact": False,
        }
    inverse = R161.poly_scale(quotient, pow(scalar, -1, prime), prime)
    inverse_identity_exact = (
        R161.poly_mul_mod(denominator, inverse, u_poly, prime) == [1]
    )
    return {
        "u_value": u_value,
        "u_at_target": scalar,
        "exceptional": False,
        "quotient_identity_exact": quotient_identity_exact,
        "quotient": quotient,
        "inverse": inverse,
        "inverse_identity_exact": inverse_identity_exact,
    }


def quotient_functional_polynomial(
    u_poly: list[int], weights: list[int], prime: int
) -> list[int]:
    degree = R161.poly_degree(u_poly)
    if len(weights) != degree:
        raise ValueError("functional weight count must equal deg(U)")
    coefficients = []
    for power in range(degree):
        value = 0
        for row in range(degree - power):
            value += weights[row] * u_poly[power + row + 1]
        coefficients.append(value % prime)
    return R161.trim(coefficients, prime)


def target_records(control: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *control["positive_targets"],
        control["empty_target"],
        control["denominator_exception_target"],
    ]


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    prime = int(curve["field_prime"])
    dimension = len(R161.R157.c_point_pairs(curve, 0))
    factor_base = R161.R160.generic_factor_base(curve, dimension, seed)
    divisor = R161.signed_c3_divisor(factor_base["representatives"], curve)
    base = R161.finite_control(curve, seed)
    records = target_records(base)
    target_x = [int(record["target"][0]) for record in records]
    direct_values = [R161.poly_eval(divisor["u"], value, prime) for value in target_x]
    batched_values = multipoint_evaluate(divisor["u"], target_x, prime)
    quotients = [
        denominator_quotient(value, divisor["u"], prime) for value in target_x
    ]
    expected_exceptional = [
        int(record.get("exceptional_left_count", 0)) > 0 for record in records
    ]
    functional_rows = []
    for functional_index in range(3):
        weights = [
            (
                seed
                + 17 * (functional_index + 1) * (coefficient_index + 1)
                + 29 * coefficient_index * coefficient_index
            )
            % prime
            for coefficient_index in range(len(divisor["u"]) - 1)
        ]
        functional_poly = quotient_functional_polynomial(
            divisor["u"], weights, prime
        )
        batch_functional_values = multipoint_evaluate(
            functional_poly, target_x, prime
        )
        direct_functional_values = [
            sum(weight * coefficient for weight, coefficient in zip(weights, row["quotient"]))
            % prime
            for row in quotients
        ]
        inverse_values_exact = all(
            row["exceptional"]
            or (
                sum(
                    weight * coefficient
                    for weight, coefficient in zip(weights, row["inverse"])
                )
                % prime
                == batch_value * pow(row["u_at_target"], -1, prime) % prime
            )
            for row, batch_value in zip(quotients, batch_functional_values)
        )
        functional_rows.append(
            {
                "functional_index": functional_index,
                "weights_sha256": sha256_json(weights),
                "functional_polynomial_sha256": sha256_json(functional_poly),
                "batch_values_sha256": sha256_json(batch_functional_values),
                "quotient_functional_values_exact": (
                    batch_functional_values == direct_functional_values
                ),
                "inverse_functional_values_exact_off_exceptional_roots": (
                    inverse_values_exact
                ),
            }
        )
    return {
        "control_id": f"{curve['family_id']}_batch_inverse_transpose_seed{seed}_d{dimension}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "seed": seed,
        "factor_base_dimension": dimension,
        "c3_divisor_degree": len(divisor["u"]) - 1,
        "target_count": len(records),
        "positive_target_count": len(base["positive_targets"]),
        "target_roles": [record["role"] for record in records],
        "target_x_sha256": sha256_json(target_x),
        "batch_u_evaluation_matches_direct": batched_values == direct_values,
        "batch_u_values_sha256": sha256_json(batched_values),
        "expected_exceptional_flags": expected_exceptional,
        "detected_exceptional_flags": [row["exceptional"] for row in quotients],
        "exceptional_detection_exact": (
            expected_exceptional == [row["exceptional"] for row in quotients]
        ),
        "all_quotient_identities_exact": all(
            row["quotient_identity_exact"] for row in quotients
        ),
        "all_regular_inverse_identities_exact": all(
            row["exceptional"] or row["inverse_identity_exact"]
            for row in quotients
        ),
        "functional_count": len(functional_rows),
        "all_quotient_functionals_exact": all(
            row["quotient_functional_values_exact"] for row in functional_rows
        ),
        "all_inverse_functionals_exact": all(
            row["inverse_functional_values_exact_off_exceptional_roots"]
            for row in functional_rows
        ),
        "functional_rows": functional_rows,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
        "finite_functional_checks_receive_attack_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "batch_denominator_inverse": (
            "For monic U and U(u) nonzero, Q_u(X)=(U(u)-U(X))/(u-X) "
            "satisfies (u-X)Q_u=U(u)-U(X), so "
            "(u-X)^(-1)=Q_u/U(u) in F_p[X]/U. U(u)=0 detects exactly the "
            "R161 denominator-exception roots."
        ),
        "multipoint_scalar_layer": (
            "All U(u_j) values for N targets are computable by a subproduct "
            "and remainder tree in soft-linear work in deg(U)+N."
        ),
        "transposed_functional": (
            "For U=sum_k c_k X^k and a fixed functional w=(w_r), "
            "<w,Q_u>=P_w(u), where the coefficient of u^s in P_w is "
            "sum_r w_r c_(s+r+1). The coefficients of P_w are a single "
            "cross-correlation, so one fixed functional of every Q_u is "
            "available by soft-linear preprocessing plus multipoint evaluation."
        ),
        "nonlinear_boundary": (
            "The transposed identity supplies linear access to the denominator "
            "inverses. It does not compute lambda_j^2, U(phi_j), "
            "V(phi_j), a source gcd, or a source backpointer without "
            "materializing or otherwise processing the target-varying inner maps."
        ),
        "scope": (
            "This is an exact shareable-linear-layer theorem and a fit audit "
            "for published modular-composition algorithms. It is not a lower "
            "bound against nonlinear aggregate algorithms."
        ),
    }


def literature_record() -> dict[str, Any]:
    return {
        "generic_precomputation": {
            "title": (
                "Generic Bivariate Multi-point Evaluation, Interpolation and "
                "Modular Composition with Precomputation"
            ),
            "authors": "Vincent Neiger, Johan Rosenkilde, Grigory Solomatov",
            "url": "https://arxiv.org/abs/2003.12468",
            "local_sha256": sha256_file(PRECOMPUTATION_PAPER),
            "fit": (
                "The quasi-linear online modular-composition result "
                "precomputes both M and A, then varies the input f. R162 fixes "
                "M=U and f=U but varies A=phi_j with every target, so the "
                "stated precomputation theorem does not share across this batch."
            ),
        },
        "two_relation_matrices": {
            "title": "Faster modular composition using two relation matrices",
            "authors": "Vincent Neiger, Bruno Salvy, Eric Schost, Gilles Villard",
            "url": "https://arxiv.org/abs/2601.17422",
            "local_sha256": sha256_file(TWO_RELATION_PAPER),
            "fit": (
                "The paper accelerates one generic modular composition to "
                "soft-O(n^((omega+3)/4)) algebraic operations and recalls the "
                "finite-field Kedlaya-Umans near-linear bit bound. It does not "
                "state a many-varying-inner algorithm with total soft-O(n+N) work."
            ),
        },
        "novelty_status": "batch_inverse_transpose_identity_standard_or_nearby_novelty_unverified",
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_batch_inverse_transpose_modcomp_fit.cost.r162.v1",
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "strict_r159_batch_cap_exponent_B": fraction_record(Fraction(5, 4)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "target_dependent_soft_linear_n_pass_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "soft_linear_n_pass_meets_strict_batch_cap": False,
        "soft_linear_n_pass_is_below_pollard_rho": True,
        "soft_linear_n_pass_rho_gap_exponent_B": fraction_record(Fraction(1, 4)),
        "independent_near_linear_composition_batch_exponent_B": fraction_record(
            Fraction(7, 2)
        ),
        "independent_near_linear_composition_is_below_rho": False,
        "monomial_batch_cost_rule": (
            "A cost n^alpha N^beta has B-exponent (9 alpha + 5 beta)/4."
        ),
        "strict_batch_fit_condition": "9 alpha + 5 beta <= 5",
        "global_below_rho_fit_condition": "9 alpha + 5 beta < 10",
        "one_soft_linear_n_pass_global_fit_pair": {
            "alpha": 1,
            "beta": 0,
            "weighted_sum": 9,
            "fits_below_rho": True,
        },
        "independent_near_linear_pair": {
            "alpha": 1,
            "beta": 1,
            "weighted_sum": 14,
            "fits_below_rho": False,
        },
        "batch_denominator_scalar_evaluation_supplied": True,
        "batch_fixed_linear_functional_access_supplied": True,
        "batch_nonlinear_composition_supplied": False,
        "batch_source_gcd_and_backpointer_supplied": False,
        "unconditional_total_attack_cost_supplied": False,
        "finite_checks_receive_attack_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_batch_eval = all(row["batch_u_evaluation_matches_direct"] for row in rows)
    all_exceptional = all(row["exceptional_detection_exact"] for row in rows)
    all_quotients = all(row["all_quotient_identities_exact"] for row in rows)
    all_inverses = all(row["all_regular_inverse_identities_exact"] for row in rows)
    all_functionals = all(
        row["all_quotient_functionals_exact"]
        and row["all_inverse_functionals_exact"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_batch_inverse_transpose_modcomp_fit.controls.r162.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "exact_batch_u_evaluation_control_count": sum(
            row["batch_u_evaluation_matches_direct"] for row in rows
        ),
        "exact_exceptional_detection_control_count": sum(
            row["exceptional_detection_exact"] for row in rows
        ),
        "exact_quotient_inverse_control_count": sum(
            row["all_quotient_identities_exact"]
            and row["all_regular_inverse_identities_exact"]
            for row in rows
        ),
        "exact_transposed_functional_control_count": sum(
            row["all_quotient_functionals_exact"]
            and row["all_inverse_functionals_exact"]
            for row in rows
        ),
        "candidate_oracle_consumed": False,
        "finite_controls_receive_attack_credit": False,
        "controls": rows,
    }
    theorem = theorem_record()
    literature = literature_record()
    cost = cost_record()
    obligations = {
        "twelve_source_bindings_verified": len(actual_bindings) == 12,
        "r161_signed_divisor_interface_inherited": True,
        "denominator_quotient_identity_complete": True,
        "denominator_inverse_identity_complete": True,
        "batch_u_multipoint_evaluation_complete": True,
        "exceptional_root_detection_complete": True,
        "transposed_fixed_functional_identity_complete": True,
        "strict_and_global_cost_caps_distinguished": True,
        "published_precomputation_orientation_audited": True,
        "published_single_composition_cost_audited": True,
        "six_finite_batch_controls_complete": len(rows) == 6,
        "all_finite_batch_evaluations_exact": all_batch_eval,
        "all_finite_exceptional_detections_exact": all_exceptional,
        "all_finite_quotient_identities_exact": all_quotients,
        "all_finite_inverse_identities_exact": all_inverses,
        "all_finite_transposed_functionals_exact": all_functionals,
        "candidate_oracles_avoided": True,
        "finite_controls_scoped_without_attack_credit": True,
        "target_batched_nonlinear_composition_complete": False,
        "target_batched_source_gcd_complete": False,
        "source_backpointer_assignment_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    admission = {
        "obligations": obligations,
        "passed_obligation_count": passed,
        "obligation_count": len(obligations),
        "batch_inverse_linear_layer_admitted": True,
        "global_below_rho_n_pass_window_admitted": True,
        "target_batched_nonlinear_layer_admitted": False,
        "lane_admitted": False,
    }
    transpose = {
        "schema": "p1553.m6_batch_inverse_functional_transpose.r162.v1",
        "identity": theorem["transposed_functional"],
        "control_records": [
            {
                "control_id": row["control_id"],
                "target_count": row["target_count"],
                "batch_u_values_sha256": row["batch_u_values_sha256"],
                "functional_rows": row["functional_rows"],
            }
            for row in rows
        ],
        "all_functional_controls_exact": all_functionals,
        "nonlinear_composition_complete": False,
    }
    replay = {
        "schema": "p1553.m6_batch_inverse_transpose_modcomp_fit.replay.r162.v1",
        "source_bindings": source_binding_records(),
        "control_records": [
            {
                "control_id": row["control_id"],
                "target_x_sha256": row["target_x_sha256"],
                "batch_u_values_sha256": row["batch_u_values_sha256"],
                "batch_u_evaluation_exact": row[
                    "batch_u_evaluation_matches_direct"
                ],
                "exceptional_detection_exact": row[
                    "exceptional_detection_exact"
                ],
                "quotient_inverse_exact": (
                    row["all_quotient_identities_exact"]
                    and row["all_regular_inverse_identities_exact"]
                ),
                "functional_transpose_exact": (
                    row["all_quotient_functionals_exact"]
                    and row["all_inverse_functionals_exact"]
                ),
            }
            for row in rows
        ],
        "all_replay_invariants_pass": (
            all_batch_eval
            and all_exceptional
            and all_quotients
            and all_inverses
            and all_functionals
        ),
    }
    frozen = {
        "schema": "p1553.m6_batch_inverse_transpose_modcomp_fit.frozen.r162.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "literature": literature,
        "cost": cost,
        "admission": admission,
        "successor_interface": {
            "persistent_degree": "n=B^(9/4+o(1))",
            "target_count": "N=B^(5/4+o(1))",
            "strict_batch_goal": "B^(5/4+o(1))",
            "global_below_rho_goal": "strictly below B^(5/2)",
            "required_output": (
                "target-labeled degree-at-most-20 source factors and backpointers"
            ),
            "open_primitive": (
                "aggregate nonlinear signed-divisor composition/gcd below the "
                "global rho cap, preferably in soft-linear n work"
            ),
        },
    }
    next_action = (
        "Construct an aggregate nonlinear signed-divisor batch operator, not "
        "another denominator-inversion routine. It may spend target-dependent "
        "soft-O(n)=B^(9/4+o(1)) work, which misses the strict R159 batch cap "
        "but remains below rho, provided total charged work stays strictly "
        "below B^(5/2), it computes the lambda-square/composition/gcd layer, "
        "labels every degree-at-most-20 source factor by target, handles "
        "exceptional roots, and replays factor logs and identical descent."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Separate the batchable linear denominator layer of R161 from "
            "the nonlinear modular-composition bottleneck, audit published "
            "algorithm fit, and state the true below-rho total-cost window."
        ),
        "source_bindings": source_binding_records(),
        "deduplication": {
            "r117": (
                "Already charges translated-divisor batches and excludes "
                "universal linear shift sketches; R162 adds an exact compact "
                "inverse/functional identity, not a new source locator."
            ),
            "r148": (
                "Already charges generic static pair-sum indexing; R162 does "
                "not promote the finite quotient materialization as an index."
            ),
            "r161": (
                "Supplies the nonlinear signed-divisor source gcd. R162 only "
                "compresses its denominator-inverse linear sublayer and "
                "widens the successor from the strict batch cap to the global "
                "below-rho total-cost window."
            ),
        },
        "theorem": theorem,
        "literature": literature,
        "cost": cost,
        "controls": controls,
        "admission": admission,
        "classification": (
            "BATCH_U_EVALUATION_SOFT_LINEAR__EXACT_DENOMINATOR_QUOTIENT_AND_"
            "INVERSE__FIXED_LINEAR_FUNCTIONALS_TRANSPOSE_TO_MULTIPOINT_"
            "EVALUATION__SIX_FINITE_CONTROLS_EXACT__PUBLISHED_PRECOMPUTATION_"
            "FIXES_INNER_MAP_AND_DOES_NOT_SHARE_ACROSS_TARGETS__INDEPENDENT_"
            "COMPOSITION_B7O2__TARGET_DEPENDENT_B9O4_AGGREGATE_PASS_STILL_"
            "BELOW_RHO__NONLINEAR_COMPOSITION_GCD_BACKPOINTER_OPEN__NO_RHO_"
            "SHOUP_BREAKTHROUGH"
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "next_action": next_action,
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "transpose": transpose,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--transpose-output", type=Path, default=DEFAULT_TRANSPOSE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.transpose_output, bundle["transpose"])
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
