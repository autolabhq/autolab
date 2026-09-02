#!/usr/bin/env python3
"""Deduplicate the R169 lambda-zero Fitting factor against the target norm."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_lambda_zero_fitting_target_norm_dedup.r170.v1"

R169_PRODUCER = ROOT / "p1553_m6_regularized_log_trace_displacement_rank_probe_r169.py"
R169_REPORT = ROOT / "p1553_m6_regularized_log_trace_displacement_rank_probe_report_r169.json"
R169_FROZEN = ROOT / "frozen_m6_regularized_log_trace_displacement_rank.json"
R169_COST = ROOT / "m6_regularized_log_trace_displacement_rank_cost_ledger.json"
R169_REPLAY = ROOT / "m6_regularized_log_trace_displacement_rank_replay.json"
R169_CONTROLS = ROOT / "m6_regularized_log_trace_displacement_rank_controls.json"
R169_PENCIL = ROOT / "regularized_trace_pencil_and_displacement_r169.json"
R169_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_regularized_log_trace_displacement_rank_probe_r169.py"
R169_GATE = ROOT / "p1553_m6_regularized_log_trace_displacement_rank_probe_gate_r169.md"
R169_PARENT = ROOT / "p1553_m6_regularized_log_trace_displacement_rank_probe_parent_report_r169.yaml"
R164_REPORT = ROOT / "p1553_m6_randomized_target_divisor_norm_union_probe_report_r164.json"
R164_GATE = ROOT / "p1553_m6_randomized_target_divisor_norm_union_probe_gate_r164.md"
R167_REPORT = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_report_r167.json"
R167_GATE = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_gate_r167.md"
ELLIPTIC_CAUCHY_PAPER = ROOT / "references/prokofev_zabrodin_elliptic_cauchy_matrices_2023.pdf"

SOURCE_BINDINGS = (
    ("r169_producer", R169_PRODUCER, "213b8d9ca7b900adef6af241efac79dbe8ca01a2283b3cd166ba56d8cfd7d36f"),
    ("r169_report", R169_REPORT, "6f9123923396d0c6478486c9a669cb73ff759db00fa8997954f5b4a35d11ce86"),
    ("r169_frozen", R169_FROZEN, "df7d50d0bf60835d4513950c0961c65cb8901c20afe3b4dcea75a5a9e488cbf5"),
    ("r169_cost", R169_COST, "bab32516614e139ed989a6d7186aa1daed08c8e2efb2815d1b273c3b1a685a1f"),
    ("r169_replay", R169_REPLAY, "f6538b5474971bb9dbdf594b3efd8eb3c47bf202b518942b35871e8407a27044"),
    ("r169_controls", R169_CONTROLS, "954308ce9d715fb0d904297923b0bb7316e7d1ff625831368a261e55daba5e5b"),
    ("r169_pencil", R169_PENCIL, "cc36dc603f9c28da1194a1b699a71f670566a40aad2af12038d8f18131648d18"),
    ("r169_test", R169_TEST, "fe149185aa0eddc72b896d48bf3ceadd05c6e5ab9502b9b809c4ab396badc2ac"),
    ("r169_gate", R169_GATE, "7aba9b4c16a35dc7db1f1ad5953ef50e597a882644260c028d8b6d7a412d185f"),
    ("r169_parent", R169_PARENT, "90778d88d7c268d7fdc5c4a5257a350e8f3b15cff5f527537bf9ca7710248e78"),
    ("r164_report", R164_REPORT, "8d9ddeaf584d8d8bd41c3373c9c765b1361aa9a287cf763217b1296ed85b389a"),
    ("r164_gate", R164_GATE, "65f78505220ff5e396fb834c9c81034861b80391578089cc295d72c9be771ea3"),
    ("r167_report", R167_REPORT, "1d2f009525ef9d54a0f538d3a0a8cefe8451d4d97933b91b31ed1bb5c0b47e3c"),
    ("r167_gate", R167_GATE, "41c4869bb231dfa2d0de1bb0d280aa34bd95903ed425792b80a84c762b845e3b"),
    ("elliptic_cauchy_2023", ELLIPTIC_CAUCHY_PAPER, "43e94a255b6999505893f6ac1d4c70facee3500dda758b30465cd793dc0fab74"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_lambda_zero_fitting_target_norm_dedup_probe_report_r170.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_lambda_zero_fitting_target_norm_dedup.json"
DEFAULT_COST = ROOT / "m6_lambda_zero_fitting_target_norm_dedup_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_lambda_zero_fitting_target_norm_dedup_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_lambda_zero_fitting_target_norm_dedup_controls.json"
DEFAULT_FITTING = ROOT / "lambda_zero_fitting_norm_and_density_r170.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R169 = load_module("p1553_r169_for_r170", R169_PRODUCER)
R167 = R169.R167
R166 = R169.R166
R161 = R169.R161


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
        raise AssertionError(f"R170 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def pad(poly: list[int], size: int) -> list[int]:
    return [*poly, *([0] * (size - len(poly)))]


def coefficient_density(poly: list[int], size: int) -> dict[str, Any]:
    padded = pad(poly, size)
    nonzero = sum(value != 0 for value in padded)
    return {
        "slot_count": size,
        "nonzero_coefficient_count": nonzero,
        "density": nonzero / size,
        "degree": R161.poly_degree(poly),
    }


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    r167 = R167.finite_control(curve, seed)
    _, divisor, target_records = R166.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    targets = [
        tuple(record["target"])
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    signed_support = selected + [
        R167.point_negate(point, curve) for point in selected
    ]
    prime = int(curve["field_prime"])
    u_poly = [int(value) for value in divisor["u"]]
    n = len(selected)

    factor_polynomials: list[list[int]] = []
    factor_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        values = []
        for left in selected:
            translated = R167.point_subtract(target, left, curve)
            if translated is None:
                raise AssertionError("R167 retained target reached a pole")
            values.append(R167.kummer_value(divisor, translated, prime))
        polynomial = R161.interpolate(
            [(int(left[0]), value) for left, value in zip(selected, values)],
            prime,
        )
        if any(
            R161.poly_eval(polynomial, int(left[0]), prime) != value
            for left, value in zip(selected, values)
        ):
            raise AssertionError("target-factor interpolation failed")
        factor_polynomials.append(polynomial)
        factor_rows.append(
            {
                "target_index": target_index,
                "target": R167.point_list(target),
                "value_sha256": sha256_json(values),
                "coefficient_sha256": sha256_json(pad(polynomial, n)),
                **coefficient_density(polynomial, n),
            }
        )

    aggregate = [1]
    for polynomial in factor_polynomials:
        aggregate = R161.poly_mul_mod(aggregate, polynomial, u_poly, prime)

    direct_values = []
    corrected_fitting_values = []
    numerator = r167["numerator_witness"]
    denominator = r167["denominator_witness"]
    anchor = tuple(r167["anchor"])
    auxiliary_points = [tuple(point) for point in r167["auxiliary_points"]]
    for left in selected:
        direct = 1
        for target in targets:
            translated = R167.point_subtract(target, left, curve)
            if translated is None:
                raise AssertionError("retained target reached direct pole")
            direct = direct * R167.kummer_value(divisor, translated, prime) % prime
        direct_values.append(direct)

        correction = 1
        for point in [anchor, *auxiliary_points]:
            translated = R167.point_subtract(point, left, curve)
            if translated is None:
                raise AssertionError("correction reached a pole")
            correction = correction * R167.kummer_value(
                divisor, translated, prime
            ) % prime
        signed_product = 1
        for support_point in signed_support:
            translated = R167.point_add(support_point, left, curve)
            value = R167.rational_value(
                numerator, denominator, translated, prime
            )
            signed_product = signed_product * value % prime
        h_left = R167.rational_value(numerator, denominator, left, prime)
        corrected = (
            correction
            * signed_product
            * pow(pow(h_left, 2 * n, prime), -1, prime)
            % prime
        )
        corrected_fitting_values.append(corrected)

    aggregate_values = [
        R161.poly_eval(aggregate, int(left[0]), prime) for left in selected
    ]
    if aggregate_values != direct_values:
        raise AssertionError("coefficient-ring aggregate differs from direct norm")
    if corrected_fitting_values != direct_values:
        raise AssertionError("lambda-zero corrected Fitting value differs from norm")

    candidate_factor = R161.poly_gcd(u_poly, aggregate, prime)
    candidate_roots = sorted(
        int(left[0])
        for left, value in zip(selected, aggregate_values)
        if value == 0
    )
    factor_density_rows = [row["density"] for row in factor_rows]
    aggregate_density = coefficient_density(aggregate, n)
    return {
        "control_id": f"{curve['family_id']}_lambda_zero_norm_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "subgroup_order": int(curve["subgroup_order"]),
        "seed": seed,
        "c3_divisor_degree": n,
        "retained_target_count": len(targets),
        "represented_factor_slot_count": n * len(targets),
        "factor_polynomials": factor_rows,
        "all_target_factor_interpolants_exact": True,
        "minimum_factor_coefficient_density": min(factor_density_rows),
        "mean_factor_coefficient_density": (
            sum(factor_density_rows) / len(factor_density_rows)
        ),
        "all_target_factor_interpolants_full_degree": all(
            row["degree"] == n - 1 for row in factor_rows
        ),
        "aggregate_polynomial": {
            "coefficient_sha256": sha256_json(pad(aggregate, n)),
            **aggregate_density,
        },
        "aggregate_is_full_degree": aggregate_density["degree"] == n - 1,
        "aggregate_matches_direct_target_norm": True,
        "corrected_lambda_zero_fitting_matches_direct_norm": True,
        "r167_all_reciprocity_identities_exact": r167[
            "all_reciprocity_identities_exact"
        ],
        "candidate_factor_degree": R161.poly_degree(candidate_factor),
        "candidate_factor_sha256": sha256_json(candidate_factor),
        "candidate_roots": candidate_roots,
        "r167_candidate_roots": r167["candidate_roots"],
        "candidate_factor_matches_r167": candidate_roots
        == r167["candidate_roots"],
        "direct_value_sha256": sha256_json(direct_values),
        "corrected_fitting_value_sha256": sha256_json(
            corrected_fitting_values
        ),
        "finite_interpolation_receives_asymptotic_attack_credit": False,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "lambda_zero_fitting_identity": (
            "In A=F_p[X]/(U), let c_j be the denominator-cleared Kummer "
            "target factor P -> U(x(T_j-P)) after the public equality split. "
            "The aggregate C=product_j c_j has gcd(U,C) equal to the R167 "
            "candidate factor. The constant term of R169's corrected scalar "
            "resolvent equals C componentwise by the exact R167 reciprocity "
            "identity. Thus lambda-zero Fitting support is the existing target "
            "norm support up to explicit units, not a new constructor."
        ),
        "standard_fraction_free_presentation": (
            "Representing every c_j in the monomial basis of A uses nN field "
            "slots. A fraction-free product tree or coefficient-ring "
            "subresultant can preserve candidate zero divisors, but it first "
            "consumes this represented input and therefore has B^(7/2) state "
            "or work at the campaign caps. The final aggregate C has only n "
            "slots; obtaining it from the compact target input is the open "
            "streaming norm problem."
        ),
        "elliptic_cauchy_literature_scope": (
            "Prokofev and Zabrodin prove determinant, inverse, product, and "
            "factorization identities for a complex sigma-function elliptic "
            "Cauchy matrix. Their factorization uses full square matrices and "
            "does not give a finite-field Weierstrass-coordinate, candidate-"
            "safe, output-sensitive norm algorithm for an arbitrary degree-N "
            "target divisor. Applying a kernel separately to N arbitrary poles "
            "would retain nN generator state."
        ),
        "scope": (
            "R170 proves an exact semantic deduplication and charges standard "
            "represented routes. Finite interpolation and density observations "
            "receive no asymptotic attack or lower-bound credit. An SLP-"
            "streaming norm, custom algebraic elliptic operator, or another "
            "implicit constructor remains open."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_lambda_zero_fitting_target_norm_dedup.cost.r170.v1",
        "field_and_subgroup_order_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "compact_target_divisor_slp_state_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "represented_target_factor_table_exponent_B": fraction_record(
            Fraction(7, 2)
        ),
        "standard_fraction_free_product_or_subresultant_exponent_B": fraction_record(
            Fraction(7, 2)
        ),
        "represented_aggregate_element_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "swapped_fitting_matrix_exponent_B": fraction_record(Fraction(9, 2)),
        "expected_candidate_factor_exponent_B": fraction_record(Fraction(3, 4)),
        "expected_signed_verification_exponent_B": fraction_record(Fraction(2)),
        "preferred_slp_streaming_norm_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "standard_represented_route_inside_rho": False,
        "aggregate_element_size_inside_rho": True,
        "aggregate_element_constructor_inside_rho_supplied": False,
        "slp_streaming_output_sensitive_norm_supplied": False,
        "elliptic_cauchy_paper_supplies_finite_field_constructor": False,
        "finite_density_receives_asymptotic_lower_bound_credit": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_exact = all(
        row["all_target_factor_interpolants_exact"]
        and row["aggregate_matches_direct_target_norm"]
        and row["corrected_lambda_zero_fitting_matches_direct_norm"]
        and row["r167_all_reciprocity_identities_exact"]
        and row["candidate_factor_matches_r167"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_lambda_zero_fitting_target_norm_dedup.controls.r170.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_coefficient_ring_replays_exact": all_exact,
        "all_target_factor_interpolants_full_degree": all(
            row["all_target_factor_interpolants_full_degree"] for row in rows
        ),
        "all_aggregate_polynomials_full_degree": all(
            row["aggregate_is_full_degree"] for row in rows
        ),
        "minimum_observed_factor_coefficient_density": min(
            row["minimum_factor_coefficient_density"] for row in rows
        ),
        "minimum_observed_aggregate_coefficient_density": min(
            row["aggregate_polynomial"]["density"] for row in rows
        ),
        "represented_factor_slot_count": sum(
            row["represented_factor_slot_count"] for row in rows
        ),
        "candidate_factor_degree_sum": sum(
            row["candidate_factor_degree"] for row in rows
        ),
        "finite_controls_receive_asymptotic_attack_credit": False,
        "candidate_oracle_consumed": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "fifteen_source_bindings_verified": len(actual_bindings) == 15,
        "r169_scalar_resolvent_interface_inherited": True,
        "r167_reciprocity_interface_inherited": True,
        "r164_aggregate_norm_semantics_deduplicated": True,
        "lambda_zero_fitting_equals_direct_norm_up_to_units": all_exact,
        "six_coefficient_ring_controls_complete": len(rows) == 6,
        "all_target_factor_interpolants_exact": all(
            row["all_target_factor_interpolants_exact"] for row in rows
        ),
        "all_aggregate_products_exact": all(
            row["aggregate_matches_direct_target_norm"] for row in rows
        ),
        "all_corrected_fitting_values_exact": all(
            row["corrected_lambda_zero_fitting_matches_direct_norm"]
            for row in rows
        ),
        "all_candidate_factors_match_r167": all(
            row["candidate_factor_matches_r167"] for row in rows
        ),
        "represented_nn_state_charged": True,
        "swapped_n2_state_charged": True,
        "aggregate_element_n_state_charged": True,
        "finite_density_scoped_without_lower_bound_credit": True,
        "elliptic_cauchy_primary_scope_bound": True,
        "candidate_oracles_avoided": True,
        "slp_streaming_output_sensitive_norm_complete": False,
        "custom_algebraic_elliptic_operator_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(bool(value) for value in obligations.values())
    classification = (
        "ADMIT_LAMBDA_ZERO_FITTING_EQUALS_R167_DIRECT_TARGET_NORM_UP_TO_UNITS__"
        "SIX_COEFFICIENT_RING_REPLAYS__STANDARD_DENSE_FACTORS_NN_B7O2__"
        "SWAPPED_FITTING_N2_B9O2__ELLIPTIC_CAUCHY_FACTORIZATION_NOT_A_FINITE_"
        "FIELD_OUTPUT_SENSITIVE_CONSTRUCTOR__SLP_STREAMING_NORM_OPEN__NO_RHO_"
        "SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Construct or refute an SLP-streaming output-sensitive elliptic target "
        "norm modulo U from the compact generalized Miller witness, below "
        "B^(5/2) and preferably B^(9/4+o(1)). The constructor must emit the "
        "aggregate element or candidate gcd without materializing N dense "
        "elements of F_p[X]/U, an nN coefficient body, or an n^2 Fitting matrix."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Determine whether R169's lambda-zero Fitting specialization is a "
            "new sub-rho primitive or the existing aggregate target norm under "
            "a different presentation."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "controls": controls,
        "cost": cost,
        "literature": {
            "elliptic_cauchy_matrices": {
                "title": "Elliptic Cauchy matrices",
                "authors": "V. Prokofev and A. Zabrodin",
                "arxiv": "2305.02837",
                "fit": theorem["elliptic_cauchy_literature_scope"],
            }
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "lambda_zero_target_norm_equivalence_admitted": all_exact,
            "standard_fraction_free_route_closed_by_cost": True,
            "slp_streaming_norm_constructor_admitted": False,
            "lane_admitted": False,
        },
        "classification": classification,
        "next_action": next_action,
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    frozen = {
        "schema": "p1553.m6_lambda_zero_fitting_target_norm_dedup.frozen.r170.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "critical_experiment": {
            "hypothesis": (
                "A compact target-divisor SLP can stream the lambda-zero norm "
                "into A=F_p[X]/U without constructing N dense A-elements."
            ),
            "decisive_test": next_action,
            "falsifier": (
                "The route materializes nN coefficients, an n^2 matrix, N "
                "independent modular compositions, candidate inverses, or a "
                "unit-cost norm/Fitting/subresultant oracle; reaches B^(5/2); "
                "or fails exact R170/R169/R167 replay."
            ),
        },
        "promotion_allowed": False,
    }
    fitting = {
        "schema": "p1553.m6_lambda_zero_fitting_target_norm_dedup.fitting.r170.v1",
        "identity": theorem["lambda_zero_fitting_identity"],
        "standard_presentation": theorem["standard_fraction_free_presentation"],
        "controls": [
            {
                "control_id": row["control_id"],
                "represented_factor_slot_count": row[
                    "represented_factor_slot_count"
                ],
                "minimum_factor_coefficient_density": row[
                    "minimum_factor_coefficient_density"
                ],
                "mean_factor_coefficient_density": row[
                    "mean_factor_coefficient_density"
                ],
                "aggregate_polynomial": row["aggregate_polynomial"],
                "candidate_factor_degree": row["candidate_factor_degree"],
                "direct_value_sha256": row["direct_value_sha256"],
                "corrected_fitting_value_sha256": row[
                    "corrected_fitting_value_sha256"
                ],
            }
            for row in rows
        ],
    }
    replay = {
        "schema": "p1553.m6_lambda_zero_fitting_target_norm_dedup.replay.r170.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "direct_value_sha256": row["direct_value_sha256"],
                "corrected_fitting_value_sha256": row[
                    "corrected_fitting_value_sha256"
                ],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "aggregate_coefficient_sha256": row["aggregate_polynomial"][
                    "coefficient_sha256"
                ],
            }
            for row in rows
        ],
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "fitting": fitting,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--fitting-output", type=Path, default=DEFAULT_FITTING)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    for path in (
        args.report_output,
        args.frozen_output,
        args.cost_output,
        args.replay_output,
        args.controls_output,
        args.fitting_output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.fitting_output, bundle["fitting"])
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane=0 breakthrough=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
