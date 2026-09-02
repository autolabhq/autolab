#!/usr/bin/env python3
"""Reduce aggregate M6 marginal linear algebra to Jacobian actions."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import pathlib
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parent
SCHEMA = "p1553.m6_matrix_free_marginal_jacobian_krylov.r151.v1"

R150_PRODUCER = ROOT / (
    "p1553_m6_rational_convolution_subalgebra_rigidity_probe_r150.py"
)
R150_REPORT = ROOT / (
    "p1553_m6_rational_convolution_subalgebra_rigidity_"
    "probe_report_r150.json"
)
R150_FROZEN = ROOT / (
    "frozen_m6_rational_convolution_subalgebra_rigidity.json"
)
R150_COST = ROOT / (
    "m6_rational_convolution_subalgebra_rigidity_cost_ledger.json"
)
R150_REPLAY = ROOT / (
    "m6_rational_convolution_subalgebra_rigidity_replay.json"
)
R150_CONTROLS = ROOT / (
    "m6_rational_convolution_subalgebra_rigidity_controls.json"
)
R150_LOGS = ROOT / "factor_logs_and_identical_descent_r150.json"
R150_TEST = ROOT / (
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_m6_rational_convolution_subalgebra_rigidity_probe_r150.py"
)
R150_GATE = ROOT / (
    "p1553_m6_rational_convolution_subalgebra_rigidity_probe_gate_r150.md"
)
R150_PARENT = ROOT / (
    "p1553_m6_rational_convolution_subalgebra_rigidity_"
    "probe_parent_report_r150.yaml"
)

R144_PRODUCER = ROOT / (
    "p1553_m6_weighted_fiber_marginal_log_operator_probe_r144.py"
)
R144_REPORT = ROOT / (
    "p1553_m6_weighted_fiber_marginal_log_operator_"
    "probe_report_r144.json"
)
R144_CONTROLS = ROOT / (
    "m6_weighted_fiber_marginal_log_operator_controls.json"
)
R144_GATE = ROOT / (
    "p1553_m6_weighted_fiber_marginal_log_operator_probe_gate_r144.md"
)
R144_PARENT = ROOT / (
    "p1553_m6_weighted_fiber_marginal_log_operator_"
    "probe_parent_report_r144.yaml"
)

BAUR_STRASSEN = ROOT / (
    "references/baur_strassen_partial_derivatives_1983.pdf"
)
WIEDEMANN = ROOT / (
    "references/wiedemann_sparse_linear_equations_1986.pdf"
)

SOURCE_BINDINGS = (
    (
        "r150_producer",
        R150_PRODUCER,
        "7f9217358c57a6cdb84f7f0d731e003e973096e4ff01a780ae0b3b8f482a755e",
    ),
    (
        "r150_report",
        R150_REPORT,
        "598fd0b95622354f22bc6ce531fc02aa2b5f61a1578d06dae36fbd39086ae6d3",
    ),
    (
        "r150_frozen",
        R150_FROZEN,
        "04f989b1af422a1e42456bda7f95fbfdbb9941f709a75aed1de456d98c261e2b",
    ),
    (
        "r150_cost",
        R150_COST,
        "57a022f3ffd5e149a95236f913cb612faff214f2ec2ef2d69a48f7af219017cd",
    ),
    (
        "r150_replay",
        R150_REPLAY,
        "4fb314326c46d9cf2988866435d4fdc7705f0dd28444c288002925d4cf833341",
    ),
    (
        "r150_controls",
        R150_CONTROLS,
        "cc838d5f5cd05b3c43ebfae096d3b9110fa06e02a2cb3ea10285155d119c2a28",
    ),
    (
        "r150_logs",
        R150_LOGS,
        "cf0aa2174f836375165628813ee8f16ea29005a6345aec5af42f071f3367d1e5",
    ),
    (
        "r150_test",
        R150_TEST,
        "e5ff8f6cc11cf7e252e1aa49b456942c7d9bcfae660707c0bf75b9626cf7bf03",
    ),
    (
        "r150_gate",
        R150_GATE,
        "1def188f9b0c5834889a937f7de9629f78b98c0e4f3b3249b1ba170a404483b6",
    ),
    (
        "r150_parent",
        R150_PARENT,
        "81761afebf32a86199a3aef3bad3520a9a4a25654ceef051a547d6829e9098b6",
    ),
    (
        "r144_producer",
        R144_PRODUCER,
        "3cd7a3e40bf696a836ff6b6cad4c59a7ca73a44d72d152232cddeffa14e90a5c",
    ),
    (
        "r144_report",
        R144_REPORT,
        "954bc230dbe28300534a765c3bde389b7ae807767b664dc82f63ede1b3a5805d",
    ),
    (
        "r144_controls",
        R144_CONTROLS,
        "0677e3a7220e7258e4dc56bff14f5e083a09ed048948667fe0582dd2fa4c6ea8",
    ),
    (
        "r144_gate",
        R144_GATE,
        "c967cc10137fe861384a80a8261763c2ecb86ca2951db8c5bf43f348c65e1e9b",
    ),
    (
        "r144_parent",
        R144_PARENT,
        "db906bd74c9c43fd12ab196a36d59ac80bfbbd22de494c4b6b2efb4d0c96c268",
    ),
    (
        "baur_strassen_primary",
        BAUR_STRASSEN,
        "828e8c9eb30af0089c48b1a5bcac3990bc0e21d3b68d4e343490b0fdaa38ecae",
    ),
    (
        "wiedemann_primary",
        WIEDEMANN,
        "8ec0b8a8b35c02bd4a84236129ab991fdbcca4b4feb8332c22b08dbdefc1218a",
    ),
)

DEFAULT_REPORT = ROOT / (
    "p1553_m6_matrix_free_marginal_jacobian_krylov_"
    "probe_report_r151.json"
)
DEFAULT_FROZEN = ROOT / (
    "frozen_m6_matrix_free_marginal_jacobian_krylov.json"
)
DEFAULT_COST = ROOT / (
    "m6_matrix_free_marginal_jacobian_krylov_cost_ledger.json"
)
DEFAULT_REPLAY = ROOT / (
    "m6_matrix_free_marginal_jacobian_krylov_replay.json"
)
DEFAULT_CONTROLS = ROOT / (
    "m6_matrix_free_marginal_jacobian_krylov_controls.json"
)
DEFAULT_LOGS = ROOT / "factor_logs_and_identical_descent_r151.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R144 = load_module("p1553_r144_for_r151", R144_PRODUCER)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> dict[str, str]:
    actual = {
        name: sha256_file(path)
        for name, path, _ in SOURCE_BINDINGS
    }
    failures = [
        name
        for name, _, expected in SOURCE_BINDINGS
        if actual[name] != expected
    ]
    if failures:
        raise AssertionError(f"R151 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        ),
        "decimal": float(value),
    }


def dot(
    left: Iterable[int], right: Iterable[int], modulus: int
) -> int:
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    ) % modulus


def matrix_vector(
    matrix: Iterable[Iterable[int]],
    vector: Iterable[int],
    modulus: int,
) -> list[int]:
    values = tuple(vector)
    return [dot(row, values, modulus) for row in matrix]


def transpose_vector(
    matrix: Iterable[Iterable[int]],
    vector: Iterable[int],
    modulus: int,
) -> list[int]:
    rows = [tuple(row) for row in matrix]
    weights = tuple(vector)
    if len(rows) != len(weights):
        raise ValueError("transpose weight length mismatch")
    width = len(rows[0])
    return [
        sum(weights[row] * rows[row][column] for row in range(len(rows)))
        % modulus
        for column in range(width)
    ]


def deterministic_vector(
    control_id: str, role: str, width: int, modulus: int
) -> list[int]:
    return [
        int.from_bytes(
            hashlib.sha256(
                f"R151|{control_id}|{role}|{index}".encode("utf-8")
            ).digest(),
            "big",
        )
        % modulus
        for index in range(width)
    ]


def actual_control(control: dict[str, Any]) -> dict[str, Any]:
    modulus = int(control["subgroup_order"])
    selected = control["selected_independent_fibers"]
    matrix = [
        list(map(int, row["reduced_marginal_row"]))
        for row in selected
    ]
    rhs = [int(row["known_rhs"]) % modulus for row in selected]
    width = int(control["meaningful_log_dimension"])
    if len(matrix) != width or any(len(row) != width for row in matrix):
        raise AssertionError("selected R144 marginal matrix is not square")

    direction = deterministic_vector(
        control["control_id"], "direction", width, modulus
    )
    contraction = deterministic_vector(
        control["control_id"], "contraction", width, modulus
    )
    forward = matrix_vector(matrix, direction, modulus)
    reverse = transpose_vector(matrix, contraction, modulus)
    bilinear_left = dot(contraction, forward, modulus)
    bilinear_right = dot(direction, reverse, modulus)
    recovered_logs = R144.solve_square_mod(matrix, rhs, modulus)
    recovered_rhs = matrix_vector(matrix, recovered_logs, modulus)

    return {
        "control_id": control["control_id"],
        "subgroup_order": modulus,
        "meaningful_log_dimension": width,
        "selected_matrix_sha256": sha256_json(matrix),
        "direction_sha256": sha256_json(direction),
        "contraction_sha256": sha256_json(contraction),
        "forward_jacobian_vector_sha256": sha256_json(forward),
        "reverse_jacobian_transpose_vector_sha256": sha256_json(reverse),
        "bilinear_pairing_forward": bilinear_left,
        "bilinear_pairing_reverse": bilinear_right,
        "jacobian_transpose_identity_exact": (
            bilinear_left == bilinear_right
        ),
        "matrix_free_recovered_logs_sha256": sha256_json(recovered_logs),
        "matrix_free_recovered_rhs_sha256": sha256_json(recovered_rhs),
        "selected_rhs_sha256": sha256_json(rhs),
        "selected_linear_system_replay_exact": recovered_rhs == rhs,
        "r144_factor_logs_recovered": bool(
            control["meaningful_factor_logs_recovered"]
        ),
        "r144_factor_logs_and_shifted_descent_replay": bool(
            control["all_cartesian_factor_logs_replay"]
            and control["all_shifted_target_samples_replay"]
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "verifier_matrix_receives_candidate_credit": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "weighted_count_map": (
            "Let Z(w) be the vector of aggregate known-target fiber counts "
            "as functions of logarithmic A/C atom weights, evaluated at "
            "w=1."
        ),
        "marginal_jacobian": (
            "The aggregate marginal matrix is M=D_log Z(1), with "
            "M_(i,a)=w_a*partial Z_i/partial w_a at w=1."
        ),
        "forward_action": (
            "For any atom vector x, Mx is the first-order coefficient of "
            "Z(w_a*(1+epsilon*x_a)); it is a forward directional "
            "derivative and does not require explicit marginal rows."
        ),
        "transpose_action": (
            "For any row vector lambda, M^T lambda is the logarithmic "
            "gradient at w=1 of the scalar contraction "
            "sum_i lambda_i Z_i(w)."
        ),
        "bilinear_certificate": (
            "lambda dot (M x)=x dot (M^T lambda) over the subgroup field."
        ),
        "baur_strassen_use": (
            "Baur-Strassen gives constant-factor nonscalar overhead for "
            "the gradient of one complete division-safe scalar contraction "
            "circuit. Forward dual evaluation gives one directional action "
            "at constant scalar width."
        ),
        "wiedemann_use": (
            "Wiedemann linear algebra accepts a matrix as an operator and "
            "uses O(n) matrix-vector applications, up to logarithmic and "
            "probabilistic factors, with an exact residual check."
        ),
        "offline_online_boundary": (
            "Differentiating a complete circuit also differentiates its "
            "setup. Baur-Strassen alone does not compile a reusable "
            "weight-parametric tangent/adjoint state. That state or a "
            "weight-independent setup remains mandatory."
        ),
        "scope": (
            "This is an exact interface reduction and conditional cost "
            "envelope, not a supplied count circuit, derivative data "
            "structure, generic-prime rank theorem, factor-log algorithm, "
            "or Shoup-bound improvement."
        ),
        "primary_references": {
            "baur_strassen": (
                "Walter Baur and Volker Strassen, The Complexity of "
                "Partial Derivatives, Theoretical Computer Science 22 "
                "(1983), 317-330, doi:10.1016/0304-3975(83)90110-X."
            ),
            "wiedemann": (
                "Douglas H. Wiedemann, Solving Sparse Linear Equations "
                "Over Finite Fields, IEEE Transactions on Information "
                "Theory 32(1) (1986), 54-62, "
                "doi:10.1109/TIT.1986.1057137."
            ),
        },
        "novelty_status": "interface_composition_novelty_unverified",
    }


def cost_ledger() -> dict[str, Any]:
    return {
        "schema": (
            "p1553.m6_matrix_free_marginal_jacobian_krylov."
            "cost.r151.v1"
        ),
        "meaningful_log_dimension_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "known_target_row_count_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "a6_markers_per_row_exponent_B": fraction_record(Fraction(1, 2)),
        "scalar_marker_batch_exponent_B": fraction_record(Fraction(5, 4)),
        "explicit_marginal_matrix_output_exponent_B": fraction_record(
            Fraction(3, 2)
        ),
        "conditional_bidirectional_operator_apply_exponent_B": (
            fraction_record(Fraction(5, 4))
        ),
        "conditional_wiedemann_iteration_exponent_B": fraction_record(
            Fraction(3, 4)
        ),
        "conditional_matrix_free_solve_exponent_B": fraction_record(
            Fraction(2)
        ),
        "setup_state_cap_exponent_B": fraction_record(Fraction(9, 4)),
        "pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "conditional_matrix_free_solve_inside_setup_cap": True,
        "conditional_matrix_free_solve_inside_pollard_rho": True,
        "conditions": [
            (
                "One frozen scalar-blind marker-batch circuit applies both "
                "M and M^T in B^(5/4+o(1)) work."
            ),
            (
                "Its setup is weight-independent or includes reusable "
                "division-safe tangent and adjoint state without replay."
            ),
            (
                "The structured generic-prime marginal operator has the "
                "required rank and Wiedemann failure is controlled and "
                "followed by an exact residual check."
            ),
            (
                "All field, extension, chart, integer-lift, memory, and bit "
                "costs remain within the charged exponents."
            ),
        ],
        "candidate_field_dlp_used": False,
        "candidate_root_oracle_used": False,
        "bidirectional_marker_batch_operator_supplied": False,
        "reusable_weight_parametric_derivative_state_supplied": False,
        "generic_prime_rank_and_density_supplied": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, Any]:
    actual_bindings = verify_source_bindings()
    r144_controls = read_json(R144_CONTROLS)
    actual = [
        actual_control(control)
        for control in r144_controls["actual_controls"]
    ]
    all_transposes = all(
        row["jacobian_transpose_identity_exact"] for row in actual
    )
    all_solves = all(
        row["selected_linear_system_replay_exact"]
        and row["r144_factor_logs_recovered"]
        and row["r144_factor_logs_and_shifted_descent_replay"]
        for row in actual
    )
    theorem = theorem_record()
    costs = cost_ledger()

    controls = {
        "schema": (
            "p1553.m6_matrix_free_marginal_jacobian_krylov."
            "controls.r151.v1"
        ),
        "actual_control_count": len(actual),
        "all_jacobian_transpose_identities_exact": all_transposes,
        "all_selected_linear_system_replays_exact": all_solves,
        "meaningful_dimensions": [
            row["meaningful_log_dimension"] for row in actual
        ],
        "candidate_discrete_log_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "actual_controls": actual,
    }

    frozen = {
        "schema": (
            "p1553.m6_matrix_free_marginal_jacobian_krylov."
            "frozen.r151.v1"
        ),
        "source_bindings": source_binding_records(),
        "source_binding_actual_sha256": actual_bindings,
        "theorem": theorem,
        "cost": costs,
        "required_open_outputs": {
            "bidirectional_marker_batch_count_jacobian_operator": "open",
            "reusable_weight_parametric_derivative_state": "open",
            "generic_prime_rank_and_density": "open",
            "factor_logs_without_verifier_matrix": "open",
            "identical_target_descent": "open",
            "generic_prime_family_algorithm": "open",
            "shoup_bound_improvement": "open",
        },
    }

    replay = {
        "schema": (
            "p1553.m6_matrix_free_marginal_jacobian_krylov."
            "replay.r151.v1"
        ),
        "controls": actual,
        "all_jacobian_transpose_identities_exact": all_transposes,
        "all_selected_linear_system_replays_exact": all_solves,
        "theorem": theorem,
    }

    logs = {
        "schema": (
            "p1553.m6_matrix_free_marginal_jacobian_krylov."
            "logs_descent.r151.v1"
        ),
        "finite_selected_systems_and_shifted_descents_replay": all_solves,
        "candidate_factor_logs_computed": False,
        "candidate_identical_target_descent_computed": False,
        "generic_prime_family_transfer_supplied": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }

    obligations = {
        "seventeen_source_bindings_verified": len(actual_bindings) == 17,
        "r150_fixed_depth_scope_inherited_without_overclaim": True,
        "r144_weighted_marginal_semantics_inherited": True,
        "eight_actual_controls_complete": len(actual) == 8,
        "all_actual_jacobian_transpose_identities_exact": all_transposes,
        "all_actual_selected_linear_system_replays_exact": all_solves,
        "forward_directional_derivative_identity_complete": True,
        "reverse_scalar_contraction_gradient_identity_complete": True,
        "matrix_free_bilinear_certificate_complete": True,
        "baur_strassen_primary_bound_pinned": True,
        "wiedemann_primary_operator_bound_pinned": True,
        "explicit_marginal_output_B3_over_2_charged": True,
        "conditional_matrix_free_B2_envelope_derived": True,
        "offline_online_derivative_state_not_inferred": True,
        "candidate_dlp_and_root_oracles_avoided": True,
        "finite_results_scoped_without_asymptotic_credit": True,
        "bidirectional_marker_batch_operator_complete": False,
        "reusable_weight_parametric_derivative_state_complete": False,
        "generic_prime_rank_and_density_complete": False,
        "factor_logs_without_verifier_matrix_complete": False,
        "identical_target_descent_complete": False,
        "generic_prime_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    passed = sum(obligations.values())
    next_action = (
        "Construct one scalar-blind weight-parametric marker-batch circuit "
        "whose frozen setup is weight-independent or carries reusable "
        "division-safe tangent and adjoint state. It must apply both the "
        "aggregate marginal Jacobian M and M^T in B^(5/4+o(1)) work without "
        "forming marginal rows, q modes, q targets, or C3 occurrences. "
        "Then run an exact-residual matrix-free solve and shifted descent "
        "with no DLP, root, Fourier, recurrence, algebra-state, count, "
        "marginal, rank, or source oracle."
    )

    report = {
        "schema": SCHEMA,
        "date": "2026-07-29",
        "classification": (
            "AGGREGATE_MARGINAL_MATRIX_IS_LOG_WEIGHT_JACOBIAN__FORWARD_"
            "DIRECTION_GIVES_MX__REVERSE_SCALAR_CONTRACTION_GIVES_MT_"
            "LAMBDA__WIEDEMANN_ACCEPTS_OPERATOR_ACTIONS__CONDITIONAL_B5O4_"
            "APPLY_TIMES_B3O4_ITERATIONS_EQUALS_B2_BELOW_SETUP_AND_RHO__"
            "BAUR_STRASSEN_DOES_NOT_PRESERVE_OFFLINE_ONLINE_STATE__"
            "BIDIRECTIONAL_WEIGHT_PARAMETRIC_COUNT_CIRCUIT_OPEN__NO_"
            "GENERIC_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
        ),
        "objective": (
            "Determine whether the V99 fixed-depth marker-count circuit "
            "needs to emit every marginal row, or can expose matrix-free "
            "Jacobian and transpose actions for factor-log linear algebra."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": costs,
        "controls": controls,
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "matrix_free_jacobian_interface_admitted": True,
            "conditional_B2_linear_algebra_envelope_admitted": True,
            "bidirectional_weight_parametric_count_circuit_admitted": False,
            "lane_admitted": False,
        },
        "next_action": next_action,
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_oracle_consumed": False,
        "finite_controls_receive_asymptotic_credit": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": costs,
        "replay": replay,
        "controls": controls,
        "logs": logs,
    }


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-output", type=pathlib.Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=pathlib.Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=pathlib.Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=pathlib.Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=pathlib.Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--logs-output", type=pathlib.Path, default=DEFAULT_LOGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.logs_output, bundle["logs"])
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
