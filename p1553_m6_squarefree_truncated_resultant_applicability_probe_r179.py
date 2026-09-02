#!/usr/bin/env python3
"""Test truncated-resultant routes for the signed norm modulo squarefree U."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_squarefree_truncated_resultant_applicability.r179.v1"

R178_PRODUCER = ROOT / "p1553_m6_marked_fitting_signed_norm_dedup_probe_r178.py"
R178_REPORT = ROOT / "p1553_m6_marked_fitting_signed_norm_dedup_probe_report_r178.json"
R178_FROZEN = ROOT / "frozen_m6_marked_fitting_signed_norm_dedup.json"
R178_COST = ROOT / "m6_marked_fitting_signed_norm_dedup_cost_ledger.json"
R178_REPLAY = ROOT / "m6_marked_fitting_signed_norm_dedup_replay.json"
R178_CONTROLS = ROOT / "m6_marked_fitting_signed_norm_dedup_controls.json"
R178_EQUIVALENCE = ROOT / "marked_fitting_signed_norm_equivalence_r178.json"
R178_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_marked_fitting_signed_norm_dedup_probe_r178.py"
R178_GATE = ROOT / "p1553_m6_marked_fitting_signed_norm_dedup_probe_gate_r178.md"
R178_PARENT = ROOT / "p1553_m6_marked_fitting_signed_norm_dedup_probe_parent_report_r178.yaml"

R172_REPORT = ROOT / "p1553_m6_target_sign_conjugate_s3_self_resultant_probe_report_r172.json"
R172_GATE = ROOT / "p1553_m6_target_sign_conjugate_s3_self_resultant_probe_gate_r172.md"
MOROZ_SCHOST = ROOT / "references/moroz_schost_truncated_resultant_1609.04259.pdf"
BIVARIATE_RESULTANTS = ROOT / "references/hyun_neiger_schost_bivariate_resultants_1905.04356.pdf"
SHOUP_PAPER = ROOT / "references/shoup_generic_dlp_lower_bound_1997.pdf"

SOURCE_BINDINGS = (
    ("r178_producer", R178_PRODUCER, "5aa01661614943739d8dc4ee623def8df94a0ecbc6119f344cee7f88b4b24312"),
    ("r178_report", R178_REPORT, "679eeeeb8031fa4b4121d2004f5778629c5f16f116dcf8b10b3603a2ad63d2a6"),
    ("r178_frozen", R178_FROZEN, "a575b4bbf47f3d43ada0fc4411d352f795a87261caae7ba388f7e427b46325a4"),
    ("r178_cost", R178_COST, "999972250b94afa500eaa1e93285d70d662192232cd2c6ae9527a36ddf8ca606"),
    ("r178_replay", R178_REPLAY, "b241914e4cdfd7b14e66fba051399dabfd84b1248be31fab0db57714bad6acae"),
    ("r178_controls", R178_CONTROLS, "ca239fb08f96845781ea8a7be8237fc0bf0bc66e1ed2918cb6ac9a1d55437716"),
    ("r178_equivalence", R178_EQUIVALENCE, "4c9d6cb8b5a7d32747e13542f42b3a3a382b04d97cc6caa709cd6aa706dd7c66"),
    ("r178_test", R178_TEST, "ae71be82ca4f0113d27459b6360de65b1c5312e84ff9951899960a6be196ea65"),
    ("r178_gate", R178_GATE, "bf0071ea97bd82f97b18c2f944f039153b5711ab9f3ace4b0aa910b805df9462"),
    ("r178_parent", R178_PARENT, "12cf8c48f4d0c3d5973c8d07dc85f92eb7459aae4952942c09a39075d4ccceaf"),
    ("r172_report", R172_REPORT, "66f986b68c6d31576c688da4509d828139275331009dba4bb05abd45aef5ce5e"),
    ("r172_gate", R172_GATE, "14fb9459c75510f26177e2ef795266a825f58c2cdd11797d6ec6e60bd31dc951"),
    ("moroz_schost_truncated_resultant", MOROZ_SCHOST, "160c68cfbb413ca27352a064cbf2d27f7ad4ed6a210c3d6ead2770e00204b709"),
    ("hyun_neiger_schost_bivariate_resultants", BIVARIATE_RESULTANTS, "32b73cf0ca7172bdec0f8f1b256adda628a86d6dd7eee8e07e8644e35a9f16f3"),
    ("shoup_1997", SHOUP_PAPER, "89d19aad3a4d98b563029de9135d30c8ed9b831d74f7348c286acc22f9af85b3"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_squarefree_truncated_resultant_applicability_probe_report_r179.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_squarefree_truncated_resultant_applicability.json"
DEFAULT_COST = ROOT / "m6_squarefree_truncated_resultant_applicability_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_squarefree_truncated_resultant_applicability_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_squarefree_truncated_resultant_applicability_controls.json"
DEFAULT_APPLICABILITY = ROOT / "squarefree_truncated_resultant_applicability_r179.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R178 = load_module("p1553_r178_for_r179", R178_PRODUCER)
R167 = R178.R167
R175 = R178.R175
R161 = R178.R161


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


def verify_source_bindings() -> None:
    failures = [
        name
        for name, path, expected in SOURCE_BINDINGS
        if sha256_file(path) != expected
    ]
    if failures:
        raise AssertionError(f"R179 source binding mismatch: {failures}")


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def control_row(payload: dict[str, Any], family_id: str, seed: int) -> dict[str, Any]:
    matches = [
        row
        for row in payload["controls"]["controls"]
        if row["family_id"] == family_id and int(row["seed"]) == seed
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one control for {family_id}:{seed}")
    return matches[0]


def poly_derivative(poly: list[int], prime: int) -> list[int]:
    if len(poly) <= 1:
        return [0]
    result = [
        index * coefficient % prime
        for index, coefficient in enumerate(poly)
    ][1:]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result or [0]


def linear_power(root: int, exponent: int, prime: int) -> list[int]:
    result = [1]
    factor = [(-root) % prime, 1]
    for _ in range(exponent):
        result = R161.poly_mul(result, factor, prime)
    return result


def finite_control(
    curve: dict[str, Any],
    seed: int,
    r178_report: dict[str, Any],
    r174_report: dict[str, Any],
) -> dict[str, Any]:
    family_id = curve["family_id"]
    expected = control_row(r178_report, family_id, seed)
    chow = control_row(r174_report, family_id, seed)
    r167 = R167.finite_control(curve, seed)
    _, divisor, _ = R175.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    prime = int(curve["field_prime"])
    u_poly = divisor["u"]
    n = len(selected)
    target_count = int(chow["retained_target_count"])

    row_norm_points: list[tuple[int, int]] = []
    pair_evaluation_count = 0
    for left in selected:
        row_norm = 1
        for right in selected:
            pair_sum = R167.point_add(left, right, curve)
            value = R167.rational_value(
                r167["numerator_witness"],
                r167["denominator_witness"],
                pair_sum,
                prime,
            )
            row_norm = row_norm * value % prime
            pair_evaluation_count += 1
        row_norm_points.append((int(left[0]), row_norm))

    aggregate = R161.interpolate(row_norm_points, prime)
    if sha256_json(aggregate) != expected["aggregate_polynomial_sha256"]:
        raise AssertionError("R179 aggregate differs from R178")
    if R161.poly_mod(aggregate, u_poly, prime) != aggregate:
        raise AssertionError("aggregate is not the canonical residue modulo U")
    if any(
        R161.poly_eval(aggregate, x_value, prime) != value
        for x_value, value in row_norm_points
    ):
        raise AssertionError("order-one CRT residues do not replay aggregate")

    u_derivative = poly_derivative(u_poly, prime)
    squarefree_gcd = R161.poly_gcd(u_poly, u_derivative, prime)
    if R161.poly_degree(squarefree_gcd) != 0:
        raise AssertionError("selected divisor is not squarefree")

    expansion_root = int(selected[0][0])
    local_modulus = linear_power(expansion_root, n, prime)
    local_zero = R161.poly_mod(local_modulus, local_modulus, prime)
    squarefree_alias_remainder = R161.poly_mod(local_modulus, u_poly, prime)
    if local_zero != [0] or squarefree_alias_remainder == [0]:
        raise AssertionError("local-power versus squarefree alias witness failed")

    canonical_from_crt = R161.interpolate(row_norm_points, prime)
    if canonical_from_crt != aggregate:
        raise AssertionError("squarefree CRT reconstruction failed")

    target_chow_slots = math.comb(target_count + 2, 2)
    return {
        "control_id": f"{family_id}_squarefree_truncated_resultant_seed{seed}",
        "family_id": family_id,
        "field_prime": prime,
        "seed": seed,
        "selected_divisor_degree": n,
        "target_count": target_count,
        "pair_evaluation_count": pair_evaluation_count,
        "aggregate_polynomial_sha256": sha256_json(aggregate),
        "aggregate_equals_r178": True,
        "aggregate_slot_count": n,
        "squarefree_modulus_sha256": sha256_json(u_poly),
        "squarefree_gcd_degree": R161.poly_degree(squarefree_gcd),
        "crt_component_count": n,
        "crt_local_precision_each": 1,
        "crt_reconstruction_exact": True,
        "global_local_power_precision": n,
        "elimination_degree_floor": n,
        "global_degree_precision_product": n * n,
        "crt_degree_precision_product_sum": n * n,
        "expansion_root": expansion_root,
        "local_power_modulus_sha256": sha256_json(local_modulus),
        "local_power_alias_remainder_mod_u_sha256": sha256_json(
            squarefree_alias_remainder
        ),
        "local_power_aliases_zero_mod_local_power": True,
        "local_power_does_not_alias_zero_mod_u": True,
        "target_chow_coefficient_slots": target_chow_slots,
        "finite_pair_scan_receives_asymptotic_credit": False,
        "moroz_schost_complexity_instantiation_receives_lower_bound_credit": False,
        "candidate_oracle_consumed": False,
    }


def theorem_record() -> dict[str, str]:
    return {
        "published_interface": (
            "Moroz-Schost computes Res_y(P,Q) modulo x^k in softly O(dk) "
            "field operations for bivariate inputs of degree at most d, under "
            "its stated characteristic conditions. The truncation ideal is the "
            "local nilpotent ideal (x^k), not an arbitrary squarefree modulus U."
        ),
        "arbitrary_modulus_mismatch": (
            "For every monic squarefree U of degree n and every root a of U, "
            "the polynomials 0 and (X-a)^n agree modulo (X-a)^n but differ "
            "modulo U. Therefore one order-n local truncation does not determine "
            "the signed aggregate norm modulo U."
        ),
        "squarefree_crt_specialization": (
            "Because U is squarefree and split on the selected divisor, the "
            "required residue algebra is the product of n order-one local fields. "
            "Applying the published local resultant separately at all components "
            "uses n local calls with k=1 and elimination degree at least n, for "
            "softly O(n^2) standard work."
        ),
        "single_expansion_specialization": (
            "Forcing one order-n x-adic expansion sets k=n while the selected "
            "elimination polynomial U(Z) has degree n, so the published softly "
            "O(dk) bound also specializes to softly O(n^2). It computes the wrong "
            "local ideal unless an additional arbitrary-modulus reduction is supplied."
        ),
        "factored_gap": (
            "The direct theorem accepts represented bivariate polynomials. "
            "Expanding the N target linear factors already exposes Theta(N^2) "
            "dual-Chow coefficients at the rho boundary. A useful successor must "
            "keep both U and the target factors factored and share all component "
            "work through an arbitrary-squarefree dynamic-evaluation or transposed operator."
        ),
        "scope": (
            "This closes only direct x-adic, per-component CRT, and standard "
            "represented-coefficient uses of the cited resultant algorithms. It "
            "is not a lower bound against factored D5, transposed, modular-"
            "composition, arithmetic-circuit, or output-sensitive constructions."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_squarefree_truncated_resultant_applicability.cost.r179.v1",
        "selected_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_factor_count_exponent_B": fraction_record(Fraction(5, 4)),
        "required_output_exponent_B": fraction_record(Fraction(9, 4)),
        "desired_factored_total_exponent_B": fraction_record(Fraction(9, 4)),
        "moroz_single_expansion_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "moroz_single_expansion_precision_exponent_B": fraction_record(Fraction(9, 4)),
        "moroz_single_expansion_total_exponent_B": fraction_record(Fraction(9, 2)),
        "moroz_squarefree_crt_total_exponent_B": fraction_record(Fraction(9, 2)),
        "represented_target_chow_exponent_B": fraction_record(Fraction(5, 2)),
        "standard_nN_target_grid_exponent_B": fraction_record(Fraction(7, 2)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "single_expansion_strictly_below_rho": False,
        "squarefree_crt_strictly_below_rho": False,
        "represented_target_chow_strictly_below_rho": False,
        "conditional_factored_dynamic_evaluation_strictly_below_rho": True,
        "factored_dynamic_evaluation_constructor_supplied": False,
        "standard_route_negative_claimed_as_circuit_lower_bound": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    verify_source_bindings()
    r178_report = json.loads(R178_REPORT.read_text())
    r174_report = json.loads(R178.R174_REPORT.read_text())
    rows = [
        finite_control(curve, seed, r178_report, r174_report)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_exact = all(
        row["aggregate_equals_r178"]
        and row["crt_reconstruction_exact"]
        and row["squarefree_gcd_degree"] == 0
        and row["local_power_aliases_zero_mod_local_power"]
        and row["local_power_does_not_alias_zero_mod_u"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_squarefree_truncated_resultant_applicability.controls.r179.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_aggregates_equal_r178": all(row["aggregate_equals_r178"] for row in rows),
        "all_squarefree_moduli_verified": all(row["squarefree_gcd_degree"] == 0 for row in rows),
        "all_crt_reconstructions_exact": all(row["crt_reconstruction_exact"] for row in rows),
        "all_local_power_alias_witnesses_exact": all(
            row["local_power_aliases_zero_mod_local_power"]
            and row["local_power_does_not_alias_zero_mod_u"]
            for row in rows
        ),
        "selected_divisor_degree_sum": sum(row["selected_divisor_degree"] for row in rows),
        "target_count_sum": sum(row["target_count"] for row in rows),
        "pair_evaluation_count": sum(row["pair_evaluation_count"] for row in rows),
        "crt_component_count": sum(row["crt_component_count"] for row in rows),
        "global_degree_precision_product_sum": sum(
            row["global_degree_precision_product"] for row in rows
        ),
        "crt_degree_precision_product_sum": sum(
            row["crt_degree_precision_product_sum"] for row in rows
        ),
        "target_chow_coefficient_slot_sum": sum(
            row["target_chow_coefficient_slots"] for row in rows
        ),
        "finite_pair_scan_receives_asymptotic_credit": False,
        "candidate_oracle_consumed": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "r178_r172_literature_and_shoup_bindings_exact": True,
        "six_signed_norm_controls_replayed": len(rows) == 6,
        "r178_aggregate_residues_replayed": controls["all_aggregates_equal_r178"],
        "selected_moduli_squarefree_complete": controls["all_squarefree_moduli_verified"],
        "order_one_crt_reconstruction_complete": controls["all_crt_reconstructions_exact"],
        "local_power_alias_witness_complete": controls["all_local_power_alias_witnesses_exact"],
        "published_xadic_interface_scoped": True,
        "single_expansion_degree_precision_charged": True,
        "squarefree_crt_degree_precision_charged": True,
        "represented_target_chow_charged": True,
        "standard_route_negative_scoped_without_lower_bound": True,
        "factored_dynamic_evaluation_constructor_complete": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "factor_logs_complete": False,
        "identical_fresh_target_descent_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    classification = (
        "ADMIT_MOROZ_SCHOST_XADIC_INTERFACE_SCOPE__SIX_R178_SIGNED_NORM_"
        "CONTROLS_REPLAYED__SQUAREFREE_U_IS_N_ORDER_ONE_CRT_COMPONENTS__"
        "LOCAL_POWER_ALIAS_WITNESSES_EXACT__ONE_ORDER_N_EXPANSION_AND_N_"
        "ORDER_ONE_EXPANSIONS_BOTH_STANDARD_N2__REPRESENTED_TARGET_CHOW_AT_"
        "RHO__DIRECT_TRUNCATED_RESULTANT_ROUTE_CLOSED__FACTORED_SQUAREFREE_"
        "DYNAMIC_EVALUATION_OPEN__NO_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Construct or refute one factored arbitrary-squarefree dynamic-evaluation "
        "operator for the unified signed elliptic translate product. Keep U,V and "
        "all N target factors factored, split U only by charged gcds at actual "
        "nonunits, share the elimination work across every CRT component, and emit "
        "C_h mod U or G_1 in softly O(n+N) total work. Reject one order-n x-adic "
        "surrogate, n independent local resultants, N quotient-ring elements, "
        "Theta(N^2) coefficient expansion, nN or n^2 grids, candidate inversions, "
        "and unit-cost resultant, norm, multipoint, root, count, marginal, rank, or source oracles."
    )
    passed = sum(obligations.values())
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Determine whether fast x-adic truncated-resultant algorithms directly "
            "supply the R178 signed aggregate norm modulo arbitrary squarefree U "
            "inside the B^(5/2) rho boundary."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "controls": controls,
        "cost": cost,
        "literature": {
            "moroz_schost_2016": {
                "result": "Res_y(P,Q) mod x^k in softly O(dk) field operations.",
                "campaign_fit": (
                    "The local x-adic theorem is replayed only as a complexity "
                    "upper bound for the two direct adaptations; arbitrary-squarefree "
                    "factored dynamic evaluation is not attributed to the paper."
                ),
            },
            "hyun_neiger_schost_2019": {
                "campaign_fit": (
                    "General represented bivariate resultant algorithms do not "
                    "provide the missing factored arbitrary-modulus output-sensitive interface."
                )
            },
            "novelty_scope": (
                "R179 contributes an applicability and cost audit for the exact "
                "R178 operator. It claims no new resultant theorem."
            ),
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "xadic_applicability_scope_admitted": all_exact,
            "direct_truncated_resultant_route_closed": all_exact,
            "factored_dynamic_evaluation_constructor_admitted": False,
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
        "schema": "p1553.m6_squarefree_truncated_resultant_applicability.frozen.r179.v1",
        "source_bindings": source_binding_records(),
        "interfaces": {
            "required_modulus": "arbitrary_squarefree_U_degree_n",
            "published_truncation_modulus": "x_power_k",
            "required_total_work": "softly_O(n+N)",
            "surviving_primitive": "factored_squarefree_dynamic_evaluation_resultant",
        },
        "promotion_allowed": False,
    }
    replay = {
        "schema": "p1553.m6_squarefree_truncated_resultant_applicability.replay.r179.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "aggregate_polynomial_sha256": row["aggregate_polynomial_sha256"],
                "squarefree_modulus_sha256": row["squarefree_modulus_sha256"],
                "local_power_modulus_sha256": row["local_power_modulus_sha256"],
                "local_power_alias_remainder_mod_u_sha256": row[
                    "local_power_alias_remainder_mod_u_sha256"
                ],
                "crt_component_count": row["crt_component_count"],
            }
            for row in rows
        ],
    }
    applicability = {
        "schema": "p1553.m6_squarefree_truncated_resultant_applicability.analysis.r179.v1",
        "published_interface": theorem["published_interface"],
        "direct_routes": {
            "single_order_n_expansion": {
                "computes_required_modulus": False,
                "standard_exponent_B": "9/2",
                "strictly_below_rho": False,
            },
            "n_order_one_crt_expansions": {
                "computes_required_modulus": True,
                "standard_exponent_B": "9/2",
                "strictly_below_rho": False,
            },
            "represented_target_chow": {
                "standard_exponent_B": "5/2",
                "strictly_below_rho": False,
            },
        },
        "open_interface": {
            "name": "factored_squarefree_dynamic_evaluation_resultant",
            "required_exponent_B": "9/4",
            "constructor_supplied": False,
        },
    }
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost,
        "replay": replay,
        "controls": controls,
        "applicability": applicability,
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
    parser.add_argument("--applicability-output", type=Path, default=DEFAULT_APPLICABILITY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    outputs = (
        (args.report_output, bundle["report"]),
        (args.frozen_output, bundle["frozen"]),
        (args.cost_output, bundle["cost"]),
        (args.replay_output, bundle["replay"]),
        (args.controls_output, bundle["controls"]),
        (args.applicability_output, bundle["applicability"]),
    )
    for path, value in outputs:
        write_json(path, value)
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} direct_route_closed=1 breakthrough=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
