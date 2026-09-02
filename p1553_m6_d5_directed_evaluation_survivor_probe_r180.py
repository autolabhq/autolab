#!/usr/bin/env python3
"""Charge D5 and directed evaluation on the signed target-factor stream."""

from __future__ import annotations

import argparse
from functools import lru_cache
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_d5_directed_evaluation_survivor.r180.v1"

R179_PRODUCER = ROOT / "p1553_m6_squarefree_truncated_resultant_applicability_probe_r179.py"
R179_REPORT = ROOT / "p1553_m6_squarefree_truncated_resultant_applicability_probe_report_r179.json"
R179_FROZEN = ROOT / "frozen_m6_squarefree_truncated_resultant_applicability.json"
R179_COST = ROOT / "m6_squarefree_truncated_resultant_applicability_cost_ledger.json"
R179_REPLAY = ROOT / "m6_squarefree_truncated_resultant_applicability_replay.json"
R179_CONTROLS = ROOT / "m6_squarefree_truncated_resultant_applicability_controls.json"
R179_APPLICABILITY = ROOT / "squarefree_truncated_resultant_applicability_r179.json"
R179_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_squarefree_truncated_resultant_applicability_probe_r179.py"
R179_GATE = ROOT / "p1553_m6_squarefree_truncated_resultant_applicability_probe_gate_r179.md"
R179_PARENT = ROOT / "p1553_m6_squarefree_truncated_resultant_applicability_probe_parent_report_r179.yaml"

R174_PRODUCER = ROOT / "p1553_m6_confluent_signed_dual_chow_pushforward_probe_r174.py"
R174_REPORT = ROOT / "p1553_m6_confluent_signed_dual_chow_pushforward_probe_report_r174.json"
R174_CONTROLS = ROOT / "m6_confluent_signed_dual_chow_pushforward_controls.json"
R174_GATE = ROOT / "p1553_m6_confluent_signed_dual_chow_pushforward_probe_gate_r174.md"

D5_PAPER = ROOT / "references/dahan_moreno_schost_xie_d5_complexity_2006.pdf"
DIRECTED_EVALUATION = ROOT / "references/van_der_hoeven_lecerf_directed_evaluation_2020.pdf"
MODCOMP_2026 = ROOT / "references/neiger_salvy_schost_villard_two_relation_modcomp_2026.pdf"
SHOUP_PAPER = ROOT / "references/shoup_generic_dlp_lower_bound_1997.pdf"

SOURCE_BINDINGS = (
    ("r179_producer", R179_PRODUCER, "00210a51d1ee5dd3f8154eeab57c5909309711c3558a0699ae9ffa0cf0b271e9"),
    ("r179_report", R179_REPORT, "3f68214851d59a68369f45cf7cc9b1c1eed768f0129cc218834fb3e25d4ac4e5"),
    ("r179_frozen", R179_FROZEN, "ef327c0c0dad05eb697a5e99bce2d8f37533fe07dc747ee8c5082652cb9edcc6"),
    ("r179_cost", R179_COST, "6315b3766fd611229bbb8394e4dd135f084e203ec823f3c712d9e08f5d7d73b7"),
    ("r179_replay", R179_REPLAY, "8d3b4e7af0f28fc0e5626ce14717fb450f54474d1e28c7c1bb6a5697794a5f06"),
    ("r179_controls", R179_CONTROLS, "07bda95159541b61d43b307e85f756ec6b7239273b73103414ef8ff52125bcbc"),
    ("r179_applicability", R179_APPLICABILITY, "037891a103819194d1c8866a779ee98fbaa554fbdd53bed8576e78c888ef4769"),
    ("r179_test", R179_TEST, "1e700b72e6934753cd939c9bcf472d9653f4e62a09e7a5af8c3f36a81f7e215e"),
    ("r179_gate", R179_GATE, "87ae37e53a7f10af580a58287f2d83848dee20ff8f5f8e157d74174b39a21791"),
    ("r179_parent", R179_PARENT, "84e4d0b6acab02415dabb6dcd917028618748318b3544868cfaaf1593792ee4c"),
    ("r174_producer", R174_PRODUCER, "9b30baf9bea77492816bd81bcbd6cfae01ae05b467b8793562101da8afe9765c"),
    ("r174_report", R174_REPORT, "a8b1d1d5fd4ffeaef17726325ebd85c343285ef61d16ca4dd1ce91dfcce28496"),
    ("r174_controls", R174_CONTROLS, "6d69f7146857cfd017281550e7a44b99aa4c53623b767c852da40b81e8bc9d40"),
    ("r174_gate", R174_GATE, "46597f7afd91ada661b5822031489b7d08445b041dfd4399b4343199457afebb"),
    ("dahan_moreno_schost_xie_d5_2006", D5_PAPER, "d2e265e13f585b9a3c9d69d27c8fa948d811a9de89b513d6983c3b8e8b87f565"),
    ("van_der_hoeven_lecerf_directed_evaluation_2020", DIRECTED_EVALUATION, "20b6959dd71e3bd16e0c072f300b07a0c3714d013a28b1b0ea03955083df5da1"),
    ("neiger_salvy_schost_villard_modcomp_2026", MODCOMP_2026, "bfa0a9fb8f3ec6cd1d2aa95907a03df131d6a4ffb3abac56983bfee42c235866"),
    ("shoup_1997", SHOUP_PAPER, "89d19aad3a4d98b563029de9135d30c8ed9b831d74f7348c286acc22f9af85b3"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_d5_directed_evaluation_survivor_probe_report_r180.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_d5_directed_evaluation_survivor.json"
DEFAULT_COST = ROOT / "m6_d5_directed_evaluation_survivor_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_d5_directed_evaluation_survivor_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_d5_directed_evaluation_survivor_controls.json"
DEFAULT_APPLICABILITY = ROOT / "d5_directed_evaluation_survivor_applicability_r180.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R179 = load_module("p1553_r179_for_r180", R179_PRODUCER)
R174 = load_module("p1553_r174_for_r180", R174_PRODUCER)
R161 = R174.R161


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.relative_to(ROOT)), "sha256": digest}
        for name, path, digest in SOURCE_BINDINGS
    }


def verify_source_bindings() -> None:
    failures = [
        name
        for name, path, expected in SOURCE_BINDINGS
        if sha256_file(path) != expected
    ]
    if failures:
        raise AssertionError(f"R180 source binding mismatch: {failures}")


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


def exact_quotient(
    numerator: list[int], denominator: list[int], prime: int
) -> list[int]:
    quotient, remainder = R161.poly_divmod(numerator, denominator, prime)
    if remainder != [0]:
        raise AssertionError("expected an exact polynomial quotient")
    return quotient


def target_norm_factor(
    curve: dict[str, Any],
    divisor: dict[str, Any],
    selected: list[tuple[int, int]],
    target: tuple[int, int],
) -> dict[str, Any]:
    prime = int(curve["field_prime"])
    values: list[tuple[int, int]] = []
    zero_roots: list[int] = []
    pair_evaluation_count = 0
    zero_biconditional_count = 0
    for left in selected:
        x_value, y_value = left
        line_poly = R174.deflated_line_polynomial(
            x_value, y_value, target, divisor["v"], prime
        )
        norm = 1
        for right in selected:
            z_value = int(right[0])
            if z_value == x_value:
                factor = (
                    (x_value - int(target[0]))
                    * (3 * x_value * x_value + int(curve["curve_a"]))
                    - 2 * y_value * (y_value + int(target[1]))
                ) % prime
            else:
                factor = R161.poly_eval(line_poly, z_value, prime)
            expected_zero = tuple(R161.R70.add(left, right, curve) or ()) == target
            if (factor == 0) != expected_zero:
                raise AssertionError("target line-factor zero biconditional failed")
            zero_biconditional_count += 1
            pair_evaluation_count += 1
            norm = norm * factor % prime
        values.append((int(x_value), norm))
        if norm == 0:
            zero_roots.append(int(x_value))

    factor_poly = R161.interpolate(values, prime)
    if any(
        R161.poly_eval(factor_poly, x_value, prime) != value
        for x_value, value in values
    ):
        raise AssertionError("target norm interpolation failed")
    return {
        "target": [int(target[0]), int(target[1])],
        "factor_polynomial": factor_poly,
        "factor_polynomial_sha256": sha256_json(factor_poly),
        "factor_degree": R161.poly_degree(factor_poly),
        "zero_roots": sorted(zero_roots),
        "zero_root_count": len(zero_roots),
        "pair_evaluation_count": pair_evaluation_count,
        "zero_biconditional_count": zero_biconditional_count,
    }


def survivor_trace(
    order: tuple[int, ...],
    u_poly: list[int],
    factors: list[dict[str, Any]],
    prime: int,
) -> dict[str, Any]:
    survivor = u_poly
    discovered_product = [1]
    visits = 0
    split_count = 0
    rows: list[dict[str, Any]] = []
    for step, factor_index in enumerate(order):
        before_degree = R161.poly_degree(survivor)
        visits += before_degree
        residue = R161.poly_mod(
            factors[factor_index]["factor_polynomial"], survivor, prime
        )
        discovered = R161.poly_gcd(survivor, residue, prime)
        discovered_degree = R161.poly_degree(discovered)
        if discovered_degree:
            split_count += 1
        survivor = exact_quotient(survivor, discovered, prime)
        discovered_product = R161.poly_mul(discovered_product, discovered, prime)
        rows.append(
            {
                "step": step,
                "factor_index": factor_index,
                "target": factors[factor_index]["target"],
                "survivor_degree_before": before_degree,
                "new_candidate_degree": discovered_degree,
                "survivor_degree_after": R161.poly_degree(survivor),
                "split_is_actual_nonunit": discovered_degree > 0,
                "discovered_factor_sha256": sha256_json(discovered),
            }
        )
    return {
        "order": list(order),
        "component_factor_visit_count": visits,
        "actual_nonunit_split_count": split_count,
        "candidate_factor": discovered_product,
        "candidate_factor_sha256": sha256_json(discovered_product),
        "survivor_polynomial": survivor,
        "survivor_polynomial_sha256": sha256_json(survivor),
        "survivor_degree": R161.poly_degree(survivor),
        "steps": rows,
    }


def optimal_order(
    n: int, root_sets: list[frozenset[int]]
) -> tuple[int, tuple[int, ...]]:
    target_count = len(root_sets)

    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[int, tuple[int, ...]]:
        if mask == (1 << target_count) - 1:
            return 0, ()
        discovered: set[int] = set()
        for index, roots in enumerate(root_sets):
            if mask & (1 << index):
                discovered.update(roots)
        current_cost = n - len(discovered)
        choices = []
        for index in range(target_count):
            if mask & (1 << index):
                continue
            tail_cost, tail_order = solve(mask | (1 << index))
            choices.append((current_cost + tail_cost, (index, *tail_order)))
        return min(choices)

    return solve(0)


def finite_control(
    curve: dict[str, Any],
    seed: int,
    r174_report: dict[str, Any],
    r178_report: dict[str, Any],
) -> dict[str, Any]:
    family_id = curve["family_id"]
    expected_r174 = control_row(r174_report, family_id, seed)
    expected_r178 = control_row(r178_report, family_id, seed)
    _, divisor, target_records = R174.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    targets = [
        tuple(record["target"])
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    prime = int(curve["field_prime"])
    n = len(selected)
    target_count = len(targets)
    if target_count != int(expected_r174["retained_target_count"]):
        raise AssertionError("R180 retained target count differs from R174")

    factors = [
        target_norm_factor(curve, divisor, selected, target)
        for target in targets
    ]
    product = [1]
    for factor in factors:
        product = R161.poly_mul_mod(
            product, factor["factor_polynomial"], divisor["u"], prime
        )
    candidate_factor = R161.poly_gcd(divisor["u"], product, prime)
    candidate_sha256 = sha256_json(candidate_factor)
    if candidate_sha256 != expected_r174["candidate_factor_sha256"]:
        raise AssertionError("target-factor product differs from R174 candidate")
    if candidate_sha256 != expected_r178["candidate_factor_sha256"]:
        raise AssertionError("target-factor product differs from R178 candidate")

    candidate_roots = sorted(
        int(point[0])
        for point in selected
        if R161.poly_eval(candidate_factor, int(point[0]), prime) == 0
    )
    factor_root_union = sorted(
        set().union(*(set(factor["zero_roots"]) for factor in factors))
    )
    if factor_root_union != candidate_roots:
        raise AssertionError("target-factor zero union differs from candidate roots")

    root_sets = [frozenset(factor["zero_roots"]) for factor in factors]
    optimal_visits, best_order = optimal_order(n, root_sets)
    natural = survivor_trace(
        tuple(range(target_count)), divisor["u"], factors, prime
    )
    optimal = survivor_trace(best_order, divisor["u"], factors, prime)
    if optimal["component_factor_visit_count"] != optimal_visits:
        raise AssertionError("optimal survivor dynamic program did not replay")
    if natural["candidate_factor"] != candidate_factor:
        raise AssertionError("natural D5 splitting lost candidate factors")
    if optimal["candidate_factor"] != candidate_factor:
        raise AssertionError("optimal D5 splitting lost candidate factors")

    noncandidate_degree = n - len(candidate_roots)
    component_visit_lower_bound = noncandidate_degree * target_count
    if optimal_visits < component_visit_lower_bound:
        raise AssertionError("survivor visit lower bound failed")
    final_survivor = optimal["survivor_polynomial"]
    all_final_factors_units = all(
        R161.poly_degree(
            R161.poly_gcd(final_survivor, factor["factor_polynomial"], prime)
        )
        == 0
        for factor in factors
    )
    if not all_final_factors_units:
        raise AssertionError("a target factor remains a nonunit on final survivor")

    factor_records = [
        {
            key: value
            for key, value in factor.items()
            if key != "factor_polynomial"
        }
        for factor in factors
    ]
    return {
        "control_id": f"{family_id}_d5_directed_survivor_seed{seed}",
        "family_id": family_id,
        "field_prime": prime,
        "seed": seed,
        "selected_divisor_degree": n,
        "target_factor_count": target_count,
        "candidate_degree": len(candidate_roots),
        "noncandidate_survivor_degree": noncandidate_degree,
        "candidate_roots": candidate_roots,
        "candidate_factor_sha256": candidate_sha256,
        "candidate_factor_equals_r174": True,
        "candidate_factor_equals_r178": True,
        "target_factor_product_mod_u_sha256": sha256_json(product),
        "target_factor_ring_multiplication_count": max(0, target_count - 1),
        "materialized_target_factor_residue_slot_count": n * target_count,
        "finite_pair_evaluation_count": sum(
            factor["pair_evaluation_count"] for factor in factors
        ),
        "zero_biconditional_count": sum(
            factor["zero_biconditional_count"] for factor in factors
        ),
        "target_factor_zero_incidence_count": sum(
            factor["zero_root_count"] for factor in factors
        ),
        "target_factor_zero_union_equals_candidate_roots": True,
        "natural_order": natural,
        "optimal_order": optimal,
        "optimal_order_dynamic_program_exact": True,
        "component_factor_visit_lower_bound": component_visit_lower_bound,
        "optimal_visits_respect_noncandidate_lower_bound": True,
        "maximum_early_split_savings": len(candidate_roots) * target_count,
        "actual_optimal_early_split_savings": n * target_count - optimal_visits,
        "all_final_survivor_factors_are_units": all_final_factors_units,
        "d5_splits_only_at_actual_nonunits": True,
        "target_factors": factor_records,
        "finite_materialization_receives_asymptotic_credit": False,
        "candidate_discrete_log_oracle_consumed": False,
    }


def theorem_record() -> dict[str, str]:
    return {
        "target_factor_decomposition": (
            "For each retained target T_j, let g_j mod U be the R174 signed "
            "deflated-line norm over the selected divisor. Then gcd(U,product_j "
            "g_j) is exactly the R174 and R178 first candidate factor G_1."
        ),
        "d5_interface": (
            "Dahan-Moreno Maza-Schost-Xie compute in a direct product of fields "
            "by splitting at zero divisors. In degree n, one coefficient-ring "
            "multiplication or quasi-inversion costs softly linear work in n; "
            "their half-GCD over coefficient-ring polynomials multiplies this "
            "arithmetic time by softly linear work in the polynomial degree."
        ),
        "directed_evaluation_interface": (
            "Van der Hoeven-Lecerf directed evaluation removes the repeated-"
            "splitting overhead. For one degree-n algebraic parameter, Theorem "
            "4.2 still charges the computation tree's tau_mul multiplications "
            "and tau_div zero-tests or inversions by softly linear algebra work."
        ),
        "survivor_lower_bound": (
            "In the literal target-factor computation tree, every component of "
            "U/G_1 is a unit for every g_j and therefore survives all N factor "
            "steps. Even an optimal order and immediate retirement of every "
            "discovered candidate incurs at least (n-deg G_1)N component-factor "
            "visits. Since deg G_1=B^(3/4) and n=B^(9/4), this is Theta(nN)."
        ),
        "half_gcd_specialization": (
            "Applying the published D5 half-GCD directly to the translated "
            "degree-n selected divisor and a degree-N target witness over the "
            "degree-n product algebra has standard softly n^2 cost because the "
            "outer polynomial degree is n. This is B^(9/2), above rho."
        ),
        "modular_composition_escape": (
            "A genuine escape must compile all N target factors before product-"
            "algebra evaluation, for example into one univariate H(a) mod U. "
            "The cited 2026 generic algebraic modular-composition bound O(n^1.343) "
            "becomes B^3.02175 at n=B^(9/4), above rho. Kedlaya-Umans-type finite-"
            "field near-linear bit complexity could matter only after an exact "
            "one-parameter fold of the elliptic two-parameter kernel is supplied."
        ),
        "scope": (
            "This closes literal factor streaming, successive D5 zero-tests, "
            "directed panoramic evaluation of that same tree, and standard D5 "
            "half-GCD. It is not a lower bound against a one-shot monogenic, "
            "bivariate modular-composition, transposed, or custom circuit compiler."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_d5_directed_evaluation_survivor.cost.r180.v1",
        "selected_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_factor_count_exponent_B": fraction_record(Fraction(5, 4)),
        "candidate_degree_exponent_B": fraction_record(Fraction(3, 4)),
        "noncandidate_survivor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "required_output_exponent_B": fraction_record(Fraction(9, 4)),
        "literal_factor_stream_lower_bound_exponent_B": fraction_record(Fraction(7, 2)),
        "successive_d5_zero_test_exponent_B": fraction_record(Fraction(7, 2)),
        "directed_evaluation_same_tree_exponent_B": fraction_record(Fraction(7, 2)),
        "standard_d5_half_gcd_exponent_B": fraction_record(Fraction(9, 2)),
        "best_cited_generic_algebraic_modcomp_exponent_n": 1.343,
        "best_cited_generic_algebraic_modcomp_exponent_B": 3.02175,
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "literal_factor_stream_strictly_below_rho": False,
        "successive_d5_zero_tests_strictly_below_rho": False,
        "directed_evaluation_same_tree_strictly_below_rho": False,
        "standard_d5_half_gcd_strictly_below_rho": False,
        "cited_generic_algebraic_modcomp_strictly_below_rho": False,
        "finite_field_near_linear_modcomp_fold_applicability_proved": False,
        "one_shot_monogenic_compiler_supplied": False,
        "standard_route_negative_claimed_as_circuit_lower_bound": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    verify_source_bindings()
    r174_report = json.loads(R174_REPORT.read_text())
    r178_report = json.loads(R179.R178_REPORT.read_text())
    r179_report = json.loads(R179_REPORT.read_text())
    if not r179_report["admission"]["direct_truncated_resultant_route_closed"]:
        raise AssertionError("R179 direct truncated-resultant boundary is not closed")
    rows = [
        finite_control(curve, seed, r174_report, r178_report)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_exact = all(
        row["candidate_factor_equals_r174"]
        and row["candidate_factor_equals_r178"]
        and row["target_factor_zero_union_equals_candidate_roots"]
        and row["optimal_order_dynamic_program_exact"]
        and row["optimal_visits_respect_noncandidate_lower_bound"]
        and row["all_final_survivor_factors_are_units"]
        and row["d5_splits_only_at_actual_nonunits"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_d5_directed_evaluation_survivor.controls.r180.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_candidate_factors_equal_r174_and_r178": all(
            row["candidate_factor_equals_r174"]
            and row["candidate_factor_equals_r178"]
            for row in rows
        ),
        "all_target_factor_zero_unions_exact": all(
            row["target_factor_zero_union_equals_candidate_roots"] for row in rows
        ),
        "all_optimal_orders_exact": all(
            row["optimal_order_dynamic_program_exact"] for row in rows
        ),
        "all_survivor_lower_bounds_exact": all(
            row["optimal_visits_respect_noncandidate_lower_bound"] for row in rows
        ),
        "all_final_survivor_factors_are_units": all(
            row["all_final_survivor_factors_are_units"] for row in rows
        ),
        "all_splits_are_actual_nonunits": all(
            row["d5_splits_only_at_actual_nonunits"] for row in rows
        ),
        "selected_divisor_degree_sum": sum(
            row["selected_divisor_degree"] for row in rows
        ),
        "target_factor_count_sum": sum(row["target_factor_count"] for row in rows),
        "candidate_degree_sum": sum(row["candidate_degree"] for row in rows),
        "noncandidate_survivor_degree_sum": sum(
            row["noncandidate_survivor_degree"] for row in rows
        ),
        "materialized_target_factor_residue_slot_count": sum(
            row["materialized_target_factor_residue_slot_count"] for row in rows
        ),
        "finite_pair_evaluation_count": sum(
            row["finite_pair_evaluation_count"] for row in rows
        ),
        "target_factor_zero_incidence_count": sum(
            row["target_factor_zero_incidence_count"] for row in rows
        ),
        "natural_component_factor_visit_count": sum(
            row["natural_order"]["component_factor_visit_count"] for row in rows
        ),
        "optimal_component_factor_visit_count": sum(
            row["optimal_order"]["component_factor_visit_count"] for row in rows
        ),
        "component_factor_visit_lower_bound": sum(
            row["component_factor_visit_lower_bound"] for row in rows
        ),
        "finite_materialization_receives_asymptotic_credit": False,
        "candidate_oracle_consumed": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "eighteen_source_bindings_exact": len(SOURCE_BINDINGS) == 18,
        "r179_squarefree_dynamic_evaluation_gap_inherited": True,
        "d5_product_algebra_interface_bound": True,
        "directed_evaluation_theorem_4_2_interface_bound": True,
        "modular_composition_2026_interface_bound": True,
        "six_r174_r178_candidate_controls_replayed": len(rows) == 6,
        "target_factor_decomposition_complete": controls[
            "all_candidate_factors_equal_r174_and_r178"
        ],
        "target_factor_zero_unions_complete": controls[
            "all_target_factor_zero_unions_exact"
        ],
        "optimal_early_split_orders_complete": controls["all_optimal_orders_exact"],
        "noncandidate_survivor_lower_bounds_complete": controls[
            "all_survivor_lower_bounds_exact"
        ],
        "final_survivor_unit_checks_complete": controls[
            "all_final_survivor_factors_are_units"
        ],
        "splits_charged_only_at_actual_nonunits": controls[
            "all_splits_are_actual_nonunits"
        ],
        "literal_factor_stream_nN_charged": True,
        "successive_d5_zero_tests_nN_charged": True,
        "directed_evaluation_same_tree_nN_charged": True,
        "standard_d5_half_gcd_n2_charged": True,
        "generic_algebraic_modcomp_above_rho_charged": True,
        "scope_limited_without_circuit_lower_bound": True,
        "one_shot_monogenic_compiler_complete": False,
        "finite_field_modcomp_fold_applicability_complete": False,
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
        "ADMIT_D5_DIRECTED_EVALUATION_INTERFACE_SCOPE__SIX_R174_R178_"
        "TARGET_FACTOR_CONTROLS_REPLAYED__OPTIMAL_EARLY_NONUNIT_SPLITS_EXACT__"
        "EVERY_NONCANDIDATE_COMPONENT_SURVIVES_ALL_N_FACTORS__LITERAL_FACTOR_"
        "STREAM_AND_DIRECTED_EVALUATION_B7O2__STANDARD_D5_HALF_GCD_B9O2__"
        "GENERIC_ALGEBRAIC_MODCOMP_B3O02175__ONE_SHOT_MONOGENIC_FINITE_FIELD_"
        "FOLD_OPEN__NO_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Construct or refute one one-shot monogenic compiler for the signed "
        "elliptic translate product. From compact U,V and the degree-N target "
        "Miller SLP, derive H,a with C_h=H(a) mod U, or a bounded-bidegree "
        "G(X,a(X)) mod U, using softly O(n+N) preprocessing and without emitting "
        "N residue elements. Then apply a charged finite-field modular-composition "
        "algorithm and verify G_1 on held-out divisors. Reject a disguised N-step "
        "product-algebra tree, nN coefficients, n^2 pair state, candidate "
        "inversions, and unit-cost composition, norm, resultant, root, count, "
        "marginal, rank, or source oracles."
    )
    passed = sum(obligations.values())
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Determine whether D5 splitting or directed panoramic evaluation "
            "turns the factored R174/R178 target stream into a softly O(n+N) "
            "constructor modulo arbitrary squarefree U."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "controls": controls,
        "cost": cost,
        "literature": {
            "dahan_moreno_maza_schost_xie_2006": {
                "campaign_fit": (
                    "Supplies fast quasi-inversion, splitting, and half-GCD over "
                    "direct products of fields; it does not compress an N-step "
                    "coefficient-ring computation tree to O(1) ring operations."
                )
            },
            "van_der_hoeven_lecerf_2020": {
                "campaign_fit": (
                    "Supplies fast panoramic evaluation without repeated-split "
                    "overhead while retaining tau_mul and tau_div in Theorem 4.2."
                )
            },
            "neiger_salvy_schost_villard_2026": {
                "campaign_fit": (
                    "The generic algebraic modular-composition exponent is charged; "
                    "no one-parameter elliptic fold is attributed to this paper."
                )
            },
            "novelty_scope": (
                "R180 contributes the exact noncandidate-survivor invariant and "
                "optimal finite split audit for the campaign operator. It claims "
                "no new D5 or modular-composition theorem."
            ),
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": passed,
            "obligation_count": len(obligations),
            "d5_directed_interface_scope_admitted": all_exact,
            "literal_d5_factor_stream_route_closed": all_exact,
            "standard_d5_half_gcd_route_closed": all_exact,
            "one_shot_monogenic_compiler_admitted": False,
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
        "schema": "p1553.m6_d5_directed_evaluation_survivor.frozen.r180.v1",
        "source_bindings": source_binding_records(),
        "interfaces": {
            "coefficient_algebra": "F[X]/(U)_squarefree_degree_n",
            "literal_computation_tree": "N_target_norm_factors_with_early_nonunit_splits",
            "closed_standard_routes": [
                "literal_factor_stream",
                "successive_d5_zero_tests",
                "directed_evaluation_of_same_tree",
                "standard_d5_half_gcd",
            ],
            "open_interface": "one_shot_monogenic_finite_field_modcomp_compiler",
            "required_total_work": "softly_O(n+N)",
        },
        "promotion_allowed": False,
    }
    replay = {
        "schema": "p1553.m6_d5_directed_evaluation_survivor.replay.r180.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "target_factor_product_mod_u_sha256": row[
                    "target_factor_product_mod_u_sha256"
                ],
                "natural_candidate_factor_sha256": row["natural_order"][
                    "candidate_factor_sha256"
                ],
                "optimal_candidate_factor_sha256": row["optimal_order"][
                    "candidate_factor_sha256"
                ],
                "optimal_order": row["optimal_order"]["order"],
                "optimal_component_factor_visit_count": row["optimal_order"][
                    "component_factor_visit_count"
                ],
                "component_factor_visit_lower_bound": row[
                    "component_factor_visit_lower_bound"
                ],
            }
            for row in rows
        ],
    }
    applicability = {
        "schema": "p1553.m6_d5_directed_evaluation_survivor.analysis.r180.v1",
        "published_interfaces": {
            "d5": theorem["d5_interface"],
            "directed_evaluation": theorem["directed_evaluation_interface"],
            "modular_composition_escape": theorem["modular_composition_escape"],
        },
        "closed_routes": {
            "literal_target_factor_stream": "B^(7/2)",
            "successive_d5_zero_tests": "B^(7/2)",
            "directed_evaluation_same_tree": "B^(7/2)",
            "standard_d5_half_gcd": "B^(9/2)",
            "best_cited_generic_algebraic_modcomp_after_hypothetical_fold": "B^3.02175",
        },
        "open_interface": {
            "name": "one_shot_monogenic_finite_field_modcomp_compiler",
            "required_exponent_B": "9/4+o(1)",
            "one_parameter_fold_supplied": False,
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
        f"{admission['obligation_count']} literal_d5_closed=1 breakthrough=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
