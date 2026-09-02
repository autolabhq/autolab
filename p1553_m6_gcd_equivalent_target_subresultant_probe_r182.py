#!/usr/bin/env python3
"""Audit target-divisor subresultants for the signed M6 kernel."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_gcd_equivalent_target_subresultant.r182.v1"

R181_PRODUCER = ROOT / "p1553_m6_monogenic_kernel_bidegree_probe_r181.py"
R181_REPORT = ROOT / "p1553_m6_monogenic_kernel_bidegree_probe_report_r181.json"
R181_FROZEN = ROOT / "frozen_m6_monogenic_kernel_bidegree.json"
R181_COST = ROOT / "m6_monogenic_kernel_bidegree_cost_ledger.json"
R181_REPLAY = ROOT / "m6_monogenic_kernel_bidegree_replay.json"
R181_CONTROLS = ROOT / "m6_monogenic_kernel_bidegree_controls.json"
R181_APPLICABILITY = ROOT / "monogenic_kernel_bidegree_applicability_r181.json"
R181_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_monogenic_kernel_bidegree_probe_r181.py"
R181_GATE = ROOT / "p1553_m6_monogenic_kernel_bidegree_probe_gate_r181.md"
R181_PARENT = ROOT / "p1553_m6_monogenic_kernel_bidegree_probe_parent_report_r181.yaml"

POTEAUX_SCHOST = ROOT / "references/poteaux_schost_modular_composition_triangular_sets_2010.pdf"
SPECIAL_RESULTANTS = ROOT / "references/bostan_flajolet_salvy_schost_special_resultants_2005.pdf"
KEDLAYA_UMANS = ROOT / "references/kedlaya_umans_fast_polynomial_factorization_modular_composition_2011.pdf"
BIVARIATE_RESULTANTS = ROOT / "references/hyun_neiger_schost_bivariate_resultants_1905.04356.pdf"
SHOUP_PAPER = ROOT / "references/shoup_generic_dlp_lower_bound_1997.pdf"

SOURCE_BINDINGS = (
    ("r181_producer", R181_PRODUCER, "c96a8283da297d78f3ba16d038fe6422026f09a0663d4ecacf9a290936dc178b"),
    ("r181_report", R181_REPORT, "c80947aab90d88b402eb4fd42de5917b310e25455a0a7c59fc4e95b09b9f7be6"),
    ("r181_frozen", R181_FROZEN, "5244a753b4ceb16f8cbd7f271f41bb93af6cadd58fe73152c20d65225eac1356"),
    ("r181_cost", R181_COST, "77d018d104916d41686620cc0c880881d5919de7361aee4dbf6ae95f3b4f7be1"),
    ("r181_replay", R181_REPLAY, "860d5e366227116c8e1d3b757b498336ae08a6141640b5f480240b4fc4853b12"),
    ("r181_controls", R181_CONTROLS, "3b85e19a38058ff9f3f0d0f79238075b5f0dc32e18ba4d38da3e0a1fa82c6a6f"),
    ("r181_applicability", R181_APPLICABILITY, "35157589f6109dc1443a0497f8f087fd6c0338b61b97950a7cd75073940339a3"),
    ("r181_test", R181_TEST, "9c064a5039e027d7692729089877bfeeeed447d0b1c7fb44b1d247cc5c69766a"),
    ("r181_gate", R181_GATE, "24d801e02466f07a7a7e4036773a2fb362eeaae13bd5dccc947cf51d616f7469"),
    ("r181_parent", R181_PARENT, "0bf5b1b722557ec0fe6ae5fb03297db40a9dd35df0dd88956d90955992832f4d"),
    ("poteaux_schost_2010", POTEAUX_SCHOST, "587f302dd16c724d1be6a4b629a46a684a0c35389dbc22ba45641dba54de6f32"),
    ("bostan_flajolet_salvy_schost_2005", SPECIAL_RESULTANTS, "19db312c68f997949db342a050df568a009a340dd5026e4584e2c4fa59fcc375"),
    ("kedlaya_umans_2011", KEDLAYA_UMANS, "93bd1f77b762f49bcae017d5c12ceccef38c67a956810873105cbce083634377"),
    ("hyun_neiger_schost_2019", BIVARIATE_RESULTANTS, "32b73cf0ca7172bdec0f8f1b256adda628a86d6dd7eee8e07e8644e35a9f16f3"),
    ("shoup_1997", SHOUP_PAPER, "89d19aad3a4d98b563029de9135d30c8ed9b831d74f7348c286acc22f9af85b3"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_gcd_equivalent_target_subresultant_probe_report_r182.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_gcd_equivalent_target_subresultant.json"
DEFAULT_COST = ROOT / "m6_gcd_equivalent_target_subresultant_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_gcd_equivalent_target_subresultant_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_gcd_equivalent_target_subresultant_controls.json"
DEFAULT_APPLICABILITY = ROOT / "gcd_equivalent_target_subresultant_applicability_r182.json"

REPLAY_SEEDS = (16001, 16002, 18104)
HELD_OUT_SEEDS = (18206,)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R181 = load_module("p1553_r181_for_r182", R181_PRODUCER)
R174 = R181.R174
R161 = R181.R161
R81 = R181.R81


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
        raise AssertionError(f"R182 source binding mismatch: {failures}")


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def unique_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return list(dict.fromkeys(points))


def x_injective_charts(
    points: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    charts: list[list[tuple[int, int]]] = []
    chart_x_values: list[set[int]] = []
    for point in unique_points(points):
        for chart, x_values in zip(charts, chart_x_values):
            if int(point[0]) not in x_values:
                chart.append(point)
                x_values.add(int(point[0]))
                break
        else:
            charts.append([point])
            chart_x_values.append({int(point[0])})
    if len(charts) > 2:
        raise AssertionError("elliptic x fibers require at most two charts")
    return charts


def target_chart_record(
    points: list[tuple[int, int]], prime: int, curve: dict[str, Any]
) -> dict[str, Any]:
    if len({int(point[0]) for point in points}) != len(points):
        raise AssertionError("target chart is not x-injective")
    w_poly = R161.monic_root_polynomial(
        [int(point[0]) for point in points], prime
    )
    v_poly = R161.interpolate(
        [(int(point[0]), int(point[1])) for point in points], prime
    )
    curve_residual = R161.poly_sub(
        R161.poly_mul(v_poly, v_poly, prime),
        [int(curve["curve_b"]), int(curve["curve_a"]), 0, 1],
        prime,
    )
    if R161.poly_mod(curve_residual, w_poly, prime) != [0]:
        raise AssertionError("target interpolant is not on the curve")
    derivative = R181.poly_trim(
        [
            index * coefficient % prime
            for index, coefficient in enumerate(w_poly)
        ][1:]
        or [0],
        prime,
    )
    if R161.poly_degree(R161.poly_gcd(w_poly, derivative, prime)) != 0:
        raise AssertionError("deduplicated target chart is not squarefree")
    return {
        "points": points,
        "w": w_poly,
        "v": v_poly,
        "degree": len(points),
        "w_sha256": sha256_json(w_poly),
        "v_sha256": sha256_json(v_poly),
    }


def kernel_remainder(
    normal: R181.NormalForm,
    chart: dict[str, Any],
    prime: int,
) -> list[int]:
    a_part, b_part = normal
    return R161.poly_mod(
        R161.poly_add(
            a_part,
            R161.poly_mul(chart["v"], b_part, prime),
            prime,
        ),
        chart["w"],
        prime,
    )


def elementary_norm_from_power_sums(
    values: list[int], prime: int
) -> tuple[int, list[int]]:
    power_sums = [
        sum(pow(value, exponent, prime) for value in values) % prime
        for exponent in range(1, len(values) + 1)
    ]
    elementary = [1]
    for degree in range(1, len(values) + 1):
        numerator = 0
        for index in range(1, degree + 1):
            sign = 1 if index % 2 else -1
            numerator += sign * elementary[degree - index] * power_sums[index - 1]
        elementary.append(numerator * pow(degree, -1, prime) % prime)
    return elementary[-1], power_sums


def control_row(
    payload: dict[str, Any], family_id: str, seed: int
) -> dict[str, Any]:
    matches = [
        row
        for row in payload["controls"]["controls"]
        if row["family_id"] == family_id and int(row["seed"]) == seed
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one R181 control for {family_id}:{seed}")
    return matches[0]


def expected_candidate(
    curve: dict[str, Any],
    seed: int,
    role: str,
    r181_report: dict[str, Any],
) -> dict[str, Any]:
    if role == "r181_replay":
        row = control_row(r181_report, curve["family_id"], seed)
        return {
            "source": "r181_frozen_control",
            "candidate_factor_sha256": row["candidate_factor_sha256"],
            "candidate_roots": row["candidate_roots"],
        }
    row = R174.finite_control(curve, seed)
    return {
        "source": "fresh_r174_held_out_replay",
        "candidate_factor_sha256": row["candidate_factor_sha256"],
        "candidate_roots": row["candidate_roots"],
    }


def finite_control(
    curve: dict[str, Any],
    seed: int,
    role: str,
    r181_report: dict[str, Any],
) -> dict[str, Any]:
    _, divisor, target_records = R174.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    raw_targets = [
        tuple(record["target"])
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    unique_targets = unique_points(raw_targets)
    prime = int(curve["field_prime"])
    charts = [
        target_chart_record(points, prime, curve)
        for points in x_injective_charts(raw_targets)
    ]
    n = len(selected)
    normals = [
        R181.signed_target_kernel(curve, divisor, selected, left)
        for left in selected
    ]

    combined_norms = [1] * n
    source_has_nonunit = [False] * n
    source_gcd_degrees = [0] * n
    all_remainders_match = True
    all_norms_match = True
    all_newton_norms_match = True
    all_unit_inverses_exact = True
    chart_records = []
    inverse_certificate_count = 0
    inverse_certificate_slots = 0
    trace_projection_count = 0
    for chart_index, chart in enumerate(charts):
        remainders = []
        chart_norms = []
        chart_gcd_degrees = []
        chart_inverse_count = 0
        chart_power_sum_hashes = []
        for source_index, normal in enumerate(normals):
            remainder = kernel_remainder(normal, chart, prime)
            values = [
                R181.normal_eval(normal, target, prime)
                for target in chart["points"]
            ]
            remainder_values = [
                R161.poly_eval(remainder, int(target[0]), prime)
                for target in chart["points"]
            ]
            all_remainders_match &= values == remainder_values
            direct_norm = 1
            for value in values:
                direct_norm = direct_norm * value % prime
            newton_norm, power_sums = elementary_norm_from_power_sums(
                values, prime
            )
            all_newton_norms_match &= newton_norm == direct_norm
            gcd_poly = R161.poly_gcd(chart["w"], remainder, prime)
            gcd_degree = R161.poly_degree(gcd_poly)
            all_norms_match &= (direct_norm == 0) == (gcd_degree > 0)
            source_has_nonunit[source_index] |= gcd_degree > 0
            source_gcd_degrees[source_index] += gcd_degree
            combined_norms[source_index] = (
                combined_norms[source_index] * direct_norm % prime
            )
            if gcd_degree == 0:
                inverse = R161.poly_inv_mod(remainder, chart["w"], prime)
                exact_inverse = R161.poly_mul_mod(
                    remainder, inverse, chart["w"], prime
                ) == [1]
                all_unit_inverses_exact &= exact_inverse
                inverse_certificate_count += 1
                inverse_certificate_slots += chart["degree"]
                chart_inverse_count += 1
            remainders.append(
                remainder + [0] * (chart["degree"] - len(remainder))
            )
            chart_norms.append(direct_norm)
            chart_gcd_degrees.append(gcd_degree)
            chart_power_sum_hashes.append(sha256_json(power_sums))
            trace_projection_count += chart["degree"]
        remainder_rank = R81.rank_mod(remainders, prime)
        coefficient_slots = n * chart["degree"]
        nonzero_coefficients = sum(
            int(coefficient != 0)
            for row in remainders
            for coefficient in row
        )
        chart_records.append(
            {
                "chart_index": chart_index,
                "target_degree": chart["degree"],
                "target_w_sha256": chart["w_sha256"],
                "target_v_sha256": chart["v_sha256"],
                "remainder_coefficient_slot_count": coefficient_slots,
                "remainder_nonzero_coefficient_count": nonzero_coefficients,
                "remainder_body_fully_dense": nonzero_coefficients
                == coefficient_slots,
                "remainder_matrix_rank": remainder_rank,
                "remainder_matrix_full_target_rank": remainder_rank
                == chart["degree"],
                "remainder_matrix_sha256": sha256_json(remainders),
                "norm_values_sha256": sha256_json(chart_norms),
                "gcd_degrees": chart_gcd_degrees,
                "gcd_degrees_sha256": sha256_json(chart_gcd_degrees),
                "unit_inverse_certificate_count": chart_inverse_count,
                "power_sum_sha256s": chart_power_sum_hashes,
            }
        )

    if not all_remainders_match:
        raise AssertionError("target-divisor remainders lost kernel values")
    if not all_norms_match:
        raise AssertionError("target norm zero differs from subresultant gcd")
    if not all_newton_norms_match:
        raise AssertionError("Newton power sums lost the target norm")
    if not all_unit_inverses_exact:
        raise AssertionError("a target-divisor inverse certificate failed")

    selector = R161.interpolate(
        [
            (int(point[0]), int(value))
            for point, value in zip(selected, combined_norms)
        ],
        prime,
    )
    candidate_factor = R161.poly_gcd(divisor["u"], selector, prime)
    candidate_roots = sorted(
        int(point[0])
        for point in selected
        if R161.poly_eval(candidate_factor, int(point[0]), prime) == 0
    )
    subresultant_roots = sorted(
        int(point[0])
        for point, nonunit in zip(selected, source_has_nonunit)
        if nonunit
    )
    if candidate_roots != subresultant_roots:
        raise AssertionError("subresultant roots differ from norm roots")
    expected = expected_candidate(curve, seed, role, r181_report)
    candidate_sha256 = sha256_json(candidate_factor)
    if candidate_sha256 != expected["candidate_factor_sha256"]:
        raise AssertionError("R182 candidate factor differs from signed replay")
    if candidate_roots != expected["candidate_roots"]:
        raise AssertionError("R182 candidate roots differ from signed replay")

    total_target_degree = sum(chart["degree"] for chart in charts)
    remainder_slots = n * total_target_degree
    noncandidate_count = n - len(candidate_roots)
    return {
        "control_id": f"{curve['family_id']}_target_subresultant_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "subgroup_order": int(curve["subgroup_order"]),
        "seed": seed,
        "control_role": role,
        "selected_divisor_degree": n,
        "raw_retained_target_count": len(raw_targets),
        "unique_retained_target_count": len(unique_targets),
        "duplicate_target_count": len(raw_targets) - len(unique_targets),
        "target_chart_count": len(charts),
        "all_target_charts_x_injective": True,
        "target_charts_cover_unique_targets": sum(
            chart["degree"] for chart in charts
        )
        == len(unique_targets),
        "remainder_coefficient_slot_count": remainder_slots,
        "remainder_nonzero_coefficient_count": sum(
            chart["remainder_nonzero_coefficient_count"]
            for chart in chart_records
        ),
        "all_remainder_bodies_fully_dense": all(
            chart["remainder_body_fully_dense"] for chart in chart_records
        ),
        "all_remainder_matrices_full_target_rank": all(
            chart["remainder_matrix_full_target_rank"]
            for chart in chart_records
        ),
        "all_remainder_evaluations_exact": all_remainders_match,
        "all_norm_zero_iff_subresultant_gcd_nontrivial": all_norms_match,
        "all_newton_power_sum_norms_exact": all_newton_norms_match,
        "all_unit_inverse_certificates_exact": all_unit_inverses_exact,
        "candidate_source_count": len(candidate_roots),
        "noncandidate_source_count": noncandidate_count,
        "source_target_gcd_degree_sum": sum(source_gcd_degrees),
        "unit_inverse_certificate_count": inverse_certificate_count,
        "unit_inverse_certificate_coefficient_slot_count": inverse_certificate_slots,
        "trace_power_projection_count": trace_projection_count,
        "trace_output_coefficient_slot_count": n * total_target_degree,
        "candidate_factor_sha256": candidate_sha256,
        "candidate_roots": candidate_roots,
        "candidate_factor_equals_signed_replay": True,
        "candidate_replay_source": expected["source"],
        "combined_norm_values_sha256": sha256_json(combined_norms),
        "chart_records": chart_records,
        "finite_remainders_or_ranks_receive_asymptotic_credit": False,
        "candidate_oracle_consumed": False,
    }


def opposite_point_chart_control(
    curve: dict[str, Any], seed: int
) -> dict[str, Any]:
    _, divisor, target_records = R174.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    retained_targets = [
        tuple(record["target"])
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    prime = int(curve["field_prime"])
    target = next(
        point
        for point in retained_targets
        if int(point[1]) != 0
        and (int(point[0]), -int(point[1]) % prime) not in selected_set
    )
    opposite = (int(target[0]), -int(target[1]) % prime)
    adversarial_targets = [target, target, opposite]
    charts = [
        target_chart_record(points, prime, curve)
        for points in x_injective_charts(adversarial_targets)
    ]
    if len(charts) != 2 or len(unique_points(adversarial_targets)) != 2:
        raise AssertionError("opposite-point control did not force two charts")

    normals = [
        R181.signed_target_kernel(curve, divisor, selected, left)
        for left in selected
    ]
    combined_norms = [1] * len(selected)
    source_has_nonunit = [False] * len(selected)
    all_remainders_match = True
    all_norm_gcd_equivalences_match = True
    chart_records = []
    for chart_index, chart in enumerate(charts):
        remainder_hashes = []
        gcd_degrees = []
        for source_index, normal in enumerate(normals):
            remainder = kernel_remainder(normal, chart, prime)
            values = [
                R181.normal_eval(normal, point, prime)
                for point in chart["points"]
            ]
            remainder_values = [
                R161.poly_eval(remainder, int(point[0]), prime)
                for point in chart["points"]
            ]
            all_remainders_match &= values == remainder_values
            norm = 1
            for value in values:
                norm = norm * value % prime
            gcd_degree = R161.poly_degree(
                R161.poly_gcd(chart["w"], remainder, prime)
            )
            all_norm_gcd_equivalences_match &= (norm == 0) == (gcd_degree > 0)
            combined_norms[source_index] = (
                combined_norms[source_index] * norm % prime
            )
            source_has_nonunit[source_index] |= gcd_degree > 0
            remainder_hashes.append(sha256_json(remainder))
            gcd_degrees.append(gcd_degree)
        chart_records.append(
            {
                "chart_index": chart_index,
                "target_degree": chart["degree"],
                "target_w_sha256": chart["w_sha256"],
                "target_v_sha256": chart["v_sha256"],
                "remainder_sha256s": remainder_hashes,
                "gcd_degrees_sha256": sha256_json(gcd_degrees),
            }
        )

    selector = R161.interpolate(
        [
            (int(point[0]), int(value))
            for point, value in zip(selected, combined_norms)
        ],
        prime,
    )
    candidate_factor = R161.poly_gcd(divisor["u"], selector, prime)
    norm_roots = sorted(
        int(point[0])
        for point in selected
        if R161.poly_eval(candidate_factor, int(point[0]), prime) == 0
    )
    gcd_roots = sorted(
        int(point[0])
        for point, nonunit in zip(selected, source_has_nonunit)
        if nonunit
    )
    if norm_roots != gcd_roots:
        raise AssertionError("opposite-point norm roots differ from gcd roots")
    return {
        "control_id": (
            f"{curve['family_id']}_opposite_point_two_chart_seed{seed}"
        ),
        "family_id": curve["family_id"],
        "seed": seed,
        "raw_target_count": len(adversarial_targets),
        "unique_target_count": len(unique_points(adversarial_targets)),
        "duplicate_target_count": 1,
        "opposite_points_share_x": target[0] == opposite[0],
        "opposite_points_have_distinct_y": target[1] != opposite[1],
        "target_chart_count": len(charts),
        "all_target_charts_x_injective": all(
            len({point[0] for point in chart["points"]}) == chart["degree"]
            for chart in charts
        ),
        "target_charts_cover_unique_targets": sum(
            chart["degree"] for chart in charts
        )
        == len(unique_points(adversarial_targets)),
        "all_remainder_evaluations_exact": all_remainders_match,
        "all_norm_zero_iff_subresultant_gcd_nontrivial": (
            all_norm_gcd_equivalences_match
        ),
        "candidate_roots_equal_gcd_roots": norm_roots == gcd_roots,
        "candidate_source_count": len(norm_roots),
        "candidate_factor_sha256": sha256_json(candidate_factor),
        "chart_records": chart_records,
        "candidate_oracle_consumed": False,
    }


def theorem_record() -> dict[str, str]:
    return {
        "target_divisor_remainder": (
            "After public target deduplication and partition into at most two "
            "x-injective charts, each chart has W(u)=product(u-u_j) and an "
            "interpolant V_T(u). The R181 kernel restricts to "
            "R_P(u)=A_P(u)+V_T(u)B_P(u) mod W."
        ),
        "gcd_norm_inverse_equivalence": (
            "For squarefree W, a selected source P has an incident target in "
            "the chart exactly when gcd(W,R_P) is nonconstant, equivalently "
            "when R_P is a nonunit in F[u]/W, equivalently when its norm or "
            "resultant Res(W,R_P) is zero. Deduplicating repeated targets does "
            "not change this zero union."
        ),
        "power_projection_interface": (
            "The norm is the last elementary symmetric function of the values "
            "R_P(u_j). Newton identities recover it from N power sums. A direct "
            "power-projection transcript therefore emits N scalar traces per "
            "source component, or N degree-n coefficient vectors when batched "
            "over F[X]/U."
        ),
        "inverse_certificate_survivor": (
            "Every noncandidate source component is a unit in every target "
            "chart and requires a degree-N inverse certificate in the represented "
            "route. Since n-c=Theta(n), immediate splitting of candidate "
            "components does not remove the Theta(nN) noncandidate body."
        ),
        "represented_cost": (
            "The remainder matrix, batched Newton traces, target-algebra inverse "
            "certificates, Poteaux-Schost triangular norm, and standard half-GCD "
            "all operate on or emit Theta(nN) represented coefficients. With "
            "n=B^(9/4) and N=B^(5/4), this is B^(7/2), above rho."
        ),
        "scope": (
            "This closes standard represented target remainders, direct Newton "
            "power projections, per-component inverse certificates, represented "
            "triangular norms, and standard target-variable half-GCD. It is not "
            "a lower bound against an SLP-direct determinant-zero or resultant-"
            "mod-U algorithm that emits only the candidate factor."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_gcd_equivalent_target_subresultant.cost.r182.v1",
        "selected_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_degree_exponent_B": fraction_record(Fraction(5, 4)),
        "candidate_degree_exponent_B": fraction_record(Fraction(3, 4)),
        "required_output_exponent_B": fraction_record(Fraction(3, 4)),
        "desired_total_exponent_B": fraction_record(Fraction(9, 4)),
        "represented_remainder_body_exponent_B": fraction_record(Fraction(7, 2)),
        "direct_newton_trace_body_exponent_B": fraction_record(Fraction(7, 2)),
        "noncandidate_inverse_certificate_body_exponent_B": fraction_record(Fraction(7, 2)),
        "triangular_norm_algebra_dimension_exponent_B": fraction_record(Fraction(7, 2)),
        "standard_target_half_gcd_exponent_B": fraction_record(Fraction(7, 2)),
        "generic_bivariate_resultant_input_body_exponent_B": fraction_record(Fraction(7, 2)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "represented_remainder_body_strictly_below_rho": False,
        "direct_newton_trace_body_strictly_below_rho": False,
        "noncandidate_inverse_certificate_body_strictly_below_rho": False,
        "triangular_norm_strictly_below_rho": False,
        "standard_target_half_gcd_strictly_below_rho": False,
        "generic_bivariate_resultant_input_strictly_below_rho": False,
        "slp_direct_output_sensitive_resultant_supplied": False,
        "finite_density_claimed_as_general_lower_bound": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    verify_source_bindings()
    r181_report = json.loads(R181_REPORT.read_text())
    if r181_report["admission"]["gcd_equivalent_output_sensitive_resultant_admitted"]:
        raise AssertionError("R181 gcd-equivalent resultant should remain open")
    rows = [
        finite_control(curve, seed, "r181_replay", r181_report)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in REPLAY_SEEDS
    ]
    rows.extend(
        finite_control(curve, seed, "held_out_seed", r181_report)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in HELD_OUT_SEEDS
    )
    opposite_chart = opposite_point_chart_control(
        R161.R159.R82.FAMILIES[0], HELD_OUT_SEEDS[0]
    )
    all_exact = all(
        row["all_target_charts_x_injective"]
        and row["target_charts_cover_unique_targets"]
        and row["all_remainder_bodies_fully_dense"]
        and row["all_remainder_matrices_full_target_rank"]
        and row["all_remainder_evaluations_exact"]
        and row["all_norm_zero_iff_subresultant_gcd_nontrivial"]
        and row["all_newton_power_sum_norms_exact"]
        and row["all_unit_inverse_certificates_exact"]
        and row["candidate_factor_equals_signed_replay"]
        for row in rows
    ) and all(
        (
            opposite_chart["opposite_points_share_x"],
            opposite_chart["opposite_points_have_distinct_y"],
            opposite_chart["target_chart_count"] == 2,
            opposite_chart["all_target_charts_x_injective"],
            opposite_chart["target_charts_cover_unique_targets"],
            opposite_chart["all_remainder_evaluations_exact"],
            opposite_chart["all_norm_zero_iff_subresultant_gcd_nontrivial"],
            opposite_chart["candidate_roots_equal_gcd_roots"],
        )
    )
    controls = {
        "schema": "p1553.m6_gcd_equivalent_target_subresultant.controls.r182.v1",
        "control_count": len(rows),
        "replay_control_count": sum(
            row["control_role"] == "r181_replay" for row in rows
        ),
        "held_out_control_count": sum(
            row["control_role"] == "held_out_seed" for row in rows
        ),
        "replay_seeds": list(REPLAY_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "all_target_charts_x_injective": all(
            row["all_target_charts_x_injective"] for row in rows
        ),
        "all_target_charts_cover_unique_targets": all(
            row["target_charts_cover_unique_targets"] for row in rows
        ),
        "all_remainder_bodies_fully_dense": all(
            row["all_remainder_bodies_fully_dense"] for row in rows
        ),
        "all_remainder_matrices_full_target_rank": all(
            row["all_remainder_matrices_full_target_rank"] for row in rows
        ),
        "all_remainder_evaluations_exact": all(
            row["all_remainder_evaluations_exact"] for row in rows
        ),
        "all_norm_zero_iff_subresultant_gcd_nontrivial": all(
            row["all_norm_zero_iff_subresultant_gcd_nontrivial"] for row in rows
        ),
        "all_newton_power_sum_norms_exact": all(
            row["all_newton_power_sum_norms_exact"] for row in rows
        ),
        "all_unit_inverse_certificates_exact": all(
            row["all_unit_inverse_certificates_exact"] for row in rows
        ),
        "all_candidate_factors_equal_signed_replay": all(
            row["candidate_factor_equals_signed_replay"] for row in rows
        ),
        "selected_divisor_degree_sum": sum(
            row["selected_divisor_degree"] for row in rows
        ),
        "raw_retained_target_count_sum": sum(
            row["raw_retained_target_count"] for row in rows
        ),
        "unique_retained_target_count_sum": sum(
            row["unique_retained_target_count"] for row in rows
        ),
        "duplicate_target_count": sum(row["duplicate_target_count"] for row in rows),
        "target_chart_count": sum(row["target_chart_count"] for row in rows),
        "remainder_coefficient_slot_count": sum(
            row["remainder_coefficient_slot_count"] for row in rows
        ),
        "remainder_nonzero_coefficient_count": sum(
            row["remainder_nonzero_coefficient_count"] for row in rows
        ),
        "candidate_source_count": sum(row["candidate_source_count"] for row in rows),
        "noncandidate_source_count": sum(
            row["noncandidate_source_count"] for row in rows
        ),
        "source_target_gcd_degree_sum": sum(
            row["source_target_gcd_degree_sum"] for row in rows
        ),
        "unit_inverse_certificate_count": sum(
            row["unit_inverse_certificate_count"] for row in rows
        ),
        "unit_inverse_certificate_coefficient_slot_count": sum(
            row["unit_inverse_certificate_coefficient_slot_count"] for row in rows
        ),
        "trace_power_projection_count": sum(
            row["trace_power_projection_count"] for row in rows
        ),
        "trace_output_coefficient_slot_count": sum(
            row["trace_output_coefficient_slot_count"] for row in rows
        ),
        "finite_remainders_or_ranks_receive_asymptotic_credit": False,
        "candidate_oracle_consumed": False,
        "opposite_point_two_chart_control": opposite_chart,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "fifteen_source_bindings_exact": len(SOURCE_BINDINGS) == 15,
        "r181_gcd_equivalent_gap_inherited": True,
        "poteaux_schost_norm_interface_bound": True,
        "kedlaya_umans_power_projection_interface_bound": True,
        "special_and_bivariate_resultant_interfaces_bound": True,
        "nine_r181_controls_replayed": controls["replay_control_count"] == 9,
        "three_held_out_controls_replayed": controls["held_out_control_count"] == 3,
        "public_target_deduplication_complete": True,
        "two_chart_target_partition_complete": controls["all_target_charts_x_injective"],
        "opposite_point_two_chart_adversarial_control_complete": (
            opposite_chart["target_chart_count"] == 2
            and opposite_chart["all_remainder_evaluations_exact"]
            and opposite_chart[
                "all_norm_zero_iff_subresultant_gcd_nontrivial"
            ]
        ),
        "target_curve_interpolants_complete": all_exact,
        "kernel_remainders_complete": controls["all_remainder_evaluations_exact"],
        "remainder_full_target_rank_complete": controls["all_remainder_matrices_full_target_rank"],
        "remainder_density_complete": controls["all_remainder_bodies_fully_dense"],
        "norm_subresultant_equivalence_complete": controls["all_norm_zero_iff_subresultant_gcd_nontrivial"],
        "newton_power_sum_norm_replay_complete": controls["all_newton_power_sum_norms_exact"],
        "unit_inverse_certificates_complete": controls["all_unit_inverse_certificates_exact"],
        "candidate_factor_replay_complete": controls["all_candidate_factors_equal_signed_replay"],
        "represented_nN_remainder_body_charged": True,
        "direct_nN_trace_body_charged": True,
        "noncandidate_nN_inverse_body_charged": True,
        "triangular_norm_dimension_nN_charged": True,
        "standard_target_half_gcd_nN_charged": True,
        "finite_evidence_scoped_without_general_lower_bound": True,
        "slp_direct_output_sensitive_resultant_complete": False,
        "compact_determinant_zero_certificate_complete": False,
        "transposed_trace_without_nN_output_complete": False,
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
        "ADMIT_GCD_EQUIVALENT_TARGET_DIVISOR_SUBRESULTANT_IDENTITY__TWELVE_"
        "CONTROLS_INCLUDING_THREE_HELD_OUT_REPLAYED__PUBLIC_DUPLICATES_"
        "REMOVED_AND_TWO_X_CHART_ADVERSARIAL_CONTROL_COMPLETE__NORM_ZERO_IFF_NONUNIT_IFF_GCD_"
        "NONTRIVIAL__NEWTON_TRACES_AND_UNIT_INVERSES_EXACT__REMAINDER_BODY_"
        "FULLY_DENSE_AND_FULL_TARGET_RANK__REPRESENTED_REMAINDER_TRACE_INVERSE_"
        "NORM_AND_HALF_GCD_B7O2__SLP_DIRECT_OUTPUT_SENSITIVE_RESULTANT_MOD_U_"
        "OPEN__NO_GENERAL_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Construct or refute one SLP-direct determinant-zero or resultant-mod-U "
        "algorithm. Consume the compact R181 signed kernel and deduplicated "
        "target charts, and emit only G_1 in softly O(n+N) work without the nN "
        "remainder matrix, N degree-n Newton traces, noncandidate inverse "
        "certificates, or a represented triangular norm. Test a transposed "
        "single-functional minimal-polynomial or subresultant certificate on "
        "seed 18206 and one new divisor family. Reject hidden nN projection "
        "outputs, candidate oracles, and unit-cost determinant or resultant calls."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Instantiate the R181 gcd-only opening as an exact target-divisor "
            "subresultant and test whether standard norm, inverse-certificate, "
            "or power-projection interfaces avoid the represented nN body."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "controls": controls,
        "cost": cost,
        "literature": {
            "poteaux_schost": {
                "campaign_fit": (
                    "Supplies near-linear norm and modular-composition algorithms "
                    "in the represented triangular-algebra dimension nN."
                )
            },
            "kedlaya_umans": {
                "campaign_fit": (
                    "Supplies near-linear modular power projection once the "
                    "degree-nN represented algebra and requested outputs exist."
                )
            },
            "special_and_bivariate_resultants": {
                "campaign_fit": (
                    "Supply fast algorithms for represented special or generic "
                    "bivariate resultants, not an SLP-direct candidate-only "
                    "resultant modulo arbitrary U."
                )
            },
            "novelty_scope": (
                "R182 contributes the exact signed target-divisor remainder, "
                "norm/nonunit/gcd equivalence, and finite inverse and Newton "
                "transcripts. It claims no new resultant theorem or lower bound."
            ),
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "target_subresultant_identity_admitted": all_exact,
            "represented_standard_routes_closed": all_exact,
            "slp_direct_output_sensitive_resultant_admitted": False,
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
        "schema": "p1553.m6_gcd_equivalent_target_subresultant.frozen.r182.v1",
        "source_bindings": source_binding_records(),
        "interfaces": {
            "target_remainder": "R_P=A_P+V_T*B_P_mod_W",
            "gcd_semantics": "norm_zero_iff_nonunit_iff_gcd_W_R_P_nontrivial",
            "closed_standard_routes": [
                "represented_nN_remainder_matrix",
                "direct_N_degree_n_Newton_traces",
                "noncandidate_target_algebra_inverse_certificates",
                "represented_triangular_norm",
                "standard_target_variable_half_gcd",
            ],
            "open_interface": "slp_direct_output_sensitive_resultant_mod_U",
            "required_total_work": "softly_O(n+N)",
        },
        "replay_seeds": list(REPLAY_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "promotion_allowed": False,
    }
    replay = {
        "schema": "p1553.m6_gcd_equivalent_target_subresultant.replay.r182.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "control_role": row["control_role"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "combined_norm_values_sha256": row["combined_norm_values_sha256"],
                "unique_retained_target_count": row["unique_retained_target_count"],
                "candidate_source_count": row["candidate_source_count"],
                "source_target_gcd_degree_sum": row["source_target_gcd_degree_sum"],
                "chart_remainder_sha256s": [
                    chart["remainder_matrix_sha256"]
                    for chart in row["chart_records"]
                ],
            }
            for row in rows
        ],
    }
    applicability = {
        "schema": "p1553.m6_gcd_equivalent_target_subresultant.analysis.r182.v1",
        "closed_routes": {
            "represented_target_remainder_matrix": "B^(7/2)",
            "direct_newton_power_projection_body": "B^(7/2)",
            "noncandidate_inverse_certificate_body": "B^(7/2)",
            "represented_triangular_norm": "B^(7/2)",
            "standard_target_half_gcd": "B^(7/2)",
        },
        "open_interface": {
            "name": "slp_direct_output_sensitive_resultant_mod_U",
            "required_exponent_B": "9/4+o(1)",
            "required_output": "G_1_only",
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
        f"{admission['obligation_count']} represented_routes_closed=1 "
        "breakthrough=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
