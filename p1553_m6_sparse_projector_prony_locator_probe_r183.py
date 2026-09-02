#!/usr/bin/env python3
"""Test sparse Prony recovery from target-subresultant projector moments."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_sparse_projector_prony_locator.r183.v1"

R182_PRODUCER = ROOT / "p1553_m6_gcd_equivalent_target_subresultant_probe_r182.py"
R182_REPORT = ROOT / "p1553_m6_gcd_equivalent_target_subresultant_probe_report_r182.json"
R182_FROZEN = ROOT / "frozen_m6_gcd_equivalent_target_subresultant.json"
R182_COST = ROOT / "m6_gcd_equivalent_target_subresultant_cost_ledger.json"
R182_REPLAY = ROOT / "m6_gcd_equivalent_target_subresultant_replay.json"
R182_CONTROLS = ROOT / "m6_gcd_equivalent_target_subresultant_controls.json"
R182_APPLICABILITY = ROOT / "gcd_equivalent_target_subresultant_applicability_r182.json"
R182_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_gcd_equivalent_target_subresultant_probe_r182.py"
R182_GATE = ROOT / "p1553_m6_gcd_equivalent_target_subresultant_probe_gate_r182.md"
R182_PARENT = ROOT / "p1553_m6_gcd_equivalent_target_subresultant_probe_parent_report_r182.yaml"
R163_GATE = ROOT / "p1553_m6_aggregate_union_factor_label_recovery_probe_gate_r163.md"
R168_GATE = ROOT / "p1553_m6_log_derivative_elliptic_cauchy_trace_probe_gate_r168.md"
R97_GATE = ROOT / "p1553_5a5c_factored_transposed_projector_trace_probe_gate_r97.md"
WIEDEMANN_PAPER = ROOT / "references/wiedemann_sparse_linear_equations_1986.pdf"
KEDLAYA_UMANS = ROOT / "references/kedlaya_umans_fast_polynomial_factorization_modular_composition_2011.pdf"
SHOUP_PAPER = ROOT / "references/shoup_generic_dlp_lower_bound_1997.pdf"

SOURCE_BINDINGS = (
    ("r182_producer", R182_PRODUCER, "0058a28e4c337df97cec7b2e87bb5cc16e18a5b47029ab72d12b852405875e15"),
    ("r182_report", R182_REPORT, "9a38150e9e7e0e41f455b000b4cda390aaff4f428c2828d5ffeed8a9781f7f5d"),
    ("r182_frozen", R182_FROZEN, "29291879484cf68bc16b83e2341818a611bb0c529c53e135f33bc181f30d462e"),
    ("r182_cost", R182_COST, "0eb0f05465fc072c72b6cccc24105d420431eb53162a6d5042b135bdcb454121"),
    ("r182_replay", R182_REPLAY, "bec170f0a0148e269b28c0499f7a8c65b20a3270a8f8dbfc41a0724d3e71e9ba"),
    ("r182_controls", R182_CONTROLS, "850099e9c6f538baea1f94b7d87dafbe8c2df7a83b3dcf9d47be9b233639b77f"),
    ("r182_applicability", R182_APPLICABILITY, "5f1700e9b00d6756daa58d38f36002e7f0dee704632f223de8f4bc55cfce620c"),
    ("r182_test", R182_TEST, "37e7ced472a7eb2244e1539cb3248fd2e170bdec343a89a69b93ce9e05d39a7e"),
    ("r182_gate", R182_GATE, "960fbcdcae0867bc8b02786f81af9eb68ed337d2dfff9019e62c61248ca303fd"),
    ("r182_parent", R182_PARENT, "d83ffeed2d66b78ec40a7b14e3a89fd5a8940775d21f5c297c6105454032ba66"),
    ("r163_gate", R163_GATE, "3064a108bf063ab59a0991be6910b70bf3a2e498974add780f6a57e157a63212"),
    ("r168_gate", R168_GATE, "7063a5f6ba686e38682ab6e29f386ab56e7fb56097e288949cce73e130b2fcae"),
    ("r97_gate", R97_GATE, "ef91d8a15dc7a218c21355b5b2f2777db948978514ebed26d6884ce929c09182"),
    ("wiedemann_1986", WIEDEMANN_PAPER, "8ec0b8a8b35c02bd4a84236129ab991fdbcca4b4feb8332c22b08dbdefc1218a"),
    ("kedlaya_umans_2011", KEDLAYA_UMANS, "93bd1f77b762f49bcae017d5c12ceccef38c67a956810873105cbce083634377"),
    ("shoup_1997", SHOUP_PAPER, "89d19aad3a4d98b563029de9135d30c8ed9b831d74f7348c286acc22f9af85b3"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_sparse_projector_prony_locator_probe_report_r183.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_sparse_projector_prony_locator.json"
DEFAULT_COST = ROOT / "m6_sparse_projector_prony_locator_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_sparse_projector_prony_locator_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_sparse_projector_prony_locator_controls.json"
DEFAULT_APPLICABILITY = ROOT / "sparse_projector_prony_locator_applicability_r183.json"

REPLAY_SEEDS = (16001, 16002, 18104, 18206)
NEW_DIVISOR_SEED = 18307


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R182 = load_module("p1553_r182_for_r183", R182_PRODUCER)
R181 = R182.R181
R174 = R182.R174
R161 = R182.R161
R81 = R182.R81


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
        raise AssertionError(f"R183 source binding mismatch: {failures}")


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def berlekamp_massey(sequence: list[int], prime: int) -> list[int]:
    """Return C with sum_i C[i] s[n-i] = 0 and C[0] = 1."""
    if not sequence:
        return [1]
    connection = [1]
    previous = [1]
    order = 0
    shift = 1
    previous_discrepancy = 1
    for index in range(len(sequence)):
        discrepancy = sequence[index] % prime
        for offset in range(1, order + 1):
            discrepancy = (
                discrepancy
                + connection[offset] * sequence[index - offset]
            ) % prime
        if discrepancy == 0:
            shift += 1
            continue
        saved = connection[:]
        scale = discrepancy * pow(previous_discrepancy, -1, prime) % prime
        required = len(previous) + shift
        if len(connection) < required:
            connection.extend([0] * (required - len(connection)))
        for offset, coefficient in enumerate(previous):
            connection[offset + shift] = (
                connection[offset + shift] - scale * coefficient
            ) % prime
        if 2 * order <= index:
            order = index + 1 - order
            previous = saved
            previous_discrepancy = discrepancy
            shift = 1
        else:
            shift += 1
    return R181.poly_trim(connection[: order + 1], prime)


def locator_from_weighted_moments(
    moments: list[int], prime: int
) -> tuple[list[int], list[int]]:
    connection = berlekamp_massey(moments, prime)
    locator = R181.poly_trim(list(reversed(connection)), prime)
    if locator[-1] != 1:
        inverse = pow(locator[-1], -1, prime)
        locator = [coefficient * inverse % prime for coefficient in locator]
    return locator, connection


def recurrence_is_exact(
    moments: list[int], connection: list[int], prime: int
) -> bool:
    order = len(connection) - 1
    return all(
        sum(
            connection[offset] * moments[index - offset]
            for offset in range(order + 1)
        )
        % prime
        == 0
        for index in range(order, len(moments))
    )


def r182_control_row(
    report: dict[str, Any], family_id: str, seed: int
) -> dict[str, Any]:
    matches = [
        row
        for row in report["controls"]["controls"]
        if row["family_id"] == family_id and int(row["seed"]) == seed
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one R182 control for {family_id}:{seed}")
    return matches[0]


def divisor_from_points(
    points: list[tuple[int, int]], prime: int
) -> dict[str, Any]:
    if len({int(point[0]) for point in points}) != len(points):
        raise AssertionError("strided divisor is not x-injective")
    u_poly = R161.monic_root_polynomial(
        [int(point[0]) for point in points], prime
    )
    v_poly = R161.interpolate(
        [(int(point[0]), int(point[1])) for point in points], prime
    )
    return {
        "records": [{"endpoint": point} for point in points],
        "u": u_poly,
        "v": v_poly,
        "x_injective": True,
        "u_sha256": sha256_json(u_poly),
        "v_sha256": sha256_json(v_poly),
    }


def direct_target_multiplicities(
    selected: list[tuple[int, int]],
    targets: list[tuple[int, int]],
    curve: dict[str, Any],
) -> list[int]:
    target_set = set(targets)
    multiplicities = []
    for left in selected:
        sums = {
            tuple(total)
            for right in selected
            if (total := R161.R70.add(left, right, curve)) is not None
        }
        multiplicities.append(len(sums & target_set))
    return multiplicities


def analyze_instance(
    *,
    curve: dict[str, Any],
    seed: int,
    role: str,
    selected: list[tuple[int, int]],
    divisor: dict[str, Any],
    raw_targets: list[tuple[int, int]],
    expected_roots: list[int] | None,
    expected_factor_sha256: str | None,
    replay_source: str,
) -> dict[str, Any]:
    prime = int(curve["field_prime"])
    unique_targets = R182.unique_points(raw_targets)
    charts = [
        R182.target_chart_record(points, prime, curve)
        for points in R182.x_injective_charts(raw_targets)
    ]
    normals = [
        R181.signed_target_kernel(curve, divisor, selected, left)
        for left in selected
    ]
    direct_multiplicities = direct_target_multiplicities(
        selected, unique_targets, curve
    )

    source_rows = []
    multiplicities = []
    full_spectrum_source_count = 0
    total_distinct_target_values = 0
    for source_index, (left, normal) in enumerate(zip(selected, normals)):
        gcd_degree_sum = 0
        projector_trace_sum = 0
        all_values = []
        chart_rows = []
        for chart_index, chart in enumerate(charts):
            remainder = R182.kernel_remainder(normal, chart, prime)
            values = [
                R181.normal_eval(normal, target, prime)
                for target in chart["points"]
            ]
            remainder_values = [
                R161.poly_eval(remainder, int(target[0]), prime)
                for target in chart["points"]
            ]
            if values != remainder_values:
                raise AssertionError("R183 target remainder evaluation failed")
            projector_values = [
                (1 - pow(value, prime - 1, prime)) % prime
                for value in values
            ]
            projector_trace = sum(projector_values) % prime
            gcd_degree = R161.poly_degree(
                R161.poly_gcd(chart["w"], remainder, prime)
            )
            if projector_trace != gcd_degree:
                raise AssertionError("Fermat projector trace differs from gcd degree")
            gcd_degree_sum += gcd_degree
            projector_trace_sum += projector_trace
            all_values.extend(values)
            chart_rows.append(
                {
                    "chart_index": chart_index,
                    "target_degree": chart["degree"],
                    "gcd_degree": gcd_degree,
                    "fermat_projector_trace": projector_trace,
                    "distinct_value_count": len(set(values)),
                    "value_spectrum_full_degree": len(set(values))
                    == chart["degree"],
                    "values_sha256": sha256_json(values),
                    "projector_sha256": sha256_json(projector_values),
                }
            )
        if projector_trace_sum >= prime:
            raise AssertionError("projector multiplicity wrapped modulo p")
        if gcd_degree_sum != direct_multiplicities[source_index]:
            raise AssertionError("projector multiplicity differs from group-law replay")
        spectrum_degree = len(set(all_values))
        full_spectrum_source_count += int(spectrum_degree == len(unique_targets))
        total_distinct_target_values += spectrum_degree
        multiplicities.append(gcd_degree_sum)
        source_rows.append(
            {
                "source_index": source_index,
                "source_x": int(left[0]),
                "target_multiplicity": gcd_degree_sum,
                "candidate": gcd_degree_sum != 0,
                "combined_target_degree": len(unique_targets),
                "combined_value_minimal_polynomial_degree": spectrum_degree,
                "combined_value_spectrum_full_degree": spectrum_degree
                == len(unique_targets),
                "chart_rows": chart_rows,
            }
        )

    candidate_roots = sorted(
        int(point[0])
        for point, multiplicity in zip(selected, multiplicities)
        if multiplicity
    )
    candidate_factor = R161.monic_root_polynomial(candidate_roots, prime)
    candidate_factor_sha256 = sha256_json(candidate_factor)
    if expected_roots is not None and candidate_roots != expected_roots:
        raise AssertionError("R183 candidate roots differ from R182 replay")
    if (
        expected_factor_sha256 is not None
        and candidate_factor_sha256 != expected_factor_sha256
    ):
        raise AssertionError("R183 candidate factor differs from R182 replay")

    candidate_degree = len(candidate_roots)
    moment_count = 2 * candidate_degree
    moments = [
        sum(
            multiplicity * pow(int(point[0]), exponent, prime)
            for point, multiplicity in zip(selected, multiplicities)
        )
        % prime
        for exponent in range(moment_count)
    ]
    prony_locator, connection = locator_from_weighted_moments(moments, prime)
    if prony_locator != candidate_factor:
        raise AssertionError("weighted Prony locator differs from candidate factor")
    if not recurrence_is_exact(moments, connection, prime):
        raise AssertionError("weighted moment recurrence failed")
    hankel_rank = (
        R81.rank_mod(
            [
                [moments[row + column] for column in range(candidate_degree)]
                for row in range(candidate_degree)
            ],
            prime,
        )
        if candidate_degree
        else 0
    )
    if hankel_rank != candidate_degree:
        raise AssertionError("weighted moment Hankel matrix lost candidate rank")

    return {
        "control_id": f"{curve['family_id']}_sparse_projector_prony_seed{seed}_{role}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "seed": seed,
        "control_role": role,
        "replay_source": replay_source,
        "selected_divisor_degree": len(selected),
        "raw_target_count": len(raw_targets),
        "unique_target_count": len(unique_targets),
        "target_chart_count": len(charts),
        "candidate_degree": candidate_degree,
        "noncandidate_degree": len(selected) - candidate_degree,
        "target_gcd_multiplicity_sum": sum(multiplicities),
        "maximum_target_gcd_multiplicity": max(multiplicities, default=0),
        "all_target_multiplicities_below_characteristic": all(
            multiplicity < prime for multiplicity in multiplicities
        ),
        "all_fermat_projector_traces_equal_gcd_degrees": True,
        "all_gcd_degrees_equal_direct_group_law_multiplicities": True,
        "candidate_support_iff_nonzero_multiplicity": all(
            (multiplicity != 0) == (int(point[0]) in set(candidate_roots))
            for point, multiplicity in zip(selected, multiplicities)
        ),
        "weighted_moment_count": moment_count,
        "weighted_moments_sha256": sha256_json(moments),
        "berlekamp_massey_order": len(connection) - 1,
        "weighted_hankel_rank": hankel_rank,
        "weighted_prony_locator_sha256": sha256_json(prony_locator),
        "candidate_factor_sha256": candidate_factor_sha256,
        "candidate_roots": candidate_roots,
        "prony_locator_equals_candidate_factor": prony_locator
        == candidate_factor,
        "candidate_factor_equals_expected_replay": expected_factor_sha256
        is None
        or candidate_factor_sha256 == expected_factor_sha256,
        "full_target_spectrum_source_count": full_spectrum_source_count,
        "source_count": len(selected),
        "total_distinct_target_value_count": total_distinct_target_values,
        "source_rows": source_rows,
        "finite_pair_or_projector_enumeration_receives_asymptotic_credit": False,
        "candidate_oracle_consumed": False,
    }


def ordinary_control(
    curve: dict[str, Any], seed: int, r182_report: dict[str, Any]
) -> dict[str, Any]:
    _, divisor, target_records = R174.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    raw_targets = [
        tuple(record["target"])
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    expected = r182_control_row(r182_report, curve["family_id"], seed)
    return analyze_instance(
        curve=curve,
        seed=seed,
        role="r182_replay",
        selected=selected,
        divisor=divisor,
        raw_targets=raw_targets,
        expected_roots=expected["candidate_roots"],
        expected_factor_sha256=expected["candidate_factor_sha256"],
        replay_source="r182_frozen_target_subresultant",
    )


def strided_divisor_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    _, original_divisor, target_records = R174.R164.target_material(curve, seed)
    original_selected = [
        tuple(record["endpoint"]) for record in original_divisor["records"]
    ]
    selected = [
        point for index, point in enumerate(original_selected) if index % 3 != 2
    ]
    if len(selected) < 6:
        raise AssertionError("strided divisor is too small")
    prime = int(curve["field_prime"])
    divisor = divisor_from_points(selected, prime)
    selected_set = set(selected)

    pair_sum_set = {
        tuple(total)
        for left in selected
        for right in selected
        if (total := R161.R70.add(left, right, curve)) is not None
    }
    constructed_targets: list[tuple[int, int]] = []
    for index in range(1, min(3, len(selected))):
        total = R161.R70.add(selected[0], selected[index], curve)
        if total is not None and tuple(total) not in selected_set:
            constructed_targets.append(tuple(total))
    for record in target_records:
        target = tuple(record["target"])
        if target not in selected_set and target not in pair_sum_set:
            constructed_targets.append(target)
        if len(R182.unique_points(constructed_targets)) >= 8:
            break
    unique = R182.unique_points(constructed_targets)
    if not unique:
        raise AssertionError("strided divisor produced no targets")
    opposite = tuple(R161.R70.negate(unique[0], curve))
    raw_targets = [unique[0], unique[0], opposite, *unique[1:]]

    return analyze_instance(
        curve=curve,
        seed=seed,
        role="new_strided_source_divisor",
        selected=selected,
        divisor=divisor,
        raw_targets=raw_targets,
        expected_roots=None,
        expected_factor_sha256=None,
        replay_source="independent_group_law_strided_divisor",
    )


def source_blind_trace_collision() -> dict[str, Any]:
    prime = 101
    source_x = [1, 2]
    rows = [[0, 3, 7], [4, 5, 6]]
    swapped = [rows[1], rows[0]]
    exponent_count = 12

    def global_power_sums(values: list[list[int]]) -> list[int]:
        return [
            sum(pow(value, exponent, prime) for row in values for value in row)
            % prime
            for exponent in range(exponent_count)
        ]

    original_multiplicities = [sum(value == 0 for value in row) for row in rows]
    swapped_multiplicities = [
        sum(value == 0 for value in row) for row in swapped
    ]
    original_locator = R161.monic_root_polynomial(
        [
            value
            for value, multiplicity in zip(source_x, original_multiplicities)
            if multiplicity
        ],
        prime,
    )
    swapped_locator = R161.monic_root_polynomial(
        [
            value
            for value, multiplicity in zip(source_x, swapped_multiplicities)
            if multiplicity
        ],
        prime,
    )
    original_sums = global_power_sums(rows)
    swapped_sums = global_power_sums(swapped)
    if original_sums != swapped_sums or original_locator == swapped_locator:
        raise AssertionError("source-blind trace collision control failed")
    return {
        "field_prime": prime,
        "source_x": source_x,
        "target_value_rows": rows,
        "row_swapped_target_values": swapped,
        "power_sum_exponent_count": exponent_count,
        "global_power_sums_sha256": sha256_json(original_sums),
        "row_swapped_power_sums_sha256": sha256_json(swapped_sums),
        "all_source_blind_power_sums_equal": original_sums == swapped_sums,
        "original_candidate_locator": original_locator,
        "row_swapped_candidate_locator": swapped_locator,
        "candidate_locators_differ": original_locator != swapped_locator,
        "scope": "unweighted_global_target_spectrum_only",
    }


def theorem_record() -> dict[str, str]:
    return {
        "nested_fermat_projector_trace": (
            "For squarefree split target charts B_r=F[u]/W_r and a source "
            "component P, Tr_Br(1-M_RP^(p-1)) equals deg gcd(W_r,R_P)."
        ),
        "noncancelling_candidate_weight": (
            "The summed target-gcd multiplicity m_P lies in [0,N]. Since "
            "N<p, m_P is nonzero in F exactly when P is a candidate."
        ),
        "weighted_prony_locator": (
            "For c candidate x-coordinates, s_k=sum_P m_P*x(P)^k has "
            "minimal recurrence polynomial product_candidate (Z-x(P)); 2c "
            "moments recover the monic locator by Berlekamp-Massey."
        ),
        "source_blind_scalar_limitation": (
            "An unweighted scalar Krylov trace sees only the global multiset "
            "of target values. Swapping complete target spectra between source "
            "labels preserves every such power sum while changing the locator."
        ),
        "scope": (
            "The identities reduce the desired output from an A-valued norm or "
            "N A-valued Krylov terms to O(c) scalar nested moments. R183 does "
            "not construct those moments from the compact elliptic SLP below nN."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_sparse_projector_prony_locator.cost.r183.v1",
        "selected_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_degree_exponent_B": fraction_record(Fraction(5, 4)),
        "candidate_degree_exponent_B": fraction_record(Fraction(3, 4)),
        "projector_moment_output_exponent_B": fraction_record(Fraction(3, 4)),
        "prony_locator_recovery_exponent_B": fraction_record(Fraction(3, 4)),
        "candidate_target_verification_exponent_B": fraction_record(Fraction(2, 1)),
        "desired_compact_setup_exponent_B": fraction_record(Fraction(9, 4)),
        "represented_tensor_projector_exponent_B": fraction_record(Fraction(7, 2)),
        "a_linear_target_krylov_output_exponent_B": fraction_record(Fraction(7, 2)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "projector_moment_output_strictly_below_rho": True,
        "prony_locator_recovery_strictly_below_rho": True,
        "candidate_target_verification_strictly_below_rho": True,
        "represented_tensor_projector_strictly_below_rho": False,
        "a_linear_target_krylov_output_strictly_below_rho": False,
        "slp_direct_nested_projector_moment_constructor_supplied": False,
        "source_blind_scalar_trace_is_label_complete": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    verify_source_bindings()
    r182_report = json.loads(R182_REPORT.read_text())
    if r182_report["admission"]["slp_direct_output_sensitive_resultant_admitted"]:
        raise AssertionError("R182 SLP-direct constructor should remain open")
    curves = R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
    rows = [
        ordinary_control(curve, seed, r182_report)
        for curve in curves
        for seed in REPLAY_SEEDS
    ]
    rows.append(strided_divisor_control(curves[0], NEW_DIVISOR_SEED))
    collision = source_blind_trace_collision()

    all_exact = all(
        row["all_fermat_projector_traces_equal_gcd_degrees"]
        and row["all_gcd_degrees_equal_direct_group_law_multiplicities"]
        and row["candidate_support_iff_nonzero_multiplicity"]
        and row["prony_locator_equals_candidate_factor"]
        and row["candidate_factor_equals_expected_replay"]
        and row["weighted_hankel_rank"] == row["candidate_degree"]
        and row["berlekamp_massey_order"] == row["candidate_degree"]
        for row in rows
    )
    controls = {
        "schema": "p1553.m6_sparse_projector_prony_locator.controls.r183.v1",
        "control_count": len(rows),
        "r182_replay_control_count": sum(
            row["control_role"] == "r182_replay" for row in rows
        ),
        "new_divisor_family_control_count": sum(
            row["control_role"] == "new_strided_source_divisor" for row in rows
        ),
        "replay_seeds": list(REPLAY_SEEDS),
        "new_divisor_seed": NEW_DIVISOR_SEED,
        "selected_divisor_degree_sum": sum(
            row["selected_divisor_degree"] for row in rows
        ),
        "unique_target_count_sum": sum(row["unique_target_count"] for row in rows),
        "candidate_degree_sum": sum(row["candidate_degree"] for row in rows),
        "noncandidate_degree_sum": sum(
            row["noncandidate_degree"] for row in rows
        ),
        "target_gcd_multiplicity_sum": sum(
            row["target_gcd_multiplicity_sum"] for row in rows
        ),
        "weighted_moment_count_sum": sum(
            row["weighted_moment_count"] for row in rows
        ),
        "full_target_spectrum_source_count": sum(
            row["full_target_spectrum_source_count"] for row in rows
        ),
        "represented_source_target_value_slot_count": sum(
            row["source_count"] * row["unique_target_count"] for row in rows
        ),
        "total_distinct_target_value_count": sum(
            row["total_distinct_target_value_count"] for row in rows
        ),
        "source_count": sum(row["source_count"] for row in rows),
        "all_target_multiplicities_below_characteristic": all(
            row["all_target_multiplicities_below_characteristic"] for row in rows
        ),
        "all_fermat_projector_traces_equal_gcd_degrees": all(
            row["all_fermat_projector_traces_equal_gcd_degrees"] for row in rows
        ),
        "all_gcd_degrees_equal_direct_group_law_multiplicities": all(
            row["all_gcd_degrees_equal_direct_group_law_multiplicities"]
            for row in rows
        ),
        "all_candidate_supports_equal_nonzero_multiplicity_support": all(
            row["candidate_support_iff_nonzero_multiplicity"] for row in rows
        ),
        "all_weighted_hankel_ranks_equal_candidate_degrees": all(
            row["weighted_hankel_rank"] == row["candidate_degree"] for row in rows
        ),
        "all_berlekamp_massey_orders_equal_candidate_degrees": all(
            row["berlekamp_massey_order"] == row["candidate_degree"]
            for row in rows
        ),
        "all_prony_locators_equal_candidate_factors": all(
            row["prony_locator_equals_candidate_factor"] for row in rows
        ),
        "all_r182_candidate_factors_replayed": all(
            row["candidate_factor_equals_expected_replay"]
            for row in rows
            if row["control_role"] == "r182_replay"
        ),
        "source_blind_trace_collision": collision,
        "candidate_oracle_consumed": False,
        "finite_pair_or_projector_enumeration_receives_asymptotic_credit": False,
        "controls": rows,
    }
    cost = cost_record()
    obligations = {
        "r182_gap_inherited": True,
        "source_bindings_complete": True,
        "twelve_r182_controls_replayed": controls["r182_replay_control_count"] == 12,
        "new_strided_divisor_family_complete": controls[
            "new_divisor_family_control_count"
        ]
        == 1,
        "nested_projector_identity_complete": controls[
            "all_fermat_projector_traces_equal_gcd_degrees"
        ],
        "group_law_multiplicity_replay_complete": controls[
            "all_gcd_degrees_equal_direct_group_law_multiplicities"
        ],
        "multiplicity_non_cancellation_complete": controls[
            "all_target_multiplicities_below_characteristic"
        ],
        "candidate_support_complete": controls[
            "all_candidate_supports_equal_nonzero_multiplicity_support"
        ],
        "weighted_hankel_rank_complete": controls[
            "all_weighted_hankel_ranks_equal_candidate_degrees"
        ],
        "berlekamp_massey_order_complete": controls[
            "all_berlekamp_massey_orders_equal_candidate_degrees"
        ],
        "prony_locator_replay_complete": controls[
            "all_prony_locators_equal_candidate_factors"
        ],
        "r182_factor_hash_replay_complete": controls[
            "all_r182_candidate_factors_replayed"
        ],
        "source_blind_collision_complete": collision[
            "all_source_blind_power_sums_equal"
        ]
        and collision["candidate_locators_differ"],
        "output_count_charged": True,
        "represented_tensor_route_charged": True,
        "a_linear_krylov_route_charged": True,
        "candidate_verification_charged": True,
        "finite_enumeration_no_credit": True,
        "candidate_oracle_absent": True,
        "slp_direct_nested_moment_constructor": False,
        "nN_body_avoided_by_supplied_constructor": False,
        "deterministic_hash_to_curve_transfer": False,
        "generic_prime_coordinate_family_algorithm": False,
        "factor_logs": False,
        "identical_target_descent": False,
        "pollard_rho_improvement": False,
        "shoup_improvement": False,
        "breakthrough": False,
    }
    passed_obligation_count = sum(obligations.values())
    disposition = (
        "ADMIT_NESTED_FERMAT_PROJECTOR_TRACE_EQUALS_TARGET_GCD_DEGREE__"
        "NONZERO_MULTIPLICITY_SUPPORT_EXACT_BECAUSE_N_LT_P__TWELVE_R182_"
        "REPLAYS_AND_ONE_NEW_STRIDED_DIVISOR_FAMILY__WEIGHTED_2C_SOURCE_"
        "MOMENTS_RECOVER_G1_BY_PRONY_BERLEKAMP_MASSEY__SOURCE_BLIND_GLOBAL_"
        "KRYLOV_TRACE_LABEL_COLLISION_EXACT__OUTPUT_AND_RECOVERY_B3O4__"
        "CANDIDATE_VERIFICATION_B2__REPRESENTED_PROJECTOR_AND_A_LINEAR_"
        "KRYLOV_B7O2__SLP_DIRECT_NESTED_MOMENT_CONSTRUCTOR_OPEN__NO_RHO_"
        "SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Construct or refute one SLP-direct nested projector-moment operator. "
        "From compact U,V, target charts, and the R181 signed line-product SLP, "
        "emit the 2c scalars Tr_A(X^k Tr_B(1-M_R^(p-1))) in softly O(n+N+c) "
        "work without representing R, the Fermat projector, an A-valued target "
        "Krylov sequence, or any nN tensor body. Test seed 18308 and a second "
        "strided divisor family; reject source-blind traces and candidate oracles."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-02",
        "role": "coordinator",
        "objective": (
            "Replace the R182 A-valued resultant output by an output-sensitive "
            "scalar moment interface and test exact candidate-locator recovery."
        ),
        "source_bindings": source_binding_records(),
        "theorems": theorem_record(),
        "controls": controls,
        "cost": cost,
        "admission": {
            "passed_obligation_count": passed_obligation_count,
            "obligation_count": len(obligations),
            "obligations": obligations,
            "nested_projector_identity_admitted": all_exact,
            "weighted_prony_output_stage_admitted": all_exact,
            "slp_direct_nested_moment_constructor_admitted": False,
            "lane_admitted": False,
        },
        "analysis": {
            "new_interface": "2c_nested_scalar_projector_moments",
            "source_label_strategy": "weighted_source_power_moments_plus_prony",
            "source_blind_scalar_krylov": "label_incomplete",
            "standard_represented_projector": "B^(7/2)",
            "a_linear_target_krylov": "B^(7/2)",
            "moment_output_and_prony": "B^(3/4)",
            "candidate_verification": "B^2",
            "missing_constructor": "slp_direct_nested_projector_moments",
        },
        "classification": disposition,
        "disposition": disposition,
        "next_action": next_action,
        "finite_controls_receive_asymptotic_credit": False,
        "candidate_discrete_log_oracle_consumed": False,
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
    }
    frozen = {
        "schema": "p1553.m6_sparse_projector_prony_locator.frozen.r183.v1",
        "source_report_sha256": sha256_json(report),
        "identity": {
            "projector": "m_P=sum_r_Tr_Br(1-M_RP^(p-1))",
            "support": "m_P_nonzero_iff_candidate_when_total_target_degree_lt_p",
            "moments": "s_k=sum_P_m_P_xP^k_for_0_le_k_lt_2c",
            "locator": "BM(s)_reversed=product_candidate(X-xP)",
        },
        "closed_routes": [
            "source_blind_global_target_power_sums",
            "represented_tensor_fermat_projector",
            "A_linear_N_term_target_Krylov_sequence",
        ],
        "open_route": "SLP_direct_2c_nested_projector_moments",
        "disposition": disposition,
        "breakthrough": False,
    }
    replay = {
        "schema": "p1553.m6_sparse_projector_prony_locator.replay.r183.v1",
        "control_count": controls["control_count"],
        "rows": [
            {
                "control_id": row["control_id"],
                "candidate_degree": row["candidate_degree"],
                "target_gcd_multiplicity_sum": row[
                    "target_gcd_multiplicity_sum"
                ],
                "weighted_moment_count": row["weighted_moment_count"],
                "weighted_moments_sha256": row["weighted_moments_sha256"],
                "locator_sha256": row["weighted_prony_locator_sha256"],
            }
            for row in rows
        ],
        "source_blind_collision_sha256": sha256_json(collision),
        "disposition": disposition,
    }
    applicability = {
        "schema": "p1553.m6_sparse_projector_prony_locator.applicability.r183.v1",
        "squarefree_split_source_divisor": "required",
        "squarefree_split_target_charts": "required",
        "total_target_degree_below_characteristic": "required_for_non_cancellation",
        "distinct_source_x_coordinates": "required_for_prony_locator",
        "known_candidate_degree": "not_required_BM_recovers_order",
        "compact_slp_direct_moment_constructor": "not_supplied",
        "generic_prime_family_transfer": "not_supplied",
        "breakthrough": False,
    }
    controls_artifact = dict(controls)
    cost_artifact = dict(cost)
    return {
        "report": report,
        "frozen": frozen,
        "cost": cost_artifact,
        "replay": replay,
        "controls": controls_artifact,
        "applicability": applicability,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--applicability", type=Path, default=DEFAULT_APPLICABILITY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report, bundle["report"])
    write_json(args.frozen, bundle["frozen"])
    write_json(args.cost, bundle["cost"])
    write_json(args.replay, bundle["replay"])
    write_json(args.controls, bundle["controls"])
    write_json(args.applicability, bundle["applicability"])
    print(json.dumps({key: str(value) for key, value in vars(args).items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
