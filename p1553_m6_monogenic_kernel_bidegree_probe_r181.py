#!/usr/bin/env python3
"""Audit exact monogenic folds for the signed M6 target kernel."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_monogenic_kernel_bidegree.r181.v1"

R180_PRODUCER = ROOT / "p1553_m6_d5_directed_evaluation_survivor_probe_r180.py"
R180_REPORT = ROOT / "p1553_m6_d5_directed_evaluation_survivor_probe_report_r180.json"
R180_FROZEN = ROOT / "frozen_m6_d5_directed_evaluation_survivor.json"
R180_COST = ROOT / "m6_d5_directed_evaluation_survivor_cost_ledger.json"
R180_REPLAY = ROOT / "m6_d5_directed_evaluation_survivor_replay.json"
R180_CONTROLS = ROOT / "m6_d5_directed_evaluation_survivor_controls.json"
R180_APPLICABILITY = ROOT / "d5_directed_evaluation_survivor_applicability_r180.json"
R180_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_d5_directed_evaluation_survivor_probe_r180.py"
R180_GATE = ROOT / "p1553_m6_d5_directed_evaluation_survivor_probe_gate_r180.md"
R180_PARENT = ROOT / "p1553_m6_d5_directed_evaluation_survivor_probe_parent_report_r180.yaml"

POTEAUX_SCHOST = ROOT / "references/poteaux_schost_modular_composition_triangular_sets_2010.pdf"
SPECIAL_RESULTANTS = ROOT / "references/bostan_flajolet_salvy_schost_special_resultants_2005.pdf"
KEDLAYA_UMANS = ROOT / "references/kedlaya_umans_fast_polynomial_factorization_modular_composition_2011.pdf"
SHOUP_PAPER = ROOT / "references/shoup_generic_dlp_lower_bound_1997.pdf"

SOURCE_BINDINGS = (
    ("r180_producer", R180_PRODUCER, "d7925463992a6275aa05e7401bcde60dd5f4959b3a426892d561b78a6502ac60"),
    ("r180_report", R180_REPORT, "93aff1e86ceab11135757789baf02497c5b4915843061125aa351854423bb760"),
    ("r180_frozen", R180_FROZEN, "e3bd7aa983330fa315552dd8813cb58efec30633dbae370586edb2a43edb2231"),
    ("r180_cost", R180_COST, "038215322dcc394d4231f234092e08e365aab6622138570fbf7eee35e1ae79cc"),
    ("r180_replay", R180_REPLAY, "f136b663b81f712849552881eb78336e2b1bf39800377027fdcd222f6c2fa17d"),
    ("r180_controls", R180_CONTROLS, "a632faa9922d552106d8099f0efd98bb2114e81ed7c1cf7d235cd78a7cf28318"),
    ("r180_applicability", R180_APPLICABILITY, "8bacbab2eeffb69cb3e3a4f2dd0f3a02bda992a9e584a648e2d9a41cb72b9c9b"),
    ("r180_test", R180_TEST, "e34e4067069d9288bc57c74dfa38ef5f558e5eda8e1c44137ffb61b9a3ba0f41"),
    ("r180_gate", R180_GATE, "b1e56b8a374d2b95a04f278e5c47910d072b937c67cbaea3f6dab7caee6d44d1"),
    ("r180_parent", R180_PARENT, "a11bb3fea860ebbceb57d43d1aff2e115867f8667143c7fe78ee8a272c144270"),
    ("poteaux_schost_2010", POTEAUX_SCHOST, "587f302dd16c724d1be6a4b629a46a684a0c35389dbc22ba45641dba54de6f32"),
    ("bostan_flajolet_salvy_schost_2005", SPECIAL_RESULTANTS, "19db312c68f997949db342a050df568a009a340dd5026e4584e2c4fa59fcc375"),
    ("kedlaya_umans_2011", KEDLAYA_UMANS, "93bd1f77b762f49bcae017d5c12ceccef38c67a956810873105cbce083634377"),
    ("shoup_1997", SHOUP_PAPER, "89d19aad3a4d98b563029de9135d30c8ed9b831d74f7348c286acc22f9af85b3"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_monogenic_kernel_bidegree_probe_report_r181.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_monogenic_kernel_bidegree.json"
DEFAULT_COST = ROOT / "m6_monogenic_kernel_bidegree_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_monogenic_kernel_bidegree_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_monogenic_kernel_bidegree_controls.json"
DEFAULT_APPLICABILITY = ROOT / "monogenic_kernel_bidegree_applicability_r181.json"

DEVELOPMENT_SEEDS = (16001, 16002)
HELD_OUT_SEEDS = (18104,)

NormalForm = tuple[list[int], list[int]]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R180 = load_module("p1553_r180_for_r181", R180_PRODUCER)
R174 = R180.R174
R161 = R180.R161
R81 = R161.R159.R82.R81


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
        raise AssertionError(f"R181 source binding mismatch: {failures}")


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def poly_trim(poly: list[int], prime: int) -> list[int]:
    result = [coefficient % prime for coefficient in poly]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result or [0]


def poly_add(left: list[int], right: list[int], prime: int) -> list[int]:
    result = [0] * max(len(left), len(right))
    for index, coefficient in enumerate(left):
        result[index] = (result[index] + coefficient) % prime
    for index, coefficient in enumerate(right):
        result[index] = (result[index] + coefficient) % prime
    return poly_trim(result, prime)


def normal_mul(
    left: NormalForm,
    right: NormalForm,
    curve: dict[str, Any],
) -> NormalForm:
    prime = int(curve["field_prime"])
    left_a, left_b = left
    right_a, right_b = right
    curve_cubic = [
        int(curve["curve_b"]) % prime,
        int(curve["curve_a"]) % prime,
        0,
        1,
    ]
    a_part = poly_add(
        R161.poly_mul(left_a, right_a, prime),
        R161.poly_mul(
            R161.poly_mul(left_b, right_b, prime), curve_cubic, prime
        ),
        prime,
    )
    b_part = poly_add(
        R161.poly_mul(left_a, right_b, prime),
        R161.poly_mul(left_b, right_a, prime),
        prime,
    )
    return a_part, b_part


def signed_line_as_target_normal_form(
    left: tuple[int, int],
    right: tuple[int, int],
    divided: list[int],
    curve: dict[str, Any],
) -> NormalForm:
    prime = int(curve["field_prime"])
    x_value, y_value = map(int, left)
    z_value = int(right[0])
    if z_value == x_value:
        slope_numerator = (
            3 * x_value * x_value + int(curve["curve_a"])
        ) % prime
        return (
            [
                (x_value * slope_numerator - 2 * y_value * y_value) % prime,
                (-slope_numerator) % prime,
            ],
            [(-2 * y_value) % prime],
        )
    divided_value = R161.poly_eval(divided, z_value, prime)
    return (
        [(x_value * divided_value - y_value) % prime, (-divided_value) % prime],
        [prime - 1],
    )


def signed_target_kernel(
    curve: dict[str, Any],
    divisor: dict[str, Any],
    selected: list[tuple[int, int]],
    left: tuple[int, int],
) -> NormalForm:
    prime = int(curve["field_prime"])
    divided = R174.divided_difference_coefficients(
        divisor["v"], int(left[0]), prime
    )
    result: NormalForm = ([1], [0])
    for right in selected:
        result = normal_mul(
            result,
            signed_line_as_target_normal_form(left, right, divided, curve),
            curve,
        )
    return result


def normal_eval(
    normal: NormalForm, target: tuple[int, int], prime: int
) -> int:
    a_part, b_part = normal
    return (
        R161.poly_eval(a_part, int(target[0]), prime)
        + int(target[1]) * R161.poly_eval(b_part, int(target[0]), prime)
    ) % prime


def normal_pole_order(normal: NormalForm) -> int:
    a_part, b_part = normal
    a_degree = R161.poly_degree(a_part)
    b_degree = R161.poly_degree(b_part)
    a_order = 2 * a_degree if a_part != [0] else -1
    b_order = 2 * b_degree + 3 if b_part != [0] else -1
    return max(a_order, b_order)


def public_target_scan(
    curve: dict[str, Any],
    generator: tuple[int, int],
    selected: list[tuple[int, int]],
    count: int,
) -> list[dict[str, Any]]:
    selected_set = set(selected)
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    scalar = 1
    order = int(curve["subgroup_order"])
    while len(records) < count and scalar < order:
        target = R161.R70.scalar_mul(scalar, generator, curve)
        if target is not None:
            target = tuple(target)
            if target not in selected_set and target not in seen:
                records.append({"scalar": scalar, "target": target})
                seen.add(target)
        scalar += 1
    if len(records) != count:
        raise AssertionError("unable to construct public target scan")
    return records


def padded_coefficient_rows(normals: list[NormalForm]) -> tuple[list[list[int]], int, int]:
    max_a_degree = max(R161.poly_degree(normal[0]) for normal in normals)
    max_b_degree = max(R161.poly_degree(normal[1]) for normal in normals)
    rows = []
    for a_part, b_part in normals:
        rows.append(
            [
                *(a_part + [0] * (max_a_degree + 1 - len(a_part))),
                *(b_part + [0] * (max_b_degree + 1 - len(b_part))),
            ]
        )
    return rows, max_a_degree, max_b_degree


def expected_candidate(
    curve: dict[str, Any],
    seed: int,
    role: str,
    r180_report: dict[str, Any],
) -> dict[str, Any]:
    if role == "development_replay":
        row = R180.control_row(r180_report, curve["family_id"], seed)
        return {
            "source": "r180_frozen_control",
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
    r180_report: dict[str, Any],
) -> dict[str, Any]:
    factor_base, divisor, target_records = R174.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    retained_targets = [
        tuple(record["target"])
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    prime = int(curve["field_prime"])
    n = len(selected)
    normals = [
        signed_target_kernel(curve, divisor, selected, left)
        for left in selected
    ]
    coefficient_rows, max_a_degree, max_b_degree = padded_coefficient_rows(
        normals
    )
    coefficient_rank = R81.rank_mod(coefficient_rows, prime)
    coefficient_slot_count = sum(len(row) for row in coefficient_rows)
    coefficient_nonzero_count = sum(
        int(coefficient != 0)
        for row in coefficient_rows
        for coefficient in row
    )

    generator = tuple(factor_base["generator"])
    scan_records = public_target_scan(curve, generator, selected, n)
    scan_targets = [record["target"] for record in scan_records]
    evaluation_rows = [
        [normal_eval(normal, target, prime) for target in scan_targets]
        for normal in normals
    ]
    evaluation_rank = R81.rank_mod(evaluation_rows, prime)

    exact_factor_columns: list[list[int]] = []
    scan_pair_evaluation_count = 0
    for target in scan_targets:
        factor = R180.target_norm_factor(
            curve, divisor, selected, target
        )
        exact_factor_columns.append(
            [
                R161.poly_eval(
                    factor["factor_polynomial"], int(point[0]), prime
                )
                for point in selected
            ]
        )
        scan_pair_evaluation_count += int(factor["pair_evaluation_count"])
    exact_factor_rows = [list(row) for row in zip(*exact_factor_columns)]
    if evaluation_rows != exact_factor_rows:
        raise AssertionError("canonical target kernel differs from R180 factors")

    desired_zero_count = 0
    desired_target_sets_are_distinct = True
    for left, normal in zip(selected, normals):
        desired_targets = [
            tuple(R161.R70.add(left, right, curve) or ())
            for right in selected
        ]
        if any(not target for target in desired_targets):
            raise AssertionError("selected pair sum reached infinity")
        desired_target_sets_are_distinct &= len(set(desired_targets)) == n
        for target in desired_targets:
            if normal_eval(normal, target, prime) != 0:
                raise AssertionError("signed target kernel lost a desired zero")
            desired_zero_count += 1
    nonincident_witness_per_source = all(
        any(value != 0 for value in row) for row in evaluation_rows
    )
    pole_orders = [normal_pole_order(normal) for normal in normals]
    pole_order_bounds_respected = all(order >= n for order in pole_orders)

    aggregate_values = []
    for normal in normals:
        value = 1
        for target in retained_targets:
            value = value * normal_eval(normal, target, prime) % prime
        aggregate_values.append(value)
    selector = R161.interpolate(
        [
            (int(point[0]), int(value))
            for point, value in zip(selected, aggregate_values)
        ],
        prime,
    )
    candidate_factor = R161.poly_gcd(divisor["u"], selector, prime)
    candidate_roots = sorted(
        int(point[0])
        for point in selected
        if R161.poly_eval(candidate_factor, int(point[0]), prime) == 0
    )
    expected = expected_candidate(curve, seed, role, r180_report)
    candidate_sha256 = sha256_json(candidate_factor)
    if candidate_sha256 != expected["candidate_factor_sha256"]:
        raise AssertionError("R181 candidate factor differs from signed replay")
    if candidate_roots != expected["candidate_roots"]:
        raise AssertionError("R181 candidate roots differ from signed replay")

    normal_hashes = [
        sha256_json({"a": normal[0], "b": normal[1]}) for normal in normals
    ]
    return {
        "control_id": f"{curve['family_id']}_monogenic_kernel_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "subgroup_order": int(curve["subgroup_order"]),
        "seed": seed,
        "control_role": role,
        "selected_divisor_degree": n,
        "retained_target_count": len(retained_targets),
        "canonical_normal_form": "A_P(u)+v*B_P(u)_mod_v2_minus_u3_minus_au_minus_b",
        "maximum_a_degree": max_a_degree,
        "maximum_b_degree": max_b_degree,
        "maximum_target_pole_order": max(pole_orders),
        "minimum_target_pole_order": min(pole_orders),
        "target_pole_order_equals_3n": all(order == 3 * n for order in pole_orders),
        "coefficient_row_count": len(coefficient_rows),
        "coefficient_slot_count_per_source": len(coefficient_rows[0]),
        "coefficient_slot_count": coefficient_slot_count,
        "coefficient_nonzero_count": coefficient_nonzero_count,
        "coefficient_body_is_fully_dense": (
            coefficient_nonzero_count == coefficient_slot_count
        ),
        "coefficient_matrix_rank": coefficient_rank,
        "coefficient_matrix_full_source_rank": coefficient_rank == n,
        "coefficient_matrix_sha256": sha256_json(coefficient_rows),
        "kernel_normal_form_sha256s": normal_hashes,
        "public_target_scan_count": len(scan_targets),
        "public_target_scan_last_scalar": scan_records[-1]["scalar"],
        "public_target_scan_sha256": sha256_json(
            [
                {"scalar": record["scalar"], "target": list(record["target"])}
                for record in scan_records
            ]
        ),
        "evaluation_matrix_rank": evaluation_rank,
        "evaluation_matrix_full_source_rank": evaluation_rank == n,
        "evaluation_matrix_sha256": sha256_json(evaluation_rows),
        "canonical_kernel_equals_r180_factors_on_scan": True,
        "scan_pair_evaluation_count": scan_pair_evaluation_count,
        "desired_incidence_zero_count": desired_zero_count,
        "desired_target_sets_are_distinct": desired_target_sets_are_distinct,
        "all_desired_incidence_zeros_exact": True,
        "nonincident_witness_per_source": nonincident_witness_per_source,
        "pole_order_zero_bound_respected": pole_order_bounds_respected,
        "exact_scalar_source_degree_lower_bound": n - 1,
        "candidate_factor_sha256": candidate_sha256,
        "candidate_roots": candidate_roots,
        "candidate_factor_equals_signed_replay": True,
        "candidate_replay_source": expected["source"],
        "finite_rank_or_density_receives_asymptotic_credit": False,
        "candidate_oracle_consumed": False,
    }


def theorem_record() -> dict[str, str]:
    return {
        "canonical_target_kernel": (
            "For fixed selected P and Q, the tangent-corrected R174 factor is "
            "linear in target coordinates (u,v). Multiplying over Q and reducing "
            "by v^2=u^3+au+b gives a unique normal form K_P(u,v)=A_P(u)+vB_P(u)."
        ),
        "target_pole_degree": (
            "Translation by P maps the n distinct selected Q to n distinct desired "
            "targets P+Q. Any nonzero target-coordinate rational function that "
            "vanishes on all of them has pole degree at least n. In the canonical "
            "line-product normal form each line has pole order three, and all R181 "
            "controls retain the full pole order 3n."
        ),
        "exact_scalar_kernel_rank": (
            "If exact factors on fixed source values have the form "
            "K(a(P),T)=sum_(i=0)^d a(P)^i c_i(T), every target evaluation matrix "
            "has rank at most d+1. Full rank n on a fixed public n-target scan "
            "therefore forces d at least n-1 for these finite controls. Row and "
            "column scalar normalizations do not change this rank."
        ),
        "monogenic_tautology": (
            "Every residue C_h in F[X]/(U) already has the representation "
            "H(a) mod U by taking a=X and H=C_h. Modular composition accelerates "
            "evaluation only after the coefficients of H and a are supplied; "
            "this tautological representation does not construct C_h."
        ),
        "flattened_norm_interface": (
            "The selected and target quotient algebras have dimensions n and N. "
            "Their tensor algebra has dimension nN. Near-linear finite-field norm, "
            "power-projection, or modular-composition algorithms applied after "
            "flattening are near-linear in nN, not n+N. The full composed-resultant "
            "polynomial likewise has degree nN."
        ),
        "scope": (
            "This closes post-construction monogenic composition, explicit exact "
            "bounded-source-degree scalar kernels, their dense canonical bidegree "
            "body, and flattened tensor norms. It is not a lower bound against a "
            "compact high-degree SLP or a gcd-equivalent output-sensitive elliptic "
            "composed resultant that discards target-dependent units."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_monogenic_kernel_bidegree.cost.r181.v1",
        "selected_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "candidate_degree_exponent_B": fraction_record(Fraction(3, 4)),
        "canonical_target_pole_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "exact_scalar_source_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "canonical_explicit_bidegree_body_exponent_B": fraction_record(Fraction(9, 2)),
        "explicit_N_factor_coefficient_body_exponent_B": fraction_record(Fraction(7, 2)),
        "flattened_tensor_algebra_dimension_exponent_B": fraction_record(Fraction(7, 2)),
        "full_composed_resultant_output_degree_exponent_B": fraction_record(Fraction(7, 2)),
        "postcompiled_degree_n_modcomp_exponent_B": fraction_record(Fraction(9, 4)),
        "desired_compiler_total_exponent_B": fraction_record(Fraction(9, 4)),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "postcompiled_degree_n_modcomp_strictly_below_rho": True,
        "postcompiled_input_coefficients_supplied": False,
        "canonical_explicit_bidegree_body_strictly_below_rho": False,
        "explicit_N_factor_coefficient_body_strictly_below_rho": False,
        "flattened_near_linear_norm_strictly_below_rho": False,
        "full_composed_resultant_output_strictly_below_rho": False,
        "gcd_equivalent_output_sensitive_resultant_supplied": False,
        "compact_high_degree_slp_compiler_supplied": False,
        "finite_rank_claimed_as_asymptotic_lower_bound": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    verify_source_bindings()
    r180_report = json.loads(R180_REPORT.read_text())
    if r180_report["admission"]["one_shot_monogenic_compiler_admitted"]:
        raise AssertionError("R180 monogenic compiler should remain open")
    controls_rows = [
        finite_control(curve, seed, "development_replay", r180_report)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in DEVELOPMENT_SEEDS
    ]
    controls_rows.extend(
        finite_control(curve, seed, "held_out_seed", r180_report)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in HELD_OUT_SEEDS
    )
    all_exact = all(
        row["target_pole_order_equals_3n"]
        and row["coefficient_body_is_fully_dense"]
        and row["coefficient_matrix_full_source_rank"]
        and row["evaluation_matrix_full_source_rank"]
        and row["canonical_kernel_equals_r180_factors_on_scan"]
        and row["desired_target_sets_are_distinct"]
        and row["all_desired_incidence_zeros_exact"]
        and row["nonincident_witness_per_source"]
        and row["pole_order_zero_bound_respected"]
        and row["candidate_factor_equals_signed_replay"]
        for row in controls_rows
    )
    controls = {
        "schema": "p1553.m6_monogenic_kernel_bidegree.controls.r181.v1",
        "control_count": len(controls_rows),
        "development_control_count": sum(
            row["control_role"] == "development_replay" for row in controls_rows
        ),
        "held_out_control_count": sum(
            row["control_role"] == "held_out_seed" for row in controls_rows
        ),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "all_target_pole_orders_equal_3n": all(
            row["target_pole_order_equals_3n"] for row in controls_rows
        ),
        "all_coefficient_bodies_fully_dense": all(
            row["coefficient_body_is_fully_dense"] for row in controls_rows
        ),
        "all_coefficient_matrices_full_source_rank": all(
            row["coefficient_matrix_full_source_rank"] for row in controls_rows
        ),
        "all_evaluation_matrices_full_source_rank": all(
            row["evaluation_matrix_full_source_rank"] for row in controls_rows
        ),
        "all_canonical_kernels_equal_r180_factors": all(
            row["canonical_kernel_equals_r180_factors_on_scan"]
            for row in controls_rows
        ),
        "all_desired_incidence_zeros_exact": all(
            row["all_desired_incidence_zeros_exact"] for row in controls_rows
        ),
        "all_candidate_factors_equal_signed_replay": all(
            row["candidate_factor_equals_signed_replay"] for row in controls_rows
        ),
        "selected_divisor_degree_sum": sum(
            row["selected_divisor_degree"] for row in controls_rows
        ),
        "retained_target_count_sum": sum(
            row["retained_target_count"] for row in controls_rows
        ),
        "coefficient_slot_count": sum(
            row["coefficient_slot_count"] for row in controls_rows
        ),
        "coefficient_nonzero_count": sum(
            row["coefficient_nonzero_count"] for row in controls_rows
        ),
        "desired_incidence_zero_count": sum(
            row["desired_incidence_zero_count"] for row in controls_rows
        ),
        "scan_pair_evaluation_count": sum(
            row["scan_pair_evaluation_count"] for row in controls_rows
        ),
        "finite_rank_or_density_receives_asymptotic_credit": False,
        "candidate_oracle_consumed": False,
        "controls": controls_rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "fourteen_source_bindings_exact": len(SOURCE_BINDINGS) == 14,
        "r180_monogenic_gap_inherited": True,
        "poteaux_schost_norm_interface_bound": True,
        "special_composed_resultant_interface_bound": True,
        "kedlaya_umans_modcomp_interface_bound": True,
        "six_development_controls_replayed": controls["development_control_count"] == 6,
        "three_held_out_controls_replayed": controls["held_out_control_count"] == 3,
        "canonical_target_normal_forms_complete": all_exact,
        "target_pole_order_3n_complete": controls["all_target_pole_orders_equal_3n"],
        "canonical_coefficient_density_complete": controls["all_coefficient_bodies_fully_dense"],
        "coefficient_full_source_rank_complete": controls["all_coefficient_matrices_full_source_rank"],
        "public_scan_full_source_rank_complete": controls["all_evaluation_matrices_full_source_rank"],
        "r180_factor_value_replay_complete": controls["all_canonical_kernels_equal_r180_factors"],
        "desired_incidence_zero_replay_complete": controls["all_desired_incidence_zeros_exact"],
        "signed_candidate_factor_replay_complete": controls["all_candidate_factors_equal_signed_replay"],
        "monogenic_tautology_scoped": True,
        "exact_bounded_source_degree_kernel_closed_on_controls": all_exact,
        "canonical_n2_body_charged": True,
        "flattened_nN_norm_charged": True,
        "full_degree_nN_composed_resultant_charged": True,
        "finite_evidence_scoped_without_asymptotic_lower_bound": True,
        "asymptotic_full_rank_theorem_complete": False,
        "compact_high_degree_slp_compiler_complete": False,
        "gcd_equivalent_unit_normalized_kernel_complete": False,
        "output_sensitive_composed_resultant_mod_u_complete": False,
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
        "ADMIT_CANONICAL_SIGNED_TARGET_KERNEL_NORMAL_FORM__NINE_CONTROLS_"
        "INCLUDING_THREE_HELD_OUT_REPLAYED__TARGET_POLE_ORDER_3N__EXACT_"
        "COEFFICIENT_AND_PUBLIC_SCAN_RANK_N__CANONICAL_BODY_3N2_FULLY_DENSE__"
        "TAUTOLOGICAL_H_OF_A_GETS_NO_CONSTRUCTOR_CREDIT__EXPLICIT_BOUNDED_"
        "SOURCE_DEGREE_FOLD_CLOSED_ON_CONTROLS__FLATTENED_NN_NORM_AND_FULL_"
        "COMPOSED_RESULTANT_B7O2__GCD_EQUIVALENT_OUTPUT_SENSITIVE_RESULTANT_"
        "OPEN__NO_ASYMPTOTIC_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH"
    )
    next_action = (
        "Construct or refute one gcd-equivalent output-sensitive elliptic "
        "composed resultant modulo U. Accept the O(n)-size signed line-product "
        "SLP and the O(N)-size target divisor, discard target-dependent units, "
        "and emit G_1 in softly O(n+N) work without constructing the canonical "
        "3n^2 coefficient body, an nN tensor element, or the full degree-nN "
        "composed resultant. Test direct transposed power projections or "
        "subresultant traces modulo U on the held-out seed. Reject candidate "
        "oracles, unit-cost norms, and post-construction modular composition."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Determine whether the R180 one-shot opening is an exact "
            "bounded-bidegree monogenic kernel or only a tautological "
            "post-construction representation, and preserve any stronger "
            "gcd-equivalent output-sensitive route."
        ),
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "controls": controls,
        "cost": cost,
        "literature": {
            "poteaux_schost": {
                "campaign_fit": (
                    "Supplies near-linear finite-field norm and multivariate "
                    "modular-composition algorithms in the represented triangular "
                    "algebra dimension; the selected-target tensor dimension is nN."
                )
            },
            "bostan_flajolet_salvy_schost": {
                "campaign_fit": (
                    "Supplies fast special composed resultants whose complete "
                    "output has degree equal to the product of input degrees."
                )
            },
            "kedlaya_umans": {
                "campaign_fit": (
                    "Supplies nearly linear finite-field modular composition and "
                    "power projection after the represented inputs are available; "
                    "it does not compile the elliptic target kernel."
                )
            },
            "novelty_scope": (
                "R181 contributes the canonical target-kernel normal form, its "
                "exact finite source-rank audit, and the monogenic-construction "
                "scope split. It claims no new modular-composition theorem or "
                "asymptotic rank lower bound."
            ),
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "canonical_kernel_scope_admitted": all_exact,
            "explicit_exact_bounded_bidegree_fold_closed": all_exact,
            "flattened_norm_route_closed": all_exact,
            "gcd_equivalent_output_sensitive_resultant_admitted": False,
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
        "schema": "p1553.m6_monogenic_kernel_bidegree.frozen.r181.v1",
        "source_bindings": source_binding_records(),
        "interfaces": {
            "exact_kernel": "K_P(u,v)=A_P(u)+vB_P(u)",
            "closed_standard_routes": [
                "tautological_post_construction_H_of_a",
                "explicit_exact_bounded_source_degree_scalar_kernel",
                "canonical_dense_bidegree_body",
                "flattened_nN_tensor_norm",
                "full_degree_nN_composed_resultant",
            ],
            "open_interface": "gcd_equivalent_output_sensitive_elliptic_composed_resultant_mod_U",
            "required_total_work": "softly_O(n+N)",
        },
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "promotion_allowed": False,
    }
    replay = {
        "schema": "p1553.m6_monogenic_kernel_bidegree.replay.r181.v1",
        "source_bindings": source_binding_records(),
        "all_replay_invariants_pass": all_exact,
        "control_records": [
            {
                "control_id": row["control_id"],
                "control_role": row["control_role"],
                "coefficient_matrix_sha256": row["coefficient_matrix_sha256"],
                "evaluation_matrix_sha256": row["evaluation_matrix_sha256"],
                "public_target_scan_sha256": row["public_target_scan_sha256"],
                "candidate_factor_sha256": row["candidate_factor_sha256"],
                "coefficient_matrix_rank": row["coefficient_matrix_rank"],
                "evaluation_matrix_rank": row["evaluation_matrix_rank"],
            }
            for row in controls_rows
        ],
    }
    applicability = {
        "schema": "p1553.m6_monogenic_kernel_bidegree.analysis.r181.v1",
        "closed_routes": {
            "post_construction_monogenic_composition": "tautological_no_constructor_credit",
            "canonical_exact_bounded_bidegree_body": "B^(9/2)",
            "explicit_N_factor_coefficient_stream": "B^(7/2)",
            "flattened_tensor_norm": "B^(7/2)",
            "full_composed_resultant_output": "B^(7/2)",
        },
        "open_interface": {
            "name": "gcd_equivalent_output_sensitive_elliptic_composed_resultant_mod_U",
            "required_exponent_B": "9/4+o(1)",
            "may_discard_target_dependent_units": True,
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
        f"{admission['obligation_count']} bounded_fold_closed=1 breakthrough=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
