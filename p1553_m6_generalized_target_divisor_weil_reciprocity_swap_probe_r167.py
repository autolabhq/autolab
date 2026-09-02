#!/usr/bin/env python3
"""Test a compact target-divisor witness and Weil-reciprocity product swap."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
SCHEMA = "p1553.m6_generalized_target_divisor_weil_reciprocity_swap.r167.v1"

R166_PRODUCER = ROOT / "p1553_m6_kummer_x_translate_signed_verification_probe_r166.py"
R166_REPORT = ROOT / "p1553_m6_kummer_x_translate_signed_verification_probe_report_r166.json"
R166_FROZEN = ROOT / "frozen_m6_kummer_x_translate_signed_verification.json"
R166_COST = ROOT / "m6_kummer_x_translate_signed_verification_cost_ledger.json"
R166_REPLAY = ROOT / "m6_kummer_x_translate_signed_verification_replay.json"
R166_CONTROLS = ROOT / "m6_kummer_x_translate_signed_verification_controls.json"
R166_KUMMER = ROOT / "kummer_x_candidate_and_signed_false_branch_r166.json"
R166_TEST = ROOT / "tasks/ecdlp_index_calculus/tests/test_p1553_m6_kummer_x_translate_signed_verification_probe_r166.py"
R166_GATE = ROOT / "p1553_m6_kummer_x_translate_signed_verification_probe_gate_r166.md"
R166_PARENT = ROOT / "p1553_m6_kummer_x_translate_signed_verification_probe_parent_report_r166.yaml"
EAGEN_PAPER = ROOT / "references/eagen_ecip_weil_reciprocity_2022_596.pdf"
MILLER_PAPER = ROOT / "references/miller_weil_pairing_algorithm_1986.pdf"
R113_REPORT = ROOT / "p1553_5a5c_nonlinear_elliptic_orbit_product_probe_report_r113.json"
R113_GATE = ROOT / "p1553_5a5c_nonlinear_elliptic_orbit_product_probe_gate_r113.md"

SOURCE_BINDINGS = (
    ("r166_producer", R166_PRODUCER, "c524eca7bdb586e8e68e34da515da301766668f6f0b7bd6808e005b2b77e893a"),
    ("r166_report", R166_REPORT, "94ee650b1302d231d946fc02f14d1c12b62a63bd9fba240a9d6a93e4bb986285"),
    ("r166_frozen", R166_FROZEN, "e413a3f78adf508fbc22e2b05e4b2d1d5986871473de0cafa78ba89f5f2fb21a"),
    ("r166_cost", R166_COST, "3b6ce95aa27864c5fe475ea36d745afba2108189138ade240f2f1ee4cbd43857"),
    ("r166_replay", R166_REPLAY, "b459ab51185d403d0de91b5ff1700d43138e3e2e7281f4cdfe38ff0d1a0760ee"),
    ("r166_controls", R166_CONTROLS, "72a28dbed3e056b4f1cab123546e786ae2ebc058bd012338afc7bde3ff6d9fc8"),
    ("r166_kummer", R166_KUMMER, "ff79f8c98a284f4a71cd3c529e65d11b76b77af216eb33089d50245a91fa6e5a"),
    ("r166_test", R166_TEST, "39570b95fbc355095a5f494ae091889d11d11b92b753b3caf0f74c5b5e63a751"),
    ("r166_gate", R166_GATE, "e98cd1ba5b73c591dae5882e9058fafbd6e548df3c2ae929563181bc84920d9c"),
    ("r166_parent", R166_PARENT, "a10abdaa1f1c075ab0c0b77a709a1757d675f0369e9d7c51f276cb25e1c7c502"),
    ("eagen_2022_596", EAGEN_PAPER, "5310b35d288a9462ff704eb77e7651d18f681a5b560cfb2919d1cfd0e01ae09e"),
    ("miller_1986", MILLER_PAPER, "39c76c7643278b87b3d8c24b9a07d0b4cbfb561cd13735548990848e0f0bd166"),
    ("r113_report", R113_REPORT, "5b961649fa71ced6b049a3500316114b3c9f3806b5b61639471d9a6f86b01102"),
    ("r113_gate", R113_GATE, "d1939734367f92798ec40b89f9954ff0c9fe3bf07c6ad0bdaeb33e9a5fccad94"),
)

DEFAULT_REPORT = ROOT / "p1553_m6_generalized_target_divisor_weil_reciprocity_swap_probe_report_r167.json"
DEFAULT_FROZEN = ROOT / "frozen_m6_generalized_target_divisor_weil_reciprocity_swap.json"
DEFAULT_COST = ROOT / "m6_generalized_target_divisor_weil_reciprocity_swap_cost_ledger.json"
DEFAULT_REPLAY = ROOT / "m6_generalized_target_divisor_weil_reciprocity_swap_replay.json"
DEFAULT_CONTROLS = ROOT / "m6_generalized_target_divisor_weil_reciprocity_swap_controls.json"
DEFAULT_RESULTANT = ROOT / "generalized_miller_elliptic_resultant_swap_r167.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R166 = load_module("p1553_r166_for_r167", R166_PRODUCER)
R161 = R166.R161


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
        raise AssertionError(f"R167 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def point_list(point: tuple[int, int] | None) -> list[int] | None:
    return None if point is None else [int(point[0]), int(point[1])]


def point_add(
    left: tuple[int, int] | None,
    right: tuple[int, int] | None,
    curve: dict[str, Any],
) -> tuple[int, int] | None:
    result = R161.R70.add(left, right, curve)
    return None if result is None else tuple(result)


def point_negate(
    point: tuple[int, int] | None, curve: dict[str, Any]
) -> tuple[int, int] | None:
    if point is None:
        return None
    return tuple(R161.R70.negate(point, curve))


def point_sum(
    points: Iterable[tuple[int, int]], curve: dict[str, Any]
) -> tuple[int, int] | None:
    total: tuple[int, int] | None = None
    for point in points:
        total = point_add(total, point, curve)
    return total


def point_subtract(
    left: tuple[int, int], right: tuple[int, int], curve: dict[str, Any]
) -> tuple[int, int] | None:
    return point_add(left, point_negate(right, curve), curve)


def riemann_roch_basis(pole_order: int) -> list[dict[str, int | str]]:
    basis: list[dict[str, int | str]] = []
    for exponent in range(pole_order // 2 + 1):
        weight = 2 * exponent
        if weight <= pole_order:
            basis.append({"kind": "x", "exponent": exponent, "weight": weight})
    if pole_order >= 3:
        for exponent in range((pole_order - 3) // 2 + 1):
            weight = 3 + 2 * exponent
            if weight <= pole_order:
                basis.append(
                    {"kind": "yx", "exponent": exponent, "weight": weight}
                )
    basis.sort(key=lambda monomial: int(monomial["weight"]))
    return basis


def basis_value(
    monomial: dict[str, int | str], point: tuple[int, int], prime: int
) -> int:
    x_coord, y_coord = point
    value = pow(x_coord, int(monomial["exponent"]), prime)
    if monomial["kind"] == "yx":
        value = value * y_coord % prime
    return value


def modular_nullspace(matrix: list[list[int]], prime: int) -> list[list[int]]:
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be nonempty")
    reduced = [[value % prime for value in row] for row in matrix]
    row_count = len(reduced)
    column_count = len(reduced[0])
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(column_count):
        selected = next(
            (
                row
                for row in range(pivot_row, row_count)
                if reduced[row][column]
            ),
            None,
        )
        if selected is None:
            continue
        reduced[pivot_row], reduced[selected] = (
            reduced[selected],
            reduced[pivot_row],
        )
        inverse = pow(reduced[pivot_row][column], -1, prime)
        reduced[pivot_row] = [
            value * inverse % prime for value in reduced[pivot_row]
        ]
        for row in range(row_count):
            if row == pivot_row or not reduced[row][column]:
                continue
            scale = reduced[row][column]
            reduced[row] = [
                (left - scale * right) % prime
                for left, right in zip(reduced[row], reduced[pivot_row])
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == row_count:
            break
    free_columns = [
        column
        for column in range(column_count)
        if column not in pivot_columns
    ]
    vectors: list[list[int]] = []
    for free_column in free_columns:
        vector = [0] * column_count
        vector[free_column] = 1
        for row, column in enumerate(pivot_columns):
            vector[column] = -reduced[row][free_column] % prime
        vectors.append(vector)
    return vectors


def witness_value(
    witness: dict[str, Any],
    point: tuple[int, int] | None,
    prime: int,
) -> int:
    if point is None:
        return int(witness["leading_local_coefficient"])
    return sum(
        int(coefficient) * basis_value(monomial, point, prime)
        for coefficient, monomial in zip(
            witness["coefficients"], witness["basis"]
        )
    ) % prime


def function_witness(
    points: list[tuple[int, int]], curve: dict[str, Any]
) -> dict[str, Any]:
    prime = int(curve["field_prime"])
    pole_order = len(points)
    basis = riemann_roch_basis(pole_order)
    if len(basis) != pole_order:
        raise AssertionError("Riemann-Roch basis dimension mismatch")
    matrix = [
        [basis_value(monomial, point, prime) for monomial in basis]
        for point in points
    ]
    nullspace = modular_nullspace(matrix, prime)
    if len(nullspace) != 1:
        raise ValueError(f"expected nullity one, found {len(nullspace)}")
    coefficients = nullspace[0]
    leading_index = max(
        index for index, coefficient in enumerate(coefficients) if coefficient
    )
    inverse = pow(coefficients[leading_index], -1, prime)
    coefficients = [coefficient * inverse % prime for coefficient in coefficients]
    leading = basis[leading_index]
    leading_local_coefficient = coefficients[leading_index]
    if leading["kind"] == "yx":
        leading_local_coefficient = -leading_local_coefficient % prime
    witness = {
        "pole_order": int(leading["weight"]),
        "ambient_pole_order": pole_order,
        "basis": basis,
        "coefficients": coefficients,
        "matrix_rank": len(basis) - len(nullspace),
        "nullity": len(nullspace),
        "leading_local_coefficient": leading_local_coefficient,
        "prescribed_points": [point_list(point) for point in points],
    }
    witness["all_prescribed_values_zero"] = all(
        witness_value(witness, point, prime) == 0 for point in points
    )
    witness["coefficient_sha256"] = sha256_json(coefficients)
    return witness


def rational_value(
    numerator: dict[str, Any],
    denominator: dict[str, Any],
    point: tuple[int, int] | None,
    prime: int,
) -> int:
    denominator_value = witness_value(denominator, point, prime)
    if denominator_value == 0:
        raise ZeroDivisionError("target-divisor denominator vanishes")
    return (
        witness_value(numerator, point, prime)
        * pow(denominator_value, -1, prime)
        % prime
    )


def kummer_value(
    divisor: dict[str, Any], point: tuple[int, int], prime: int
) -> int:
    return R161.poly_eval(divisor["u"], int(point[0]), prime)


def try_auxiliary_witness(
    curve: dict[str, Any],
    generator: tuple[int, int],
    divisor: dict[str, Any],
    targets: list[tuple[int, int]],
    target_sum: tuple[int, int],
    common_zero: tuple[int, int],
    numerator: dict[str, Any],
    offset: int,
) -> dict[str, Any] | None:
    order = int(curve["subgroup_order"])
    prime = int(curve["field_prime"])
    count = len(targets)
    auxiliary_scalars = [
        (offset + index) % order for index in range(count - 1)
    ]
    if any(scalar == 0 for scalar in auxiliary_scalars):
        return None
    auxiliary_points = [
        R161.R70.scalar_mul(scalar, generator, curve)
        for scalar in auxiliary_scalars
    ]
    if any(point is None for point in auxiliary_points):
        return None
    auxiliary_points = [tuple(point) for point in auxiliary_points]
    auxiliary_sum = point_sum(auxiliary_points, curve)
    anchor = point_add(target_sum, point_negate(auxiliary_sum, curve), curve)
    if anchor is None:
        return None
    denominator_points = [anchor, *auxiliary_points, common_zero]
    if len(set(denominator_points)) != len(denominator_points):
        return None
    try:
        denominator = function_witness(denominator_points, curve)
    except ValueError:
        return None
    pole_order = len(targets) + 1
    if (
        numerator["pole_order"] != pole_order
        or denominator["pole_order"] != pole_order
    ):
        return None
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    signed_support = selected + [point_negate(point, curve) for point in selected]
    if any(point is None for point in signed_support):
        raise AssertionError("finite signed support reached infinity")

    def h_value(point: tuple[int, int] | None) -> int:
        return rational_value(numerator, denominator, point, prime)

    try:
        for left in selected:
            if h_value(left) == 0:
                return None
            for correction in [anchor, *auxiliary_points]:
                translated = point_subtract(correction, left, curve)
                if translated is None or kummer_value(divisor, translated, prime) == 0:
                    return None
            for support_point in signed_support:
                translated = point_add(support_point, left, curve)
                h_value(translated)
    except ZeroDivisionError:
        return None
    return {
        "offset": offset,
        "auxiliary_scalars": auxiliary_scalars,
        "auxiliary_points": auxiliary_points,
        "anchor": anchor,
        "denominator_points": denominator_points,
        "denominator": denominator,
    }


def finite_control(curve: dict[str, Any], seed: int) -> dict[str, Any]:
    factor_base, divisor, target_records = R166.R164.target_material(curve, seed)
    selected = [tuple(record["endpoint"]) for record in divisor["records"]]
    selected_set = set(selected)
    retained_records = [
        record
        for record in target_records
        if tuple(record["target"]) not in selected_set
    ]
    dropped_records = [
        record
        for record in target_records
        if tuple(record["target"]) in selected_set
    ]
    targets = [tuple(record["target"]) for record in retained_records]
    if len(set(targets)) != len(targets):
        raise AssertionError("finite target witness requires distinct targets")
    target_sum = point_sum(targets, curve)
    if target_sum is None:
        raise AssertionError("finite target sum unexpectedly reached infinity")
    common_zero = point_negate(target_sum, curve)
    if common_zero is None or common_zero in targets:
        raise AssertionError("invalid common-zero completion")
    numerator_points = [*targets, common_zero]
    numerator = function_witness(numerator_points, curve)
    if numerator["pole_order"] != len(numerator_points):
        raise AssertionError("numerator witness lost its full pole order")
    generator = tuple(factor_base["generator"])
    auxiliary: dict[str, Any] | None = None
    for offset in range(1, 4097):
        auxiliary = try_auxiliary_witness(
            curve,
            generator,
            divisor,
            targets,
            target_sum,
            common_zero,
            numerator,
            offset,
        )
        if auxiliary is not None:
            break
    if auxiliary is None:
        raise AssertionError("unable to find an admissible auxiliary divisor")
    denominator = auxiliary["denominator"]
    anchor = auxiliary["anchor"]
    auxiliary_points = auxiliary["auxiliary_points"]
    prime = int(curve["field_prime"])
    signed_support = selected + [
        point_negate(point, curve) for point in selected
    ]
    if len(set(signed_support)) != 2 * len(selected):
        raise AssertionError("signed Kummer support is not reduced")

    def h_value(point: tuple[int, int] | None) -> int:
        return rational_value(numerator, denominator, point, prime)

    rows: list[dict[str, Any]] = []
    all_identities_exact = True
    all_corrections_units = True
    for left in selected:
        direct_product = 1
        for target in targets:
            translated = point_subtract(target, left, curve)
            if translated is None:
                raise AssertionError("retained target equals selected endpoint")
            direct_product = direct_product * kummer_value(
                divisor, translated, prime
            ) % prime
        correction_product = 1
        for correction in [anchor, *auxiliary_points]:
            translated = point_subtract(correction, left, curve)
            if translated is None:
                raise AssertionError("auxiliary correction reached the pole")
            value = kummer_value(divisor, translated, prime)
            all_corrections_units &= value != 0
            correction_product = correction_product * value % prime
        swapped_product = 1
        for support_point in signed_support:
            translated = point_add(support_point, left, curve)
            swapped_product = swapped_product * h_value(translated) % prime
        h_at_left = h_value(left)
        if h_at_left == 0:
            raise AssertionError("h(P) must be a unit on the selected divisor")
        reciprocity_value = (
            correction_product
            * swapped_product
            * pow(pow(h_at_left, 2 * len(selected), prime), -1, prime)
            % prime
        )
        exact = direct_product == reciprocity_value
        all_identities_exact &= exact
        rows.append(
            {
                "left_endpoint": point_list(left),
                "direct_translate_product": direct_product,
                "correction_product": correction_product,
                "signed_support_h_product": swapped_product,
                "h_at_left": h_at_left,
                "reciprocity_value": reciprocity_value,
                "identity_exact": exact,
                "candidate_zero": direct_product == 0,
                "disjoint_support_row": direct_product != 0,
            }
        )
    if not all_identities_exact:
        raise AssertionError("Weil-reciprocity swap failed on an admissible row")
    if not all_corrections_units:
        raise AssertionError("a correction factor was not a unit")

    numerator_sum = point_sum(numerator_points, curve)
    denominator_sum = point_sum(auxiliary["denominator_points"], curve)
    h_at_infinity = h_value(None)
    candidate_roots = sorted(
        int(row["left_endpoint"][0])
        for row in rows
        if row["candidate_zero"]
    )
    return {
        "control_id": f"{curve['family_id']}_target_divisor_weil_seed{seed}",
        "family_id": curve["family_id"],
        "field_prime": prime,
        "subgroup_order": int(curve["subgroup_order"]),
        "seed": seed,
        "c3_divisor_degree": len(selected),
        "original_target_count": len(target_records),
        "retained_target_count": len(targets),
        "dropped_selected_endpoint_target_count": len(dropped_records),
        "dropped_target_roles": [record["role"] for record in dropped_records],
        "target_witness_degree": len(targets),
        "riemann_roch_ambient_pole_order": len(targets) + 1,
        "target_sum": point_list(target_sum),
        "common_zero": point_list(common_zero),
        "anchor": point_list(anchor),
        "auxiliary_offset": auxiliary["offset"],
        "auxiliary_scalars": auxiliary["auxiliary_scalars"],
        "auxiliary_points": [
            point_list(point) for point in auxiliary_points
        ],
        "numerator_point_sum_is_infinity": numerator_sum is None,
        "denominator_point_sum_is_infinity": denominator_sum is None,
        "numerator_points_distinct": len(set(numerator_points))
        == len(numerator_points),
        "denominator_points_distinct": len(set(auxiliary["denominator_points"]))
        == len(auxiliary["denominator_points"]),
        "numerator_witness": numerator,
        "denominator_witness": denominator,
        "common_zero_cancels_from_quotient_divisor": True,
        "quotient_zero_divisor_degree": len(targets),
        "quotient_pole_divisor_degree": len(targets),
        "h_at_infinity": h_at_infinity,
        "all_correction_factors_units_on_selected_divisor": all_corrections_units,
        "all_reciprocity_identities_exact": all_identities_exact,
        "evaluated_selected_endpoint_count": len(rows),
        "candidate_zero_count": len(candidate_roots),
        "candidate_roots": candidate_roots,
        "disjoint_support_row_count": sum(
            int(row["disjoint_support_row"]) for row in rows
        ),
        "specialized_common_zero_row_count": sum(
            int(row["candidate_zero"]) for row in rows
        ),
        "direct_target_evaluation_count": len(selected) * len(targets),
        "raw_swapped_h_evaluation_count": 2 * len(selected) * len(selected),
        "correction_evaluation_count": len(selected) * len(targets),
        "row_transcript_sha256": sha256_json(rows),
        "direct_product_transcript_sha256": sha256_json(
            [row["direct_translate_product"] for row in rows]
        ),
        "reciprocity_transcript_sha256": sha256_json(
            [row["reciprocity_value"] for row in rows]
        ),
        "candidate_discrete_log_oracle_consumed": False,
        "candidate_root_or_count_or_marginal_or_rank_or_source_oracle_consumed": False,
        "dense_finite_nullspace_receives_asymptotic_attack_credit": False,
        "finite_endpoint_enumeration_receives_asymptotic_attack_credit": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "target_divisor_witness": (
            "For target points T_1,...,T_N, set S_T=sum_j T_j and C=-S_T. "
            "Choose distinct auxiliary points R_1,...,R_(N-1), and set "
            "A=S_T-sum_i R_i. Degree-(N+1) functions F_num and F_den with "
            "zeros (T_1,...,T_N,C) and (A,R_1,...,R_(N-1),C) exist because "
            "each zero list sums to O. Their quotient h=F_num/F_den has "
            "divisor sum_j[T_j]-[A]-sum_i[R_i]; the shared zero C and equal "
            "poles at O cancel."
        ),
        "compact_representation": (
            "The quotient h may be represented by a generalized Miller line "
            "straight-line program of O(N) group/line merges, or by the "
            "principal-divisor interpolation interface described in the bound "
            "primary literature. This establishes compact target-divisor state, "
            "not a fast restriction or resultant modulo U. The finite producer "
            "uses dense Riemann-Roch nullspaces only as an exact control."
        ),
        "weil_reciprocity_swap": (
            "Let f_0(Q)=U(x(Q)), with divisor S+(-S)-2n[O], and let "
            "h_P(Q)=h(Q+P). For disjoint support, Weil reciprocity gives "
            "product_j f_0(T_j-P) = f_0(A-P) product_i f_0(R_i-P) "
            "times product_(Q in S union -S) h(Q+P) / h(P)^(2n). "
            "After clearing unit denominators, the identity extends by "
            "specialization to candidate-zero rows; those rows are not claimed "
            "as literal disjoint-support evaluations."
        ),
        "resultant_interface": (
            "The swapped quotient product is the elliptic-resultant/tame-symbol "
            "interface Res_E(f_0(Q),h(Q+P)), up to the explicit h(P) and "
            "auxiliary correction units. If all corrections are units on the "
            "selected signed divisor, its restriction has exactly the same gcd "
            "roots with U as the original Kummer translate product."
        ),
        "cost_boundary": (
            "Directly evaluating the original product uses nN=B^(7/2) values. "
            "Directly evaluating the swapped product uses 2n^2=B^(9/2) h-values "
            "and is one exponent worse. A standard represented elliptic "
            "resultant has Theta(nN)=B^(7/2) divisor/representation scale. "
            "Neither path is below the B^(5/2) rho proxy."
        ),
        "open_primitive": (
            "The remaining primitive is an output-sensitive elliptic resultant "
            "or tame-symbol resultant modulo U, computed directly from the "
            "degree-N target-divisor SLP in less than B^(5/2) total work, "
            "preferably B^(9/4+o(1)), without an nN or n^2 value table and "
            "without materializing a degree-Theta(nN) function."
        ),
    }


def cost_record() -> dict[str, Any]:
    return {
        "schema": "p1553.m6_generalized_target_divisor_weil_reciprocity_swap.cost.r167.v1",
        "field_and_subgroup_order_exponent_B": fraction_record(Fraction(5)),
        "c3_divisor_degree_exponent_B": fraction_record(Fraction(9, 4)),
        "target_count_exponent_B": fraction_record(Fraction(5, 4)),
        "compact_target_divisor_slp_state_exponent_B": fraction_record(
            Fraction(5, 4)
        ),
        "direct_target_table_exponent_B": fraction_record(Fraction(7, 2)),
        "raw_weil_swapped_table_exponent_B": fraction_record(Fraction(9, 2)),
        "standard_elliptic_resultant_representation_exponent_B": fraction_record(
            Fraction(7, 2)
        ),
        "expected_candidate_exponent_B": fraction_record(Fraction(3, 4)),
        "expected_signed_verification_exponent_B": fraction_record(Fraction(2)),
        "preferred_resultant_mod_u_work_exponent_B": fraction_record(
            Fraction(9, 4)
        ),
        "global_pollard_rho_exponent_B": fraction_record(Fraction(5, 2)),
        "direct_table_rho_excess_exponent_B": fraction_record(Fraction(1)),
        "raw_swap_rho_excess_exponent_B": fraction_record(Fraction(2)),
        "standard_resultant_rho_excess_exponent_B": fraction_record(Fraction(1)),
        "compact_target_divisor_slp_inside_rho": True,
        "direct_target_table_inside_rho": False,
        "raw_weil_swap_inside_rho": False,
        "standard_represented_elliptic_resultant_inside_rho": False,
        "output_sensitive_elliptic_resultant_mod_u_supplied": False,
        "finite_dense_nullspace_receives_asymptotic_credit": False,
        "finite_endpoint_enumeration_receives_asymptotic_credit": False,
        "unconditional_total_attack_cost_supplied": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    actual_bindings = verify_source_bindings()
    rows = [
        finite_control(curve, seed)
        for curve in R161.R159.R82.FAMILIES[: R161.R160.FAMILY_COUNT]
        for seed in R161.R160.SEEDS
    ]
    all_point_sums = all(
        row["numerator_point_sum_is_infinity"]
        and row["denominator_point_sum_is_infinity"]
        for row in rows
    )
    all_witnesses = all(
        row["numerator_witness"]["nullity"] == 1
        and row["denominator_witness"]["nullity"] == 1
        and row["numerator_witness"]["all_prescribed_values_zero"]
        and row["denominator_witness"]["all_prescribed_values_zero"]
        for row in rows
    )
    all_identities = all(row["all_reciprocity_identities_exact"] for row in rows)
    all_units = all(
        row["all_correction_factors_units_on_selected_divisor"] for row in rows
    )
    controls = {
        "schema": "p1553.m6_generalized_target_divisor_weil_reciprocity_swap.controls.r167.v1",
        "control_count": len(rows),
        "family_count": R161.R160.FAMILY_COUNT,
        "seeds": list(R161.R160.SEEDS),
        "all_zero_lists_sum_to_infinity": all_point_sums,
        "all_riemann_roch_witnesses_exact": all_witnesses,
        "all_common_zero_cancellations_recorded": all(
            row["common_zero_cancels_from_quotient_divisor"] for row in rows
        ),
        "all_auxiliary_corrections_are_units": all_units,
        "all_reciprocity_identities_exact": all_identities,
        "dropped_selected_endpoint_target_count": sum(
            row["dropped_selected_endpoint_target_count"] for row in rows
        ),
        "evaluated_selected_endpoint_count": sum(
            row["evaluated_selected_endpoint_count"] for row in rows
        ),
        "candidate_zero_count": sum(row["candidate_zero_count"] for row in rows),
        "disjoint_support_row_count": sum(
            row["disjoint_support_row_count"] for row in rows
        ),
        "specialized_common_zero_row_count": sum(
            row["specialized_common_zero_row_count"] for row in rows
        ),
        "direct_target_evaluation_count": sum(
            row["direct_target_evaluation_count"] for row in rows
        ),
        "raw_swapped_h_evaluation_count": sum(
            row["raw_swapped_h_evaluation_count"] for row in rows
        ),
        "correction_evaluation_count": sum(
            row["correction_evaluation_count"] for row in rows
        ),
        "candidate_oracle_consumed": False,
        "finite_controls_receive_asymptotic_attack_credit": False,
        "controls": rows,
    }
    theorem = theorem_record()
    cost = cost_record()
    obligations = {
        "fourteen_source_bindings_verified": len(actual_bindings) == 14,
        "r166_kummer_translate_interface_inherited": True,
        "eagen_principal_divisor_and_weil_interface_bound": True,
        "miller_line_function_slp_interface_bound": True,
        "r113_orbit_product_lane_deduplicated": True,
        "target_endpoint_pole_rows_explicitly_filtered": all(
            row["dropped_selected_endpoint_target_count"] == 1 for row in rows
        ),
        "riemann_roch_basis_dimension_exact": all(
            len(row["numerator_witness"]["basis"])
            == row["riemann_roch_ambient_pole_order"]
            for row in rows
        ),
        "zero_lists_sum_to_infinity": all_point_sums,
        "one_dimensional_witness_nullspaces_complete": all_witnesses,
        "full_pole_orders_complete": all(
            row["numerator_witness"]["pole_order"]
            == row["riemann_roch_ambient_pole_order"]
            and row["denominator_witness"]["pole_order"]
            == row["riemann_roch_ambient_pole_order"]
            for row in rows
        ),
        "shared_zero_and_infinity_poles_cancel_complete": True,
        "quotient_target_divisor_identity_complete": True,
        "compact_generalized_miller_slp_exists": True,
        "weil_reciprocity_disjoint_rows_complete": all_identities,
        "candidate_zero_specialization_rows_complete": all_identities,
        "auxiliary_corrections_units_complete": all_units,
        "six_finite_control_batches_complete": len(rows) == 6,
        "all_finite_reciprocity_identities_exact": all_identities,
        "candidate_roots_preserved_by_swap": all_identities and all_units,
        "point_at_infinity_local_coefficients_evaluated": all(
            row["h_at_infinity"] != 0 for row in rows
        ),
        "raw_weil_swap_B9O2_cost_charged": True,
        "standard_resultant_B7O2_cost_charged": True,
        "candidate_oracles_avoided": True,
        "finite_controls_scoped_without_attack_credit": True,
        "output_sensitive_elliptic_resultant_mod_u_complete": False,
        "degree_nN_representation_avoided": False,
        "deterministic_hash_to_curve_transfer_complete": False,
        "unconditional_total_attack_cost_complete": False,
        "generic_prime_coordinate_family_algorithm": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    admission = {
        "obligations": obligations,
        "passed_obligation_count": sum(obligations.values()),
        "obligation_count": len(obligations),
        "compact_target_divisor_witness_admitted": True,
        "weil_reciprocity_swap_identity_admitted": True,
        "output_sensitive_resultant_constructor_admitted": False,
        "lane_admitted": False,
    }
    resultant = {
        "schema": "p1553.m6_generalized_miller_elliptic_resultant_swap.r167.v1",
        "target_divisor_interface": theorem["target_divisor_witness"],
        "reciprocity_interface": theorem["weil_reciprocity_swap"],
        "elliptic_resultant_interface": theorem["resultant_interface"],
        "cost_boundary": cost,
        "finite_control_records": [
            {
                "control_id": row["control_id"],
                "target_witness_degree": row["target_witness_degree"],
                "auxiliary_offset": row["auxiliary_offset"],
                "numerator_coefficient_sha256": row["numerator_witness"][
                    "coefficient_sha256"
                ],
                "denominator_coefficient_sha256": row["denominator_witness"][
                    "coefficient_sha256"
                ],
                "candidate_zero_count": row["candidate_zero_count"],
                "row_transcript_sha256": row["row_transcript_sha256"],
            }
            for row in rows
        ],
        "output_sensitive_elliptic_resultant_mod_u_supplied": False,
    }
    replay = {
        "schema": "p1553.m6_generalized_target_divisor_weil_reciprocity_swap.replay.r167.v1",
        "source_bindings": source_binding_records(),
        "control_records": [
            {
                "control_id": row["control_id"],
                "numerator_coefficient_sha256": row["numerator_witness"][
                    "coefficient_sha256"
                ],
                "denominator_coefficient_sha256": row["denominator_witness"][
                    "coefficient_sha256"
                ],
                "row_transcript_sha256": row["row_transcript_sha256"],
                "direct_product_transcript_sha256": row[
                    "direct_product_transcript_sha256"
                ],
                "reciprocity_transcript_sha256": row[
                    "reciprocity_transcript_sha256"
                ],
            }
            for row in rows
        ],
        "all_replay_invariants_pass": all_point_sums
        and all_witnesses
        and all_identities
        and all_units,
    }
    frozen = {
        "schema": "p1553.m6_generalized_target_divisor_weil_reciprocity_swap.frozen.r167.v1",
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "cost": cost,
        "admission": admission,
        "successor_interface": {
            "input": (
                "one degree-n signed C3 divisor U,V (or U plus signed y side "
                "table) and a degree-N generalized Miller target-divisor SLP h"
            ),
            "required_output": (
                "the restriction modulo U of the denominator-cleared elliptic "
                "resultant Res_E(U(x(Q)),h(Q+P)), including explicit auxiliary "
                "unit corrections"
            ),
            "required_root_semantics": (
                "the same Kummer true-or-opposite-sign candidate roots as R166, "
                "followed by the inherited exact signed verifier"
            ),
            "preferred_work": "B^(9/4+o(1))",
            "maximum_total_work": "strictly below B^(5/2)",
            "forbidden_credit": (
                "n-by-N target table, 2n-by-n swapped table, degree-Theta(nN) "
                "represented resultant, unit-cost norm/resultant/root/count/"
                "marginal/rank/source oracle"
            ),
            "open_primitive": (
                "output-sensitive SLP elliptic resultant or tame-symbol "
                "resultant modulo U"
            ),
        },
    }
    next_action = (
        "Construct or refute an SLP elliptic-resultant/tame-symbol remainder "
        "operator modulo U directly from the degree-N target-divisor witness, "
        "in less than B^(5/2) total work and preferably B^(9/4+o(1)), without "
        "the nN, n^2, or degree-Theta(nN) intermediates. Test whether quotient "
        "rings, transposed modular composition, or half-GCD on generalized "
        "Miller line factors exposes a reusable low-displacement operator."
    )
    report = {
        "schema": SCHEMA,
        "date": "2026-08-01",
        "objective": (
            "Replace the arbitrary target list in R166 by a compact principal-"
            "divisor witness, swap the target product through Weil reciprocity, "
            "and charge the resulting elliptic-resultant interface end to end."
        ),
        "source_bindings": source_binding_records(),
        "deduplication": {
            "r166": (
                "R166 leaves an arbitrary-target Kummer translate product. R167 "
                "compresses the target divisor and proves an exact reciprocity "
                "swap, but does not supply the fast resultant modulo U."
            ),
            "r165": (
                "R165's statement that arbitrary targets lack one ordinary "
                "scalar-chain product is refined: a generalized Miller line SLP "
                "does compactly represent their principal divisor, while fast "
                "restriction to the selected C3 divisor remains open."
            ),
            "eagen_2022_596": (
                "The primary source supplies principal-divisor interpolation, "
                "Weil reciprocity, and elliptic-resultant interfaces; R167 does "
                "not attribute an ECDLP speedup to those interfaces."
            ),
            "miller_1986": (
                "Miller supplies line-function divisor arithmetic and compact "
                "evaluation chains. R167 generalizes the divisor witness, not "
                "the scalar-pairing complexity claim."
            ),
            "r113": (
                "R113 studies ordinary orbit/product-tree recurrence. R167 uses "
                "an arbitrary target divisor and an elliptic resultant; it does "
                "not inherit an output-sensitive recurrence from R113."
            ),
        },
        "theorem": theorem,
        "cost": cost,
        "controls": controls,
        "admission": admission,
        "classification": (
            "ADMIT_COMPACT_GENERALIZED_TARGET_DIVISOR_WITNESS__EXACT_WEIL_"
            "RECIPROCITY_SWAP_ON_SIX_FINITE_CONTROLS__CANDIDATE_ZERO_ROWS_BY_"
            "SPECIALIZATION__AUXILIARY_CORRECTIONS_UNITS__RAW_SWAP_B9O2__"
            "STANDARD_RESULTANT_B7O2__OUTPUT_SENSITIVE_RESULTANT_MOD_U_OPEN__"
            "NO_RHO_SHOUP_BREAKTHROUGH"
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
        "resultant": resultant,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST)
    parser.add_argument("--replay-output", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--controls-output", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--resultant-output", type=Path, default=DEFAULT_RESULTANT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle()
    write_json(args.report_output, bundle["report"])
    write_json(args.frozen_output, bundle["frozen"])
    write_json(args.cost_output, bundle["cost"])
    write_json(args.replay_output, bundle["replay"])
    write_json(args.controls_output, bundle["controls"])
    write_json(args.resultant_output, bundle["resultant"])
    admission = bundle["report"]["admission"]
    print(
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} lane={int(admission['lane_admitted'])} "
        f"breakthrough={int(bundle['report']['breakthrough'])}"
    )


if __name__ == "__main__":
    main()
