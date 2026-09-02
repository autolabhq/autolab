#!/usr/bin/env python3
"""Audit Khatri-Rao rank amplification for the torus C5 selector route."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
from fractions import Fraction
from typing import Any, Iterable, Sequence


SCHEMA = "p1553.torus_c5_khatri_rao_kruskal_amplification.r135.v1"
SETUP_CAP = Fraction(9, 4)
QUERY_CAP = Fraction(0)
SOURCE_DEGREE = 5
DETERMINISTIC_ATOM_KRANK = 2
DETERMINISTIC_PRODUCT_KRANK = 6
RANDOM_BOUND_EPSILON = Fraction(1, 2)

R134_PRODUCER = pathlib.Path(
    "p1553_torus_c5_two_atom_geometric_progression_probe_r134.py"
)
R134_PRODUCER_SHA256 = (
    "42be98b69e5ba120e79d75d52b66e85719e4e81e59e0a6ba9f5dfb1fd804420c"
)
R134_REPORT = pathlib.Path(
    "p1553_torus_c5_two_atom_geometric_progression_"
    "probe_report_r134.json"
)
R134_REPORT_SHA256 = (
    "29b274cce5b88242f31e07906147157a44a3e783774413a230a1c4e4a679d9cf"
)
R134_FROZEN = pathlib.Path(
    "frozen_torus_c5_two_atom_geometric_progression.json"
)
R134_FROZEN_SHA256 = (
    "2f9fb995e8a0b6821365c7ee404130cc1d6d58006e59012806a2105ffc2e3829"
)
R134_COST = pathlib.Path(
    "torus_c5_two_atom_geometric_progression_cost_ledger.json"
)
R134_COST_SHA256 = (
    "bd223baa1e454f7b9b09cf5bb4b5c733bc94f46daa1e853c8cfd2ace4c7454a6"
)
R134_REPLAY = pathlib.Path(
    "torus_c5_two_atom_geometric_progression_replay.json"
)
R134_REPLAY_SHA256 = (
    "384774470b2d92c2eaade2038ec1eb753e263f756e3a2c32ec1fd858b4082e04"
)
R134_CONTROLS = pathlib.Path(
    "torus_c5_two_atom_geometric_progression_controls.json"
)
R134_CONTROLS_SHA256 = (
    "dfc7c9d0f13d21cd3bee1bba5a0dd7d727ef5df0fb7969ff66d9384293ea0a56"
)
R134_LOGS = pathlib.Path("factor_logs_and_identical_descent_r134.json")
R134_LOGS_SHA256 = (
    "4784ebaa2d16e9dbc7fad24588a7c924b32f4a3efbc5633540d33f8fdb45a859"
)
R134_TEST = pathlib.Path(
    "tasks/ecdlp_index_calculus/tests/"
    "test_p1553_torus_c5_two_atom_geometric_progression_probe_r134.py"
)
R134_TEST_SHA256 = (
    "1b71034e755e072967fff685cba29d65240e62b585090c64ac6702ca88379ad5"
)
R134_GATE = pathlib.Path(
    "p1553_torus_c5_two_atom_geometric_progression_probe_gate_r134.md"
)
R134_GATE_SHA256 = (
    "161eca40da4f5fd1a861892a628168f10176b592f786913f24d07bf37f634f7c"
)
R134_PARENT = pathlib.Path(
    "p1553_torus_c5_two_atom_geometric_progression_"
    "probe_parent_report_r134.yaml"
)
R134_PARENT_SHA256 = (
    "40c2423dc8939c25335465091840cc13e3a08e4f60dc9ff7e5ad556090e98dc3"
)
BHASKARA_PDF = pathlib.Path(
    "references/"
    "bhaskara_charikar_vijayaraghavan_tensor_uniqueness_1304.8087.pdf"
)
BHASKARA_PDF_SHA256 = (
    "1667cbac1975b46257ac46ef887e11fca01d9f45d8bc951824671ef5d5e23618"
)


def load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R134 = load_module("p1553_r134_for_r135", R134_PRODUCER)
R133 = R134.R133
R132 = R134.R132
R131 = R134.R131
R129 = R134.R129
R126 = R134.R126
R121 = R134.R121
R82 = R134.R82
Field = R134.Field
Fp2 = tuple[int, int]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding_records() -> dict[str, dict[str, str]]:
    rows = (
        ("r134_producer", R134_PRODUCER, R134_PRODUCER_SHA256),
        ("r134_report", R134_REPORT, R134_REPORT_SHA256),
        ("r134_frozen", R134_FROZEN, R134_FROZEN_SHA256),
        ("r134_cost", R134_COST, R134_COST_SHA256),
        ("r134_replay", R134_REPLAY, R134_REPLAY_SHA256),
        ("r134_controls", R134_CONTROLS, R134_CONTROLS_SHA256),
        ("r134_logs", R134_LOGS, R134_LOGS_SHA256),
        ("r134_test", R134_TEST, R134_TEST_SHA256),
        ("r134_gate", R134_GATE, R134_GATE_SHA256),
        ("r134_parent", R134_PARENT, R134_PARENT_SHA256),
        ("bhaskara_pdf", BHASKARA_PDF, BHASKARA_PDF_SHA256),
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
        raise AssertionError(f"R135 source binding mismatch: {failures}")
    return actual


def fraction_record(value: Fraction) -> dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return {"exact": exact, "decimal": float(value)}


def fp2_matrix_rank(
    matrix: Iterable[Iterable[Fp2]],
    field: Field,
) -> int:
    rows = [list(row) for row in matrix]
    if not rows:
        return 0
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise ValueError("matrix rows must have equal length")
    pivot_row = 0
    for column in range(column_count):
        pivot = next(
            (
                index
                for index in range(pivot_row, len(rows))
                if rows[index][column] != field.zero
            ),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inverse = field.inv(rows[pivot_row][column])
        rows[pivot_row] = [
            field.mul(value, inverse) for value in rows[pivot_row]
        ]
        for index, row in enumerate(rows):
            if index == pivot_row or row[column] == field.zero:
                continue
            scale = row[column]
            rows[index] = [
                field.sub(value, field.mul(scale, pivot_value))
                for value, pivot_value in zip(row, rows[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


def column_submatrix(
    matrix: Sequence[Sequence[Fp2]],
    columns: Sequence[int],
) -> list[list[Fp2]]:
    return [[row[column] for column in columns] for row in matrix]


def kruskal_rank(
    matrix: Sequence[Sequence[Fp2]],
    field: Field,
) -> int:
    if not matrix:
        return 0
    column_count = len(matrix[0])
    maximum = min(len(matrix), column_count)
    result = 0
    for size in range(1, maximum + 1):
        if not all(
            fp2_matrix_rank(column_submatrix(matrix, columns), field)
            == size
            for columns in itertools.combinations(range(column_count), size)
        ):
            break
        result = size
    return result


def sample_modes(subgroup_order: int) -> tuple[int, ...]:
    modes = (
        0,
        1,
        2,
        3,
        subgroup_order - 3,
        subgroup_order - 2,
        subgroup_order - 1,
    )
    if len(set(modes)) != 7:
        raise AssertionError("seven-mode control bank collided")
    return modes


def evaluation_matrix(
    values: Sequence[Fp2],
    modes: Sequence[int],
    field: Field,
) -> list[list[Fp2]]:
    return [
        [field.pow(value, exponent) for exponent in modes]
        for value in values
    ]


def two_atom_product_values(
    left: Fp2,
    right: Fp2,
    field: Field,
) -> tuple[Fp2, ...]:
    return tuple(
        field.mul(
            field.pow(left, SOURCE_DEGREE - right_count),
            field.pow(right, right_count),
        )
        for right_count in range(SOURCE_DEGREE + 1)
    )


def ordered_khatri_rao_rows(
    atoms: Sequence[Fp2],
    modes: Sequence[int],
    field: Field,
) -> list[list[Fp2]]:
    rows = []
    for ordered_source in itertools.product(atoms, repeat=SOURCE_DEGREE):
        product = field.product(ordered_source)
        rows.append(
            [field.pow(product, exponent) for exponent in modes]
        )
    return rows


def canonical_row_set(
    matrix: Iterable[Iterable[Fp2]],
) -> set[tuple[Fp2, ...]]:
    return {tuple(row) for row in matrix}


def random_atom_krank_diagnostic(
    subgroup_order: int,
    minimum_color_size: int,
) -> dict[str, Any]:
    floor_log2_q = subgroup_order.bit_length() - 1
    maximum_atom_mode_count = 1 + floor_log2_q // 2
    amplified_product_mode_count = (
        SOURCE_DEGREE * (maximum_atom_mode_count - 1) + 1
    )
    root_density = Fraction(1, 2)
    log2_bound = (
        2
        + math.log2(maximum_atom_mode_count)
        + maximum_atom_mode_count * math.log2(36)
        + 3 * maximum_atom_mode_count * math.log2(subgroup_order)
        + minimum_color_size * math.log2(float(root_density))
    )
    return {
        "epsilon": fraction_record(RANDOM_BOUND_EPSILON),
        "maximum_atom_mode_count": maximum_atom_mode_count,
        "amplified_product_mode_count": amplified_product_mode_count,
        "root_density_upper_bound": fraction_record(root_density),
        "minimum_color_atom_count": minimum_color_size,
        "log2_union_bound": log2_bound,
        "union_bound_below_one": log2_bound < 0,
        "actual_control_receives_probability_credit": False,
    }


def color_control(
    field: Field,
    deck: tuple[Fp2, ...],
    c5_by_source: dict[tuple[int, ...], Fp2],
    color: int,
    part: tuple[int, ...],
    subgroup_order: int,
) -> dict[str, Any]:
    base = {
        "color": color,
        "part_indices": list(part),
        "part_size": len(part),
        "witness_available": len(part) >= 2,
        "finite_control_receives_asymptotic_credit": False,
    }
    if len(part) < 2:
        return {
            **base,
            "unavailable_reason": "finite_color_part_has_fewer_than_two_atoms",
        }
    left_index, right_index = part[:2]
    atoms = (deck[left_index], deck[right_index])
    modes = sample_modes(subgroup_order)
    atom_matrix = evaluation_matrix(atoms, modes, field)
    product_values = two_atom_product_values(atoms[0], atoms[1], field)
    product_matrix = evaluation_matrix(product_values, modes, field)
    ordered_matrix = ordered_khatri_rao_rows(atoms, modes, field)
    sources = R134.progression_sources(left_index, right_index)
    return {
        **base,
        "left_index": left_index,
        "right_index": right_index,
        "sample_modes": list(modes),
        "atom_matrix_rank": fp2_matrix_rank(atom_matrix, field),
        "atom_matrix_kruskal_rank": kruskal_rank(atom_matrix, field),
        "expected_atom_kruskal_rank": DETERMINISTIC_ATOM_KRANK,
        "product_matrix_row_count": len(product_matrix),
        "product_matrix_rank": fp2_matrix_rank(product_matrix, field),
        "product_matrix_kruskal_rank": kruskal_rank(
            product_matrix,
            field,
        ),
        "expected_product_kruskal_rank": DETERMINISTIC_PRODUCT_KRANK,
        "seven_product_columns_dependent": (
            fp2_matrix_rank(product_matrix, field)
            < len(modes)
        ),
        "ordered_khatri_rao_row_count": len(ordered_matrix),
        "ordered_khatri_rao_unique_row_count": len(
            canonical_row_set(ordered_matrix)
        ),
        "deduplicated_khatri_rao_rows_equal_product_rows": (
            canonical_row_set(ordered_matrix)
            == canonical_row_set(product_matrix)
        ),
        "source_count": len(sources),
        "all_sources_in_c5_replay": all(
            source in c5_by_source for source in sources
        ),
        "all_source_values_match_product_rows": all(
            c5_by_source[source] == value
            for source, value in zip(sources, product_values)
        ),
        "all_sources_accepted_by_color": all(
            R131.source_color_multiplicity(source, color)
            >= R134.COLOR_ACCEPTANCE_THRESHOLD
            for source in sources
        ),
    }


def finite_control(curve: dict[str, Any], offset: int) -> dict[str, Any]:
    field, _, deck_values = R121.pairing_deck(curve, offset)
    deck = tuple(deck_values)
    c5 = R126.source_products(field, deck, SOURCE_DEGREE)
    c5_by_source = {source: value for source, value in c5}
    parts = tuple(
        part
        for part in R129.balanced_four_parts(len(deck))
        if part
    )
    colors = [
        color_control(
            field,
            deck,
            c5_by_source,
            color,
            part,
            curve["subgroup_order"],
        )
        for color, part in enumerate(parts)
    ]
    available = [row for row in colors if row["witness_available"]]
    minimum_color_size = min(len(part) for part in parts)
    return {
        "control_id": f"{curve['family_id']}_offset{offset}",
        "field_prime": field.p,
        "subgroup_order": curve["subgroup_order"],
        "subgroup_order_probable_prime": R82.R70.is_prime(
            curve["subgroup_order"]
        ),
        "field_characteristic_equals_six_q_minus_one": (
            field.p == 6 * curve["subgroup_order"] - 1
        ),
        "coefficient_field_size_below_36_q_squared": (
            field.p * field.p
            < 36 * curve["subgroup_order"] * curve["subgroup_order"]
        ),
        "deck_size": len(deck),
        "c5_source_count": len(c5),
        "active_color_count": len(colors),
        "available_two_atom_control_count": len(available),
        "color_controls": colors,
        "all_available_atom_kranks_equal_two": all(
            row["atom_matrix_kruskal_rank"]
            == DETERMINISTIC_ATOM_KRANK
            for row in available
        ),
        "all_available_product_kranks_equal_six": all(
            row["product_matrix_kruskal_rank"]
            == DETERMINISTIC_PRODUCT_KRANK
            for row in available
        ),
        "all_available_seven_column_matrices_dependent": all(
            row["seven_product_columns_dependent"] for row in available
        ),
        "all_available_khatri_rao_rows_replay": all(
            row["ordered_khatri_rao_unique_row_count"] == 6
            and row["deduplicated_khatri_rao_rows_equal_product_rows"]
            for row in available
        ),
        "all_available_sources_replay_and_are_accepted": all(
            row["all_sources_in_c5_replay"]
            and row["all_source_values_match_product_rows"]
            and row["all_sources_accepted_by_color"]
            for row in available
        ),
        "random_atom_krank_diagnostic": random_atom_krank_diagnostic(
            curve["subgroup_order"],
            minimum_color_size,
        ),
        "candidate_discrete_logs_consumed": False,
        "finite_control_receives_asymptotic_credit": False,
    }


def finite_controls() -> dict[str, Any]:
    controls = [
        finite_control(curve, offset)
        for curve in R82.FAMILIES
        for offset in (0, 1)
    ]
    witness_count = sum(
        row["available_two_atom_control_count"] for row in controls
    )
    return {
        "schema": (
            "p1553.torus_c5_khatri_rao_kruskal_amplification_"
            "controls.r135.v1"
        ),
        "controls": controls,
        "control_count": len(controls),
        "active_color_control_count": sum(
            row["active_color_count"] for row in controls
        ),
        "available_two_atom_control_count": witness_count,
        "all_subgroup_orders_probable_prime": all(
            row["subgroup_order_probable_prime"] for row in controls
        ),
        "all_fields_satisfy_p_equals_six_q_minus_one": all(
            row["field_characteristic_equals_six_q_minus_one"]
            for row in controls
        ),
        "all_coefficient_fields_below_36_q_squared": all(
            row["coefficient_field_size_below_36_q_squared"]
            for row in controls
        ),
        "all_available_atom_kranks_equal_two": (
            witness_count > 0
            and all(
                row["all_available_atom_kranks_equal_two"]
                for row in controls
            )
        ),
        "all_available_product_kranks_equal_six": (
            witness_count > 0
            and all(
                row["all_available_product_kranks_equal_six"]
                for row in controls
            )
        ),
        "all_available_seven_column_matrices_dependent": (
            witness_count > 0
            and all(
                row["all_available_seven_column_matrices_dependent"]
                for row in controls
            )
        ),
        "all_available_khatri_rao_rows_replay": (
            witness_count > 0
            and all(
                row["all_available_khatri_rao_rows_replay"]
                for row in controls
            )
        ),
        "all_available_sources_replay_and_are_accepted": (
            witness_count > 0
            and all(
                row["all_available_sources_replay_and_are_accepted"]
                for row in controls
            )
        ),
        "all_actual_random_union_bounds_below_one": all(
            row["random_atom_krank_diagnostic"]["union_bound_below_one"]
            for row in controls
        ),
        "actual_random_union_bounds_receive_credit": False,
        "finite_controls_receive_asymptotic_credit": False,
        "candidate_discrete_logs_consumed": False,
    }


def theorem_record() -> dict[str, Any]:
    return {
        "primary_source": {
            "authors": [
                "Aditya Bhaskara",
                "Moses Charikar",
                "Aravindan Vijayaraghavan",
            ],
            "title": (
                "Uniqueness of Tensor Decompositions with Applications "
                "to Polynomial Identifiability"
            ),
            "identifier": "arXiv:1304.8087",
            "source_result": (
                "Definition 3.8 and Lemma A.4, used only as provenance "
                "for the Khatri-Rao and Kruskal-rank inequality"
            ),
            "pinned_sha256": BHASKARA_PDF_SHA256,
        },
        "finite_field_khatri_rao_lemma": (
            "Over any field, let A and B have the same R indexed columns "
            "and Kruskal ranks k_A and k_B. Then the columnwise tensor "
            "product A odot B has Kruskal rank at least "
            "min(k_A+k_B-1,R). For a dependence on a set S of at most "
            "k_A+k_B-1 columns, choose a nonzero coefficient i. Partition "
            "S minus i into sets of sizes at most k_A-1 and k_B-1. "
            "Kruskal independence supplies dual functionals killing the "
            "respective sets but not column i. Their tensor product "
            "isolates the nonzero coefficient, a contradiction."
        ),
        "fifth_power_amplification": (
            "Induction gives krank(A odot A odot A odot A odot A) at "
            "least min(5*(krank(A)-1)+1,R)."
        ),
        "c5_matrix_identification": (
            "For atom rows a and exponent-mode columns e, A[a,e]=a^e. "
            "A fifth Khatri-Rao row indexed by (a1,...,a5) has entry "
            "product(a_j)^e, exactly the represented-mode evaluation row "
            "of the degree-five source product. Permutations and product "
            "collisions create duplicate rows only; deleting duplicate "
            "rows does not change column dependencies. The deduplicated "
            "matrix is the all-five-atoms-from-this-color submatrix of the "
            "C5 color-acceptance evaluation matrix, which is sufficient to "
            "obstruct vanishing on the full acceptance support."
        ),
        "deterministic_structured_consequence": (
            "Every asymptotic balanced color has two distinct atoms, so "
            "its all-mode atom matrix has Kruskal rank at least two. The "
            "fifth-power lemma therefore gives product Kruskal rank at "
            "least six. This recovers the R134 one-through-six-mode "
            "obstruction, but supplies no deterministic seventh-mode "
            "credit."
        ),
        "random_deck_model": {
            "assumptions": [
                "q is prime and q is not 5",
                "the deck is an ordered uniform sample without replacement from H",
                "the four balanced color parts are fixed independently of deck values",
                "the coefficient field has size p^2 with p<6q",
                "predicates are represented sparse sums with distinct exponents modulo q",
            ],
            "atom_krank_bound": (
                "For 0<epsilon<1 set "
                "T=1+floor((1-epsilon)*log2(q)). Kelley-adapted sparse "
                "root bounds give root density at most "
                "rho_epsilon=2^(-epsilon/(1-epsilon)) for every nonzero "
                "represented polynomial with at most T modes."
            ),
            "projective_polynomial_count": (
                "For each t<=T, support choices are at most q^t and "
                "projective coefficient choices are below "
                "36^t*q^(2t), so all represented predicates through T "
                "modes number below T*36^T*q^(3T)."
            ),
            "union_bound": (
                "If m is the smallest color part, the probability that "
                "any color atom matrix has Kruskal rank below T is at "
                "most 4*T*36^T*q^(3T)*rho_epsilon^m."
            ),
            "amplified_consequence": (
                "With the complementary probability, every represented "
                "C5 product predicate through 5*(T-1)+1 modes has a "
                "nonzero evaluation on every color acceptance support."
            ),
            "near_five_log_threshold": (
                "Since m=q^(3/20+o(1)), epsilon may tend to zero slowly "
                "while epsilon*m dominates (log q)^2. Thus the excluded "
                "product-mode count is (5-o(1))*log2(q) with "
                "overwhelming probability under this model only."
            ),
            "structured_factor_base_transfer_proved": False,
            "receives_candidate_credit": False,
        },
        "field_independent_ordinary_rank_lemma": True,
        "robust_real_rank_claim_imported": False,
        "maximum_deterministic_excluded_mode_count": 6,
        "random_model_excluded_mode_asymptotic": "(5-o(1))*log2(q)",
        "not_covered": [
            "deterministic structured-factor-base atom Kruskal rank above two",
            "multiple-predicate or adaptive decision DAGs",
            "nonzero-value tests and coordinate comparisons",
            "polynomials with large expansion but compact straight-line programs",
            "general arithmetic-circuit, RAM, or cell-probe lower bounds",
        ],
    }


def cost_ledger(theorem: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "p1553.torus_c5_khatri_rao_kruskal_amplification_"
            "cost_ledger.r135.v1"
        ),
        "caps": {
            "setup_exponent_B": fraction_record(SETUP_CAP),
            "per_arbitrary_target_query_exponent_B": fraction_record(
                QUERY_CAP
            ),
        },
        "theorem": theorem,
        "routes": [
            {
                "route_id": "one_to_six_mode_structured_zero_or_pole",
                "random_model_required": False,
                "structured_factor_base_theorem": True,
                "status": (
                    "rejected_deterministically_by_khatri_rao_"
                    "amplification_from_atom_krank_two"
                ),
            },
            {
                "route_id": "near_five_log_mode_random_deck_zero_or_pole",
                "maximum_mode_count": "(5-o(1))*log2(q)",
                "random_model_required": True,
                "structured_factor_base_theorem": False,
                "status": (
                    "rejected_with_overwhelming_probability_under_"
                    "uniform_random_deck_model_only"
                ),
            },
            {
                "route_id": "seven_plus_mode_structured_zero_or_pole",
                "structured_atom_krank_above_two_proved": False,
                "inside_cap_exact_predicate_constructed": False,
                "status": "open",
            },
            {
                "route_id": "multiple_small_mode_predicate_dag",
                "single_predicate_krank_obstruction_sufficient": False,
                "inside_cap_exact_dag_constructed": False,
                "status": "open",
            },
            {
                "route_id": "low_slp_expanded_extension_predicate",
                "represented_mode_bound_applicable": False,
                "inside_cap_exact_predicate_constructed": False,
                "status": "open",
            },
            {
                "route_id": "nonzero_value_frobenius_coordinate_dag",
                "zero_set_obstruction_sufficient": False,
                "inside_cap_exact_dag_constructed": False,
                "status": "open",
            },
        ],
        "candidate_work_credit": False,
    }


def build_bundle() -> dict[str, dict[str, Any]]:
    source_hashes = verify_source_bindings()
    inherited = json.loads(R134_REPORT.read_text(encoding="utf-8"))
    if inherited.get("breakthrough") or inherited.get(
        "shoup_bound_improvement"
    ):
        raise AssertionError("R134 nonclaim boundary drifted")
    theorem = theorem_record()
    controls = finite_controls()
    cost = cost_ledger(theorem)
    routes = {row["route_id"]: row for row in cost["routes"]}
    obligations = {
        "eleven_source_bindings_verified": len(source_hashes) == 11,
        "r134_six_mode_interface_inherited": (
            inherited["admission"][
                "deterministic_one_to_six_mode_negative_admitted"
            ]
            and not inherited["admission"]["lane_admitted"]
        ),
        "primary_khatri_rao_source_pinned": (
            theorem["primary_source"]["pinned_sha256"]
            == BHASKARA_PDF_SHA256
        ),
        "finite_field_dual_functional_proof_explicit": (
            "dual functionals"
            in theorem["finite_field_khatri_rao_lemma"]
        ),
        "fifth_power_krank_amplification_explicit": (
            "5*(krank(A)-1)+1"
            in theorem["fifth_power_amplification"]
        ),
        "c5_evaluation_matrix_identification_explicit": (
            "degree-five source product"
            in theorem["c5_matrix_identification"]
        ),
        "deterministic_credit_capped_at_six_modes": (
            theorem["maximum_deterministic_excluded_mode_count"] == 6
        ),
        "random_model_near_five_log_bound_explicit": (
            theorem["random_model_excluded_mode_asymptotic"]
            == "(5-o(1))*log2(q)"
        ),
        "random_model_receives_no_structured_credit": (
            not theorem["random_deck_model"][
                "structured_factor_base_transfer_proved"
            ]
            and not theorem["random_deck_model"]["receives_candidate_credit"]
        ),
        "eight_actual_controls_complete": controls["control_count"] == 8,
        "twelve_two_atom_rank_controls_complete": (
            controls["available_two_atom_control_count"] == 12
        ),
        "all_available_khatri_rao_replays_exact": (
            controls["all_available_atom_kranks_equal_two"]
            and controls["all_available_product_kranks_equal_six"]
            and controls[
                "all_available_seven_column_matrices_dependent"
            ]
            and controls["all_available_khatri_rao_rows_replay"]
            and controls[
                "all_available_sources_replay_and_are_accepted"
            ]
        ),
        "finite_and_random_diagnostics_receive_no_credit": (
            not controls["actual_random_union_bounds_receive_credit"]
            and not controls["finite_controls_receive_asymptotic_credit"]
        ),
        "structured_seven_multi_dag_and_slp_routes_preserved": all(
            routes[route_id]["status"] == "open"
            for route_id in (
                "seven_plus_mode_structured_zero_or_pole",
                "multiple_small_mode_predicate_dag",
                "low_slp_expanded_extension_predicate",
                "nonzero_value_frobenius_coordinate_dag",
            )
        ),
        "inside_cap_asymmetric_predicate_complete": False,
        "inside_cap_five_source_index_complete": False,
        "known_rhs_relation_rank_complete": False,
        "factor_logs_complete": False,
        "identical_target_descent_complete": False,
        "pollard_rho_improvement_complete": False,
        "shoup_improvement_complete": False,
        "breakthrough_complete": False,
    }
    next_action = (
        "Attack the surviving grammar rather than increasing sparse support: "
        "test a multiple-small-predicate decision DAG, a nonzero-value "
        "Frobenius-coordinate branch, or a high-expansion low-SLP predicate "
        "against the full C5 color supports. Freeze every branch and "
        "coefficient, replay exact positive and empty paths and C2+C3 "
        "sources, fit B^(9/4+o(1)) state and polylogarithmic arbitrary-target "
        "work, avoid field DLP, and charge rank, logs, identical descent, "
        "memory, field operations, and bits."
    )
    frozen = {
        "schema": (
            "p1553.frozen_torus_c5_khatri_rao_kruskal_"
            "amplification.r135.v1"
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
            "single represented structured color predicates through six modes",
            (
                "single represented predicates through "
                "(5-o(1))*log2(q) modes under the uniform-random-deck "
                "model only"
            ),
        ],
        "preserved_interface": (
            "structured seven-plus-mode predicate, multiple-predicate DAG, "
            "nonzero-value Frobenius-coordinate branch, or high-expansion "
            "low-SLP predicate"
        ),
        "random_deck_model_required_above_six_modes": True,
        "general_finite_field_circuit_lower_bound_claimed": False,
    }
    replay = {
        "schema": (
            "p1553.torus_c5_khatri_rao_kruskal_amplification_"
            "replay.r135.v1"
        ),
        "actual_control_count": controls["control_count"],
        "active_color_control_count": controls[
            "active_color_control_count"
        ],
        "available_two_atom_control_count": controls[
            "available_two_atom_control_count"
        ],
        "all_available_atom_kranks_equal_two": controls[
            "all_available_atom_kranks_equal_two"
        ],
        "all_available_product_kranks_equal_six": controls[
            "all_available_product_kranks_equal_six"
        ],
        "all_available_khatri_rao_rows_replay": controls[
            "all_available_khatri_rao_rows_replay"
        ],
        "actual_random_union_bounds_receive_credit": False,
        "inside_cap_surviving_selector_constructed": False,
        "candidate_work_credit": False,
    }
    logs_descent = {
        "schema": "p1553.factor_logs_and_identical_descent.r135.v1",
        "r134_two_atom_progression_audit_complete": True,
        "r135_khatri_rao_amplification_audit_complete": True,
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
        "C5_COLOR_EVALUATION_IS_FIFTH_KHATRI_RAO_POWER_OF_ATOM_MATRIX__"
        "ORDINARY_KRUSKAL_RANK_INEQUALITY_PROVED_OVER_ANY_FIELD__ATOM_"
        "KRANK_TWO_RECOVERS_DETERMINISTIC_SIX_MODE_OBSTRUCTION__UNIFORM_"
        "RANDOM_DECK_ATOM_KRANK_AMPLIFIES_TO_NEAR_FIVE_LOG2_Q_PRODUCT_"
        "MODES_WITH_OVERWHELMING_PROBABILITY__TWELVE_EXACT_FINITE_RANK_"
        "REPLAYS__NO_STRUCTURED_CREDIT_ABOVE_SIX__MULTI_PREDICATE_NONZERO_"
        "VALUE_LOW_SLP_ROUTES_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH"
    )
    report = {
        "schema": SCHEMA,
        "claim_status": (
            "FINITE_FIELD_KHATRI_RAO_AMPLIFICATION_AND_RANDOM_MODEL_"
            "SPARSE_EXCLUSION_ONLY_WITHHOLD_PROMOTION"
        ),
        "classification": classification,
        "source_bindings": source_binding_records(),
        "theorem": theorem,
        "finite_evidence": {
            "control_count": controls["control_count"],
            "active_color_control_count": controls[
                "active_color_control_count"
            ],
            "available_two_atom_control_count": controls[
                "available_two_atom_control_count"
            ],
            "all_available_atom_kranks_equal_two": controls[
                "all_available_atom_kranks_equal_two"
            ],
            "all_available_product_kranks_equal_six": controls[
                "all_available_product_kranks_equal_six"
            ],
            "all_actual_random_union_bounds_below_one": controls[
                "all_actual_random_union_bounds_below_one"
            ],
            "asymptotic_credit": False,
        },
        "admission": {
            "obligations": obligations,
            "passed_obligation_count": sum(obligations.values()),
            "obligation_count": len(obligations),
            "deterministic_one_to_six_mode_negative_admitted": True,
            "random_model_near_five_log_mode_negative_admitted": True,
            "structured_factor_base_above_six_admitted": False,
            "lane_admitted": False,
        },
        "artifacts": {
            "frozen": (
                "frozen_torus_c5_khatri_rao_kruskal_amplification.json"
            ),
            "cost": (
                "torus_c5_khatri_rao_kruskal_amplification_cost_ledger.json"
            ),
            "source_replay": (
                "torus_c5_khatri_rao_kruskal_amplification_replay.json"
            ),
            "controls": (
                "torus_c5_khatri_rao_kruskal_amplification_controls.json"
            ),
            "logs_descent": "factor_logs_and_identical_descent_r135.json",
        },
        "next_action": next_action,
        "non_claims": [
            "The primary paper is provenance; the finite-field ordinary-rank proof is self-contained.",
            "The random-deck union bound does not transfer to the structured factor base.",
            "The actual small controls receive no probability or asymptotic credit.",
            "No deterministic structured seventh-mode obstruction is supplied.",
            "Multiple-predicate, nonzero-value, and low-SLP DAGs remain open.",
            "No general circuit, RAM, or cell-probe lower bound is claimed.",
            "No source index, rank, logs, descent, rho, or Shoup result is supplied.",
        ],
        "pollard_rho_improvement": False,
        "shoup_bound_improvement": False,
        "breakthrough": False,
        "disposition": (
            "ADMIT_FIELD_INDEPENDENT_KHATRI_RAO_KRUSKAL_RANK_LEMMA__"
            "ADMIT_DETERMINISTIC_SIX_MODE_RECOVERY__ADMIT_NEAR_FIVE_LOG2_Q_"
            "EXCLUSION_UNDER_UNIFORM_RANDOM_DECK_MODEL_ONLY__ADMIT_TWELVE_"
            "EXACT_FINITE_RANK_REPLAYS_WITHOUT_ASYMPTOTIC_CREDIT__PRESERVE_"
            "STRUCTURED_SEVEN_MODE_MULTI_PREDICATE_NONZERO_VALUE_AND_LOW_"
            "SLP_ROUTES__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_"
            "SHOUP__NO_BREAKTHROUGH"
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
            "p1553_torus_c5_khatri_rao_kruskal_amplification_"
            "probe_report_r135.json"
        ),
    )
    parser.add_argument(
        "--frozen-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "frozen_torus_c5_khatri_rao_kruskal_amplification.json"
        ),
    )
    parser.add_argument(
        "--cost-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_khatri_rao_kruskal_amplification_cost_ledger.json"
        ),
    )
    parser.add_argument(
        "--replay-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_khatri_rao_kruskal_amplification_replay.json"
        ),
    )
    parser.add_argument(
        "--controls-output",
        type=pathlib.Path,
        default=pathlib.Path(
            "torus_c5_khatri_rao_kruskal_amplification_controls.json"
        ),
    )
    parser.add_argument(
        "--logs-output",
        type=pathlib.Path,
        default=pathlib.Path("factor_logs_and_identical_descent_r135.json"),
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
        f"R135 classification={report['classification']} "
        f"obligations={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} "
        f"lane_admitted={admission['lane_admitted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
