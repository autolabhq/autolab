#!/usr/bin/env python3
"""Hash-bind and evaluate the IDEA-340 public-chart attack obligations."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ecdlp.p1436_idea340_public_chart_preflight.v1"
EXPECTED_SCHEMAS = {
    "p1407": "ecdlp.low_term_total2_p1407_coordinate_advice_compression_expansion_after_p1406.v1",
    "p1408": "ecdlp.low_term_total2_p1408_rational_map_image_factor_bases_after_p1407.v1",
    "p1416": "ecdlp.p1416_coordinate_additive_energy_recursive_s3.v1",
    "p1432": "ecdlp.p1432_prospective_high_additive_energy_factor_bases.v1",
    "r68": "p1553.ffe_fixed_sum_information_conservation.r68.v1",
}
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
LIVE_ROOT = Path("/Volumes/Volume/git/autolab")
LIVE_STATE = LIVE_ROOT / "ecdlp_index_calculus_state"
DEFAULT_PATHS = {
    "p1407": LIVE_STATE
    / "p1407_coordinate_advice_compression_expansion_after_p1406_probe.json",
    "p1408": LIVE_STATE
    / "p1408_rational_map_image_factor_bases_after_p1407_probe.json",
    "p1416": LIVE_STATE
    / "p1416_coordinate_additive_energy_recursive_s3_after_p1415_probe.json",
    "p1432": LIVE_STATE / "p1432_high_energy_factor_bases_after_p1431_probe.json",
    "r68": WORKTREE_ROOT
    / "p1553_ffe_fixed_sum_information_conservation_report_r68.json",
}
DEFAULT_OUTPUT = (
    WORKTREE_ROOT
    / "ecdlp_index_calculus_state"
    / "p1436_idea340_public_chart_preflight.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def validate_schemas(inputs: dict[str, dict[str, Any]]) -> None:
    mismatches = {
        name: {
            "expected": EXPECTED_SCHEMAS[name],
            "observed": payload.get("schema"),
        }
        for name, payload in inputs.items()
        if payload.get("schema") != EXPECTED_SCHEMAS[name]
    }
    if mismatches:
        raise ValueError(f"input schema mismatch: {json.dumps(mismatches, sort_keys=True)}")


def evaluate(inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the preflight independently of filesystem provenance."""

    validate_schemas(inputs)
    p1407 = inputs["p1407"]
    p1408 = inputs["p1408"]
    p1416 = inputs["p1416"]
    p1432 = inputs["p1432"]
    r68 = inputs["r68"]
    s1407 = p1407["summary"]
    s1408 = p1408["summary"]
    s1416 = p1416["summary"]
    s1432 = p1432["summary"]

    recovery_receipts = []
    for label, payload, summary in (
        ("p1407_coordinate", p1407, s1407),
        ("p1408_map", p1408, s1408),
    ):
        tested = int(summary["evaluated_target_policy_count"])
        recovered = int(summary["public_recovery_success_count"])
        invalid = int(summary["invalid_advice_witness_count"])
        forbidden = list(payload.get("forbidden_selector_inputs_used") or [])
        recovery_receipts.append(
            {
                "source": label,
                "tested": tested,
                "recovered": recovered,
                "invalid_witnesses": invalid,
                "forbidden_selector_inputs_used": forbidden,
                "pass": recovered == tested and invalid == 0 and not forbidden,
            }
        )
    public_construction_pass = all(row["pass"] for row in recovery_receipts)

    policy_audit = s1416["promotion_audit"]["policies"]
    public_policy_names = sorted(policy_audit)
    exact_policy_names = sorted(
        name
        for name, audit in policy_audit.items()
        if audit["exactness_rank_descent_gate"]
    )
    exactness_pass = (
        int(s1416["invalid_witness_count"]) == 0
        and int(s1416["full_rank_policy_curve_count"])
        == int(s1416["policy_curve_count"])
        and int(s1416["target_success_count"]) == int(s1416["target_total"])
        and exact_policy_names == public_policy_names
    )

    energy_policy_names = sorted(
        name
        for name, audit in policy_audit.items()
        if audit["additive_energy_gate"]
    )
    energy_pass = bool(energy_policy_names)

    full_rank_cells = int(
        s1432["full_factor_rank_cell_count_in_fixed_512_target_batch"]
    )
    prospective_cells = int(s1432["policy_curve_count"])
    prospective_rank_fraction = (
        full_rank_cells / prospective_cells if prospective_cells else 0.0
    )
    rank_persistence_pass = (
        bool(s1432["fresh_curve_persistence_gate"])
        and prospective_cells > 0
        and full_rank_cells == prospective_cells
    )

    symbolic_source_policy_names = sorted(
        name
        for name, audit in policy_audit.items()
        if audit["symbolic_B2_5_gate"]
    )
    explicit_join_exponent = float(
        s1432["explicit_pair_triple_join_symbolic_basis_exponent_in_n"]
    )
    selected_work_exponent = float(
        s1432["projected_selected_basis_work_exponent_in_n"]
    )
    subquadratic_source_pass = (
        bool(symbolic_source_policy_names)
        and explicit_join_exponent < 0.5
        and selected_work_exponent < 0.5
    )

    heldout_ratios: dict[str, list[float]] = {
        name: [] for name in public_policy_names
    }
    heldout_curve_count = 0
    for record in p1416["curve_records"]:
        if record["split"] != "heldout":
            continue
        heldout_curve_count += 1
        for name in public_policy_names:
            heldout_ratios[name].append(
                float(
                    record["policies"][name]["costs"][
                        "total_tested_batch_field_units_over_sign_rho"
                    ]
                )
            )
    qualifying_below_rho_policies = sorted(
        name for name, ratios in heldout_ratios.items() if ratios and max(ratios) < 1.0
    )
    flattened_ratios = [
        ratio for ratios in heldout_ratios.values() for ratio in ratios
    ]
    below_rho_pass = bool(qualifying_below_rho_policies)

    fresh_persistence_pass = (
        bool(s1432["prospective_generation_started_after_training_freeze"])
        and bool(s1432["fresh_curve_persistence_gate"])
    )

    obligations = {
        "public_dlp_free_factor_base_construction": {
            "pass": public_construction_pass,
            "receipts": recovery_receipts,
            "generator_multiple_positive_control_credited": False,
        },
        "exact_relation_rank_and_target_descent": {
            "pass": exactness_pass,
            "full_rank_policy_curve_count": s1416[
                "full_rank_policy_curve_count"
            ],
            "policy_curve_count": s1416["policy_curve_count"],
            "target_success_count": s1416["target_success_count"],
            "target_total": s1416["target_total"],
            "invalid_witness_count": s1416["invalid_witness_count"],
            "exact_public_policies": exact_policy_names,
        },
        "superuniform_heldout_additive_energy": {
            "pass": energy_pass,
            "qualifying_public_policies": energy_policy_names,
            "tested_public_policies": public_policy_names,
        },
        "stable_prospective_independent_rank": {
            "pass": rank_persistence_pass,
            "fresh_curve_persistence_gate": s1432[
                "fresh_curve_persistence_gate"
            ],
            "full_rank_cells": full_rank_cells,
            "policy_curve_cells": prospective_cells,
            "full_rank_fraction": prospective_rank_fraction,
        },
        "sub_sqrt_source_enumeration": {
            "pass": subquadratic_source_pass,
            "symbolic_B2_5_passing_public_policies": symbolic_source_policy_names,
            "explicit_pair_triple_join_exponent_in_n": explicit_join_exponent,
            "selected_projected_basis_work_exponent_in_n": selected_work_exponent,
            "required_exponent_strictly_below": 0.5,
        },
        "heldout_end_to_end_below_rho": {
            "pass": below_rho_pass,
            "heldout_curve_count": heldout_curve_count,
            "tested_public_policy_count": len(public_policy_names),
            "qualifying_public_policies": qualifying_below_rho_policies,
            "minimum_ratio_over_all_public_policy_cells": (
                min(flattened_ratios) if flattened_ratios else None
            ),
            "maximum_ratio_over_all_public_policy_cells": (
                max(flattened_ratios) if flattened_ratios else None
            ),
            "per_policy_maximum_ratio": {
                name: max(ratios) if ratios else None
                for name, ratios in heldout_ratios.items()
            },
        },
        "fresh_heldout_persistence": {
            "pass": fresh_persistence_pass,
            "prospective_generation_started_after_training_freeze": s1432[
                "prospective_generation_started_after_training_freeze"
            ],
            "fresh_curve_persistence_gate": s1432[
                "fresh_curve_persistence_gate"
            ],
        },
    }
    failed_obligations = [
        name for name, obligation in obligations.items() if not obligation["pass"]
    ]
    lane_admitted = not failed_obligations
    return {
        "classification": (
            "IDEA340_PUBLIC_CHART_LANE_ADMITTED"
            if lane_admitted
            else "IDEA340_PUBLIC_CHART_PREFLIGHT_REJECTED"
        ),
        "obligations": obligations,
        "admission": {
            "lane_admitted": lane_admitted,
            "failed_obligations": failed_obligations,
            "passed_obligation_count": len(obligations) - len(failed_obligations),
            "obligation_count": len(obligations),
        },
        "r68_binding": {
            "information_conservation_pass": bool(r68["pass"]),
            "ffe_product_relations_add_information_beyond_fixed_sum_rows": r68[
                "result"
            ]["ffe_product_relations_add_information_beyond_fixed_sum_rows"],
            "pencil_factorization_can_only_help_by_finding_new_fixed_sum_rows": r68[
                "result"
            ]["pencil_factorization_can_only_help_by_finding_new_fixed_sum_rows"],
            "product_or_quotient_relation_rank_credit": 0,
        },
        "claim_boundary": {
            "algorithm_breakthrough": lane_admitted,
            "generic_prime_field_speedup": lane_admitted,
            "shoup_bound_improvement": lane_admitted,
            "positive_controls_are_attack_evidence": False,
        },
        "next_action": (
            "Freeze a disjoint train/heldout split for the 17-of-18 preserving "
            "slice-quadratic surfaces and require a public pre-choice slice selector "
            "with all rejected candidates charged."
        ),
    }


def build_payload(
    paths: dict[str, Path],
) -> dict[str, Any]:
    inputs = {name: read_json(path) for name, path in paths.items()}
    result = evaluate(inputs)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "IDEA-340 public coordinate/rational-map factor bases",
        "source_bindings": {
            name: {
                "path": str(paths[name]),
                "sha256": sha256_file(paths[name]),
                "schema": inputs[name]["schema"],
                "claim_status": inputs[name].get("claim_status"),
            }
            for name in sorted(paths)
        },
        **result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in DEFAULT_PATHS.items():
        parser.add_argument(f"--{name}", type=Path, default=path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {name: getattr(args, name) for name in DEFAULT_PATHS}
    payload = build_payload(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    admission = payload["admission"]
    rho = payload["obligations"]["heldout_end_to_end_below_rho"]
    print(
        f"output={args.output} admitted={admission['lane_admitted']} "
        f"passed={admission['passed_obligation_count']}/"
        f"{admission['obligation_count']} min_rho_ratio="
        f"{rho['minimum_ratio_over_all_public_policy_cells']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
