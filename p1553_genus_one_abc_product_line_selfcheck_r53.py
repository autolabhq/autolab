#!/usr/bin/env python3
"""Check the exact arithmetic scope of the R53 genus-one abc gate."""

import json


def abc_case(name, genus, line_degree, gcd_degree, radical_support):
    moving_degree = line_degree - gcd_degree
    bound_rhs = radical_support + 2 * genus - 2
    return {
        "name": name,
        "genus": genus,
        "line_bundle_degree": line_degree,
        "common_divisor_degree": gcd_degree,
        "moving_map_degree": moving_degree,
        "three_fiber_radical_support": radical_support,
        "abc_bound_rhs": bound_rhs,
        "abc_slack": bound_rhs - moving_degree,
        "abc_admissible": moving_degree <= bound_rhs,
    }


def run():
    with open(
        "p1553_translated_product_shared_factor_audit_report_r52.json",
        encoding="utf-8",
    ) as input_file:
        r52_report = json.load(input_file)
    r52_result = r52_report["result"]
    r52_lines = r52_report["line_classifications"]
    r52_base_degree = r52_result["base_divisor_degree"]
    r52_moving_degree = r52_result["residual_map_degree"]
    r52_line_degree = r52_base_degree + r52_moving_degree
    r52_radical_supports = [
        len(
            set().union(
                *(set(points) for points in line["residual_point_blocks"])
            )
        )
        for line in r52_lines
    ]

    r52 = abc_case(
        name="r52_after_common_divisor_cancellation",
        genus=1,
        line_degree=r52_line_degree,
        gcd_degree=r52_base_degree,
        radical_support=r52_radical_supports[0],
    )
    primitive = abc_case(
        name="hypothetical_primitive_reduced_degree_nine",
        genus=1,
        line_degree=9,
        gcd_degree=0,
        radical_support=27,
    )
    radical_deficit = abc_case(
        name="impossible_genus_one_radical_deficit_control",
        genus=1,
        line_degree=9,
        gcd_degree=0,
        radical_support=8,
    )

    reduced_product_controls = []
    factor_degree = 3
    for factor_count in range(1, 33):
        moving_degree = factor_count * factor_degree
        radical_support = 3 * moving_degree
        reduced_product_controls.append(
            {
                "factor_count": factor_count,
                "factor_degree": factor_degree,
                "moving_degree": moving_degree,
                "radical_support": radical_support,
                "abc_slack": radical_support - moving_degree,
                "expected_slack": 2 * factor_count * factor_degree,
                "abc_admissible": moving_degree <= radical_support,
            }
        )

    characteristic = 193
    frozen_toy_moving_degree = r52["moving_map_degree"]
    checks = {
        "pinned_r52_report_passes": r52_report["pass"] is True,
        "pinned_r52_report_has_228_lines": len(r52_lines) == 228,
        "pinned_r52_base_divisor_degree_is_six": r52_base_degree == 6,
        "pinned_r52_line_bundle_degree_is_nine": r52_line_degree == 9,
        "all_r52_residuals_are_three_disjoint_three_point_fibers": all(
            len(line["residual_point_blocks"]) == 3
            and all(len(points) == 3 for points in line["residual_point_blocks"])
            and support == 9
            for line, support in zip(r52_lines, r52_radical_supports)
        ),
        "r52_common_divisor_cancels_degree_six": r52["moving_map_degree"] == 3,
        "r52_residual_radical_is_three_reduced_three_point_fibers": r52[
            "three_fiber_radical_support"
        ]
        == 9,
        "r52_abc_is_admissible_with_slack_six": r52["abc_admissible"]
        and r52["abc_slack"] == 6,
        "primitive_degree_nine_control_is_admissible_with_slack_eighteen": primitive[
            "abc_admissible"
        ]
        and primitive["abc_slack"] == 18,
        "strict_genus_one_radical_deficit_is_rejected": not radical_deficit[
            "abc_admissible"
        ],
        "all_reduced_product_controls_have_slack_2rd": all(
            control["abc_admissible"]
            and control["abc_slack"] == control["expected_slack"]
            for control in reduced_product_controls
        ),
        "frozen_toy_map_degree_below_characteristic": frozen_toy_moving_degree
        < characteristic,
    }
    if not all(checks.values()):
        raise AssertionError(f"R53 abc scope self-check failed: {checks}")

    return {
        "schema": "p1553.genus_one_abc_product_line_selfcheck.r53.v1",
        "classification": [
            "theorem-scope-arithmetic",
            "exact",
            "non-run-cryptanalysis",
            "model-bound",
            "novelty-unverified",
        ],
        "theorem_scope": {
            "separable_curve_bound": "m <= n + 2g - 2",
            "genus_one_bound": "m <= n",
            "m": "moving map degree after common divisor cancellation",
            "n": "distinct support across zero, one, and pole fibers",
            "factor_labels_visible_to_bound": False,
        },
        "frozen_characteristic": characteristic,
        "cases": {
            "r52": r52,
            "primitive_degree_nine": primitive,
            "radical_deficit_negative_control": radical_deficit,
        },
        "reduced_degree_three_factor_controls": reduced_product_controls,
        "checks": checks,
        "result": {
            "single_three_fiber_inequality_closes_r52": False,
            "single_three_fiber_inequality_excludes_primitive_reduced_product_pencils": False,
            "single_three_fiber_inequality_detects_only_total_radical_deficit": True,
            "auxiliary_factor_sensitive_abc_or_s_unit_arguments": "open",
            "factor_sensitive_trisecant_classification_required": True,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "limits": [
            "the arithmetic self-check does not prove Riemann-Hurwitz",
            "the gate does not classify trisecants of the multiplication image",
            "the gate supplies no asymptotic pencil family or target locator",
            "no R10 output, relation-rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
