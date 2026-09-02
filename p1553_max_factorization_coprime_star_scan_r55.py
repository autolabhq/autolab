#!/usr/bin/env python3
"""Exhaust the coprime catalog-line star through the R54 center section."""

import json

import p1553_coprime_product_trisecant_probe_r54 as r54


CENTER_BLOCK = tuple(range(36, 45))


def run():
    catalog = r54.build_catalog()
    center_mask = sum(1 << point for point in CENTER_BLOCK)
    center_index = catalog["block_masks"].index(center_mask)
    center_section = catalog["sections"][center_index]
    center_multiplicity = catalog["factorizations_per_block"][center_mask]

    coprime_partner_indices = [
        index
        for index, mask in enumerate(catalog["block_masks"])
        if not (center_mask & mask)
    ]
    positive_lines = []
    for partner_index in coprime_partner_indices:
        hits = r54.interior_catalog_hits(
            center_section,
            catalog["sections"][partner_index],
            catalog["section_index"],
        )
        if not hits:
            continue
        positive_lines.append(
            {
                "partner_block_index": partner_index,
                "partner_block": list(catalog["blocks"][partner_index]),
                "interior_hits": [
                    {
                        "line_scalar": scalar,
                        "block_index": hit_index,
                        "block": list(catalog["blocks"][hit_index]),
                    }
                    for scalar, hit_index in hits
                ],
            }
        )

    inherited_control = r54.inherited_positive_control(catalog)
    checks = {
        "center_block_is_middle_window": catalog["blocks"][center_index]
        == CENTER_BLOCK,
        "center_factorization_multiplicity_is_280": center_multiplicity == 280,
        "every_partner_is_coprime_to_center": all(
            not (center_mask & catalog["block_masks"][index])
            for index in coprime_partner_indices
        ),
        "every_coprime_catalog_partner_scanned": len(coprime_partner_indices)
        == sum(not (center_mask & mask) for mask in catalog["block_masks"]),
        "r52_inherited_positive_control_detected": inherited_control[
            "expected_third_block_detected"
        ],
    }
    if not all(checks.values()):
        raise AssertionError(f"R55 maximum-factorization star scan failed: {checks}")

    return {
        "schema": "p1553.max_factorization_coprime_star_scan.r55.v1",
        "classification": [
            "toy",
            "exact-structured-star",
            "exhaustive-coprime-partners",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": r54.PRIME,
        "subgroup_order": r54.r44.r38.SUBGROUP_ORDER,
        "catalog": {
            "unique_product_sections": len(catalog["sections"]),
            "center_block_index": center_index,
            "center_block": list(CENTER_BLOCK),
            "center_factorization_multiplicity": center_multiplicity,
        },
        "star_scan": {
            "coprime_catalog_partners": len(coprime_partner_indices),
            "interior_projective_points_per_line": r54.PRIME - 1,
            "interior_projective_points_checked": len(coprime_partner_indices)
            * (r54.PRIME - 1),
            "positive_coprime_trisecant_lines": len(positive_lines),
            "positive_lines": positive_lines,
        },
        "inherited_positive_control": inherited_control,
        "checks": checks,
        "result": {
            "primitive_coprime_trisecant_found": bool(positive_lines),
            "maximum_factorization_star_exhausted": True,
            "full_catalog_pair_space_exhausted": False,
            "asymptotic_product_line_theorem": False,
            "asymptotic_interval_pencil_family": False,
            "fresh_target_action": False,
            "r10_queried_coefficients": False,
            "independent_rank": False,
            "factor_base_logs": False,
            "scalar_blind_descent": False,
            "shoup_bound_improvement": False,
            "breakthrough": False,
        },
        "limits": [
            "the exhaustive statement covers only the one star through the unique 280-factorization section",
            "other centers and the full pair space remain unclassified",
            "the frozen product catalog covers only three within-window degree-three factors",
            "no asymptotic theorem, target locator, R10 output, rank campaign, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
