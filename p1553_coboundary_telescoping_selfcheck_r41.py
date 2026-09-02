#!/usr/bin/env python3
"""Check raw-hit loss under telescoping of the R40 coboundary locator."""

import json

import p1553_orbit_transfer_divisor_selfcheck_r40 as r40


def translated_path_product_orders(transfer_orders, path_length):
    order = len(transfer_orders)
    ratio_orders = [
        transfer_orders[(index + 1) % order] - transfer_orders[index]
        for index in range(order)
    ]
    reduced_orders = [
        sum(
            ratio_orders[(target - offset) % order]
            for offset in range(path_length)
        )
        for target in range(order)
    ]
    endpoint_orders = [
        transfer_orders[(target + 1) % order]
        - transfer_orders[(target - path_length + 1) % order]
        for target in range(order)
    ]
    if reduced_orders != endpoint_orders:
        raise AssertionError("coboundary path product did not telescope")
    return ratio_orders, reduced_orders


def support_summary(transfer_orders, path_length):
    order = len(transfer_orders)
    ratio_orders, reduced_orders = translated_path_product_orders(
        transfer_orders, path_length
    )
    raw_zero_counts = [
        sum(
            ratio_orders[(target - offset) % order] > 0
            for offset in range(path_length)
        )
        for target in range(order)
    ]
    raw_pole_counts = [
        sum(
            ratio_orders[(target - offset) % order] < 0
            for offset in range(path_length)
        )
        for target in range(order)
    ]
    raw_zero_support = {
        target for target, count in enumerate(raw_zero_counts) if count
    }
    raw_pole_support = {
        target for target, count in enumerate(raw_pole_counts) if count
    }
    reduced_zero_support = {
        target for target, value in enumerate(reduced_orders) if value > 0
    }
    reduced_pole_support = {
        target for target, value in enumerate(reduced_orders) if value < 0
    }
    return {
        "path_length": path_length,
        "raw_zero_support": sorted(raw_zero_support),
        "raw_zero_support_size": len(raw_zero_support),
        "raw_pole_support": sorted(raw_pole_support),
        "raw_pole_support_size": len(raw_pole_support),
        "raw_zero_pole_overlap": sorted(raw_zero_support & raw_pole_support),
        "raw_zero_pole_overlap_size": len(raw_zero_support & raw_pole_support),
        "unambiguous_raw_zero_support": sorted(raw_zero_support - raw_pole_support),
        "unambiguous_raw_zero_support_size": len(raw_zero_support - raw_pole_support),
        "reduced_zero_support": sorted(reduced_zero_support),
        "reduced_zero_support_size": len(reduced_zero_support),
        "reduced_pole_support": sorted(reduced_pole_support),
        "reduced_pole_support_size": len(reduced_pole_support),
        "raw_zeros_lost_by_reduction": sorted(
            raw_zero_support - reduced_zero_support
        ),
        "raw_zeros_lost_by_reduction_size": len(
            raw_zero_support - reduced_zero_support
        ),
        "reduced_orders": {
            str(target): value
            for target, value in enumerate(reduced_orders)
            if value
        },
    }


def run():
    transfer = r40.run()
    if not transfer["pass"]:
        raise AssertionError("R40 positive coboundary control did not replay")

    order = transfer["subgroup_order"]
    transfer_orders = [0] * order
    for index, value in transfer["optimization"][
        "optimal_transfer_orders"
    ].items():
        transfer_orders[int(index)] = value

    summaries = [
        support_summary(transfer_orders, path_length)
        for path_length in range(1, 10)
    ]
    full = summaries[-1]

    expected = {
        "raw_zero_support_size": 17,
        "raw_pole_support_size": 28,
        "raw_zero_pole_overlap_size": 16,
        "unambiguous_raw_zero_support_size": 1,
        "reduced_zero_support_size": 9,
        "reduced_pole_support_size": 18,
        "raw_zeros_lost_by_reduction_size": 8,
    }
    checks = {
        "r40_replay": True,
        "all_path_products_telescope": True,
        "length_nine_statistics": all(
            full[key] == value for key, value in expected.items()
        ),
        "length_nine_raw_hit_interval": full["raw_zero_support"]
        == list(range(17)),
        "length_nine_lost_endpoints": full["raw_zeros_lost_by_reduction"]
        == [0, 1, 2, 3, 13, 14, 15, 16],
        "length_nine_reduced_zero_support": full["reduced_zero_support"]
        == list(range(4, 13)),
    }
    if not all(checks.values()):
        raise AssertionError("coboundary telescoping self-check failed")

    return {
        "schema": "p1553.coboundary_telescoping_selfcheck.r41.v1",
        "classification": [
            "toy",
            "exact",
            "model-bound",
            "novelty-unverified",
        ],
        "field_prime": transfer["field_prime"],
        "subgroup_order": order,
        "selected_interval_length": 9,
        "transfer_degree": transfer["optimization"]["optimal_transfer_degree"],
        "transfer_support_size": transfer["optimization"][
            "optimal_transfer_support_size"
        ],
        "identity": "product_(j=0)^(M-1) F(R-jq)=h(R+q)/h(R-(M-1)q)",
        "path_summaries": summaries,
        "length_nine_result": full,
        "asymptotic_path_family": {
            "selected_path_length": "L=B^2",
            "other_pair_support": "V=union of K q-paths",
            "path_lengths": "M_1+...+M_K=|V|<=B^2",
            "target_support_upper_bound": "K*L+|V|-K",
            "explicit_endpoint_ratio_work_lower_bound": "K",
            "independent_target_expected_work_lower_bound": "N/(2*L)=Omega(B^3)",
            "assumptions_for_bound": [
                "pole translates are disjoint from hit translates",
                "each h endpoint evaluation and path result costs at least one operation",
                "path results are processed explicitly",
            ],
            "remaining_exception": "one global cancellation-free marked across-path locator",
        },
        "checks": checks,
        "limits": [
            "the cancellation statistics use the exact R40 toy optimum",
            "a different pole placement can avoid zero-pole overlap at greater transfer cost",
            "the path-level expected-work theorem is scoped to explicit path results",
            "an unrestricted cancellation-free across-path arithmetic circuit is not excluded",
            "no degree-nine pencil, R10 output, rank, logs, or descent is constructed",
        ],
        "pass": True,
    }


def main():
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
