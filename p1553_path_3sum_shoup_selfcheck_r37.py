#!/usr/bin/env python3
"""Check the R37 path-3SUM generic-DLP reduction in a cyclic group."""

import argparse
import json


def ceil_fifth_root(n: int) -> int:
    b = max(1, int(round(n ** 0.2)))
    while b**5 < n:
        b += 1
    while b > 1 and (b - 1) ** 5 >= n:
        b -= 1
    return b


def run(order: int, secret: int) -> dict:
    if order <= 2:
        raise ValueError("order must exceed two")
    secret %= order
    b = ceil_fifth_root(order)
    m = b * b
    step3 = b**3
    step4 = b**4

    targets = [(r, r % order) for r in range(m)]
    z_rows = [
        (i, j, (-i * m + j) % order)
        for i in range(b)
        for j in range(b)
    ]
    v_rows = [
        (i, j, (secret - i * step4 - j * step3) % order)
        for i in range(b)
        for j in range(b)
    ]

    hits = []
    bad_recoveries = 0
    for r_label, target in targets:
        for vi, vj, v in v_rows:
            for zi, zj, z in z_rows:
                if (v + z) % order != target:
                    continue
                recovered = (
                    (vi * b + vj) * step3
                    + r_label
                    - (-zi * m + zj)
                ) % order
                if recovered != secret:
                    bad_recoveries += 1
                if len(hits) < 16:
                    hits.append(
                        {
                            "target_label": r_label,
                            "v_label": [vi, vj],
                            "z_label": [zi, zj],
                            "recovered": recovered,
                        }
                    )

    k = secret // step3
    residual = secret - k * step3
    witness_vi, witness_vj = divmod(k, b)
    witness_zi, witness_zj = divmod(residual, m)[0], 0
    witness_r = residual - witness_zi * m
    canonical_valid = (
        witness_vi < b
        and witness_vj < b
        and witness_zi < b
        and witness_r < m
        and (
            (secret - k * step3) + (-witness_zi * m + witness_zj)
        )
        % order
        == witness_r % order
    )

    return {
        "schema": "p1553.path_3sum_shoup_selfcheck.r37.v1",
        "order": order,
        "secret": secret,
        "B": b,
        "B5_covers_order": b**5 >= order,
        "B3_below_order": b**3 < order,
        "target_count": len(targets),
        "z_occurrence_count": len(z_rows),
        "z_distinct_count": len({row[2] for row in z_rows}),
        "v_occurrence_count": len(v_rows),
        "v_distinct_count": len({row[2] for row in v_rows}),
        "hit_count": sum(
            1
            for _, target in targets
            for _, _, v in v_rows
            for _, _, z in z_rows
            if (v + z) % order == target
        ),
        "bad_recoveries": bad_recoveries,
        "canonical_witness": {
            "target_label": witness_r,
            "v_label": [witness_vi, witness_vj],
            "z_label": [witness_zi, witness_zj],
            "valid": canonical_valid,
        },
        "first_hits": hits,
        "pass": canonical_valid and bad_recoveries == 0 and bool(hits),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=1009)
    parser.add_argument("--secret", type=int, default=777)
    args = parser.parse_args()
    print(json.dumps(run(args.order, args.secret), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
