import json
from collections import defaultdict


F = GF(193)
E = EllipticCurve(F, [2, 3])
P = E(1, 44)
R = PolynomialRing(F, "x")
x = R.gen()

ENDPOINT_SCALARS = list(range(1, 9))
TRAINING_FIFTH_SCALARS = list(range(9, 29))
HOLDOUT_FIFTH_SCALARS = list(range(29, 49))
DENOMINATOR_COEFFICIENTS = [147, 140, 100, 41, 188, 123, 182, 46, 1]

assert P.order() == 103

endpoints = [a * P for a in ENDPOINT_SCALARS]
endpoint_x = [F(A[0]) for A in endpoints]
psi_num = prod(x - value for value in endpoint_x)
psi_den = sum(F(value) * x**i for i, value in enumerate(DENOMINATOR_COEFFICIENTS))


def point_x(point):
    assert not point.is_zero()
    return F(point[0])


def evaluate(fifth_scalars):
    fibers = defaultdict(list)
    pole_sources = []
    for endpoint_scalar, A in zip(ENDPOINT_SCALARS, endpoints):
        for fifth_scalar in fifth_scalars:
            Q = fifth_scalar * P
            plus_x = point_x(A + Q)
            minus_x = point_x(A - Q)
            if psi_den(plus_x) == 0 or psi_den(minus_x) == 0:
                pole_sources.append([endpoint_scalar, fifth_scalar])
                continue
            plus = psi_num(plus_x) / psi_den(plus_x)
            minus = psi_num(minus_x) / psi_den(minus_x)
            key = (int(plus + minus), int(plus * minus))
            fibers[key].append([endpoint_scalar, fifth_scalar])

    source_count = len(ENDPOINT_SCALARS) * len(fifth_scalars)
    collision_fibers = [
        (key, values) for key, values in fibers.items() if len(values) > 1
    ]
    zero_collision_fibers = [
        (key, values) for key, values in collision_fibers if key[1] == 0
    ]
    return {
        "source_count": source_count,
        "evaluated_source_count": source_count - len(pole_sources),
        "pole_sources": pole_sources,
        "distinct_complete_keys": len(fibers),
        "compression_ratio": (
            (source_count - len(pole_sources)) / len(fibers) if fibers else 0
        ),
        "collision_excess": sum(len(values) - 1 for _, values in collision_fibers),
        "zero_branch_collision_excess": sum(
            len(values) - 1 for _, values in zero_collision_fibers
        ),
        "max_fiber": max((len(values) for values in fibers.values()), default=0),
        "collision_fiber_count": len(collision_fibers),
    }


training = evaluate(TRAINING_FIFTH_SCALARS)
holdout = evaluate(HOLDOUT_FIFTH_SCALARS)

decision = (
    "RETAIN_HOLDOUT_COMPRESSION_FOR_REVIEW"
    if not holdout["pole_sources"]
    and holdout["distinct_complete_keys"] <= 106
    and holdout["compression_ratio"] >= 1.5
    else "REJECT_HOLDOUT_TRANSFER_BELOW_1P5_OR_POLE"
)

report = {
    "schema": "crypto.autoresearch.exact_toy_report.v1",
    "report_id": "P1553-LIST-ONLY-MASS-COLLISION-HOLDOUT-R22D-REPORT",
    "date": "2026-07-19",
    "classification": [
        "toy",
        "exact",
        "fresh-holdout",
        "model-bound",
        "novelty-unverified",
    ],
    "field": 193,
    "subgroup_order": int(P.order()),
    "degree": 8,
    "endpoint_scalars": ENDPOINT_SCALARS,
    "training_fifth_scalars": TRAINING_FIFTH_SCALARS,
    "holdout_fifth_scalars": HOLDOUT_FIFTH_SCALARS,
    "psi_numerator": str(psi_num),
    "psi_denominator": str(psi_den),
    "training": training,
    "holdout": holdout,
    "decision": decision,
    "breakthrough": False,
    "shoup_bound_improvement": False,
    "scope": (
        "One fixed finite-field holdout. No asymptotic containment, target, "
        "R10, rank, factor-log, descent, lower-bound, or breakthrough claim."
    ),
}

print(json.dumps(report, indent=2, sort_keys=True, default=lambda value: int(value)))
