import json
import random
import statistics
from collections import Counter, defaultdict


F = GF(193)
E = EllipticCurve(F, [2, 3])
P = E(1, 44)
R = PolynomialRing(F, "x")
x = R.gen()

DEGREE = 8
ENDPOINT_SCALARS = list(range(1, 9))
FIFTH_SCALARS = list(range(9, 29))
CANDIDATE_COUNT = 4096
RANDOM_SEED = 155322

assert P.order() == 103

endpoints = [a * P for a in ENDPOINT_SCALARS]
fifth_points = [q * P for q in FIFTH_SCALARS]
endpoint_x = [F(A[0]) for A in endpoints]
psi_num = prod(x - value for value in endpoint_x)


def point_x(point):
    assert not point.is_zero()
    return F(point[0])


sources = []
evaluated_x = set(endpoint_x)
for endpoint_scalar, A in zip(ENDPOINT_SCALARS, endpoints):
    for fifth_scalar, Q in zip(FIFTH_SCALARS, fifth_points):
        plus_x = point_x(A + Q)
        minus_x = point_x(A - Q)
        evaluated_x.add(plus_x)
        evaluated_x.add(minus_x)
        sources.append(
            {
                "endpoint_scalar": endpoint_scalar,
                "fifth_scalar": fifth_scalar,
                "plus_x": plus_x,
                "minus_x": minus_x,
            }
        )


def complete_key(denominator, source):
    plus = psi_num(source["plus_x"]) / denominator(source["plus_x"])
    minus = psi_num(source["minus_x"]) / denominator(source["minus_x"])
    return int(plus + minus), int(plus * minus)


def evaluate_candidate(denominator):
    fibers = defaultdict(list)
    for source in sources:
        fibers[complete_key(denominator, source)].append(
            [source["endpoint_scalar"], source["fifth_scalar"]]
        )
    sizes = sorted((len(values) for values in fibers.values()), reverse=True)
    return {
        "distinct_keys": len(fibers),
        "max_fiber": sizes[0],
        "sum_squared_fibers": sum(size * size for size in sizes),
        "fiber_sizes": sizes,
        "fibers": fibers,
    }


def geometric_fiber_min_support(denominator):
    minimum = DEGREE + 1
    witnesses = []
    for value in F:
        fiber_poly = psi_num - value * denominator
        finite_support = fiber_poly.squarefree_part().degree() if fiber_poly else 0
        infinity_support = 1 if fiber_poly.degree() < DEGREE else 0
        support = finite_support + infinity_support
        if support < minimum:
            minimum = support
            witnesses = [int(value)]
        elif support == minimum:
            witnesses.append(int(value))
    pole_support = denominator.squarefree_part().degree()
    if pole_support < minimum:
        minimum = pole_support
        witnesses = ["infinity"]
    elif pole_support == minimum:
        witnesses.append("infinity")
    return int(minimum), witnesses[:16]


identity_fibers = defaultdict(list)
for source in sources:
    plus = source["plus_x"]
    minus = source["minus_x"]
    key = (int(plus + minus), int(plus * minus))
    identity_fibers[key].append(
        [source["endpoint_scalar"], source["fifth_scalar"]]
    )

rng = random.Random(int(RANDOM_SEED))
best = None
distinct_counts = []
valid_candidates = 0
attempts = 0

while valid_candidates < CANDIDATE_COUNT:
    attempts += 1
    coefficients = [F(rng.randrange(193)) for _ in range(DEGREE)]
    denominator = x**DEGREE + sum(coefficients[i] * x**i for i in range(DEGREE))
    if gcd(psi_num, denominator) != 1:
        continue
    if any(denominator(value) == 0 for value in evaluated_x):
        continue
    derivative_num = psi_num.derivative() * denominator - psi_num * denominator.derivative()
    if derivative_num == 0:
        continue

    result = evaluate_candidate(denominator)
    valid_candidates += 1
    distinct_counts.append(result["distinct_keys"])
    score = (
        result["distinct_keys"],
        -result["sum_squared_fibers"],
        -result["max_fiber"],
    )
    if best is None or score < best["score"]:
        best = {
            "score": score,
            "denominator": denominator,
            "coefficients_low_to_high": [int(value) for value in coefficients] + [1],
            "result": result,
        }

assert best is not None

source_count = len(sources)
best_result = best["result"]
compression_ratio = source_count / best_result["distinct_keys"]
min_support, min_support_values = geometric_fiber_min_support(best["denominator"])

collision_fibers = []
for key, values in sorted(
    best_result["fibers"].items(), key=lambda item: (-len(item[1]), item[0])
):
    if len(values) > 1:
        collision_fibers.append(
            {
                "key": list(key),
                "multiplicity": len(values),
                "sources": values,
            }
        )


def expected_sampled_mass_coverage(sample_count):
    total = source_count
    uncovered = sum(
        (size / total) * ((1 - size / total) ** sample_count)
        for size in best_result["fiber_sizes"]
    )
    return 1 - uncovered


universe = 193**2
random_key_expected_distinct = universe * (1 - (1 - 1 / universe) ** source_count)

sorted_counts = sorted(distinct_counts)


def percentile(numerator, denominator):
    index = (len(sorted_counts) - 1) * numerator // denominator
    return int(sorted_counts[index])


decision = (
    "RETAIN_LIST_ONLY_COMPRESSION_FOR_REVIEW"
    if best_result["distinct_keys"] <= 80
    else "REJECT_NO_TWOFOLD_LIST_ONLY_COMPRESSION"
)

report = {
    "schema": "crypto.autoresearch.exact_toy_report.v1",
    "report_id": "P1553-LIST-ONLY-MASS-COLLISION-TOY-R22-REPORT",
    "date": "2026-07-19",
    "classification": ["toy", "exact", "model-bound", "novelty-unverified"],
    "field": 193,
    "curve_order": int(E.order()),
    "subgroup_order": int(P.order()),
    "degree": DEGREE,
    "endpoint_scalars": ENDPOINT_SCALARS,
    "fifth_scalars": FIFTH_SCALARS,
    "endpoint_block_size": len(ENDPOINT_SCALARS),
    "source_count": source_count,
    "candidate_count": CANDIDATE_COUNT,
    "candidate_attempts": attempts,
    "random_seed": RANDOM_SEED,
    "psi_numerator": str(psi_num),
    "best_denominator": str(best["denominator"]),
    "best_denominator_coefficients_low_to_high": best["coefficients_low_to_high"],
    "best_distinct_complete_keys": best_result["distinct_keys"],
    "best_compression_ratio": compression_ratio,
    "best_collision_excess": source_count - best_result["distinct_keys"],
    "best_max_fiber": best_result["max_fiber"],
    "best_sum_squared_fibers": best_result["sum_squared_fibers"],
    "best_geometric_fiber_min_support": min_support,
    "best_geometric_fiber_min_support_values": min_support_values,
    "best_collision_fibers": collision_fibers,
    "sampled_mass_coverage": {
        "R_source_count_over_4": expected_sampled_mass_coverage(source_count // 4),
        "R_source_count_over_2": expected_sampled_mass_coverage(source_count // 2),
        "R_distinct_key_count": expected_sampled_mass_coverage(
            best_result["distinct_keys"]
        ),
    },
    "candidate_distinct_key_summary": {
        "minimum": int(sorted_counts[0]),
        "p10": percentile(1, 10),
        "median": int(statistics.median(sorted_counts)),
        "p90": percentile(9, 10),
        "maximum": int(sorted_counts[-1]),
    },
    "controls": {
        "identity_psi_distinct_complete_keys": len(identity_fibers),
        "identity_psi_max_fiber": max(len(values) for values in identity_fibers.values()),
        "random_key_universe": universe,
        "random_key_expected_distinct": random_key_expected_distinct,
    },
    "decision": decision,
    "breakthrough": False,
    "shoup_bound_improvement": False,
    "scope": (
        "Finite regular-chart search only. No asymptotic containment, target, "
        "R10, rank, factor-log, descent, lower-bound, or breakthrough claim."
    ),
}

print(json.dumps(report, indent=2, sort_keys=True, default=lambda value: int(value)))
