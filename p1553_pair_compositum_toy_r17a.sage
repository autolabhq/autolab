import json

F = GF(193)
E = EllipticCurve(F, [2, 3])
P = E(1, 44)
d = 3
endpoint_scalars = [1, 2, 3]
endpoints = [j * P for j in endpoint_scalars]
endpoint_x = [F(A[0]) for A in endpoints]

R0 = PolynomialRing(F, "x")
x = R0.gen()
psi_num = prod(x - a for a in endpoint_x)
psi_den = x**3 + 5 * x + 7
derivative_num = psi_num.derivative() * psi_den - psi_num * psi_den.derivative()

assert P.order() == 103
assert endpoint_x == [F(1), F(184), F(62)]
assert gcd(psi_num, psi_den) == 1
assert all(psi_den(a) != 0 for a in endpoint_x)
assert gcd(derivative_num, derivative_num.derivative()) == 1
assert (psi_num - psi_den).degree() == 2

Rq = PolynomialRing(F, "q")
q = Rq.gen()
K = Rq.fraction_field()
S = PolynomialRing(K, "X")
X = S.gen()


def curve_rhs(value):
    return value**3 + 2 * value + 3


def lift_base_polynomial(poly):
    return sum(K(poly[i]) * X**i for i in range(poly.degree() + 1))


psi_num_X = lift_base_polynomial(psi_num)
psi_den_X = lift_base_polynomial(psi_den)


def branch_map(a):
    a = K(a)
    delta = K(q) - a
    c = -(K(q) + a)
    rhs_sum = curve_rhs(K(q)) + curve_rhs(a)
    addition_trace = 2 * rhs_sum / delta**2 + 2 * c
    quotient = K(q) ** 2 + a * K(q) + a**2 + 2
    addition_norm = quotient**2 / delta**2 + 2 * c * rhs_sum / delta**2 + c**2
    quadratic = X**2 - addition_trace * X + addition_norm
    p_rem = psi_num_X % quadratic
    r_rem = psi_den_X % quadratic
    p0, p1 = K(p_rem[0]), K(p_rem[1])
    r0, r1 = K(r_rem[0]), K(r_rem[1])
    denominator_norm = r1**2 * addition_norm + r1 * r0 * addition_trace + r0**2
    value_norm = (p1**2 * addition_norm + p1 * p0 * addition_trace + p0**2) / denominator_norm
    value_trace = (
        (p1 * r0 + p0 * r1) * addition_trace
        + 2 * p1 * r1 * addition_norm
        + 2 * p0 * r0
    ) / denominator_norm
    return K(value_trace), K(value_norm)


branch_maps = [branch_map(a) for a in endpoint_x[:2]]


def psi(value):
    return F(psi_num(value)) / F(psi_den(value))


direct_checks = []
for endpoint_index, A in enumerate(endpoints[:2]):
    trace_map, norm_map = branch_maps[endpoint_index]
    for scalar in [4, 5, 6]:
        Q = scalar * P
        plus = psi(F((A + Q)[0]))
        minus = psi(F((A - Q)[0]))
        q_value = F(Q[0])
        assert trace_map(q_value) == plus + minus
        assert norm_map(q_value) == plus * minus
        direct_checks.append([int(endpoint_scalars[endpoint_index]), int(scalar)])

B = PolynomialRing(F, names=("q0", "q1"))
q0, q1 = B.gens()


def lift_univariate(poly, variable):
    poly = Rq(poly)
    return sum(B(poly[i]) * variable**i for i in range(poly.degree() + 1))


def equality_polynomial(function):
    numerator = Rq(function.numerator())
    denominator = Rq(function.denominator())
    return (
        lift_univariate(numerator, q0) * lift_univariate(denominator, q1)
        - lift_univariate(numerator, q1) * lift_univariate(denominator, q0)
    )


equalities = []
for trace_map, norm_map in branch_maps:
    equalities.append(equality_polynomial(trace_map))
    equalities.append(equality_polynomial(norm_map))

diagonal = q0 - q1
assert all(poly % diagonal == 0 for poly in equalities)
common = equalities[0]
for poly in equalities[1:]:
    common = gcd(common, poly)

factorization = [(str(factor), int(multiplicity)) for factor, multiplicity in factor(common)]
off_diagonal = common
while off_diagonal % diagonal == 0:
    off_diagonal //= diagonal
off_diagonal_is_constant = off_diagonal.total_degree() == 0

report = {
    "schema": "crypto.autoresearch.exact_toy_report.v1",
    "report_id": "P1553-PAIR-COMPOSITUM-TOY-R17A-REPORT",
    "date": "2026-07-19",
    "classification": ["toy", "exact", "model-bound", "novelty-unverified"],
    "field": 193,
    "curve_order": int(E.order()),
    "subgroup_order": int(P.order()),
    "endpoint_x": [int(a) for a in endpoint_x],
    "psi_numerator": str(psi_num),
    "psi_denominator": str(psi_den),
    "derivative_numerator": str(derivative_num),
    "derivative_squarefree": True,
    "infinity_unramified": True,
    "fiber_uniform_min_support_lower_bound": 2,
    "non_galois_degree_three": True,
    "direct_formula_checks": direct_checks,
    "paired_endpoints": [1, 2],
    "equality_total_degrees": [int(poly.total_degree()) for poly in equalities],
    "common_gcd_total_degree": int(common.total_degree()),
    "common_gcd_factorization": factorization,
    "off_diagonal_is_constant": bool(off_diagonal_is_constant),
    "decision": "REJECT_TOY_CANDIDATE_BIRATIONAL_PAIR" if off_diagonal_is_constant else "RETAIN_OFF_DIAGONAL_FACTOR_FOR_REVIEW",
    "breakthrough": False,
    "shoup_bound_improvement": False,
    "scope": "One fixed degree-three exact toy; no asymptotic, finite-deck, target, R10, rank, log, or descent claim.",
}

print(json.dumps(report, indent=2, sort_keys=True))
