# P1553 S6 centered-carry rank-minor probe R72

## Classification

- Owner: existing P1553 R7-R10 and IDEA-049/IDEA-198 frontier; no P1554.
- Evidence: exact modular minor computation on four large prime-order curves.
- Status: `REJECT_TWO_NATURAL_S6_CENTERED_CARRY_LIFTS`.
- Labels: `exact-finite`, `representation-bound`, `scalar-blind-source`,
  `novelty-unverified`.
- Cryptanalytic result: no projector-trace contraction, relation campaign,
  factor-log solve, scalar-blind descent, Shoup-bound improvement, or ECDLP
  breakthrough.

R71 found full mode rank in the canonical `S4`, `k=1` centered carry but left
open an arity-six-specific cancellation. R72 tests that exception directly on
the five-label predicate

```text
F_R(x_1,...,x_5)
  = Res_z(
      S_4(x_1,x_2,x_3,z),
      S_4(x_4,x_5,x(R),z)
    )
  = S_6(x_1,...,x_5,x(R)).
```

For each frozen integer coefficient lift it computes

```text
R_R = ctr_p(F_R),
K_R = (F_R-R_R)/p.
```

The decisive deck size is `B=18`. Fixed-target `S6` has degree 16 in each
variable, so every raw legal-grid mode flattening has rational rank at most
17. A carry rank of 18 therefore detects centering-induced expansion beyond
the raw degree ceiling.

Every raw mode has exact rank 17. Every centered remainder and every carry mode
has rank 18 under both auxiliary primes, for both natural lift schedules, on
all four curves and both target types. The hoped-for arity-six low-carry-rank
exception is absent from these two lifts.

This does not contradict R9. The Fermat zero projector can have rank zero or
one on empty or singleton fibers, and carry rank is not lift invariant. R72
closes only the two exact resultant lifts it measures.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R71 S4 centered-carry report | `6ee62d61f6c567a5a65947b6b47026182bd961565dde3c1aee540cf3cbe5b72f` |
| R71 S4 centered-carry gate | `f194ed996a24659e2b11d762dd471c0ff75ab98d6e64a9df8b782c676acca2f7` |
| R9 projector-trace router gate | `400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81` |
| R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |

Curve parameters are the SEC 2 secp256k1 parameters and the NIST P-256,
P-384, and P-521 parameters:

- SEC 2, version 2: <https://www.secg.org/sec2-v2.pdf>
- NIST SP 800-186: <https://csrc.nist.gov/pubs/sp/800/186/final>

## Frozen curves

| Family | field bits | subgroup bits | cofactor |
|---|---:|---:|---:|
| `secp256k1` | 256 | 256 | 1 |
| `nist_p256` | 256 | 256 | 1 |
| `nist_p384` | 384 | 384 | 1 |
| `nist_p521` | 521 | 521 | 1 |

For every family, R72 verifies:

- the field modulus and subgroup order pass strong probable-prime tests;
- the discriminant is nonzero;
- the standard generator lies on the curve;
- multiplication by the stated subgroup order gives the identity;
- `32 * 18^4 < p`, so the R9 no-wrap threshold holds at the probe size; and
- `18^5 < q`.

The probable-prime checks replay the standardized parameters; they are not new
primality certificates.

## Scalar-blind source

Five domain-separated SHA-256 x-coordinate streams produce five public decks
of 18 points per curve. Canonical square roots are selected and duplicate
x-classes are removed within each color. No scalar label or discrete logarithm
is consumed.

Each family uses:

1. one `blind_hash_target` from an independent x-coordinate stream; and
2. one `forced_positive_target`, the negative sum of five public deck points.

On each size-three prefix, R72 exhausts all `3^5=243` label tuples and all 32
normalized source-sign choices. Across eight targets this gives 1,944 exact
predicate tuples. Every blind prefix has zero roots, every forced prefix has
exactly one root, and

```text
S_6=0 mod p  iff  one signed six-point group relation exists.
```

There are zero predicate or sampled-symmetry mismatches. Every forced witness
verifies by complete group addition.

## Integer lifts

R72 applies the same recursive resultant formula under two coefficient
schedules:

1. `least_nonnegative_coefficients`, using `A,B in [0,p-1]`; and
2. `centered_coefficients`, replacing `A,B` by their centered integer
   representatives.

Both schedules define the same predicate modulo `p` but generally different
integer raw values and carries. secp256k1 has identical schedules because
`A=0` and `B=7`; the three NIST curves exercise genuinely different
coefficient lifts.

For an auxiliary prime `ell`, the exact carry is evaluated without large
integer materialization through

```text
K_R mod ell
  = (F_R mod ell - ctr_p(F_R)) * p^(-1) mod ell.
```

This follows from the exact integer identity `F_R-ctr_p(F_R)=p K_R`; it is not
an approximate quotient.

## S6 compiler replay

R72 first constructs the quartic

```text
S_4(x_1,x_2,x_3,z)
```

coefficientwise from the exact quadratic-resultant formula for `S3`. The unit
test evaluates that quartic and matches R71's direct integer `S4` resultant.
Two quartics are then eliminated with the fixed-degree `8 x 8` Sylvester
determinant modulo each field or auxiliary prime.

The fixed formal degree is retained under specialization. This evaluates the
specialized Semaev polynomial, including its proper-subsum and leading-term
strata, rather than silently replacing it with a lower-degree resultant.

## Rank-minor certificate

For each target and each of five tensor modes, columns of the conceptual
`18 x 18^4` flattening are visited in a frozen SHA-256-derived permutation.
An incremental exact column basis is maintained separately for:

- the centered remainder under auxiliary primes `1000003` and `1000033`;
- the raw tensor under both primes and both lift schedules; and
- the centered carry under both primes and both lift schedules.

Every profile reaches its target rank after exactly 18 columns. Independent
column numbers are retained in the JSON report.

The aggregate is:

```text
curve families                         4
target instances                       8
tensor-mode instances                 40
carry profiles: modes*lifts*primes   160
full-rank carry profiles             160
predicate tuples                    1944
unique S6 tuple evaluations         12961
```

For every mode:

```text
raw lift rank              17
centered remainder rank    18
centered carry rank        18
```

The rank-17 raw lower bound holds under both primes. The degree-16 theorem
gives a matching rational upper bound, so raw rational mode rank is exactly
17.

A rank-18 minor modulo either auxiliary prime is a nonzero integer minor.
Therefore each named carry has rational matrix rank at least 18 and rational
CP rank at least 18 on the frozen legal grid. Both primes independently give
the same certificate.

## Consequence

R72 rejects the proposed tiny separated carry object for:

- the exact split `S4|S4` resultant lift with least-nonnegative curve
  coefficients; and
- the same resultant lift with centered curve coefficients.

The result is stronger than R71 only in this scoped sense: it tests the actual
five-label `S6` predicate at a deck size beyond its raw mode-degree ceiling.
It remains finite. It does not prove rank growth for every increasing deck
family, every integer lift, higher powers modulo `p^k`, or every arithmetic
circuit.

The surviving branch-complete interface is R9's exact trace count:

```text
C_(R,I) = Tr(1-F_R^(p-1)) = |Z_(R,I)|.
```

R9 already proves that `O(log B)` exact restricted counts return one source.
R10 identifies the missing operation as an exact non-CP balanced contraction
of that scalar count without the `B^3` triple table. Carry tensors no longer
deserve priority unless a new lift comes with an explicit cancellation theorem
and direct source interface.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_s6_centered_carry_rank_minor_probe_r72.py` | `e074f0f8894d7a194b4cdc25a6ef707c326d65751b3b3df0dc22b891ca8c451f` |
| `p1553_s6_centered_carry_rank_minor_probe_report_r72.json` | `7e63b52fc7667be14aadc1db3aeeb22b43876ff2c62e3dce87b312c056e85e43` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_s6_centered_carry_rank_minor_probe_r72.py` | `97cc80978468e593e9e7551362a742b5f00daf5c9d806063d84a85e80a5bccab` |

Targeted compiler tests:

```text
Ran 3 tests in 0.092s
OK
```

Full deterministic rank-minor replay:

```text
families=4
carry_full=160/160
lane_admitted=False
```

## Disposition

```text
REJECT_TWO_NATURAL_S6_CENTERED_CARRY_LIFTS__SECP256K1_P256_P384_P521__FIVE_SCALAR_BLIND_B18_DECKS__BLIND_AND_FORCED_POSITIVE_TARGETS__1944_EXACT_S6_SIGN_REPLAYS__ZERO_PREDICATE_MISMATCHES__R9_NO_WRAP_THRESHOLD_HOLDS__LEAST_NONNEGATIVE_AND_CENTERED_COEFFICIENT_LIFTS__RAW_MODE_RANK_EXACTLY_17_FROM_MODULAR_MINOR_AND_DEGREE16_BOUND__CENTERED_REMAINDER_MODE_RANK18__CENTERED_CARRY_MODE_RANK18__ALL_160_LIFT_TARGET_MODE_PRIME_PROFILES_FULL__RATIONAL_CP_RANK_AT_LEAST18_PER_NAMED_CARRY__FINITE_LIFT_SCOPED_ONLY__PROJECTOR_TRACE_AND_NON_CP_PULLBACK_OPEN__NO_TRACE_COUNT_CONSTRUCTOR__NO_FACTOR_LOG_SOLVE__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Close explicit centered-carry CP construction for the two natural `S6`
resultant lifts. Return to the R9/R10 surviving object: define one exact
non-CP balanced contraction computing

```text
C_(R,I)
  = sum_(a_3,a_4,a_5) h_I(V_R(a_3,a_4,a_5))
```

on the full box and every queried dyadic child without materializing the
`B^3` triple table. Before another run, require a frozen circuit grammar that
handles zero signatures, exact integer multiplicities, one occurrence
backpointer, blind and positive targets, and the rank-two sparse
multiplicative-convolution control. Reject any proposal that merely renames a
carry, dense character table, source incidence tensor, or uncharged selector.
