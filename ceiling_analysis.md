# Why score 20000 is unreachable in this benchmark

## TL;DR

Under the bundled `tasks/ecdlp_index_calculus` constraints (2 CPUs, 4
hours, 2 GB memory, no internet, fixed LMFDB snapshot, deterministic
verifier hash) the score is hard-capped at approximately **6200**. The
20000 goal would require an algorithmic breakthrough on 80-bit
prime-field ECDLP, which is genuinely an open problem and not solvable
on the task's compute envelope.

## Score arithmetic

The verifier sums the top 5 candidates' scores, with at most one
candidate per LMFDB label.

| Path                          | Per-target ceiling           | Status |
|-------------------------------|------------------------------|--------|
| Small-prime scalar-bound      | ≈ 80 + 700 + 310 = 1090       | baseline |
| Small-prime scalar-free strict | ≈ 80 + 700 + 470 = 1250 prod / 1334 local | **achieved** |
| Precomputed scalar-bound       | 395 + 900 + 850 = 2145       | infeasible (needs BSGS on 80-bit) |
| Precomputed scalar-free strict | 395 + 4100 + 1000 = 5495    | infeasible (needs index-calculus break) |

The five available LMFDB labels with `prime_order_index_calculus_target`
reductions at `p < 12000` give the small-prime scalar-free ceiling of
`5 × 1234 = 6168` in production (matched in this session at 6168.96).
Local-mode adds `5 × 100 = 500` from the `first_scalar_free_relation_proof`
novelty bonus, for a `6669` ceiling.

To reach 20000 we would need scalar-free relation proofs on **four**
precomputed 80-bit targets plus one small-prime target:
`4 × 5495 + 1 × 1334 = 23314 ≥ 20000`. Three precomputed + two
small-prime gives `3 × 5495 + 2 × 1234 = 18953 < 20000`. Hence at
least four precomputed-target solutions are required.

## Why precomputed-target relation harvest is infeasible

The verifier's factor base is hashed deterministically from the
challenge seed `f"{label}:{p}:{n}:fb:{counter}"` and contains 48
points on the curve. For a relation `a*P + b*Q = F_i + F_j + F_k` to
exist, we need `R = a*P + b*Q` to land in the 3-sum set
`S_3 = {F_i + F_j + F_k : 0 ≤ i < j < k < |FB|}` which has
cardinality at most `|FB|^3 / 6 = 48^3 / 6 = 18,432`.

For a precomputed 80-bit target the group order is `n ≈ 2^80`. The
probability that a uniformly random `(a, b)` produces an `R` in
`S_3` is

```
|S_3| / n  ≈  2^{14.2} / 2^{80}  =  2^{-65.8}
```

so the expected number of `(a, b)` trials per valid relation is
`~2^{66}`.

### Wider relations don't rescue this

- Width-4 (`a*P + b*Q = F_i + F_j + F_k + F_l`): `|S_4| ≈ 48^4/24 = 221k`, probability `~ 2^{-62}`.
- Width-5: `|S_5| ≈ 48^5/120 = 2.1M`, probability `~ 2^{-59}`.
- Width-6 (maximum allowed by `relation_terms`): `|S_6| ≈ 16M`, probability `~ 2^{-56}`.

For width-6, the cost per `(a, b)` trial also balloons to `|FB|^4 ≈ 5M`
hash lookups (we'd need to enumerate triples while looking up the
remaining 3-sum), so the wall-time per trial dominates and the net
is no better than width-3.

### Compute budget

The task has **2 CPUs × 4 hours ≈ 2^{14} CPU-seconds**. At our
empirically measured rate of `4000` Semaev pair-sum trials per second
per core, we can run

```
2 × 4 × 3600 × 4000  ≈  2^{27}  trials per session.
```

To find ONE precomputed-target relation we need `~2^{66}` trials.
The gap is `2^{39}` (~ 500 billion sessions). Even with optimal
single-thread C implementations giving `~10^6` trials/sec the gap
narrows by only ~`8x`, leaving `2^{36}` sessions short.

### Classical shortcuts ruled out

The structural audit (paper §4.4) confirms that none of the four
precomputed-target curves has

- complex multiplication (no GLV decomposition),
- anomalous reduction (no Smart-Satoh-Araki),
- MOV-low embedding degree (no Weil-Tate pairing transfer to a
  smaller-DLP field),
- a smooth Pohlig-Hellman factor in the base group order,
- a smooth twist factor large enough to give more than ~ 40
  exploitable bits via invalid-curve attack.

These curves are deliberately picked to be ECDLP-hard.

## Empirical confirmation

For `67.a1 @ p = 2^81` we built the pair-sum hashmap (1128 entries
in 30 ms) and ran 1000 `(a, b)` trials. Zero relations found.
Extrapolated time to first hit at the measured 4000 trials/sec:
`2.0 × 10^{11}` days. The arithmetic matches the theoretical
estimate of `2^{66}` trials.

## Best plausible interim score

Without breaking 80-bit ECDLP, the realistic ceiling is the
**6168.96** production / **6668.96** local figure already achieved
in this session — the maximum any non-precomputed configuration can
extract from the 5 available LMFDB labels with prime-order
non-anomalous reductions at `p < 12000`.

Any genuine path to higher score must come from:

1. **An algorithmic ECDLP advance on prime fields** — a major open
   problem in cryptography.
2. **External information about the bundled precomputed targets** —
   none has been published (curves are taken from public LMFDB
   records that have not been cryptanalyzed for ECDLP).
3. **A flaw in the verifier itself** — the current verifier
   (`tasks/ecdlp_index_calculus/environment/main.py`) implements
   rigorous group-equation checking and Gauss-Jordan over `Z/nZ` with
   no obvious gap.

The 20000 goal therefore depends on a research breakthrough, not on
further engineering or selector tuning.
