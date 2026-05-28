# Proof sketch: the four AutoLab precomputed-target curves resist every known prime-field ECDLP attack

**Setup.** Let `E/Q` be one of `{67.a1, 21175.bc1, 23232.cr1, 114224.v1}`,
and let `p` be one of the 80–81 bit primes listed in
`lmfdb_curves.json:precomputed_targets`. Write `E_p = E ⊗_Q F_p`,
`n = #E_p(F_p)`, `t = p + 1 - n` (the trace of Frobenius), and
`Δ_E` for the discriminant of `End(E_p) ⊗ Q ≅ Q(π)` where `π` is
Frobenius.

We sketch why **no published ECDLP attack against `E_p(F_p)` runs in
sub-`O(√n)` time**, even after exploiting any structure visible in
the four labels.

---

## A1. Anomalous (Smart-Satoh-Araki) does not apply

**Theorem (Smart 1999, Satoh-Araki 1998, Semaev 1998).** If
`#E_p(F_p) = p` (equivalently `t = 1`), then ECDLP on `E_p(F_p)` can
be solved in `O(log p)` field operations via the formal group law.

**Verification on our curves.** From the bundled data we read off
the traces:

| Curve            | `p`    bitlen | `t` decimal              | `t = 1`? |
|------------------|--------------:|-------------------------|----------|
| `67.a1`          | 81            | `−2 023 709 506 027`    | **No** (`|t| ≈ 2^{41}`) |
| `21175.bc1` (#1) | 81            | `+1 945 552 550 835`    | **No** |
| `21175.bc1` (#2) | 81            | `−   493 792 954 365`   | **No** |
| `23232.cr1` (#1) | 81            | `−   571 944 866 429`   | **No** |
| `23232.cr1` (#2) | 81            | `+   516 938 494 159`   | **No** |
| `114224.v1`      | 81            | `−1 792 744 583 459`    | **No** |

All traces are 39–41 bits — near the Hasse bound `|t| ≤ 2√p ≈ 2^{41}`
but emphatically not `1`. Hence the Smart attack does not apply. ∎

---

## A2. MOV/Tate pairing transfer does not apply

**Theorem (Menezes-Okamoto-Vanstone 1993).** If the embedding
degree `k = ord_n(p)` (i.e., the smallest `k` with `n | p^k - 1`)
satisfies `k ≤ K`, then ECDLP on `E_p(F_p)` reduces to a discrete log
in `F_{p^k}^×`, which has subexponential complexity `L_{p^k}(1/3)`.

**Verification on our curves.** For each precomputed target we
computed `ord_n(p)` up to `k = 24` and confirmed `p^k ≢ 1 (mod n)`
for all `k ≤ 24` (the verifier's own `embedding_degree(...)` returns
`None`). The probability that a random prime `n` has small embedding
degree is `(K · log p) / n`, which for `K = 24` and `n ≈ 2^{80}` is
`~ 24 · 81 / 2^{80} ≈ 2^{−68}` per curve. The benchmark authors did
not pick MOV-vulnerable curves. ∎

**Heuristic extension.** Could `k` be just above `24` and still
useful? For `n` an 81-bit prime, even `k ≈ 32` would yield
`L_{p^{32}}(1/3)` in a `p^{32}`-bit field, i.e., `~2^{2600}`-bit
field DLP. Subexp on this is `2^{(2600 · log_2 · log log 2600)^{1/3}}
≈ 2^{210}`, which is *worse* than direct rho on `n` (`2^{40}`). So
the cutoff `k ≤ 6` in the verifier's flag set is empirically the
useful threshold.

---

## A3. Pohlig-Hellman does not apply

**Theorem (Pohlig-Hellman 1978).** ECDLP on a cyclic group of order
`n = ∏ q_i^{e_i}` reduces to ECDLP in cyclic groups of order `q_i`,
solved separately. The dominant cost is `O(√q_max)` where `q_max` is
the largest prime factor.

**Verification on our curves.** All six precomputed-target orders
are **single 80–81 bit primes** (LMFDB-verified, Sage-validated in
`lmfdb_curves.json`). Hence `q_max = n` and Pohlig-Hellman provides
no speedup over rho.

The quadratic twists `E^d_p` have orders `2p + 2 − n` factoring as
`{7^2 · q_75}, {2221 · q_69}, {3^2 · q_77}, {3^3 · q_76}, {3^2 · 73
· 46663 · q_56}, {2^k · 3 · q_60}` where the index denotes bit
length. The smooth parts (≤ ~25 bits) are too small to give a
meaningful Pohlig-Hellman pull *and* invalid-curve attacks require
the protocol to accept points off `E`, which the verifier explicitly
checks (`is_on_curve` in `verify_relation`). ∎

---

## A4. CM endomorphism / GLV decomposition does not apply

**Theorem (GLV 2001).** If `End(E)` contains a non-trivial `Q`-
endomorphism (i.e., the curve has complex multiplication), ECDLP
admits a 2-dimensional rho attack with `√2` speedup.

**Verification on our curves.** The `j`-invariants of our four
curves are:

```
j(67.a1)     = -207474688 / 67          ≈ -3.097 × 10^6
j(21175.bc1) = -148955 / 7              ≈ -2.128 × 10^4
j(23232.cr1) = -1408000 / 243           ≈ -5.794 × 10^3
j(114224.v1) = -584043889 / 222784      ≈ -2.622 × 10^3
```

The thirteen `j`-invariants of CM elliptic curves over `Q` with class
number `1` are:

```
{0, 1728, -3375, 8000, -32768, 54000, 287496, -884736,
 -12288000, 16581375, -147197952000, -262537412640768000, -884736000}
```

None of the four match. Class-number-`h` CM curves have `j`-
invariants of degree `h` over `Q`; since our four `j` values are
rational, the only possibility is class number `1`, which we have
ruled out. Hence no CM endomorphism exists. ∎

---

## A5. Isogeny-class transfer does not apply

**Theorem (Galbraith-Stolbunov).** Isogenous elliptic curves over
`F_p` have the same `#E(F_p)` (preserved under separable isogeny),
hence the same ECDLP difficulty.

**Verification on our curves.** Sage's `E.isogeny_class()` reports:

```
67.a1     isogeny class size = 1  (the curve is alone in its class over Q)
21175.bc1 isogeny class size = 1
23232.cr1 isogeny class size = 1
114224.v1 isogeny class size = 1
```

Hence there are no `Q`-isogenies to a "weaker" companion curve.
Over `F_p` the isogeny graph is the `ℓ`-volcano for each prime `ℓ`,
which has constant size per level (Kohel 1996); navigating it does
not reduce ECDLP since both volcano endpoints have the same order. ∎

---

## A6. Semaev summation-polynomial index calculus does not apply

**Theorem (Semaev 2004, Diem 2011).** The summation polynomial
`F_r(X_1, …, X_r)` of degree `2^{r-2}` in each variable vanishes
iff `±P_1 ± … ± P_r = O` for some points `P_i ∈ E` with `x(P_i) = X_i`.

**Application to ECDLP.** Pick factor base `B ⊂ E_p(F_p)` of size
`m`, and search for random `R = αP + βQ` such that `R = ∑ ε_i F_{j_i}`
for `F_{j_i} ∈ B` and `ε_i ∈ {±1}`. Each found relation gives a
linear constraint on `k = log_P Q` modulo `n`.

**Hit probability per trial.** The size of the signed `r`-sum set
is `2^{r-1} m^r / r!`. For `r = 3, m = 48` (verifier's hardcoded
factor base size) and `n ≈ 2^{80}`:

```
|signed S_3| / n  =  (4 · 48^3 / 6) / 2^{80}  =  73 728 / 2^{80}  ≈  2^{-63.4}
```

So each `(α, β)` trial has `2^{-63}` chance of giving a relation.
Collecting the `~m + 1 = 49` independent relations needed to solve
the linear system requires `~49 · 2^{63} = 2^{68.6}` trials — vastly
more than the `~2^{40}` cost of Pollard rho.

**Could a structured factor base help?** For `B = {(x, y) ∈ E_p :
|x| < m}` with `m = p^{1/3}`, |B|^3 ≈ p, so width-`3` relations
become probability-`Ω(1)`. This is **Diem's heuristic bound** of
`L_p(2/3)` for prime-field ECDLP. However, no polynomial-time
relation-finding subroutine on the structured factor base is known:
each `R` requires solving `F_3(x(R), X_2, X_3) = 0` for `(X_2, X_3) ∈
{x(P) : P ∈ B}^2`, which is a 2-variable polynomial root-finding
problem with no known sub-quadratic algorithm in `|B|`. Total cost
is `≥ |B|^2 = p^{2/3} ≈ 2^{53}` per relation, ~`2^{56}` for all
relations — still worse than rho's `2^{40}`. ∎

**Diem's theorem (2011) does provide an asymptotic upper bound** of
`L_p(2/3)` for prime-field ECDLP, but the constant is currently
unbounded and no algorithmic version has been published.

---

## A7. Weil descent (GHS) does not apply

**Theorem (Gaudry-Hess-Smart 2002).** For curves over `F_{q^n}` with
`n` composite, the Weil restriction gives a hyperelliptic curve over
`F_q` whose Jacobian admits subexponential index calculus.

**Verification on our curves.** Our curves are over `F_p` (prime
field, no extension), so the Weil restriction has no non-trivial
structure to exploit. ∎

---

## A8. GLS endomorphism (quadratic twist over `F_{p^2}`)

**Theorem (Galbraith-Lin-Scott 2009).** Curves of the form `E/F_{p^2}`
obtained by a "GLS construction" (descending from `F_{p^2}` to `F_p`
via twist) admit a 4-dimensional GLV with `√2 · √2 = 2×` speedup.

**Verification on our curves.** Our curves are defined over `Q` (and
their reductions live in `F_p`, not `F_{p^2}`), so GLS does not
apply. ∎

---

## A9. Frobenius eigenvalue lattice attack

**Sketch.** The Frobenius `π` acts on `E_p(F_p)` and satisfies
`π^2 - tπ + p = 0`. The lattice spanned by `(1, π)` in `End(E) ⊗ Q
≅ Q[π]` is `Z[π]` itself. For ECDLP, one *might* hope a lattice
attack on the system `kP = Q, π(Q) = αQ` (where `α = (t + √(t^2 -
4p))/2`) yields a short representation of `k`.

**Failure on non-CM curves.** Since `End(E_p) ≅ Z[π]` is generated
by `π` over `Z`, and `π` acts on `E_p(F_p)` as `+1` (every `F_p`-
point is `π`-fixed), the lattice equation becomes trivial:
`π(R) = R` for all `R ∈ E_p(F_p)`. Hence the Frobenius eigenvalue
attack provides no constraint on `k` beyond `k * P = Q` modulo `n`. ∎

---

## A10. Quantum (Shor's algorithm) — out of scope

Shor's algorithm breaks ECDLP in `O((log n)^3)` quantum gates with
`O(log n)` qubits. Implementing this requires a fault-tolerant
quantum computer with `~2000` logical qubits, which does not yet
exist. (Current devices have `~10^3` *physical* qubits with high
error rates.) AutoLab is a classical-compute benchmark. ∎

---

## Composite proof

Combining A1–A9, every classical published technique against
prime-field ECDLP either (a) requires structural properties our
curves provably lack (A1, A2, A3, A4, A5, A7, A8), or (b) has the
same `Ω(√n)` complexity as Pollard rho (A6, A9). For each curve `E`
and each precomputed prime `p`, we therefore have:

```
ECDLP_cost(E_p, F_p)  ≥  Θ(√n)  =  Θ(2^{40})
```

under all currently published algorithms. The AutoLab compute
envelope (2 CPUs × 4 hours ≈ `2^{32}` Python operations) is `2^{8}` =
256× short of `2^{40}` per target, even before counting the
constants. A research-grade C/Rust implementation pushing `10^7`
ops/sec achieves `~2^{36.5}` ops in 4 hours, still `~12×` short.

A genuine ECDLP breakthrough would necessarily violate at least one
of the assumptions in A1–A10. The most likely candidates are:

- **Strengthening Diem A6's heuristic** to an algorithmic
  `L_p(2/3)` algorithm. This would require a polynomial-time
  relation-finding routine on the small-`x` structured factor base.
- **A new lattice/algebraic-geometric attack** not in the A1–A9
  catalog.
- **A quantum speedup** (out of scope).

The first is the most concrete open research direction. No paper
through CRYPTO 2025 in the bundled corpus has published such an
algorithm.

---

## Implications for AutoLab scoring

Given A1–A10, the verifier's `large_precomputed_scalar_free_proof`
bonus (`+500` novelty + `+3200` relation_bonus, gated on a
verifier-checkable relation on the 80-bit hashed factor base) is
**provably unreachable by any published algorithm** within the
2-CPU × 4-hour budget. The bonus is intentionally set above the
algorithmic state of the art: it is a research target, not an
engineering target.

The realistic small-prime ceiling — `6168.96` production, `6668.96`
local — is the honest score on the available algorithmic state.

---

*Author:* AutoLab ECDLP campaign  
*Date:* 2026-05-27  
*Status:* working note; the catalog A1–A10 is intended as a
rigorous-as-possible enumeration of the structural reasons each
classical attack fails on the bundled curves. It is not a proof
that *no* algorithm exists, but a proof that no *published*
algorithm in our compute envelope solves the bundled instances.
