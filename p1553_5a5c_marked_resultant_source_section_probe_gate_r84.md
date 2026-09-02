# P1553 5A+5C marked-resultant source-section gate R84

## Classification

- Owner: existing P1539/P1553/R82 Cartesian-sum factor-base lane.
- Evidence: exact finite scalar-blind construction, exhaustive containment
  replay, coefficient-level controls, and prospective exponent accounting.
- Status:
  `EXPLICIT_MARKED_SOURCE_SECTION_EXACT__COEFFICIENT_BODY_OVER_CAP`.
- Labels: `exact-finite`, `scalar-blind-source`, `prospective`,
  `verifier-dlp-separated`, `scoped-negative`, `novelty-unverified`.
- Cryptanalytic result: no factor-log solve, blind descent, Shoup-bound
  improvement, or ECDLP breakthrough.

R84 constructs the source section requested by R83. It uses an injective
point key over `F_(p^2)`, a radical endpoint polynomial, and one packed
mixed-radix source interpolant for each side of

```text
(2A+3C) + (3A+2C) = T.
```

The section is exact on every frozen instance. It returns all ten atom
indices as two coupled packed codes sharing the same exact join root. The
passing representation is nevertheless over cap: the two explicit side
coefficient bodies have prospective generic degrees `B^(2.6)` and
`B^(2.4)`, and even the smaller one exceeds the `B^(9/4)` setup limit. A
fresh target-specific translated-right polynomial emits `B^(2.4)`
coefficients before gcd or source opening, far above the `B^(5/4)` online
limit.

This is a scoped explicit-coefficient result. P1510's audited
output-sensitive compiler and P1511's pre-leaf exception remain intact.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R83 coordinate-filtration report | `1478cdf21493ffbeaed0859af849ea6f2835027f23db7e3e4c008f3f24db500c` |
| R83 gate | `7907df5232c7a6322c797ec33b7600042e223c2c21cb5c832b285995a77fafdf` |
| R82 Cartesian-sum report | `ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832` |
| P1510 output-sensitive compiler | `20c7e26c55801aba57d2095254823f20de66ca499c8281ac761603391e8f0d68` |
| P1510 independent audit | `e89c11c5a57ae2ac90f4d42b3d33558cb1c1ba1765d7409a7940accba3098452` |
| P1511 factorized-semijoin gate | `4b393c9805e5d7bc008451a9e275e5b41cf917fd9d87c7f99237783a0f4440d6` |
| R16 common-right-factor gate | `9f7ec4ecb6821e30affce39b25b459891f6be0b945b6d64332971dd86df2237c` |

The producer verifies all seven hashes before building an instance.

## Frozen construction

Let

```text
|A|=u=B^(2/5+o(1)),
|C|=v=B^(3/5+o(1)),
F_(i,j)=A_i+C_j,
|F|=uv=B,
|<P>|=Theta(B^5).
```

The two source decks are the unordered multisets

```text
L = 2A+3C,
R = 3A+2C.
```

Their exact source counts are

```text
K_L = binomial(u+1,2) binomial(v+2,3) = B^(2.6+o(1)),
K_R = binomial(u+2,3) binomial(v+1,2) = B^(2.4+o(1)).
```

For a nonsquare `nu in F_p`, use

```text
F_(p^2)=F_p[w]/(w^2-nu),
kappa(O)=0,
kappa((x,y))=x+w*y.
```

The affine encoding is injective because `1,w` are an `F_p` basis.
`(0,0)` is not an affine point on the frozen curve `y^2=x^3+1`, so the
identity tag is also disjoint. No scalar labels or verifier DLP values enter
this key.

For each side retain one lexicographically first source per endpoint and
construct

```text
L_rad(Z) = product_(ell in support(L)) (Z-kappa(ell)),
R_rad(Z) = product_(r in support(R)) (Z-kappa(r)).
```

Pack a complete side source in one mixed-radix base-field value. Lagrange
interpolation gives `I_L` and `I_R` satisfying

```text
I_L(kappa(ell)) = packed_source(ell),
I_R(kappa(r))   = packed_source(r).
```

These are jointly coupled selectors, not independent marker moments.

For a fresh target `T`, the explicit marked join is

```text
gcd(
  L_rad(Z),
  product_(r in support(R)) (Z-kappa(T-r))
).
```

After choosing one common root, the exact point key identifies `ell`, the
point `r=T-ell` identifies the right endpoint, and `I_L,I_R` return the two
sources. Every decoded atom tuple is replayed on the elliptic curve.

## Exact finite receipts

| `B` | copies | left support | right support | split pairs per copy | direct 5A+5C sources | target support range |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 2 | 30 | 24 | 720 | 126 | 126 |
| 15 | 2 | 210 | 150 | 31,500 | 2,646 | 2,631-2,646 |
| 18 | 2 | 336 | 210 | 70,560 | 5,292 | 5,292 |
| 28 | 2 | 840 | 560 | 470,400 | 25,872 | 25,816-25,872 |

Across all eight instances:

- every left and right multiset source has a distinct side endpoint;
- all radical roots vanish exactly;
- every selector value decodes to its frozen source and group-replays;
- 1,146,360 side pairs are enumerated for audit only;
- their target union is exactly the direct `5A+5C` target union;
- that union contains 67,801 distinct targets;
- one coupled split source replays for every one of those targets;
- all 32 target-specific coefficient gcd controls have degree equal to their
  exact point-set intersection and return a coupled source.

The gap between split-pair count and target count is primarily expected
split multiplicity: one final multiset can be partitioned into left and right
in several ways. Direct `5A+5C` enumeration separately measures accidental
endpoint collisions. Its collision excess is `0,0,15,0,0,0,56,0` over the
eight frozen instances.

The all-target enumeration is a completeness verifier. It receives no
candidate work credit.

## Coefficient and cap accounting

The source-product polynomial before radicalization has exact degree `K_L`
or `K_R`. On every finite fixture the side endpoint maps are injective, so
the radical polynomials have those same finite degrees. R84 does not promote
this finite observation into an unrestricted asymptotic injectivity theorem.

Under the frozen prospective generic-support model:

| Object | Exponent in `B` | Gate |
|---|---:|---|
| left radical/source coefficients | 2.6 | above setup |
| right radical/source coefficients | 2.4 | above setup |
| smaller explicit side | 2.4 | above `9/4` |
| fresh translated-right output/work | 2.4 | above `5/4` |
| exhaustive side-pair containment audit | 5.0 | verifier only |
| generic collision baseline | 2.5 | `N^(1/2)` |

One packed selector value returns all indices, but its explicit interpolant
still has one dense coefficient body of side-support length. Packing removes
uncoupled provenance; it does not remove coefficient count.

## P1510 and P1511 controls

P1510 is independently verified on its frozen family. It constructs 15
degree-`O(r^2)` marked-resultant coefficient polynomials in
`O(r^2 polylog r)` work and `O(r^2)` state. That is a real output-sensitive
positive result.

R84 preserves it exactly. Applying an output-linear compiler to R84's
explicit side outputs still emits `B^(2.6)` or `B^(2.4)` coefficients. R84
does not claim that every target-specific resultant must emit those side
outputs.

P1511 separately closes P1510-per-target products only after their
provenance leaves are supplied. It leaves open a target-uniform
representation constructed before leaf emission. R84 leaves the same
exception open, together with:

- a compact common right factor or source-returning quotient circuit;
- a circuit-valued Chow, Tate, subresultant, or exterior-syzygy object;
- an arbitrary summation-polynomial or FFE solver;
- a structured asymptotic side family with proven endpoint compression.

## Rank, logs, and descent

R82's public rectangle identities remain exact, but R84's source section is
not inside the setup or fresh-query caps. The exhaustive finite source audit
therefore supplies no algorithmic known-RHS relation rows.

Consequently:

- known-RHS rank without verifier DLP is absent;
- factor logs are not recovered or verified algorithmically;
- identical scalar-blind target descent is absent;
- no generic-prime theorem or Shoup-bound improvement is established.

## Scoped disposition

R84 accepts the exact finite marked source section and rejects its explicit
coefficient-body pipeline. It closes only dense endpoint-polynomial and
packed-interpolant implementations of the frozen `2A+3C` versus `3A+2C`
split under the stated generic-support accounting.

It is not an arithmetic-circuit lower bound and does not reject a
target-uniform representation that acts before endpoint coefficients or
provenance leaves are emitted.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Producer | `80cd0887fd24bfc37f0568f03a8af4c98eb148e030a3112576c1a0940f4ad99d` |
| Main report | `c9b1c5fb0f58f2c5118562623fd5dfff5d55d7238b513892d4178a67af5ccf0b` |
| Frozen construction | `842076c19914abe0e969d159ef3847c9cfb3e87d0f5f3fcefeb058de94c2aa8c` |
| Coefficient/containment receipts | `c2f3b02eed1e763aa4a6608b63617ad7c9f201e1e2339fee4e5855950064cc2b` |
| Joint-source replay | `2cb0f2a2e66d0f25bf5b8ec98912a0ef392a41cf51a4a80848b53b41338ec364` |
| Fresh-query/state ledger | `64c3c66a655a3261d05fcbfca4da50eac9d7ee60331be7187af0a0fb2e7aad53` |
| Factor-log/descent receipt | `6d677a7a00ac3b98168442bba820a8ef176b9d437a2bd20849e2192ec36f0d53` |
| Unit test | `332674727196772d688befd32b558a2c0b64009058c72cdeffbf1304c40197d8` |

## Exactly one next action

Construct or refute one target-uniform pre-coefficient circuit for the
`2A+3C` versus `3A+2C` join. It must consume only the compact `D_A,D_C`
inputs, specialize a fresh target inside `B^(5/4)`, emit no `B^(2.4)` side
polynomial or provenance leaves, return one jointly coupled source, and
preserve exact containment, rank, factor-log, and identical-descent receipts.
