# P1436 autoresearch focus harness V101 result

Date: 2026-07-29

## Result

V101 binds R152 as the 88th closed frontier lane and routes the highest
priority action to
`s44_signed_weight_separable_ffe_elimination_dag`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

For each public R82 A/C atom deck with distinct x-coordinates, R152 freezes

```text
P(X) = product_i (X-x_i),
```

its subproduct-tree geometry, and the nonzero constants
`P'(x_i)^(-1)`. Arbitrary atom payloads are represented by the unique
degree-below-`n` interpolant

```text
W_v(X)
  = sum_i v_i P(X)/((X-x_i)P'(x_i)).
```

The map is linear:

```text
W_(v+epsilon*d) = W_v + epsilon*W_d.
```

Consequently arbitrary tangent payloads change only leaf coefficients.
They do not rebuild geometry or divide by weights. The transpose map
returns atom adjoints and satisfies

```text
<g,Iv> = <I^Tg,v>.
```

All eight actual controls, covering sixteen A/C decks, pass interpolation
roundtrip, tangent linearity, transpose pairing, distinct-coordinate, and
division-safety checks using only public point coordinates. No verifier
scalar labels are consumed.

At the M6 scaling, the dominant C deck has size `B^(3/4+o(1))`.
Subproduct-tree state, one tangent compilation, and one adjoint application
therefore cost `B^(3/4+o(1))`, below the `B^(5/4)` marker cap.

This supplies reusable weight-independent derivative state only at the
leaves. The public y-coordinate side table remains bound, but an x-only
Semaev relation may introduce sign branches. R152 does not construct a
signed, weight-separable internal summation-polynomial/FFE elimination DAG,
nor control its degree growth, pivots, or marker-batch cost.

No candidate DLP, root, count, marginal, rank, or source oracle is
supplied. No generic rank, factor-log recovery, identical descent,
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R152 tests: 11 passed.
- Full harness tests: 118 passed.
- Full ECDLP suite: 794 passed.
- R152 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R152: 77 receipts, 1,434 bindings, 0 mismatches.
- V101 frontier preflights: 88 provided, 88 closed.
- R152 obligations: 15 of 25 passed; lane admission false.
- R152 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V101 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R152 producer: `0ccb39253ab66c97961bd08722dab7ad45248e4b6faaf74f9e752344c4491fb2`
- R152 report: `f2dd5a43593e9e747d969c73afefac9ee7ce7aac3ca79ca2dd009ea1ffe4e46c`
- R152 frozen interface: `6f51f9a2cb8199f8f45c919cfc0e3d2fbdcfb9b9aa2755b57c1d7e56b29e1c25`
- R152 cost ledger: `abb38aac0549ecdf871955000accf92e89fee3d2697a5ebbccf49c25adb9fbf9`
- R152 source replay: `858e0802bf8f0702d6a596b97bb5b7ae514d2a34849a8e55f3ea66dff0ddf5bf`
- R152 controls: `31bfcb6da0a040ee79885ffbb5c2ca5190d2a7c1d97efb9d4e69e00698ab3db4`
- R152 logs/descent: `551b053ad3b4730d0f615b784becb4ae1e11d9665bedd8b365d460e69db7ee92`
- R152 tests: `fbcaad6bac3619973c27bbd990d9a23e8890b5318626e12a9132c53831852282`
- R152 gate: `57c336c6e337e06092efc54af710b8cd5ae8de46edf184d9c3335cd44f4bb38a`
- R152 parent: `7cce460153d6249f9bc01a77dbe4fe6ef6f45fff770b1a013373f4f22c656182`
- harness: `b9d7ba08feb110c61434ab0f6c490179f8fc532b7659a0585d5359c85962d6c2`
- harness tests: `6a32dd1e1f93d4b77dc684c3e7634e0329c831a318cf500d1e1a1dc9c65e8d02`
- report: `6a87d8c2fdac2b8d2f070b040af3bd71fb3fb45e1ad714d82887664b8222a974`
- note: `198a7b07d5b89619afe3575ad4dd6402f88faf127ed0d0516bb8f620c396e016`
- inventory: `91ebb159d1de908e7e6d61f883cbbb89e2cd513ab7727778fe9f8b6f9038cdb4`
- replay plan: `d7322dbb7d25023c72938bf9611ffab7c637cf2f167dfa8d5ca1886de6cca302`
