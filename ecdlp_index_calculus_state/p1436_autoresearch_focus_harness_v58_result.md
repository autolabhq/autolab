# P1436 Autoresearch Focus Harness V58 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, target descent, or algorithmic breakthrough is
claimed.

## R107: explicit non-character resultants

All 34 nontrivial source-root partitions were exhausted. The best standard
Sylvester/subresultant split is `12/5` against `13/5`, so its coefficient
interface costs `B^(13/5)` and exceeds both campaign caps. An unrestricted
resultant computes source-dependent partition weights rather than the
canonical source count: the two actual double fibers have weights `[15, 12]`
and `[8, 35]`, while a collapsed-deck control has order 31500 rather than a
generic order two.

This closes the frozen standard resultant, truncated-subresultant, and
Macaulay grammars only. It is not a general arithmetic-circuit lower bound.

```text
R107 producer  2598f635a8572c816eec2c79873da60e71af241fddbce7a0deb90d0918ffe00a
R107 report    0f85b6c29df4b91a56b6f17badc08c08f34c95e21c3b5219261a9308f493d65d
R107 gate      74343856b9bd1865d92bde6055b9d3b441d2ad716797f478d99ab8ee85cc1c51
R107 parent    7859557b1d418699aabde0da890cbe5445f987001950e84b7ba8c2682b5d850c
R107 tests     aa17ad00d2642d2d671cb0315b757089487667309ee7eac8a5f7dab468060c5a
```

## R108: canonical cycle-index identity

The degree-five cycle identity

`120 h5 = p1^5 + 10 p1^3 p2 + 15 p1 p2^2 + 20 p1^2 p3 + 20 p2 p3 + 30 p1 p4 + 24 p5`

gives a seven-term canonical side divisor. Its two-deck product is an exact
49-term `5A+5C` identity. All eight actual and eight matched decks replay
exactly; every side source has weight 120, every full source has weight
14400, and all ten R105 markers are preserved. This is a genuine exact local
positive.

It does not yet yield an inside-cap evaluator. The coefficient-one
`(1^5,1^5)` term retains the original `B^5` source body and `B^(13/5)` root
interface. The theorem of the cube factors the line-bundle class, not the
target section.

```text
R108 producer  99fb8e79f3e22ae82b47d3282edbc17ad0616bc9d30511114d38c919d6ee3e19
R108 report    91fab519c6a59851a672aed1a9e18fac8c434359c91a9e1d9fc0acb112ee7b41
R108 gate      9d7bf5d8b5420d52950ce28a08b7d59a94307bf66752805967338727f3883ab7
R108 parent    6705c882ec1b80538bd3a7e853f33c4f832afae5e35c2063a19a04794d928888
R108 tests     a662fb2e77db846bb03ce797d38501e3804c6fa9be5376690f87293b0d414360
```

## R109: target-section rank

The equality fiber `sum(P_i)=T` contains no zero cylinder depending on at
most two coordinates, refuting every regular finite-valued pure product of
one-body and pairwise factors. Distinct signed section translates have
distinct order-three poles, so their uniform two-block separation rank is at
least `B^(12/5)`, above both caps.

All 56 unary/pairwise subsets were checked on all 16 actual and matched
instances. The balanced side maps are injective, and every sampled signed
section matrix is full rank; the smallest complete matrix has rank 24.
Rational pole-cancellation networks and actual-deck-specific implicit
high-rank circuits remain open.

```text
R109 producer  085b3df3d7442a78800dcd01e67c8fc155a9dfb1181d9a141ed9097c2af65c2e
R109 report    73c826964d17877ea26436c40bd52753f2b5d0c416fee86ef04f2b4c8f030e4d
R109 gate      fc876cc0b83fc43c61c2b95eff1cb0229663f1974b024d4a8f4619b264b7cc00
R109 parent    dd826bb338e9eeada0692f75ec94f7cecc2ae2b407b0830120d4c9c742e1d9ef
R109 tests     8370de8599fdd2a7bf1b3ddae6472467b5c68ab4c0cde32dcd206b0c7dc158cc
```

## Harness routing

V58 is bound to the real two-map collision record with source hash
`c01933da5cebe3c654404f84951ddd8fd4783b4e4791863aca8dac296cc6d1b7`.
Schema v51 has 45 provided and closed hash-bound lanes. Promotion remains
false, and the first action is:

```text
s6_5a5c_theta_addition_cancellation_network
```

That lane must construct or refute one exact finite-field theta-addition
network which generates the high-rank target section implicitly while
keeping every intermediate bond and contraction inside both caps. It must
audit rational poles and cancellations, preserve the exact R108 weight 14400
and R105 markers, and continue through multiplicity, integer lifting, rank,
factor logs, and identical target descent.

```text
harness              91be704fa121c372746fdfe194a041d93c6c250ee8df7fa701c3a8b83416c0f8
harness tests        004b32bee74d5c55d3f090683fb29439b6a5d3679c71594f6dd7e12f8ad1eebe
V58 report           e1230a303a7da593ab7e5e8dd28592878b350a39b5acbe57a03c71dd180183e0
V58 focus note       04b61afbdad39b722775bb32fae655f8a47ac2f7c98e1ddc6c6a0f566aeb7fe6
V58 FFE inventory    8a018079c93a9be862db44dd41238732df89a854cda14888b7622ed8a532fb42
V58 FFE replay       9fd0fb4f48f715d02c03e7561a67439323fbca144348aa7428849863c2c2fa23
```

## Verification

- R107-R109 plus harness targeted tests: 96 passed.
- Full ECDLP task suite: 334 tests passed.
- Eight producer, harness, and test modules compiled.
- All 34 R76-R109 parent YAML receipts parsed.
- All 544 declared R76-R109 input and artifact hashes matched.
- No true breakthrough, Shoup-improvement, or promotion flag appears in the
  R107-R109 structured claim records.
- Clean R107-R109 reruns reproduced all 18 generated JSON artifacts
  byte-for-byte.
- V58 has 45 closed lanes, `promotion_allowed=false`, and the expected R110
  theta-addition cancellation-network action.
