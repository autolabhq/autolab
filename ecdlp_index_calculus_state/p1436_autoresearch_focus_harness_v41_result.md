# P1436 Autoresearch Focus Harness V41 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R91: unequal-list subfunction-inversion screen

R91 binds Dinur and Golovnev's current v2 paper and applies its exact
unequal-list theorem to the R82 colored source. For integer lists of lengths
`n<=m`, the theorem gives

```text
S = soft-O(n^(3/2-delta) m)
T = soft-O(n^delta)
0 <= delta <= 1.
```

For the intended five-A versus five-C endpoint split,

```text
n = B^2
m = B^3
S = B^(6-2*delta)
T = B^(2*delta).
```

Setup would require `delta>=15/8`, outside the theorem range, while online
work requires `delta<=5/8`. At `delta=5/8`, the online cap is tight and setup
is `B^(19/4)=B^4.75`. The explicit large-list auxiliary alone has `B^3`
words.

Every nonempty bipartition of the five `B^(2/5)` A decks and five `B^(3/5)`
C decks was checked. The best online-compatible theorem point is

```text
small side  B^(6/5)
large side  B^(19/5)
delta       1
space       B^(22/5) = B^4.4
query       B^(6/5).
```

No deck partition meets setup. The balanced kSUM theorem, optimistically
given scalar labels for a five-factor query, bottoms out at `B^(9/2)` space
and `B` query.

A finite integer-list control verifies exact source reporting. This does not
transfer to a generic prime-order elliptic group: the paper's residue maps
act on additive integer labels, proper homomorphic filters are trivial, and
public point encodings are not addition-compatible.

This closes only direct applications of the bound unequal-list and balanced
kSUM theorems to explicit endpoint lists. Compact elliptic subfunction maps,
non-Fiat-Naor indices, and representation-changing FFE identities remain
open.

```text
paper   e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c
report  6bfaaaa72dd1135f8244a70a0ce2a6697e5f3fe63ce8ac0ce107bd05f19a71da
gate    57469654e0b535ccbf1d62edd9548cd827a09d244deaecb06b8aedbd410ac6d3
parent  d7d1de33b7587956e78a053a028f9201f8b499848472ae9dbafa69e2c247b620
```

Primary reference:
<https://arxiv.org/abs/2512.04258v2>.

## Harness routing

V41 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v41.json`

SHA-256:

`5e375a705e3558055ff534f3da3344bf2c0eaabde4bce82150872c9dfc84a1b4`

Schema: `ecdlp.p1436_autoresearch_focus_report.v33`.

Twenty-seven hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_compact_elliptic_subfunction_map
```

The next experiment must realize public `MAP1`, `MAP2`, `f_d`, and `TR`
directly on compact `D_A,D_C` and one target. It must select one subfunction,
fit `B^(9/4)` setup/state and `B^(5/4)` fresh work/workspace, and return one
exact signed source or bottom on every projective branch without DLP labels,
proper quotients, explicit endpoint lists, or hidden elimination oracles.

V41 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
R91 producer         307be1d618e0cc1aed3736b576da40af7f70207cc0cb5a24e3ad7034df371dbd
R91 tests            eeba582be5168f34a572279d519910519028e08d8236def23c1597f81bdc3315
harness              3bb101c4be5959b351acbdc9f15927ccb5c3629d0a8dc48f933c7acff4cb35a6
harness tests        79be7584f757cae2d9f495cbab93e4eee27f87264aad6b4a6a0335c65e67d27b
V41 focus note       5916b3abf06ba770fe167fd0d3aa8ec8ae2282a81b5536987aaa9cc66f93a315
V41 FFE inventory    50df0045f35429230970789798ce82578faafc193541b9879886560ad44d548b
V41 FFE replay       bccb2f51cdf06c6582d1847f3a271ce68bee97c4f3bb4a67ad5174659fab1ba5
```

## Verification

- R91 targeted tests: 7 passed.
- Harness tests: 57 passed.
- Full ECDLP task suite: 189 tests passed.
- Seventeen R76-R91 and harness Python modules compiled.
- Sixteen R76-R91 parent YAML receipts parsed.
- All 203 declared input/artifact hashes matched.
- Every parent receipt and nested result has `breakthrough=false`.
- A clean R91 rerun reproduced all six generated JSON artifacts
  byte-for-byte.
- V41 has 27 closed lanes, `promotion_allowed=false`, and the expected compact
  elliptic subfunction-map action.
