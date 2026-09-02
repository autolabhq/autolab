# P1553 torus C5 explicit hash-correction support gate R126

## Claim boundary

R126 closes represented product-correction descendants of the R125
nonhomomorphic-fingerprint interface. It covers explicit per-bucket-pair
distinct-product lists and a globally deduplicated exact product dictionary.

It does not cover implicit correction circuits, adaptive probes, nonlinear
algebraic certificates that do not list targets, bounded-error structures,
or general arithmetic circuits and data structures. It supplies no complete
source index, rank, factor logs, identical descent, Shoup improvement, or
ECDLP breakthrough.

Classification:

```text
ARBITRARY_COORDINATE_HASH_C2C3_EXPLICIT_CORRECTION_UNION_EQUALS_C5_SUPPORT__SUM_BUCKET_PAIR_PRODUCT_ENTRIES_AT_LEAST_C5__GLOBAL_DEDUP_DICTIONARY_STILL_C5__IID_STATE_B15O4_OVER_CAP__ACTUAL_R82_NO_DLP_HASH_AND_SOURCE_CONTROLS_EXACT__IMPLICIT_ADAPTIVE_CORRECTION_CIRCUIT_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Explicit correction model

Let `h2:C2->A` and `h3:C3->B` be arbitrary target-independent hashes.
For every occupied bucket pair `(a,b)`, an exact explicit correction list
contains

```text
L_(a,b) = {xy : x in C2, h2(x)=a,
                y in C3, h3(y)=b}.
```

The correction lists satisfy

```text
union_(a,b) L_(a,b) = C2*C3 = C5.
```

Therefore

```text
sum_(a,b) |L_(a,b)| >= |C5|.
```

Deduplicating products globally does not escape the boundary: the global
exact product dictionary has exactly `|C5|` keys.

Under the inherited R119 iid-distinct-support theorem at
`|C|=B^(3/4+o(1))`,

```text
|C5| = B^(15/4+o(1))
```

with high probability. Both explicit representations exceed the
`B^(9/4+o(1))` setup cap. Adding one C2+C3 source backpointer to each entry
cannot reduce the number of represented product keys.

## Actual controls

Twenty-four controls use all eight R82 pairing decks with coordinate hashes

```text
h(a+b*u) = a + salt*b mod m
```

at `m=2,3,5`. These hashes inspect field coordinates directly and consume no
pairing-image discrete logarithms.

Every control verifies:

- the union of bucket-pair correction sets equals the exact canonical C5
  support;
- total correction entries are at least the number of distinct C5 products;
- every correction entry has a C2 and C3 source backpointer;
- every merged five-source backpointer replays the represented product.

All C5 products are distinct on these inherited controls. Finite enumeration
receives no asymptotic credit.

## Scope limits

The counting argument charges represented product targets. An implicit
circuit might evaluate whether a target belongs to a correction set without
listing that set, and an adaptive structure might probe a compressed
certificate. R126 proves no lower bound for those routes. Any survivor must
still prove deterministic exact empty semantics, expose five projective
sources, and charge every correction, preprocessing, and query operation.

## Admission

Twelve of twenty obligations pass. The coordinate-hash correction semantics
and scoped explicit-support negative are admitted. The implicit/adaptive
membership-source interface, known-RHS rank, logs, identical descent,
Pollard-rho improvement, Shoup improvement, and breakthrough obligations
remain false.

Disposition:

```text
ADMIT_COORDINATE_HASH_EXACT_CORRECTION_AND_SOURCE_CONTROLS_ONLY__REJECT_EXPLICIT_PER_BUCKET_AND_GLOBAL_DEDUP_PRODUCT_DICTIONARIES_AT_C5_B15O4_STATE__PRESERVE_IMPLICIT_ADAPTIVE_CORRECTION_CIRCUIT__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Freeze one implicit correction circuit or adaptive probe for a
nonhomomorphic coordinate hash. It must answer exact arbitrary-target
membership without listing bucket-pair or globally deduplicated C5
products, prove deterministic noncancellation and empty semantics, return
five projective sources, fit `B^(9/4+o(1))` setup and polylogarithmic query,
avoid field DLP, and charge complete rank, logs, identical descent, memory,
field-operation, and bit costs.
