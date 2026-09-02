# P1553 maximal-class orbit gate R57

## Classification

- Owner: existing P1553/IDEA-195 primitive split-pencil frontier; no new idea
  ID.
- Evidence: exact class enumeration, reflection-orbit reduction, three exact
  multiplication matrices, and exhaustive product-pair scans; no
  cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_MAXIMAL_CLASS_ORBIT_RULING_GATE`.
- Labels: `toy`, `exact-maximal-class-enumeration`,
  `exhaustive-orbit-representative-product-lines`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: among 1,459 nonempty unordered degree-three class
  triples, exactly four maximize the finite catalog at 3,136 products. They
  form three orbits under path reflection. Every orbit representative has a
  rank-nine `27->9` multiplication map with kernel dimension 18; exhaustive
  pair scans find only one-factor ruling lines and zero primitive coprime
  trisecants. This closes only the maximal-product class orbits. No Shoup-bound
  improvement or ECDLP breakthrough follows.

## Inputs

| Input | SHA-256 |
|---|---|
| R56 fixed-class multiplication gate | `73c02f10b511e142235e16e1fd6238fbef079c2eaec70c37d7e19115bbee582c` |
| R56 exact report | `c59653b3c59f5c04bd468c03c9f3271443694eebad783a90e192fcac267e0d2f` |

## Class orbits

The 756 within-window triples occupy 97 class residues. Exhausting unordered
residue triples with total class `51 mod 103` gives 1,459 nonempty triples.
The maximum product count is 3,136, attained exactly by

```text
(16,42,96), (17,41,96), (17,42,95), (18,41,95). (1)
```

Path reflection `k -> 80-k` sends a triple class residue to

```text
rho -> 240-rho mod 103.                           (2)
```

It pairs `(16,42,96)` with `(18,41,95)` and fixes the other two unordered
triples. Factor permutation does not change the product set. Thus three
representatives suffice.

## Exact classifications

| Representative | Trisecant lines | Line sizes | Base degrees | Primitive |
|---|---:|---|---|---:|
| `(16,42,96)` | 5,348 | `3:4900, 4:448` | `6:196, 7:5152` | 0 |
| `(17,41,96)` | 5,152 | `3:4508, 4:644` | `6:196, 7:4956` | 0 |
| `(17,42,95)` | 5,152 | `3:4508, 4:644` | `6:196, 7:4956` | 0 |

Each representative contains 3,136 distinct products and 4,915,680 section
pairs. Each multiplication matrix has rank nine and kernel dimension 18. In
all three scans, every trisecant line varies exactly one factor. The three
canonical line-classification digests are preserved in the R57 JSON report.

## Scope

R57 closes the four maximal-product class triples only. The other 1,455
nonempty class triples, full rational factor spaces, cross-window factors, and
nonproduct sections remain open. The result is one finite toy and supplies no
asymptotic family, target locator, R10 coefficients, rank campaign, logs, or
descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_MAXIMAL_CLASS_ORBIT_RULING_GATE
97_FACTOR_RESIDUES_1459_NONEMPTY_UNORDERED_CLASS_TRIPLES
FOUR_MAXIMAL_3136_PRODUCT_TRIPLES
THREE_REFLECTION_ORBITS_EXHAUSTED
ALL_MULTIPLICATION_MAPS_RANK9_KERNEL18
ALL14747040_PRODUCT_PAIRS_SCANNED_ACROSS_REPRESENTATIVES
ALL_TRISECANTS_VARY_EXACTLY_ONE_FACTOR
ZERO_PRIMITIVE_COPRIME_TRISECANTS
LOWER_PRODUCT_CLASSES_OPEN
NO_TARGET_R10_RELATION_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: leave the sparse path catalog and search the full
rational curve for a primitive product pencil. Generate coprime fixed-class
products from marked subgroup triples, scan each pencil's rational fibers,
and require a third fiber that splits into the same class factors.
