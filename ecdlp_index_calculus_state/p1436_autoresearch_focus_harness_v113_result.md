# P1436 autoresearch focus harness V113 result

Date: 2026-08-01

## Result

V113 binds R164 as the 100th closed frontier lane and routes the highest
priority action to `s56_output_sensitive_elliptic_translation_target_norm`.
The report remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R164 gives every public target a distinct field label and samples one uniform
degree-below-`N` randomizer polynomial in the target-label algebra. For a
regular endpoint-target pair it replaces R163's Fermat projector by

```text
c_j(P) = a_j(P) + r_j b_j(P).
```

Every true same-target match remains a root. A nonmatch cancels with
probability at most `1/p`, so the complete false-union probability is at most

```text
nN/p = B^(9/4+5/4-5) = B^(-3/2).
```

Factoring the candidate and applying R163's direct target scan removes every
false root and gives expected `B^2` exact verification. A forced `r=-a/b`
control creates a false root and verifies that it is removed.

Denominator incidents are now a separate affordable branch:

```text
J = gcd(U, product_j (X-x(T_j))).
```

Signed C3 x-injectivity bounds incidence pairs by `N`; the factor and x-bucket
checks cost `B^(9/4+o(1))`. The positive `P+(-2P)=-P` tangent control and its
opposite regular orientation both pass.

The remaining primitive is exactly the regular target norm
`gcd(U,Norm_target(a+R*b))`. A standard represented coefficient-ring product
or resultant still costs `nN=B^(7/2)`, above rho. Dahan's dynamic evaluation
supplies correct quotient-ring splitting but no output-sensitive simultaneous
norm for this elliptic-translation family.

The finite producer enumerates endpoint-target pairs and receives no attack
credit. No below-rho norm constructor, unconditional generic-prime algorithm,
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is claimed.

## Verification

- Focused R164 tests: 15 passed.
- Full harness tests: 130 passed.
- Full ECDLP suite: 982 passed in 155.529 seconds.
- R164 clean replay: all six JSON outputs byte-identical.
- V113 clean replay: note byte-identical; three JSON outputs identical after
  removing `generated_at` timestamps.
- Parent audit R76-R164: 89 receipts, 1,689 bindings, 0 mismatches, no missing
  paths, and no missing or duplicate rounds.
- V113 frontier preflights: 100 provided, 100 closed.
- R164 obligations: 20 of 27 passed; lane admission false.
- R164 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V113 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R164 producer: `770294394c4f9b4e39652224ef44918242198c6c99d9fb786f9883774bb4044b`
- R164 report: `8d9ddeaf584d8d8bd41c3373c9c765b1361aa9a287cf763217b1296ed85b389a`
- R164 frozen interface: `bfd2d51835b22c70ddb84217fc12c21187220ef7c2e40993cbcfe3b21cef2781`
- R164 cost ledger: `d2ad661d6fec3bbb1bb9bad95a549fa89ab7528a2abdbbd17dee486e8d607ea1`
- R164 replay: `db29165a34163d1cab89569c883a41f1e61eb1c2ae56f341316b50fcd6c708e4`
- R164 controls: `6604f752e894d891afe39e44eb984260a705d0eac86b5734e9dca636e1574cc9`
- R164 label algebra: `718cbcb4a79a238eb2995b16c56098a7c5cab510dd2b6d1b7faaf730f72dfd62`
- R164 tests: `3056cfbbd9abea6dd1b436216112b2d073cd604eb3fc12882b253b510341327a`
- R164 gate: `65f78505220ff5e396fb834c9c81034861b80391578089cc295d72c9be771ea3`
- R164 parent: `b0aa13e8a47535b223824c97b18f4f577ad7723296fb54d5f11f9be27d6622fc`
- Dynamic-evaluation paper: `e17f13261cab77b08313c5524764e2a7b1030dfc83b56afc1a88c954034c667f`
- Harness: `ab007dd7ca169fed86751eafa0f6cfc77087fc02963cf5812d7005e3030bf9ea`
- Harness tests: `f9ab24952e08f1568d2e42266d5373c686878e76fb1f9b41b70199bb439c81eb`
- Report: `f91da97f13ad6f2b1d5802808965a76291c30732d01b3726dae76e3cb5b69243`
- Note: `de44143e96f81ed2a314b5122e14c791b77f8c462ee90449ae6b275f96ddb8e8`
- Inventory: `30d5274411c71d990fe33da73f3e6d620677b654a761569489e29f4cf56a3a6f`
- Replay plan: `feb82a1a2f00cf33fa7c15d3272c425134c4a5cbe5ca719db5e0d9b3081d0d7a`
