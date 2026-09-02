# P1436 autoresearch focus harness V114 result

Date: 2026-08-01

## Result

V114 binds R165 as the 101st closed frontier lane and routes the highest
priority action to
`s57_single_function_arbitrary_translate_product_remainder`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R165 proves that R164 does not need independent target randomizers. One
uniform scalar `r` gives every fixed regular nonmatch cancellation probability
at most `1/p`, so the complete union bound remains

```text
nN/p = B^(-3/2).
```

All target factors are now translations of one elliptic function:

```text
f_r(Q) = U(x(Q)) + r(y(Q)-V(x(Q))),
Phi_r(P) = product_j f_r(T_j-P).
```

The function vanishes on every signed C3 endpoint. Since `U` is monic of
degree `n` and `deg(V)<n`, `f_r` has exact pole order `2n` at `O` and a zero
divisor of degree `2n`.

The global translation morphism handles the `P=-T` tangent branch directly.
Only `P=T` maps to the pole `O`; that component receives semantic product
factor one because infinity is outside the affine signed C3 set. A dedicated
control verifies both tangent orientations and the pole equality.

Six global-randomizer controls retain every true root and verify to R163's
exact union. A forced global `r=-a/b` creates correlated false roots, and
direct verification removes them all.

For arbitrary targets, explicitly representing `Phi_r` has zero and pole
divisor degrees `2nN=B^(7/2)`, still above rho. Miller's scalar-chain function
compression does not apply to an arbitrary target divisor without an
additional representation theorem. The surviving primitive is an
output-sensitive remainder of this one fixed-function translate product
modulo `U`.

The finite producer enumerates endpoint-target pairs and receives no attack
credit. No below-rho translate-product constructor, unconditional
generic-prime algorithm, Pollard-rho improvement, Shoup improvement, or ECDLP
breakthrough is claimed.

## Verification

- Focused R165 tests: 15 passed.
- Full harness tests: 131 passed.
- Full ECDLP suite: 998 passed in 166.425 seconds.
- R165 clean replay: all six JSON outputs byte-identical.
- V114 clean replay: note byte-identical; three JSON outputs identical after
  removing `generated_at` timestamps.
- Parent audit R76-R165: 90 receipts, 1,713 bindings, 0 mismatches, no missing
  paths, and no missing or duplicate rounds.
- V114 frontier preflights: 101 provided, 101 closed.
- R165 obligations: 21 of 28 passed; lane admission false.
- R165 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V114 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R165 producer: `32b21c70e961b92a54d975d7294fb13403106e2c8e82f289e19458798d9d615b`
- R165 report: `139277bf3141b7efabf0115399fc4e17377a7ba15783bfc2344dfc84826fc0c2`
- R165 frozen interface: `2d76c303138ecff1d53be8f277012fc87e4ac3ba8874a6f7eafe9c88f5b623e7`
- R165 cost ledger: `5521dc3ec760444f8403ecaa002092550ba7f78596cdbad7cc4ce170932e2daa`
- R165 replay: `5753a28235f5c5ace5ad708df6f7939ed8319ce4951637af84522b542fce7304`
- R165 controls: `619ed335270854a95a6c2b6a66ece31140a13f581fc0656bef0f09bd54d29a37`
- R165 translate product: `fd21e63a694c30913ffc9e540290209ea8315d75f7c7539405dd7593458e83ca`
- R165 tests: `95ef2fddd6ca3e193115c2a869b79161c2342a48bd19b2054493c830a5a68ddf`
- R165 gate: `b0d812d186d2a1bb000d3bb94bdf8f23a74a0d5a338f6636e1f0b7e4fa0002c2`
- R165 parent: `b8a81e2e9b259e5280ce2470bb0d57dc57c41172639832191ee83408945cb08d`
- Harness: `645811ec5058f7a986008453e3cb9ee12acc0af9dcc902ed4c5cb19d0a5e13b5`
- Harness tests: `faaf0e235565ac463381ea12a1583406464eaa671daa1767cba741230e9fb9fd`
- Report: `4795eaf70b901f9f052b21bd87dabf60c1d176b233f4e60900689be3ae524115`
- Note: `a0836b825068b33eb16452afd6c72d6d1a6889b57a1fb19096a341b0ad1248e5`
- Inventory: `c6e09d31a61d1c2370ea039c926aa29356dbb073925103b91d093b2c35631124`
- Replay plan: `f1d5b59ec3d1e9a4af89d6c0fe46c25ad4c0dc089aa6a40e54b57bb4ede89d20`
