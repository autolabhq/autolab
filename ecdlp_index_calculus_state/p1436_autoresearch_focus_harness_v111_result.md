# P1436 autoresearch focus harness V111 result

Date: 2026-08-01

## Result

V111 binds R162 as the 98th closed frontier lane and routes the highest
priority action to `s54_aggregate_nonlinear_signed_divisor_below_rho`. The
report remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R162 proves that the denominator inversions in R161 are shareable. For monic
`U`,

```text
Q_u(X) = (U(u)-U(X))/(u-X)
(u-X)^(-1) = Q_u(X)/U(u) mod U
```

when `U(u)` is nonzero; `U(u)=0` detects the exceptional branch. All scalar
`U(u_j)` values admit soft-linear multipoint evaluation. For a fixed
coefficient functional `w`, `<w,Q_u>` is evaluation at `u` of one polynomial
whose coefficients are a cross-correlation of `w` and the coefficients of
`U`. Thus a constant number of linear views of all inverses can also be
batched without materializing one degree-`n` quotient per target.

This does not batch the nonlinear operations `lambda_j^2`, `U(phi_j)`,
`V(phi_j)`, source gcd extraction, or target-labeled backpointers. The
primary literature checked here accelerates one composition, or assumes that
the varying-inner map is fixed during precomputation; neither supplies the
required many-varying-inner operator as stated.

R162 also corrects the successor cost gate. With `n=B^(9/4)`,
`N=B^(5/4)`, and rho `B^(5/2)`, a cost `n^alpha N^beta` is below rho whenever
`9 alpha+5 beta<10`. A target-dependent soft-linear pass over `n` therefore
misses the strict R159 `B^(5/4)` batch phase cap but remains below rho by
exponent `B^(1/4)`. Independent near-linear composition costs `B^(7/2)` and
still fails.

All six controls verify batched `U(u_j)` evaluation, exceptional detection,
quotient/inverse identities, and three transposed functionals. They consume
no candidate oracles and receive no attack credit.

No aggregate nonlinear operator, source locator, unconditional generic-prime
algorithm, Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough
is claimed.

## Verification

- Focused R162 tests: 16 passed.
- Full harness tests: 128 passed.
- Full ECDLP suite: 949 passed in 118.077 seconds.
- R162 clean replay: all six JSON outputs byte-identical.
- V111 clean replay: note byte-identical; three JSON outputs identical after
  removing `generated_at` timestamps.
- Parent audit R76-R162: 87 receipts, 1,646 bindings, 0 mismatches, no missing
  paths, and no missing or duplicate rounds.
- V111 frontier preflights: 98 provided, 98 closed.
- R162 obligations: 18 of 27 passed; lane admission false.
- R162 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V111 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R162 producer: `734938d6e5f7c79b10539dd57583f7a012b66406bfb55e1a3bbcb96e24ee89f1`
- R162 report: `55a46fbdefc4e1c9a84d5286cf825beb1a0c2659ca6653d63cc5b95fbfea1742`
- R162 frozen interface: `69411c74d272c73d721d3a863ea2f25262906d2461eb43f3835ef912531e58a7`
- R162 cost ledger: `ab274fdea53dc0c89b140ab910e6b288d25a56a1912f5b6f26c6ee716450b7e4`
- R162 replay: `a51f01eee7da30b83dcbd00703ba52af085c8b849272f85c1921ab4d21335022`
- R162 controls: `ba3e8bd83c66f2876b599fc095eb69c637963e89783a73bfc9481bff0d95d2d8`
- R162 transpose: `dd373da7cbe4f98ac2f354f7e082bc16f7415345afb90838a94b7576a5596319`
- R162 tests: `863ac5fce63370328976b4c6ce2b67bafb63d29f27bf7d84a7371a40326011ab`
- R162 gate: `e5fa538ae4f3bef7f6764d7271a64f48da7e8e340e0f10eae0af6ce345a10338`
- R162 parent: `a49a2631c8a7b9313fe20794df0c56b7ca6081bc2242b5b05b6fca0eec914927`
- Harness: `812efcdc3d6f9c1c5f6eecfc34230240512b37167e8d58b494e856536998a50c`
- Harness tests: `b78295f9d4747658bdd698be6c851706a88093278fb72aa150df45e577e70432`
- Report: `12081361069a94f2acdef5c410c39f9e7d8c69168b3e3e9e2f4114ffb3602acb`
- Note: `ad1946c96708da7aa80254f6f029c9d2e4f649b902bf8e37b66d2cdecc972532`
- Inventory: `4dffd0f930a2d58743ac1a37ac3d454abebf913d10b91a1e032f3f06b14cbb84`
- Replay plan: `b4515d40d660ebd073a60fec856642392e397b0ebd2a25ff82139db2859a5c76`
