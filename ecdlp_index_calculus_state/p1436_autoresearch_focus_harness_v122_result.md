# P1436 autoresearch harness V122 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V122 binds R173 as the 109th closed frontier lane and routes the first
experiment to `s65_commutative_target_sign_divisor_pushforward_mod_u` at
priority 308.

## R173 Result

For

```text
s1 = X + Z + u,
s2 = XZ + Xu + Zu,
s3 = XZu,
```

Semaev's third summation polynomial has the exact determinant form

```text
S3(X,Z,u) = (s2-a)^2 - 4 s1 (s3+b)
           = det([[s2-a,2(s3+b)],[2s1,s2-a]]).
```

Writing this matrix as `M_u(Z)=A(X,u)+Z B(X,u)` gives

```text
A = [[Xu-a, 2b],       B = [[X+u, 2Xu],
     [2(X+u), Xu-a]]        [2,   X+u]].
```

The coefficient matrices do not generically commute:

```text
[A,B] = diag(delta,-delta),
delta = 4(b-Xu(X+u)),

M_u(z1)M_u(z2)-M_u(z2)M_u(z1) = (z2-z1)[A,B].
```

Six controls verify 68,326 determinant evaluations, 1,486 discriminants, and
1,675,890 ordered distinct-root pair identities. Every tested root pair and
all 118 target-factor pairs are noncommuting; zero tested points lie on the
exceptional `delta=0` locus. Reversing target order changes two matrix entries
in every control.

The determinant is nevertheless order invariant:

```text
det(product_j M_{u_j}(X,Z))
  = product_j S3(X,Z,u_j)
  = Res_Y(W(Y),S3(X,Z,Y)).
```

Both target orders replay all six R172 reverse resultants exactly.

## Discriminant Boundary

As a quadratic in `Z`, the exact discriminant is

```text
disc_Z S3(X,Z,u) = 16 (X^3+aX+b)(u^3+au+b).
```

On the selected and target curve divisors, its square root is
`4 V(X) V_T(u)`. Diagonalizing the pencil therefore returns the plus and
target-sign-conjugate branches already exposed and charged by R172. It does
not supply a third compact branch or the missing aggregate operator.

The represented transfer cost is:

```text
factored pencil input state:                 B^(5/4)
represented ordered 2 by 2 transfer body:   B^(5/2)
represented scalar determinant body:         B^(5/2)
standard outer root/factor grid:              B^(7/2)
represented aggregate output:                 B^(9/4)
rho proxy:                                    B^(5/2)
```

The six represented matrix products reserve 1,448 coefficient slots and use
1,418, with coefficient-rank sum 184. Materializing the transfer is therefore
at the rho exponent before the outer self-resultant. Finite noncommutativity,
density, and rank receive no asymptotic or general lower-bound credit.

R173 closes only the naive order-independent `2 by 2` matrix transfer and its
represented product. Custom commutative, transposed, implicit, and arbitrary
squarefree-modulus algorithms remain open. R173 passes 14 of 23 obligations;
lane admission, rho improvement, Shoup improvement, and breakthrough flags
remain false.

## V122 Routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v109`.
- Bound and closed frontier lanes: 109.
- First focus: `s65_commutative_target_sign_divisor_pushforward_mod_u`.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.

## Verification

- R173 focused tests: 14 passed in 96.144 seconds.
- Harness tests: 139 passed in 0.741 seconds.
- Full ECDLP suite: 1,131 passed in 532.825 seconds.
- R173 clean replay: all six JSON outputs byte-identical.
- V122 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R173: 98 receipts, 1,904 recursive path/hash bindings, zero
  mismatches, missing paths or rounds, and duplicate rounds.

## R173 Hashes

- Producer: `dd63e3404bb9cd6cf24de48234b8240e6a7918eefb200bd76292c2e410119f67`
- Report: `2a9d6fafbee9c3c8e3d9323af291476bb662de075090cb497ad9e8555dafe75c`
- Frozen interface: `2032a4ef8c743de4db64d0e49bf8763fd23270120f26bb937629e9ef60410533`
- Cost ledger: `7a0bb4812e6e7facbe4528e4717e34dfdd6b08f4d2b19af50f72ae921b490588`
- Replay: `57840c79d02126b9a4510273465df74fc0316c94913d2739a131ca0a880f1c4b`
- Controls: `c53431049b58348471d8cf8d5797c7e9e7fcf52f0fc0bdd05e3631f26e1a2d64`
- Transfer ledger: `4bd35b11d490b12445c8c98bd568e40d91259609b7a2976984c3f76c9454004c`
- Tests: `2f0b521d7e78901247119376a167951a4805e202976576aafe729f26e0107740`
- Gate: `437a665efca2d9c0903e801f11b44f0c69f1d0a14a64ae4b69603593049442f7`
- Parent: `c57613845a891c5a1f917bd0721a7b7041676c6abc289c194f92d083022a2a84`

## V122 Hashes

- Harness: `f5762d29a3d70b699a8fd4e3959f7d43c28b9be2556030a8bb124fc8a105523c`
- Harness tests: `6169e3919db0ddc05276296288ed45d1b74a8afdfa1dfde6938547e9c2827350`
- Focus report: `732a55d0dd6c202949e449b90e830cc2a372e78d014d08fd1c1364862042167c`
- Note: `04c03505bc25a3eed2bd5415e685eeb53499c195360ee40a86cd12e1835e6490`
- Evidence inventory: `9990d43ba7efcdf3a4c11e6d1bba92eff3973939cea45ea8f1de1ce831d5da67`
- Replay plan: `e9c2e05fb19f9129c11fb0a137bb6939dc0ba9ada452a81e9ebde0c9cf3a10b7`

## Claim Boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, or breakthrough was produced. Exact identities, finite controls,
order obstructions, density/rank observations, source recovery, and verifier
passes receive no asymptotic attack or lower-bound credit.

## Next Action

Use the known square root `4 V(X) V_T(u)` directly. Formulate the two diagonal
target-sign branches as compact divisor pushforwards modulo arbitrary
squarefree `U`, then test a transposed modular-composition, multipoint, or
equivalent commutative arithmetic DAG. Charge all target-dependent setup,
squarefree transforms, quotient reduction, exceptional branches,
factorization, output, and signed verification. Require total work strictly
below `B^(5/2)`, preferably `B^(9/4)`, and reject `N^2` coefficient bodies,
`nN` pair visits, candidate inversions, or unit-cost norm, resultant, root,
count, marginal, rank, source, or multipoint oracles.
