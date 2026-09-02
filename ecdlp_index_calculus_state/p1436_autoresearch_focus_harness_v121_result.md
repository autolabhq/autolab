# P1436 autoresearch harness V121 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V121 binds R172 as the 108th closed frontier lane and routes the first
experiment to `s64_factored_self_s3_resultant_mod_u` at priority 306.

## R172 Result

R172 removes target-y dependence from the R171 Kummer target locator. For
`P=(X,V(X))`, `T=(u,v)`,

```text
D = (u-X)^2,
A = u^2 X + u X^2 + a(u+X) + 2b,
H_U(K,D) = D^n U(K/D).
```

The target and target-sign-conjugate factors are `H_U(A+2vV,D)` and
`H_U(A-2vV,D)`. The exact identity

```text
(A-DZ)^2 - 4(u^3+au+b)(X^3+aX+b) = D S3(X,Z,u)
```

gives

```text
H_U(A+2vV,D) H_U(A-2vV,D)
  = D^n Res_Z(U(Z),S3(X,Z,u)) mod U(X).
```

For `W(Y)=product_j(Y-u_j)`, multiplication over targets gives the iterated
resultant

```text
G_+(X)G_-(X) = R(X) / W(X)^(2n),
R(X) = Res_Z(U(Z), Res_Y(W(Y),S3(X,Z,Y))).
```

The denominator is one modular exponentiation after the public
target/selected x-equality split.

Six controls verify 2,972 homogenized factors, 1,486 target-sign conjugate
identities, 8,922 factor-product rows, and the quotient-ring denominator
identity. The plus branch exactly replays all 140 R171 candidate roots. The
conjugate branch contributes one additional finite-control root and no overlap;
all roots retain signed verification. These finite counts receive no
asymptotic attack credit.

## Representation Boundary

The reverse resultant is the factored product

```text
Res_Y(W,S3) = product_j S3(X,Z,u_j).
```

Its represented bivariate grid has `(2N+1)^2` coefficient slots. Across the
six controls all 1,270 slots are nonzero and the coefficient matrices have
full rank, with rank sum 86. This is finite evidence about these controls, not
a resultant, arithmetic-circuit, RAM, cell-probe, or generic-group lower bound.

The cost boundary is:

```text
compact target divisor / factored S3 state:  B^(5/4)
batched denominator:                         B^(9/4+o(1))
represented aggregate output:                B^(9/4)
represented reverse-resultant body:          B^(5/2)
standard point/factor-local route:            B^(7/2)
rho proxy:                                    B^(5/2)
```

Materializing the reverse resultant is exactly at the rho exponent before the
final self-resultant. The standard local route is above rho. The only surviving
interface is an unproved algorithm that keeps the `N` quadratic `S3` factors
and emits the self-resultant remainder modulo arbitrary squarefree `U` in
softly `O(n+N)` work.

Semaev supplies `S3`, not that factored self-resultant. Moroz-Schost compute a
local truncation from represented bivariate inputs, and Hyun-Neiger-Schost
compute generic bivariate resultants from represented polynomial-matrix inputs.
Neither audited contract avoids constructing the relevant represented input.

R172 passes 24 of 31 obligations. Factored-self-resultant admission, lane
admission, rho improvement, Shoup improvement, and breakthrough flags remain
false.

## V121 Routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v108`.
- Bound and closed frontier lanes: 108.
- First focus: `s64_factored_self_s3_resultant_mod_u`.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.

## Verification

- R172 focused tests: 16 passed within the integrated run.
- R172 plus harness tests: 154 passed and 6 subtests passed in 58.74 seconds.
- Full ECDLP suite: 1,116 passed and 10 subtests passed in 430.91 seconds.
- R172 clean replay: all six JSON outputs byte-identical.
- V121 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R172: 97 receipts, 1,883 recursive path/hash bindings, zero
  mismatches, missing paths or rounds, and duplicate rounds.

## R172 Hashes

- Producer: `297dc3d3195926341fbbd79c7d0346e59bfcb1cd721e5b4f0d4b3e2cb4d615c8`
- Report: `66f986b68c6d31576c688da4509d828139275331009dba4bb05abd45aef5ce5e`
- Frozen interface: `5a6938348733ec3d103ec2044198bcb5aef06c2119da01d45d5f746ed9e737bf`
- Cost ledger: `b8ceb6c4d9cc9d6365ce861ba17b6ac631c0e8f4f951d56dc4d94d0e351602f0`
- Replay: `7975e14f92f9fedd73193ace641d3ea6e2d97387b512be16ad9b01133fb15920`
- Controls: `d176c75580c5810c37e6e9debb0a2a91636fe17f28bbec0b516f2a3cd164bc99`
- Resultant ledger: `281fcf996ceddeeef5b5da57930e6b5dfc44da6da743719726fc2ebffd7f299d`
- Tests: `b9b663f3e502971257047a6b39d09a4d969030ebfd928927b14be4f474385dd1`
- Gate: `14fb9459c75510f26177e2ef795266a825f58c2cdd11797d6ec6e60bd31dc951`
- Parent: `45e476d650cb7c54bb3f9859175cded032d823881d76c706f60722c3e9d6a1bb`
- Hyun-Neiger-Schost reference: `32b73cf0ca7172bdec0f8f1b256adda628a86d6dd7eee8e07e8644e35a9f16f3`

## V121 Hashes

- Harness: `df9c4c3878b504036febe2ed19232b8fe188d813bee04496e3ac79e4178595b3`
- Harness tests: `e1994032252c5436def8493edd91ad354f0085ea812cfabe4d37bc70cd15a16d`
- Focus report: `0c8466d500b64c77d4f579f33014367317208d7f7df43b63e226838e48227e3b`
- Note: `c326f1fc218c79e708fdb3ae254746f7e05446f24e752309b35d2275997d0360`
- Evidence inventory: `d5af175ec20c36e7240b5fdb6b21c5a425428a7ed32e53c41b7eb2eed630e159`
- Replay plan: `5091256520d22b3f5fed9fd4a804af559a5aad4d39c2fd6c0cf8d6bad15d7f58`

## Claim Boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, or breakthrough was produced. Exact identities, finite controls,
density/rank observations, source recovery, and verifier passes receive no
asymptotic attack or lower-bound credit.

## Next Action

Construct or refute a factored self-`S3` resultant algorithm that emits

```text
Res_Z(U(Z), product_j S3(X,Z,u_j)) mod U(X)
```

in softly `O(n+N)` work, preferably `B^(9/4+o(1))`, without materializing the
`N^2` reverse-resultant body, visiting `nN` point/factor pairs, building an
`nN` or `n^2` polynomial matrix, inverting candidate nonunits, substituting a
local `x^k` truncation for an arbitrary squarefree-`U` remainder, or invoking
an uncharged resultant, norm, root, count, marginal, rank, source, or
multipoint oracle.
