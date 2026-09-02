# P1553 M6 target-sign conjugate S3 self-resultant gate R172

Date: 2026-08-01

## Scope

R172 removes target-y dependence from the R171 Kummer locator by multiplying
the target branch by its target-sign conjugate. This gives an exact iterated
Semaev `S3` resultant and one batched denominator. The represented reverse
resultant reaches the rho exponent before the final self-resultant, so the only
surviving route is an unproved algorithm on the factored `S3` input.

This is an exact algebraic reduction and a new bounded interface. It is not an
ECDLP algorithm, a resultant lower bound, or a Pollard-rho or Shoup improvement.

## Conjugate Identity

Let `E: y^2=x^3+ax+b`, let the selected divisor be `(U,V)`, and write
`P=(X,V(X))`, `T=(u,v)`. Set

```text
D = (u-X)^2
A = u^2 X + u X^2 + a(u+X) + 2b
H_U(K,D) = D^n U(K/D).
```

Then the two target-sign branches are

```text
H_U(A+2vV,D)  and  H_U(A-2vV,D).
```

The polynomial identity

```text
(A-DZ)^2 - 4(u^3+au+b)(X^3+aX+b) = D S3(X,Z,u)
```

therefore gives, modulo `U(X)`,

```text
H_U(A+2vV,D) H_U(A-2vV,D)
  = D^n Res_Z(U(Z),S3(X,Z,u)).
```

The controls verify 1,486 instances of this identity. The plus branch exactly
replays all 140 R171 candidate roots. The conjugate branch contributes one
additional root across the six controls; this finite count receives no
asymptotic attack credit and every root retains signed verification.

## Target Batch

For `W(Y)=product_j(Y-u_j)`, multiplication over all targets gives

```text
G_+(X)G_-(X) = R(X) / W(X)^(2n),

R(X) = Res_Z(U(Z), Res_Y(W(Y),S3(X,Z,Y))).
```

The denominator is a single modular exponentiation and is a unit after the
public target/selected x-equality branch is split. All six finite controls
verify the quotient-ring identity exactly.

The inner reverse resultant is

```text
Res_Y(W,S3) = product_j S3(X,Z,u_j).
```

Its represented coefficient grid has `(2N+1)^2` slots. The controls instantiate
1,270 slots: every slot is nonzero and each coefficient matrix has full rank.
That finite observation is not a circuit, data-structure, or resultant lower
bound.

## Cost Boundary

```text
compact target divisor / factored S3 state:  B^(5/4)
batched denominator:                         B^(9/4+o(1))
represented aggregate output:                B^(9/4)
represented reverse-resultant body:          B^(5/2)
standard factor-local or pair-local route:    B^(7/2)
rho proxy:                                    B^(5/2)
```

Materializing the reverse resultant is not strictly below rho, and standard
local evaluation is above rho. No cost is assigned to an absent factored
self-resultant algorithm.

## Literature Boundary

Semaev supplies `S3`, not the required factored self-resultant remainder.
Moroz and Schost accept represented bivariate inputs and return a local
truncation modulo `x^k`; their stated contract is not an arbitrary squarefree
`U` remainder of this factored three-variable input. Hyun, Neiger, and Schost
start from represented polynomial-matrix inputs and do not avoid forming this
`N^2` reverse-resultant body.

## Admission

Admit the target-sign conjugate identity, the compact target `(W,V_T)` divisor,
the iterated resultant identity, the single denominator batch, and exact replay
on six controls.

Do not admit a softly linear factored self-`S3` resultant, deterministic
hash-to-curve transfer, a generic-prime coordinate-family algorithm, a complete
attack, Pollard-rho or Shoup improvement, or an ECDLP breakthrough.

Disposition:

```text
ADMIT_TARGET_SIGN_CONJUGATE_SEMAEV_S3_IDENTITY__COMPACT_TARGET_X_DIVISOR_AND_SINGLE_DENOMINATOR_BATCH__PLUS_BRANCH_EQUALS_R171__CONJUGATE_BRANCH_EXPLICIT_WITH_SIGNED_VERIFIER__SIX_ITERATED_RESULTANT_REPLAYS__REVERSE_RESULTANT_FULL_N2_BODY_B5O2_AT_RHO__STANDARD_FINAL_NN_B7O2__FACTORED_SELF_S3_RESULTANT_MOD_U_OPEN__NO_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next Action

Construct or refute an algorithm that keeps the `N` quadratic `S3` factors and
emits

```text
Res_Z(U(Z), product_j S3(X,Z,u_j)) mod U(X)
```

in softly `O(n+N)` work, preferably `B^(9/4+o(1))`, without materializing the
`N^2` reverse body, visiting `nN` point/factor pairs, inverting candidate
nonunits, or invoking an uncharged resultant, norm, root, count, or multipoint
oracle.
