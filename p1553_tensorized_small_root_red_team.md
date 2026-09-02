# Independent P1553 tensorized small-root R7 red-team transcript

Reviewer: `019f7c7a-2cb9-75a3-b012-8c93e3377e23`
Record: coordinator transcription of the independent theorem-only response
Evidence: read-only review, no run

## Terminal verdict

```text
REVISE_SCOPED_CONDITIONAL_BACKEND__EXACT_IMPLICIT_CP_GRAM_LLL_CONSTRUCTED_FOR_SUPPLIED_SMALL_SHIFT_FAMILY__LITERAL_POWER_RANK_BINOMIAL__OUTPUT_RANK_BOUNDED_BY_TOTAL_INPUT_A__NAIVE_CAP_A_LE_B1_OVER_8__P_K_CENTERING_CARRY_UNBOUNDED__STANDARD_CANCELLATION_QUOTIENT_B5__NO_FIVE_USEFUL_SHORT_OUTPUTS__STANDARD_EXACT_RECOVERY_B5_OR_B3__NO_EXACT_QUERY2P1__NO_P1554__NO_RUN__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Accepted positive construction

Choose a constant-rank separated integer lift `tilde(F_R)` of the exact R6
predicate. For

```text
g_(k,alpha)=p^(m-k) z^alpha tilde(F_R)^k,
```

every modular root makes `g_(k,alpha)` divisible by `p^m`, and

```text
rank_CP(g_(k,alpha)) <= binomial(r+k-1,k).
```

Integer reduction modulo each domain polynomial `Q_i` is factorwise,
preserves legal-label values, and retains CP separation. For CP generators,
the bound-scaled Gram entry is

```text
G_(j,l)=sum_(a,b) product_i
          sum_d u_(j,a,i,d)u_(l,b,i,d)B_i^(2d).
```

Gram-form LLL therefore needs no `B^5` coefficient vector. If it returns
combination matrix `U`, retaining `h_j=sum_l U_(j,l)g_l` bounds each output
rank by the total input rank `A`. If its certified norm satisfies the
Howgrave-Graham threshold, the desired modular root is rigorously an integer
root of `h_j`.

The naive pairwise implementation costs `B^(2+o(1))A^2` setup bit work and
`B^(1+o(1))A^2` target work, forcing `A<=B^(1/8+o(1))` under both caps. This
bound is scoped to that implementation.

## Carry and shortness gate

The uncentered separated lift is a valid lattice input, so a low-rank centered
carry is sufficient but not necessary. If recentered after the `k`th power,
the correct modulus is `p^k`:

```text
ctr_(p^k)(tilde(F_R)^k)=tilde(F_R)^k-p^k K_k.
```

Modular CP rank does not bound the integer rank of `K_k`. The natural
separated lift has root-scaled coefficients of size `B^(Omega(B))` on
full-degree decks; multiplying its `k=1` row by `p^(m-1)` does not improve its
ratio to the `p^m` short-vector threshold. Cancellations among shifts are
therefore load bearing.

The conventional cancellation module is the integer label-domain quotient

```text
Z[z_1,...,z_5]/(Q_1,...,Q_5),
```

of rank `product_i s_i=Theta(B^5)`. No cap-sized target-symbolic submodule is
constructed that contains enough cancellations for five useful short outputs.
This is a representation-specific over-budget object, not a general tensor or
lattice lower bound.

## Recovery gate

Five certified short outputs may remain algebraically dependent, lie in the
domain ideal, or have extraneous common roots. The standard exact quotient or
Groebner/resultant recovery has `B^5` states. The balanced finite-grid route
first materializes a source-faithful third-label table or norm of size `B^3`.
Both exceed the online `B^(5/4)` cap. Exact bisection needs either exhaustive
extraneous-root handling or a source-relevant existence certificate.

The favorable `B^(9/4)` relation campaign, `B^2` factor-log algebra, and
`B^(5/4)` descent remain conditional. Falling back to `B^3` source recovery
gives `B^4` relation collection and `B^3` descent.

Ryan's 2025 construction optimizes explicit shift spans; its lower-rank graph
selection and multivariate recovery remain heuristic. It does not supply the
missing tensor-native exact source locator.

## Exactly one next action

Prove or refute one target-symbolic quotient-lattice module whose literal-power
shifts preserve `p^m` divisibility, have total separated rank
`A<=B^(1/8+o(1))`, yield five useful Howgrave-Graham-short outputs, and support
exact subset-stable decision-and-source recovery inside `B^(5/4+o(1))`.
