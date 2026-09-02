# P1553 R114 transposed nonuniform C5 leaf-generator gate

## Scope

R114 tests the surviving R113 exception in two standard forms: reverse-mode
transposition after the canonical `C5` membership leaves are evaluated, and
typed source-support propagation before those leaves are formed.

## Exact replay

- All 14 unique-zero instances are localized by the first product gradient.
- Both multiplicity-two instances have zero first gradient and are localized
  exactly by the product Hessian applied to the all-ones direction.
- The resulting adjoint supports return all 18 R113 sources, including R105
  markers and R108 weight 14400.
- Typed preleaf support profiles are frozen at every `C` prefix depth from
  zero through five on all 16 actual and matched instances.

## Cost boundary

Reverse-mode checkpointing can reduce the product-tree state to logarithmic
size, represented by exponent zero, but still evaluates the complete
`B^(3+o(1))` membership-leaf trace. Higher product derivatives handle
multiplicity but do not reduce that primal work.

Moving the transpose before leaf formation begins with `B^2` terminal `A5`
support. Expanding only one `C` atom produces a typed state of exponent
`2 + 3/5 = 13/5`, already above the `B^(9/4)` setup cap and the `B^(5/4)`
fresh-work cap. Five atoms restore `B^5` source occurrences.

This closes standard product reverse AD, first/second derivative
localization, and the canonical typed preleaf support propagation. It does
not prove a lower bound for arbitrary transposed group-algebra circuits.

## Disposition

Preserve no inside-cap locator credit. Rebalance relation arity and
asymmetric factor-base exponents before another toy run, charging relation
supply, summation-polynomial/FFE degree, known-RHS rank, factor logs, and
identical target descent.

No factor-log solve, identical target descent, generic-prime Shoup
improvement, or ECDLP breakthrough is supplied.

## Exactly one next action

Freeze and solve the relation-arity and asymmetric factor-base exponent
feasibility inequalities under the setup and fresh-work caps.
