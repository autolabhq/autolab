# P1436 autoresearch focus harness V78 result

Date: 2026-07-29

## Result

V78 binds R129 as the 65th closed frontier lane and routes the highest
priority action to `s21_shared_predicate_torus_c5_selector_dag`. The
report remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R129 studies piecewise-constant C2 translate branches in the inherited
injective source-monomial model. With

```text
n = B^(3/4+o(1))
```

deck atoms, every pure fifth power forces its square C2 branch. Cross-term
branches must then hit every square-free five-subset. The graph of
unselected cross terms is `K5`-free, so Turan's theorem gives the exact
minimum

```text
Lmin = n + binom(n,2) - ex(n,K5)
     = Theta(n^2)
     = B^(3/2+o(1)).
```

A matching construction partitions atoms into four balanced parts and
selects all squares and all within-part pairs.

With one shared exact C3 dictionary, the setup fits
`B^(9/4+o(1))`, but sequentially scanning the optimal branch family costs
`B^(3/2+o(1))` per arbitrary target. An explicit target-to-branch router
costs `B^(15/4+o(1))` state.

All eight actual controls have injective C2, C3, and C5 source products.
For deck sizes `3, 5, 6, 7`, exhaustive minimization agrees with exact
branch counts `3, 6, 8, 10`. Every positive target returns a replayable
C2+C3 source; zero is rejected after scanning every selected branch. No
pairing-image discrete logarithm is consumed and finite controls receive no
asymptotic credit.

Branch count is not a decision-DAG query lower bound. A balanced DAG could
have logarithmic depth if its union predicates are compact and exact.
Likewise, the inherited degree bound forces only `Omega(log B)`
multiplication depth in an unrestricted SLP. Compact shared-predicate DAGs
and high-degree low-SLP selectors remain open. No relation-rank
construction, factor logs, identical target descent, Pollard-rho
improvement, Shoup improvement, or ECDLP breakthrough is claimed.

## Verification

- Focused R129 tests: 10 passed.
- Full harness tests: 95 passed.
- Full ECDLP suite: 529 passed.
- R129 clean rerun: all six JSON outputs byte-identical.
- Harness clean rerun against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R129: 54 receipts, 964 bindings, 0 mismatches.
- V78 frontier preflights: 65 provided, 65 closed.
- R129 breakthrough, Shoup-improvement, and rho-improvement claims: false.
- V78 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R129 producer: `0296c7a71284b8d376720791eb73a768fe6446ea90ac0ce29a7b6a3b9435f1f5`
- R129 report: `3ff1da9d4eb76fe023432492ad0ad6452d1aa69dd51cea98e6469b3191de9402`
- R129 frozen interface: `a10da885fbae4070e5a88591d53e1e716e40b26f31848661980aeb13a8e49e3d`
- R129 cost ledger: `2948aea2107d53d5c9b700f20611c28f28c3a3933378510c9d625b80272334b6`
- R129 source replay: `cb3f53214494107fe9b81cc73a11131da23743fa65c290d19d376236c4593bd4`
- R129 controls: `72d32caf2ee1fa748fc90481cf8061bec62e37e6b61f598e83b77e841d39aaa0`
- R129 logs/descent: `d4d28cc412d139ed9f9dc25a050b43cce225a9c0a87dca307b6a955f04a37a40`
- R129 tests: `a8c34c8d3f173328390735d2cae1a5172436c1cd9b99712aaa99860e3903c621`
- R129 gate: `35026763fe6805e22935bb8f62480609f67a1f7e0c303c67f3d302a14d853c32`
- R129 parent: `b9f9d21e9572e9978511b34e01a90dc4b9df62087aa8f915342386610783970e`
- harness: `30117d9852c61293538ddf09d65c795d949172e0a4518c1b8cf0ee4a567bb623`
- harness tests: `a9c3ec1c62b247ca658e9dabcf3507a121f26897637ee2d1eda56c586e83fe60`
- report: `15835283593c5037bd7705af4dc317d0f215fcceb0eea7ac7894ccd992b53cdb`
- note: `b01ff78b35089dc9a604a69f2d85adf8156d4308d434606216e302b152e58140`
- inventory: `21fa6c6b0624bf9074179a3c5cb77d3bbe3ab0b51b9364a89fafbc63db34179f`
- replay plan: `684c2dfb81553c6636e706c052a2c07cf19e29d0fba7c079d471bc1853f84d9e`
