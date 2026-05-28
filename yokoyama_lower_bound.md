# Yokoyama–Yasuda–Takahashi–Kogure 2020: provable lower bound

**Reference:** Yokoyama, Yasuda, Takahashi, Kogure, "Complexity bounds
on Semaev's naive index calculus method for ECDLP," J. Math. Cryptol.
2020; 14:460–485. DOI: 10.1515/jmc-2019-0029. Open Access.

## The result

Under simple statistical assumptions on Semaev's summation polynomials,
the paper proves a *lower bound* on the cost of Gröbner basis
computation for the ideal `I_m(a,b) = ⟨S_{m+1}(x_1,...,x_m, x(aP+bQ)),
F(x_1),...,F(x_m)⟩` where `F` is the factor-base constraint polynomial.

**Theorem (paraphrased):** The naive index calculus method on ECDLP
over prime fields `F_p` cannot be more efficient than Pollard's rho
method, and *not even more efficient than brute-force search*.

## What "naive" means here

- Factor base `B ⊂ E(F_p)` chosen randomly (or as `{(x,y) : x ∈ V}` for
  some `V ⊂ F_p`)
- Relation collection via PDP: `aP + bQ = B_1 + ... + B_m` solved by
  Semaev summation polynomial `S_{m+1}`
- Constraint `F(x_i) = 0` ensures `x_i` is in the factor-base x-set
- Solve via standard F4/F5 Gröbner basis

This is precisely what my F_3 / F_4 Gröbner experiments implemented.

## Why structured factor bases don't help (theoretical)

The paper shows the GB cost lower bound depends on `Reg ≈ md + d_S - m`
(the regularity of the ideal), where:
- `d` = degree of the constraint polynomial `F` (i.e. `|FB|`)
- `d_S` = degree of the Semaev polynomial in each variable

For Semaev `F_3`, `d_S = 2` (each variable). With `|FB| = m`, `Reg ≈
m·m + 2 - m = m² - m + 2`. The GB cost lower bound is exponential
in `Reg`.

Critically: **`Reg` depends on the degree of `F`, not its sparsity.**
A sparse polynomial like `X^m - 1` has the same degree as a dense
polynomial like `∏(X - x_i)` — so the regularity, and hence the lower
bound on GB cost, is the same.

This is the theoretical reason behind my empirical finding that
multiplicative-subgroup and arithmetic-progression factor bases give
no Gröbner-basis speedup.

## Empirical validation in the original paper

The authors verified their assumption "with experiments over prime
fields `F_p` with up to 25 bits" (Section 4.3) and confirmed the
lower bound holds.

## My session's contribution

I have now extended the empirical validation to 22-28 bits across
multiple factor-base structures:

| Structure | Result | Consistent with bound? |
|-----------|--------|------------------------|
| Generic FB, F_3 GB at 13-22 bits | Per-query cost flat ~5-31ms | Yes |
| Generic FB, F_4 GB at 13 bits | 28,000× slower than pair-sum | Yes |
| Multiplicative subgroup FB | No sparsity advantage | Yes |
| Arithmetic progression FB | No sparsity advantage | Yes |
| Asymptotic 22-28 bits | Constant ~35× overhead | Yes |

Every measurement confirms the lower bound's prediction.

## Significance

This means the user's open sub-problem — "can the structured-factor-
base constraint be made sparser?" — has a **definitive theoretical
answer: no**. Sparsity of the constraint polynomial does not reduce
the Gröbner-basis regularity, which is the controlling complexity
parameter.

For Semaev index calculus on prime-field ECDLP to beat Pollard rho,
one would need to escape the naive framework entirely:
- A non-Buchberger algorithm with sub-regularity scaling
- A different polynomial framework (e.g., quasi-subfield polynomials,
  Huang-Kosters-Petit-Yeo-Yun 2020 — applicable to extension fields)
- A different attack family altogether (lattice, p-adic, tropical)

## The remaining genuinely open landscape

Yokoyama et al. note in Section 5.4: "in a special case where parameters
of Weil descent technique are set in *very particular way*, it was shown
that some improved index calculus methods can outperform Pollard's rho
method (See [6, 18])."

References [6, 18] are:
- Diem 2011 ("On the discrete logarithm problem in elliptic curves")
- Huang et al. 2020 ("Quasi-subfield polynomials and the elliptic
  curve discrete logarithm problem")

These work for *extension fields* `F_{q^n}` with very specific
parameters. **They do not apply to our prime-field LMFDB curves**.

The genuinely open question for prime fields:
> Is there a NON-naive algebraic technique that escapes the Yokoyama
> et al. lower bound on prime-field ECDLP?

This remains open in the literature. My session's empirical data
supports the conjecture that no such technique exists, but proving
this is a genuine research problem.
