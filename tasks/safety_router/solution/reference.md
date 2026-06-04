# Safety Router — Reference

## Background
Safety routers gate whether an LLM answers or refuses an incoming prompt. Production routers are usually small encoder networks attached to a frozen feature extractor; the open question is how compact such a router can be while still satisfying behavior constraints (overall accuracy, unsafe-recall floor, safe-recall floor) on a held-out split. This task fixes the input feature pipeline (128-dim dense vectors precomputed offline) and scores submissions by trainable parameter count, conditional on passing the verifier's three behavior thresholds. The benchmark targets robustness under aggressive size pressure rather than wall-clock speed or representation learning.

## Baseline Approach (16 641 parameters)
A standard 2-layer MLP with `hidden_dim=128`, ReLU activation, weighted BCE loss, and threshold selection on the validation split. With 128 input features, this gives `128*128 + 128 + 128 + 1 = 16 641` trainable parameters. The architecture is intentionally over-parameterized for a binary classifier on dense pre-projected features and easily clears the verifier behavior thresholds, which makes it a generous starting point but a long way from the achievable frontier.

## Possible Optimization Directions
1. **Reduce hidden width** — the input feature space is already SVD-compressed to 128 dims, so the hidden layer can be far narrower than the input. Width 16 still has comfortable representational capacity.
2. **Linear or near-linear router** — for many feature spaces a single `Linear(128, 1)` (129 params) is enough; only switch to a hidden layer if the linear model fails the unsafe-recall floor.
3. **Low-rank factorization** — replace `W1 (128 x H)` with `U (128 x r) @ V (r x H)` for small `r`, reducing parameter count when `H` is non-trivial.
4. **Feature selection then retrain** — pick the top-k SVD components by validation-set mutual information with the label, then train a tiny model only on those components.
5. **Sparse parameterization with fixed masks** — keep `hidden_dim` modest but mask out a large fraction of weights; only the unmasked weights count as trainable.
6. **Threshold tuning** — once accuracy is borderline, sweep the decision threshold on validation to trade off unsafe-recall against safe-recall and recover headroom on the binding constraint.

## Reference Solution (~2 100 parameters)
**Key insight**: the input features are already a strong representation, so the router only needs enough capacity to carve a near-linear decision boundary plus a small non-linearity to handle borderline buckets.

A 2-layer MLP with `hidden_dim=16` gives `128*16 + 16 + 16 + 1 = 2 081` trainable parameters and reliably passes the verifier's accuracy / unsafe-recall / safe-recall thresholds on the private split. Training reuses the baseline recipe (weighted BCE, validation threshold sweep over `[0.35, 0.70]`) with a slightly higher positive class weight (~1.2) to keep unsafe-recall stable as capacity drops. This is roughly an 8x reduction over the baseline and a comfortable target for a first pass.

The achievable frontier is much smaller than this reference: agents have demonstrated solutions in the low-hundreds of parameters by aggressively narrowing the hidden layer (down to `hidden_dim=1`, ~131 params) or by replacing the MLP with a tuned linear head plus a learned threshold. Treat this section as a starter point, not the floor.

## Data Construction

**Data layout**
- `environment/data/{train,val,test_public}.npz` hold precomputed dense features (`X`) and binary labels (`y`): `0` = answer, `1` = refuse.
- The private verifier split lives in `tests/data/test_private.npz` with prompt metadata in `tests/data/prompt_bank_private.jsonl`.

**Source datasets** (mixed, with explicit labels)
- `Paul/XSTest`
- `JailbreakBench/JBB-Behaviors`
- `PKU-Alignment/BeaverTails-Evaluation`
- `lmsys/toxic-chat` (using `human_annotation` as the label source)

**Feature pipeline**
- `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=12000)` fit on the training split.
- `TruncatedSVD` projection to 128 dims (`random_state=2026`).

## Sources
- Han, Mao & Dally, *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding* (ICLR 2016) — parameter compression under accuracy constraints
- Frankle & Carbin, *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks* (ICLR 2019) — sparse subnetwork extraction
- Tishby & Zaslavsky, *Deep Learning and the Information Bottleneck Principle* (ITW 2015) — low-dimensional sufficient features for classification
