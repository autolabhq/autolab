# ResNet Bit-Flip Attack — Reference

## Background

Deep neural networks store their parameters as IEEE-754 float32 numbers. Flipping
a single bit of a weight — especially the sign bit (the MSB) — can change the
sign or magnitude of that weight by orders of magnitude. *Deep Neural Lesion*
(arXiv:2502.07408) showed that on ImageNet ResNet-50, **a single sign-bit flip
on a well-chosen high-magnitude weight drops top-1 accuracy from ~76% to ~1%**,
and a handful of such flips collapse a wide range of vision models. This is a
**weight-perturbation** attack (the attacker touches the model, not the input).

This puzzle asks the agent to find the **minimum number of bit flips** that
drives a pretrained CIFAR-10 MiniResNet's top-1 test accuracy below 12% — a
couple of points below the 10% random-chance floor on a balanced 10-class
dataset, so the model must be broken *beyond* a mere class collapse. Any bit of
any float32 weight may be flipped; in practice sign-bit flips produce the
strongest effects while keeping activations finite (exponent-bit flips on
high-magnitude weights tend to overflow to ±inf and propagate NaN through
BatchNorm, which the harness rejects).

## Baseline Approach

The baseline ranks the classifier head's FC weights (`fc.weight`, 640 entries)
by absolute magnitude and flips the sign bit of the top K. Each flipped entry
inverts one of the class-vs-hidden-unit preferences, so enough of them drive the
softmax toward wrong classes. The attack is trivial to write but very wasteful —
it only ever touches one layer and ignores how useful a given sign flip actually
is for increasing loss, so K needs to be ~95 for the MiniResNet to drop below
12% accuracy on this task.

## Possible Optimization Directions

1. **Saliency-weighted selection (1P-DNL)** — rank parameters by
   `|w| × |∂L/∂w|` after one forward + backward pass on a sample batch. This
   surfaces weights whose sign directly swings the loss, rather than weights
   that merely happen to be large. On this MiniResNet it drops the required
   flip count from ~95 to ~40.

2. **Target every tensor, not just the FC** — convolutional weights have far
   more reach (each kernel entry affects a whole feature map) but also lots of
   low-impact entries. Mixing FC flips with a few early-conv flips found by
   saliency is the win.

3. **Iterative / greedy search** — flip one bit, re-evaluate loss or sample
   accuracy, pick the next flip with the largest marginal effect. Avoids the
   redundancy a one-shot top-K ranking can have (many "good" flips whose
   effects overlap). More expensive per flip, but can produce smaller attack
   sets.

4. **Targeted class suppression** — for each class c, find and flip the FC
   weights whose sign reduces that class's logit the most. Fewer structural
   changes, larger effect on per-class accuracy.

5. **Non-sign bits** — the harness caps every post-flip weight at `|w| ≤ 100`,
   which rules out exponent-bit flips that blow small values up to ~10³⁸ (and
   would otherwise let a single flip on `fc.bias[i]` dominate every logit).
   Mantissa flips are almost always ignorable. Sign flips are the only bit
   class that produces a meaningful attack per flip on this model.

## Reference Solution

`solve_optimized.py` runs the 1P-DNL saliency recipe: one forward/backward pass
on the sample batch to get gradients, then rank every 2D+ float32 weight by
`|w| × |∂L/∂w|` and flip the sign bit of the top K. This lands the reference
attack at 40 flips with ~11.7% test accuracy.

## Source

Guzman, J.L., Vashisth, P., Tessenyi, D., Abdel Magid, E., Sadaphal, V.,
Thomas, A., Khan, M.S. & Park, D.S. (2025). *Deep Neural Lesion: Class-wise
Network Attacks on Deep Neural Networks.* arXiv:2502.07408.
