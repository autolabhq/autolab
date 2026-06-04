"""
solve.py — Baseline bit-flip attack: flip sign bits of the largest FC weights.

Ranks the classifier head's FC weights (`fc.weight`) by absolute magnitude,
descending, and flips the sign bit (bit 31 of IEEE-754 float32) of the top K.
The FC row for class c directly controls class c's logit, so sign flips on
large FC entries invert the model's class preferences and drag top-1 accuracy
below the 12% gate — but K must be large compared to what a saliency-ranked
or gradient-guided attack needs.

Edit this file to find smaller flip sets. Allowed imports: numpy, torch, math, struct.
"""

import torch

BASELINE_K = 95  # tuned during calibration; see task.toml baseline


def find_bit_flips(model, images, labels):
    """Return a list of (param_name, flat_index, bit_index) bit-flip triples.

    Args:
        model:  MiniResNet loaded with frozen pretrained weights (eval mode).
        images: float32 tensor of shape (N, 3, 32, 32), CIFAR-10 test subset (normalized).
        labels: int64 tensor of shape (N,), true class labels.

    Returns:
        flips: list of (str, int, int) triples. Each triple XORs one bit of one
               float32 weight in the model's state_dict.
    """
    fc = model.state_dict()['fc.weight'].detach()
    abs_flat = fc.flatten().abs().numpy()
    order = abs_flat.argsort()[::-1]   # descending magnitude
    top = order[:BASELINE_K]
    return [('fc.weight', int(i), 31) for i in top]
