# ResNet Bit-Flip Attack — Minimum Flip Count

Optimize `find_bit_flips()` in `/app/solve.py` to find the **fewest bit flips**
that drive a frozen CIFAR-10 ResNet's top-1 test accuracy **below 12%**.

## Problem

A small pre-trained ResNet (`MiniResNet`, ~77K float32 parameters, test accuracy
~78%) classifies the 10 000-image CIFAR-10 test set. The target of the attack
is the **model's weights**, not the input images: the agent returns a list of
bit-level edits to apply to the float32 parameters. After the edits are applied,
the model's accuracy on the full test set must be strictly below 12%.

```python
def find_bit_flips(model, images, labels):
    """
    Args:
        model:  MiniResNet loaded with the frozen pretrained weights (eval mode).
        images: float32 tensor of shape (1000, 3, 32, 32), CIFAR-10 test sample
                (already normalized with standard CIFAR stats).
        labels: int64 tensor of shape (1000,), true class labels.

    Returns:
        flips: list of (param_name: str, flat_index: int, bit_index: int)
               Each triple XORs bit `bit_index` of the uint32 reinterpretation
               of state_dict[param_name].view(-1)[flat_index] (a float32 scalar).
    """
```

The harness deduplicates the returned list, validates every entry, and applies
the flips to a **fresh** copy of the frozen state dict. The metric is
`bits_flipped = len(unique(flips))` — **lower is better**.

If the resulting model's accuracy is ≥ 12%, the attack does not count and the
score is 0.

## Files

| File | Role |
|------|------|
| `solve.py` | **Edit this file only** |
| `model.py` | `MiniResNet` architecture (read-only) |
| `main.py` | Harness: selects sample, calls `find_bit_flips`, applies flips, evaluates full test set (read-only) |
| `train_model.py` | Training script run once at build time (read-only) |
| `model.pth` | Frozen weights (read-only) |
| `/data/cifar10` | CIFAR-10 dataset (read-only) |

## Rules

- Edit `/app/solve.py` only.
- Allowed imports: `numpy`, `torch`, `math`, `struct`. No other libraries.
- The model architecture and weights are fixed — do not modify `model.py`,
  `main.py`, `train_model.py`, or `model.pth`.
- At most **10 000** flips may be returned (duplicates are collapsed). Each flip
  must reference an existing float32 **learnable parameter**
  (`model.named_parameters()`); BatchNorm running statistics and other buffers
  are not flippable. `bit_index` must be in `[0, 31]`.
- The flipped model must produce finite outputs; weights that become NaN/Inf
  invalidate the attack.
- **Magnitude cap:** after the flips are applied, every float32 weight must
  satisfy `|w| ≤ 10`.
- No internet access.
- Time budget: 2 hours
