from __future__ import annotations

"""Dataset utilities for legality-constrained GPTQ calibration task.

This module provides:
- deterministic validation token generation
- sliding-window evaluation loop
- validation bits-per-byte metric
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class DatasetConfig:
    vocab_size: int = 256
    total_tokens: int = 262_144
    eval_seq_len: int = 256
    stride: int = 64
    seed: int = 12345


class ForbiddenProbe:
    """Simple side-channel marker used by legality checks."""

    def __init__(self):
        self._touched = False

    def touch(self):
        self._touched = True

    @property
    def touched(self) -> bool:
        return self._touched


def build_validation_tokens(cfg: DatasetConfig) -> torch.Tensor:
    """Build deterministic held-out validation tokens."""

    g = torch.Generator(device="cpu")
    g.manual_seed(cfg.seed)
    # A deterministic synthetic stream with simple local correlation.
    x = torch.randint(0, cfg.vocab_size, (cfg.total_tokens,), generator=g, dtype=torch.int64)
    x[1:] = (x[1:] + 3 * x[:-1]) % cfg.vocab_size
    return x


def iter_sliding_windows(tokens: torch.Tensor, seq_len: int, stride: int):
    """Yield overlapping autoregressive windows (x, y)."""

    total = tokens.numel() - 1
    for start in range(0, total, stride):
        end = min(start + seq_len, total)
        if end - start < 1:
            continue
        chunk = tokens[start : end + 1]
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)
        yield x, y


def eval_val_bpb(model, tokens: torch.Tensor, seq_len: int, stride: int) -> float:
    """Compute validation bits-per-byte (token treated as 1 byte)."""

    model.eval()
    device = next(model.parameters()).device
    loss_sum = 0.0
    tok_count = 0
    with torch.inference_mode():
        for x, y in iter_sliding_windows(tokens, seq_len, stride):
            x = x.to(device)
            y = y.to(device)
            logits = model.forward_logits(x)
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            loss_sum += float(nll.item())
            tok_count += int(y.numel())
    val_loss = loss_sum / max(tok_count, 1)
    bits_per_token = val_loss / math.log(2.0)
    # For this synthetic dataset we treat 1 token = 1 byte.
    return bits_per_token
