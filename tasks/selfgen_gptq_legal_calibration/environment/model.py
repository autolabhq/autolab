from __future__ import annotations

"""Tiny language model used for quantization benchmark."""

import torch
from torch import nn


class TinyLM(nn.Module):
    """Small residual MLP language model over byte tokens."""

    def __init__(self, vocab_size: int = 256, d_model: int = 128, n_layers: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 4 * d_model, bias=False),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model, bias=False),
                )
                for _ in range(n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for each sequence position."""

        x = self.tok_emb(token_ids)
        for blk in self.blocks:
            x = x + blk(x)
        x = self.final_ln(x)
        return self.lm_head(x)

    def forward(self, token_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Return mean cross-entropy loss for convenience."""

        logits = self.forward_logits(token_ids)
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="mean",
        )
