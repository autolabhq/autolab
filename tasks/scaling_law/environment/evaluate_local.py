#!/usr/bin/env python3
"""Local evaluation script — check validation perplexity before submitting."""

import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial

from litgpt import GPT, Config


class TokenDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        self.n_chunks = len(data) // (seq_length + 1)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * (self.seq_length + 1)
        return self.data[start : start + self.seq_length + 1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "/app/output/checkpoint.pt"

    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = ckpt["model_config"]
    print(f"Model config: {model_config}")
    print(f"Training step: {ckpt.get('step', '?')}")
    print(f"Saved val PPL: {ckpt.get('val_ppl', '?')}")
    print(f"Parameters: {ckpt.get('n_params', '?'):,}")

    # Reconstruct model
    config = Config(**model_config)
    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Evaluate with fixed seq_length=1024 (same as verifier)
    eval_seq_length = 1024
    print(f"\nEvaluating on validation set with seq_length={eval_seq_length}")

    val_data = torch.load("/data/wikitext103/validation.pt", weights_only=True)
    val_dataset = TokenDataset(val_data, eval_seq_length)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    print(f"\nValidation loss: {avg_loss:.4f}")
    print(f"Validation perplexity: {ppl:.2f}")

    # Show expected reward (multiplier: reference / score)
    reference_ppl = 23.0
    reward = reference_ppl / ppl if ppl > 0 else 0.0
    print(f"\nEstimated reward: {reward:.4f}")
    print(f"  (reference={reference_ppl}, reward=1.0 means matched reference)")


if __name__ == "__main__":
    main()
