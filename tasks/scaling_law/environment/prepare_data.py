#!/usr/bin/env python3
"""Download and tokenize WikiText-103 for pretraining."""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    out_dir = "/data/wikitext103"
    tok_dir = "/models/tokenizer"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    print("Downloading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(tok_dir)
    print(f"Vocab size: {tokenizer.vocab_size}")

    for split in ["train", "validation", "test"]:
        print(f"Tokenizing {split}...")
        texts = ds[split]["text"]
        # Filter empty lines, concatenate all text
        all_text = "\n".join(t for t in texts if t.strip())
        tokens = tokenizer.encode(all_text)
        tensor = torch.tensor(tokens, dtype=torch.long)
        torch.save(tensor, os.path.join(out_dir, f"{split}.pt"))
        print(f"  {split}: {len(tokens):,} tokens")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
