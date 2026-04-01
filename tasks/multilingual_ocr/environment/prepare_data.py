"""
Prepare Persian + Bengali OCR datasets for training and evaluation.

Creates:
1. /data/ocr_train/ — mixed Persian+Bengali training image-text pairs
2. /orig/data/ocr_eval.json — held-out evaluation pairs (both languages)
"""

import json
import os
import random

from datasets import load_from_disk, load_dataset, Dataset
from PIL import Image


def main():
    random.seed(42)
    os.makedirs("/orig/data/ocr_eval_images", exist_ok=True)
    os.makedirs("/data/ocr_train_images", exist_ok=True)

    # ── Load Persian data ──
    print("Loading Persian OCR data...")
    ds_fa = load_from_disk("/data/parsynth_ocr_5k")
    fa_examples = list(ds_fa)
    random.shuffle(fa_examples)

    # ── Load Bengali data ──
    print("Loading Bengali OCR data...")
    ds_bn = load_from_disk("/data/bengali_ocr_5k")
    bn_examples = list(ds_bn)
    random.shuffle(bn_examples)

    # Split: 4500 train + 200 eval per language = 9000 train, 400 eval total
    fa_train = fa_examples[:4500]
    fa_eval = fa_examples[4500:4700]
    bn_train = bn_examples[:4500]
    bn_eval = bn_examples[4500:4700]

    # ── Training data (mixed) ──
    print("Preparing training data...")
    train_records = []
    for i, ex in enumerate(fa_train):
        img_path = f"/data/ocr_train_images/fa_{i:04d}.png"
        img = ex["image_path"]
        if isinstance(img, Image.Image):
            img.save(img_path)
        train_records.append({
            "image_path": img_path,
            "text": ex["text"],
            "language": "fa",
        })

    for i, ex in enumerate(bn_train):
        img_path = f"/data/ocr_train_images/bn_{i:04d}.png"
        img = ex["image"]
        if isinstance(img, Image.Image):
            img.save(img_path)
        train_records.append({
            "image_path": img_path,
            "text": ex["text"],
            "language": "bn",
        })

    random.shuffle(train_records)
    train_ds = Dataset.from_list(train_records)
    train_ds.save_to_disk("/data/ocr_train")
    print(f"  -> {len(train_records)} training examples ({sum(1 for r in train_records if r['language']=='fa')} fa, {sum(1 for r in train_records if r['language']=='bn')} bn)")

    # ── Eval data ──
    print("Preparing evaluation data...")
    eval_records = []
    for i, ex in enumerate(fa_eval):
        img_path = f"/orig/data/ocr_eval_images/fa_{i:04d}.png"
        img = ex["image_path"]
        if isinstance(img, Image.Image):
            img.save(img_path)
        eval_records.append({
            "image_path": img_path,
            "text": ex["text"],
            "language": "fa",
        })

    for i, ex in enumerate(bn_eval):
        img_path = f"/orig/data/ocr_eval_images/bn_{i:04d}.png"
        img = ex["image"]
        if isinstance(img, Image.Image):
            img.save(img_path)
        eval_records.append({
            "image_path": img_path,
            "text": ex["text"],
            "language": "bn",
        })

    with open("/orig/data/ocr_eval.json", "w") as f:
        json.dump(eval_records, f, ensure_ascii=False)
    print(f"  -> {len(eval_records)} evaluation examples")

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
