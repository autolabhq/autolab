#!/usr/bin/env python3
"""
Download and prepare training data for FLUX.2 klein LoRA concept learning.
Uses 15 images from lambdalabs/naruto-blip-captions dataset.
Creates caption .txt files alongside each image with trigger word.
"""

import os

CONCEPT_DIR = "/data/concept"
NUM_IMAGES = 15
TRIGGER = "naruto style"


def main():
    os.makedirs(CONCEPT_DIR, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset("lambdalabs/naruto-blip-captions", split=f"train[:{NUM_IMAGES}]")

    for i, item in enumerate(ds):
        img = item["image"].convert("RGB").resize((512, 512))
        img_path = os.path.join(CONCEPT_DIR, f"{i:03d}.png")
        img.save(img_path)

        # Create caption file with trigger word
        caption = f"illustration in {TRIGGER}"
        txt_path = os.path.join(CONCEPT_DIR, f"{i:03d}.txt")
        with open(txt_path, "w") as f:
            f.write(caption)

        print(f"  Saved {img_path} + caption")

    print(f"Prepared {NUM_IMAGES} training images in {CONCEPT_DIR}")


if __name__ == "__main__":
    main()
