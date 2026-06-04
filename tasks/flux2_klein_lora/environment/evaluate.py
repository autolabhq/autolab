#!/usr/bin/env python3
"""
Evaluate LoRA quality for FLUX.2 klein using a 3-metric composite score:
  score = 0.40 * clip_sim + 0.35 * dino_sim + 0.25 * text_align

Uses musubi-tuner's flux_2_generate_image.py for inference, then computes
CLIP/DINO/text-alignment metrics on the generated images.

Usage:
    python3 evaluate.py [--lora_path /app/output/lora.safetensors] [--output_json result.json]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path


CONCEPT_DIR = "/data/concept"
PROMPTS_FILE = "/app/eval_prompts.txt"
MUSUBI_DIR = "/opt/musubi-tuner/src/musubi_tuner"
DIT_PATH = "/models/dit/flux2-klein-base-9b.safetensors"
VAE_PATH = "/models/vae/ae.safetensors"
TEXT_ENC_PATH = "/models/text_encoder/model-00001-of-00004.safetensors"
MODEL_VERSION = "klein-base-9b"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 4.0
SEED = 12345


def load_prompts(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_reference_images(concept_dir):
    """Load all PNG images from concept directory."""
    images = []
    for p in sorted(Path(concept_dir).glob("*.png")):
        images.append(Image.open(p).convert("RGB").resize((512, 512)))
    return images


def generate_images(lora_path, prompts, save_images_dir=None):
    """Generate images using musubi-tuner's FLUX.2 klein inference script.

    If save_images_dir is set, also copy each successfully generated PNG to that
    directory as gen_{i:03d}.png (so artifacts persist past the temp dir).
    """
    gen_dir = tempfile.mkdtemp(prefix="flux2_eval_")
    images = []

    if save_images_dir:
        os.makedirs(save_images_dir, exist_ok=True)
        with open(os.path.join(save_images_dir, "prompts.txt"), "w") as f:
            for i, p in enumerate(prompts):
                f.write(f"{i:03d}\t{p}\n")

    for i, prompt in enumerate(prompts):
        out_path = os.path.join(gen_dir, f"gen_{i:03d}")
        cmd = [
            "python", f"{MUSUBI_DIR}/flux_2_generate_image.py",
            "--model_version", MODEL_VERSION,
            "--dit", DIT_PATH,
            "--vae", VAE_PATH,
            "--text_encoder", TEXT_ENC_PATH,
            "--prompt", prompt,
            "--image_size", "512", "512",
            "--infer_steps", str(NUM_INFERENCE_STEPS),
            "--seed", str(SEED + i),
            "--fp8_scaled",
            "--save_path", out_path,
            "--output_type", "images",
        ]

        if lora_path and os.path.exists(lora_path):
            cmd.extend(["--lora_weight", lora_path, "--lora_multiplier", "1.0"])

        print(f"  Generating {i+1}/{len(prompts)}: {prompt}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    WARNING: Generation failed: {result.stderr[:200]}")
            # Create a blank image as fallback
            images.append(Image.new("RGB", (512, 512), (128, 128, 128)))
            continue

        # Find the generated image
        gen_images = list(Path(out_path).glob("*.png")) + list(Path(out_path).glob("*.jpg"))
        if gen_images:
            src = gen_images[0]
            images.append(Image.open(src).convert("RGB").resize((512, 512)))
            if save_images_dir:
                try:
                    shutil.copy(src, os.path.join(save_images_dir, f"gen_{i:03d}{src.suffix}"))
                except Exception as e:
                    print(f"    WARNING: Failed to persist image: {e}")
        else:
            print(f"    WARNING: No output image found in {out_path}")
            images.append(Image.new("RGB", (512, 512), (128, 128, 128)))

    torch.cuda.empty_cache()
    return images


def compute_clip_image_similarity(generated, references):
    """Compute average CLIP ViT-B/32 image similarity between generated and reference images."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device="cuda"
    )
    model.eval()

    def encode_images(imgs):
        tensors = torch.stack([preprocess(img) for img in imgs]).to("cuda")
        with torch.no_grad():
            feats = model.encode_image(tensors)
            feats = F.normalize(feats, dim=-1)
        return feats

    gen_feats = encode_images(generated)
    ref_feats = encode_images(references)

    ref_mean = F.normalize(ref_feats.mean(dim=0, keepdim=True), dim=-1)
    sims = (gen_feats @ ref_mean.T).squeeze(-1)
    raw_score = sims.mean().item()

    # Normalize: typical CLIP image-image cosine sim range [0.5, 0.9] -> [0, 1]
    normalized = max(0.0, min(1.0, (raw_score - 0.5) / 0.4))

    del model
    torch.cuda.empty_cache()
    return raw_score, normalized


def compute_dino_similarity(generated, references):
    """Compute average DINOv2 ViT-S/14 similarity between generated and reference images."""
    from torchvision import transforms

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to("cuda").eval()

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def encode_images(imgs):
        tensors = torch.stack([preprocess(img) for img in imgs]).to("cuda")
        with torch.no_grad():
            feats = model(tensors)
            feats = F.normalize(feats, dim=-1)
        return feats

    gen_feats = encode_images(generated)
    ref_feats = encode_images(references)

    ref_mean = F.normalize(ref_feats.mean(dim=0, keepdim=True), dim=-1)
    sims = (gen_feats @ ref_mean.T).squeeze(-1)
    raw_score = sims.mean().item()

    # Normalize: typical DINO image-image range [0.3, 0.8] -> [0, 1]
    normalized = max(0.0, min(1.0, (raw_score - 0.3) / 0.5))

    del model
    torch.cuda.empty_cache()
    return raw_score, normalized


def compute_text_alignment(generated, prompts):
    """Compute CLIP text-image alignment score."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device="cuda"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    img_tensors = torch.stack([preprocess(img) for img in generated]).to("cuda")
    text_tokens = tokenizer(prompts).to("cuda")

    with torch.no_grad():
        img_feats = F.normalize(model.encode_image(img_tensors), dim=-1)
        txt_feats = F.normalize(model.encode_text(text_tokens), dim=-1)

    # Pairwise diagonal: each image matched to its own prompt
    sims = (img_feats * txt_feats).sum(dim=-1)
    raw_score = sims.mean().item()

    # Normalize: typical CLIP text-image range [0.20, 0.35] -> [0, 1]
    normalized = max(0.0, min(1.0, (raw_score - 0.20) / 0.15))

    del model
    torch.cuda.empty_cache()
    return raw_score, normalized


def main():
    parser = argparse.ArgumentParser(description="Evaluate FLUX.2 klein LoRA quality")
    parser.add_argument("--lora_path", type=str, default="/app/output/lora.safetensors",
                        help="Path to LoRA safetensors file")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to write results JSON")
    parser.add_argument("--no_lora", action="store_true",
                        help="Evaluate base model without LoRA")
    parser.add_argument("--save_images_dir", type=str, default=None,
                        help="If set, persist generated images to this directory")
    args = parser.parse_args()

    prompts = load_prompts(PROMPTS_FILE)
    references = load_reference_images(CONCEPT_DIR)
    print(f"Loaded {len(references)} reference images, {len(prompts)} prompts")

    # Generate images
    lora = None if args.no_lora else args.lora_path
    generated = generate_images(lora, prompts, save_images_dir=args.save_images_dir)

    # Compute metrics
    print("\nComputing CLIP image similarity...")
    clip_raw, clip_norm = compute_clip_image_similarity(generated, references)
    print(f"  CLIP sim: {clip_raw:.4f} (normalized: {clip_norm:.4f})")

    print("Computing DINO similarity...")
    dino_raw, dino_norm = compute_dino_similarity(generated, references)
    print(f"  DINO sim: {dino_raw:.4f} (normalized: {dino_norm:.4f})")

    print("Computing text alignment...")
    text_raw, text_norm = compute_text_alignment(generated, prompts)
    print(f"  Text align: {text_raw:.4f} (normalized: {text_norm:.4f})")

    # Composite score
    score = 0.40 * clip_norm + 0.35 * dino_norm + 0.25 * text_norm
    score = round(score, 4)

    result = {
        "score": score,
        "clip_similarity": {"raw": round(clip_raw, 4), "normalized": round(clip_norm, 4)},
        "dino_similarity": {"raw": round(dino_raw, 4), "normalized": round(dino_norm, 4)},
        "text_alignment": {"raw": round(text_raw, 4), "normalized": round(text_norm, 4)},
        "num_prompts": len(prompts),
        "lora_path": args.lora_path if not args.no_lora else None,
    }

    result_json = json.dumps(result, indent=2)
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(result_json)

    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(result_json)
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
