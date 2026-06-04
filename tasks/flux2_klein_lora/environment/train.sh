#!/bin/bash
set -euo pipefail

MUSUBI=/opt/musubi-tuner/src/musubi_tuner
DIT=/models/dit/flux2-klein-base-9b.safetensors
VAE=/models/vae/ae.safetensors
TEXT_ENC=/models/text_encoder/model-00001-of-00004.safetensors
DATASET=/app/dataset.toml

echo "============================================"
echo "  FLUX.2 klein LoRA Training Pipeline"
echo "============================================"

# ── Step 1: Cache latents ──
echo ""
echo "[Step 1/3] Caching latents..."
python $MUSUBI/flux_2_cache_latents.py \
    --dataset_config $DATASET \
    --vae $VAE \
    --vae_dtype bfloat16 \
    --model_version klein-9b

# ── Step 2: Cache text encoder outputs ──
echo ""
echo "[Step 2/3] Caching text encoder outputs..."
python $MUSUBI/flux_2_cache_text_encoder_outputs.py \
    --dataset_config $DATASET \
    --text_encoder $TEXT_ENC \
    --batch_size 4 \
    --model_version klein-9b

# ── Step 3: Train LoRA ──
echo ""
echo "[Step 3/3] Training LoRA..."
echo "Model:     FLUX.2 klein 9B"
echo "Data:      15 naruto concept images × 10 repeats"
echo ""

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    $MUSUBI/flux_2_train_network.py \
    --model_version klein-9b \
    --dit $DIT \
    --vae $VAE \
    --text_encoder $TEXT_ENC \
    --dataset_config $DATASET \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift \
    --weighting_scheme none \
    --optimizer_type adamw8bit \
    --learning_rate 1e-3 \
    --max_train_epochs 50 \
    --lr_scheduler constant \
    --vae_dtype float32 \
    --network_module networks.lora_flux_2 \
    --network_dim 2 \
    --network_alpha 1 \
    --save_every_n_epochs 10 \
    --seed 42 \
    --output_dir /app/output --output_name lora

echo ""
echo "============================================"
echo "  Training complete"
echo "============================================"
echo "LoRA saved to /app/output/"
ls -la /app/output/
