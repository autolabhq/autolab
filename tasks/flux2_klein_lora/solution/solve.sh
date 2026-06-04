#!/bin/bash
set -euo pipefail

# See solution/reference.md for the reasoning behind these fixes.

cat > /app/dataset.toml << 'DSEOF'
[general]
resolution = 512
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]
image_directory = "/data/concept"
num_repeats = 20
DSEOF

cat > /app/train.sh << 'EOF'
#!/bin/bash
set -euo pipefail

MUSUBI=/opt/musubi-tuner/src/musubi_tuner
DIT=/models/dit/flux2-klein-base-9b.safetensors
VAE=/models/vae/ae.safetensors
TEXT_ENC=/models/text_encoder/model-00001-of-00004.safetensors
DATASET=/app/dataset.toml

python $MUSUBI/flux_2_cache_latents.py \
    --dataset_config $DATASET \
    --vae $VAE \
    --vae_dtype bfloat16 \
    --model_version klein-base-9b

python $MUSUBI/flux_2_cache_text_encoder_outputs.py \
    --dataset_config $DATASET \
    --text_encoder $TEXT_ENC \
    --batch_size 4 \
    --model_version klein-base-9b

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    $MUSUBI/flux_2_train_network.py \
    --model_version klein-base-9b \
    --dit $DIT \
    --vae $VAE \
    --text_encoder $TEXT_ENC \
    --dataset_config $DATASET \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling flux2_shift \
    --weighting_scheme none \
    --optimizer_type adamw8bit \
    --learning_rate 1e-4 \
    --max_train_epochs 16 \
    --lr_scheduler cosine_with_restarts \
    --lr_warmup_steps 50 \
    --gradient_accumulation_steps 4 \
    --vae_dtype bfloat16 \
    --gradient_checkpointing \
    --blocks_to_swap 8 \
    --network_module networks.lora_flux_2 \
    --network_dim 32 \
    --network_alpha 16 \
    --save_every_n_epochs 4 \
    --seed 42 \
    --output_dir /app/output --output_name lora
EOF

chmod +x /app/train.sh
bash /app/train.sh
python3 /app/evaluate.py
