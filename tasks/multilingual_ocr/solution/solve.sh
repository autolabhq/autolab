#!/bin/bash
set -euo pipefail

# Reference solution: optimize hyperparameters for multilingual OCR
# Baseline is already correct (image_size=640, crop_mode=True, train_on_responses_only=True)
# but uses suboptimal training hyperparams (LoRA r=16, 60 steps, lr=2e-4, linear schedule).
#
# Optimizations:
# 1. Increase LoRA rank: 16 → 128 (much more capacity for two scripts)
# 2. More training steps: 60 → 700 (many epochs over bilingual data)
# 3. Lower learning rate: 2e-4 → 1e-4 (more stable convergence)
# 4. Cosine scheduler (better than linear for longer training)
# 5. More warmup steps: 5 → 20

sed -i 's/r=16, lora_alpha=16/r=128, lora_alpha=128/' /app/train.py
sed -i 's/max_steps=60/max_steps=700/' /app/train.py
sed -i 's/learning_rate=2e-4/learning_rate=1e-4/' /app/train.py
sed -i 's/lr_scheduler_type="linear"/lr_scheduler_type="cosine"/' /app/train.py
sed -i 's/warmup_steps=5/warmup_steps=20/' /app/train.py

echo "Running training..."
bash /app/train.sh
