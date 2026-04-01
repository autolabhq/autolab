#!/bin/bash
set -euo pipefail

echo "=== Step 1: Running data selection ==="
python3 /app/select_data.py

echo "=== Step 2: Converting to LLaMA-Factory format ==="
python3 /app/convert_to_sharegpt.py

# Validate selection size
SELECTED_COUNT=$(python3 -c "import json; print(len(json.load(open('/app/data/selected_train_sharegpt.json'))))")
echo "Selected $SELECTED_COUNT samples"

if [ "$SELECTED_COUNT" -gt 5000 ]; then
    echo "WARNING: Selected more than 5000 samples ($SELECTED_COUNT). Truncating to 5000."
    python3 -c "
import json
data = json.load(open('/app/data/selected_train_sharegpt.json'))[:5000]
json.dump(data, open('/app/data/selected_train_sharegpt.json', 'w'), ensure_ascii=False)
print(f'Truncated to {len(data)} samples')
"
fi

if [ "$SELECTED_COUNT" -eq 0 ]; then
    echo "ERROR: No samples selected. Aborting."
    exit 1
fi

echo "=== Step 3: Training LoRA SFT ==="
llamafactory-cli train /app/train_config.yaml

echo "=== Training complete ==="
echo "LoRA adapter saved to: /app/output/"
