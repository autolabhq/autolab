#!/bin/bash
set -euo pipefail

cp /solution/optimized_select_data.py /app/select_data.py

echo "Running training..."
bash /app/train.sh
