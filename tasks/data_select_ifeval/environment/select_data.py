"""Data selection script: selects training samples from the pool.

Input:  /data/pool.json          (~50k samples, tulu messages format)
Output: /app/data/selected_train.json  (max 5000 samples, same format)
"""

import json
import random

random.seed(0)

MAX_SELECTED = 5000

with open("/data/pool.json") as f:
    pool = json.load(f)

print(f"Pool size: {len(pool)}")

selected = random.sample(pool, min(MAX_SELECTED, len(pool)))

with open("/app/data/selected_train.json", "w") as f:
    json.dump(selected, f, ensure_ascii=False)

print(f"Selected {len(selected)} samples (random baseline)")
