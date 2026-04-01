"""Convert selected data from tulu messages format to LLaMA-Factory sharegpt format.

Input:  /app/data/selected_train.json         (tulu: role/content)
Output: /app/data/selected_train_sharegpt.json (LLaMA-Factory: from/value)
"""

import json

ROLE_MAP = {"user": "human", "assistant": "gpt", "system": "system"}

INPUT_PATH = "/app/data/selected_train.json"
OUTPUT_PATH = "/app/data/selected_train_sharegpt.json"

with open(INPUT_PATH) as f:
    selected = json.load(f)

converted = []
for sample in selected:
    conversations = []
    for msg in sample["messages"]:
        role = ROLE_MAP.get(msg["role"], msg["role"])
        conversations.append({"from": role, "value": msg["content"]})
    converted.append({"conversations": conversations})

with open(OUTPUT_PATH, "w") as f:
    json.dump(converted, f, ensure_ascii=False)

print(f"Converted {len(converted)} samples to sharegpt format -> {OUTPUT_PATH}")
