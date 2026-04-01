"""Source-aware data selection for IFEval optimization."""

import json
import random
from collections import Counter

random.seed(42)

MAX_SELECTED = 5000

with open("/data/pool.json") as f:
    pool = json.load(f)

source_counts = Counter(s["source"] for s in pool)
print("Pool source distribution:")
for src, cnt in source_counts.most_common():
    print(f"  {src}: {cnt}")

by_source: dict[str, list] = {}
for sample in pool:
    by_source.setdefault(sample["source"], []).append(sample)

PRIORITY = [
    ("ai2-adapt-dev/tulu_v3.9_persona_ifeval_turn_1", 2500),
    ("ai2-adapt-dev/coconot_sft_v2.0", 1000),
    ("ai2-adapt-dev/no_robots_sft", 500),
    ("ai2-adapt-dev/wildchat_gpt-4_sft_v2.0", 500),
    ("ai2-adapt-dev/flan_v2_converted", 300),
    ("ai2-adapt-dev/oasst1_converted", 200),
]


def is_english_likely(text):
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) > 0.85


def quality_filter(sample):
    messages = sample.get("messages", [])
    if len(messages) < 2:
        return False
    user_msgs = [m for m in messages if m["role"] == "user"]
    if not user_msgs:
        return False
    first_user = user_msgs[0]["content"]
    if len(first_user) < 10:
        return False
    asst_msgs = [m for m in messages if m["role"] == "assistant"]
    if not asst_msgs:
        return False
    if not is_english_likely(first_user):
        return False
    return True


selected = []
used_ids = set()

for source_key, budget in PRIORITY:
    candidates = []
    for src_name, samples in by_source.items():
        if source_key in src_name or src_name in source_key:
            candidates.extend(samples)
    candidates = [s for s in candidates if quality_filter(s) and s["id"] not in used_ids]
    random.shuffle(candidates)
    take = min(budget, len(candidates))
    for s in candidates[:take]:
        selected.append(s)
        used_ids.add(s["id"])
    print(f"Selected {take} from {source_key} (candidates: {len(candidates)})")

remaining = MAX_SELECTED - len(selected)
if remaining > 0:
    fill_sources = [
        "ai2-adapt-dev/tulu_v3.9_persona_ifeval_turn_1",
        "ai2-adapt-dev/coconot_sft_v2.0",
        "ai2-adapt-dev/wildchat_gpt-4_sft_v2.0",
    ]
    extras = []
    for source_key in fill_sources:
        for src_name, samples in by_source.items():
            if source_key in src_name or src_name in source_key:
                for s in samples:
                    if s["id"] not in used_ids and quality_filter(s):
                        extras.append(s)
    random.shuffle(extras)
    for s in extras[:remaining]:
        selected.append(s)
        used_ids.add(s["id"])
    print(f"Filled {min(remaining, len(extras))} additional samples")

selected = selected[:MAX_SELECTED]
random.shuffle(selected)

final_sources = Counter(s["source"] for s in selected)
print(f"\nTotal selected: {len(selected)}")
print("Final source distribution:")
for src, cnt in final_sources.most_common():
    print(f"  {src}: {cnt}")

with open("/app/data/selected_train.json", "w") as f:
    json.dump(selected, f, ensure_ascii=False)
