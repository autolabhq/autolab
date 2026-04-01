"""Build-time data preparation: download tulu-3-sft-mixture and create pool."""

import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

CAP_PER_SOURCE = 5000
TOTAL_TARGET = 50000
POOL_PATH = Path("/data/pool.json")
META_PATH = Path("/data/pool_metadata.json")
DATASET_INFO_PATH = Path("/app/data/dataset_info.json")


def main():
    from datasets import load_dataset

    print("Downloading allenai/tulu-3-sft-mixture (pinned revision) ...")
    ds = load_dataset(
        "allenai/tulu-3-sft-mixture",
        split="train",
        revision="b14afda60f1bbebe55d5d2fa1e4df5042f97f8be",
    )
    print(f"Full dataset: {len(ds)} samples")

    # Group by source
    by_source: dict[str, list[int]] = {}
    for idx in range(len(ds)):
        src = ds[idx]["source"]
        by_source.setdefault(src, []).append(idx)

    print(f"\nSources ({len(by_source)}):")
    for src, indices in sorted(by_source.items(), key=lambda x: -len(x[1])):
        print(f"  {src}: {len(indices)}")

    # Stratified sample: cap per source
    selected_indices = []
    for src, indices in by_source.items():
        random.shuffle(indices)
        take = min(len(indices), CAP_PER_SOURCE)
        selected_indices.extend(indices[:take])

    # If over target, randomly downsample
    if len(selected_indices) > TOTAL_TARGET:
        random.shuffle(selected_indices)
        selected_indices = selected_indices[:TOTAL_TARGET]

    selected_indices.sort()

    # Build pool
    pool = []
    for idx in selected_indices:
        row = ds[idx]
        messages = []
        for msg in row["messages"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        pool.append({
            "id": row["id"],
            "messages": messages,
            "source": row["source"],
        })

    random.shuffle(pool)

    # Save pool
    Path("/data").mkdir(parents=True, exist_ok=True)
    POOL_PATH.write_text(json.dumps(pool, ensure_ascii=False))
    print(f"\nPool saved: {len(pool)} samples -> {POOL_PATH}")

    # Save metadata
    source_counts = Counter(s["source"] for s in pool)
    meta = {
        "total": len(pool),
        "sources": dict(source_counts.most_common()),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved -> {META_PATH}")

    # Create LLaMA-Factory dataset_info.json
    Path("/app/data").mkdir(parents=True, exist_ok=True)
    dataset_info = {
        "selected_train": {
            "file_name": "selected_train_sharegpt.json",
            "formatting": "sharegpt",
        }
    }
    DATASET_INFO_PATH.write_text(json.dumps(dataset_info, indent=2))
    print(f"dataset_info.json saved -> {DATASET_INFO_PATH}")


if __name__ == "__main__":
    main()
