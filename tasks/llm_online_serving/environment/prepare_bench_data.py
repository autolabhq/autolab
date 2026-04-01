#!/usr/bin/env python3
"""Prepare benchmark prompts from Magpie multi-turn conversations."""

import json
import random
from datasets import load_dataset

def main():
    random.seed(42)

    print("Loading Magpie multi-turn dataset...")
    ds = load_dataset("Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered", split="train")

    requests = []
    for row in ds:
        conversations = row.get("conversations", [])
        if len(conversations) < 2:
            continue

        prompt_parts = []
        last_assistant = None
        for msg in conversations:
            role = msg.get("from", "")
            content = msg.get("value", "")
            if role == "human":
                prompt_parts.append(content)
            elif role == "gpt":
                last_assistant = content

        if not prompt_parts or not last_assistant:
            continue

        # Use first user message as prompt (keep it simple for serving benchmark)
        prompt = prompt_parts[0]
        if len(prompt) < 20 or len(prompt) > 2000:
            continue

        # Output length based on actual reply, capped
        output_len = min(max(len(last_assistant.split()) * 2, 64), 256)

        requests.append({
            "prompt": prompt,
            "output_len": output_len,
        })

        if len(requests) >= 500:
            break

    print(f"Collected {len(requests)} requests")
    with open("/data/bench_requests.json", "w") as f:
        json.dump(requests, f, ensure_ascii=False)
    print("Saved to /data/bench_requests.json")

if __name__ == "__main__":
    main()
