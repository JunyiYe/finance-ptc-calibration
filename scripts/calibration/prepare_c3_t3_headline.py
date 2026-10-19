"""
C3 calibration — T3 Headline (Gold Commodity News Classification).

Token budget: 128 * 2048 / 10 = 26,214 tokens (LLaMA-3.1-8B tokenizer).
Source: data/raw_datasets/flare_headlines/train.json, field "query".
Output: data/calibration_datasets/c3_t3_headline.jsonl

Prompt format (already in query field, 90 templates = 9 labels × 10 variants):
    <instruction>
    Text: <headline>
    Answer:
"""

import json
import random
from pathlib import Path
from transformers import AutoTokenizer

SEED = 42
TOKEN_BUDGET = 128 * 2048 // 10  # 26,214
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/raw_datasets/flare_headlines/train.json"
OUTPUT = ROOT / "data/calibration_datasets/c3_t3_headline.jsonl"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    with open(INPUT) as f:
        data = json.load(f)

    texts = [item["query"] for item in data if item.get("query", "").strip()]

    rng = random.Random(SEED)
    rng.shuffle(texts)

    selected, total_tokens = [], 0
    for text in texts:
        n = len(tokenizer(text, add_special_tokens=False).input_ids)
        if total_tokens + n > TOKEN_BUDGET:
            break
        selected.append(text)
        total_tokens += n

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for text in selected:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"T3 Headline: {len(selected)} samples, ~{total_tokens} tokens → {OUTPUT.name}")


if __name__ == "__main__":
    main()
