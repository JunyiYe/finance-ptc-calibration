"""
C3 calibration — T8 Causal Classification (flare-causal20-sc).

Token budget: 128 * 2048 / 10 = 26,214 tokens (LLaMA-3.1-8B tokenizer).
Source: data/raw_datasets/flare_causal20_sc/calib_dedup.json (7128 samples,
        deduped from flare_causal20_sc_test.json by removing 1500 eval samples
        selected via seed=42 shuffle in eval_t8_causal.py).
Output: data/calibration_datasets/c3_t8_causal.jsonl

Prompt format (mirrors eval_t8_causal.py score_ab_rotation — the actual model input):
    In this task, ... Your goal is to classify each sentence into either
    'causal' or 'noise' ...
    Text: <sentence>
    Choose one of the following options:
    A. <label1>
    B. <label2>

    Answer with only A or B.
    Answer:

2 rotations: (causal, noise) and (noise, causal), cycled with i % 2.
"""

import json
import re
import random
from pathlib import Path
from transformers import AutoTokenizer

SEED = 42
TOKEN_BUDGET = 128 * 2048 // 10  # 26,214
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LETTERS = ("A", "B")
ROTATIONS = [("causal", "noise"), ("noise", "causal")]

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/raw_datasets/flare_causal20_sc/calib_dedup.json"
OUTPUT = ROOT / "data/calibration_datasets/c3_t8_causal.jsonl"


def build_prompt(query: str, rotation: tuple) -> str:
    choice_str = "\n".join(f"{l}. {v}" for l, v in zip(LETTERS, rotation))
    modified = re.sub(r"Please return only the category 'causal' or 'noise'\.", "", query)
    modified = modified.replace(
        "\nAnswer:",
        f"\nChoose one of the following options:\n{choice_str}\n\nAnswer with only A or B.\nAnswer:",
    )
    return modified


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    with open(INPUT) as f:
        data = json.load(f)

    rng = random.Random(SEED)
    rng.shuffle(data)

    selected, total_tokens = [], 0
    for i, item in enumerate(data):
        query = item.get("query", "").strip()
        if not query:
            continue
        rotation = ROTATIONS[i % 2]
        prompt = build_prompt(query, rotation)
        n = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if total_tokens + n > TOKEN_BUDGET:
            break
        selected.append(prompt)
        total_tokens += n

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for text in selected:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"T8 Causal: {len(selected)} samples, ~{total_tokens} tokens → {OUTPUT.name}")


if __name__ == "__main__":
    main()
