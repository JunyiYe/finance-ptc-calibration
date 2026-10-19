"""
C3 calibration — T10 Stock Movement Prediction (flare-sm-bigdata).

Token budget: 128 * 2048 / 10 = 26,214 tokens (LLaMA-3.1-8B tokenizer).
Source: data/raw_datasets/flare_sm_bigdata/train.json (4897 samples, no overlap
        with the eval set which uses flare_sm_bigdata_test.json).
Output: data/calibration_datasets/c3_t10_stock.jsonl

Prompt format (mirrors eval_t10_stock.py score_ab_rotation):
    Based on the data provided, predict the stock price movement.
    Context: date,open,high,low,close,...
    ...

    Choose one of the following options:
    A. Rise
    B. Fall

    Answer with only A or B.
    Answer:

2 rotations: (Rise, Fall) and (Fall, Rise), cycled with i % 2.
The raw query's variable instruction prefix is dropped; only the
\nContext: ... block is kept (matching the eval's extraction logic).
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
ROTATIONS = [("Rise", "Fall"), ("Fall", "Rise")]

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/raw_datasets/flare_sm_bigdata/train.json"
OUTPUT = ROOT / "data/calibration_datasets/c3_t10_stock.jsonl"

_ANSWER_CUE = re.compile(r"\n+Answer:\s*$")


def build_prompt(query: str, rotation: tuple) -> str:
    context_start = query.find("\nContext:")
    context_part = query[context_start:] if context_start != -1 else "\n" + query
    context_part = _ANSWER_CUE.sub("", context_part)
    choice_str = "\n".join(f"{l}. {v}" for l, v in zip(LETTERS, rotation))
    return (
        f"Based on the data provided, predict the stock price movement."
        f"{context_part}\n\n"
        f"Choose one of the following options:\n{choice_str}\n\n"
        f"Answer with only A or B.\nAnswer:"
    )


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

    print(f"T10 Stock: {len(selected)} samples, ~{total_tokens} tokens → {OUTPUT.name}")


if __name__ == "__main__":
    main()
