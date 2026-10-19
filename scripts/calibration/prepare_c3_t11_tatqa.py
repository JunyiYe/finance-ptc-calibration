"""
C3 calibration — T11 TAT-QA (Table-and-Text Question Answering).

Token budget: 128 * 2048 / 10 = 26,214 tokens (LLaMA-3.1-8B tokenizer).
Source: data/raw_datasets/flare_tatqa/test_dedup.json (168 samples —
        the non-eval portion of flare_tatqa_test.json after seed=42
        shuffle + first 1500 are reserved for eval).
Output: data/calibration_datasets/c3_t11_tatqa.jsonl

Prompt format (mirrors eval_t11_tatqa.py build_cot_prompt):
    Please answer the given financial question based on the context.
    Context: <table/text>
    ...
    Question: <question>

    Think step by step, then end your response with "Answer: X" where X
    is the final answer. X may be a number, a short phrase, or — if the
    question asks for multiple values — a list of items separated by commas.
"""

import json
import re
import random
from pathlib import Path
from transformers import AutoTokenizer

SEED = 42
TOKEN_BUDGET = 128 * 2048 // 10  # 26,214
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/raw_datasets/flare_tatqa/test_dedup.json"
OUTPUT = ROOT / "data/calibration_datasets/c3_t11_tatqa.jsonl"

_ANSWER_CUE = re.compile(r"\n?Answer:\s*$", re.IGNORECASE)

COT_SUFFIX = (
    '\n\nThink step by step, then end your response with "Answer: X" '
    "where X is the final answer. X may be a number, a short phrase, "
    "or — if the question asks for multiple values — a list of items "
    "separated by commas."
)


def build_prompt(query: str) -> str:
    question = _ANSWER_CUE.sub("", query.rstrip())
    return f"{question}{COT_SUFFIX}"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    with open(INPUT) as f:
        data = json.load(f)

    rng = random.Random(SEED)
    rng.shuffle(data)

    selected, total_tokens = [], 0
    for item in data:
        query = item.get("query", "").strip()
        if not query:
            continue
        prompt = build_prompt(query)
        n = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if total_tokens + n > TOKEN_BUDGET:
            break
        selected.append(prompt)
        total_tokens += n

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for text in selected:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"T11 TATQa: {len(selected)} samples, ~{total_tokens} tokens → {OUTPUT.name}")


if __name__ == "__main__":
    main()
