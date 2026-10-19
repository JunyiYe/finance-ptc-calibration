"""
C3 calibration — T9 ConvFinQA (Conversational Financial QA).

Token budget: 128 * 2048 / 10 = 26,214 tokens (LLaMA-3.1-8B tokenizer).
Source: data/raw_datasets/flare_convfinqa/train.json, field "query".
Output: data/calibration_datasets/c3_t9_convfinqa.jsonl

Prompt format (mirrors eval_t9_convfinqa.py build_cot_prompt):
    In the context of this series of interconnected finance-related queries
    and the additional information provided by the pretext, table data, and
    posttext from a company's financial filings, please provide a response
    to the final question. ...
    Context: <context>
    Conversations: <prior Q&A>
    Question: <final question>

    Think step by step, then end your response with "Answer: X" where X is
    the final answer — a number, or "Yes"/"No" if the question asks for a
    yes/no judgment.

The raw query ends with "\nAnswer:" which is stripped and replaced by the CoT
instruction above, matching the eval script exactly.
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
INPUT = ROOT / "data/raw_datasets/flare_convfinqa/train.json"
OUTPUT = ROOT / "data/calibration_datasets/c3_t9_convfinqa.jsonl"

_ANSWER_CUE = re.compile(r"\n?Answer:\s*$", re.IGNORECASE)

COT_SUFFIX = (
    '\n\nThink step by step, then end your response with "Answer: X" '
    'where X is the final answer — a number, or "Yes"/"No" if the '
    "question asks for a yes/no judgment."
)


def build_prompt(query: str) -> str:
    question = _ANSWER_CUE.sub("", query.rstrip())
    return f"{question}{COT_SUFFIX}"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    with open(INPUT) as f:
        data = json.load(f)

    queries = [item["query"] for item in data if item.get("query", "").strip()]

    rng = random.Random(SEED)
    rng.shuffle(queries)

    selected, total_tokens = [], 0
    for query in queries:
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

    print(f"T9 ConvFinQA: {len(selected)} samples, ~{total_tokens} tokens → {OUTPUT.name}")


if __name__ == "__main__":
    main()
