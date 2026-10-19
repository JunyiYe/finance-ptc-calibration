"""
C3 calibration — T4 CFA Exam QA.

Token budget: 128 * 2048 / 10 = 26,214 tokens (LLaMA-3.1-8B tokenizer).
Source: data/evaluation_datasets/fineval_cfa_challenge_test.json, field "query".
       (Different dataset from the eval set flare_cfa_test.json — zero overlap.)
Output: data/calibration_datasets/c3_t4_cfa.jsonl

Prompt format (mirrors eval_t4_cfa.py build_cot_prompt output):
    Read the question and choose the most appropriate option among A, B, and C.
    Q: <question>
    A: <choice>
    B: <choice>
    C: <choice>

    Think step by step, then end your response with "Answer: X" where X is A, B, or C.

fineval_cfa_challenge_test.json only has 90 samples (~8,442 tokens), well under the
26,214-token budget. To fill the budget and improve label-position balance, we generate
3 cyclic rotations per sample (same technique as T6/T8/T10), then shuffle and sample.
"""

import json
import re
import random
from pathlib import Path
from transformers import AutoTokenizer

SEED = 42
TOKEN_BUDGET = 128 * 2048 // 10  # 26,214
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LETTERS = ("A", "B", "C")

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/evaluation_datasets/fineval_cfa_challenge_test.json"
OUTPUT = ROOT / "data/calibration_datasets/c3_t4_cfa.jsonl"

# Matches "Question: <text>;" and "Answer Choices: A: ..., B: ..., C: ..."
_Q_RE = re.compile(r"Question:(.*?);?\s*\nAnswer Choices:\s*(.*?)\s*\nAnswer:", re.DOTALL)
# Split choices on ", " followed by A:, B:, or C: (same as eval's CFA_CHOICE_SPLIT_PATTERN)
_CHOICE_SPLIT = re.compile(r",\s*(?=[ABC]:\s)")

COT_SUFFIX = (
    '\n\nThink step by step, then end your response with "Answer: X" '
    "where X is A, B, or C."
)


def parse_query(query: str):
    m = _Q_RE.search(query)
    if not m:
        return None
    question = m.group(1).strip().rstrip(";").strip()
    # Strip trailing ". " or ".." (the sentence-ending period added after the last choice).
    # Each choice's own trailing "." is preserved to match the eval prompt format.
    choices_str = m.group(2).strip().rstrip(". ")
    parts = _CHOICE_SPLIT.split(choices_str)
    choices_ordered = []
    for part in parts:
        label, _, val = part.partition(":")
        choices_ordered.append(val.strip())
    if len(choices_ordered) != 3:
        return None
    return question, choices_ordered


def build_rotation_prompt(question: str, rotation: list) -> str:
    choice_str = "\n".join(f"{l}: {v}" for l, v in zip(LETTERS, rotation))
    return (
        "Read the question and choose the most appropriate option among A, B, and C.\n"
        f"Q: {question}\n"
        f"{choice_str}"
        f"{COT_SUFFIX}"
    )


def cyclic_rotations(choices: list) -> list:
    n = len(choices)
    return [choices[i:] + choices[:i] for i in range(n)]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    with open(INPUT) as f:
        data = json.load(f)

    all_prompts = []
    for item in data:
        parsed = parse_query(item.get("query", "").strip())
        if parsed is None:
            continue
        question, choices = parsed
        for rotation in cyclic_rotations(choices):
            all_prompts.append(build_rotation_prompt(question, rotation))

    rng = random.Random(SEED)
    rng.shuffle(all_prompts)

    selected, total_tokens = [], 0
    for prompt in all_prompts:
        n = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if total_tokens + n > TOKEN_BUDGET:
            break
        selected.append(prompt)
        total_tokens += n

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for text in selected:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"T4 CFA: {len(selected)} samples, ~{total_tokens} tokens → {OUTPUT.name}")


if __name__ == "__main__":
    main()
