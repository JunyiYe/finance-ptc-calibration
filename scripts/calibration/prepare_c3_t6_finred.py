"""
C3 calibration — T6 FinRED (Financial Relation Classification).

Token budget: 128 * 2048 / 10 = 26,214 tokens (LLaMA-3.1-8B tokenizer).
Source: data/raw_datasets/fingpt_finred_re/fingpt_finred_cls_calib.json (7428 samples,
        deduped from fingpt_finred_cls_test.json by removing the 1500 eval samples
        selected via seed=42 shuffle in eval_t6_finred.py).
Output: data/calibration_datasets/c3_t6_finred.jsonl

Prompt format (mirrors eval_t6_finred.py build_rotation_prompt — the actual model input):
    <instruction>
    Options:
    A. <c1>
    B. <c2>
    C. <c3>
    D. <c4>
    Text: <text>
    Answer with only A, B, C, or D.
    Answer:

All 4 cyclic rotations are generated per sample, the pool is shuffled (seed=42),
then samples are accumulated until the token budget is reached.
"""

import json
import re
import random
from pathlib import Path
from transformers import AutoTokenizer

SEED = 42
TOKEN_BUDGET = 128 * 2048 // 10  # 26,214
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LETTERS = ("A", "B", "C", "D")

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/raw_datasets/fingpt_finred_re/fingpt_finred_cls_calib.json"
OUTPUT = ROOT / "data/calibration_datasets/c3_t6_finred.jsonl"

PROMPT_RE = re.compile(r"^(.*?)\nOptions:(.*?)\nText:(.*)\nAnswer:\s*$", re.DOTALL)


def cyclic_rotations(choices):
    return [choices[i:] + choices[:i] for i in range(len(choices))]


def build_rotation_prompt(instruction: str, text: str, rotation: list) -> str:
    choice_str = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, rotation))
    return (
        f"{instruction}\n"
        f"Options:\n{choice_str}\n"
        f"Text:{text}\n"
        f"Answer with only A, B, C, or D.\nAnswer:"
    )


def parse_query(query: str):
    m = PROMPT_RE.match(query.strip())
    if not m:
        return None
    instruction = m.group(1).strip()
    options_str = m.group(2).strip()
    text = m.group(3)
    choices = [c.strip() for c in options_str.split(",") if c.strip()]
    return instruction, text, choices


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    with open(INPUT) as f:
        data = json.load(f)

    all_prompts = []
    for item in data:
        parsed = parse_query(item.get("query", ""))
        if not parsed:
            continue
        instruction, text, choices = parsed
        for rotation in cyclic_rotations(choices):
            all_prompts.append(build_rotation_prompt(instruction, text, rotation))

    rng = random.Random(SEED)
    rng.shuffle(all_prompts)

    selected, total_tokens = [], 0
    for text in all_prompts:
        n = len(tokenizer(text, add_special_tokens=False).input_ids)
        if total_tokens + n > TOKEN_BUDGET:
            break
        selected.append(text)
        total_tokens += n

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for text in selected:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"T6 FinRED: {len(selected)} samples, ~{total_tokens} tokens → {OUTPUT.name}")


if __name__ == "__main__":
    main()
