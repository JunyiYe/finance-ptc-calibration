"""
Verify actual token counts for each C3 per-task JSONL file.
Uses the same tokenizer as the generation scripts.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET = 128 * 2048 // 10  # 26,214 per task
TOTAL_TARGET = 128 * 2048  # 262,144

ROOT = Path(__file__).resolve().parents[2]
CAL_DIR = ROOT / "data/calibration_datasets"

TASK_FILES = [
    ("T1  Sentiment ", "c3_t1_sentiment.jsonl"),
    ("T2  FOMC      ", "c3_t2_fomc.jsonl"),
    ("T3  Headline  ", "c3_t3_headline.jsonl"),
    ("T4  CFA       ", "c3_t4_cfa.jsonl"),
    ("T5  FinQA     ", "c3_t5_finqa.jsonl"),
    ("T6  FinRED    ", "c3_t6_finred.jsonl"),
    ("T8  Causal    ", "c3_t8_causal.jsonl"),
    ("T9  ConvFinQA ", "c3_t9_convfinqa.jsonl"),
    ("T10 Stock     ", "c3_t10_stock.jsonl"),
    ("T11 TATQa     ", "c3_t11_tatqa.jsonl"),
]


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    print()

    header = f"{'Task':<20} {'Samples':>7} {'Tokens':>8} {'Target':>8} {'Diff':>7} {'min_tok':>8} {'max_tok':>8}"
    print(header)
    print("-" * len(header))

    grand_samples = 0
    grand_tokens = 0

    for label, fname in TASK_FILES:
        path = CAL_DIR / fname
        tokens_list = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    text = json.loads(line)["text"]
                    tokens_list.append(count_tokens(tokenizer, text))

        n = len(tokens_list)
        total = sum(tokens_list)
        diff = total - TARGET
        sign = "+" if diff >= 0 else ""
        mn = min(tokens_list)
        mx = max(tokens_list)

        print(f"{label:<20} {n:>7} {total:>8,} {TARGET:>8,} {sign}{diff:>6,} {mn:>8,} {mx:>8,}")
        grand_samples += n
        grand_tokens += total

    print("-" * len(header))
    print(f"{'TOTAL':<20} {grand_samples:>7} {grand_tokens:>8,} {TOTAL_TARGET:>8,} "
          f"{'+' if grand_tokens >= TOTAL_TARGET else ''}{grand_tokens - TOTAL_TARGET:>6,}")


if __name__ == "__main__":
    main()
