"""
C3 calibration — T1 Sentiment (FinancialPhraseBank).

Token budget: 128 * 2048 / 10 = 26,214 tokens (LLaMA-3.1-8B tokenizer).
Source: data/raw_datasets/en_fpb/{train,validation}.json, field "query".
        (validation.json: 776 samples, 0 overlap with the en_fpb_test.json eval
        set — combined with train.json's 3,100 samples to enlarge the candidate
        pool for future non-overlapping multi-subset calibration sets.)
Output: data/calibration_datasets/c3_t1_sentiment.jsonl

Prompt format (user message content, same pattern as T3):
    Analyze the sentiment of this statement extracted from a financial news
    article. Provide your answer as either negative, positive, or neutral.
    Text: <sentence>
    Answer:

The eval wraps this in a chat template then appends the assistant-turn scoring
prefix "The sentiment is" before scoring positive/neutral/negative. We use only
the user-message content here, consistent with how T3 uses the raw query ending
in "Answer:" (where the model scores Yes/No on the next token directly).
"""

import json
import random
from pathlib import Path
from transformers import AutoTokenizer

SEED = 42
TOKEN_BUDGET = 128 * 2048 // 10  # 26,214
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

ROOT = Path(__file__).resolve().parents[2]
INPUT_TRAIN = ROOT / "data/raw_datasets/en_fpb/train.json"
INPUT_VAL = ROOT / "data/raw_datasets/en_fpb/validation.json"
OUTPUT = ROOT / "data/calibration_datasets/c3_t1_sentiment.jsonl"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    with open(INPUT_TRAIN) as f:
        train_data = json.load(f)
    with open(INPUT_VAL) as f:
        val_data = json.load(f)

    seen = set()
    texts = []
    for item in train_data + val_data:
        query = item.get("query", "").strip()
        if query and query not in seen:
            seen.add(query)
            texts.append(query)

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

    print(f"T1 Sentiment: {len(selected)} samples, ~{total_tokens} tokens → {OUTPUT.name}")


if __name__ == "__main__":
    main()
