"""
C3 calibration — T2 FOMC (Central Bank Policy Tendency).

Token budget: 128 * 2048 / 10 = 26,214 tokens (rough estimate: len//4).
Source: data/raw_datasets/financemteb_fomc/{train_dedup,test_dedup}.json,
        field "sentence". Both files are already deduped against the 496
        sentences used by finben_fomc_test.json (the T2 eval set); combining
        them enlarges the candidate pool for future non-overlapping
        multi-subset calibration sets.
Output: data/calibration_datasets/c3_t2_fomc.jsonl

Prompt format (matching finben_fomc_test.json / bf16 detailed results):
    <instruction (lowercase labels)>
    Text: <sentence>
    Choose one of the following options:
    A. <l1>
    B. <l2>
    C. <l3>

    Answer with only A, B, or C.
    Answer:

10 instruction templates × 3 label rotations cycled across samples.
"""

import json
import random
from pathlib import Path
from transformers import AutoTokenizer

SEED = 42
TOKEN_BUDGET = 128 * 2048 // 10  # 26,214
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

ROOT = Path(__file__).resolve().parents[2]
INPUT_TRAIN = ROOT / "data/raw_datasets/financemteb_fomc/train_dedup.json"
INPUT_TEST = ROOT / "data/raw_datasets/financemteb_fomc/test_dedup.json"
OUTPUT = ROOT / "data/calibration_datasets/c3_t2_fomc.jsonl"

TEMPLATES = [
    "Analyse the following sentence from a central bank's report. Tag it as hawkish if it alludes to a tightening of the monetary policy, dovish if it implies an easing of the monetary policy, or neutral if the stance is impartial. ",
    "Assess the following statement from a central bank's circular. Indicate it as hawkish if it conveys a tightening of monetary policy, dovish if it insinuates an easing of monetary policy, or neutral if the stance is dispassionate. ",
    "Consider the below statement from a central bank's bulletin. Identify it as hawkish if it supports a tightening of monetary policy, dovish if it hints at an easing of monetary policy, or neutral if the stance is balanced. ",
    "Dissect the given sentence from a central bank's transcript. Tag it as hawkish if it denotes a tightening of monetary policy, dovish if it implies an easing of monetary policy, or neutral if the stance is impartial. ",
    "Evaluate the provided sentence from a central bank's statement. Mark it as hawkish if it signifies a tightening of monetary policy, dovish if it points towards an easing of monetary policy, or neutral if the stance is equanimous. ",
    "Examine the excerpt from a central bank's release below. Classify it as hawkish if it advocates for a tightening of monetary policy, dovish if it suggests an easing of monetary policy, or neutral if the stance is unbiased. ",
    "Reflect on the given sentence from a central bank's announcement. Assign it as hawkish if it endorses a tightening of the monetary policy, dovish if it indicates an easing of the monetary policy, or neutral if the stance is even-handed. ",
    "Review the sentence from a central bank's communiqué. Designate it as hawkish if it expresses a tightening of monetary policy, dovish if it reveals an easing of monetary policy, or neutral if the stance is detached. ",
    "Scrutinize the sentence provided from a central bank's press release. Determine it as hawkish if it proposes a tightening of monetary policy, dovish if it discloses an easing of monetary policy, or neutral if the stance is unbiased. ",
    "Study the sentence below from a central bank's briefing. Categorize it as hawkish if it promotes a tightening of monetary policy, dovish if it represents an easing of monetary policy, or neutral if the stance is nonpartisan.",
]

ROTATIONS = [
    ("dovish", "neutral", "hawkish"),
    ("neutral", "hawkish", "dovish"),
    ("hawkish", "dovish",  "neutral"),
]
LETTERS = ("A", "B", "C")


def build_query(sentence: str, idx: int) -> str:
    template = TEMPLATES[idx % len(TEMPLATES)]
    rotation = ROTATIONS[idx % len(ROTATIONS)]
    choices = "\n".join(f"{l}. {v}" for l, v in zip(LETTERS, rotation))
    return (
        f"{template}\n"
        f"Text: {sentence}\n"
        f"Choose one of the following options:\n"
        f"{choices}\n\n"
        f"Answer with only A, B, or C.\n"
        f"Answer:"
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    with open(INPUT_TRAIN) as f:
        train_data = json.load(f)
    with open(INPUT_TEST) as f:
        test_data = json.load(f)

    seen = set()
    sentences = []
    for item in train_data + test_data:
        s = item.get("sentence", "").strip()
        if s and s not in seen:
            seen.add(s)
            sentences.append(s)

    rng = random.Random(SEED)
    rng.shuffle(sentences)

    selected, total_tokens = [], 0
    for i, sentence in enumerate(sentences):
        query = build_query(sentence, i)
        n = len(tokenizer(query, add_special_tokens=False).input_ids)
        if total_tokens + n > TOKEN_BUDGET:
            break
        selected.append(query)
        total_tokens += n

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for text in selected:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"T2 FOMC: {len(selected)} samples, ~{total_tokens} tokens → {OUTPUT.name}")


if __name__ == "__main__":
    main()
