"""
E2 control corpus: WikiText-2 sampled to match FinMix's *shape* --
128 short sequences of ~127 tokens (~16k tokens total) -- but with generic
domain. If GPTQ calibrated on this behaves like full-budget Wiki, the
FinMix advantage is domain, not token budget or sequence shape.

Output: data/calibration_datasets/c1small_wikitext_16k.jsonl
"""

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
NSAMPLES = 128
TARGET_TOKENS = 120  # accumulate lines until >= this; matches FinMix mean ~127
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "calibration_datasets"


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    lines = [r["text"] for r in ds if r["text"].strip()]
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_fast=False)
    lengths = [len(tokenizer(l, add_special_tokens=False).input_ids) for l in lines]

    chunks, cur, cur_len = [], [], 0
    for line, n in zip(lines, lengths):
        cur.append(line)
        cur_len += n
        if cur_len >= TARGET_TOKENS:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
    logger.info(f"built {len(chunks)} short chunks (>= {TARGET_TOKENS} tokens)")

    random.seed(SEED)
    samples = random.sample(chunks, NSAMPLES)
    total = sum(len(tokenizer(s, add_special_tokens=False).input_ids) for s in samples)
    logger.info(f"sampled {NSAMPLES} chunks, total {total} tokens "
                f"(mean {total/NSAMPLES:.0f})")

    out = DATA_DIR / "c1small_wikitext_16k.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
    logger.info(f"saved -> {out}")


if __name__ == "__main__":
    main()
