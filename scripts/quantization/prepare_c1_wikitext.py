"""
Prepare C1 Generic calibration dataset: WikiText-2 (raw), no-decode chunking.

Source:
  - Salesforce/wikitext, config "wikitext-2-raw-v1", split "train"

Method (no encode/decode round-trip — preserves original text exactly):
  1. Take all non-empty lines from wikitext-2 train split, in order
     (each line already ends with its own "\n", as stored by the dataset)
  2. Tokenize each line individually to get its token length
  3. Greedily group consecutive lines into chunks until cumulative token
     length >= SEQLEN + BUFFER, joining each chunk's lines with "\n" so
     that consecutive lines end up separated by "\n\n" (1 from the line's
     own trailing newline + 1 join separator) — standard paragraph spacing
  4. Randomly sample NSAMPLES chunks (seed=42) from all chunks produced
  5. Save the ORIGINAL (un-decoded) chunk text as {"text": "..."}

Output: data/calibration_datasets/c1_wikitext.jsonl
"""

import argparse
import json
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
NSAMPLES = 128
SEQLEN = 2048
BUFFER = 64  # margin so chunks stay >= SEQLEN after re-tokenization downstream
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "calibration_datasets"

WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"


def load_wikitext_lines() -> list[str]:
    from datasets import load_dataset

    logger.info(f"Loading {WIKITEXT_DATASET} ({WIKITEXT_CONFIG}, train split)...")
    ds = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split="train")
    lines = [row["text"] for row in ds if row["text"].strip()]
    logger.info(f"  {len(ds)} lines total, {len(lines)} non-empty")
    return lines


def build_chunks(lines: list[str], model_id: str) -> list[str]:
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    logger.info("Tokenizing each line to measure length...")
    lengths = [len(tokenizer(line, add_special_tokens=False).input_ids) for line in lines]

    threshold = SEQLEN + BUFFER
    chunks = []
    current_lines: list[str] = []
    current_len = 0
    for line, n in zip(lines, lengths):
        current_lines.append(line)
        current_len += n
        if current_len >= threshold:
            chunks.append("\n".join(current_lines))
            current_lines, current_len = [], 0

    logger.info(f"  Built {len(chunks)} chunks (>= {threshold} tokens each)")
    if len(chunks) < NSAMPLES:
        raise ValueError(f"Not enough chunks: need {NSAMPLES} but only {len(chunks)} available")
    return chunks


def sample_chunks(chunks: list[str], seed: int = SEED) -> list[str]:
    random.seed(seed)
    return random.sample(chunks, NSAMPLES)


def save_jsonl(texts: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(texts)} samples → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED,
                        help="chunk sampling seed; seeds != 42 produce seed-variant "
                             "files named c1s<seed>_wikitext_seed<seed>.jsonl for the "
                             "calibration seed-robustness ablation")
    args = parser.parse_args()

    if args.seed == SEED:
        output_path = DATA_DIR / "c1_wikitext.jsonl"
    else:
        output_path = DATA_DIR / f"c1s{args.seed}_wikitext_seed{args.seed}.jsonl"

    logger.info("=== Preparing C1 calibration dataset (WikiText-2, no-decode chunking) ===")
    logger.info(f"  nsamples={NSAMPLES}, seqlen={SEQLEN}, buffer={BUFFER}, seed={args.seed}")
    logger.info(f"  output={output_path}")

    lines = load_wikitext_lines()
    chunks = build_chunks(lines, DEFAULT_MODEL)
    samples = sample_chunks(chunks, args.seed)
    save_jsonl(samples, output_path)
    logger.info("=== C1 ready ===")


if __name__ == "__main__":
    main()
