"""
Prepare C2 calibration dataset from allenai/c4 (English Common Crawl).

Sampling method:
  - Stream allenai/c4 en, keep only documents that tokenize to >= seqlen tokens
  - Randomly sample nsamples from the qualifying pool (seed=42)
  - Save full document text as {"text": "..."} — llm-compressor handles truncation

Output: data/calibration_datasets/c2_c4_<nsamples>.jsonl
"""

import argparse
import json
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
DEFAULT_NSAMPLES = 128
DEFAULT_SEQLEN = 2048
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "calibration_datasets"


def get_c4_samples(nsamples: int, seed: int, seqlen: int, model_id: str) -> list[str]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    buffer_size = max(nsamples * 10, 2000)

    logger.info(f"Streaming allenai/c4 en (collecting {buffer_size} docs with >= {seqlen} tokens)...")
    stream = load_dataset("allenai/c4", "en", split="train", streaming=True)

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    candidates = []
    for doc in stream:
        n = tokenizer(doc["text"], return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
        if n >= seqlen:
            candidates.append(doc["text"])
        if len(candidates) >= buffer_size:
            break

    logger.info(f"  Buffered {len(candidates)} qualifying docs")
    random.seed(seed)
    chosen = random.sample(candidates, nsamples)
    logger.info(f"Done: {len(chosen)} samples selected")
    return chosen


def save_jsonl(texts: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(texts)} samples → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare C11 calibration dataset from allenai/c4")
    parser.add_argument("--nsamples", type=int, default=DEFAULT_NSAMPLES,
                        help=f"Number of samples (default: {DEFAULT_NSAMPLES})")
    parser.add_argument("--seqlen", type=int, default=DEFAULT_SEQLEN,
                        help=f"Token window length per sample (default: {DEFAULT_SEQLEN})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Tokenizer model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: data/calibration_datasets/c2_c4_<nsamples>.jsonl)")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"c2_c4_{args.nsamples}.jsonl"

    logger.info("=== Preparing C2 calibration dataset (allenai/c4 GPTQ-style) ===")
    logger.info(f"  nsamples={args.nsamples}, seqlen={args.seqlen}, seed={args.seed}")
    logger.info(f"  tokenizer={args.model}")
    logger.info(f"  output={output_path}")

    texts = get_c4_samples(
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        model_id=args.model,
    )
    save_jsonl(texts, output_path)
    logger.info("=== C2 ready ===")


if __name__ == "__main__":
    main()
