"""
Download FinanceMTEB/FOMC train+test splits and remove sentences that
overlap with the T2 evaluation set (data/evaluation_datasets/finben_fomc_test.json).

FinanceMTEB/FOMC and TheFinAI/finben-fomc both derive from the same
underlying FOMC sentence-classification corpus (Trillion Dollar Words,
Shah et al. 2023). 264/496 (53%) of finben_fomc_test.json sentences are
literally present in FinanceMTEB/FOMC's train split (206/496 in its test
split) — using either as-is for T2 calibration would leak a large chunk
of the eval set's text.

For each split:
  1. Load the split
  2. Drop internal duplicate sentences
  3. Drop any sentence also present in finben_fomc_test.json

Output:
  data/raw_datasets/financemteb_fomc/train.json        (original, untouched)
  data/raw_datasets/financemteb_fomc/test.json         (original, untouched)
  data/raw_datasets/financemteb_fomc/train_dedup.json
  data/raw_datasets/financemteb_fomc/test_dedup.json
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_datasets"
EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation_datasets"


def dedup_split(split: str, eval_texts: set[str], out_dir: Path) -> None:
    logger.info(f"Loading FinanceMTEB/FOMC [{split}]...")
    ds = load_dataset("FinanceMTEB/FOMC", split=split)
    records = [dict(ex) for ex in ds]
    logger.info(f"  {len(records)} raw rows")

    raw_path = out_dir / f"{split}.json"
    with open(raw_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(records)} records (original) -> {raw_path}")

    seen = set()
    deduped = []
    for r in records:
        s = r["sentence"]
        if s not in seen:
            seen.add(s)
            deduped.append(r)
    logger.info(f"  {len(deduped)} after removing internal duplicates")

    clean = [r for r in deduped if r["sentence"] not in eval_texts]
    logger.info(f"  {len(clean)} remaining after removing overlap with T2 eval set")

    dedup_path = out_dir / f"{split}_dedup.json"
    with open(dedup_path, "w") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(clean)} records (dedup) -> {dedup_path}")


def main():
    out_dir = RAW_DIR / "financemteb_fomc"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "finben_fomc_test.json") as f:
        finben_test = json.load(f)
    eval_texts = set(d["text"] for d in finben_test)
    logger.info(f"finben_fomc_test.json: {len(eval_texts)} unique eval sentences")

    dedup_split("train", eval_texts, out_dir)
    dedup_split("test", eval_texts, out_dir)


if __name__ == "__main__":
    main()
