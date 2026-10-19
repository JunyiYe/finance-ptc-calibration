"""
Download additional splits (train/valid) for T3/T6/T9/T10 evaluation datasets,
which only had their `test` split downloaded so far.

These splits are intended for use as task-matched calibration data candidates
(see data/calibration_datasets/README.md).

Datasets (split availability on HuggingFace, checked 2026-06-12):
  - T3  TheFinAI/flare-headlines   : train, valid, test
  - T6  FinGPT/fingpt-finred-re    : train, test
  - T9  TheFinAI/flare-convfinqa   : train, valid, test
  - T10 TheFinAI/flare-sm-bigdata  : train, validation, test

T2 (TheFinAI/finben-fomc), T4 (TheFinAI/flare-cfa) and T8
(TheFinAI/flare-causal20-sc) only expose a `test` split on HuggingFace —
nothing new to download for those.

Output: data/raw_datasets/<name>/<split>.json  (list of dicts, indent=2)
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_datasets"

DATASETS = {
    "flare_headlines": ("TheFinAI/flare-headlines", ["train", "valid", "test"]),
    "fingpt_finred_re": ("FinGPT/fingpt-finred-re", ["train", "test"]),
    "flare_convfinqa": ("TheFinAI/flare-convfinqa", ["train", "valid", "test"]),
    "flare_sm_bigdata": ("TheFinAI/flare-sm-bigdata", ["train", "validation", "test"]),
}


def main():
    for name, (hf_name, splits) in DATASETS.items():
        out_dir = RAW_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            out_path = out_dir / f"{split}.json"
            if out_path.exists():
                logger.info(f"  {out_path} already exists, skipping")
                continue
            logger.info(f"Downloading {hf_name} [{split}] ...")
            ds = load_dataset(hf_name, split=split)
            records = [dict(ex) for ex in ds]
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
