"""
Download the raw source datasets behind the FinMix calibration corpus
(data/raw_datasets/, not shipped with the repo — the built corpus is).

Per task (split availability on HuggingFace):
  - T1  TheFinAI/en-fpb            : train, validation, test
  - T2  FinanceMTEB/FOMC           : train, test   (dedup via prepare_financemteb_fomc_dedup.py)
  - T3  TheFinAI/flare-headlines   : train, valid, test
  - T5  TheFinAI/flare-finqa       : train, valid, test
  - T6  FinGPT/fingpt-finred-cls   : test only     (the shipped eval pool; calib
                                     split via prepare_fingpt_finred_dedup.py)
  - T7  TheFinAI/flare-causal20-sc : test only     (split via prepare_flare_causal20sc_dedup.py)
  - T8  TheFinAI/flare-convfinqa   : train, valid, test
  - T9  TheFinAI/flare-sm-bigdata  : train, validation, test
  - T10 TheFinAI/flare-tatqa       : test only     (split via prepare_flare_tatqa_dedup.py)

T4 has no train split anywhere; its calibration source (FinEval CFA Challenge)
ships with the repo and can be re-fetched with
scripts/calibration/download_cfa_challenge.py.

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
    "en_fpb": ("TheFinAI/en-fpb", ["train", "validation", "test"]),
    "financemteb_fomc": ("FinanceMTEB/FOMC", ["train", "test"]),
    "flare_headlines": ("TheFinAI/flare-headlines", ["train", "valid", "test"]),
    "flare_finqa": ("TheFinAI/flare-finqa", ["train", "valid", "test"]),
    "flare_causal20_sc": ("TheFinAI/flare-causal20-sc", ["test"]),
    "flare_convfinqa": ("TheFinAI/flare-convfinqa", ["train", "valid", "test"]),
    "flare_sm_bigdata": ("TheFinAI/flare-sm-bigdata", ["train", "validation", "test"]),
    "flare_tatqa": ("TheFinAI/flare-tatqa", ["test"]),
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
