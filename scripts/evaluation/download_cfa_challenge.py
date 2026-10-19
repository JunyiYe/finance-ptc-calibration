"""
Download the CFA-Challenge subset of Salesforce/FinEval.

This is a distinct CFA dataset from TheFinAI/flare-cfa (the current T4 eval
set, mirrored 1:1 by FinEval's CFA-Easy subset): CFA-Challenge contains 90
long scenario-based questions (CFA Level III style — one scenario followed
by multiple A/B/C sub-questions), sourced from official CFA sample exams
("source": "sample_test"). No train/valid split exists.

Output: data/evaluation_datasets/fineval_cfa_challenge_test.json
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation_datasets"


def main():
    out_path = EVAL_DIR / "fineval_cfa_challenge_test.json"

    logger.info("Downloading Salesforce/FinEval [CFA-Challenge, test]...")
    ds = load_dataset("Salesforce/FinEval", "CFA-Challenge", split="test")
    records = [dict(ex) for ex in ds]

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
