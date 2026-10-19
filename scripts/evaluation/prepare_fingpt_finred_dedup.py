"""
Split fingpt_finred_cls_test.json (the full 8928-question pool) into the 1500
samples actually used by T6 eval and the remaining 7428 — saved as
fingpt_finred_cls_calib.json so the remainder can be used as the T6
task-matched calibration source without overlapping the eval set.

The shipped pool (data/evaluation_datasets/fingpt_finred_cls_test.json) is the
test split of FinGPT/fingpt-finred-cls reformatted per question as
    query   = instruction + "\nText: " + input + "\nAnswer:"
    answer  = output
    choices = the instruction's "Options: ..." list
    id      = "finredcls{row}"
Pass --verify-hf to re-download that split and check the identity row by row.

Replicates eval_t6_finred.py's exact sample-selection logic:
    random.seed(42); random.shuffle(dataset); dataset = dataset[:1500]
The remainder is re-sorted by numeric id before saving.

Output:
  data/raw_datasets/fingpt_finred_re/fingpt_finred_cls_calib.json  (7428 samples)
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation_datasets"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_datasets"

NUM_EVAL_SAMPLES = 1500


def verify_against_hf(pool):
    from datasets import load_dataset

    ds = load_dataset("FinGPT/fingpt-finred-cls", split="test")
    assert len(ds) == len(pool), f"size mismatch: HF {len(ds)} vs local {len(pool)}"
    for i, (hf, local) in enumerate(zip(ds, pool)):
        query = hf["instruction"] + "\nText: " + hf["input"] + "\nAnswer:"
        options_line = hf["instruction"].split("Options: ", 1)[1].split("\n")[0]
        choices = [c.strip() for c in options_line.split(",")]
        assert local["id"] == f"finredcls{i}", f"id mismatch at row {i}"
        assert local["query"] == query, f"query mismatch at row {i}"
        assert local["answer"] == hf["output"], f"answer mismatch at row {i}"
        assert local["choices"] == choices, f"choices mismatch at row {i}"
    logger.info(f"Verified: all {len(pool)} rows match FinGPT/fingpt-finred-cls [test]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-hf", action="store_true",
                        help="check the shipped pool against FinGPT/fingpt-finred-cls [test]")
    args = parser.parse_args()

    src_path = EVAL_DIR / "fingpt_finred_cls_test.json"
    with open(src_path) as f:
        pool = json.load(f)
    logger.info(f"Loaded {len(pool)} samples from {src_path}")

    if args.verify_hf:
        verify_against_hf(pool)

    dataset = list(pool)
    random.seed(42)
    random.shuffle(dataset)

    eval_used = dataset[:NUM_EVAL_SAMPLES]
    remaining = dataset[NUM_EVAL_SAMPLES:]
    remaining.sort(key=lambda s: int(s["id"][len("finredcls"):]))
    logger.info(f"  eval-used (matches eval_t6_finred.py --num-samples {NUM_EVAL_SAMPLES}): {len(eval_used)}")
    logger.info(f"  remaining: {len(remaining)}")

    out_dir = RAW_DIR / "fingpt_finred_re"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fingpt_finred_cls_calib.json"
    with open(out_path, "w") as f:
        json.dump(remaining, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(remaining)} records (dedup) -> {out_path}")


if __name__ == "__main__":
    main()
