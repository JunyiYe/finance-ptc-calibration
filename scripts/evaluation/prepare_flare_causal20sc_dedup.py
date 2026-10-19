"""
Split flare_causal20_sc_test.json into the 1500 samples actually used by T8
eval (run_t8_logprob_llama8b.sh / run_llama70b_t8_causal.sh, --num-samples
1500) and the remaining samples — saved as test_dedup.json so the remainder
can be used as a T8 task-matched calibration candidate pool without
overlapping the eval set.

Replicates eval_t8_causal.py's exact sample-selection logic:
    random.seed(42); random.shuffle(dataset); dataset = dataset[:1500]

Output:
  data/raw_datasets/flare_causal20_sc/test.json        (original, untouched, 8628 samples)
  data/raw_datasets/flare_causal20_sc/test_dedup.json  (remaining 7128 samples)
"""

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation_datasets"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_datasets"

NUM_EVAL_SAMPLES = 1500


def main():
    src_path = EVAL_DIR / "flare_causal20_sc_test.json"
    with open(src_path) as f:
        original = json.load(f)
    logger.info(f"Loaded {len(original)} samples from {src_path}")

    out_dir = RAW_DIR / "flare_causal20_sc"
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_out_path = out_dir / "test.json"
    with open(orig_out_path, "w") as f:
        json.dump(original, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(original)} records (original) -> {orig_out_path}")

    dataset = list(original)
    random.seed(42)
    random.shuffle(dataset)

    eval_used = dataset[:NUM_EVAL_SAMPLES]
    remaining = dataset[NUM_EVAL_SAMPLES:]
    logger.info(f"  eval-used (matches eval_t8_causal.py --num-samples {NUM_EVAL_SAMPLES}): {len(eval_used)}")
    logger.info(f"  remaining: {len(remaining)}")

    out_path = out_dir / "test_dedup.json"
    with open(out_path, "w") as f:
        json.dump(remaining, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(remaining)} records (dedup) -> {out_path}")


if __name__ == "__main__":
    main()
