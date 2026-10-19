"""
Merge all 10 per-task C3 calibration JSONLs into one shuffled file.

Input:  data/calibration_datasets/c3_t{1-6,8-11}_*.jsonl  (10 files)
Output: data/calibration_datasets/c3_finmix.jsonl

The merged file is shuffled (seed=42) so samples from different tasks
are interleaved, giving the quantizer diverse calibration signal across
the full forward pass.
"""

import json
import random
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
CAL_DIR = ROOT / "data/calibration_datasets"

TASK_FILES = [
    "c3_t1_sentiment.jsonl",
    "c3_t2_fomc.jsonl",
    "c3_t3_headline.jsonl",
    "c3_t4_cfa.jsonl",
    "c3_t5_finqa.jsonl",
    "c3_t6_finred.jsonl",
    "c3_t8_causal.jsonl",
    "c3_t9_convfinqa.jsonl",
    "c3_t10_stock.jsonl",
    "c3_t11_tatqa.jsonl",
]

OUTPUT = CAL_DIR / "c3_finmix.jsonl"


def main():
    all_records = []
    per_task = {}

    for fname in TASK_FILES:
        path = CAL_DIR / fname
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        per_task[fname] = len(records)
        all_records.extend(records)

    random.seed(SEED)
    random.shuffle(all_records)

    with open(OUTPUT, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"C3 task-mix merged: {len(all_records)} total samples → {OUTPUT.name}")
    print("\nPer-task sample counts:")
    for fname, n in per_task.items():
        task = fname.replace("c3_", "").replace(".jsonl", "")
        print(f"  {task:25s}: {n:4d} samples")


if __name__ == "__main__":
    main()
