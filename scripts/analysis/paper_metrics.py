"""
Single source of truth for the paper's per-task metric policy.

Policy:
  - Classification tasks report weighted-average F1 (class imbalance:
    FPB 59/29/12, Headlines 67/33, ...); FinCausal reports macro F1
    (94:6 imbalance -> equal class weights).
  - Multiple-choice and generative QA tasks report (judge-verified) accuracy.

All table generators and analysis scripts must import TASK_METRIC and
load_scores from here; do not hand-roll loaders.
"""

import glob
import json
import os

TASK_METRIC = {
    1: "weighted_f1",   # FPB
    2: "weighted_f1",   # FOMC
    3: "weighted_f1",   # Headlines
    4: "accuracy",      # CFA
    5: "accuracy",      # FinQA (judge)
    6: "weighted_f1",   # FinRED
    7: "macro_f1",      # FinCausal (94:6 imbalance -> equal class weights)
    8: "accuracy",      # ConvFinQA (judge)
    9: "weighted_f1",   # BigData22
    10: "accuracy",     # TAT-QA (judge)
}

# tasks whose accuracy comes from LLM adjudication
JUDGED = {5, 8, 10}

# config-name redirects (kept for API compatibility; every FinMix run is
# published under the plain c3 config name, so this is empty)
FINMIX_REDIRECT = {}


def load_scores(model):
    """Return {(config, task): score_pct} using the paper's metric policy.

    For judged tasks, *_llm_adjudicated.json accuracy takes precedence.
    Test-set consistency overrides (see build_testset_overrides.py) win last.
    """
    rows = {}
    ov_path = "experiments/results/analysis/testset_overrides.json"
    overrides = json.load(open(ov_path)) if os.path.exists(ov_path) else {}
    for f in glob.glob(f"experiments/results/{model}/*.json"):
        b = os.path.basename(f)
        if "_detailed" in b:
            continue
        import re
        m = re.search(r"_t(\d+)", b)
        if not m:
            continue
        t = int(m.group(1))
        if t not in TASK_METRIC:
            continue
        adj = "_llm_adjudicated" in b
        cfg = b.split(f"_{model}")[0]
        try:
            d = json.load(open(f))
        except Exception:
            continue
        metrics = d.get("metrics") or d
        key = TASK_METRIC[t]
        v = metrics.get(key)
        if v is None and adj:
            v = metrics.get("accuracy")
        if not isinstance(v, (int, float)):
            continue
        if adj or (cfg, t) not in rows:
            rows[(cfg, t)] = v * 100
    for key, mets in overrides.items():
        m_, cfg, t = key.split("|")
        if m_ == model:
            rows[(cfg, int(t))] = mets[TASK_METRIC[int(t)]] * 100
    return rows
