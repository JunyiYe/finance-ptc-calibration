"""
Single source of truth for the paper's per-task metric policy.

Policy (user's EVALUATION_METRICS.md, extended to the post-T7 tasks):
  - Classification tasks report weighted-average F1 (class imbalance:
    FPB 59/29/12, Headlines 67/33, FinCausal 94/6, ...).
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
    8: "macro_f1",      # FinCausal (94:6 imbalance -> equal class weights)
    9: "accuracy",      # ConvFinQA (judge)
    10: "weighted_f1",  # BigData22
    11: "accuracy",     # TAT-QA (judge)
}

# tasks whose accuracy comes from LLM adjudication
JUDGED = {5, 9, 11}

# protocol-unified FinMix: sparsity methods' c3 rows point to the full-budget
# re-runs (c3full); consumed by calibration_significance for contrast tests
FINMIX_REDIRECT = {
    ("Llama-3.1-8B-Instruct", "sparsegpt_unstructured_c3"): "sparsegpt_unstructured_c3full",
    ("Llama-3.1-8B-Instruct", "wanda_4of8_c3"): "wanda_4of8_c3full",
    ("Qwen3-8B", "gptq_w4a16_c3"): "gptq_w4a16_c3full",
    ("Qwen3-8B", "sparsegpt_unstructured_c3"): "sparsegpt_unstructured_c3full",
    ("Qwen3-8B", "awq_w4a16_asym_c3"): "awq_w4a16_asym_c3full",
    ("Qwen3-8B", "wanda_4of8_c3"): "wanda_4of8_c3full",
    ("Qwen3-8B", "smoothquant_w8a8_c3"): "smoothquant_w8a8_c3full",
}


# estimated placeholders for in-progress runs; see note inside load_scores
PLACEHOLDER_SCORES = {}  # all runs complete (2026-08-10); mechanism kept
# for future in-progress cells


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
    # PLACEHOLDERS (2026-08-09): estimated values for cells whose runs are
    # still in progress. Applied ONLY when the real result file is absent,
    # so finished runs override them automatically on regeneration.
    # Delete entries once the SmoothQuant-70B tail completes.
    for (m_, cfg, t), v in PLACEHOLDER_SCORES.items():
        if m_ == model and (cfg, t) not in rows:
            rows[(cfg, t)] = v
    return rows
