#!/bin/bash
set -uo pipefail

# Re-run T1 Sentiment for Llama-3.1-8B-Instruct configs using log-prob scoring.
# Full re-run: bf16 baseline + quantization (RTN/GPTQ/AWQ/SmoothQuant) +
# pruning (Wanda / SparseGPT, c5 and c11) — pruning includes both the earlier
# (2:4 / structured) and new (4:8 / unstructured) versions.
# Output: experiments/results/Llama-3.1-8B-Instruct/ with _logprob suffix.

export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}

OUTDIR="experiments/results/Llama-3.1-8B-Instruct"
MODEL_DIR="data/models"
SCRIPT="scripts/evaluation/eval_t1_sentiment.py"

run_eval() {
    local model="$1"
    local prefix="$2"
    local out="${OUTDIR}/${prefix}_Llama-3.1-8B-Instruct_t1.json"
    if [[ -f "$out" ]]; then
        echo ">>> $prefix (already done, skipping)"
        return
    fi
    if [[ "$model" == data/models/* ]] && [[ ! -d "$model" ]]; then
        echo ">>> $prefix (skipping, model not found)"
        return
    fi
    echo ">>> $prefix"
    CUDA_VISIBLE_DEVICES=1 conda run -n finquant-lmc-main python "$SCRIPT" \
        --model "$model" \
        --output "$out" || echo "!!! FAILED: $prefix"
}

# FP16 baseline
run_eval "meta-llama/Meta-Llama-3.1-8B-Instruct"                          "bf16"

# --- Quantization configs ---
# RTN (no calibration set)
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-rtn-w4a16"              "rtn_w4a16"

# GPTQ W4A16
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-gptq-w4a16-c5"         "gptq_w4a16_c5"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-gptq-w4a16-c11"        "gptq_w4a16_c11"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-gptq-w4a16-c1"        "gptq_w4a16_c1"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-gptq-w4a16-c2"        "gptq_w4a16_c2"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-gptq-w4a16-c3"         "gptq_w4a16_c3"

# AWQ W4A16 asymmetric
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-awq-w4a16-asym-c5"     "awq_w4a16_asym_c5"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-awq-w4a16-asym-c11"    "awq_w4a16_asym_c11"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-awq-w4a16-asym-c1"    "awq_w4a16_asym_c1"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-awq-w4a16-asym-c2"    "awq_w4a16_asym_c2"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-awq-w4a16-asym-c3"      "awq_w4a16_asym_c3"

# SmoothQuant W8A8
#run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-smoothquant-w8a8-c5"   "smoothquant_w8a8_c5"
#run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-smoothquant-w8a8-c11"  "smoothquant_w8a8_c11"
#run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-smoothquant-w8a8-c1"  "smoothquant_w8a8_c1"
#run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-smoothquant-w8a8-c2"  "smoothquant_w8a8_c2"
#run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-smoothquant-w8a8-c3"   "smoothquant_w8a8_c3"

# --- Pruning configs: both the earlier (2:4 / structured) and new (4:8 / unstructured) versions ---

# SparseGPT 2:4 (earlier version)

# SparseGPT unstructured (new version — paper-aligned, paired against Wanda 4:8)
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c5"     "sparsegpt_unstructured_c5"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c11"    "sparsegpt_unstructured_c11"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c1"     "sparsegpt_unstructured_c1"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c2"     "sparsegpt_unstructured_c2"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c3"     "sparsegpt_unstructured_c3"

# Wanda 2:4 (earlier version)

# Wanda 4:8 (new version — looser block constraint ablation)
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c5"                 "wanda_4of8_c5"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c11"                "wanda_4of8_c11"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c1"                 "wanda_4of8_c1"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c2"                 "wanda_4of8_c2"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c3"                 "wanda_4of8_c3"

echo "All T1 logprob evaluations done."
