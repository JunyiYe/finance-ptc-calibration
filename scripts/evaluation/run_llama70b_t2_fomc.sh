#!/bin/bash
set -uo pipefail

# T2 FOMC A/B/C rotation log-prob evaluation for Llama-3.1-70B-Instruct (c5, c11 + baselines).

export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}

OUTDIR="experiments/results/Llama-3.1-70B-Instruct"
MODEL_DIR="data/models"
SCRIPT="scripts/evaluation/eval_t2_fomc.py"

run_eval() {
    local model="$1"
    local prefix="$2"
    local out="${OUTDIR}/${prefix}_Llama-3.1-70B-Instruct_t2_fomc_logprob.json"
    if [[ -f "$out" ]]; then
        echo ">>> $prefix (already done, skipping)"
        return
    fi
    echo ">>> $prefix"
    CUDA_VISIBLE_DEVICES=0,1 conda run -n finquant-lmc-main python "$SCRIPT" \
        --model "$model" \
        --output "$out" || echo "!!! FAILED: $prefix"
}

# FP16 baseline
run_eval "meta-llama/Meta-Llama-3.1-70B-Instruct"                                    "bf16"

# RTN
run_eval "${MODEL_DIR}/Meta-Llama-3.1-70B-Instruct-rtn-w4a16"                        "rtn_w4a16"

# GPTQ W4A16
run_eval "${MODEL_DIR}/Meta-Llama-3.1-70B-Instruct-gptq-w4a16-c5"                   "gptq_w4a16_c5"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-70B-Instruct-gptq-w4a16-c11"                  "gptq_w4a16_c11"

# AWQ W4A16 asymmetric
run_eval "${MODEL_DIR}/Meta-Llama-3.1-70B-Instruct-awq-w4a16-asym-c5"               "awq_w4a16_asym_c5"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-70B-Instruct-awq-w4a16-asym-c11"              "awq_w4a16_asym_c11"

# SmoothQuant W8A8
run_eval "${MODEL_DIR}/Meta-Llama-3.1-70B-Instruct-smoothquant-w8a8-c5"             "smoothquant_w8a8_c5"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-70B-Instruct-smoothquant-w8a8-c11"            "smoothquant_w8a8_c11"

# SparseGPT 2:4

# Wanda 2:4

echo "All 70B T2 logprob evaluations done."
