#!/bin/bash
set -uo pipefail

# Full T5 FinQA pipeline for Llama-3.1-8B-Instruct.
#   1. Generate responses with eval_t5_finqa.py  (full 1147 samples, no --num-samples)
#   2. Adjudicate with adjudicate_t5.py    (updated judge: original prompt + directional-word rule)
#
# Output: experiments/results/Llama-3.1-8B-Instruct/
#
# Usage:
#   bash scripts/evaluation/run_llama8b_t5_finqa.sh

export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-$(cat /data/yej/.deepseek_key 2>/dev/null)}"

MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_DIR="data/models"
OUTDIR="experiments/results/${MODEL_NAME}"
LOGDIR="experiments/logs/${MODEL_NAME}"
EVAL_SCRIPT="scripts/evaluation/eval_t5_finqa.py"
JUDGE_SCRIPT="scripts/llm_judgement/adjudicate_t5.py"

mkdir -p "$LOGDIR"
# Append to the most recent existing log if present, otherwise create new
LATEST_LOG=$(ls -t "${LOGDIR}"/run_t5_llama8b_*.log 2>/dev/null | head -1) || true
if [[ -n "$LATEST_LOG" ]]; then
    LOG_FILE="$LATEST_LOG"
    echo "" >> "$LOG_FILE"
    echo "================================================================" >> "$LOG_FILE"
    echo "RESUMED: $(date)" >> "$LOG_FILE"
    echo "================================================================" >> "$LOG_FILE"
else
    LOG_FILE="${LOGDIR}/run_t5_llama8b_$(date +%Y%m%d_%H%M%S).log"
fi
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

run_eval() {
    local model="$1"
    local prefix="$2"
    local out="${OUTDIR}/${prefix}_${MODEL_NAME}_t5.json"
    local detailed="${OUTDIR}/${prefix}_${MODEL_NAME}_t5_detailed.json"
    local adjudicated="${OUTDIR}/${prefix}_${MODEL_NAME}_t5_detailed_llm_adjudicated.json"

    # Skip generation if full data (>=1147 samples) already exists
    if [[ -f "$detailed" ]] && python3 -c "import json; d=json.load(open('$detailed')); exit(0 if len(d)>=1147 else 1)" 2>/dev/null; then
        echo "[$(date +%H:%M)] === ${prefix}: generate (skipping, full data exists) ==="
    elif [[ "$model" == data/models/* ]] && [[ ! -d "$model" ]]; then
        echo "[$(date +%H:%M)] === ${prefix}: skipping (model not found) ==="
        return
    else
        echo "[$(date +%H:%M)] === ${prefix}: generate ==="
        CUDA_VISIBLE_DEVICES=1 conda run -n finquant-lmc-main python "$EVAL_SCRIPT" \
            --model "$model" \
            --output "$out"
    fi

    # Adjudication temporarily disabled — rerunning generation only after the
    # numbers_match fix (dropped 2-sig-fig tolerance tier); re-enable once
    # the new generation results are confirmed.
    # if [[ -f "$adjudicated" ]] && python3 -c "import json; d=json.load(open('$adjudicated')); exit(0 if len(d)>=1147 else 1)" 2>/dev/null; then
    #     echo "[$(date +%H:%M)] === ${prefix}: adjudicate (skipping, full data exists) ==="
    # else
    #     echo "[$(date +%H:%M)] === ${prefix}: adjudicate ==="
    #     conda run -n finquant-lmc-main python "$JUDGE_SCRIPT" \
    #         --input "$detailed"
    # fi
}

echo "================================================================"
echo "Llama-3.1-8B T5 FinQA — full pipeline (generate + adjudicate)"
echo "Start: $(date)"
echo "================================================================"

# ── BF16 baseline ────────────────────────────────────────────────
run_eval "meta-llama/Meta-Llama-3.1-8B-Instruct"                                  "bf16"

# ── RTN W4A16 ────────────────────────────────────────────────────
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-rtn-w4a16"                      "rtn_w4a16"

# ── GPTQ W4A16 ───────────────────────────────────────────────────
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-gptq-w4a16-c5"                  "gptq_w4a16_c5"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-gptq-w4a16-c11"                 "gptq_w4a16_c11"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-gptq-w4a16-c1"                 "gptq_w4a16_c1"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-gptq-w4a16-c2"                 "gptq_w4a16_c2"

# ── AWQ W4A16 asymmetric ─────────────────────────────────────────
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-awq-w4a16-asym-c5"              "awq_w4a16_asym_c5"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-awq-w4a16-asym-c11"             "awq_w4a16_asym_c11"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-awq-w4a16-asym-c1"             "awq_w4a16_asym_c1"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-awq-w4a16-asym-c2"             "awq_w4a16_asym_c2"

# ── SmoothQuant W8A8 ─────────────────────────────────────────────
#run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-smoothquant-w8a8-c5"            "smoothquant_w8a8_c5"
#run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-smoothquant-w8a8-c11"           "smoothquant_w8a8_c11"
#run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-smoothquant-w8a8-c1"           "smoothquant_w8a8_c1"
#run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-smoothquant-w8a8-c2"           "smoothquant_w8a8_c2"

# ── SparseGPT 2:4 ────────────────────────────────────────────────

# ── SparseGPT unstructured ───────────────────────────────────────
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c5"      "sparsegpt_unstructured_c5"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c11"     "sparsegpt_unstructured_c11"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c1"  "sparsegpt_unstructured_c1"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c2"  "sparsegpt_unstructured_c2"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c3"  "sparsegpt_unstructured_c3"

# ── Wanda 2:4 ────────────────────────────────────────────────────

# ── Wanda 4:8 ────────────────────────────────────────────────────
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c5"                  "wanda_4of8_c5"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c11"                 "wanda_4of8_c11"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c1"  "wanda_4of8_c1"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c2"  "wanda_4of8_c2"
run_eval "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c3"  "wanda_4of8_c3"

echo ""
echo "================================================================"
echo "All T5 FinQA runs done: $(date)"
echo "================================================================"
