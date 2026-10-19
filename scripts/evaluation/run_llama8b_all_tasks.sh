#!/bin/bash
# Llama-3.1-8B-Instruct evaluation — full paper grid: BF16 reference, RTN, and
# 5 methods (GPTQ/AWQ/SmoothQuant/SparseGPT/Wanda) × 3 calibration corpora
# (c1=WikiText, c2=C4, c3=FinMix), over all ten tasks. Tasks with large test
# pools (T3/T6/T7/T10) are evaluated on 1500 samples (seed-42 shuffle, first
# 1500 — paper Table 2); the rest use the full test file.
#
# Usage:
#   bash scripts/evaluation/run_llama8b_all_tasks.sh          # everything
#   bash scripts/evaluation/run_llama8b_all_tasks.sh t5       # one task only
#
# Completed (config, task) pairs are skipped, so the sweep is safe to stop and
# resume. Adjudicate the generative tasks afterwards (requires DEEPSEEK_API_KEY):
#   bash scripts/llm_judgement/run_adjudicate_all.sh Llama-3.1-8B-Instruct
set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ONLY_TASK="${1:-}"

MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_DIR="data/models"
OUTDIR="experiments/results/${MODEL_NAME}"
mkdir -p "$OUTDIR"

CONFIG_NAMES=(bf16 rtn_w4a16)
CONFIG_PATHS=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-rtn-w4a16"
)
for C in c1 c2 c3; do
    for M in "gptq_w4a16:gptq-w4a16" "awq_w4a16:awq-w4a16" \
             "smoothquant_w8a8:smoothquant-w8a8" \
             "sparsegpt_unstructured:sparsegpt-unstructured" "wanda_4of8:wanda-4of8"; do
        IFS=':' read -r cfg_name dir_name <<< "$M"
        CONFIG_NAMES+=("${cfg_name}_${C}")
        CONFIG_PATHS+=("${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-${dir_name}-${C}")
    done
done

TASKS=(t1 t2 t3 t4 t5 t6 t7 t8 t9 t10)
TASK_SCRIPTS=(
    "eval_t1_sentiment.py"
    "eval_t2_fomc.py"
    "eval_t3_headline.py"
    "eval_t4_cfa.py"
    "eval_t5_finqa.py"
    "eval_t6_finred.py"
    "eval_t7_causal.py"
    "eval_t8_convfinqa.py"
    "eval_t9_stock.py"
    "eval_t10_tatqa.py"
)
# paper protocol (Table 2): cap large test pools at 1500 (deterministic
# seed-42 selection inside each eval script); empty = full test file
TASK_NSAMPLES=("" "" "1500" "" "" "1500" "1500" "" "" "1500")

for i in "${!CONFIG_NAMES[@]}"; do
    cfg="${CONFIG_NAMES[$i]}"
    model="${CONFIG_PATHS[$i]}"

    if [[ "$model" == data/models/* && ! -d "$model" ]]; then
        echo "[$(date +%H:%M)] === ${cfg}: model not found, skipping ==="
        continue
    fi

    for j in "${!TASKS[@]}"; do
        task="${TASKS[$j]}"
        if [[ -n "$ONLY_TASK" && "$task" != "$ONLY_TASK" ]]; then
            continue
        fi
        out="${OUTDIR}/${cfg}_${MODEL_NAME}_${task}.json"
        if [[ -f "$out" ]]; then
            echo "[$(date +%H:%M)] === ${cfg} ${task}: done already, skipping ==="
            continue
        fi
        echo "[$(date +%H:%M)] === ${cfg} ${task} ==="
        ns="${TASK_NSAMPLES[$j]}"
        python "scripts/evaluation/${TASK_SCRIPTS[$j]}" \
            --model "$model" \
            ${ns:+--num-samples "$ns"} \
            --output "$out" || echo "!!! FAILED: ${cfg} ${task}"
    done
done

echo "All 8B evaluations done: $(date)"
