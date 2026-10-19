#!/bin/bash
# Qwen3-8B evaluation — paper grid for the model-family transfer: BF16
# reference, RTN, and 4 methods (GPTQ/AWQ/SparseGPT/Wanda) × c1(WikiText)/
# c3(FinMix), over all ten tasks (thinking mode OFF). Tasks with large test
# pools (T3/T6/T7/T10) are evaluated on 1500 samples (seed-42 shuffle, first
# 1500 — paper Table 2); the rest use the full test file. This is
# the out-of-sample test of the damage–recovery relationship (including the
# SparseGPT prediction).
#
# Quantize first with scripts/quantization/run_qwen3_8b_quantize_all.sh;
# configurations whose model directory is missing are skipped.
#
# If DEEPSEEK_API_KEY is set, the generative tasks (t5/t8/t10) are adjudicated
# inline; otherwise run the adjudicate_*.py scripts afterwards.
#
# Results: experiments/results/Qwen3-8B/{cfg}_Qwen3-8B_t{N}.json
# (same naming as Llama so scripts/analysis/* work unchanged)
set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME="Qwen3-8B"
MODEL_DIR="data/models"
OUTDIR="experiments/results/${MODEL_NAME}"
mkdir -p "$OUTDIR"

CONFIG_NAMES=(bf16 rtn_w4a16)
CONFIG_PATHS=(
    "Qwen/Qwen3-8B"
    "${MODEL_DIR}/${MODEL_NAME}-rtn-w4a16"
)
for C in c1 c3; do
    for M in "gptq_w4a16:gptq-w4a16" "awq_w4a16:awq-w4a16" \
             "sparsegpt_unstructured:sparsegpt-unstructured" "wanda_4of8:wanda-4of8"; do
        IFS=':' read -r cfg_name dir_name <<< "$M"
        CONFIG_NAMES+=("${cfg_name}_${C}")
        CONFIG_PATHS+=("${MODEL_DIR}/${MODEL_NAME}-${dir_name}-${C}")
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

judge_script() {
    case "$1" in
        t5)  echo scripts/llm_judgement/adjudicate_t5_finqa.py ;;
        t8)  echo scripts/llm_judgement/adjudicate_t8_convfinqa.py ;;
        t10) echo scripts/llm_judgement/adjudicate_t10_tatqa.py ;;
        *)   echo "" ;;
    esac
}

for i in "${!CONFIG_NAMES[@]}"; do
    cfg="${CONFIG_NAMES[$i]}"
    model="${CONFIG_PATHS[$i]}"

    if [[ "$model" == data/models/* && ! -d "$model" ]]; then
        echo "[$(date +%H:%M)] === ${cfg}: model not found, skipping ==="
        continue
    fi

    for j in "${!TASKS[@]}"; do
        task="${TASKS[$j]}"
        out="${OUTDIR}/${cfg}_${MODEL_NAME}_${task}.json"
        detailed="${OUTDIR}/${cfg}_${MODEL_NAME}_${task}_detailed.json"
        if [[ -f "$out" ]]; then
            echo "[$(date +%H:%M)] === ${cfg} ${task}: done already, skipping ==="
        else
            echo "[$(date +%H:%M)] === ${cfg} ${task} ==="
            ns="${TASK_NSAMPLES[$j]}"
            python "scripts/evaluation/${TASK_SCRIPTS[$j]}" \
                --model "$model" \
                ${ns:+--num-samples "$ns"} \
                --output "$out" || echo "!!! FAILED: ${cfg} ${task}"
        fi
        # adjudicate the generative tasks for comparability with the Llama grid
        judge=$(judge_script "$task")
        if [[ -n "$judge" && -n "${DEEPSEEK_API_KEY:-}" && -f "$detailed" ]]; then
            adjud="${OUTDIR}/${cfg}_${MODEL_NAME}_${task}_detailed_llm_adjudicated.json"
            if [[ ! -f "$adjud" ]]; then
                echo "[$(date +%H:%M)] === adjudicate ${cfg} ${task} ==="
                python "$judge" --input "$detailed" || echo "!!! FAILED: adjudicate ${cfg} ${task}"
            fi
        fi
    done
done

echo "All Qwen3-8B evaluations done: $(date)"
echo "Analyze with:"
echo "  python3 scripts/analysis/calibration_significance.py --model ${MODEL_NAME} --pairs c3:c1"
