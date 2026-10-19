#!/bin/bash
# E2: token-budget control for the "16k FinMix beats 262k generic" claim.
# GPTQ calibrated on c1small (WikiText, 128 short seqs ~16k tokens, matching
# FinMix's shape), evaluated on FinQA/ConvFinQA/CFA/FPB + adjudication.
#
# Usage: bash scripts/quantization/run_gptq_wiki16k_control.sh

set -uo pipefail

export HF_HOME=/data/yej/.cache/huggingface
export CUDA_VISIBLE_DEVICES=1   # hard-set: ~/.bashrc exports stale 0,1,2,3
export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-$(grep -oP 'sk-[a-z0-9]+' scripts/evaluation/run_llama8b_t5_finqa.sh | head -1)}"

MODEL_NAME="Llama-3.1-8B-Instruct"
OUTDIR="experiments/results/${MODEL_NAME}"
LOGDIR="experiments/logs/${MODEL_NAME}"
mkdir -p "$LOGDIR"
LOG_FILE="${LOGDIR}/wiki16k_control_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

PY="conda run -n finquant-lmc-main python"

CALIB="data/calibration_datasets/c1small_wikitext_16k.jsonl"
if [[ ! -f "$CALIB" ]]; then
    echo "[$(date +%H:%M)] preparing c1small calibration file"
    $PY scripts/quantization/prepare_c1small_wikitext.py
fi

SAVE_DIR="data/models/Meta-${MODEL_NAME}-gptq-w4a16-c1small"
if [[ ! -d "$SAVE_DIR" ]]; then
    echo "[$(date +%H:%M)] === quantize GPTQ c1small ==="
    $PY scripts/quantization/gptq_w4a16.py \
        --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
        --calib-path "$CALIB" \
        --max-seq-length 2048 --num-calib-samples 128
fi

for task in t5 t9 t4 t1; do
    case "$task" in
        t5) script=scripts/evaluation/eval_t5_finqa.py ;;
        t9) script=scripts/evaluation/eval_t9_convfinqa.py ;;
        t4) script=scripts/evaluation/eval_t4_cfa.py ;;
        t1) script=scripts/evaluation/eval_t1_sentiment.py ;;
    esac
    out="${OUTDIR}/gptq_w4a16_c1small_${MODEL_NAME}_${task}.json"
    detailed="${OUTDIR}/gptq_w4a16_c1small_${MODEL_NAME}_${task}_detailed.json"
    if [[ -f "$detailed" ]]; then
        echo "[$(date +%H:%M)] ${task}: exists, skipping"
        continue
    fi
    echo "[$(date +%H:%M)] === eval gptq_w4a16_c1small ${task} ==="
    $PY "$script" --model "$SAVE_DIR" --output "$out"
done

for task in t5 t9; do
    judge="scripts/llm_judgement/adjudicate_${task/t5/t5_finqa}"
    judge="${judge/t9/t9_convfinqa}.py"
    detailed="${OUTDIR}/gptq_w4a16_c1small_${MODEL_NAME}_${task}_detailed.json"
    adjud="${OUTDIR}/gptq_w4a16_c1small_${MODEL_NAME}_${task}_detailed_llm_adjudicated.json"
    if [[ -f "$detailed" && ! -f "$adjud" ]]; then
        echo "[$(date +%H:%M)] === adjudicate ${task} ==="
        $PY "$judge" --input "$detailed"
    fi
done

echo "Done: $(date)"
