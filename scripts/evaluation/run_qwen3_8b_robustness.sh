#!/bin/bash
# Qwen3-8B robustness check for the calibration-domain finding.
#
# Protocol matches the Llama-3.1-8B main matrix: full eval samples,
# thinking mode OFF, tasks T1-T6 + T8-T11, calibration c1 (WikiText) vs
# c3 (financial task mix, random 128-subset via llmcompressor sampler).
# Configs in priority order: BF16 baseline, GPTQ c1, GPTQ c3, RTN (bonus).
#
# Results: experiments/results/Qwen3-8B/{cfg}_Qwen3-8B_t{N}.json
# (same naming as Llama so scripts/analysis/* work unchanged)
#
# Usage:
#   bash scripts/evaluation/run_qwen3_8b_robustness.sh

set -uo pipefail

export HF_HOME=/data/yej/.cache/huggingface
# Hard-set, NOT a :-default — ~/.bashrc exports a stale CUDA_VISIBLE_DEVICES=0,1,2,3
# which would let device_map="auto" place layers on the full GPU 0 and OOM.
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}

BASE="Qwen/Qwen3-8B"
MODEL_NAME="Qwen3-8B"
MODEL_DIR="data/models"
OUTDIR="experiments/results/${MODEL_NAME}"
LOGDIR="experiments/logs/${MODEL_NAME}"
TASKS=(t1 t2 t3 t4 t5 t6 t8 t9 t10 t11)

mkdir -p "$OUTDIR" "$LOGDIR"
LOG_FILE="${LOGDIR}/robustness_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

PY="conda run -n finquant-lmc-main python"

eval_script() {
    case "$1" in
        t1)  echo scripts/evaluation/eval_t1_sentiment.py ;;
        t2)  echo scripts/evaluation/eval_t2_fomc.py ;;
        t3)  echo scripts/evaluation/eval_t3_headline.py ;;
        t4)  echo scripts/evaluation/eval_t4_cfa.py ;;
        t5)  echo scripts/evaluation/eval_t5_finqa.py ;;
        t6)  echo scripts/evaluation/eval_t6_finred.py ;;
        t8)  echo scripts/evaluation/eval_t8_causal.py ;;
        t9)  echo scripts/evaluation/eval_t9_convfinqa.py ;;
        t10) echo scripts/evaluation/eval_t10_stock.py ;;
        t11) echo scripts/evaluation/eval_t11_tatqa.py ;;
    esac
}

judge_script() {
    case "$1" in
        t5)  echo scripts/llm_judgement/adjudicate_t5_finqa.py ;;
        t9)  echo scripts/llm_judgement/adjudicate_t9_convfinqa.py ;;
        t11) echo scripts/llm_judgement/adjudicate_t11_tatqa.py ;;
        *)   echo "" ;;
    esac
}

run_all_tasks() {
    local model="$1" prefix="$2"
    if [[ "$model" == data/models/* && ! -d "$model" ]]; then
        echo "!! model missing: $model — skipping all ${prefix} evals"
        return
    fi
    for task in "${TASKS[@]}"; do
        local out="${OUTDIR}/${prefix}_${MODEL_NAME}_${task}.json"
        local detailed="${OUTDIR}/${prefix}_${MODEL_NAME}_${task}_detailed.json"
        if [[ -f "$detailed" ]]; then
            echo "[$(date +%H:%M)] ${prefix} ${task}: exists, skipping"
        else
            echo "[$(date +%H:%M)] === eval ${prefix} ${task} ==="
            $PY "$(eval_script "$task")" --model "$model" --output "$out"
        fi
        # adjudicate numerical tasks for comparability with Llama main table
        local judge; judge=$(judge_script "$task")
        if [[ -n "$judge" && -n "${DEEPSEEK_API_KEY:-}" && -f "$detailed" ]]; then
            local adjud="${OUTDIR}/${prefix}_${MODEL_NAME}_${task}_detailed_llm_adjudicated.json"
            if [[ ! -f "$adjud" ]]; then
                echo "[$(date +%H:%M)] === adjudicate ${prefix} ${task} ==="
                $PY "$judge" --input "$detailed"
            fi
        fi
    done
}

quantize_gptq() {
    local calib_file="$1"
    local calib_name; calib_name=$(basename "$calib_file" | cut -d'_' -f1)
    local save_dir="${MODEL_DIR}/${MODEL_NAME}-gptq-w4a16-${calib_name}"
    if [[ -d "$save_dir" ]]; then
        echo "[$(date +%H:%M)] quantize ${calib_name}: exists, skipping"
        return
    fi
    echo "[$(date +%H:%M)] === quantize GPTQ ${calib_name} (${MODEL_NAME}) ==="
    $PY scripts/quantization/gptq_w4a16.py \
        --model-id "$BASE" \
        --calib-path "$calib_file" \
        --max-seq-length 2048 --num-calib-samples 128
}

echo "================================================================"
echo "Qwen3-8B robustness: BF16 + GPTQ c1/c3 (+RTN) x ${#TASKS[@]} tasks"
echo "Start: $(date)"
echo "================================================================"

# 1. BF16 baseline
run_all_tasks "$BASE" "bf16"

# 2. GPTQ c1 / c3
quantize_gptq "data/calibration_datasets/c1_wikitext.jsonl"
run_all_tasks "${MODEL_DIR}/${MODEL_NAME}-gptq-w4a16-c1" "gptq_w4a16_c1"

quantize_gptq "data/calibration_datasets/c3_finmix.jsonl"
run_all_tasks "${MODEL_DIR}/${MODEL_NAME}-gptq-w4a16-c3" "gptq_w4a16_c3"

# 3. RTN (bonus — model already quantized earlier)
if [[ -d "${MODEL_DIR}/${MODEL_NAME}-rtn-w4a16" ]]; then
    run_all_tasks "${MODEL_DIR}/${MODEL_NAME}-rtn-w4a16" "rtn_w4a16"
fi

echo ""
echo "Done: $(date)"
echo "Analyze with:"
echo "  python3 scripts/analysis/calibration_significance.py --model ${MODEL_NAME} --pairs c3:c1"
