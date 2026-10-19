#!/bin/bash
# Calibration-sampling seed robustness for the GPTQ c1-vs-c3 finding.
#
# Motivation: both corpora are themselves sampled constructions — c1's 128
# chunks are drawn from the WikiText chunk pool and c3/FinMix is built from
# shuffled task training splits, both with seed 42. This script re-quantizes
# GPTQ with the corpora rebuilt under two further seeds and re-evaluates the
# four headline tasks, to show the FinQA/ConvFinQA gain is larger than
# calibration-sampling noise.
#
#   c1 variants: regenerate WikiText chunk sample with seed 43 / 44
#   c3 variants: full FinMix reconstructions under seeds 43 / 44 (ship with the repo)
#
# Tasks evaluated: T5 FinQA, T8 ConvFinQA (sensitive) + T4 CFA, T1 Sentiment
# (controls). T5/T8 are DeepSeek-adjudicated afterwards for comparability with
# the main-table numbers.
#
# Usage:
#   bash scripts/quantization/run_gptq_seed_robustness.sh
#
# Runtime: ~4 quantizations (~20 min each) + 16 evals on a single 80GB+ GPU.

set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_DIR="data/models"
CALIB_DIR="data/calibration_datasets"
OUTDIR="experiments/results/${MODEL_NAME}"
LOGDIR="experiments/logs/${MODEL_NAME}"
SEEDS=(43 44)

mkdir -p "$LOGDIR"
LOG_FILE="${LOGDIR}/seed_robustness_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"

PY="python"

# ── Step 0: prepare seed-variant calibration files ───────────────────────────
for SEED in "${SEEDS[@]}"; do
    C1_FILE="${CALIB_DIR}/c1s${SEED}_wikitext_seed${SEED}.jsonl"
    if [[ ! -f "$C1_FILE" ]]; then
        echo "[$(date +%H:%M)] preparing $C1_FILE"
        $PY scripts/quantization/prepare_c1_wikitext.py --seed "$SEED"
    fi

    C3_FILE="${CALIB_DIR}/c3s${SEED}_finmix_seed${SEED}.jsonl"
    if [[ ! -f "$C3_FILE" ]]; then
        echo "ERROR: $C3_FILE not found. The seed-variant FinMix corpora ship with the repo"
        echo "       (full reconstructions of FinMix under seeds 43/44, built with scripts/calibration/)."
        exit 1
    fi
done

# ── Step 1: quantize ─────────────────────────────────────────────────────────
quantize() {
    local calib_file="$1"       # e.g. data/calibration_datasets/c1s43_wikitext_seed43.jsonl
    local calib_name             # gptq_w4a16.py derives it: stem before first "_"
    calib_name=$(basename "$calib_file" | cut -d'_' -f1)
    local save_dir="${MODEL_DIR}/Meta-${MODEL_NAME}-gptq-w4a16-${calib_name}"
    if [[ -d "$save_dir" ]]; then
        echo "[$(date +%H:%M)] quantize ${calib_name}: exists, skipping"
        return
    fi
    echo "[$(date +%H:%M)] === quantize GPTQ ${calib_name} ==="
    $PY scripts/quantization/gptq_w4a16.py \
        --model-id "$MODEL_ID" \
        --calib-path "$calib_file" \
        --max-seq-length 2048
}

for SEED in "${SEEDS[@]}"; do
    quantize "${CALIB_DIR}/c1s${SEED}_wikitext_seed${SEED}.jsonl"
    quantize "${CALIB_DIR}/c3s${SEED}_finmix_seed${SEED}.jsonl"
done

# ── Step 2: evaluate T5 / T8 / T4 / T1 ──────────────────────────────────────
evaluate() {
    local calib_name="$1"        # c1s43 ...
    local model_path="${MODEL_DIR}/Meta-${MODEL_NAME}-gptq-w4a16-${calib_name}"
    local prefix="gptq_w4a16_${calib_name}"
    if [[ ! -d "$model_path" ]]; then
        echo "!! model missing: $model_path — skipping evals"
        return
    fi
    for task in t5 t8 t4 t1; do
        local script
        case "$task" in
            t5) script=scripts/evaluation/eval_t5_finqa.py ;;
            t8) script=scripts/evaluation/eval_t8_convfinqa.py ;;
            t4) script=scripts/evaluation/eval_t4_cfa.py ;;
            t1) script=scripts/evaluation/eval_t1_sentiment.py ;;
        esac
        local out="${OUTDIR}/${prefix}_${MODEL_NAME}_${task}.json"
        local detailed="${OUTDIR}/${prefix}_${MODEL_NAME}_${task}_detailed.json"
        if [[ -f "$detailed" ]]; then
            echo "[$(date +%H:%M)] ${prefix} ${task}: exists, skipping"
            continue
        fi
        echo "[$(date +%H:%M)] === eval ${prefix} ${task} ==="
        $PY "$script" --model "$model_path" --output "$out"
    done
}

for SEED in "${SEEDS[@]}"; do
    evaluate "c1s${SEED}"
    evaluate "c3s${SEED}"
done

# ── Step 3: adjudicate T5/T8 (needed for comparability with main table) ─────
if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    for SEED in "${SEEDS[@]}"; do
        for calib in "c1s${SEED}" "c3s${SEED}"; do
            for task in t5 t8; do
                detailed="${OUTDIR}/gptq_w4a16_${calib}_${MODEL_NAME}_${task}_detailed.json"
                adjud="${OUTDIR}/gptq_w4a16_${calib}_${MODEL_NAME}_${task}_detailed_llm_adjudicated.json"
                judge="scripts/llm_judgement/adjudicate_${task/t5/t5_finqa}"
                judge="${judge/t8/t8_convfinqa}.py"
                if [[ -f "$detailed" && ! -f "$adjud" ]]; then
                    echo "[$(date +%H:%M)] === adjudicate ${calib} ${task} ==="
                    $PY "$judge" --input "$detailed"
                fi
            done
        done
    done
else
    echo "DEEPSEEK_API_KEY not set — skipping T5/T8 adjudication (run later)."
fi

echo ""
echo "Done: $(date)"
echo "Analyze with:"
echo "  python3 scripts/analysis/calibration_significance.py --model ${MODEL_NAME} \\"
echo "      --pairs c3s43:c1s43 c3s44:c1s44 c3:c1"
