#!/bin/bash
# Calibration-sampling seed robustness for the GPTQ c1-vs-c3 finding.
#
# Motivation: the original c3 run drew a RANDOM 128-of-2042 calibration subset
# (llmcompressor shuffle_calibration_samples=True, unseeded), and c1's 128
# chunks were sampled from the WikiText chunk pool with seed=42. This script
# re-quantizes GPTQ with two extra sampling seeds per calibration set and
# re-evaluates the four headline tasks, to show the FinQA/ConvFinQA gain is
# larger than calibration-sampling noise.
#
#   c1 variants: regenerate WikiText chunk sample with seed 43 / 44
#   c3 variants: pre-sample an explicit 128-subset of c3_finmix (seed 43 / 44)
#
# Tasks evaluated: T5 FinQA, T9 ConvFinQA (sensitive) + T4 CFA, T1 Sentiment
# (controls). T5/T9 are DeepSeek-adjudicated afterwards for comparability with
# the main-table numbers.
#
# Usage:
#   bash scripts/quantization/run_gptq_seed_robustness.sh
#
# Runtime: ~4 quantizations (~20 min each) + 16 evals on a shared H200.

set -uo pipefail

export HF_HOME=/data/yej/.cache/huggingface
export CUDA_VISIBLE_DEVICES=1   # hard-set: ~/.bashrc exports stale 0,1,2,3; GPU 0 is full
export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}

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

PY="conda run -n finquant-lmc-main python"

# ── Step 0: prepare seed-variant calibration files ───────────────────────────
for SEED in "${SEEDS[@]}"; do
    C1_FILE="${CALIB_DIR}/c1s${SEED}_wikitext_seed${SEED}.jsonl"
    if [[ ! -f "$C1_FILE" ]]; then
        echo "[$(date +%H:%M)] preparing $C1_FILE"
        $PY scripts/quantization/prepare_c1_wikitext.py --seed "$SEED"
    fi

    C3_FILE="${CALIB_DIR}/c3s${SEED}_finmix_seed${SEED}.jsonl"
    if [[ ! -f "$C3_FILE" ]]; then
        echo "[$(date +%H:%M)] preparing $C3_FILE (128-subset of c3_finmix, seed=$SEED)"
        # NOTE: plain python3, NOT conda run — conda run does not forward stdin,
        # so a heredoc script would silently no-op.
        python3 - "$SEED" "$C3_FILE" <<'PYEOF'
import json, random, sys
seed, out = int(sys.argv[1]), sys.argv[2]
lines = open("data/calibration_datasets/c3_finmix.jsonl").read().splitlines()
random.seed(seed)
subset = random.sample(lines, 128)
with open(out, "w") as f:
    f.write("\n".join(subset) + "\n")
print(f"wrote {out}: 128 of {len(lines)} samples, seed={seed}")
PYEOF
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
        --max-seq-length 2048 \
        --num-calib-samples 128
}

for SEED in "${SEEDS[@]}"; do
    quantize "${CALIB_DIR}/c1s${SEED}_wikitext_seed${SEED}.jsonl"
    quantize "${CALIB_DIR}/c3s${SEED}_finmix_seed${SEED}.jsonl"
done

# ── Step 2: evaluate T5 / T9 / T4 / T1 ──────────────────────────────────────
evaluate() {
    local calib_name="$1"        # c1s43 ...
    local model_path="${MODEL_DIR}/Meta-${MODEL_NAME}-gptq-w4a16-${calib_name}"
    local prefix="gptq_w4a16_${calib_name}"
    if [[ ! -d "$model_path" ]]; then
        echo "!! model missing: $model_path — skipping evals"
        return
    fi
    for task in t5 t9 t4 t1; do
        local script
        case "$task" in
            t5) script=scripts/evaluation/eval_t5_finqa.py ;;
            t9) script=scripts/evaluation/eval_t9_convfinqa.py ;;
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

# ── Step 3: adjudicate T5/T9 (needed for comparability with main table) ─────
if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    for SEED in "${SEEDS[@]}"; do
        for calib in "c1s${SEED}" "c3s${SEED}"; do
            for task in t5 t9; do
                detailed="${OUTDIR}/gptq_w4a16_${calib}_${MODEL_NAME}_${task}_detailed.json"
                adjud="${OUTDIR}/gptq_w4a16_${calib}_${MODEL_NAME}_${task}_detailed_llm_adjudicated.json"
                judge="scripts/llm_judgement/adjudicate_${task/t5/t5_finqa}"
                judge="${judge/t9/t9_convfinqa}.py"
                if [[ -f "$detailed" && ! -f "$adjud" ]]; then
                    echo "[$(date +%H:%M)] === adjudicate ${calib} ${task} ==="
                    $PY "$judge" --input "$detailed"
                fi
            done
        done
    done
else
    echo "DEEPSEEK_API_KEY not set — skipping T5/T9 adjudication (run later)."
fi

echo ""
echo "Done: $(date)"
echo "Analyze with:"
echo "  python3 scripts/analysis/calibration_significance.py --model ${MODEL_NAME} \\"
echo "      --pairs c3s43:c1s43 c3s44:c1s44 c3:c1"
