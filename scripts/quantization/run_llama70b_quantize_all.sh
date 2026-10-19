#!/bin/bash
# Compress Llama-3.1-70B-Instruct — paper grid (appendix): RTN
# (calibration-free) and GPTQ/AWQ × 2 calibration corpora (c1=WikiText,
# c3=FinMix). SmoothQuant and the pruning methods are not part of the 70B
# grid. Each corpus is consumed in full at max_seq_length=2048 (the method
# scripts' defaults). 70B needs 2 GPUs (device_map="auto" shards across them).
#
# Compressed models land in data/models/Meta-Llama-3.1-70B-Instruct-<method>-<calib>;
# existing output dirs are skipped. Evaluate afterwards with
# scripts/evaluation/run_llama70b_all_tasks.sh.
set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL_BASE="Meta-Llama-3.1-70B-Instruct"
MODEL_DIR="data/models"

CALIBS=(
    "data/calibration_datasets/c1_wikitext.jsonl"
    "data/calibration_datasets/c3_finmix.jsonl"
)

METHODS=(
    "gptq_w4a16:gptq-w4a16"
    "awq_w4a16:awq-w4a16"
)

# RTN is calibration-free
if [[ -d "${MODEL_DIR}/${MODEL_BASE}-rtn-w4a16" ]]; then
    echo ">>> rtn-w4a16: exists, skipping"
else
    echo ">>> rtn-w4a16 | $(date)"
    python scripts/quantization/rtn_w4a16.py --model-id "$MODEL" \
        || echo "!!! FAILED: rtn-w4a16"
fi

for M in "${METHODS[@]}"; do
    IFS=':' read -r script_name dir_name <<< "$M"
    for CALIB in "${CALIBS[@]}"; do
        CALIB_NAME=$(basename "$CALIB" | cut -d'_' -f1)
        OUT="${MODEL_DIR}/${MODEL_BASE}-${dir_name}-${CALIB_NAME}"
        if [[ -d "$OUT" ]]; then
            echo ">>> ${dir_name}-${CALIB_NAME}: exists, skipping"
            continue
        fi
        echo ">>> ${dir_name}-${CALIB_NAME} | $(date)"
        python "scripts/quantization/${script_name}.py" \
            --model-id "$MODEL" \
            --calib-path "$CALIB" \
            || echo "!!! FAILED: ${dir_name}-${CALIB_NAME}"
    done
done

echo "All Llama-3.1-70B compression jobs done: $(date)"
