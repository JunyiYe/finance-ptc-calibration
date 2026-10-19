#!/bin/bash
# Compress Qwen3-8B — paper grid for the model-family transfer: RTN
# (calibration-free) and 4 methods (GPTQ/AWQ/SparseGPT/Wanda) × 2 calibration
# corpora (c1=WikiText, c3=FinMix). SmoothQuant is not part of the Qwen grid.
# Each corpus is consumed in full at max_seq_length=2048 (the method defaults).
#
# Compressed models land in data/models/Qwen3-8B-<method>-<calib>; existing
# output dirs are skipped. Evaluate afterwards with
# scripts/evaluation/run_qwen3_8b_all_tasks.sh.
set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="Qwen/Qwen3-8B"
MODEL_BASE="Qwen3-8B"
MODEL_DIR="data/models"

CALIBS=(
    "data/calibration_datasets/c1_wikitext.jsonl"
    "data/calibration_datasets/c3_finmix.jsonl"
)

METHODS=(
    "gptq_w4a16:gptq-w4a16"
    "awq_w4a16:awq-w4a16"
    "sparsegpt_unstructured:sparsegpt-unstructured"
    "wanda_4of8:wanda-4of8"
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

echo "All Qwen3-8B compression jobs done: $(date)"
