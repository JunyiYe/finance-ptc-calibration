#!/bin/bash
# Compress Llama-3.1-8B-Instruct — full paper grid: RTN (calibration-free) and
# 5 methods (GPTQ/AWQ/SmoothQuant/SparseGPT/Wanda) × 3 calibration corpora
# (c1=WikiText, c2=C4, c3=FinMix). Each corpus is consumed in full at
# max_seq_length=2048 (the method scripts' defaults).
#
# Compressed models land in data/models/Meta-Llama-3.1-8B-Instruct-<method>-<calib>;
# existing output dirs are skipped, so the sweep is safe to stop and resume.
set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_BASE="Meta-Llama-3.1-8B-Instruct"
MODEL_DIR="data/models"

CALIBS=(
    "data/calibration_datasets/c1_wikitext.jsonl"
    "data/calibration_datasets/c2_c4web.jsonl"
    "data/calibration_datasets/c3_finmix.jsonl"
)

METHODS=(
    "gptq_w4a16:gptq-w4a16"
    "awq_w4a16:awq-w4a16"
    "smoothquant_w8a8:smoothquant-w8a8"
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
        CALIB_NAME=$(basename "$CALIB" | cut -d'_' -f1)   # c1_wikitext.jsonl -> c1
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

echo "All Llama-3.1-8B compression jobs done: $(date)"
