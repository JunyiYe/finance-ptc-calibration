#!/bin/bash
# Run SmoothQuant W8A8 quantization for all 5 calibration sets (C1-C5).
set -e

export HF_HOME=/data/yej/.cache/huggingface
export CUDA_VISIBLE_DEVICES=1

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ=2048
N_CALIB=128

CALIB_SETS=(
    "data/calibration_datasets/c1_wikitext.jsonl"
    "data/calibration_datasets/c2_c4web.jsonl"
)

for CALIB in "${CALIB_SETS[@]}"; do
    CALIB_NAME=$(basename "${CALIB}" | cut -d'_' -f1)
    echo "=========================================="
    echo "SmoothQuant W8A8 | model=${MODEL} | calib=${CALIB_NAME}"
    echo "=========================================="
    conda run -n finquant-lmc-main python scripts/quantization/smoothquant_w8a8.py \
        --model-id "${MODEL}" \
        --calib-path "${CALIB}" \
        --max-seq-length "${MAX_SEQ}" \
        --num-calib-samples "${N_CALIB}"
    echo "Done: data/models/$(basename ${MODEL})-smoothquant-w8a8-${CALIB_NAME}"
    echo ""
done

echo "All 5 calibration sets done."
