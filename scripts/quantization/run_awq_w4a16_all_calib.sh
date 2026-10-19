#!/bin/bash
# Run AWQ W4A16 quantization for the calibration sets (C1, C2, C3).
set -e

export HF_HOME=/data/yej/.cache/huggingface
export CUDA_VISIBLE_DEVICES=1

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
# MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
MAX_SEQ=2048
N_CALIB=128

CALIB_SETS=(
    "data/calibration_datasets/c1_wikitext.jsonl"
    "data/calibration_datasets/c2_c4web.jsonl"
)

for CALIB in "${CALIB_SETS[@]}"; do
    CALIB_NAME=$(basename "${CALIB}" | cut -d'_' -f1)  # c1, c2, ...
    echo "=========================================="
    echo "AWQ W4A16 | model=${MODEL} | calib=${CALIB_NAME}"
    echo "=========================================="
    conda run -n finquant-lmc-main python scripts/quantization/awq_w4a16_asym.py \
        --model-id "${MODEL}" \
        --calib-path "${CALIB}" \
        --max-seq-length "${MAX_SEQ}" \
        --num-calib-samples "${N_CALIB}"
    echo "Done: data/models/$(basename ${MODEL})-awq-w4a16-${CALIB_NAME}"
    echo ""
done

echo "All 5 calibration sets done."
