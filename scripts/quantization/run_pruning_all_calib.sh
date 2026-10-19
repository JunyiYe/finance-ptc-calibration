#!/bin/bash
# SparseGPT-unstructured (50%) + Wanda-4:8 pruning for C1, C2, C3 calibration sets.
# Runs sequentially on GPU 0 (GPU 1 is reserved for eval pipeline).
set -euo pipefail

export HF_HOME=/data/yej/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ=512
N_CALIB=128

CALIB_SETS=(
    "data/calibration_datasets/c1_wikitext.jsonl"
    "data/calibration_datasets/c2_c4web.jsonl"
    "data/calibration_datasets/c3_finmix.jsonl"
)

echo "================================================================"
echo "SparseGPT-unstructured + Wanda-4:8 | C1/C2/C3"
echo "Start: $(date)"
echo "================================================================"

for CALIB in "${CALIB_SETS[@]}"; do
    CALIB_NAME=$(basename "${CALIB}" | cut -d'_' -f1)

    SPARSE_OUT="data/models/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-${CALIB_NAME}"
    if [[ -d "$SPARSE_OUT" ]]; then
        echo ">>> sparsegpt-unstructured-${CALIB_NAME}: already exists, skipping"
    else
        echo ""
        echo ">>> sparsegpt-unstructured-${CALIB_NAME} | $(date)"
        conda run -n finquant-lmc-main python scripts/quantization/sparsegpt_unstructured.py \
            --model-id "${MODEL}" \
            --calib-path "${CALIB}" \
            --max-seq-length "${MAX_SEQ}" \
            --num-calib-samples "${N_CALIB}"
        echo "Done: ${SPARSE_OUT} | $(date)"
    fi

    WANDA_OUT="data/models/Meta-Llama-3.1-8B-Instruct-wanda-4of8-${CALIB_NAME}"
    if [[ -d "$WANDA_OUT" ]]; then
        echo ">>> wanda-4of8-${CALIB_NAME}: already exists, skipping"
    else
        echo ""
        echo ">>> wanda-4of8-${CALIB_NAME} | $(date)"
        conda run -n finquant-lmc-main python scripts/quantization/wanda_4of8.py \
            --model-id "${MODEL}" \
            --calib-path "${CALIB}" \
            --max-seq-length "${MAX_SEQ}" \
            --num-calib-samples "${N_CALIB}"
        echo "Done: ${WANDA_OUT} | $(date)"
    fi
done

echo ""
echo "================================================================"
echo "All done: $(date)"
echo "================================================================"
