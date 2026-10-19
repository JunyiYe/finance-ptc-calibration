#!/bin/bash
set -uo pipefail

# Quantize Llama-3.1-70B-Instruct: GPTQ + AWQ-asym, calib sets c1/c2/c3.
# SmoothQuant skipped (paused — see 8B findings: no real speedup on plain HF
# inference path, revisit later).
# 70B needs 2 GPUs (device_map="auto" shards across them).

export CUDA_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}

MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL_DIR="data/models"
MAX_SEQ=2048

CALIB=(
    "c1:data/calibration_datasets/c1_wikitext.jsonl:128"
    "c2:data/calibration_datasets/c2_c4web.jsonl:128"
    "c3:data/calibration_datasets/c3_finmix.jsonl:2042"
)

run_quant() {
    local script="$1" method_dir="$2" calib_name="$3" calib_path="$4" n_samples="$5"
    local out_dir="${MODEL_DIR}/Meta-Llama-3.1-70B-Instruct-${method_dir}-${calib_name}"
    if [[ -d "$out_dir" ]]; then
        echo ">>> ${method_dir} ${calib_name} (already done, skipping)"
        return
    fi
    echo ">>> ${method_dir} ${calib_name}"
    conda run -n finquant-lmc-main python "$script" \
        --model-id "$MODEL" \
        --calib-path "$calib_path" \
        --max-seq-length "$MAX_SEQ" \
        --num-calib-samples "$n_samples" || echo "!!! FAILED: ${method_dir} ${calib_name}"
}

for entry in "${CALIB[@]}"; do
    IFS=':' read -r name path nsamp <<< "$entry"
    run_quant scripts/quantization/gptq_w4a16.py "gptq-w4a16" "$name" "$path" "$nsamp"
done

for entry in "${CALIB[@]}"; do
    IFS=':' read -r name path nsamp <<< "$entry"
    run_quant scripts/quantization/awq_w4a16_asym.py "awq-w4a16-asym" "$name" "$path" "$nsamp"
done

echo "All 70B quantization jobs done."
