#!/bin/bash
# Multi-model baseline runner script
# Usage: bash scripts/evaluation/run_baseline_all_models.sh 70B

set -e

MODEL_SIZE=${1:-8B}
export LD_LIBRARY_PATH="/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}"
CONDA_CMD="conda run -n finquant-lmc-main python"

# Model mapping
if [ "$MODEL_SIZE" = "70B" ]; then
    MODEL="meta-llama/Llama-3.1-70B"
    CUDA_DEVICES="0,1"  # Use two GPUs
    OUTPUT_PREFIX="baseline_70b"
    SAMPLES_T3=2000  # Use only 2000 samples for the large model to speed things up
elif [ "$MODEL_SIZE" = "8B" ]; then
    MODEL="meta-llama/Llama-3.1-8B"
    CUDA_DEVICES="0"   # Use a single GPU
    OUTPUT_PREFIX="baseline_8b"
    SAMPLES_T3=5000
else
    echo "❌ Unsupported model: $MODEL_SIZE"
    echo "Supported: 8B, 70B"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  FP16 Baseline - Llama-3.1-${MODEL_SIZE} (GPU: $CUDA_DEVICES)  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL"
echo "Output prefix: $OUTPUT_PREFIX"
echo ""

# Check GPU
echo "[Hardware Info]"
nvidia-smi -L
echo ""

RESULTS_DIR="experiments"
mkdir -p $RESULTS_DIR

# Define all tasks
declare -a TASKS=(
    "t1:970"
    "t2:496"
    "t3:$SAMPLES_T3"
    "t4:1032"
    "t5:1147"
    "t6:1068"
    "t7:98"
)

echo "[Launching Evaluation]"
echo ""

for task_spec in "${TASKS[@]}"; do
    IFS=':' read -r task samples <<< "$task_spec"

    script="scripts/evaluation/eval_${task}_*.py"
    script_file=$(ls $script | head -1)

    if [ ! -f "$script_file" ]; then
        echo "❌ Script not found: $script"
        continue
    fi

    output_file="$RESULTS_DIR/${OUTPUT_PREFIX}_${task}.json"

    echo "Running $(basename $script_file)..."

    # Pass model arguments to the existing script
    $CONDA_CMD "$script_file" \
        --model "$MODEL" \
        --output "$output_file" \
        --num-samples $samples

    if [ -f "$output_file" ]; then
        echo "✓ $task done"
    else
        echo "⚠ $task may have failed"
    fi

    echo ""
done

echo "[Results Summary]"
ls -lh $RESULTS_DIR/${OUTPUT_PREFIX}_*.json
echo ""
echo "✓ All baseline evaluations complete!"
echo ""
echo "View results:"
