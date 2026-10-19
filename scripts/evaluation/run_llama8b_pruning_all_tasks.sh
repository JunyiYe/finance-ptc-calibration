#!/bin/bash
# Evaluate all downstream tasks for the 6 new pruned models on GPU 0:
#   sparsegpt-unstructured-c1/c2/c3 + wanda-4of8-c1/c2/c3
# GPU 0 is used (GPU 1 is reserved for the existing quantization eval pipeline).
set -uo pipefail

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-$(cat /data/yej/.deepseek_key 2>/dev/null)}"

MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_DIR="data/models"
OUTDIR="experiments/results/${MODEL_NAME}"
LOGDIR="experiments/logs/${MODEL_NAME}"

mkdir -p "$LOGDIR"
LOG_FILE="${LOGDIR}/run_all_tasks_sparsegpt_wanda4of8_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $LOG_FILE"
echo "================================================================"
echo "All tasks: sparsegpt-unstructured + wanda-4of8 × C1/C2/C3"
echo "Start: $(date)"
echo "================================================================"

MODELS=(
    "sparsegpt_unstructured_c1 ${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c1"
    "sparsegpt_unstructured_c2 ${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c2"
    "sparsegpt_unstructured_c3 ${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c3"
    "wanda_4of8_c1 ${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c1"
    "wanda_4of8_c2 ${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c2"
    "wanda_4of8_c3 ${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c3"
)

# ── Simple log-prob / rotation tasks (T1/T2/T3/T4/T6/T8/T10) ────────────────

run_simple() {
    local prefix="$1" model="$2" task="$3" script="$4" n_samples="${5:-}"
    local out="${OUTDIR}/${prefix}_${MODEL_NAME}_${task}.json"
    if [[ -f "$out" ]]; then
        echo "[$(date +%H:%M)] $prefix $task — skip (exists)"; return
    fi
    echo "[$(date +%H:%M)] $prefix $task"
    local extra_args=""
    [[ -n "$n_samples" ]] && extra_args="--num-samples $n_samples"
    conda run -n finquant-lmc-main python "$script" \
        --model "$model" --output "$out" $extra_args \
        || echo "!!! FAILED: $prefix $task"
}

# ── COT generation + adjudication tasks (T5/T9/T11) ─────────────────────────

run_cot() {
    local prefix="$1" model="$2" task="$3" script="$4" judge_script="$5" min_samples="$6"
    local out="${OUTDIR}/${prefix}_${MODEL_NAME}_${task}.json"
    local detailed="${OUTDIR}/${prefix}_${MODEL_NAME}_${task}_detailed.json"
    local adjudicated="${OUTDIR}/${prefix}_${MODEL_NAME}_${task}_detailed_llm_adjudicated.json"

    if [[ -f "$detailed" ]] && python3 -c "import json; d=json.load(open('$detailed')); exit(0 if len(d)>=${min_samples} else 1)" 2>/dev/null; then
        echo "[$(date +%H:%M)] $prefix $task generate — skip (exists)"
    else
        echo "[$(date +%H:%M)] $prefix $task generate"
        conda run -n finquant-lmc-main python "$script" \
            --model "$model" --output "$out" \
            || echo "!!! FAILED generate: $prefix $task"
    fi

    if [[ -f "$adjudicated" ]] && python3 -c "import json; d=json.load(open('$adjudicated')); exit(0 if len(d)>=${min_samples} else 1)" 2>/dev/null; then
        echo "[$(date +%H:%M)] $prefix $task adjudicate — skip (exists)"
    else
        echo "[$(date +%H:%M)] $prefix $task adjudicate"
        conda run -n finquant-lmc-main python "$judge_script" \
            --input "$detailed" \
            || echo "!!! FAILED adjudicate: $prefix $task"
    fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────

for entry in "${MODELS[@]}"; do
    prefix="${entry%% *}"
    model="${entry#* }"

    echo ""
    echo "════════════════════════════════════════"
    echo "Model: $prefix  |  $(date)"
    echo "════════════════════════════════════════"

    run_simple "$prefix" "$model" "t1" "scripts/evaluation/eval_t1_sentiment.py"
    run_simple "$prefix" "$model" "t2" "scripts/evaluation/eval_t2_fomc.py"
    run_simple "$prefix" "$model" "t3" "scripts/evaluation/eval_t3_headline.py"
    run_simple "$prefix" "$model" "t4" "scripts/evaluation/eval_t4_cfa.py"
    run_cot    "$prefix" "$model" "t5" \
        "scripts/evaluation/eval_t5_finqa.py" \
        "scripts/llm_judgement/adjudicate_t5_finqa.py" 1147
    run_simple "$prefix" "$model" "t6" "scripts/evaluation/eval_t6_finred.py"
    run_simple "$prefix" "$model" "t8" "scripts/evaluation/eval_t8_causal.py" "" 1500
    run_cot    "$prefix" "$model" "t9" \
        "scripts/evaluation/eval_t9_convfinqa.py" \
        "scripts/llm_judgement/adjudicate_t9_convfinqa.py" 1490
    run_simple "$prefix" "$model" "t10" "scripts/evaluation/eval_t10_stock.py"
    run_cot    "$prefix" "$model" "t11" \
        "scripts/evaluation/eval_t11_tatqa.py" \
        "scripts/llm_judgement/adjudicate_t11_tatqa.py" 1500
done

echo ""
echo "================================================================"
echo "All done: $(date)"
echo "================================================================"
