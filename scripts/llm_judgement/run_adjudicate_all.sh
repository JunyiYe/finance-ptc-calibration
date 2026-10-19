#!/bin/bash
# Adjudicate every generative-task result of one model with the DeepSeek judge.
#
# Scans experiments/results/<model>/ for *_t5/_t8/_t10_detailed.json files
# (FinQA, ConvFinQA, TAT-QA — the three judged tasks) and adjudicates each one
# that does not have a *_detailed_llm_adjudicated.json yet, so the sweep is
# safe to stop and resume. Requires DEEPSEEK_API_KEY.
#
# Usage:
#   bash scripts/llm_judgement/run_adjudicate_all.sh                          # Llama-3.1-8B-Instruct
#   bash scripts/llm_judgement/run_adjudicate_all.sh Qwen3-8B
#   bash scripts/llm_judgement/run_adjudicate_all.sh Llama-3.1-70B-Instruct
set -uo pipefail

MODEL_NAME="${1:-Llama-3.1-8B-Instruct}"
OUTDIR="experiments/results/${MODEL_NAME}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set."
    exit 1
fi
if [[ ! -d "$OUTDIR" ]]; then
    echo "ERROR: no results directory: $OUTDIR"
    exit 1
fi

judge_script() {
    case "$1" in
        t5)  echo scripts/llm_judgement/adjudicate_t5_finqa.py ;;
        t8)  echo scripts/llm_judgement/adjudicate_t8_convfinqa.py ;;
        t10) echo scripts/llm_judgement/adjudicate_t10_tatqa.py ;;
    esac
}

for task in t5 t8 t10; do
    judge=$(judge_script "$task")
    for detailed in "$OUTDIR"/*_"${MODEL_NAME}"_"${task}"_detailed.json; do
        [[ -e "$detailed" ]] || continue
        adjud="${detailed%.json}_llm_adjudicated.json"
        if [[ -f "$adjud" ]]; then
            echo "[$(date +%H:%M)] $(basename "$detailed"): adjudicated already, skipping"
            continue
        fi
        echo "[$(date +%H:%M)] === adjudicate $(basename "$detailed") ==="
        python "$judge" --input "$detailed" || echo "!!! FAILED: $(basename "$detailed")"
    done
done

echo "All ${MODEL_NAME} adjudications done: $(date)"
