#!/bin/bash
set -euo pipefail

# ================================================================
# Llama-3.1-70B-Instruct BF16 Evaluation — All 10 Tasks
# GPU: CUDA_VISIBLE_DEVICES=0,1
# ================================================================

export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/data/yej/.cache/huggingface

MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct"
PREFIX="experiments/bf16_Llama-3.1-70B-Instruct"

echo "================================================================"
echo "Llama-3.1-70B BF16 Evaluation — All 10 Tasks"
echo "Model: ${MODEL}"
echo "Start: $(date)"
echo "================================================================"

echo "[$(date +%H:%M)] T1 Sentiment..."
conda run -n finquant-lmc-main python scripts/evaluation/eval_t1_sentiment.py \
  --model "${MODEL}" --num-samples 100 --output "${PREFIX}_t1_sentiment.json"

echo "[$(date +%H:%M)] T2 FOMC..."
conda run -n finquant-lmc-main python scripts/evaluation/eval_t2_fomc.py \
  --model "${MODEL}" --num-samples 100 --output "${PREFIX}_t2_fomc.json"

echo "[$(date +%H:%M)] T3 Headline..."
conda run -n finquant-lmc-main python scripts/evaluation/eval_t3_headline.py \
  --model "${MODEL}" --num-samples 100 --output "${PREFIX}_t3_headline.json"

echo "[$(date +%H:%M)] T4 CFA..."
conda run -n finquant-lmc-main python scripts/evaluation/eval_t4_cfa.py \
  --model "${MODEL}" --num-samples 100 --output "${PREFIX}_t4_cfa.json"

echo "[$(date +%H:%M)] T5 FinQA..."
conda run -n finquant-lmc-main python scripts/evaluation/eval_t5_finqa.py \
  --model "${MODEL}" --num-samples 100 --output "${PREFIX}_t5_finqa.json"
conda run -n finquant-lmc-main python scripts/llm_judgement/adjudicate_t5_finqa.py \
  --input "${PREFIX}_t5_finqa_detailed.json"

echo "[$(date +%H:%M)] T6 Relation Cls..."
conda run -n finquant-lmc-main python scripts/evaluation/eval_t6_finred.py \
  --model "${MODEL}" --num-samples 100 --output "${PREFIX}_t6_finred.json"


echo "[$(date +%H:%M)] T8 Causal..."
conda run -n finquant-lmc-main python scripts/evaluation/eval_t8_causal.py \
  --model "${MODEL}" --num-samples 100 --output "${PREFIX}_t8_causal.json"

echo "[$(date +%H:%M)] T9 ConvFinQA..."
conda run -n finquant-lmc-main python scripts/evaluation/eval_t9_convfinqa.py \
  --model "${MODEL}" --num-samples 100 --output "${PREFIX}_t9_convfinqa.json"
conda run -n finquant-lmc-main python scripts/llm_judgement/adjudicate_t9_convfinqa.py \
  --input "${PREFIX}_t9_convfinqa_detailed.json"

echo "[$(date +%H:%M)] T10 Stock..."
conda run -n finquant-lmc-main python scripts/evaluation/eval_t10_stock.py \
  --model "${MODEL}" --num-samples 100 --output "${PREFIX}_t10_stock.json"

echo ""
echo "================================================================"
echo "70B BF16 complete: $(date)"
echo "Results: ${PREFIX}_t*.json"
echo "================================================================"
