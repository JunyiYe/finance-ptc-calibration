#!/bin/bash
set -euo pipefail

# ================================================================
# T10 Stock Movement Prediction Evaluation (flare-sm-bigdata)
# How to run eval_t10_stock.py
# ================================================================

export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}

# 1. Llama 3.1 8B
CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t10_stock.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num-samples 100

# 2. Llama 3.1 70B
# CUDA_VISIBLE_DEVICES=0,1 conda run -n finquant-lmc-main python scripts/evaluation/eval_t10_stock.py \
#   --model meta-llama/Meta-Llama-3.1-70B-Instruct

# 3. Gemma-4 31B
# CUDA_VISIBLE_DEVICES=0,1 conda run -n finquant-lmc-main python scripts/evaluation/eval_t10_stock.py \
#   --model google/gemma-4-31B-it

# 4. Qwen 7B
# CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t10_stock.py \
#   --model Qwen/Qwen2.5-7B-Instruct

# ================================================================
# Usage
# ================================================================
#
# Args:
#   --model          : required, model ID
#   --num-samples    : optional, sample count (default: all 1472)
#   --output         : optional, output file path (default: auto-generated)
#
# Task: predict stock movement (Rise/Fall) from historical price data and tweets
# Label distribution: Rise 55.2% / Fall 44.8% (fairly balanced)
#
# Output files:
#   - experiments/bf16_<model_name>_t10_stock.json
#   - experiments/bf16_<model_name>_t10_stock_detailed.json
#
# ================================================================
