#!/bin/bash
set -euo pipefail

# ================================================================
# T9 Conversational Financial QA Evaluation
# How to run eval_t9_convfinqa.py
# ================================================================

export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}

# 1. Llama 3.1 8B
CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t9_convfinqa.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num-samples 100

# 2. Llama 3.1 70B
# CUDA_VISIBLE_DEVICES=0,1 conda run -n finquant-lmc-main python scripts/evaluation/eval_t9_convfinqa.py \
#   --model meta-llama/Meta-Llama-3.1-70B-Instruct

# 3. Gemma-4 31B
# CUDA_VISIBLE_DEVICES=0,1 conda run -n finquant-lmc-main python scripts/evaluation/eval_t9_convfinqa.py \
#   --model google/gemma-4-31B-it

# 4. Qwen 7B
# CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t9_convfinqa.py \
#   --model Qwen/Qwen2.5-7B-Instruct

# ================================================================
# Usage
# ================================================================
#
# Args:
#   --model          : required, model ID
#   --num-samples    : optional, sample count (default: all 1490)
#   --output         : optional, output file path (default: auto-generated)
#
# Output files:
#   - experiments/bf16_<model_name>_t9_convfinqa.json
#   - experiments/bf16_<model_name>_t9_convfinqa_detailed.json
#
# Note: multi-turn dialogue fills placeholders with gold prior-turn answers (oracle setting)
#
# ================================================================
