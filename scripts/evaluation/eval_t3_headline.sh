#!/bin/bash
set -euo pipefail

# ================================================================
# T3 Headline Classification Evaluation
# How to run eval_t3_headline.py
# ================================================================

export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}

# 1. Llama 3.1 8B
CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t3_headline.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num-samples 100

# 2. Llama 3.1 70B
# export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}
# CUDA_VISIBLE_DEVICES=0,1 conda run -n finquant-lmc-main python scripts/evaluation/eval_t3_headline.py \
#   --model meta-llama/Meta-Llama-3.1-70B-Instruct

# 3. Gemma-4 31B
# export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}
# CUDA_VISIBLE_DEVICES=1 conda run -n finquant-lmc-main python scripts/evaluation/eval_t3_headline.py \
#   --model google/gemma-4-31B-it

# 5. Qwen 7B
# CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t3_headline.py \
#   --model Qwen/Qwen2.5-7B-Instruct

# 6. Custom model and output path
# CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t3_headline.py \
#   --model <your_model_id> \
#   --num-samples <num> \
#   --output <output_path>

# ================================================================
# Usage
# ================================================================
#
# Args:
#   --model          : required, model ID (e.g. meta-llama/Meta-Llama-3.1-8B-Instruct)
#   --num-samples    : optional, sample count (default: all)
#   --output         : optional, output file path (default: auto-generated)
#
# Output files:
#   - experiments/bf16_<model_name>_t3_headline.json (summary results)
#   - experiments/bf16_<model_name>_t3_headline_detailed.json (detailed results)
#
# GPU selection:
#   CUDA_VISIBLE_DEVICES=0          (use GPU 0)
#   CUDA_VISIBLE_DEVICES=0,1        (use GPU 0 and 1)
#
# ================================================================
