#!/bin/bash
set -euo pipefail

# ================================================================
# T8 Causal Classification Evaluation
# How to run eval_t8_causal.py
# ================================================================

export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}

# 1. Llama 3.1 8B
CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t8_causal.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num-samples 100

# 2. Llama 3.1 70B
# CUDA_VISIBLE_DEVICES=0,1 conda run -n finquant-lmc-main python scripts/evaluation/eval_t8_causal.py \
#   --model meta-llama/Meta-Llama-3.1-70B-Instruct

# 3. Gemma-4 31B
# CUDA_VISIBLE_DEVICES=0,1 conda run -n finquant-lmc-main python scripts/evaluation/eval_t8_causal.py \
#   --model google/gemma-4-31B-it

# 4. Qwen 7B
# CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t8_causal.py \
#   --model Qwen/Qwen2.5-7B-Instruct

# 5. Custom model and output path
# CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t8_causal.py \
#   --model <your_model_id> \
#   --num-samples <num> \
#   --output <output_path>

# ================================================================
# Usage
# ================================================================
#
# Args:
#   --model          : required, model ID (e.g. meta-llama/Meta-Llama-3.1-8B-Instruct)
#   --num-samples    : optional, sample count (default: all 8628)
#   --output         : optional, output file path (default: auto-generated)
#
# Output files:
#   - experiments/bf16_<model_name>_t8_causal.json (summary results)
#   - experiments/bf16_<model_name>_t8_causal_detailed.json (detailed results)
#
# Note: dataset is heavily imbalanced (noise:causal = 93%:7%);
#       macro_f1 is the primary metric, accuracy is inflated.
#
# GPU selection:
#   CUDA_VISIBLE_DEVICES=0          (use GPU 0)
#   CUDA_VISIBLE_DEVICES=0,1        (use GPU 0 and 1)
#
# ================================================================
