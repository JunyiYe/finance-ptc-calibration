#!/bin/bash
set -euo pipefail


export LD_LIBRARY_PATH=/data/yej/miniconda3/envs/finquant-lmc-main/lib:${LD_LIBRARY_PATH:-}

conda run -n finquant-lmc-main python scripts/llm_judgement/adjudicate_t5_finqa.py \
  --input experiments/bf16_Llama-3.1-8B-Instruct_t5_finqa_detailed.json \


# conda run -n finquant-lmc-main python scripts/llm_judgement/adjudicate_t5_finqa.py \
#   --input experiments/bf16_Llama-3.1-8B-Instruct_t5_finqa_detailed.json

# conda run -n finquant-lmc-main python scripts/llm_judgement/adjudicate_t5_finqa.py \
#   --input experiments/bf16_Llama-3.1-70B-Instruct_t5_finqa_detailed.json
