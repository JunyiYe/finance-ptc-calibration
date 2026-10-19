"""
T10 Stock Movement Prediction Evaluation Script (flare-sm-bigdata)
Supports multiple model architectures: Llama, Gemma, Qwen

Eval method: A/B rotation log-prob (2 rotations of Rise/Fall → A/B).
Rise=2 tokens, Fall=1 token → A/B rotation removes length bias.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t10_stock.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --num-samples 1472
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from sklearn.metrics import f1_score, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation_datasets"

ENABLE_THINKING = False
MAX_INPUT_LENGTH = 2048
DEVICE_MAP = "auto"
DTYPE = torch.bfloat16

model = None
processor_or_tokenizer = None
model_type = None

STOCK_LABELS = ("Rise", "Fall")
INVALID_LABEL_IDX = -1

# A/B rotation: 2 cyclic permutations
LETTERS = ("A", "B")
ROTATIONS = [
    ("Rise", "Fall"),
    ("Fall", "Rise"),
]

# ============================================================
# CURATED "OBVIOUS" SANITY-CHECK EXAMPLES
# ============================================================
# 12 examples (6 Rise / 6 Fall), drawn from the BF16 baseline run (re-run with
# the corrected prompt — see score_ab_rotation): the highest log-prob-margin,
# correctly-predicted example per gold class.
#
# IMPORTANT DEVIATION from T1/T2/T3/T8's curation convention: those tasks'
# "obvious" examples were additionally human-eyeballed for unambiguity, because
# a human reading the input text *can* independently judge the gold label
# (a sentence is plainly "causal" or plainly "hawkish"). T10 has no such
# anchor — its gold label is the *actual subsequent stock price movement*,
# which is not determinable by inspecting an OHLC percentage-change table (that
# is precisely what makes stock-movement prediction hard/interesting). There is
# no "obviously Rise" or "obviously Fall" pattern a human can point to in this
# input format, so the human-eyeballing filter is dropped — these 12 are
# selected purely as "cases the strong reference model handles confidently and
# correctly," which is the only kind of "obvious" that meaningfully exists for
# this task. The check's semantics shift accordingly: not "does the quantized
# model still get the *objectively obvious* case right" but "does it still
# agree with the *reference model's most confident, correct* calls" — a
# regression/consistency anchor rather than a ground-truth-obviousness anchor.
#
# Despite the dataset being only mildly imbalanced (Rise 55.2% / Fall 44.8%),
# these are still deliberately balanced 6/6 for consistency with the established
# convention. Note also a striking asymmetry in the source margins themselves:
# the top correct-Rise margins reach ~3.4-3.5, while the top correct-Fall
# margins only reach ~1.0-1.6 — i.e. even the reference model is dramatically
# more confident (and on this evidence, better-calibrated) when it predicts
# "Rise" than when it predicts "Fall". This Rise-leaning bias is itself a
# pattern worth tracking across quantized configs.
OBVIOUS_SANITY_EXAMPLES = [
    {"id": "bigdatasm5772", "gold_label": "Rise", "query":
        "Examine the data and tweets to deduce if the closing price of $ma will boost or lower at 2020-11-03. Kindly confirm either Rise or Fall.\nContext: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30\n2020-10-20,0.4,1.5,-0.0,0.0,0.0,1.5,3.2,2.9,2.2,2.0,1.8\n2020-10-21,-0.3,0.9,-0.6,0.5,0.5,0.5,2.4,2.3,1.8,1.4,1.3\n2020-10-22,-0.1,0.5,-1.3,0.1,0.1,0.1,2.0,2.1,1.8,1.2,1.2\n2020-10-23,1.8,1.8,-0.4,-1.2,-1.2,0.7,2.5,3.1,3.0,2.4,2.5\n2020-10-26,2.6,2.9,-1.2,-3.5,-3.5,3.5,5.2,6.4,6.5,6.0,6.0\n2020-10-27,0.6,0.8,-1.5,-0.3,-0.3,2.9,4.6,6.3,6.5,6.1,6.0\n2020-10-28,3.5,3.6,-1.0,-8.1,-8.1,9.1,12.0,14.5,15.1,15.0,14.8\n2020-10-29,0.0,0.7,-1.9,0.6,0.6,5.7,9.8,12.6,13.5,13.8,13.6\n2020-10-30,0.4,1.1,-2.6,-1.5,-1.5,4.5,9.8,12.9,14.5,15.0,14.9\n2020-11-02,1.5,1.7,-0.6,0.5,0.5,2.1,7.8,10.9,13.0,13.8,13.9\n\n\nAnswer:"},
    {"id": "bigdatasm7066", "gold_label": "Rise", "query":
        "With the help of the data and tweets given, can you forecast whether the closing price of $wmt will climb or drop at 2020-12-28? Please state either Rise or Fall.\nContext: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30\n2020-12-11,-0.1,0.3,-0.7,-0.0,-0.0,0.4,1.3,1.7,1.8,1.3,0.4\n2020-12-14,1.4,1.4,-0.2,-0.9,-0.9,1.1,1.8,2.5,2.6,2.2,1.5\n2020-12-15,-0.0,0.5,-0.5,-0.0,-0.0,0.7,1.4,2.3,2.5,2.4,1.7\n2020-12-16,0.6,1.0,-0.2,-0.1,-0.1,0.5,1.2,2.2,2.4,2.5,1.8\n2020-12-17,-0.0,0.2,-0.7,0.5,0.5,-0.1,0.6,1.5,1.9,2.0,1.5\n2020-12-18,0.5,0.8,-0.5,-0.1,-0.1,-0.1,0.5,1.3,1.8,2.0,1.7\n2020-12-21,-0.3,0.5,-0.6,0.0,0.0,-0.1,0.4,1.0,1.7,1.9,1.7\n2020-12-22,0.7,1.4,-0.7,-1.2,-1.2,0.9,1.3,1.9,2.7,2.9,2.9\n2020-12-23,0.2,0.7,-0.6,-0.7,-0.7,1.3,1.7,2.3,3.1,3.5,3.6\n2020-12-24,0.0,0.6,-0.2,0.2,0.2,0.7,1.2,1.8,2.7,3.1,3.3\n\n\nAnswer:"},
    {"id": "bigdatasm6595", "gold_label": "Rise", "query":
        "Scrutinize the data and tweets to envisage if the closing price of $nee will swell or contract at 2020-12-07. Please affirm either Rise or Fall.\nContext: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30\n2020-11-20,-0.4,1.0,-0.4,0.6,0.6,0.8,1.1,0.5,0.1,-0.1,-0.0\n2020-11-23,1.9,1.9,-0.6,-1.3,-1.3,1.3,2.3,1.7,1.3,1.2,1.2\n2020-11-24,0.7,0.9,-0.6,0.5,0.5,0.3,1.6,1.1,0.7,0.7,0.6\n2020-11-25,-1.7,0.1,-1.9,1.3,1.8,-1.3,-0.4,-0.4,-1.0,-1.0,-1.1\n2020-11-27,1.2,1.3,-0.6,-1.1,-1.1,-0.1,0.6,0.8,0.2,0.1,0.0\n2020-11-30,2.3,2.5,-0.5,-2.0,-2.0,1.5,2.3,2.7,2.4,2.1,2.0\n2020-12-01,-0.0,1.9,-0.3,0.8,0.8,0.7,1.1,1.8,1.6,1.3,1.2\n2020-12-02,-0.2,0.1,-1.9,0.2,0.2,0.4,0.6,1.5,1.3,1.0,1.0\n2020-12-03,0.9,1.6,-0.4,-1.3,-1.3,1.0,1.6,2.5,2.6,2.3,2.3\n2020-12-04,0.8,1.4,-0.8,-1.1,-1.1,1.5,2.5,3.3,3.7,3.4,3.3\n\n\nAnswer:"},
    {"id": "bigdatasm5842", "gold_label": "Rise", "query":
        "Contemplate the data and tweets to guess whether the closing price of $cmcsa will surge or decline at 2020-11-05. Please declare either Rise or Fall.\nContext: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30\n2020-10-22,-1.4,0.3,-1.7,1.3,1.3,-0.8,-0.0,-0.1,0.5,0.5,0.5\n2020-10-23,0.8,1.1,-1.0,-0.1,-0.1,-0.9,0.1,0.1,0.5,0.6,0.7\n2020-10-26,2.1,2.4,-0.5,-2.9,-2.9,1.8,2.6,2.7,3.2,3.5,3.6\n2020-10-27,0.4,0.9,-0.2,-0.8,-0.8,2.2,2.8,3.4,3.7,4.0,4.1\n2020-10-28,2.2,2.6,-0.2,-3.4,-3.4,4.5,5.6,6.5,6.8,7.3,7.4\n2020-10-29,-2.3,1.5,-2.7,2.6,2.6,0.9,2.4,3.5,3.7,4.4,4.5\n2020-10-30,0.5,1.9,-1.2,-1.9,-1.9,1.6,3.6,5.1,5.4,6.1,6.3\n2020-11-02,3.3,3.7,-1.0,-2.0,-2.0,2.5,5.0,6.4,7.1,7.8,8.2\n2020-11-03,-1.6,0.8,-2.0,2.9,2.9,-0.8,1.6,2.9,3.8,4.4,4.9\n2020-11-04,1.3,2.1,-0.7,2.5,2.5,-2.5,-1.1,0.1,1.1,1.6,2.2\n\n\nAnswer:"},
    {"id": "bigdatasm5763", "gold_label": "Rise", "query":
        "Assess the data and tweets to estimate whether the closing price of $cmcsa will escalate or deflate at 2020-11-03. Respond with either Rise or Fall.\nContext: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30\n2020-10-20,0.8,0.9,-0.3,-0.1,-0.1,1.4,2.0,2.2,2.5,2.6,2.3\n2020-10-21,-1.2,0.9,-1.2,0.6,0.6,0.5,1.3,1.4,1.9,1.8,1.8\n2020-10-22,-1.4,0.3,-1.7,1.3,1.3,-0.8,-0.0,-0.1,0.5,0.5,0.5\n2020-10-23,0.8,1.1,-1.0,-0.1,-0.1,-0.9,0.1,0.1,0.5,0.6,0.7\n2020-10-26,2.1,2.4,-0.5,-2.9,-2.9,1.8,2.6,2.7,3.2,3.5,3.6\n2020-10-27,0.4,0.9,-0.2,-0.8,-0.8,2.2,2.8,3.4,3.7,4.0,4.1\n2020-10-28,2.2,2.6,-0.2,-3.4,-3.4,4.5,5.6,6.5,6.8,7.3,7.4\n2020-10-29,-2.3,1.5,-2.7,2.6,2.6,0.9,2.4,3.5,3.7,4.4,4.5\n2020-10-30,0.5,1.9,-1.2,-1.9,-1.9,1.6,3.6,5.1,5.4,6.1,6.3\n2020-11-02,3.3,3.7,-1.0,-2.0,-2.0,2.5,5.0,6.4,7.1,7.8,8.2\n\n\nAnswer:"},
    {"id": "bigdatasm7017", "gold_label": "Rise", "query":
        "Given the data and tweets, could you project whether the closing price of $ppl will grow or shrink at 2020-12-23? Please specify either Rise or Fall.\nContext: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30\n2020-12-09,-0.8,0.3,-1.1,-0.6,0.9,-1.5,-1.7,-0.7,-0.3,-0.8,-1.5\n2020-12-10,2.5,2.9,-0.1,-2.6,-2.6,1.2,0.7,1.8,2.3,1.9,1.3\n2020-12-11,-0.3,0.5,-0.5,-0.1,-0.1,1.2,0.6,1.8,2.3,2.1,1.6\n2020-12-14,2.6,3.0,-0.1,-1.7,-1.7,2.4,2.2,3.3,3.8,4.0,3.4\n2020-12-15,-0.4,0.5,-2.2,1.2,1.2,0.7,1.0,1.7,2.2,2.7,2.2\n2020-12-16,2.5,3.2,-0.1,-2.2,-2.2,1.8,2.9,3.2,4.2,4.7,4.3\n2020-12-17,0.2,1.7,0.0,0.1,0.1,1.1,2.6,2.7,3.8,4.3,4.2\n2020-12-18,1.5,2.3,-0.8,-1.5,-1.5,1.8,3.6,3.8,5.0,5.7,5.7\n2020-12-21,-0.6,0.1,-2.1,-0.9,-0.9,2.0,3.9,4.4,5.5,6.2,6.5\n2020-12-22,1.2,1.3,-0.4,-1.4,-1.4,2.3,4.5,5.5,6.5,7.2,7.8\n\n\nAnswer:"},
    {"id": "bigdatasm6921", "gold_label": "Fall", "query":
        "Examine the data and tweets to deduce if the closing price of $orcl will boost or lower at 2020-12-21. Kindly confirm either Rise or Fall.\nContext: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30\n2020-12-07,0.3,0.7,-0.3,-0.3,-0.3,-0.7,-2.4,-3.4,-3.8,-4.1,-4.4\n2020-12-08,-0.1,1.2,-1.4,-1.3,-1.3,0.7,-0.7,-2.0,-2.4,-2.7,-3.0\n2020-12-09,-1.4,0.3,-1.6,1.2,1.2,-0.3,-1.5,-2.8,-3.3,-3.6,-4.0\n2020-12-10,0.1,0.6,-0.7,-0.4,-0.4,0.2,-0.7,-2.1,-2.7,-3.0,-3.4\n2020-12-11,0.5,0.6,-2.0,1.9,1.9,-1.5,-2.1,-3.4,-4.2,-4.6,-5.0\n2020-12-14,0.1,1.5,-0.1,0.2,0.2,-1.4,-1.8,-3.1,-4.1,-4.6,-4.9\n2020-12-15,-1.2,0.5,-1.6,1.8,1.8,-2.2,-3.1,-4.2,-5.4,-5.9,-6.3\n2020-12-16,-1.4,0.7,-1.5,1.5,1.5,-2.7,-3.9,-5.1,-6.3,-6.9,-7.4\n2020-12-17,-1.0,0.7,-1.4,1.3,1.3,-2.7,-4.5,-5.7,-7.0,-7.7,-8.2\n2020-12-18,-2.4,0.4,-2.5,2.3,2.3,-3.5,-5.8,-7.0,-8.4,-9.3,-9.8\n\n\nAnswer:"},
    {"id": "bigdatasm6730", "gold_label": "Fall", "query":
        "Examine the data and tweets to deduce if the closing price of $ge will boost or lower at 2020-12-11. Kindly confirm either Rise or Fall.\nContext: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30\n2020-11-27,1.4,1.5,-0.9,-1.0,-1.0,-1.6,-4.7,-8.6,-12.9,-16.1,-18.3\n2020-11-30,2.1,2.1,-2.2,-2.1,-2.1,1.4,-1.8,-5.3,-9.7,-13.3,-15.6\n2020-12-01,1.8,3.3,-0.4,-0.3,-0.3,1.8,-0.9,-4.1,-8.1,-12.0,-14.4\n2020-12-02,-3.4,0.5,-4.0,2.8,2.8,-0.9,-2.8,-5.7,-9.3,-13.1,-15.7\n2020-12-03,-1.9,0.9,-2.0,1.6,1.6,-2.3,-3.6,-6.2,-9.5,-13.3,-16.0\n2020-12-04,-1.9,0.5,-3.4,2.6,2.6,-4.0,-4.9,-7.3,-10.5,-14.2,-17.2\n2020-12-07,0.3,2.2,-0.9,-0.2,-0.2,-2.5,-3.8,-6.1,-9.1,-12.8,-16.0\n2020-12-08,-1.3,0.6,-1.6,0.9,0.9,-2.0,-3.8,-6.1,-8.9,-12.3,-15.7\n2020-12-09,-3.4,0.9,-3.4,3.9,3.9,-4.0,-6.6,-8.7,-11.3,-14.4,-17.7\n2020-12-10,-2.1,1.4,-2.3,-0.6,-0.6,-2.1,-5.3,-7.2,-9.6,-12.6,-16.0\n\n\nAnswer:"},
    {"id": "bigdatasm6707", "gold_label": "Fall", "query":
        "By reviewing the data and tweets, can we predict if the closing price of $jnj will go upwards or downwards at 2020-12-10? Please indicate either Rise or Fall.\nContext: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30\n2020-11-25,0.3,0.3,-0.6,-0.1,-0.1,0.6,1.9,1.3,-0.2,-0.3,-0.2\n2020-11-27,0.3,0.5,-0.5,0.2,0.2,0.1,1.4,1.3,-0.1,-0.6,-0.4\n2020-11-30,-0.3,0.1,-1.1,0.5,0.5,-0.5,0.6,1.0,-0.3,-1.0,-1.0\n2020-12-01,-0.8,1.1,-1.1,1.9,1.9,-1.8,-1.4,-0.8,-1.8,-2.8,-2.7\n2020-12-02,-0.2,1.1,-0.3,0.5,0.5,-1.7,-1.9,-1.2,-1.9,-3.1,-3.1\n2020-12-03,-0.6,0.4,-0.9,0.6,0.6,-1.6,-2.3,-1.7,-2.1,-3.3,-3.5\n2020-12-04,-0.3,0.0,-0.7,0.9,0.9,-1.6,-2.8,-2.4,-2.6,-3.7,-4.2\n2020-12-07,0.3,0.6,-0.3,-0.9,-0.9,-0.1,-1.7,-1.6,-1.5,-2.6,-3.2\n2020-12-08,-1.0,0.4,-1.3,1.7,1.7,-1.3,-2.9,-3.2,-2.9,-3.9,-4.7\n2020-12-09,-0.2,0.3,-0.9,1.0,1.0,-1.6,-3.3,-3.9,-3.7,-4.4,-5.4\n\n\nAnswer:"},
    {"id": "bigdatasm6685", "gold_label": "Fall", "query":
        "Examine the data and tweets to deduce if the closing price of $mrk will boost or lower at 2020-12-10. Kindly confirm either Rise or Fall.\nContext: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30\n2020-11-25,0.1,0.5,-0.6,-0.1,-0.1,0.3,0.4,0.6,-0.5,-0.8,-0.9\n2020-11-27,0.6,1.0,-0.4,-0.2,-0.2,0.4,0.7,0.8,-0.0,-0.5,-0.7\n2020-11-30,-0.5,0.2,-1.2,0.7,0.7,-0.3,-0.0,0.1,-0.3,-1.2,-1.3\n2020-12-01,-0.7,1.1,-0.9,1.4,1.4,-1.4,-1.3,-1.2,-1.5,-2.4,-2.6\n2020-12-02,0.1,1.1,-0.7,0.3,0.3,-1.3,-1.6,-1.5,-1.5,-2.6,-2.8\n2020-12-03,0.6,0.6,-0.8,-0.5,-0.5,-0.5,-1.0,-1.0,-1.0,-1.9,-2.2\n2020-12-04,-1.0,0.0,-1.0,0.6,0.6,-0.6,-1.4,-1.4,-1.5,-2.2,-2.7\n2020-12-07,-0.5,1.2,-0.6,0.7,0.7,-0.8,-1.8,-2.0,-2.0,-2.5,-3.2\n2020-12-08,-1.1,0.7,-1.1,0.8,0.8,-1.2,-2.3,-2.6,-2.7,-3.0,-3.8\n2020-12-09,0.4,0.5,-1.1,0.3,0.3,-1.2,-2.2,-2.7,-2.9,-3.0,-3.9\n\n\nAnswer:"},
    {"id": "bigdatasm6677", "gold_label": "Fall", "query":
        "Utilize the data and tweets at hand to foresee if the closing price of $agfs will elevate or diminish at 2020-12-09. Only answer with Rise or Fall.\nContext: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30\n2020-11-24,0.0,0.9,-2.6,2.2,2.2,-4.2,-4.0,-3.3,-4.7,-4.8,-3.9\n2020-11-25,-2.5,0.0,-4.2,1.3,1.3,-3.4,-5.2,-3.8,-5.3,-5.9,-5.0\n2020-11-27,-0.8,0.0,-4.6,-0.4,-0.4,-1.8,-4.4,-3.3,-4.3,-5.3,-4.7\n2020-11-30,1.7,1.7,-4.3,-3.0,-3.0,1.7,-1.3,-0.3,-0.9,-2.3,-1.9\n2020-12-01,-6.5,0.0,-7.3,7.4,7.4,-3.9,-7.2,-6.9,-7.1,-8.6,-8.5\n2020-12-02,3.0,4.3,-3.9,-5.7,-5.7,1.7,-0.8,-1.5,-1.5,-2.8,-3.0\n2020-12-03,-0.8,1.7,-3.0,1.3,1.3,0.3,-1.1,-2.9,-2.2,-3.5,-4.2\n2020-12-04,-5.2,0.0,-5.6,5.9,5.9,-4.3,-5.6,-7.7,-7.4,-8.3,-9.3\n2020-12-07,0.8,3.3,-2.4,-2.0,-2.0,-1.1,-2.8,-5.3,-5.1,-5.8,-7.1\n2020-12-08,-4.7,2.0,-6.3,4.1,4.1,-4.4,-5.6,-8.2,-8.5,-8.9,-10.3\n\n\nAnswer:"},
    {"id": "bigdatasm6360", "gold_label": "Fall", "query":
        "Utilize the data and tweets at hand to foresee if the closing price of $xom will elevate or diminish at 2020-11-25. Only answer with Rise or Fall.\nContext: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30\n2020-11-11,1.7,1.8,-0.8,-1.0,-1.0,-4.7,-7.9,-8.8,-9.0,-8.7,-9.0\n2020-11-12,1.8,3.0,-1.0,-3.4,-3.4,0.3,-3.7,-5.4,-5.6,-5.4,-5.5\n2020-11-13,-1.9,0.5,-1.9,2.4,2.4,0.2,-4.8,-7.1,-7.4,-7.4,-7.4\n2020-11-16,-1.5,0.0,-2.8,5.8,5.8,-4.2,-8.7,-11.2,-11.7,-11.9,-12.0\n2020-11-17,-2.2,0.1,-3.3,1.3,1.3,-4.5,-8.4,-11.2,-12.1,-12.6,-12.6\n2020-11-18,4.9,5.2,-0.0,-3.8,-3.8,-0.4,-3.5,-6.6,-8.1,-8.7,-8.8\n2020-11-19,-1.4,0.3,-2.2,0.5,0.5,0.3,-2.6,-6.1,-8.1,-8.8,-9.0\n2020-11-20,1.1,1.3,-0.4,-1.2,-1.2,2.0,-0.1,-4.0,-6.4,-7.3,-7.6\n2020-11-23,-4.7,0.2,-5.3,6.6,6.6,-3.7,-5.4,-8.9,-11.3,-12.3,-12.8\n2020-11-24,-3.5,0.2,-4.2,6.7,6.7,-8.1,-10.1,-13.1,-15.7,-16.9,-17.6\n\n\nAnswer:"},
]


def sanitize_model_name(model_id: str) -> str:
    model_name = model_id.rstrip("/").split("/")[-1]
    if model_name.startswith("Meta-Llama-"):
        model_name = model_name.replace("Meta-Llama-", "Llama-", 1)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")

# Per-base-model log subdirectories (experiments/logs/<base_model>/...).
KNOWN_MODEL_LOG_DIRS = (
    "Llama-3.1-70B-Instruct",
    "Llama-3.1-8B-Instruct",
    "gemma-4-31B-it",
    "Qwen3-8B",
)


def model_log_dir(model_name: str, base_log_dir: Path) -> Path:
    """Route a model's log file into its base-model subdirectory, if recognized."""
    for base in KNOWN_MODEL_LOG_DIRS:
        if model_name == base or model_name.startswith(base + "-"):
            return base_log_dir / base
    return base_log_dir


def setup_model(model_id: str):
    global model, processor_or_tokenizer, model_type

    logger.info(f"Detecting model type from: {model_id}")
    model_id_lower = model_id.lower()

    if "llama" in model_id_lower:
        model_type = "llama"
        logger.info("Loading Llama model with AutoTokenizer")
        processor_or_tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE, device_map=DEVICE_MAP)

    elif "gemma" in model_id_lower:
        model_type = "gemma"
        logger.info("Loading Gemma model with AutoProcessor")
        processor_or_tokenizer = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE, device_map=DEVICE_MAP)

    elif "qwen" in model_id_lower:
        model_type = "qwen"
        logger.info("Loading Qwen model with AutoTokenizer")
        processor_or_tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE, device_map=DEVICE_MAP)

    else:
        raise ValueError(f"Unsupported model: {model_id}")

    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    logger.info(f"✓ Model loaded: {model_type.upper()}")


def _get_letter_token_ids():
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    ids = {}
    for letter in LETTERS:
        toks = tokenizer.encode(letter, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Letter '{letter}' tokenizes to {len(toks)} tokens — expected 1")
        ids[letter] = toks[0]
    return ids


def score_ab_rotation(query: str) -> Tuple[str, Dict[str, float], list, bool]:
    """
    Score Rise/Fall via A/B single-token options with 2 cyclic rotations.

    Each of the 2 cyclic rotations maps the labels onto fixed single-token
    letters "A"/"B" in a different order; averaging the per-label log-probs
    across rotations cancels any positional/identity bias the model has
    toward a given letter (each label occupies each position exactly once).
    This also sidesteps the token-length mismatch between "Rise" (2 tokens)
    and "Fall" (1 token) that rules out direct single-token label scoring.

    Returns (best_label, label_scores, rotation_details, rotation_consistent):
      - label_scores: per-label log-prob averaged over the 2 rotations (i.e.
        sum of the 2 single-letter log-probs / len(ROTATIONS)) — on the same
        scale as a typical single-token log-prob, so margins/magnitudes read
        naturally. argmax is unaffected by this scaling.
      - rotation_details: one entry per rotation — full prompt, raw per-letter
        log-probs, and that rotation's argmax label (mapped back through the
        rotation's letter->label assignment). Persisted into detailed_results
        so every prompt and its scored log-probs can be inspected post-hoc.
      - rotation_consistent: True iff both rotations independently picked the
        same label as their argmax — i.e. the model is tracking label content
        rather than letter position. This is the core assumption the rotation
        scheme relies on; tracking it is this method's analogue of T1's
        tokenization boundary-merge assertion (mirrors T2's
        score_abc_rotation/T8's score_ab_rotation rotation_consistent design).
    """
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    letter_ids = _get_letter_token_ids()
    label_scores = {label: 0.0 for label in STOCK_LABELS}
    rotation_details = []

    context_start = query.find("\nContext:")
    context_part = query[context_start:] if context_start != -1 else "\n" + query
    # Strip the dataset's own trailing "Answer:" cue so it isn't stranded
    # mid-prompt — the new closing block re-adds it as the final cue,
    # matching the convention in score_abc_rotation (T2) / score_ab_rotation (T8).
    context_part = re.sub(r"\n+Answer:\s*$", "", context_part)

    for rotation in ROTATIONS:
        choice_str = "\n".join(f"{l}. {v}" for l, v in zip(LETTERS, rotation))
        modified_query = (
            f"Based on the data provided, predict the stock price movement."
            f"{context_part}\n\n"
            f"Choose one of the following options:\n{choice_str}\n\n"
            f"Answer with only A or B.\nAnswer:"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": modified_query},
        ]
        try:
            prefix = processor_or_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING,
            )
        except TypeError:
            prefix = processor_or_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        inp = torch.tensor([prefix_ids], device=model.device)
        with torch.no_grad():
            lp = torch.log_softmax(model(inp).logits[0], dim=-1)
        pos = len(prefix_ids) - 1

        letter_logprobs = {letter: lp[pos, letter_ids[letter]].item() for letter in LETTERS}
        argmax_letter = max(letter_logprobs, key=letter_logprobs.__getitem__)
        argmax_label = rotation[LETTERS.index(argmax_letter)]

        rotation_details.append({
            "prompt": modified_query,
            "letter_order": dict(zip(LETTERS, rotation)),
            "letter_logprobs": letter_logprobs,
            "argmax_label": argmax_label,
        })

        for letter, label in zip(LETTERS, rotation):
            label_scores[label] += letter_logprobs[letter]

    # Normalize sum -> average per-rotation log-prob. argmax is invariant to
    # this (positive constant divisor), so `best`/predictions are unaffected;
    # it only rescales `label_scores`/`margin` to read as a typical
    # single-token log-prob magnitude instead of a sum of 2.
    label_scores = {label: total / len(ROTATIONS) for label, total in label_scores.items()}
    best = max(label_scores, key=label_scores.__getitem__)
    rotation_consistent = len({d["argmax_label"] for d in rotation_details}) == 1
    return best, label_scores, rotation_details, rotation_consistent


def run_obvious_sanity_checks() -> list:
    """
    Re-score the 12 curated "obvious" examples (OBVIOUS_SANITY_EXAMPLES) and
    record whether the model reproduces the expected (highest-margin, BF16-
    correct) label.

    score_ab_rotation() has no fail-fast assertion to wrap (the AB scheme
    scores fixed single-token letters, so there's no tokenization
    boundary-merge risk the way score_logprob()/score_choices() has in
    T1/T3) — these calls simply re-run the same scoring path the main loop
    uses, on cases the BF16 reference model called correctly with the
    largest margin. Mirrors T8's run_obvious_sanity_checks() for its own
    AB-rotation scheme.
    """
    records = []
    for ex in OBVIOUS_SANITY_EXAMPLES:
        pred_label, label_scores, rotation_details, rotation_consistent = score_ab_rotation(ex["query"])
        ranked = sorted(label_scores.values(), reverse=True)
        records.append({
            "id": ex["id"],
            "query": ex["query"],
            "gold_label": ex["gold_label"],
            "predicted_label": pred_label,
            "label_scores": label_scores,
            "margin": ranked[0] - ranked[1],
            "rotation_consistent": rotation_consistent,
            "rotations": rotation_details,
            "matches_expected": pred_label == ex["gold_label"],
        })
    return records


def evaluate_t10_stock(num_samples: int = None) -> Tuple[Dict[str, Any], list]:
    """Evaluate T10 Stock Movement Prediction task (flare-sm-bigdata)"""

    logger.info("=" * 60)
    logger.info("T10 Stock Movement Prediction Evaluation (flare-sm-bigdata)")
    logger.info("=" * 60)

    data_path = DATA_DIR / "flare_sm_bigdata_test.json"
    if not data_path.exists():
        logger.error(f"Dataset file not found: {data_path}")
        return {"error": "dataset_not_found"}, []

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if num_samples is not None:
        import random as _rnd; _rnd.seed(42); _rnd.shuffle(dataset)
        dataset = dataset[:num_samples]

    logger.info(f"Loaded {len(dataset)} samples")

    predictions = []
    gold_labels = []
    label_map = {"Rise": 1, "Fall": 0}
    reverse_map = {0: "Fall", 1: "Rise"}
    detailed_results = []
    rotation_consistent_count = 0

    for i, example in enumerate(dataset):
        if i % 200 == 0:
            logger.info(f"  Progress: {i}/{len(dataset)}")

        query = example.get("query", "").strip()
        answer = example.get("answer", "").strip()
        example_id = example.get("id", str(i))

        if not query or not answer:
            continue

        pred_label, label_scores, rotation_details, rotation_consistent = score_ab_rotation(query)
        if rotation_consistent:
            rotation_consistent_count += 1

        pred_idx = label_map.get(pred_label, INVALID_LABEL_IDX)
        gold_idx = label_map.get(answer, INVALID_LABEL_IDX)
        if gold_idx == INVALID_LABEL_IDX:
            raise ValueError(f"Invalid gold label for example {example_id}: {answer}")

        predictions.append(pred_idx)
        gold_labels.append(gold_idx)

        detailed_results.append({
            "id": example_id,
            "gold_label": answer,
            "predicted_label": pred_label,
            "label_scores": label_scores,
            "rotation_consistent": rotation_consistent,
            "rotations": rotation_details,
            "is_correct": pred_label == answer,
        })

        if i < 3:
            logger.info(f"\nSample {i}:")
            logger.info(f"  Ground truth: {answer}")
            logger.info(f"  Prediction: {pred_label}")
            logger.info(f"  Scores: {label_scores}")

    accuracy = accuracy_score(gold_labels, predictions)
    valid_label_ids = list(reverse_map.keys())
    weighted_f1 = f1_score(gold_labels, predictions, labels=valid_label_ids, average="weighted", zero_division=0)
    macro_f1 = f1_score(gold_labels, predictions, labels=valid_label_ids, average="macro", zero_division=0)

    per_class_f1 = {}
    for label_idx in range(2):
        label_name = reverse_map[label_idx]
        binary_gold = [1 if g == label_idx else 0 for g in gold_labels]
        binary_pred = [1 if p == label_idx else 0 for p in predictions]
        per_class_f1[label_name] = f1_score(binary_gold, binary_pred, zero_division=0)

    invalid_count = sum(1 for pred in predictions if pred == INVALID_LABEL_IDX)

    results = {
        "task": "T10_stock_classification",
        "dataset": "flare_sm_bigdata",
        "eval_method": "ab_rotation",
        "num_samples": len(gold_labels),
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "invalid_predictions": invalid_count,
        "rotation_consistency": {
            "description": (
                "Fraction of samples where both A/B rotations independently "
                "picked the same label as their argmax — i.e. the model is "
                "tracking label content rather than letter position. This is "
                "the core assumption the rotation-averaging scheme relies on; "
                "per-sample detail is in 'rotation_consistent' / 'rotations' "
                "within the _detailed.json file. Mirrors T8's "
                "rotation_consistency for its own AB-rotation scheme."
            ),
            "num_consistent": rotation_consistent_count,
            "num_total": len(detailed_results),
            "fraction_consistent": (
                rotation_consistent_count / len(detailed_results) if detailed_results else None
            ),
        },
    }

    logger.info("\n" + "=" * 60)
    logger.info("T10 Evaluation Results:")
    logger.info(f"  Num samples: {len(gold_labels)}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Macro F1: {macro_f1:.4f}")
    logger.info(f"  Per-class F1: {per_class_f1}")
    logger.info(f"  Rotation consistency: {rotation_consistent_count}/{len(detailed_results)} "
                f"({results['rotation_consistency']['fraction_consistent']:.4f})")
    logger.info("=" * 60)

    return results, detailed_results


def main():
    parser = argparse.ArgumentParser(description="T10 Stock Movement Prediction Evaluation")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--num-samples", type=int, default=None)

    args = parser.parse_args()
    if args.enable_thinking:
        global ENABLE_THINKING
        ENABLE_THINKING = True

    model_name = sanitize_model_name(args.model)

    log_dir = model_log_dir(model_name, Path(__file__).resolve().parents[2] / "experiments" / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_name}_t10_stock.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to: {log_path}")

    if args.output is None:
        args.output = f"experiments/results/bf16_{model_name}_t10_stock_logprob.json"

    logger.info(f"Loading model: {args.model}...")
    setup_model(args.model)
    logger.info("✓ Model setup complete")

    # Sanity check: a clear rise signal should predict Rise
    sanity_query = (
        "Examine the data and tweets to deduce if the closing price of $aapl will boost or lower at 2020-01-10. "
        "Kindly confirm either Rise or Fall."
        "\nContext: date,open,high,low,close\n"
        "2020-01-02,5.0,6.0,4.0,5.5\n"
        "2020-01-03,5.5,7.0,5.0,6.5\n"
        "2020-01-06,6.5,8.0,6.0,7.8\n"
        "Tweet: $aapl surging, record earnings incoming!\nAnswer:"
    )
    pred, sc, _, _ = score_ab_rotation(sanity_query)
    logger.info(f"Sanity check (expected=Rise): pred={pred}, scores={sc}")
    if pred != "Rise":
        logger.warning("WARNING: sanity check did not predict 'Rise'.")

    # Sanity check 2: 12 curated "obvious" examples — highest-margin,
    # correctly-predicted-by-BF16-reference cases, 6 per gold class (see
    # OBVIOUS_SANITY_EXAMPLES for the selection-criterion adaptation
    # rationale: stock movement isn't humanly judgable from an OHLC table the
    # way T1/T2/T3/T8's text-classification gold labels are, so the anchor
    # here is the reference model's most confident correct calls rather than
    # human-eyeballed unambiguity). Re-scored every run; the full sentences +
    # per-rotation prompts/scores + pass/fail are persisted into the summary
    # JSON below so correctness can be confirmed post-hoc. Mirrors T8's
    # sanity check 2 for its own AB-rotation scheme.
    logger.info("\nSanity check (12 curated 'obvious' examples):")
    obvious_sanity_results = run_obvious_sanity_checks()
    obvious_sanity_passed = sum(r["matches_expected"] for r in obvious_sanity_results)
    obvious_sanity_inconsistent = [r for r in obvious_sanity_results if not r["rotation_consistent"]]
    for r in obvious_sanity_results:
        status = "OK  " if r["matches_expected"] else "MISS"
        rot = "  " if r["rotation_consistent"] else "**"
        logger.info(f"  [{status}] id={r['id']} gold={r['gold_label']:<5} pred={str(r['predicted_label']):<5} "
                    f"margin={r['margin']:.4f} rotation_consistent={r['rotation_consistent']}{rot}")
    logger.info(f"  -> {obvious_sanity_passed}/{len(obvious_sanity_results)} matched expected label, "
                f"{len(obvious_sanity_inconsistent)} rotation-inconsistent")
    if obvious_sanity_passed < len(obvious_sanity_results):
        logger.warning("WARNING: not all 'obvious' sanity-check examples matched their expected label — inspect 'obvious_sanity_check' in the output JSON.")
    if obvious_sanity_inconsistent:
        logger.warning(f"WARNING: {len(obvious_sanity_inconsistent)} rotation-inconsistent sanity-check example(s) — see 'obvious_sanity_check' in the output JSON.")

    results, detailed_results = evaluate_t10_stock(args.num_samples)
    results.update({
        "model": args.model,
        "dtype": str(DTYPE),
        "max_input_length": MAX_INPUT_LENGTH,
        "obvious_sanity_check": {
            "description": (
                "12 examples drawn from the BF16 baseline run with the "
                "corrected prompt — the highest-margin, correctly-predicted "
                "case per gold class direction (6 Rise / 6 Fall), balanced "
                "despite the dataset's milder 55.2%/44.8% Rise/Fall skew. "
                "Unlike T1/T2/T3/T8's 'human-eyeballed for unambiguity' "
                "criterion, stock movement isn't determinable from an OHLC "
                "percentage-change table by inspection — that's the whole "
                "point of the task being hard — so the anchor here is "
                "'most confident correct call by the reference model' rather "
                "than ground-truth-obviousness; this reframes the check's "
                "semantics from a ground-truth anchor to a regression/"
                "consistency anchor against the reference model's most "
                "confident, correct calls. Re-scored every run as a built-in "
                "correctness check on the scoring pipeline (prompt rewriting "
                "/ rotation / tokenization). Each entry also carries its own "
                "'rotations' (full per-rotation prompt + raw A/B log-probs), "
                "mirroring the main loop's detail. Mirrors T8's "
                "obvious_sanity_check for its own AB-rotation scheme."
            ),
            "num_matches_expected": obvious_sanity_passed,
            "num_total": len(obvious_sanity_results),
            "examples": obvious_sanity_results,
        },
    })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Results saved to: {output_path}")

    detailed_path = output_path.parent / f"{output_path.stem}_detailed.json"
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Detailed predictions saved to: {detailed_path}")


if __name__ == "__main__":
    main()
