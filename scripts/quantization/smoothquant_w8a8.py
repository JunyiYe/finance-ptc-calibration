#!/usr/bin/env python3
"""
SmoothQuant W8A8 Quantization — Llama 3.1-8B-Instruct
Smooths activation outliers then quantizes weights+activations to INT8.

Usage:
    # LLaMA:
    CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/quantization/smoothquant_w8a8.py \
        --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
        --calib-path data/calibration_datasets/c1_wikitext.jsonl

    # Gemma:
    CUDA_VISIBLE_DEVICES=0,1 conda run -n finquant-lmc-main2 python scripts/quantization/smoothquant_w8a8.py \
        --model-id google/gemma-4-31B-it \
        --calib-path data/calibration_datasets/c1_wikitext.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from compressed_tensors.offload import dispatch_model
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--calib-path", default="data/calibration_datasets/c1_wikitext.jsonl")
parser.add_argument("--max-seq-length", type=int, default=512)
parser.add_argument("--num-calib-samples", type=int, default=128)
parser.add_argument("--max-gpu-mem", default=None,
                    help="comma list of per-visible-GPU memory caps; enables "
                         "the big-model path (oneshot save, no probe)")
parser.add_argument("--smoothing-strength", type=float, default=0.8)
args = parser.parse_args()

MODEL_ID = args.model_id
CALIB_PATH = args.calib_path
MAX_SEQUENCE_LENGTH = args.max_seq_length
NUM_CALIBRATION_SAMPLES = args.num_calib_samples
SMOOTHING_STRENGTH = args.smoothing_strength

is_gemma = "gemma" in MODEL_ID.lower()

# Derive calibration set name (e.g. "c1") from filename for save dir.
calib_name = Path(CALIB_PATH).stem.split("_")[0]  # c1_wikitext_128 → c1

_mm = None
if args.max_gpu_mem:
    caps = args.max_gpu_mem.split(",")
    _mm = {i: c for i, c in enumerate(caps)}
    _mm["cpu"] = "600GiB"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", max_memory=_mm)

if is_gemma:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
else:
    processor = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Inner tokenizer (for decode); AutoProcessor wraps one under .tokenizer
inner_tok = getattr(processor, "tokenizer", processor)

# Load calibration dataset from JSONL.
texts = []
with open(CALIB_PATH) as f:
    for line in f:
        texts.append(json.loads(line)["text"])

ds = Dataset.from_dict({"text": texts})


def tokenize(sample):
    # Both AutoProcessor and AutoTokenizer accept text= keyword arg
    return processor(
        text=sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Verify calibration dataset.
lengths = [len(x) for x in ds["input_ids"]]
print(f"\n[Calib] path={CALIB_PATH}")
print(f"[Calib] samples={len(ds)}, max_seq_length={MAX_SEQUENCE_LENGTH}")
print(f"[Calib] token lengths — min={min(lengths)}, mean={sum(lengths)/len(lengths):.0f}, max={max(lengths)}")
print(f"[Calib] truncated={sum(l == MAX_SEQUENCE_LENGTH for l in lengths)}/{len(ds)}")
print(f"[Calib] sample[0] decoded: {inner_tok.decode(ds[0]['input_ids'][:40])}...")
print(f"[SmoothQuant] smoothing_strength={SMOOTHING_STRENGTH}")
print()

recipe = [
    SmoothQuantModifier(smoothing_strength=SMOOTHING_STRENGTH),
    QuantizationModifier(targets=["Linear"], scheme="W8A8", ignore=["lm_head"]),
]

model_name = MODEL_ID.rstrip("/").split("/")[-1]
SAVE_DIR_EARLY = f"data/models/{model_name}-smoothquant-w8a8-{calib_name}"
oneshot_kwargs = {}
if args.max_gpu_mem:
    oneshot_kwargs["output_dir"] = SAVE_DIR_EARLY
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    **oneshot_kwargs,
)
if args.max_gpu_mem:
    processor.save_pretrained(SAVE_DIR_EARLY)
    print(f"Saved via oneshot to {SAVE_DIR_EARLY}; skipping probe and manual save.")
    raise SystemExit(0)

# Sanity-check generation (new tokens only).
print("\n========== SAMPLE GENERATION ==============")
dispatch_model(model)

if is_gemma:
    probe_in = processor(text="Hello my name is", return_tensors="pt",
                         add_special_tokens=False).to(model.device)
else:
    probe_in = {"input_ids": processor("Hello my name is", return_tensors="pt").input_ids.to(model.device)}

input_len = probe_in["input_ids"].shape[-1]
probe_out = model.generate(**probe_in, max_new_tokens=50)
print(inner_tok.decode(probe_out[0][input_len:], skip_special_tokens=True))
print("==========================================\n")

# Save to disk compressed.
model_name = MODEL_ID.rstrip("/").split("/")[-1]
SAVE_DIR = f"data/models/{model_name}-smoothquant-w8a8-{calib_name}"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)