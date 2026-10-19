# Source: third_party/llm-compressor-examples/awq/llama_example.py
# LLaMA: CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/quantization/awq_w4a16_asym.py
# Gemma: CUDA_VISIBLE_DEVICES=0,1 conda run -n finquant-lmc-main2 python scripts/quantization/awq_w4a16_asym.py --model-id google/gemma-4-31B-it
import argparse
import json
from pathlib import Path

import torch
from compressed_tensors.offload import dispatch_model
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--calib-path", default="data/calibration_datasets/c1_wikitext.jsonl")
parser.add_argument("--max-seq-length", type=int, default=512)
parser.add_argument("--num-calib-samples", type=int, default=128)
parser.add_argument("--save-suffix", default="", help="Appended to the save dir name to avoid overwriting an existing quantized model")
args = parser.parse_args()

MODEL_ID = args.model_id
CALIB_PATH = args.calib_path
MAX_SEQUENCE_LENGTH = args.max_seq_length
NUM_CALIBRATION_SAMPLES = args.num_calib_samples
is_gemma = "gemma" in MODEL_ID.lower()

# Derive calibration set name (e.g. "c1") from filename for save dir.
calib_name = Path(CALIB_PATH).stem.split("_")[0]  # c1_wikitext_128 → c1


model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

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
print()

ignore = ["re:.*lm_head"] + (["re:model.vision_tower.*"] if is_gemma else [])
recipe = [
    AWQModifier(duo_scaling="both"),
    QuantizationModifier(
        ignore=ignore,
        scheme="W4A16_ASYM",
        targets=["Linear"],
    ),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Sanity-check generation (new tokens only).
print("\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
if is_gemma:
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello, what is 2+2?"}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    probe_in = processor(text=prompt, return_tensors="pt").to(model.device)
else:
    probe_in = {"input_ids": processor("Hello my name is", return_tensors="pt").input_ids.to(model.device)}
input_len = probe_in["input_ids"].shape[-1]
probe_out = model.generate(**probe_in, max_new_tokens=50)
print(inner_tok.decode(probe_out[0][input_len:], skip_special_tokens=True))
print("==========================================\n")

# Save to disk compressed.
model_name = MODEL_ID.rstrip("/").split("/")[-1]
SAVE_DIR = f"data/models/{model_name}-awq-w4a16-asym-{calib_name}{args.save_suffix}"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
