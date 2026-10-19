# Source: llama3/recipes/rtn_w4a16.py
# LLaMA: CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/quantization/rtn_w4a16.py
# LLaMA: CUDA_VISIBLE_DEVICES=0,1 conda run -n finquant-lmc-main python scripts/quantization/rtn_w4a16.py --model-id meta-llama/Meta-Llama-3.1-70B-Instruct
# Gemma: CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main2 python scripts/quantization/rtn_w4a16.py --model-id google/gemma-4-31B-it
import argparse

import torch
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
args = parser.parse_args()

MODEL_ID = args.model_id
is_gemma = "gemma" in MODEL_ID.lower()

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

if is_gemma:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
else:
    processor = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Inner tokenizer (for decode); AutoProcessor wraps one under .tokenizer
inner_tok = getattr(processor, "tokenizer", processor)

# RTN: data-free weight-only quantization (no calibration dataset needed).
ignore = ["re:.*lm_head"] + (["re:model.vision_tower.*"] if is_gemma else [])
recipe = QuantizationModifier(targets="Linear", scheme="W4A16", ignore=ignore)

oneshot(model=model, recipe=recipe)

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
SAVE_DIR = f"data/models/{model_name}-rtn-w4a16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
