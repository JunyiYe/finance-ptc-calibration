#!/usr/bin/env python3
"""
E1: WikiText-2 test perplexity for BF16 + all quantized Llama-3.1-8B configs.

Purpose (thesis C, "hidden damage"): show that the conventional metric barely
moves while numerical-reasoning accuracy collapses.

Protocol: standard GPTQ-paper style — concatenate the WikiText-2 test split,
score non-overlapping 2048-token windows, ppl = exp(mean NLL over tokens).

Usage (one config per process — compressed-tensors models are not reliably
freed in-process, so a loop over configs leaks GPU memory; drive the sweep
with the bash loop below instead):

    for c in $(python scripts/analysis/eval_perplexity.py --list); do
        CUDA_VISIBLE_DEVICES=1 python scripts/analysis/eval_perplexity.py --only "$c"
    done

Output: experiments/results/analysis/perplexity_Llama-3.1-8B-Instruct.{json,md}
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SEQLEN = 2048
BASE = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_DIR = "data/models"
OUT = "experiments/results/analysis/perplexity_Llama-3.1-8B-Instruct"

CONFIGS = [("bf16", BASE), ("rtn_w4a16", f"{MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-rtn-w4a16"),
           ("gptq_w4a16_c1small", f"{MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-gptq-w4a16-c1small"),
           ("sparsegpt_unstructured_c3full", f"{MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-sparsegpt-unstructured-c3full"),
           ("wanda_4of8_c3full", f"{MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-wanda-4of8-c3full")]
for meth_key, meth_dir in [("gptq_w4a16", "gptq-w4a16"),
                           ("awq_w4a16_asym", "awq-w4a16-asym"),
                           ("smoothquant_w8a8", "smoothquant-w8a8"),
                           ("sparsegpt_unstructured", "sparsegpt-unstructured"),
                           ("wanda_4of8", "wanda-4of8")]:
    for c in ["c1", "c2", "c3"]:
        CONFIGS.append((f"{meth_key}_{c}",
                        f"{MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-{meth_dir}-{c}"))


@torch.no_grad()
def perplexity(model, input_ids):
    nll_sum, tok_count = 0.0, 0
    n_windows = input_ids.shape[1] // SEQLEN
    for i in range(n_windows):
        window = input_ids[:, i * SEQLEN:(i + 1) * SEQLEN].to(model.device)
        out = model(window, labels=window)
        # out.loss is mean NLL over (SEQLEN - 1) predicted tokens
        nll_sum += out.loss.float().item() * (window.shape[1] - 1)
        tok_count += window.shape[1] - 1
    return float(torch.exp(torch.tensor(nll_sum / tok_count)))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="evaluate a single config name")
    ap.add_argument("--list", action="store_true", help="print config names and exit")
    ap.add_argument("--corpus", default="wikitext", choices=["wikitext", "fin"],
                    help="fin = held-out earnings-call text (never used for calibration or eval)")
    args = ap.parse_args()
    if args.corpus == "fin":
        OUT = OUT.replace("perplexity_", "fin_perplexity_")
    if args.list:
        done = json.load(open(f"{OUT}.json")) if os.path.exists(f"{OUT}.json") else {}
        for name, _ in CONFIGS:
            if name not in done:
                print(name)
        return
    configs = [(n, p) for n, p in CONFIGS if args.only is None or n == args.only]

    tok = AutoTokenizer.from_pretrained(BASE)
    if args.corpus == "fin":
        text = "\n\n".join(json.loads(l)["text"]
                           for l in open("data/calibration_datasets/finppl_earnings_calls.jsonl"))
    else:
        test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(r["text"] for r in test)
    input_ids = tok(text, return_tensors="pt").input_ids
    print(f"WikiText-2 test: {input_ids.shape[1]} tokens, "
          f"{input_ids.shape[1] // SEQLEN} windows of {SEQLEN}")

    results = {}
    if os.path.exists(f"{OUT}.json"):
        results = json.load(open(f"{OUT}.json"))
    for name, path in configs:
        if name in results:
            print(f"{name}: cached {results[name]:.2f}")
            continue
        if path.startswith(MODEL_DIR) and not os.path.isdir(path):
            print(f"{name}: model missing, skipping")
            continue
        print(f"{name}: loading {path} ...")
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        ppl = perplexity(model, input_ids)
        results[name] = ppl
        print(f"{name}: ppl = {ppl:.2f}")
        del model
        torch.cuda.empty_cache()
        with open(f"{OUT}.json", "w") as f:
            json.dump(results, f, indent=2)

    with open(f"{OUT}.md", "w") as f:
        f.write("# WikiText-2 test perplexity (2048-token windows)\n\n")
        f.write("| Config | PPL |\n|---|---|\n")
        for name, _ in CONFIGS:
            if name in results:
                f.write(f"| {name} | {results[name]:.2f} |\n")
    print(f"wrote {OUT}.md")


if __name__ == "__main__":
    main()
