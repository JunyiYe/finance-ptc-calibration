"""
Classify CFA exam questions as requiring multi-step reasoning or not,
using Llama-3.1-8B-Instruct with log-prob Yes/No scoring.
"""

import json
import re
import logging
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_PATH = Path("data/evaluation_datasets/flare_cfa_test.json")
OUTPUT_PATH = Path("experiments/results/cfa_multistep_classification.json")

SYSTEM = "You are an expert in financial analysis and the CFA curriculum."

def make_prompt(query: str, tokenizer) -> str:
    m = re.search(r'Q:(.*?)Answer:', query, re.DOTALL)
    q = m.group(1).strip() if m else query

    user_msg = (
        "A question requires multi-step reasoning if answering it correctly requires "
        "combining multiple concepts, performing calculations, or following a chain of "
        "logical steps. A question is single-step if it can be answered by recalling "
        "a single fact or definition.\n\n"
        "Does the following CFA exam question require multi-step reasoning?\n\n"
        f"Question: {q}\n\n"
        "Answer Yes or No."
    )
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    logger.info(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    logger.info("Model loaded.")

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)
    no_id  = tokenizer.encode("No",  add_special_tokens=False)
    assert len(yes_id) == 1 and len(no_id) == 1, "Yes/No must be single tokens"
    yes_id, no_id = yes_id[0], no_id[0]
    logger.info(f"Yes token: {yes_id}, No token: {no_id}")

    dataset = json.load(open(DATA_PATH))
    logger.info(f"Loaded {len(dataset)} samples")

    results = []
    multistep_count = 0

    for i, example in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  Progress: {i}/{len(dataset)}  (multi-step so far: {multistep_count})")

        query = example.get("query", "").strip()
        prefix = make_prompt(query, tokenizer)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        inp = torch.tensor([prefix_ids], device=model.device)

        with torch.no_grad():
            lp = torch.log_softmax(model(inp).logits[0], dim=-1)

        pos = len(prefix_ids) - 1
        score_yes = lp[pos, yes_id].item()
        score_no  = lp[pos, no_id].item()
        is_multistep = score_yes > score_no
        if is_multistep:
            multistep_count += 1

        results.append({
            "id": example.get("id", str(i)),
            "query": query,
            "score_yes": score_yes,
            "score_no": score_no,
            "is_multistep": is_multistep,
        })

    total = len(results)
    ratio = multistep_count / total
    logger.info(f"\n{'='*50}")
    logger.info(f"Total: {total}")
    logger.info(f"Multi-step: {multistep_count} ({ratio:.1%})")
    logger.info(f"Single-step: {total - multistep_count} ({1-ratio:.1%})")
    logger.info(f"{'='*50}")

    output = {
        "model": MODEL_ID,
        "total": total,
        "multistep_count": multistep_count,
        "singlestep_count": total - multistep_count,
        "multistep_ratio": ratio,
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(OUTPUT_PATH, "w"), indent=2, ensure_ascii=False)
    logger.info(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
