"""
T8 Causal Classification Evaluation Script (flare-causal20-sc)
Supports multiple model architectures: Llama, Gemma, Qwen

Eval method: A/B rotation log-prob (2 rotations of causal/noise → A/B).
causal=2 tokens, noise=1 token → A/B rotation removes length bias.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t8_causal.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --num-samples 1500
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

CAUSAL_LABELS = ("causal", "noise")
INVALID_LABEL_IDX = -1

# A/B rotation: 2 cyclic permutations
LETTERS = ("A", "B")
ROTATIONS = [
    ("causal", "noise"),
    ("noise",  "causal"),
]

# ============================================================
# CURATED "OBVIOUS" SANITY-CHECK EXAMPLES
# ============================================================
# 12 examples drawn from the BF16 baseline run: the highest log-prob-margin,
# correctly-predicted example per gold class (6 x causal/noise), human-eyeballed
# and confirmed unambiguous — every "causal" pick contains an explicit causal
# connector ("as", "because", "due to", "because of"); every "noise" pick is
# clear non-financial boilerplate/chit-chat/photo-caption filler. Re-scored at
# the start of every run — if the scoring pipeline (prompt rewriting / rotation
# / tokenization) is broken, these "obvious" cases are the first thing to slip.
# Note: the dataset itself is heavily skewed toward "noise" (93.4% / 6.6% —
# only 572/8628 examples are "causal"), so unlike the dataset's natural
# distribution these 12 are deliberately balanced 6/6: the goal is to test
# recognition of each class's obvious form, not to mirror the label prior.
OBVIOUS_SANITY_EXAMPLES = [
    {"id": "causal20sc6962", "gold_label": "causal", "query":
        "In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either 'causal' or 'noise' based on whether or not it indicates a causal relationship between financial events. Please return only the category 'causal' or 'noise'.\n"
        "Text:  The risk of missing its fiscal deficit target of 3.3% increases significantly as tax revenue growth is already weak.\n"
        "Answer:"},
    {"id": "causal20sc5695", "gold_label": "causal", "query":
        "In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either 'causal' or 'noise' based on whether or not it indicates a causal relationship between financial events. Please return only the category 'causal' or 'noise'.\n"
        "Text:  With that in mind, our GMV declined to $98 million from $127 million in the same period last year. The main reason for the low GMV was because we had to continue to scale back investments in marketing and inventory, which resulted in the decline in a number of buyers and orders.\n"
        "Answer:"},
    {"id": "causal20sc3796", "gold_label": "causal", "query":
        "In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either 'causal' or 'noise' based on whether or not it indicates a causal relationship between financial events. Please return only the category 'causal' or 'noise'.\n"
        "Text:  Analysis of MRMD's Condensed Consolidated Statements of Operations  As the above data indicates, the company's gross margin decreased significantly from 60% in Q1 to ~35% in Q2. The reason for this decline of 40+% is due to the ~83% o f Q2 sales that were from GenCanna were at a much lower margin than the ~17% sales from non-related parties.\n"
        "Answer:"},
    {"id": "causal20sc2918", "gold_label": "causal", "query":
        "In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either 'causal' or 'noise' based on whether or not it indicates a causal relationship between financial events. Please return only the category 'causal' or 'noise'.\n"
        "Text:  Our dividend accrual figures will show that dividend accrual is slightly over 100%. That happens because AI-C has an unusual schedule for ex-dividend dates.\n"
        "Answer:"},
    {"id": "causal20sc2543", "gold_label": "causal", "query":
        "In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either 'causal' or 'noise' based on whether or not it indicates a causal relationship between financial events. Please return only the category 'causal' or 'noise'.\n"
        "Text:  According to a Gallup poll, 75% of people who leave their jobs do so because of their bosses, not their positions.\n"
        "Answer:"},
    {"id": "causal20sc7844", "gold_label": "causal", "query":
        "In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either 'causal' or 'noise' based on whether or not it indicates a causal relationship between financial events. Please return only the category 'causal' or 'noise'.\n"
        "Text:  Net operating profit in the global markets division plunged 26 per cent to 251 billion yen (S$3.19 billion) last year, largely because of a big fall in Europe, and the bank said it struggled in customer business due to sluggish markets.\n"
        "Answer:"},
    {"id": "causal20sc5316", "gold_label": "noise", "query":
        "In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either 'causal' or 'noise' based on whether or not it indicates a causal relationship between financial events. Please return only the category 'causal' or 'noise'.\n"
        "Text:  I was excited to listen…and thanks to Paula for the shutout of never using an iron again! (Pssst to Joe and Len, I'm a dude, it was funny every time you said she)  So what questions came up from the panel?\n"
        "Answer:"},
    {"id": "causal20sc5798", "gold_label": "noise", "query":
        "In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either 'causal' or 'noise' based on whether or not it indicates a causal relationship between financial events. Please return only the category 'causal' or 'noise'.\n"
        "Text:  Tip Fleming  Thank you everyone for joining the call today. If you have any further questions, please don't hesitate to contact us or the company directly. Goodbye.\n"
        "Answer:"},
    {"id": "causal20sc3233", "gold_label": "noise", "query":
        "In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either 'causal' or 'noise' based on whether or not it indicates a causal relationship between financial events. Please return only the category 'causal' or 'noise'.\n"
        "Text:  Our database includes 500,000+ market research reports from over 95 leading global publishers & in-depth market research studies of over 5000 micro markets. SUBSCRIBE Please Enter a Valid Email ID Thank you!\n"
        "Answer:"},
    {"id": "causal20sc8379", "gold_label": "noise", "query":
        "In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either 'causal' or 'noise' based on whether or not it indicates a causal relationship between financial events. Please return only the category 'causal' or 'noise'.\n"
        "Text:  CVM TV aims to maintain high quality, first-world broadcasting standards and ongoing coverage across the entire island at all times. Your email address will not be published. Required fields are marked September 22, 2019  September 22, 2019  September 22, 2019  September 22, 2019  September 22, 2019  CVM TV\n"
        "Answer:"},
    {"id": "causal20sc2789", "gold_label": "noise", "query":
        "In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either 'causal' or 'noise' based on whether or not it indicates a causal relationship between financial events. Please return only the category 'causal' or 'noise'.\n"
        "Text:  Leicestershire tourist among Thomas Cook holidaymakers 'held hostage' by Tunisian hotel  Like us on Facebook Follow us on Twitter  leicestershirelive  Comments  More On  PA reporters\n"
        "Answer:"},
    {"id": "causal20sc8597", "gold_label": "noise", "query":
        "In this task, you are provided with sentences extracted from financial news and SEC data. Your goal is to classify each sentence into either 'causal' or 'noise' based on whether or not it indicates a causal relationship between financial events. Please return only the category 'causal' or 'noise'.\n"
        "Text:  Demonstrators protest President Trump's attempts to end the DACA program, in Washington on March 5, 2018.Samuel Corum / Anadolu Agency/Getty Images file  Get breaking news alerts and special reports.\n"
        "Answer:"},
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
    Score causal/noise via A/B single-token options with 2 cyclic rotations.

    Each of the 2 cyclic rotations maps the labels onto fixed single-token
    letters "A"/"B" in a different order; averaging the per-label log-probs
    across rotations cancels any positional/identity bias the model has
    toward a given letter (each label occupies each position exactly once).
    This also sidesteps the token-length mismatch between "causal" (2 tokens:
    ['ca','usal']) and "noise" (1 token) that rules out direct single-token
    label scoring (the approach T1/T3 use via score_logprob/score_choices).

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
        score_abc_rotation/rotation_consistent design for its 3-way A/B/C
        rotation).
    """
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    letter_ids = _get_letter_token_ids()
    label_scores = {label: 0.0 for label in CAUSAL_LABELS}
    rotation_details = []

    for rotation in ROTATIONS:
        choice_str = "\n".join(f"{l}. {v}" for l, v in zip(LETTERS, rotation))
        modified_query = re.sub(r"Please return only the category 'causal' or 'noise'\.", "", query)
        modified_query = modified_query.replace(
            "\nAnswer:",
            f"\nChoose one of the following options:\n{choice_str}\n\nAnswer with only A or B.\nAnswer:",
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
    record whether the model reproduces the human-verified expected label.

    score_ab_rotation() has no fail-fast assertion to wrap (the AB scheme
    scores fixed single-token letters, so there's no tokenization
    boundary-merge risk the way score_logprob()/score_choices() has in
    T1/T3) — these calls simply re-run the same scoring path the main loop
    uses, on cases whose answer a human has confirmed. Mirrors T2's
    run_obvious_sanity_checks() for its own letter-rotation scheme.
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


def evaluate_t8_causal(num_samples: int = None) -> Tuple[Dict[str, Any], list]:
    """Evaluate T8 Causal Classification task (flare-causal20-sc)"""

    logger.info("=" * 60)
    logger.info("T8 Causal Classification Evaluation (flare-causal20-sc)")
    logger.info("=" * 60)

    data_path = DATA_DIR / "flare_causal20_sc_test.json"
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
    label_map = {"causal": 1, "noise": 0}
    reverse_map = {0: "noise", 1: "causal"}
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
        gold_idx = label_map.get(answer.lower(), INVALID_LABEL_IDX)
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
            "is_correct": pred_label == answer.lower(),
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
        "task": "T8_causal_classification",
        "dataset": "flare_causal20_sc",
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
                "within the _detailed.json file. Mirrors T2's "
                "rotation_consistency for its own letter-rotation scheme."
            ),
            "num_consistent": rotation_consistent_count,
            "num_total": len(detailed_results),
            "fraction_consistent": (
                rotation_consistent_count / len(detailed_results) if detailed_results else None
            ),
        },
    }

    logger.info("\n" + "=" * 60)
    logger.info("T8 Evaluation Results:")
    logger.info(f"  Num samples: {len(gold_labels)}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Macro F1: {macro_f1:.4f}")
    logger.info(f"  Per-class F1: {per_class_f1}")
    logger.info(f"  Rotation consistency: {rotation_consistent_count}/{len(detailed_results)} "
                f"({results['rotation_consistency']['fraction_consistent']:.4f})")
    logger.info("=" * 60)

    return results, detailed_results


def main():
    parser = argparse.ArgumentParser(description="T8 Causal Classification Evaluation")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", default=None, help="Output file (auto-generated if not specified)")
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--num-samples", type=int, default=None)

    args = parser.parse_args()
    if args.enable_thinking:
        global ENABLE_THINKING
        ENABLE_THINKING = True

    model_name = sanitize_model_name(args.model)

    log_dir = model_log_dir(model_name, Path(__file__).resolve().parents[2] / "experiments" / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_name}_t8_causal.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to: {log_path}")

    if args.output is None:
        args.output = f"experiments/results/bf16_{model_name}_t8_causal_logprob.json"

    logger.info(f"Loading model: {args.model}...")
    setup_model(args.model)
    logger.info("✓ Model setup complete")

    # Sanity check 1: a clearly causal sentence should score causal highest
    sanity_query = (
        "In this task, you are provided with sentences extracted from financial news and SEC data. "
        "Your goal is to classify each sentence into either 'causal' or 'noise' based on whether "
        "or not it indicates a causal relationship between financial events. "
        "Please return only the category 'causal' or 'noise'.\n"
        "Text: The company's revenue fell sharply because of a sudden drop in consumer demand.\n"
        "Answer:"
    )
    pred, sc, _, _ = score_ab_rotation(sanity_query)
    logger.info(f"Sanity check (answer=causal): pred={pred}, scores={sc}")
    if pred != "causal":
        logger.warning("WARNING: sanity check did not predict 'causal'.")

    # Sanity check 2: 12 curated "obvious" examples (human-verified, high
    # log-prob-margin, 6 per gold class — see OBVIOUS_SANITY_EXAMPLES for the
    # selection rationale). Re-scored every run; the full sentences + per-
    # rotation prompts/scores + pass/fail are persisted into the summary JSON
    # below so correctness can be confirmed post-hoc. Mirrors T2's sanity
    # check 3 for its own letter-rotation scheme.
    logger.info("\nSanity check (12 curated 'obvious' examples):")
    obvious_sanity_results = run_obvious_sanity_checks()
    obvious_sanity_passed = sum(r["matches_expected"] for r in obvious_sanity_results)
    obvious_sanity_inconsistent = [r for r in obvious_sanity_results if not r["rotation_consistent"]]
    for r in obvious_sanity_results:
        status = "OK  " if r["matches_expected"] else "MISS"
        rot = "  " if r["rotation_consistent"] else "**"
        logger.info(f"  [{status}] id={r['id']} gold={r['gold_label']:<8} pred={str(r['predicted_label']):<8} "
                    f"margin={r['margin']:.4f} rotation_consistent={r['rotation_consistent']}{rot}")
    logger.info(f"  -> {obvious_sanity_passed}/{len(obvious_sanity_results)} matched expected label, "
                f"{len(obvious_sanity_inconsistent)} rotation-inconsistent")
    if obvious_sanity_passed < len(obvious_sanity_results):
        logger.warning("WARNING: not all 'obvious' sanity-check examples matched their expected label — inspect 'obvious_sanity_check' in the output JSON.")
    if obvious_sanity_inconsistent:
        logger.warning(f"WARNING: {len(obvious_sanity_inconsistent)} rotation-inconsistent sanity-check example(s) — see 'obvious_sanity_check' in the output JSON.")

    results, detailed_results = evaluate_t8_causal(args.num_samples)
    results.update({
        "model": args.model,
        "dtype": str(DTYPE),
        "max_input_length": MAX_INPUT_LENGTH,
        "obvious_sanity_check": {
            "description": (
                "12 human-verified, high log-prob-margin examples drawn from "
                "the BF16 baseline run (eyeball check on obvious cases), 6 per "
                "gold class — deliberately balanced 6/6 despite the dataset's "
                "heavy 93.4%/6.6% noise/causal skew, since the goal is to test "
                "recognition of each class's obvious form, not to mirror the "
                "label prior. Re-scored every run as a built-in correctness "
                "check on the scoring pipeline (prompt rewriting / rotation / "
                "tokenization). Each entry also carries its own 'rotations' "
                "(full per-rotation prompt + raw A/B log-probs), mirroring the "
                "main loop's detail. Mirrors T2's obvious_sanity_check for its "
                "own letter-rotation scheme."
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
