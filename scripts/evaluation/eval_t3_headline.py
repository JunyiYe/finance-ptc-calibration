"""
T3 Headline Classification Evaluation Script (FLARE Headlines)
Supports multiple model architectures: Llama, Gemma, Qwen

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t3_headline.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --num-samples 20547
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

# ============================================================
# COMMON INFERENCE PARAMETERS (shared across all models)
# ============================================================
# Generation parameters
GENERATION_CONFIG = {
    "max_new_tokens": 50,
    "do_sample": False,  # Greedy decoding
}

ENABLE_THINKING = False      # override with --enable-thinking at runtime
MAX_NEW_TOKENS_THINKING = 8192


# Input tokenization parameters
MAX_INPUT_LENGTH = 2048  # Max input tokens (leave room for generation)

# Model loading parameters
DEVICE_MAP = "auto"
DTYPE = torch.bfloat16

# ============================================================
# MODEL-SPECIFIC SETUP (loaded in main())
# ============================================================
model = None
processor_or_tokenizer = None  # Either AutoTokenizer or AutoProcessor
model_type = None  # "llama", "gemma", "qwen"

HEADLINE_LABELS = ("no", "yes")
HEADLINE_PATTERN = re.compile(r"\b(no|yes)\b")
INVALID_LABEL_IDX = -1


def score_logprob(query: str, choices: list) -> Tuple[str, Dict[str, float]]:
    """Score each single-token choice in one forward pass on the prompt."""
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)

    # Resolve token IDs for each choice (expects single token per choice)
    choice_token_ids = {}
    for choice in choices:
        toks = tokenizer.encode(choice, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Choice '{choice}' is not a single token: {toks}")
        choice_token_ids[choice] = toks[0]

    prefix = format_chat_prompt(query)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    inp = torch.tensor([prefix_ids], device=model.device)
    with torch.no_grad():
        lp = torch.log_softmax(model(inp).logits[0], dim=-1)

    pos = len(prefix_ids) - 1  # last position predicts first generated token
    scores = {c: lp[pos, tid].item() for c, tid in choice_token_ids.items()}
    best = max(scores, key=scores.__getitem__)
    return best, scores


OBVIOUS_SANITY_EXAMPLES = [
    {"id": "headlines84870", "gold_label": "Yes", "query":
        "Consider whether the headline mentions the price of gold. Does the news headline indicate a Price or Not in the context of gold commodities? Your answer should be Yes or No.\n"
        "Text: gold prices settle at record high\n"
        "Answer:"},
    {"id": "headlines95464", "gold_label": "Yes", "query":
        "Look for indications that the price of gold is increasing. Does the news headline provide information on a Direction Up regarding gold? Your answer should be either Yes or No.\n"
        "Text: Gold prices end at highest level since early June, up 0.5% for the week\n"
        "Answer:"},
    {"id": "headlines82847", "gold_label": "Yes", "query":
        "Look for indications that the price of gold is decreasing. Does the news headline provide information on a Direction Down regarding gold? Your answer should be either Yes or No.\n"
        "Text: gold futures end lower for a fourth straight session\n"
        "Answer:"},
    {"id": "headlines82183", "gold_label": "Yes", "query":
        "Consider whether the headline references past prices of gold. Does the news headline provide information on a PastPrice regarding gold? Your answer should be either Yes or No.\n"
        "Text: Dec. gold drops $10.70, or 0.8%, to settle at $1,279.70/oz\n"
        "Answer:"},
    {"id": "headlines95936", "gold_label": "Yes", "query":
        "Look for any mention of future gold prices. Does the news headline provide information on a FuturePrice regarding gold? Your answer should be either Yes or No.\n"
        "Text: Gold prices to trade lower today: Angel Commodities\n"
        "Answer:"},
    {"id": "headlines88847", "gold_label": "Yes", "query":
        "Consider if the headline compares gold with other assets. In the context of gold, is there a Asset Comparision implied in the news headline? Respond with Yes or No.\n"
        "Text: Gold Prices Remain Higher on Weaker Dollar\n"
        "Answer:"},
    {"id": "headlines96291", "gold_label": "No", "query":
        "Consider whether the headline mentions the price of gold. Is there a Price or Not in the gold commodity market indicated in the news headline? Please answer Yes or No.\n"
        "Text: Trump’s attack on Gold Star widow raises questions about civilian-military divide\n"
        "Answer:"},
    {"id": "headlines89623", "gold_label": "No", "query":
        "Look for indications that the price of gold is increasing. Is there a mention of a Direction Up in relation to gold in the news headline? Please answer with Yes or No.\n"
        "Text: amex gold bugs index falls 3.6% to close at 309.27\n"
        "Answer:"},
    {"id": "headlines94457", "gold_label": "No", "query":
        "Look for indications that the price of gold is decreasing. Can a Direction Down in the gold market be inferred from the news headline? Your response should be either Yes or No.\n"
        "Text: Gold ends higher as U.S. dollar index falls to a nearly 3-week low\n"
        "Answer:"},
    {"id": "headlines97747", "gold_label": "No", "query":
        "Look for any references to future events concerning gold. Does the news headline provide information on a FutureNews regarding gold? Your answer should be either Yes or No.\n"
        "Text: gold ends up 0.8% at $934.90 an ounce\n"
        "Answer:"},
    {"id": "headlines96794", "gold_label": "No", "query":
        "Consider if the headline compares gold with other assets. Is there a mention of a Asset Comparision in relation to gold in the news headline? Please answer with Yes or No.\n"
        "Text: Tariff value on gold imports cut sharply\n"
        "Answer:"},
    {"id": "headlines84586", "gold_label": "No", "query":
        "Consider whether the headline references past prices of gold. Is the news headline discussing a PastPrice related to gold? Please respond with Yes or No.\n"
        "Text: around 1,500 workers rescued from gold mine: report\n"
        "Answer:"},
]


def run_obvious_sanity_checks() -> list:
    """
    Re-score the 12 curated "obvious" examples (OBVIOUS_SANITY_EXAMPLES — 6
    human-verified Yes / 6 No, one per label_type, chosen so the correct
    answer is unambiguous to a human reader) and record whether the model
    reproduces the expected label.

    Unlike the main evaluation loop — where score_logprob() is intentionally
    fail-fast (raises ValueError immediately if a choice does not encode to a
    single token, "fail loudly rather than silently return a wrong result") —
    calls here are wrapped in try/except so a failure on one of these 12 fixed
    examples is *recorded* into the summary JSON rather than crashing the whole run.
    """
    records = []
    for ex in OBVIOUS_SANITY_EXAMPLES:
        prompt = format_chat_prompt(ex["query"])
        try:
            pred_label, choice_scores = score_logprob(ex["query"], ["Yes", "No"])
            ranked = sorted(choice_scores.values(), reverse=True)
            records.append({
                "id": ex["id"],
                "query": ex["query"],
                "prompt": prompt,
                "gold_label": ex["gold_label"],
                "predicted_label": pred_label,
                "choice_scores": choice_scores,
                "margin": ranked[0] - ranked[1],
                "matches_expected": pred_label == ex["gold_label"],
                "assert_error": None,
            })
        except ValueError as e:
            records.append({
                "id": ex["id"],
                "query": ex["query"],
                "prompt": prompt,
                "gold_label": ex["gold_label"],
                "predicted_label": None,
                "choice_scores": None,
                "margin": None,
                "matches_expected": False,
                "assert_error": str(e),
            })
    return records


def sanitize_model_name(model_id: str) -> str:
    """Make a HuggingFace model ID or local path safe for filenames."""
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
    """
    Load model and tokenizer/processor based on model_id.
    Sets global variables: model, processor_or_tokenizer, model_type
    """
    global model, processor_or_tokenizer, model_type

    logger.info(f"Detecting model type from: {model_id}")

    model_id_lower = model_id.lower()

    # Detect model type and load accordingly
    if "llama" in model_id_lower:
        model_type = "llama"
        logger.info("Loading Llama model with AutoTokenizer")

        processor_or_tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=DTYPE,
            device_map=DEVICE_MAP,
        )

    elif "gemma" in model_id_lower:
        model_type = "gemma"
        logger.info("Loading Gemma model with AutoProcessor")

        processor_or_tokenizer = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=DTYPE,
            device_map=DEVICE_MAP,
        )

    elif "qwen" in model_id_lower:
        model_type = "qwen"
        logger.info("Loading Qwen model with AutoTokenizer")

        processor_or_tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=DTYPE,
            device_map=DEVICE_MAP,
        )

    else:
        raise ValueError(f"Unsupported model: {model_id}")

    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    logger.info(f"✓ Model loaded: {model_type.upper()}")


def format_chat_prompt(query: str) -> str:
    """Apply the model chat template when available."""
    if not hasattr(processor_or_tokenizer, "apply_chat_template"):
        return query

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]

    try:
        return processor_or_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    except TypeError:
        return processor_or_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except ValueError:
        logger.warning("Chat template is unavailable for this model; using raw prompt.")
        return query


def prepare_input(query: str) -> Tuple[Dict, int]:
    """
    Prepare input for model-specific inference.

    Returns:
        (tokenized_inputs, input_length)
    """
    if model_type not in {"llama", "gemma", "qwen"}:
        raise ValueError(f"Unsupported model type: {model_type}")

    text = format_chat_prompt(query)
    inputs = processor_or_tokenizer(
        text=text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        add_special_tokens=False,  # chat template already includes BOS
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    return inputs, input_len


def generate_response(inputs: Dict, input_len: int) -> Tuple[str, Optional[str]]:
    """
    Generate response and decode it (model-specific).

    Returns:
        (response_text, thinking_text)  — thinking_text is None unless thinking mode enabled (Gemma or Qwen3)
    """
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    generation_config = dict(GENERATION_CONFIG)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        generation_config["pad_token_id"] = tokenizer.pad_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config,
        )

    new_token_ids = outputs[0][input_len:]
    thinking_text = None

    if model_type == "gemma" and ENABLE_THINKING:
        raw = processor_or_tokenizer.decode(new_token_ids, skip_special_tokens=False)
        parsed = processor_or_tokenizer.parse_response(raw)
        thinking_text = parsed.get("thinking", None)
        response = parsed.get("content", raw)
    elif model_type == "qwen" and ENABLE_THINKING:
        ids = new_token_ids.tolist()
        try:
            index = len(ids) - ids[::-1].index(151668)
        except ValueError:
            index = 0
        _think_raw = tokenizer.decode(ids[:index], skip_special_tokens=True)
        if _think_raw.startswith("<think>"):
            _think_raw = _think_raw[len("<think>"):]
        if _think_raw.endswith("</think>"):
            _think_raw = _think_raw[:-len("</think>")]
        thinking_text = _think_raw.strip("\n") or None
        response = tokenizer.decode(ids[index:], skip_special_tokens=True).strip("\n")
    else:
        response = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    return response.strip(), thinking_text



def extract_predicted_label(prediction_text: str, choices: list) -> Optional[str]:
    """Extract the first valid yes/no label from generated text."""
    allowed_labels = {choice.lower() for choice in choices} or set(HEADLINE_LABELS)
    text = prediction_text.strip().lower()

    direct_match = re.match(r"^(no|yes)\b", text)
    if direct_match and direct_match.group(1) in allowed_labels:
        return direct_match.group(1)

    answer_match = re.search(
        r"\b(?:answer|response|label|classification)\s*(?:is|:)?\s*(no|yes)\b",
        text,
    )
    if answer_match and answer_match.group(1) in allowed_labels:
        return answer_match.group(1)

    labels = [
        match.group(1)
        for match in HEADLINE_PATTERN.finditer(text)
        if match.group(1) in allowed_labels
    ]
    unique_labels = list(dict.fromkeys(labels))
    if len(unique_labels) == 1:
        return unique_labels[0]

    return None


def evaluate_t3_headline(num_samples: int = None) -> Tuple[Dict[str, Any], list]:
    """Evaluate T3 Headline Classification task (FLARE Headlines)"""

    logger.info("=" * 60)
    logger.info("T3 Headline Classification Evaluation (FLARE Headlines)")
    logger.info("=" * 60)

    # Load dataset
    data_path = DATA_DIR / "flare_headlines_test.json"
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
    label_map = {"No": 0, "Yes": 1}
    reverse_map = {0: "No", 1: "Yes"}

    # Store detailed prediction results
    detailed_results = []

    for i, example in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  Progress: {i}/{len(dataset)}")

        # Extract example information
        query = example.get("query", "").strip()
        answer = example.get("answer", "").strip()  # ground truth (original casing)
        choices = [choice.strip() for choice in example.get("choices", [])]
        example_id = example.get("id", str(i))

        if not query or not answer:
            continue

        # Score with log-prob (single forward pass)
        pred_label, choice_scores = score_logprob(query, choices)

        pred_idx = label_map.get(pred_label, INVALID_LABEL_IDX)
        gold_idx = label_map.get(answer, INVALID_LABEL_IDX)
        if gold_idx == INVALID_LABEL_IDX:
            raise ValueError(f"Invalid gold label for example {example_id}: {answer}")

        predictions.append(pred_idx)
        gold_labels.append(gold_idx)

        # Save detailed information
        detailed_results.append({
            "id": example_id,
            "prompt": query,
            "gold_label": answer,
            "predicted_label": pred_label,
            "choice_scores": choice_scores,
            "is_correct": pred_label == answer,
        })

        if i < 3:
            logger.info(f"\nSample {i}:")
            logger.info(f"  Ground truth: {answer}")
            logger.info(f"  Prediction: {pred_label}")
            logger.info(f"  Scores: {choice_scores}")

    # Calculate metrics. Invalid predictions stay in `predictions` as -1, so
    # accuracy counts them as wrong and F1 counts them as misses for gold classes.
    accuracy = accuracy_score(gold_labels, predictions)
    valid_label_ids = list(reverse_map.keys())
    weighted_f1 = f1_score(
        gold_labels,
        predictions,
        labels=valid_label_ids,
        average="weighted",
        zero_division=0,
    )
    macro_f1 = f1_score(
        gold_labels,
        predictions,
        labels=valid_label_ids,
        average="macro",
        zero_division=0,
    )

    # Per-class F1 scores
    per_class_f1 = {}
    for label_idx in range(2):
        label_name = reverse_map[label_idx]
        binary_gold = [1 if g == label_idx else 0 for g in gold_labels]
        binary_pred = [1 if p == label_idx else 0 for p in predictions]
        f1 = f1_score(binary_gold, binary_pred, zero_division=0)
        per_class_f1[label_name] = f1

    invalid_count = sum(1 for pred in predictions if pred == INVALID_LABEL_IDX)
    invalid_rate = invalid_count / len(predictions) if predictions else 0.0

    results = {
        "task": "T3_headline_classification",
        "dataset": "flare_headlines",
        "eval_method": "log_prob",
        "num_samples": len(gold_labels),
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "invalid_predictions": invalid_count,
        "invalid_prediction_rate": invalid_rate,
        "generation_config": GENERATION_CONFIG,
    }

    logger.info("\n" + "=" * 60)
    logger.info("T3 Evaluation Results:")
    logger.info(f"  Num samples: {len(gold_labels)}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Weighted F1: {weighted_f1:.4f}")
    logger.info(f"  Per-class F1:")
    for label, f1 in per_class_f1.items():
        logger.info(f"    {label}: {f1:.4f}")
    logger.info("=" * 60)

    return results, detailed_results


def main():
    parser = argparse.ArgumentParser(description="T3 Headline Classification Evaluation")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", default=None, help="Output file (auto-generated if not specified)")
    parser.add_argument("--enable-thinking", action="store_true", default=False, help="Enable thinking mode (Gemma-4 or Qwen3)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate")

    args = parser.parse_args()
    if args.enable_thinking:
        global ENABLE_THINKING
        ENABLE_THINKING = True
        GENERATION_CONFIG["max_new_tokens"] = MAX_NEW_TOKENS_THINKING

    model_name = sanitize_model_name(args.model)

    # Set up file logging
    log_dir = model_log_dir(model_name, Path(__file__).resolve().parents[2] / "experiments" / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_name}_t3_headline.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to: {log_path}")

    # Auto-generate output path if not specified
    if args.output is None:
        args.output = f"experiments/bf16_{model_name}_t3_headline.json"

    logger.info(f"Loading model: {args.model}...")
    setup_model(args.model)
    logger.info("✓ Model setup complete")

    # Sanity check 1: generation output
    sanity_prompt = "What is quantization in machine learning?"
    logger.info(f"\nSanity check prompt: {sanity_prompt}")
    inputs, input_len = prepare_input(sanity_prompt)
    sanity_output, _ = generate_response(inputs, input_len)
    logger.info(f"Sanity check output: {sanity_output}\n")

    # Sanity check 2: log-prob scoring on a headline that obviously mentions gold price.
    # "Yes" must score highest; if not, the prompt formatting or tokenization is wrong.
    sanity_query = (
        "Consider whether the headline mentions the price of gold. "
        "Is there a Price or Not in the gold commodity market indicated in the news headline? "
        "Please answer Yes or No.\n"
        "Text: Gold prices surge to record high amid inflation fears\n"
        "Answer:"
    )
    _, sc_scores = score_logprob(sanity_query, ["Yes", "No"])
    logger.info(f"Sanity check (obvious yes) scores: {sc_scores}")
    if max(sc_scores, key=sc_scores.__getitem__) != "Yes":
        logger.warning("WARNING: 'Yes' did not score highest on obvious yes sample — check prompt/tokenization.")

    # Sanity check 3: 12 curated "obvious" examples (6 Yes / 6 No, one per
    # label_type, chosen so the correct answer is unambiguous to a human).
    # Re-scored every run; the full queries + scores + pass/fail are persisted
    # into the summary JSON below so correctness can be confirmed post-hoc
    # without redoing the analysis.
    logger.info("\nSanity check (12 curated 'obvious' examples):")
    obvious_sanity_results = run_obvious_sanity_checks()
    obvious_sanity_passed = sum(r["matches_expected"] for r in obvious_sanity_results)
    obvious_sanity_failed_asserts = [r for r in obvious_sanity_results if r["assert_error"] is not None]
    for r in obvious_sanity_results:
        status = "OK  " if r["matches_expected"] else "MISS"
        margin_str = f"{r['margin']:.4f}" if r["margin"] is not None else "n/a"
        logger.info(f"  [{status}] id={r['id']} gold={r['gold_label']:<4} pred={str(r['predicted_label']):<4} margin={margin_str}")
    logger.info(f"  -> {obvious_sanity_passed}/{len(obvious_sanity_results)} matched expected label, "
                f"{len(obvious_sanity_failed_asserts)} single-token-assert failure(s)")
    if obvious_sanity_passed < len(obvious_sanity_results):
        logger.warning("WARNING: not all 'obvious' sanity-check examples matched their expected label — inspect 'obvious_sanity_check' in the output JSON.")
    if obvious_sanity_failed_asserts:
        logger.warning(f"WARNING: {len(obvious_sanity_failed_asserts)} single-token tokenization failure(s) on sanity-check examples — see 'failed_assert_checks' in the output JSON.")

    # Run evaluation
    results, detailed_results = evaluate_t3_headline(args.num_samples)
    main_loop_assert_checks = len(detailed_results) * 2  # 2 choices ("Yes"/"No") per sample
    results.update({
        "model": args.model,
        "dtype": str(DTYPE),
        "max_input_length": MAX_INPUT_LENGTH,
        "sanity_check_generation": {
            "description": (
                "Sanity check 1 — free-form generation on a generic prompt, run "
                "right after model load to confirm the model produces coherent "
                "text before the dataset evaluation begins."
            ),
            "prompt": sanity_prompt,
            "output": sanity_output,
        },
        "obvious_sanity_check": {
            "description": (
                "12 human-verified examples (6 Yes / 6 No, one per label_type, "
                "chosen so the correct answer is unambiguous to a human reader). "
                "Re-scored every run as a built-in correctness check on the "
                "scoring pipeline (prompt formatting / chat template / tokenization)."
            ),
            "num_matches_expected": obvious_sanity_passed,
            "num_total": len(obvious_sanity_results),
            "examples": obvious_sanity_results,
        },
        "failed_assert_checks": {
            "description": (
                "Single-token tokenization assertion failures (score_logprob raises "
                "ValueError when a choice does not encode to exactly one token). The "
                "main evaluation loop is intentionally fail-fast — it raises immediately "
                "and aborts the run rather than silently scoring the wrong token, so this "
                "summary file existing at all is itself proof that zero such failures "
                f"occurred there (across {main_loop_assert_checks} choice checks: "
                f"{len(detailed_results)} samples x 2 choices). "
                "Only the 12 sanity-check examples above are scored with try/except, so "
                "failures on those specifically are captured here instead of crashing."
            ),
            "main_loop_checks_performed": main_loop_assert_checks,
            "main_loop_failures": 0,
            "sanity_check_failures": obvious_sanity_failed_asserts,
        },
    })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Results saved to: {output_path}")

    # Save detailed predictions
    detailed_path = output_path.parent / f"{output_path.stem}_detailed.json"
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Detailed predictions saved to: {detailed_path}")


if __name__ == "__main__":
    main()
