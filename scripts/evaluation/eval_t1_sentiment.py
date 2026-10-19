"""
T1 Sentiment Analysis Evaluation Script (en-fpb)
Supports multiple model architectures: Llama, Gemma, Qwen

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t1_sentiment.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --num-samples 970
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

SENTIMENT_LABELS = ("negative", "neutral", "positive")
SENTIMENT_PATTERN = re.compile(r"\b(negative|neutral|positive)\b")
INVALID_LABEL_IDX = -1

# Partial assistant prefix that puts the label in a natural semantic position.
# The model scores "The sentiment is <label>" inside the assistant turn.
SCORING_PREFIX = "The sentiment is"

# ============================================================
# CURATED "OBVIOUS" SANITY-CHECK EXAMPLES
# ============================================================
# 12 examples drawn from the BF16 baseline run: the highest log-prob-margin,
# correctly-predicted example per gold class (4 x negative/neutral/positive),
# human-eyeballed and confirmed unambiguous. Re-scored at the start of every
# run — if the scoring pipeline (prompt formatting / tokenization / chat
# template) is broken, these "obvious" cases are the first thing to slip.
OBVIOUS_SANITY_EXAMPLES = [
    {"id": "fpb4259", "gold_label": "negative", "query":
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.\n"
        "Text: Finnish communication electronics components supplier Scanfil Oyj Tuesday said sales in the first half of 2006 will be 15 % lower than during the same period a year ago .\n"
        "Answer:"},
    {"id": "fpb4193", "gold_label": "negative", "query":
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.\n"
        "Text: YIT 's Baltic sales in the first three quarters of 2008 totaled 106.2 million euros , representing a drop of 29 percent year on year .\n"
        "Answer:"},
    {"id": "fpb4169", "gold_label": "negative", "query":
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.\n"
        "Text: The Elcoteq group recently announced that the last three months of the previous year brought to it a major loss of more than half a billion kroons ( EUR 32 mln ) for the fifth quarter running .\n"
        "Answer:"},
    {"id": "fpb4181", "gold_label": "negative", "query":
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.\n"
        "Text: The announcement pushed Freenet shares down 6.3 % , or EUR0 .71 , in Frankfurt trade to EUR10 .65 as investors gave up hope United Internet AG and Drillisch would pursue their own takeover and breakup of Freenet .\n"
        "Answer:"},
    {"id": "fpb4695", "gold_label": "neutral", "query":
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.\n"
        "Text: The decision will have to be made whether the group agrees to the import of Estonian meat and meat products to Finland , the paper added .\n"
        "Answer:"},
    {"id": "fpb4658", "gold_label": "neutral", "query":
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.\n"
        "Text: The cost of the deal could range from 70 million to 90 million euros depending on the financial results of the two companies in 2008 , the statement says .\n"
        "Answer:"},
    {"id": "fpb4352", "gold_label": "neutral", "query":
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.\n"
        "Text: Please inform IR Johanna Koskinen of your participation no later than 20 April at 10 a.m. A telephone conference for financial analysts and investors , conducted in English , will begin at 3:00 p.m. Finnish time ( EET ) .\n"
        "Answer:"},
    {"id": "fpb4737", "gold_label": "neutral", "query":
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.\n"
        "Text: The company 's share is listed in the Mid Cap segment of the NASDAQ OMX Helsinki .\n"
        "Answer:"},
    {"id": "fpb3928", "gold_label": "positive", "query":
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.\n"
        "Text: Altia 's operating profit jumped to EUR 47 million from EUR 6.6 million .\n"
        "Answer:"},
    {"id": "fpb3943", "gold_label": "positive", "query":
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.\n"
        "Text: `` We 're delighted with the move '' says Morna Cowie , co-owner , above , `` it 's double the size of our current shop and has a lovely feel to it . ''\n"
        "Answer:"},
    {"id": "fpb3993", "gold_label": "positive", "query":
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.\n"
        "Text: Dirk Jones , head of Financial Institutions Client Sales Management , GTS , Citigroup , Inc , said , ` Citi is extremely pleased to be providing global custody services to Pohjola Group Bank .\n"
        "Answer:"},
    {"id": "fpb4041", "gold_label": "positive", "query":
        "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.\n"
        "Text: Olvi has posted a strong set of figures for the first six months of this year .\n"
        "Answer:"},
]


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
    """Extract the first valid sentiment label from generated text."""
    allowed_labels = {choice.lower() for choice in choices} or set(SENTIMENT_LABELS)
    text = prediction_text.strip().lower()

    direct_match = re.match(r"^(negative|neutral|positive)\b", text)
    if direct_match and direct_match.group(1) in allowed_labels:
        return direct_match.group(1)

    answer_match = re.search(
        r"\b(?:answer|sentiment|label)\s*(?:is|:)?\s*(negative|neutral|positive)\b",
        text,
    )
    if answer_match and answer_match.group(1) in allowed_labels:
        return answer_match.group(1)

    labels = [
        match.group(1)
        for match in SENTIMENT_PATTERN.finditer(text)
        if match.group(1) in allowed_labels
    ]
    unique_labels = list(dict.fromkeys(labels))
    if len(unique_labels) == 1:
        return unique_labels[0]

    return None


def score_choices(prompt_text: str, choices: list) -> Tuple[str, Dict[str, float]]:
    """
    Score each choice by its length-normalized log-probability — the joint
    log-probability over all answer tokens, divided by token count (average
    log-probability per token). This avoids bias toward shorter labels.
    Uses full-sequence encoding so multi-token labels are handled correctly.
    """
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)

    # prompt_text = chat-formatted prefix + partial assistant response.
    # Chat template embeds BOS, so don't add special tokens again.
    prefix_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    scores = {}
    for choice in choices:
        # Score label as continuation: "The sentiment is <label>"
        full_ids = tokenizer.encode(prompt_text + " " + choice, add_special_tokens=False)

        # Safety check: prefix must be a true prefix of the full sequence.
        # If tokenization at the boundary merges tokens, this fires immediately
        # rather than silently computing wrong log-probs.
        if full_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"Tokenization boundary merge detected for choice '{choice}'. "
                f"prefix_ids={prefix_ids[-3:]}, full_ids={full_ids[:len(prefix_ids)+3]}"
            )

        answer_ids = full_ids[len(prefix_ids):]
        if not answer_ids:
            raise ValueError(f"Choice '{choice}' produced no answer tokens after prefix.")

        inputs = torch.tensor([full_ids], device=model.device)
        with torch.no_grad():
            logits = model(inputs).logits[0]          # [seq_len, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)

        # Position i predicts token i+1 (autoregressive off-by-one)
        total = sum(
            log_probs[len(prefix_ids) + j - 1, tok_id].item()
            for j, tok_id in enumerate(answer_ids)
        )
        scores[choice] = total / len(answer_ids)      # length-normalize

    best = max(scores, key=scores.__getitem__)
    return best, scores


def run_obvious_sanity_checks() -> list:
    """
    Re-score the 12 curated "obvious" examples (OBVIOUS_SANITY_EXAMPLES, validation set 1)
    and record whether the model reproduces the human-verified expected label.

    Unlike the main evaluation loop — where score_choices() is intentionally
    fail-fast (raises ValueError immediately on a tokenization boundary-merge,
    "fail loudly rather than silently return a wrong result") — calls here are
    wrapped in try/except so a failure on one of these 12 fixed examples is
    *recorded* into the summary JSON rather than crashing the whole run.
    """
    records = []
    for ex in OBVIOUS_SANITY_EXAMPLES:
        scoring_prefix = format_chat_prompt(ex["query"]) + SCORING_PREFIX
        try:
            pred_label, choice_scores = score_choices(scoring_prefix, list(SENTIMENT_LABELS))
            ranked = sorted(choice_scores.values(), reverse=True)
            records.append({
                "id": ex["id"],
                "query": ex["query"],
                "prompt": scoring_prefix,
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
                "prompt": scoring_prefix,
                "gold_label": ex["gold_label"],
                "predicted_label": None,
                "choice_scores": None,
                "margin": None,
                "matches_expected": False,
                "assert_error": str(e),
            })
    return records


def evaluate_t1_sentiment(num_samples: int = None) -> Tuple[Dict[str, Any], list]:
    """Evaluate T1 Sentiment Analysis task (en-fpb)"""

    logger.info("=" * 60)
    logger.info("T1 Sentiment Analysis Evaluation (en-fpb)")
    logger.info("=" * 60)

    # Load dataset
    data_path = DATA_DIR / "en_fpb_test.json"
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
    label_map = {"positive": 2, "neutral": 1, "negative": 0}
    reverse_map = {0: "negative", 1: "neutral", 2: "positive"}

    # Store detailed prediction results
    detailed_results = []

    for i, example in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  Progress: {i}/{len(dataset)}")

        # Extract example information
        query = example.get("query", "").strip()
        answer = example.get("answer", "").strip()  # ground truth (original casing)
        choices = [choice.strip().lower() for choice in example.get("choices", [])]
        example_id = example.get("id", str(i))

        if not query or not answer:
            continue

        scoring_prefix = format_chat_prompt(query) + SCORING_PREFIX
        pred_label, choice_scores = score_choices(scoring_prefix, choices)

        pred_idx = label_map.get(pred_label, INVALID_LABEL_IDX)
        gold_idx = label_map.get(answer.lower(), INVALID_LABEL_IDX)
        if gold_idx == INVALID_LABEL_IDX:
            raise ValueError(f"Invalid gold label for example {example_id}: {answer}")

        predictions.append(pred_idx)
        gold_labels.append(gold_idx)

        # Save detailed information
        detailed_results.append({
            "id": example_id,
            "prompt": scoring_prefix,
            "gold_label": answer,
            "predicted_label": pred_label,
            "choice_scores": choice_scores,
            "is_correct": pred_label == answer.lower(),
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
    for label_idx in range(3):
        label_name = reverse_map[label_idx]
        binary_gold = [1 if g == label_idx else 0 for g in gold_labels]
        binary_pred = [1 if p == label_idx else 0 for p in predictions]
        f1 = f1_score(binary_gold, binary_pred, zero_division=0)
        per_class_f1[label_name] = f1

    results = {
        "task": "T1_sentiment_analysis",
        "dataset": "en_fpb",
        "eval_method": "log_prob",
        "num_samples": len(gold_labels),
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
    }

    logger.info("\n" + "=" * 60)
    logger.info("T1 Evaluation Results:")
    logger.info(f"  Num samples: {len(gold_labels)}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Weighted F1: {weighted_f1:.4f}")
    logger.info(f"  Per-class F1:")
    for label, f1 in per_class_f1.items():
        logger.info(f"    {label}: {f1:.4f}")
    logger.info("=" * 60)

    return results, detailed_results


def main():
    parser = argparse.ArgumentParser(description="T1 Sentiment Analysis Evaluation")
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
    log_path = log_dir / f"{model_name}_t1_sentiment.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to: {log_path}")

    # Auto-generate output path if not specified
    if args.output is None:
        args.output = f"experiments/bf16_{model_name}_t1_sentiment.json"

    logger.info(f"Loading model: {args.model}...")
    setup_model(args.model)
    logger.info("✓ Model setup complete")

    # Sanity check 1: generation output
    sanity_prompt = "What is quantization in machine learning?"
    logger.info(f"\nSanity check prompt: {sanity_prompt}")
    inputs, input_len = prepare_input(sanity_prompt)
    sanity_output, _ = generate_response(inputs, input_len)
    logger.info(f"Sanity check output: {sanity_output}\n")

    # Sanity check 2: log-prob scoring on an obviously positive sentence.
    # "positive" must score highest; if not, the scoring prefix or tokenization is wrong.
    positive_query = (
        "Analyze the sentiment of this statement extracted from a financial news article. "
        "Provide your answer as either negative, positive, or neutral.\n"
        "Text: The company reported record profits and strong revenue growth this quarter.\n"
        "Answer:"
    )
    sc_prefix = format_chat_prompt(positive_query) + SCORING_PREFIX
    _, sc_scores = score_choices(sc_prefix, list(SENTIMENT_LABELS))
    logger.info(f"Sanity check (obviously positive sentence) scores: {sc_scores}")
    if max(sc_scores, key=sc_scores.__getitem__) != "positive":
        logger.warning("WARNING: 'positive' did not score highest on obviously positive sentence — check scoring prefix.")

    # Sanity check 3: 12 curated "obvious" examples (validation set 1 — human-verified,
    # high log-prob-margin, 4 per gold class). Re-scored every run; the full
    # sentences + scores + pass/fail are persisted into the summary JSON below
    # so correctness can be confirmed post-hoc without redoing the analysis.
    logger.info("\nSanity check (12 curated 'obvious' examples):")
    obvious_sanity_results = run_obvious_sanity_checks()
    obvious_sanity_passed = sum(r["matches_expected"] for r in obvious_sanity_results)
    obvious_sanity_failed_asserts = [r for r in obvious_sanity_results if r["assert_error"] is not None]
    for r in obvious_sanity_results:
        status = "OK  " if r["matches_expected"] else "MISS"
        margin_str = f"{r['margin']:.4f}" if r["margin"] is not None else "n/a"
        logger.info(f"  [{status}] id={r['id']} gold={r['gold_label']:<8} pred={str(r['predicted_label']):<8} margin={margin_str}")
    logger.info(f"  -> {obvious_sanity_passed}/{len(obvious_sanity_results)} matched expected label, "
                f"{len(obvious_sanity_failed_asserts)} boundary-assert failure(s)")
    if obvious_sanity_passed < len(obvious_sanity_results):
        logger.warning("WARNING: not all 'obvious' sanity-check examples matched their expected label — inspect 'obvious_sanity_check' in the output JSON.")
    if obvious_sanity_failed_asserts:
        logger.warning(f"WARNING: {len(obvious_sanity_failed_asserts)} tokenization boundary-merge failure(s) on sanity-check examples — see 'failed_assert_checks' in the output JSON.")

    # Run evaluation
    results, detailed_results = evaluate_t1_sentiment(args.num_samples)
    main_loop_assert_checks = len(detailed_results) * len(SENTIMENT_LABELS)
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
                "12 human-verified, high log-prob-margin examples from the BF16 "
                "baseline run (eyeball check on obvious cases), 4 per gold "
                "class. Re-scored every run as a built-in correctness check on the "
                "scoring pipeline (prompt formatting / chat template / tokenization)."
            ),
            "num_matches_expected": obvious_sanity_passed,
            "num_total": len(obvious_sanity_results),
            "examples": obvious_sanity_results,
        },
        "failed_assert_checks": {
            "description": (
                "Tokenization boundary-merge assertion failures (score_choices raises "
                "ValueError when prefix_ids is not a true prefix of full_ids). The main "
                "evaluation loop is intentionally fail-fast — it raises immediately and "
                "aborts the run rather than silently recording a wrong score, so this "
                "summary file existing at all is itself proof that zero such failures "
                f"occurred there (across {main_loop_assert_checks} prefix checks: "
                f"{len(detailed_results)} samples x {len(SENTIMENT_LABELS)} choices). "
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
