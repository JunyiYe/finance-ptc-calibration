"""
T9 Conversational Financial QA Evaluation Script (flare-convfinqa)
Supports multiple model architectures: Llama, Gemma, Qwen

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t9_convfinqa.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --num-samples 1490
"""

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

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
    "max_new_tokens": 2048,
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



# Matches: optional sign, digits, optional decimal, optional % or x suffix
# e.g. "50%", "-3.14", "1,234.5", "2x"
_NUMERIC_PATTERN = re.compile(
    r"-?\d[\d,]*(?:\.\d+)?(?:\s*%|\s*x\b)?",
    re.IGNORECASE,
)

# Matches "Answer: 50%", "Answer is 50%", "The answer is 50%", "Answer: Yes", etc.
_ANSWER_MARKER_PATTERN = re.compile(
    r"(?:the\s+)?answer\s*(?:is|:)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?(?:\s*%|\s*x\b)?|yes|no)",
    re.IGNORECASE,
)

# Fallback: last standalone "yes"/"no" token (for yes/no questions when the
# model doesn't follow the "Answer: X" format).
_YESNO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

# Strips a trailing "Answer:" cue from the dataset's pre-formatted query.
_ANSWER_CUE_PATTERN = re.compile(r"\n?Answer:\s*$", re.IGNORECASE)


def normalize_number(text: str) -> str:
    """Strip formatting characters and lowercase for comparison."""
    return text.replace(",", "").replace("$", "").replace(" ", "").strip().lower()


def build_cot_prompt(query: str) -> str:
    """Replace the dataset's trailing 'Answer:' cue with a CoT instruction
    asking the model to end its response with 'Answer: X'."""
    question = _ANSWER_CUE_PATTERN.sub("", query.rstrip())
    return (
        f"{question}\n\n"
        'Think step by step, then end your response with "Answer: X" '
        'where X is the final answer — a number, or "Yes"/"No" if the '
        "question asks for a yes/no judgment."
    )


def extract_predicted_answer(generated_text: str) -> str:
    """
    Extract the final numeric answer from generated text.

    Prefers the last explicit "Answer: X" / "the answer is X" marker,
    then falls back to the last numeric token in the text.
    Returns the raw matched string (not yet normalized).
    """
    text = generated_text.strip()

    # Prefer the last explicit answer marker (CoT responses may mention
    # intermediate numbers before the final "Answer: X").
    marker_matches = _ANSWER_MARKER_PATTERN.findall(text)
    if marker_matches:
        return marker_matches[-1]

    # Fall back to last numeric token in the text
    all_matches = list(_NUMERIC_PATTERN.finditer(text))
    if all_matches:
        return all_matches[-1].group(0)

    # Fall back to last standalone "yes"/"no" token (yes/no questions where
    # the model didn't follow the "Answer: X" format)
    yesno_matches = _YESNO_PATTERN.findall(text)
    if yesno_matches:
        return yesno_matches[-1]

    return text  # nothing found; return full text so it fails comparison gracefully


def _parse_number(text: str) -> Optional[float]:
    """Parse a number string to float, converting a trailing '%' to a decimal
    fraction (e.g. "14.46%" -> 0.1446). Returns None if not parseable."""
    text = text.strip().lower()
    if not text:
        return None
    is_percent = text.endswith("%")
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").replace("x", "").strip()
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value / 100.0 if is_percent else value


def _round_sig(value: float, sig: int) -> float:
    """Round to `sig` significant figures (scale-invariant, unlike decimal rounding)."""
    if value == 0:
        return 0.0
    if not math.isfinite(value):
        return value
    digits = sig - int(math.floor(math.log10(abs(value)))) - 1
    return round(value, digits)


def _sig_fig_count(text: str) -> int:
    """Count significant figures in a numeric string (after stripping
    percent/currency/comma/sign formatting and leading zeros). Returns at
    least 1."""
    cleaned = text.strip().lower().rstrip("%").replace(",", "").replace("$", "").replace("x", "").strip()
    cleaned = cleaned.lstrip("-").lstrip("+")
    digits = cleaned.replace(".", "").lstrip("0")
    return len(digits) if digits else 1


def numbers_match(pred: str, gold: str) -> bool:
    """Numeric comparison with percent-to-decimal normalization, using
    "min-precision" rounding: round both values to the smaller of the two
    operands' significant-figure counts (minimum 1) and compare exactly.
    This tolerates the model reporting a less precise rounding of the same
    value (e.g. "45%" vs gold 0.45103 both round to 0.45 at 2 sig figs)
    while avoiding the rounding-bucket collisions that a fixed coarse
    precision creates — e.g. 9.864% and 9.87% both round to 9.9% at 2 sig
    figs despite being genuinely different values, but at min-precision 3
    (the precision of "9.87%") they round to 9.86% and 9.87% and correctly
    differ.
    Falls back to exact string match if either side isn't a number."""
    pred_val = _parse_number(pred)
    gold_val = _parse_number(gold)
    if pred_val is None or gold_val is None:
        return normalize_number(pred) == normalize_number(gold)
    if pred_val == 0 or gold_val == 0:
        return pred_val == gold_val
    n = max(min(_sig_fig_count(pred), _sig_fig_count(gold)), 1)
    return _round_sig(pred_val, n) == _round_sig(gold_val, n)


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


def generate_response(inputs: Dict, input_len: int) -> Tuple[str, bool, Optional[str]]:
    """
    Generate response and decode it (model-specific).

    Returns:
        (response_text, hit_max_tokens, thinking_text)
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

    output_tokens = outputs[0][input_len:]
    hit_max = len(output_tokens) >= GENERATION_CONFIG["max_new_tokens"]
    thinking_text = None

    if model_type == "gemma" and ENABLE_THINKING:
        raw = processor_or_tokenizer.decode(output_tokens, skip_special_tokens=False)
        parsed = processor_or_tokenizer.parse_response(raw)
        thinking_text = parsed.get("thinking", None)
        response = parsed.get("content", raw)
    elif model_type == "qwen" and ENABLE_THINKING:
        ids = output_tokens.tolist()
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
        response = tokenizer.decode(output_tokens, skip_special_tokens=True)

    return response.strip(), hit_max, thinking_text



def evaluate_t9_convfinqa(num_samples: int = None) -> Tuple[Dict[str, Any], list]:
    """Evaluate T9 Conversational Financial QA task (flare-convfinqa)"""

    logger.info("=" * 60)
    logger.info("T9 Conversational Financial QA Evaluation (flare-convfinqa)")
    logger.info("=" * 60)

    # Load dataset
    data_path = DATA_DIR / "flare_convfinqa_test.json"
    if not data_path.exists():
        logger.error(f"Dataset file not found: {data_path}")
        return {"error": "dataset_not_found"}, []

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if num_samples is not None:
        dataset = dataset[:num_samples]

    logger.info(f"Loaded {len(dataset)} samples")

    correct = 0
    total = 0
    num_truncated = 0
    detailed_results = []

    for i, example in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  Progress: {i}/{len(dataset)}")

        query = example.get("query", "")
        answer = example.get("answer", "")
        example_id = example.get("id", str(i))

        if not query or not answer:
            continue

        prompt = build_cot_prompt(query)
        inputs, input_len = prepare_input(prompt)
        prediction_text, hit_max, thinking_text = generate_response(inputs, input_len)

        predicted_answer = extract_predicted_answer(prediction_text)
        is_correct = numbers_match(predicted_answer, answer)

        if is_correct:
            correct += 1
        if hit_max:
            num_truncated += 1
        total += 1

        detailed_results.append({
            "id": example_id,
            "prompt": prompt,
            "generated_text": prediction_text,
            "thinking": thinking_text,
            "gold_label": answer,
            "predicted_label": predicted_answer,  # extracted answer, for LLM judge later
            "is_correct": is_correct,
            "hit_max_tokens": hit_max,
        })

        if i < 3:
            logger.info(f"\nSample {i}:")
            logger.info(f"  Ground truth: {answer}")
            logger.info(f"  Prediction: {prediction_text}")
            logger.info(f"  Correct: {is_correct}")

    accuracy = correct / total if total > 0 else 0

    results = {
        "task": "T9_convfinqa",
        "dataset": "flare_convfinqa",
        "eval_method": "generation_cot",
        "num_samples": total,
        "accuracy": float(accuracy),
        "match_method": "min_precision",
        "num_truncated": num_truncated,
        "pct_truncated": round(num_truncated / total * 100, 2) if total > 0 else 0,
        "generation_config": GENERATION_CONFIG,
    }

    logger.info("\n" + "=" * 60)
    logger.info("T9 Evaluation Results:")
    logger.info(f"  Num samples: {total}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Truncated (hit max_new_tokens): {num_truncated} / {total} ({results['pct_truncated']}%)")
    logger.info("=" * 60)

    return results, detailed_results


def main():
    parser = argparse.ArgumentParser(description="T9 Conversational Financial QA Evaluation")
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
    log_path = log_dir / f"{model_name}_t9_convfinqa.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to: {log_path}")

    # Auto-generate output path if not specified
    if args.output is None:
        args.output = f"experiments/bf16_{model_name}_t9_convfinqa.json"

    logger.info(f"Loading model: {args.model}...")
    setup_model(args.model)
    logger.info("✓ Model setup complete")

    # Sanity check: compare output with test_llm scripts
    sanity_prompt = "What is quantization in machine learning?"
    logger.info(f"\nSanity check prompt: {sanity_prompt}")
    inputs, input_len = prepare_input(sanity_prompt)
    sanity_output, _, _ = generate_response(inputs, input_len)
    logger.info(f"Sanity check output: {sanity_output}\n")

    # Run evaluation
    results, detailed_results = evaluate_t9_convfinqa(args.num_samples)
    results.update({
        "model": args.model,
        "dtype": str(DTYPE),
        "max_input_length": MAX_INPUT_LENGTH,
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
