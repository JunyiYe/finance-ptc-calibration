"""
T4 CFA Exam QA Evaluation Script (flare-cfa)
Supports multiple model architectures: Llama, Gemma, Qwen

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t4_cfa.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --num-samples 1032
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from sklearn.metrics import accuracy_score

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

CFA_LABELS = ("A", "B", "C")
INVALID_LABEL_IDX = -1


# "Q:<question>,CHOICES: A: <a>,B: <b>,C: <c>" -> split into question + per-letter choice text.
CFA_TEXT_PATTERN = re.compile(r"^Q:(?P<question>.*),CHOICES:\s*(?P<choices>.*)$", re.DOTALL)
CFA_CHOICE_SPLIT_PATTERN = re.compile(r",(?=[ABC]:\s)")


def build_cot_prompt(text: str) -> str:
    """Reformat the dataset's 'Q:...,CHOICES: A: ...,B: ...,C: ...' text into a
    multi-line CoT prompt asking the model to end with 'Answer: X'."""
    match = CFA_TEXT_PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"Unexpected CFA text format: {text!r}")

    question = match.group("question").strip()
    choices = {}
    for part in CFA_CHOICE_SPLIT_PATTERN.split(match.group("choices").strip()):
        label, _, value = part.partition(":")
        choices[label.strip()] = value.strip()

    return (
        "Read the question and choose the most appropriate option among A, B, and C.\n"
        f"Q: {question}\n"
        f"A: {choices['A']}\n"
        f"B: {choices['B']}\n"
        f"C: {choices['C']}\n"
        "\n"
        'Think step by step, then end your response with "Answer: X" where X is A, B, or C.'
    )


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



ANSWER_PATTERN = re.compile(r"answer\s*(?:is|:)\s*(?:option\s*)?\(?([ABC])\)?\b", re.IGNORECASE)
CFA_PATTERN = re.compile(r"\b([ABC])\b")


def extract_predicted_label(prediction_text: str, choices: list, hit_max: bool = False) -> Optional[str]:
    """Extract the model's final A/B/C answer from a CoT generation.

    Prefers the last explicit "Answer: X" (the prompt asks the model to end
    with this), since earlier reasoning may mention other options by letter.
    If generation was truncated before an "Answer: X" marker appeared, treat
    as invalid rather than falling back to a stray A/B/C — truncated CFA
    reasoning often repeats formula variables like "A = P(1 + r/n)^(nt)",
    which are not the model's chosen option.
    """
    allowed_labels = {choice.upper() for choice in choices} or set(CFA_LABELS)
    text = prediction_text.strip()

    answer_matches = ANSWER_PATTERN.findall(text)
    if answer_matches:
        label = answer_matches[-1].upper()
        if label in allowed_labels:
            return label

    if hit_max:
        return None

    labels = [m.upper() for m in CFA_PATTERN.findall(text) if m.upper() in allowed_labels]
    unique_labels = list(dict.fromkeys(labels))
    if len(unique_labels) == 1:
        return unique_labels[0]

    return None


def evaluate_t4_cfa(num_samples: int = None) -> Tuple[Dict[str, Any], list]:
    """Evaluate T4 CFA Exam QA task (flare-cfa)"""

    logger.info("=" * 60)
    logger.info("T4 CFA Exam QA Evaluation (flare-cfa)")
    logger.info("=" * 60)

    # Load dataset
    data_path = DATA_DIR / "flare_cfa_test.json"
    if not data_path.exists():
        logger.error(f"Dataset file not found: {data_path}")
        return {"error": "dataset_not_found"}, []

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if num_samples is not None:
        dataset = dataset[:num_samples]

    logger.info(f"Loaded {len(dataset)} samples")

    predictions = []
    gold_labels = []
    choice_map = {"A": 0, "B": 1, "C": 2}

    # Store detailed prediction results
    detailed_results = []
    num_truncated = 0

    for i, example in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  Progress: {i}/{len(dataset)}")

        # Extract example information
        text = example.get("text", "").strip()
        answer = example.get("answer", "").strip().upper()  # ground truth, normalized to A/B/C
        choices = [choice.strip() for choice in example.get("choices", [])]
        example_id = example.get("id", str(i))

        if not text or not answer:
            continue

        # Generate a CoT response and extract the final "Answer: X"
        query = build_cot_prompt(text)
        inputs, input_len = prepare_input(query)
        response, hit_max, thinking_text = generate_response(inputs, input_len)
        if hit_max:
            num_truncated += 1

        pred_answer = extract_predicted_label(response, choices, hit_max)

        pred_idx = choice_map.get(pred_answer, INVALID_LABEL_IDX)
        gold_idx = choice_map.get(answer, INVALID_LABEL_IDX)
        if gold_idx == INVALID_LABEL_IDX:
            raise ValueError(f"Invalid gold label for example {example_id}: {answer}")

        predictions.append(pred_idx)
        gold_labels.append(gold_idx)

        # Save detailed information
        detailed_results.append({
            "id": example_id,
            "prompt": query,
            "gold_label": answer,
            "predicted_label": pred_answer,
            "response": response,
            "hit_max_tokens": hit_max,
            "is_correct": pred_answer == answer,
        })

        if i < 3:
            logger.info(f"\nSample {i}:")
            logger.info(f"  Ground truth: {answer}")
            logger.info(f"  Prediction: {pred_answer}")
            logger.info(f"  Response: {response}")

    # Calculate metrics. Invalid predictions stay in `predictions` as -1, so
    # accuracy counts them as wrong.
    accuracy = accuracy_score(gold_labels, predictions)
    invalid_count = sum(1 for pred in predictions if pred == INVALID_LABEL_IDX)
    invalid_rate = invalid_count / len(predictions) if predictions else 0.0

    total = len(gold_labels)
    results = {
        "task": "T4_cfa_qa",
        "dataset": "flare_cfa",
        "eval_method": "generation_cot",
        "num_samples": total,
        "accuracy": float(accuracy),
        "invalid_predictions": invalid_count,
        "invalid_prediction_rate": invalid_rate,
        "num_truncated": num_truncated,
        "pct_truncated": round(100 * num_truncated / total, 2) if total else 0.0,
    }

    logger.info("\n" + "=" * 60)
    logger.info("T4 Evaluation Results:")
    logger.info(f"  Num samples: {total}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Truncated (hit max_new_tokens): {num_truncated} / {total} ({results['pct_truncated']}%)")
    logger.info("=" * 60)

    return results, detailed_results


def main():
    parser = argparse.ArgumentParser(description="T4 CFA Exam QA Evaluation")
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
    log_path = log_dir / f"{model_name}_t4_cfa.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to: {log_path}")

    # Auto-generate output path if not specified
    if args.output is None:
        args.output = f"experiments/bf16_{model_name}_t4_cfa.json"

    logger.info(f"Loading model: {args.model}...")
    setup_model(args.model)
    logger.info("✓ Model setup complete")

    # Sanity check: an obvious CFA question (answer=C) should be answered correctly.
    sanity_text = (
        "Q:The nominal risk-free rate is best described as the sum of the real risk-free "
        "rate and a premium for:,CHOICES: A: maturity.,B: liquidity.,C: expected inflation."
    )
    sanity_query = build_cot_prompt(sanity_text)
    sanity_inputs, sanity_input_len = prepare_input(sanity_query)
    sanity_response, _, _ = generate_response(sanity_inputs, sanity_input_len)
    sanity_pred = extract_predicted_label(sanity_response, ["A", "B", "C"])
    logger.info(f"Sanity check (answer=C) response: {sanity_response}")
    logger.info(f"Sanity check (answer=C) predicted: {sanity_pred}")
    if sanity_pred != "C":
        logger.warning("WARNING: sanity check did not predict 'C'.")

    # Run evaluation
    results, detailed_results = evaluate_t4_cfa(args.num_samples)
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
