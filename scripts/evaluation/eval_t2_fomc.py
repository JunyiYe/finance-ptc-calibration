"""
T2 Central Bank Policy Tendency Evaluation Script (finben-fomc)
Supports multiple model architectures: Llama, Gemma, Qwen

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t2_fomc.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --num-samples 496
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

FOMC_LABELS = ("dovish", "neutral", "hawkish")
FOMC_PATTERN = re.compile(r"\b(dovish|neutral|hawkish)\b")
INVALID_LABEL_IDX = -1

# A/B/C rotation: 3 cyclic permutations of labels → average scores to remove position bias.
LETTERS = ("A", "B", "C")
ROTATIONS = [
    ("dovish", "neutral", "hawkish"),
    ("neutral", "hawkish", "dovish"),
    ("hawkish", "dovish",  "neutral"),
]

# ============================================================
# CURATED "OBVIOUS" SANITY-CHECK EXAMPLES
# ============================================================
# 12 examples drawn from the BF16 baseline run: the highest log-prob-margin,
# correctly-predicted example per gold class (4 x dovish/neutral/hawkish),
# human-eyeballed and confirmed unambiguous. Re-scored at the start of every
# run — if the scoring pipeline (prompt rewriting / rotation / tokenization)
# is broken, these "obvious" cases are the first thing to slip.
OBVIOUS_SANITY_EXAMPLES = [
    {"id": "fomc273", "gold_label": "dovish", "query":
        "Study the sentence below from a central bank's briefing. Categorize it as HAWKISH if it promotes a tightening of monetary policy, DOVISH if it represents an easing of monetary policy, or NEUTRAL if the stance is nonpartisan. Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: Many members were concerned about the still-sensitive state of financial markets and thought that an easing of policy would help to support improvements in market functioning, thereby mitigating some of the downside risks to economic growth.\n"
        "Answer:"},
    {"id": "fomc95", "gold_label": "dovish", "query":
        "Review the sentence from a central bank's communiqué. Designate it as HAWKISH if it expresses a tightening of monetary policy, DOVISH if it reveals an easing of monetary policy, or NEUTRAL if the stance is detached. Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: As a result, the likely pace and extent of policy easing expected by investors increased, and yields on nominal and inflation-indexed Treasury coupon securities fell.\n"
        "Answer:"},
    {"id": "fomc426", "gold_label": "dovish", "query":
        "Study the sentence below from a central bank's briefing. Categorize it as HAWKISH if it promotes a tightening of monetary policy, DOVISH if it represents an easing of monetary policy, or NEUTRAL if the stance is nonpartisan. Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: In response to evidence of a slowdown in economic activity and a rapid waning of inflationary pressures, central banks around the world eased policy sharply.\n"
        "Answer:"},
    {"id": "fomc110", "gold_label": "dovish", "query":
        "Reflect on the given sentence from a central bank's announcement. Assign it as HAWKISH if it endorses a tightening of the monetary policy, DOVISH if it indicates an easing of the monetary policy, or NEUTRAL if the stance is even-handed. Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: Long-term sovereign bond yields declined notably in the advanced economies, in part as foreign central banks announced additional monetary policy easing measures.\n"
        "Answer:"},
    {"id": "fomc460", "gold_label": "neutral", "query":
        "Scrutinize the sentence provided from a central bank's press release. Determine it as HAWKISH if it proposes a tightening of monetary policy, DOVISH if it discloses an easing of monetary policy, or NEUTRAL if the stance is unbiased. Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: In some models, these factors can be explicitly tied to observable economic variables, such as inflation; in other models, the factors represent statistical summaries of the data and have no explicit economic interpretations.\n"
        "Answer:"},
    {"id": "fomc140", "gold_label": "neutral", "query":
        "Scrutinize the sentence provided from a central bank's press release. Determine it as HAWKISH if it proposes a tightening of monetary policy, DOVISH if it discloses an easing of monetary policy, or NEUTRAL if the stance is unbiased. Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: King (1999) shows that, if a long enough interval is given to hit the target, there may be little difference between a price level target and an inflation rate target.\n"
        "Answer:"},
    {"id": "fomc123", "gold_label": "neutral", "query":
        "Dissect the given sentence from a central bank's transcript. Tag it as HAWKISH if it denotes a tightening of monetary policy, DOVISH if it implies an easing of monetary policy, or NEUTRAL if the stance is impartial. Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: Martin Baily and Robert Lawrence (2004) present a much more thorough analysis of this issue and come to a similar conclusion.4 To investigate further the relationship between restructuring and labor productivity, we compared the performance of industry employment to the performance of industry productivity for the period from 2000 to 2002 (chart 5).\n"
        "Answer:"},
    {"id": "fomc383", "gold_label": "neutral", "query":
        "Dissect the given sentence from a central bank's transcript. Tag it as HAWKISH if it denotes a tightening of monetary policy, DOVISH if it implies an easing of monetary policy, or NEUTRAL if the stance is impartial. Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: As you can see, transforming the information available from published statistics into concepts useful for assessing the consequences of monetary policy is, in fact, quite complex.\n"
        "Answer:"},
    {"id": "fomc249", "gold_label": "hawkish", "query":
        "Reflect on the given sentence from a central bank's announcement. Assign it as HAWKISH if it endorses a tightening of the monetary policy, DOVISH if it indicates an easing of the monetary policy, or NEUTRAL if the stance is even-handed. Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: The members also agreed that the risks remained weighted mainly in the direction of greater inflation pressures and that further tightening actions might be necessary to bring about financial conditions that were sufficiently firm to contain upward pressures on labor costs and prices.\n"
        "Answer:"},
    {"id": "fomc489", "gold_label": "hawkish", "query":
        "Review the sentence from a central bank's communiqué. Designate it as HAWKISH if it expresses a tightening of monetary policy, DOVISH if it reveals an easing of monetary policy, or NEUTRAL if the stance is detached. Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: And, as I think all of us—having that expectation and that if the economy continued to progress along the lines that we expected and we continued to see the risks as balanced—do regard it as appropriate to gradually remove accommodation that’s in place by having several interest rate increases this year.\n"
        "Answer:"},
    {"id": "fomc472", "gold_label": "hawkish", "query":
        "Examine the excerpt from a central bank's release below. Classify it as HAWKISH if it advocates for a tightening of monetary policy, DOVISH if it suggests an easing of monetary policy, or NEUTRAL if the stance is unbiased. Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: Inflation is much too high, and I strongly believe that bringing inflation back to our target is a necessary condition for meeting the goals mandated by Congress of price stability and maximum employment on a sustainable basis.\n"
        "Answer:"},
    {"id": "fomc382", "gold_label": "hawkish", "query":
        "Scrutinize the sentence provided from a central bank's press release. Determine it as HAWKISH if it proposes a tightening of monetary policy, DOVISH if it discloses an easing of monetary policy, or NEUTRAL if the stance is unbiased. Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: Participants expected that, with further gradual increases in the federal funds rate, economic activity would expand at a solid rate during the remainder of this year and a moderate pace in the medium term, and that labor market conditions would remain strong.\n"
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
    """Extract the first valid FOMC policy label from generated text."""
    allowed_labels = {choice.lower() for choice in choices} or set(FOMC_LABELS)
    text = prediction_text.strip().lower()

    direct_match = re.match(r"^(dovish|neutral|hawkish)\b", text)
    if direct_match and direct_match.group(1) in allowed_labels:
        return direct_match.group(1)

    answer_match = re.search(
        r"\b(?:answer|stance|label|policy)\s*(?:is|:)?\s*(dovish|neutral|hawkish)\b",
        text,
    )
    if answer_match and answer_match.group(1) in allowed_labels:
        return answer_match.group(1)

    labels = [
        match.group(1)
        for match in FOMC_PATTERN.finditer(text)
        if match.group(1) in allowed_labels
    ]
    unique_labels = list(dict.fromkeys(labels))
    if len(unique_labels) == 1:
        return unique_labels[0]

    return None


def _get_letter_token_ids() -> Dict[str, int]:
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    return {letter: tokenizer.encode(letter, add_special_tokens=False)[0]
            for letter in LETTERS}


def score_abc_rotation(query: str, choices: list) -> Tuple[str, Dict[str, float], list, bool]:
    """
    Score labels using A/B/C single-token options with 3 cyclic rotations.

    Each of the 3 cyclic rotations maps the labels onto fixed single-token
    letters "A"/"B"/"C" in a different order; averaging the per-label
    log-probs across rotations cancels any positional/identity bias the model
    has toward a given letter (each label occupies each position exactly once).

    Returns (best_label, label_scores, rotation_details, rotation_consistent):
      - label_scores: per-label log-prob averaged over the 3 rotations (i.e.
        sum of the 3 single-letter log-probs / len(ROTATIONS)) — on the same
        scale as a typical single-token log-prob, so margins/magnitudes read
        naturally. argmax is unaffected by this scaling.
      - rotation_details: one entry per rotation — full prompt, raw per-letter
        log-probs, and that rotation's argmax label (mapped back through the
        rotation's letter->label assignment). Persisted into detailed_results
        so every prompt and its scored log-probs can be inspected post-hoc.
      - rotation_consistent: True iff all 3 rotations independently picked the
        same label as their argmax — i.e. the model is tracking label content
        rather than letter position. This is the core assumption the rotation
        scheme relies on; tracking it is this method's analogue of T1's
        tokenization boundary-merge assertion.
    """
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    letter_ids = _get_letter_token_ids()
    label_scores = {c: 0.0 for c in choices}
    rotation_details = []

    for rotation in ROTATIONS:
        # Build prompt with this rotation's ordering
        choice_str = "\n".join(f"{l}. {v}" for l, v in zip(LETTERS, rotation))
        q = query.replace("HAWKISH", "hawkish").replace("DOVISH", "dovish").replace("NEUTRAL", "neutral")
        q = q.replace("Your response should return only hawkish, dovish, or neutral.", "")
        q = q.replace(
            "\nAnswer:",
            f"\nChoose one of the following options:\n{choice_str}\n\nAnswer with only A, B, or C.\nAnswer:",
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
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
            "prompt": q,
            "letter_order": dict(zip(LETTERS, rotation)),
            "letter_logprobs": letter_logprobs,
            "argmax_label": argmax_label,
        })

        for letter, label in zip(LETTERS, rotation):
            label_scores[label] += letter_logprobs[letter]

    # Normalize sum -> average per-rotation log-prob. argmax is invariant to
    # this (positive constant divisor), so `best`/predictions are unaffected;
    # it only rescales `choice_scores`/`margin` to read as a typical
    # single-token log-prob magnitude instead of a sum of 3.
    label_scores = {label: total / len(ROTATIONS) for label, total in label_scores.items()}
    best = max(label_scores, key=label_scores.__getitem__)
    rotation_consistent = len({d["argmax_label"] for d in rotation_details}) == 1
    return best, label_scores, rotation_details, rotation_consistent


def run_obvious_sanity_checks() -> list:
    """
    Re-score the 12 curated "obvious" examples (OBVIOUS_SANITY_EXAMPLES)
    and record whether the model reproduces the human-verified expected label.

    score_abc_rotation() has no fail-fast assertion to wrap (the ABC scheme
    scores fixed single-token letters, so there's no tokenization boundary-merge
    risk the way score_choices() has in T1) — these calls simply re-run the same
    scoring path the main loop uses, on cases whose answer a human has confirmed.
    """
    records = []
    for ex in OBVIOUS_SANITY_EXAMPLES:
        pred_label, choice_scores, rotation_details, rotation_consistent = score_abc_rotation(
            ex["query"], list(FOMC_LABELS)
        )
        ranked = sorted(choice_scores.values(), reverse=True)
        records.append({
            "id": ex["id"],
            "query": ex["query"],
            "gold_label": ex["gold_label"],
            "predicted_label": pred_label,
            "choice_scores": choice_scores,
            "margin": ranked[0] - ranked[1],
            "rotation_consistent": rotation_consistent,
            "rotations": rotation_details,
            "matches_expected": pred_label == ex["gold_label"],
        })
    return records


def evaluate_t2_fomc(num_samples: int = None) -> Tuple[Dict[str, Any], list]:
    """Evaluate T2 Central Bank Policy Tendency task (finben-fomc)"""

    logger.info("=" * 60)
    logger.info("T2 Central Bank Policy Tendency Evaluation (finben-fomc)")
    logger.info("=" * 60)

    # Load dataset
    data_path = DATA_DIR / "finben_fomc_test.json"
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
    label_map = {"dovish": 0, "neutral": 1, "hawkish": 2}
    reverse_map = {0: "dovish", 1: "neutral", 2: "hawkish"}

    # Store detailed prediction results
    detailed_results = []
    rotation_consistent_count = 0

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

        pred_label, choice_scores, rotation_details, rotation_consistent = score_abc_rotation(query, choices)
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
            "choice_scores": choice_scores,
            "rotation_consistent": rotation_consistent,
            "rotations": rotation_details,
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
        "task": "T2_fomc_policy",
        "dataset": "finben_fomc",
        "eval_method": "abc_rotation",
        "num_samples": len(gold_labels),
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "rotation_consistency": {
            "description": (
                "Fraction of samples where all 3 A/B/C rotations independently "
                "picked the same label as their argmax — i.e. the model is "
                "tracking label content rather than letter position. This is "
                "the core assumption the rotation-averaging scheme relies on; "
                "per-sample detail is in 'rotation_consistent' / 'rotations' "
                "within the _detailed.json file."
            ),
            "num_consistent": rotation_consistent_count,
            "num_total": len(detailed_results),
            "fraction_consistent": (
                rotation_consistent_count / len(detailed_results) if detailed_results else None
            ),
        },
    }

    logger.info("\n" + "=" * 60)
    logger.info("T2 Evaluation Results:")
    logger.info(f"  Num samples: {len(gold_labels)}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Weighted F1: {weighted_f1:.4f}")
    logger.info(f"  Per-class F1:")
    for label, f1 in per_class_f1.items():
        logger.info(f"    {label}: {f1:.4f}")
    logger.info(f"  Rotation consistency: {rotation_consistent_count}/{len(detailed_results)} "
                f"({results['rotation_consistency']['fraction_consistent']:.4f})")
    logger.info("=" * 60)

    return results, detailed_results


def main():
    parser = argparse.ArgumentParser(description="T2 Central Bank Policy Tendency Evaluation")
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
    log_path = log_dir / f"{model_name}_t2_fomc.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to: {log_path}")

    # Auto-generate output path if not specified
    if args.output is None:
        args.output = f"experiments/bf16_{model_name}_t2_fomc.json"

    logger.info(f"Loading model: {args.model}...")
    setup_model(args.model)
    logger.info("✓ Model setup complete")

    # Sanity check 1: generation output
    sanity_prompt = "What is quantization in machine learning?"
    logger.info(f"\nSanity check prompt: {sanity_prompt}")
    inputs, input_len = prepare_input(sanity_prompt)
    sanity_output, _ = generate_response(inputs, input_len)
    logger.info(f"Sanity check output: {sanity_output}\n")

    # Sanity check 2: an obviously hawkish statement should score "hawkish" highest.
    hawkish_query = (
        "Examine the excerpt from a central bank's release below. Classify it as HAWKISH if it "
        "advocates for a tightening of monetary policy, DOVISH if it suggests an easing of "
        "monetary policy, or NEUTRAL if the stance is unbiased. "
        "Your response should return only HAWKISH, DOVISH, or NEUTRAL.\n"
        "Text: We must raise interest rates aggressively to combat persistent inflation.\n"
        "Answer:"
    )
    _, sc_scores, _, _ = score_abc_rotation(hawkish_query, list(FOMC_LABELS))
    logger.info(f"Sanity check (obviously hawkish) scores: {sc_scores}")
    if max(sc_scores, key=sc_scores.__getitem__) != "hawkish":
        logger.warning("WARNING: 'hawkish' did not score highest — check scoring.")

    # Sanity check 3: 12 curated "obvious" examples (human-verified,
    # high log-prob-margin, 4 per gold class). Re-scored every run; the full
    # sentences + per-rotation prompts/scores + pass/fail are persisted into
    # the summary JSON below so correctness can be confirmed post-hoc.
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

    # Run evaluation
    results, detailed_results = evaluate_t2_fomc(args.num_samples)
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
                "scoring pipeline (prompt rewriting / rotation / tokenization). "
                "Each entry also carries its own 'rotations' (full per-rotation "
                "prompt + raw A/B/C log-probs), mirroring the main loop's detail."
            ),
            "num_matches_expected": obvious_sanity_passed,
            "num_total": len(obvious_sanity_results),
            "examples": obvious_sanity_results,
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
