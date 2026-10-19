"""
T6 Relation Classification Evaluation Script (fingpt-finred-cls)
Supports multiple model architectures: Llama, Gemma, Qwen

Eval method: A/B/C/D rotation log-prob (4 cyclic rotations of each example's
own 4 choices -> A/B/C/D). FinRED relation labels vary widely in token length
("industry" vs "parent organization" vs "chief executive officer"), which
rules out direct multi-token label scoring (T1's score_choices) due to length
bias. Mapping each example's 4 choices onto fixed single-token letters and
averaging over 4 cyclic rotations (each choice occupies each letter position
exactly once) removes both the length bias and any positional/identity bias
the model has toward a given letter — mirrors T2's score_abc_rotation /
T8's score_ab_rotation, generalized to per-example (not global) choice sets.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n finquant-lmc-main python scripts/evaluation/eval_t6_finred.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --num-samples 8928
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
# Only used for sanity check 1 (generic free-form generation at startup).
GENERATION_CONFIG = {
    "max_new_tokens": 50,
    "do_sample": False,  # Greedy decoding
}

ENABLE_THINKING = False      # override with --enable-thinking at runtime
MAX_NEW_TOKENS_THINKING = 8192

MAX_INPUT_LENGTH = 2048
DEVICE_MAP = "auto"
DTYPE = torch.bfloat16

# ============================================================
# MODEL-SPECIFIC SETUP (loaded in main())
# ============================================================
model = None
processor_or_tokenizer = None  # Either AutoTokenizer or AutoProcessor
model_type = None  # "llama", "gemma", "qwen"

# A/B/C/D rotation: each example supplies its own 4 choices; cyclic_rotations()
# produces 4 cyclic permutations of THOSE choices (one per letter-position).
LETTERS = ("A", "B", "C", "D")

# Matches the dataset's standard prompt shape:
#   <instruction mentioning 'entity1' and 'entity2'>
#   Options: <comma-separated choices>
#   Text: <context>
#   Answer:
PROMPT_RE = re.compile(r"^(.*?)\nOptions:.*?\nText:(.*)\nAnswer:\s*$", re.DOTALL)

# ============================================================
# CURATED "OBVIOUS" SANITY-CHECK EXAMPLES
# ============================================================
# 12 examples drawn from the BF16 baseline run: rotation-consistent,
# correctly-predicted, highest-margin example per gold label across 12
# distinct relation types, human-eyeballed and confirmed unambiguous.
# Re-scored at the start of every run — if the scoring pipeline (prompt
# rewriting / rotation / tokenization) is broken, these "obvious" cases are
# the first thing to slip. Mirrors T2's/T8's OBVIOUS_SANITY_EXAMPLES.
OBVIOUS_SANITY_EXAMPLES = [
    {"id": "finredcls2241", "gold_label": "member of",
     "choices": ["distribution format", "business division", "location of formation", "member of"],
     "query": "What is the relationship between 'David Gilmour' and 'Pink Floyd' in the context of the input sentence.\nOptions: distribution format, business division, location of formation, member of\nText: 2 COMMENTS David Gilmour spent the last 50 years making music with Pink Floyd , and in that time he has released just a handful of solo albums.\nAnswer:"},
    {"id": "finredcls8570", "gold_label": "chief executive officer",
     "choices": ["chief executive officer", "publisher", "creator", "manufacturer"],
     "query": "Take context from the input text and decide on the accurate relationship between 'Rosneft' and 'Igor Sechin' from the options provided.\nOptions: chief executive officer, publisher, creator, manufacturer\nText: The goods trade between the two countries will grow by more than 50 percent,\" Rosneft chairman Igor Sechin said in a statement.\nAnswer:"},
    {"id": "finredcls1450", "gold_label": "brand",
     "choices": ["employer", "stock exchange", "distribution format", "brand"],
     "query": "In the context of the input sentence, determine the relationship between 'Chevrolet Camaro' and 'Chevrolet'.\nOptions: employer, stock exchange, distribution format, brand\nText: Meanwhile, the other Oshawa assembly line manufactures the current models of Chevrolet Impala, Buick Regal, Chevrolet Camaro and Cadillac XTS.In 2005, General Motors announced for the first time, the plans of closing the plant in 2008.\nAnswer:"},
    {"id": "finredcls1909", "gold_label": "chairperson",
     "choices": ["chairperson", "distributed by", "location of formation", "original broadcaster"],
     "query": "Refer to the input text as context and select the correct relationship between 'Eurogroup' and 'Jeroen Dijsselbloem' from the available options.\nOptions: chairperson, distributed by, location of formation, original broadcaster\nText: Eurozone finance ministers will make a 'major' decision Saturday on the latest proposals from Athens, Eurogroup chief Jeroen Dijsselbloem said Friday.\nAnswer:"},
    {"id": "finredcls2137", "gold_label": "position held",
     "choices": ["position held", "owner of", "manufacturer", "publisher"],
     "query": "Refer to the input text as context and select the correct relationship between 'Marcantonio Barbarigo' and 'Cardinal' from the available options.\nOptions: position held, owner of, manufacturer, publisher\nText: The sight would have deeply distressed Cardinal Marcantonio Barbarigo who founded the seminary in the 1600s.\nAnswer:"},
    {"id": "finredcls8203", "gold_label": "owned by",
     "choices": ["owned by", "publisher", "chairperson", "manufacturer"],
     "query": "Refer to the input text as context and select the correct relationship between 'PayPal' and 'eBay' from the available options.\nOptions: owned by, publisher, chairperson, manufacturer\nText: Instead, within a year, PayPal was sold to eBay for $1.5 billion, and its founders and executives went on to become Silicon Valley luminaries.\nAnswer:"},
    {"id": "finredcls3068", "gold_label": "product/material produced",
     "choices": ["location of formation", "chief executive officer", "product/material produced", "distributed by"],
     "query": "Take context from the input text and decide on the accurate relationship between 'Lockheed Martin' and 'Orion' from the options provided.\nOptions: location of formation, chief executive officer, product/material produced, distributed by\nText: Leveraging designs and technologies used on our Orion program, Lockheed Martin will produce the crew ascent element, the vehicle which will transport astronauts from the lunar surface to begin their journey back to earth.\nAnswer:"},
    {"id": "finredcls8763", "gold_label": "subsidiary",
     "choices": ["subsidiary", "creator", "manufacturer", "developer"],
     "query": "What is the relationship between 'Lufthansa' and 'Austrian Airlines' in the context of the input sentence.\nOptions: subsidiary, creator, manufacturer, developer\nText: Lufthansa Passenger Airlines, SWISS and Austrian Airlines are network carriers serving the global market and all passenger segments.\nAnswer:"},
    {"id": "finredcls1394", "gold_label": "owner of",
     "choices": ["location of formation", "position held", "platform", "owner of"],
     "query": "Take context from the input text and decide on the accurate relationship between 'Merlin Entertainments' and 'Legoland' from the options provided.\nOptions: location of formation, position held, platform, owner of\nText: Britain's Merlin Entertainments (LSE: MERL.L - news ) rose 2 after announcing it will take its Legoland amusement park to Shanghai as part of a deal agreed with investment firm China Media (Other OTC: CHMD - news ) Capital (Other OTC: CGHC - news ) during the state visit of President Xi Jinping.\nAnswer:"},
    {"id": "finredcls7030", "gold_label": "currency",
     "choices": ["creator", "employer", "currency", "business division"],
     "query": "In the context of the input sentence, determine the relationship between 'eurozone' and 'euro'.\nOptions: creator, employer, currency, business division\nText: That helps lay the groundwork for investors to resume so-called ECB trade, a combination of long eurozone stock, long eurozone bond and short euro positions.\nAnswer:"},
    {"id": "finredcls1523", "gold_label": "employer",
     "choices": ["employer", "industry", "subsidiary", "operator"],
     "query": "Analyze the relationship between 'Dennis Muilenburg' and 'Boeing' within the context of the input sentence.\nOptions: employer, industry, subsidiary, operator\nText: “By continuing to profitably deliver on our large and diverse backlog, we are driving strong growth in revenue, earnings and cash flow,” Boeing president and CEO Dennis Muilenburg said.\nAnswer:"},
    {"id": "finredcls2679", "gold_label": "industry",
     "choices": ["manufacturer", "industry", "distributed by", "currency"],
     "query": "What is the relationship between 'Glencore' and 'mining' in the context of the input sentence.\nOptions: manufacturer, industry, distributed by, currency\nText: On the London exchange, shares in Swiss-based mining giant Glencore Xstrata fell more than 30 per cent.\nAnswer:"},
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
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING,
        )
    except TypeError:
        return processor_or_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except ValueError:
        logger.warning("Chat template is unavailable for this model; using raw prompt.")
        return query


def prepare_input(query: str) -> Tuple[Dict, int]:
    if model_type not in {"llama", "gemma", "qwen"}:
        raise ValueError(f"Unsupported model type: {model_type}")

    text = format_chat_prompt(query)
    inputs = processor_or_tokenizer(
        text=text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        add_special_tokens=False,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    return inputs, input_len


def generate_response(inputs: Dict, input_len: int) -> Tuple[str, Any]:
    """
    Generate response and decode it (model-specific). Only used for sanity
    check 1 (generic free-form generation at startup).

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


def _get_letter_token_ids() -> Dict[str, int]:
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    return {letter: tokenizer.encode(letter, add_special_tokens=False)[0]
            for letter in LETTERS}


def cyclic_rotations(choices: List[str]) -> List[List[str]]:
    """4 cyclic permutations of this example's own 4 choices — each choice
    occupies each letter position (A/B/C/D) exactly once."""
    n = len(choices)
    return [choices[i:] + choices[:i] for i in range(n)]


def build_rotation_prompt(instruction: str, text: str, rotation: List[str]) -> str:
    choice_str = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, rotation))
    return (
        f"{instruction}\n"
        f"Options:\n{choice_str}\n"
        f"Text:{text}\n"
        f"Answer with only A, B, C, or D.\nAnswer:"
    )


def score_abcd_rotation(query: str, choices: List[str]) -> Tuple[str, Dict[str, float], list, bool]:
    """
    Score relation labels using A/B/C/D single-token options with 4 cyclic
    rotations of THIS EXAMPLE's own 4 choices.

    Unlike T2/T8 (global, fixed label set -> a single hardcoded ROTATIONS
    list), T6 draws each example's 4 choices from a ~25-label pool, so the
    rotations are computed per example via cyclic_rotations(choices).
    Averaging the per-label log-probs across the 4 rotations cancels any
    positional/identity bias the model has toward a given letter (each choice
    occupies each position exactly once), and sidesteps the severe
    token-length mismatch between candidate labels (e.g. "industry" = 1
    token vs "chief executive officer" = 4 tokens) that rules out direct
    multi-token label scoring (T1's score_choices).

    Returns (best_label, label_scores, rotation_details, rotation_consistent):
      - label_scores: per-label log-prob averaged over the 4 rotations.
      - rotation_details: one entry per rotation — full prompt, raw per-letter
        log-probs, and that rotation's argmax label (mapped back through the
        rotation's letter->label assignment). Persisted into detailed_results
        so every prompt and its scored log-probs can be inspected post-hoc.
      - rotation_consistent: True iff all 4 rotations independently picked the
        same label as their argmax — i.e. the model is tracking label content
        rather than letter position.
    """
    m = PROMPT_RE.match(query)
    if not m:
        raise ValueError(f"Query did not match expected T6 prompt pattern: {query[:200]!r}")
    instruction, text = m.group(1), m.group(2)

    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    letter_ids = _get_letter_token_ids()
    label_scores = {c: 0.0 for c in choices}
    rotation_details = []
    rotations = cyclic_rotations(choices)

    for rotation in rotations:
        q = build_rotation_prompt(instruction, text, rotation)
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

    label_scores = {label: total / len(rotations) for label, total in label_scores.items()}
    best = max(label_scores, key=label_scores.__getitem__)
    rotation_consistent = len({d["argmax_label"] for d in rotation_details}) == 1
    return best, label_scores, rotation_details, rotation_consistent


def run_obvious_sanity_checks() -> list:
    """
    Re-score the 12 curated "obvious" examples (OBVIOUS_SANITY_EXAMPLES) and
    record whether the model reproduces the human-verified expected label.

    Mirrors T2's/T8's run_obvious_sanity_checks() — these calls simply re-run
    the same scoring path the main loop uses, on cases whose answer a human
    has confirmed.
    """
    records = []
    for ex in OBVIOUS_SANITY_EXAMPLES:
        pred_label, choice_scores, rotation_details, rotation_consistent = score_abcd_rotation(
            ex["query"], ex["choices"]
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


def evaluate_t6_finred(num_samples: int = None) -> Tuple[Dict[str, Any], list]:
    """Evaluate T6 Relation Classification task (fingpt-finred-cls)"""

    logger.info("=" * 60)
    logger.info("T6 Relation Classification Evaluation (fingpt-finred-cls)")
    logger.info("=" * 60)

    data_path = DATA_DIR / "fingpt_finred_cls_test.json"
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
    detailed_results = []
    rotation_consistent_count = 0

    for i, example in enumerate(dataset):
        if i % 500 == 0:
            logger.info(f"  Progress: {i}/{len(dataset)}")

        query = example.get("query", "").strip()
        answer = example.get("answer", "").strip()  # ground truth (original casing)
        choices = [c.strip().lower() for c in example.get("choices", [])]
        example_id = example.get("id", str(i))

        if not query or not answer:
            continue

        pred_label, choice_scores, rotation_details, rotation_consistent = score_abcd_rotation(query, choices)
        if rotation_consistent:
            rotation_consistent_count += 1

        gold = answer.lower()
        is_correct = pred_label == gold

        predictions.append(pred_label)
        gold_labels.append(gold)

        detailed_results.append({
            "id": example_id,
            "prompt": query,
            "gold_label": answer,
            "predicted_label": pred_label,
            "choices": choices,
            "choice_scores": choice_scores,
            "rotation_consistent": rotation_consistent,
            "rotations": rotation_details,
            "is_correct": is_correct,
        })

        if i < 3:
            logger.info(f"\nSample {i}:")
            logger.info(f"  Ground truth: {answer}")
            logger.info(f"  Prediction: {pred_label}")
            logger.info(f"  Scores: {choice_scores}")
            logger.info(f"  Rotation consistent: {rotation_consistent}")

    accuracy = accuracy_score(gold_labels, predictions) if gold_labels else 0.0
    weighted_f1 = f1_score(gold_labels, predictions, average="weighted", zero_division=0) if gold_labels else 0.0
    macro_f1 = f1_score(gold_labels, predictions, average="macro", zero_division=0) if gold_labels else 0.0

    results = {
        "task": "T6_relation_classification",
        "dataset": "fingpt_finred_cls",
        "eval_method": "abcd_rotation",
        "num_samples": len(gold_labels),
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "rotation_consistency": {
            "description": (
                "Fraction of samples where all 4 A/B/C/D rotations independently "
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
        "generation_config": GENERATION_CONFIG,
    }

    logger.info("\n" + "=" * 60)
    logger.info("T6 Evaluation Results:")
    logger.info(f"  Num samples: {len(gold_labels)}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Weighted F1: {weighted_f1:.4f}")
    logger.info(f"  Macro F1: {macro_f1:.4f}")
    logger.info(f"  Rotation consistency: {rotation_consistent_count}/{len(detailed_results)} "
                f"({results['rotation_consistency']['fraction_consistent']:.4f})")
    logger.info("=" * 60)

    return results, detailed_results


def main():
    parser = argparse.ArgumentParser(description="T6 Relation Classification Evaluation")
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
    log_path = log_dir / f"{model_name}_t6_finred.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to: {log_path}")

    if args.output is None:
        args.output = f"experiments/bf16_{model_name}_t6_finred.json"

    logger.info(f"Loading model: {args.model}...")
    setup_model(args.model)
    logger.info("✓ Model setup complete")

    # Sanity check 1: generic generation
    sanity_prompt = "What is quantization in machine learning?"
    logger.info(f"\nSanity check prompt: {sanity_prompt}")
    inputs, input_len = prepare_input(sanity_prompt)
    sanity_output, _ = generate_response(inputs, input_len)
    logger.info(f"Sanity check output: {sanity_output}\n")

    # Sanity check 2: an obvious "owned by" example (PayPal/eBay) should score
    # "owned by" highest.
    obvious_ex = next(ex for ex in OBVIOUS_SANITY_EXAMPLES if ex["id"] == "finredcls8203")
    _, sc_scores, _, _ = score_abcd_rotation(obvious_ex["query"], obvious_ex["choices"])
    logger.info(f"Sanity check (PayPal/eBay 'owned by') scores: {sc_scores}")
    if max(sc_scores, key=sc_scores.__getitem__) != obvious_ex["gold_label"]:
        logger.warning("WARNING: 'owned by' did not score highest — check scoring.")

    # Sanity check 3: 12 curated "obvious" examples (human-verified,
    # rotation-consistent, high log-prob-margin, one per gold label). Re-scored
    # every run; the full prompts + per-rotation prompts/scores + pass/fail are
    # persisted into the summary JSON below so correctness can be confirmed
    # post-hoc.
    logger.info("\nSanity check (12 curated 'obvious' examples):")
    obvious_sanity_results = run_obvious_sanity_checks()
    obvious_sanity_passed = sum(r["matches_expected"] for r in obvious_sanity_results)
    obvious_sanity_inconsistent = [r for r in obvious_sanity_results if not r["rotation_consistent"]]
    for r in obvious_sanity_results:
        status = "OK  " if r["matches_expected"] else "MISS"
        rot = "  " if r["rotation_consistent"] else "**"
        logger.info(f"  [{status}] id={r['id']} gold={r['gold_label']:<25} pred={str(r['predicted_label']):<25} "
                    f"margin={r['margin']:.4f} rotation_consistent={r['rotation_consistent']}{rot}")
    logger.info(f"  -> {obvious_sanity_passed}/{len(obvious_sanity_results)} matched expected label, "
                f"{len(obvious_sanity_inconsistent)} rotation-inconsistent")
    if obvious_sanity_passed < len(obvious_sanity_results):
        logger.warning("WARNING: not all 'obvious' sanity-check examples matched their expected label — inspect 'obvious_sanity_check' in the output JSON.")
    if obvious_sanity_inconsistent:
        logger.warning(f"WARNING: {len(obvious_sanity_inconsistent)} rotation-inconsistent sanity-check example(s) — see 'obvious_sanity_check' in the output JSON.")

    # Run evaluation
    results, detailed_results = evaluate_t6_finred(args.num_samples)
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
                "12 human-verified, rotation-consistent, high log-prob-margin "
                "examples from the BF16 baseline run (eyeball check on obvious "
                "cases), one per distinct gold relation label. Re-scored every "
                "run as a built-in correctness check on the scoring pipeline "
                "(prompt rewriting / rotation / tokenization). Each entry also "
                "carries its own 'rotations' (full per-rotation prompt + raw "
                "A/B/C/D log-probs), mirroring the main loop's detail."
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
