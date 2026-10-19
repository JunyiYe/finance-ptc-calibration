#!/usr/bin/env python3
"""
LLM error attribution for calibration-discordant samples (T5 FinQA / T9 ConvFinQA).

For samples where the base-calibration model (e.g. c1) is wrong and the
treat-calibration model (e.g. c3) is right, ask DeepSeek to attribute the
base model's error to one of:

    extraction  — picked the wrong number(s) from the table/text
    arithmetic  — right numbers, wrong computation
    unit_scale  — right computation, wrong unit/scale/sign convention
                  (percent vs decimal, millions vs raw, sign flip)
    reasoning   — wrong solution approach / misunderstood the question
    format      — final value effectively right but expressed unparseably
    other       — none of the above

Requires DEEPSEEK_API_KEY.

Usage:
    conda run -n finquant-lmc-main python scripts/llm_judgement/attribute_discordant_errors.py \
        --model Llama-3.1-8B-Instruct --method gptq_w4a16 --task t5 \
        [--base c1 --treat c3] [--num-samples 20] [--max-workers 100]

Output:
    experiments/results/analysis/error_attribution_<method>_<base>to<treat>_<model>_<task>.json
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

JUDGE_MODEL = "deepseek-v4-pro"
RESULTS_ROOT = Path("experiments/results")
import sys as _sys; _sys.path.insert(0, "scripts/analysis")
from paper_metrics import FINMIX_REDIRECT
OUT_DIR = RESULTS_ROOT / "analysis"

CATEGORIES = ["extraction", "arithmetic", "unit_scale", "reasoning", "format", "other"]
_CAT_RE = re.compile(r"\b(extraction|arithmetic|unit_scale|reasoning|format|other)\b",
                     re.IGNORECASE)

MAX_PROMPT_CHARS = 9000  # cap the source context sent to the judge


def make_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def load_detailed(model: str, cfg: str, tnum: str) -> dict | None:
    cfg = FINMIX_REDIRECT.get((model, cfg), cfg)
    base = RESULTS_ROOT / model
    for suffix in (f"_t{tnum}_detailed_llm_adjudicated.json",
                   f"_t{tnum}_detailed.json"):
        p = base / f"{cfg}_{model}{suffix}"
        if p.exists():
            with open(p) as f:
                return {str(r["id"]): r for r in json.load(f)}
    return None


def build_prompt(row: dict) -> str:
    src = row.get("prompt", "")
    if len(src) > MAX_PROMPT_CHARS:
        src = src[:MAX_PROMPT_CHARS // 2] + "\n...[truncated]...\n" + src[-MAX_PROMPT_CHARS // 2:]
    return (
        "A financial QA model answered the following question INCORRECTLY. "
        "Your job is to diagnose the PRIMARY cause of the error.\n\n"
        f"=== QUESTION AND CONTEXT (may be truncated) ===\n{src}\n\n"
        f"=== CORRECT ANSWER ===\n{row.get('gold_label')}\n\n"
        f"=== MODEL'S FULL RESPONSE ===\n{row.get('generated_text')}\n\n"
        f"=== MODEL'S EXTRACTED FINAL ANSWER ===\n{row.get('predicted_label')}\n\n"
        "Classify the primary error into exactly one category:\n"
        "- extraction: the model selected the wrong number(s) from the table/text\n"
        "- arithmetic: the model selected the right numbers but computed incorrectly\n"
        "- unit_scale: the computation is essentially right but the answer is off "
        "by units, scale, percent-vs-decimal, or sign convention\n"
        "- reasoning: the model misunderstood the question or used a wrong approach "
        "(wrong operation, wrong rows/years compared, answered a different question)\n"
        "- format: the model's final value is effectively correct but was expressed "
        "in a way that failed answer extraction\n"
        "- other: none of the above\n\n"
        "Reply with the category name on the first line, then one short sentence "
        "of justification."
    )


def ask_judge(client: OpenAI, row: dict, judge_model: str) -> tuple[str | None, str]:
    prompt = build_prompt(row)
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are a meticulous error analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                temperature=0,
                stream=False,
            )
            if response.choices and (response.choices[0].message.content or "").strip():
                break
        except Exception:
            pass
        time.sleep(10 * (attempt + 1))
    else:
        return None, "API_EMPTY"
    raw = response.choices[0].message.content.strip()
    m = _CAT_RE.search(raw.split("\n", 1)[0]) or _CAT_RE.search(raw)
    return (m.group(1).lower() if m else None), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama-3.1-8B-Instruct")
    ap.add_argument("--method", required=True, help="e.g. gptq_w4a16, awq_w4a16_asym")
    ap.add_argument("--task", required=True, choices=["t5", "t9"])
    ap.add_argument("--base", default="c1")
    ap.add_argument("--treat", default="c3")
    ap.add_argument("--judge", default=JUDGE_MODEL)
    ap.add_argument("--subset", choices=["fixed", "unfixed"], default="fixed",
                    help="fixed: base wrong → treat right (default); "
                         "unfixed: base wrong → treat still wrong (control group, "
                         "sampled with seed 42 when --num-samples is set)")
    ap.add_argument("--num-samples", type=int, default=None,
                    help="only attribute N samples (head for fixed, "
                         "seeded random sample for unfixed)")
    ap.add_argument("--max-workers", type=int, default=100)
    args = ap.parse_args()

    tnum = args.task[1:]
    rows_b = load_detailed(args.model, f"{args.method}_{args.base}", tnum)
    rows_t = load_detailed(args.model, f"{args.method}_{args.treat}", tnum)
    if not rows_b or not rows_t:
        raise SystemExit("Missing detailed results for base or treat config.")

    want_treat_correct = args.subset == "fixed"
    fixed_ids = [i for i, r in rows_b.items()
                 if not r["is_correct"]
                 and i in rows_t
                 and rows_t[i]["is_correct"] == want_treat_correct]
    if args.num_samples:
        if args.subset == "unfixed":
            import random
            random.seed(42)
            fixed_ids = sorted(random.sample(fixed_ids,
                                             min(args.num_samples, len(fixed_ids))))
        else:
            fixed_ids = fixed_ids[: args.num_samples]
    desc = "right" if want_treat_correct else "still wrong"
    print(f"{len(fixed_ids)} samples ({args.base} wrong → {args.treat} {desc}, "
          f"subset={args.subset}) to attribute.")

    client = make_client()
    results = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(ask_judge, client, rows_b[i], args.judge): i
                for i in fixed_ids}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            cat, raw = fut.result()
            results[i] = {"category": cat, "judge_reply": raw}
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(fixed_ids)}")

    counts = {c: 0 for c in CATEGORIES + ["unresolved"]}
    for r in results.values():
        counts[r["category"] or "unresolved"] += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "" if args.subset == "fixed" else f"_{args.subset}"
    out = OUT_DIR / (f"error_attribution_{args.method}_{args.base}to{args.treat}"
                     f"{tag}_{args.model}_{args.task}.json")
    with open(out, "w") as f:
        json.dump({
            "model": args.model, "method": args.method, "task": args.task,
            "base": args.base, "treat": args.treat, "judge": args.judge,
            "n_attributed": len(results), "category_counts": counts,
            "per_sample": results,
        }, f, indent=2)

    total = max(1, len(results))
    print("\nCategory distribution of errors fixed by "
          f"{args.treat} ({args.method}, {args.task}):")
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"  {c:<12} {n:>4}  ({n / total * 100:.1f}%)")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
