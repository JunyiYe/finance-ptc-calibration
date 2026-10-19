#!/usr/bin/env python3
"""
T9 ConvFinQA - Adjudicate all predictions using DeepSeek as judge.

Exact-match is unreliable for FinQA because the model outputs values in
different but equivalent forms (e.g. gold=0.14464 vs model=14.46%,
gold=94.0 vs model=94 million). The judge re-reads the generated_text,
extracts the model's final numerical answer, and determines whether it is
numerically equivalent to the gold label.

Usage:
    conda run -n finquant-lmc-main python scripts/llm_judgement/adjudicate_t9_convfinqa.py \
      --input experiments/bf16_Llama-3.1-8B-Instruct_t9_convfinqa_detailed.json

Output files (auto-named from input):
    experiments/bf16_Llama-3.1-8B-Instruct_t9_convfinqa_detailed_llm_adjudicated.json
    experiments/bf16_Llama-3.1-8B-Instruct_t9_convfinqa_llm_adjudicated.json
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

_YES_NO_PATTERN = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def make_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def build_prompt(generated_text: str, gold_label: str) -> str:
    return (
        f"A financial reasoning model was asked a numerical question.\n"
        f"The correct answer is: {gold_label}\n\n"
        f"The model produced the following response:\n"
        f"\"{generated_text}\"\n\n"
        f"Does the model's final answer match the correct answer? "
        f"Consider these equivalent: a decimal and its percentage form "
        f"(e.g. 0.14464 = 14.464% ≈ 14.46%), different rounding of the same value "
        f"(e.g. 9.864% ≈ 9.9%), and equivalent magnitudes (e.g. 94.0 = 94). "
        f"A value expressed as positive with a directional word (e.g. 'decreased by 62.5%') "
        f"is equivalent to the corresponding negative decimal (e.g. -0.625). "
        f"Focus on the model's concluding sentence or the value it explicitly states as the answer, "
        f"not intermediate calculation steps. "
        f"If the model did not reach a clear numerical answer, reply: null. "
        f"Otherwise reply with exactly one word: yes or no."
    )


def ask_judge(client: OpenAI, generated_text: str, gold_label: str, judge_model: str) -> tuple[bool | None, str]:
    """
    Ask the judge whether the model's generated answer matches the gold label.

    Returns (verdict, raw_reply):
        verdict: True (correct), False (incorrect), or None (judge could not decide)
        raw_reply: the judge's raw output text
    """
    prompt = build_prompt(generated_text, gold_label)
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                temperature=0,
                stream=False,
            )
            if response.choices:
                break
        except Exception:
            pass
        wait = 10 * (attempt + 1)
        print(f"  [retry {attempt+1}/5] API returned empty, waiting {wait}s...")
        time.sleep(wait)
    else:
        raise RuntimeError("DeepSeek API returned empty response after 5 retries")
    raw_reply = response.choices[0].message.content.strip()
    reply = raw_reply.lower()

    if re.match(r"^\s*null\b", reply):
        return None, raw_reply

    m = _YES_NO_PATTERN.match(reply)
    if m:
        return m.group(1).lower() == "yes", raw_reply

    return None, raw_reply  # judge gave an unexpected response


def compute_metrics(detailed_results: list) -> dict:
    correct = 0
    total = 0
    null_count = 0

    for entry in detailed_results:
        resolved = entry.get("judge_resolved", False)
        is_correct = entry.get("is_correct", False)
        if not resolved:
            null_count += 1
        else:
            if is_correct:
                correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "num_samples": total,
        "accuracy": float(accuracy),
        "judge_null_count": null_count,
        "judge_null_rate": null_count / total if total > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="T9 ConvFinQA: adjudicate all predictions with DeepSeek judge")
    parser.add_argument("--input", required=True,
                        help="Path to *_t9_convfinqa_detailed.json file")
    parser.add_argument("--judge", default=JUDGE_MODEL,
                        help=f"DeepSeek model name (default: {JUDGE_MODEL})")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Only adjudicate the first N samples (for testing)")
    parser.add_argument("--max-workers", type=int, default=450,
                        help="Concurrent API threads (default: 450)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    stem = input_path.stem
    exp_name = stem.replace("_detailed", "")

    detailed_out = input_path.parent / f"{exp_name}_detailed_llm_adjudicated.json"
    metrics_out  = input_path.parent / f"{exp_name}_llm_adjudicated.json"

    print(f"Experiment: {exp_name}")
    print(f"Input:      {input_path}")
    print(f"Output:     {detailed_out}")
    print(f"Metrics:    {metrics_out}")
    print()

    with open(input_path, "r", encoding="utf-8") as f:
        detailed_results = json.load(f)

    if args.num_samples is not None:
        detailed_results = detailed_results[:args.num_samples]
        print(f"Smoke test mode: adjudicating first {args.num_samples} samples only.")

    print(f"Total entries: {len(detailed_results)}")
    print("Adjudicating all entries (exact-match is unreliable for numerical answers).")
    print()

    client = make_client()
    n = len(detailed_results)

    # Print sample prompt before launching concurrent jobs
    entry0 = detailed_results[0]
    print("\n" + "=" * 60)
    print("SAMPLE JUDGE PROMPT (entry 0):")
    print("=" * 60)
    print(build_prompt(entry0["generated_text"], entry0["gold_label"]))
    print("=" * 60 + "\n")

    # results_map[i] = (verdict, raw_reply)  — filled concurrently, read back in order
    results_map: dict[int, tuple] = {}
    done_count = 0

    def judge_one(i: int) -> tuple[int, bool | None, str]:
        entry = detailed_results[i]
        if entry.get("hit_max_tokens"):
            return i, False, "no (skipped: hit_max_tokens)"
        verdict, raw_reply = ask_judge(client, entry["generated_text"], entry["gold_label"], args.judge)
        return i, verdict, raw_reply

    skipped_hit_max = sum(1 for e in detailed_results if e.get("hit_max_tokens"))
    print(f"Launching {n - skipped_hit_max} concurrent judge calls "
          f"({skipped_hit_max} hit_max_tokens entries skipped, marked incorrect) "
          f"(max_workers={args.max_workers}) ...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(judge_one, i): i for i in range(n)}
        for future in as_completed(futures):
            i, verdict, raw_reply = future.result()
            results_map[i] = (verdict, raw_reply)
            done_count += 1
            if done_count % 100 == 0 or done_count == n:
                print(f"  Progress: {done_count}/{n}")

    # Write results back in original order — guaranteed no scrambling
    correct_count = 0
    null_count = 0
    for i, entry in enumerate(detailed_results):
        verdict, raw_reply = results_map[i]
        entry["is_correct"] = bool(verdict)     # None → False
        entry["judge"] = args.judge
        entry["judge_output"] = raw_reply
        entry["adjudicated_by_llm"] = not entry.get("hit_max_tokens", False)
        entry["judge_resolved"] = verdict is not None

        if verdict is None:
            null_count += 1
        elif verdict:
            correct_count += 1

        if i < 5 or verdict is None:
            print(f"[{i}] gold={entry['gold_label']}  pred={entry.get('predicted_label')}  "
                  f"judge={verdict}  (is_correct={entry['is_correct']})")

    print(f"\n✓ Adjudicated {n} entries")
    print(f"  - correct:      {correct_count}")
    print(f"  - incorrect:    {n - correct_count - null_count}")
    print(f"  - judge null:   {null_count}")

    with open(detailed_out, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {detailed_out}")

    new_metrics = compute_metrics(detailed_results)

    original_metrics_path = input_path.parent / f"{exp_name}.json"
    if original_metrics_path.exists():
        with open(original_metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    api_error_count = sum(1 for e in detailed_results if e.get("judge_output") == "API_ERROR")
    metrics.update(new_metrics)
    metrics["adjudicated_count"] = len(detailed_results)
    metrics["api_error_count"] = api_error_count
    metrics["judge_model"] = args.judge

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✓ Metrics saved: {metrics_out}")
    print("\n" + "=" * 60)
    print(f"Results for: {exp_name}")
    print(f"  Accuracy:    {new_metrics['accuracy']:.4f}")
    print(f"  Judge nulls: {null_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
