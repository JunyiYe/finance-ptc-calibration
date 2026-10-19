# Calibration Datasets

Calibration corpora used for post-training compression (weight calibration only — never for evaluation).

| Set | File | Samples | Token budget | Domain |
|-----|------|---------|--------------|--------|
| C1 (WikiText) | `c1_wikitext.jsonl` | 128 | 128 × 2048 | Generic (WikiText-2 encyclopedia text) |
| C2 (C4) | `c2_c4web.jsonl` | 128 | 128 × 2048 | Generic (C4 Common Crawl web text) |
| C3 (FinMix) | `c3_finmix.jsonl` | 2042 | 128 × 2048 | Financial 10-task mix (T1–T6, T8–T11) |

All three files are built to the same file-level token budget (≈128 × 2048 = 262k tokens, LLaMA-3.1-8B tokenizer); compression scripts use `max_seq_len=2048`.

> **Effective token usage.** llm-compressor draws `num_calibration_samples=128` with a shuffling sampler. C1/C2 samples are each ≥2048 tokens, so ≈262k tokens participate in calibration. C3/FinMix consists of 2042 short task-formatted samples (mean ≈127 tokens), so only a random 128-subset ≈16k tokens is actually consumed — a ~16× smaller effective budget, as reported in the paper. Seed-variant files (`c1s43/c1s44_wikitext_seed*`, `c3s43/c3s44_finmix_seed*`) reproduce this sampling with seeds 43/44.

## Construction

**C1 — WikiText-2** ([prepare_c1_wikitext.py](../../scripts/quantization/prepare_c1_wikitext.py)): the standard domain-mismatched baseline of the GPTQ/AWQ papers. Non-empty train-split lines are greedily concatenated in order into chunks of ≥2048+64 tokens; 128 chunks sampled with seed 42. Verbatim text, no tokenize→decode round-trip.

**C2 — C4** ([prepare_c2_c4.py](../../scripts/quantization/prepare_c2_c4.py)): a second generic baseline. The English C4 train split is streamed, keeping only documents ≥2048 tokens until 2000 are collected; 128 sampled with seed 42. Full documents are stored; truncation is left to llm-compressor.

**C3 — FinMix** ([scripts/calibration/](../../scripts/calibration/), one prep script per task + `merge_c3.py`): a mixture over ten financial tasks, with each task's contribution controlled by token count (≈26.2k tokens per task) rather than sample count. Per task, candidate texts are shuffled with seed 42 and accumulated until the token budget is reached; `merge_c3.py` merges and reshuffles. Each sample is formatted exactly like the corresponding evaluation prompt (including answer-choice rotation), without chat-template wrapping.

| Task | Samples | Tokens |
|------|---------|--------|
| T1 Sentiment (FPB) | 442 | ~26.2k |
| T2 FOMC | 219 | ~26.2k |
| T3 Headlines | 532 | ~26.2k |
| T4 CFA | 270 | ~25.3k |
| T5 FinQA | 28 | ~26.2k |
| T6 FinRED | 236 | ~26.2k |
| T8 FinCausal | 221 | ~26.2k |
| T9 ConvFinQA | 23 | ~25.8k |
| T10 BigData22 | 29 | ~25.6k |
| T11 TAT-QA | 42 | ~26.1k |
| **Total** | **2042** | **~260k** |

Calibration texts are drawn from each task's train split (or a held-out subset deduplicated against the test set) — never from evaluation data.

All files are JSONL with a single `{"text": "..."}` field per line.

`finppl_earnings_calls.jsonl` is the held-out financial text (earnings-call transcripts) used only for the FinPPL perplexity evaluation (`scripts/analysis/eval_perplexity.py`) — never for calibration or task evaluation.
