# Calibration Datasets

Calibration corpora used for post-training compression (weight calibration only — never for evaluation).

| Set | File | Samples | Tokens | Domain |
|-----|------|---------|--------|--------|
| C1 (WikiText) | `c1_wikitext.jsonl` | 128 | 128 × 2048 = 262,144 | Generic (WikiText-2 encyclopedia text) |
| C2 (C4) | `c2_c4web.jsonl` | 128 | 128 × 2048 = 262,144 | Generic (C4 Common Crawl web text) |
| C3 (FinMix) | `c3_finmix.jsonl` | 2042 | ~260k over 2042 short prompts | Financial 10-task mix (T1–T10) |

The corpora are budget-matched at the point where calibration reads them (LLaMA-3.1-8B tokenizer, sequences truncated to 2048 tokens): the two generic corpora each provide exactly 262,144 tokens in 128 long sequences, and FinMix provides ~260k tokens over 2042 short task prompts.

> **Full consumption, matched budgets, seed 42.** Every configuration consumes its corpus in full, so all calibration contrasts compare matched (~260k-token) budgets. **All main-paper experiments use the seed-42 corpora** (`c1_wikitext.jsonl`, `c2_c4web.jsonl`, `c3_finmix.jsonl` — each sampled/constructed with seed 42). The seed-variant files exist only for the sampling-robustness panel: `c1s43/c1s44_wikitext_seed*` re-draw the WikiText chunk sample, and `c3s43/c3s44_finmix_seed*` re-construct FinMix end to end, under seeds 43 and 44.

## Construction

**C1 — WikiText-2** ([prepare_c1_wikitext.py](../../scripts/quantization/prepare_c1_wikitext.py)): the standard domain-mismatched baseline of the GPTQ/AWQ papers. Non-empty train-split lines are greedily concatenated in order into chunks of ≥2048+64 tokens; 128 chunks sampled with seed 42. Verbatim text, no tokenize→decode round-trip.

**C2 — C4** ([prepare_c2_c4.py](../../scripts/quantization/prepare_c2_c4.py)): a second generic baseline. The English C4 train split is streamed, keeping only documents ≥2048 tokens until 2000 are collected; 128 sampled with seed 42. Full documents are stored; truncation is left to llm-compressor.

**C3 — FinMix** ([scripts/calibration/](../../scripts/calibration/), one prep script per task + `merge_c3.py`): a mixture over the ten evaluation tasks, with each task contributing an (approximately) equal share of the token budget — ≈26.2k tokens per task, controlled by token count rather than sample count. Per task, candidate texts are shuffled with seed 42 and accumulated until the budget is reached; `merge_c3.py` merges and reshuffles. Each sample is formatted exactly like the corresponding evaluation prompt (including answer-choice rotation), without chat-template wrapping.

### Where the calibration texts come from

**FinMix never contains evaluation data.** Per task, texts are taken from the
task's **training (and validation) splits**. Where those are missing or too
small, we fall back — in this order — to (a) a held-out portion of the task's
public release, **deduplicated against the evaluated test set**, or (b) a
separate dataset of the same task type with no overlap with the test set.
As a result, every calibration sample is disjoint from every evaluated
test set.

| Task | Calibration source | Samples | Tokens |
|------|--------------------|---------|--------|
| T1 Sentiment (FPB) | FPB train + validation splits | 442 | ~26.2k |
| T2 FOMC | FOMC train split + held-out portion, dedup vs. test | 219 | ~26.2k |
| T3 Headlines | Headlines train split | 532 | ~26.2k |
| T4 CFA | FinEval **CFA Challenge** (`fineval_cfa_challenge.json`, shipped here) — a separate CFA dataset, no overlap with the evaluated FLARE CFA set (CFA has no train split) | 270 | ~25.3k |
| T5 FinQA | FinQA train split | 28 | ~26.2k |
| T6 FinRED | FinRED held-out portion, dedup vs. the evaluated set | 236 | ~26.2k |
| T7 FinCausal | FinCausal held-out portion, dedup vs. the evaluated set | 221 | ~26.2k |
| T8 ConvFinQA | ConvFinQA train split | 23 | ~25.8k |
| T9 BigData22 | BigData22 train split | 29 | ~25.6k |
| T10 TAT-QA | TAT-QA public release minus the evaluated test set (dedup) | 42 | ~26.1k |
| **Total** | | **2042** | **~260k** |

The per-task sample counts differ because prompt lengths differ (FinQA and
ConvFinQA prompts carry long table contexts; FPB prompts are one sentence) —
the token shares are what is equalized. The exact source file for each task is
hard-coded at the top of its `prepare_c3_t*.py` script.

All files are JSONL with a single `{"text": "..."}` field per line.

`finppl_earnings_calls.jsonl` is the held-out financial text (earnings-call transcripts) used only for the FinPPL perplexity evaluation (`scripts/analysis/eval_perplexity.py`) — never for calibration or task evaluation.
