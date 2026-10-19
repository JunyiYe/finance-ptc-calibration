# finance-ptc-calibration

Code release for the ICAIF 2026 paper:

> **Damage Predicts Recovery: When Calibration Data Matters in Compressing Financial LLMs**
> Junyi Ye, Mengjia Yu, Debapriya Hazra, Guiling Wang
> *7th ACM International Conference on AI in Finance (ICAIF '26), Milan, Italy.* [doi:10.1145/3858063.3858724](https://doi.org/10.1145/3858063.3858724)

**TL;DR:** Whether compression needs domain-matched calibration data depends on the *task-level damage* the compression causes, not on domain mismatch. If compression preserves the target capability, changing the calibration corpus has little effect. Where it causes large losses (numerical QA under pruning, up to 41 points), FinMix — a mixture of financial task examples — recovers roughly 20–60% of the damage, while another generic corpus does not. The damage–recovery link holds across model families and scales, supporting a practical rule: measure task-specific damage first, and construct specialized calibration data only when the damage is large.

## Setup

- **Models:** Llama-3.1-8B-Instruct (main), Qwen3-8B (robustness + prediction test), Llama-3.1-70B-Instruct (scale appendix)
- **Compression:** RTN / GPTQ / AWQ (W4A16), SmoothQuant (W8A8), SparseGPT 50% unstructured, Wanda 4:8 — all via [llm-compressor](https://github.com/vllm-project/llm-compressor)
- **Calibration corpora (`data/calibration_datasets/`):** C1 = WikiText-2, C2 = C4-web (~262k tokens each), C3 = **FinMix**, the financial 10-task mix (random 128-subset ≈16k tokens actually consumed by the sampler — see `data/calibration_datasets/README.md`), plus seed variants (s43/s44)
- **Tasks:** T1–T6, T8–T11 (sentiment, FOMC, headlines, CFA, FinQA, FinRED, FinCausal, ConvFinQA, BigData22, TAT-QA); full test sets
- **Scoring:** classification tasks via exact match / log-prob; generative numerical QA (FinQA, ConvFinQA, TAT-QA) via LLM-judge adjudication (DeepSeek, temperature 0), validated by a second independent LLM judge and a human audit of judge disagreements and concordant samples
- **Statistics:** per-sample exact McNemar tests; 3 calibration-sampling seeds for GPTQ

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=...   # only needed for the LLM-judge scripts
```

## Repository layout

```
data/calibration_datasets/   C1/C2/C3 calibration corpora (+ seed variants) and docs
data/evaluation_datasets/    test sets for the ten tasks, exactly as evaluated
scripts/quantization/        per-method compression (gptq/awq/rtn/smoothquant/sparsegpt/wanda)
                             + run_* pipelines and calibration-set preparation
scripts/calibration/         construction of the C3 financial task mix (per-task prep + merge)
scripts/evaluation/          eval_t*.py per-task evaluation + run_* pipelines
scripts/llm_judgement/       LLM-judge adjudication of FinQA/ConvFinQA/TAT-QA outputs
                             and error attribution
scripts/analysis/            significance tests, damage/recovery analysis, paper tables & figures
```

## Pipeline

1. **Prepare calibration sets** — `scripts/quantization/prepare_c1_wikitext.py`, `prepare_c2_c4.py`, `scripts/calibration/prepare_c3_*.py` + `merge_c3.py` (pre-built JSONL files are included in `data/calibration_datasets/`)
2. **Compress** — e.g. `scripts/quantization/gptq_w4a16.py --calib c1|c2|c3`, or the `run_*_all_calib.sh` pipelines
3. **Evaluate** — `scripts/evaluation/eval_t5_finqa.py` etc., or `run_llama8b_t*.sh` / `run_llama70b_*.sh` pipelines
4. **Adjudicate generative tasks** — `scripts/llm_judgement/adjudicate_t5_finqa.py --input <detailed.json>` (likewise t9/t11)
5. **Analyze** — `scripts/analysis/calibration_significance.py`, `damage_significance.py`, `generate_*_table_tex.py`, `plot_damage_recovery.py`; finish with `paper_consistency_check.py`

> Shell pipelines contain machine-specific paths (`/data/yej/...`, `CUDA_VISIBLE_DEVICES`) from the original experiment environment — adapt to your setup. The Python entry points are path-agnostic.

## Citation

```bibtex
@inproceedings{ye2026damage,
  title     = {Damage Predicts Recovery: When Calibration Data Matters in Compressing Financial LLMs},
  author    = {Ye, Junyi and Yu, Mengjia and Hazra, Debapriya and Wang, Guiling},
  booktitle = {Proceedings of the 7th ACM International Conference on AI in Finance (ICAIF '26)},
  year      = {2026},
  address   = {Milan, Italy},
  publisher = {ACM},
  doi       = {10.1145/3858063.3858724}
}
```

## Contact

Junyi Ye — yej@montclair.edu
