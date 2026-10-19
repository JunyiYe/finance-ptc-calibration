# Evaluation Datasets

Test sets for the ten financial tasks, exactly as evaluated in the paper (deduplicated where the public release contains duplicates). Each `eval_t*.py` script loads its file from this directory.

Four tasks have large public test pools (T3/T6/T7/T10). The paper evaluates 1500 samples of each (paper Table 2), selected deterministically inside the eval script: `random.seed(42); random.shuffle(dataset); dataset[:1500]`. The `run_*_all_tasks.sh` sweeps pass `--num-samples 1500` for exactly these tasks; the shipped files are the full pools, so the selection is reproducible. The unevaluated remainders of T6/T7/T10 are the task-matched calibration sources (see `scripts/evaluation/prepare_*_dedup.py`), which is why calibration data never overlaps the evaluated samples. All other tasks are evaluated on the full file.

| Task | File | File size | Evaluated n | Source benchmark |
|------|------|-----------|-------------|------------------|
| T1 Sentiment | `en_fpb_test.json` | 970 | 970 | Financial PhraseBank (FLARE split) |
| T2 FOMC | `finben_fomc_test.json` | 496 | 496 | FinBen / FinanceMTEB FOMC |
| T3 Headlines | `flare_headlines_test.json` | 20547 | 1500 | FLARE Headlines |
| T4 CFA | `flare_cfa_test.json` | 1032 | 1032 | FLARE CFA |
| T5 FinQA | `flare_finqa_test.json` | 1147 | 1147 | FLARE FinQA |
| T6 FinRED | `fingpt_finred_cls_test.json` | 8928 | 1500 | FinGPT FinRED (relation classification) |
| T7 FinCausal | `flare_causal20_sc_test.json` | 8628 | 1500 | FLARE FinCausal-20 SC |
| T8 ConvFinQA | `flare_convfinqa_test.json` | 1490 | 1490 | FLARE ConvFinQA |
| T9 BigData22 | `flare_sm_bigdata_test.json` | 1472 | 1472 | FLARE BigData22 (stock movement) |
| T10 TAT-QA | `flare_tatqa_test.json` | 1668 | 1500 | FLARE TAT-QA |

These files are redistributed from publicly available benchmarks (FLARE/PIXIU, FinBen, FinGPT) for exact reproducibility; the original datasets remain under their respective licenses — see the source benchmarks for terms. The raw training splits used to build the FinMix calibration corpus are not included (the built corpus ships in `data/calibration_datasets/`); `scripts/evaluation/download_train_splits.py` fetches all of them, and each `scripts/calibration/prepare_c3_t*.py` names its exact source at the top.
