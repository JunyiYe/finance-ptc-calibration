# Evaluation Datasets

Test sets for the ten financial tasks, exactly as evaluated in the paper (full test sets; deduplicated where the public release contains duplicates). Each `eval_t*.py` script loads its file from this directory.

| Task | File | Source benchmark |
|------|------|------------------|
| T1 Sentiment | `en_fpb_test.json` | Financial PhraseBank (FLARE split) |
| T2 FOMC | `finben_fomc_test.json` | FinBen / FinanceMTEB FOMC |
| T3 Headlines | `flare_headlines_test.json` | FLARE Headlines |
| T4 CFA | `flare_cfa_test.json`, `fineval_cfa_challenge_test.json` | FLARE CFA + FinEval CFA Challenge |
| T5 FinQA | `flare_finqa_test.json` | FLARE FinQA |
| T6 FinRED | `fingpt_finred_cls_test.json` | FinGPT FinRED (relation classification) |
| T8 FinCausal | `flare_causal20_sc_test.json` | FLARE FinCausal-20 SC |
| T9 ConvFinQA | `flare_convfinqa_test.json` | FLARE ConvFinQA |
| T10 BigData22 | `flare_sm_bigdata_test.json` | FLARE BigData22 (stock movement) |
| T11 TAT-QA | `flare_tatqa_test.json` | FLARE TAT-QA |

These files are redistributed from publicly available benchmarks (FLARE/PIXIU, FinBen, FinGPT, FinEval) for exact reproducibility; the original datasets remain under their respective licenses — see the source benchmarks for terms. Training splits used to build the FinMix calibration corpus are not included; fetch them with `scripts/evaluation/download_train_splits.py`.
