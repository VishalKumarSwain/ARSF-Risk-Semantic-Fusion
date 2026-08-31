# ARSF — Risk-Semantic Fusion for KSERESNET-Based SDC Test Prioritization

Implementation and results for **Adaptive Risk-Semantic Fusion (ARSF)**: combining a frozen learned risk predictor (KSERESNET) with a frozen offline LLM (Qwen3-8B) to prioritize self-driving-car (SDC) test scenarios, evaluated on the SensoData/SensoDat benchmark (36,006 scenarios).

This repository contains the **implementation and results only**. The manuscript (LaTeX source, figures, tables) and third-party reference literature are kept separately.

## Repository layout

```
.
├── scripts/    complete source code for every pipeline stage (see below)
├── configs/    frozen per-run experiment configs (model, dtype, seed, vLLM settings, ...)
├── data/       split definitions, R/L/G risk scores, hidden representations, leakage audits
├── models/     small checkpoints: KSERESNET (original + native-diagnostic), GBDT baseline
├── results/    all final metrics, statistical tests, reports, figures, per-scenario rankings
├── logs/       training/serving/resource logs
├── MANIFEST.md            full artifact inventory with sizes and provenance
└── transfer_manifest.csv  per-file checksums (211 files)
```

## What's frozen vs. reproducible here

- **Frozen, not re-run**: KSERESNET weights, Qwen3-8B scoring, the fixed-fusion α sweep, and the ARSF gate's held-out test evaluation. Every number in `results/FINAL_TEST_REPORT.md` and `results/ARSF_FINAL_TEST_REPORT.md` is the authoritative, one-time result for this study.
- **Reproducible**: all scripts in `scripts/` can be re-run against the included data/config artifacts, or from scratch against the public SensoData corpus (see `scripts/parse_sensodat_xodr.py`).

## Not included (see `MANIFEST.md` §F for full list + regeneration path)

| Item | Why excluded | How to get it |
|---|---|---|
| Qwen3-8B / Ministral-8B / DeepSeek-R1-Distill-Qwen-7B raw weights | 15–16 GB each, public | `vllm serve <HF id>` — see `configs/*_config.json` for exact settings |
| `data/Sensodata/` raw `.xodr` corpus (23 GB) | Large, publicly archived | SensoData/SensoDat Zenodo archive, then `scripts/parse_sensodat_xodr.py` |
| `sensodat_full_pool.json` (1.79 GB) | Large derived file; its outputs (scores/features/splits) are already included | `scripts/compute_features.py` |
| Python virtualenv (11 GB) | Fully reproducible | `pip install -r requirements.txt` (see below) |
| No Docker image exists for this project | — | run scripts directly per the instructions below |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`requirements.txt` is derived from `results/pip_freeze.txt` (the exact environment used for all reported results). Key dependencies: PyTorch, vLLM 0.9.1, XGBoost, NumPy/Pandas/SciPy, scikit-learn.

LLM scoring requires a local GPU (this study used a single NVIDIA A100 MIG partition) and vLLM serving the model per the relevant `configs/*.json`. KSERESNET/GBDT scoring and all downstream metric/fusion computation run on CPU.

## Running the pipeline

The stages below mirror `scripts/` and can be run independently against the included `data/`/`results/` artifacts, or end-to-end from raw scenario data.

1. **Dataset prep**: `parse_sensodat_xodr.py` → `compute_features.py` → `build_splits.py` (produces the frozen train/val/test partition; already in `data/split_*_ids.json`) → `build_stage1_pool.py`
2. **KSERESNET scoring**: `score_kseresnet_batch.py` (uses `models/kseresnet_original/crash_model.pth`); `extract_kseresnet_hidden.py` for the 64-d hidden representation h(s) used by ARSF
3. **GBDT baseline**: `train_gbdt.py`
4. **LLM candidate screening**: `score_llm_stage1.py` + `llm_prompt_template.py`, evaluated via `compute_ranking_metrics.py` / `correlation_analysis.py`; full-scale scoring via `validate_llm_pipeline.py` / `profile_llm_validation.py` using the frozen `configs/qwen3_8b_*_config.json`
5. **Fixed fusion**: `alpha_sweep.py` (validation-only α selection)
6. **ARSF (adaptive gate)**: `arsf_train.py` → `arsf_hp_sweep.py` / `arsf_ablation.py` (validation-only) → `arsf_final_test_evaluation.py` (one-time held-out test, integrity-checked by `arsf_final_test_integrity_audit.py`)
7. **Final fixed-fusion evaluation**: `final_test_evaluation.py`, integrity-checked by `final_test_integrity_audit.py`

Each script reads/writes to `data/` and `results/` using the paths already populated in this repo, so any stage can be re-run in isolation without redoing earlier stages.

## Key results

See `results/FINAL_TEST_REPORT.md` and `results/ARSF_FINAL_TEST_REPORT.md` for the complete, authoritative held-out test numbers (APFD, APFDc, statistical comparisons, complementarity analysis). `MANIFEST.md` indexes every file in `results/` by category.

## Provenance

Artifacts transferred from a DGX training environment (`~/VishResearch/KSERESNET_RS/`) plus locally-generated results, 2026-08-31. Full per-file checksums in `transfer_manifest.csv`.
