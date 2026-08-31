# KSERESNET-RS / ARSF Research Artifact Manifest

Transferred from DGX (`~/VishResearch/KSERESNET_RS/`, host `172.20.27.27`, user `user2`) plus locally-generated artifacts from this session, on 2026-08-31. Full checksums in `transfer_manifest.csv` (211 files, 61.77 MB).

## Directory structure

```
KSERESNET_RS_JSS/
├── data/            split definitions, R/L/G score files, hidden representations, audits
├── models/          small checkpoints (KSERESNET original + native, GBDT)
├── configs/         frozen experiment configuration JSONs (LLM/vLLM settings per run)
├── results/         all final results, reports, figures, statistical tests
│   └── rankings/    per-scenario ranking CSVs for Qwen3-8B (val/test/ARSF-train-subset)
├── scripts/         complete source code for every stage of the pipeline
├── logs/            training/serving/resource logs (excluding one 56MB raw stdout dump)
└── transfer_manifest.csv, MANIFEST.md
```

## A. Final Results (`results/`)

**Fixed-fusion final test evaluation:** `final_test_metrics.csv`, `final_test_metrics_with_ci.csv`, `final_complementarity.json`, `final_failure_discovery_curves.csv` (+`_full.json`), `final_statistical_tests.csv`, `final_test_integrity_audit.json`, `final_cost_report.json`, `FINAL_TEST_REPORT.md`, `final_test_arrays.npz`, figures `fig1`–`fig6`.

**Alpha sweep (fixed fusion, validation-only):** `alpha_sweep.csv`, `alpha_sweep_report.md`, `alpha_sweep_summary.json`, `alpha_sweep_baselines.json`, `alpha_sweep_fdc.json`, `alpha_sweep_plots.png`.

**ARSF validation:** `arsf_validation_metrics.csv`, `arsf_ablation.csv`, `arsf_seed_results.csv` (+`_summary.json`), `arsf_gate_values.csv`, `arsf_gate_analysis.json`, `arsf_complementarity.json`, `arsf_training_configs.json`, `arsf_objective_comparison.json`, `arsf_model_complexity.json`, `arsf_vs_fixed_fusion_stats.json`, `arsf_train_subset_audit.json`, `arsf_report.md`, `arsf_val_arrays.npz`, figures `arsf_fig1`–`arsf_fig5`.

**ARSF final held-out test:** `arsf_final_test_metrics.csv` (+`_with_ci.csv`), `arsf_final_comparison.csv`, `arsf_final_complementarity.json`, `arsf_final_failure_discovery_curves.csv` (+`_full.json`), `arsf_final_statistical_tests.csv`, `arsf_final_test_integrity_audit.json`, `arsf_final_gate_stats_test.json`, `arsf_generalization_analysis.json`, `arsf_key_comparisons.json`, `arsf_final_test_arrays.npz`, `ARSF_FINAL_TEST_REPORT.md`, figures `arsf_final_fig1`–`arsf_final_fig7`.

**Stage-1 LLM candidate screening (1,000-scenario subset):** `stage1_ranking_quality.csv` (+`_with_codellama.csv`), `ranking_correlation.csv` (+`_with_codellama.csv`), `complementarity.json` (+`_with_codellama.json`), `STAGE1_RANKING_QUALITY_REPORT.md`.

**Resource/reliability profiling** for every validated LLM candidate (Qwen3-8B, Ministral-8B, DeepSeek-R1-Distill-Qwen-7B, CodeLlama-7B, Octocoder — full Stage-1 set plus 100-scenario validation runs where applicable): `resource_profile_<model>[.±_stage1].csv`, `resource_samples_<model>.csv`, `summary_<model>[_stage1].json`, `raw_<model>.json`.

**Environment/hardware snapshots:** `environment.txt`, `pip_freeze.txt`, `nvidia_smi_snapshot.txt`, `nvidia_smi_mig_snapshot.txt`, `timing_feature_and_rank.json`.

**Per-scenario ranking CSVs** (`results/rankings/`): 1,000-scenario Stage-1 rankings for every candidate model + Random + KSERESNET + GBDT; full-scale Qwen3-8B rankings for val/test/ARSF-train-subset.

## B. Data / Audit Artifacts (`data/`)

- `split_train_ids.json` (21,605), `split_val_ids.json` (7,196), `split_test_ids.json` (7,205) — the frozen train/val/test partition of the 36,006-scenario SensoData pool.
- `split_leakage_audit.json` — machine-verified confirmation of no cross-split overlap.
- `arsf_train_subset_ids.json` — the 4,000-scenario stratified train subset used for ARSF gate training, with `arsf_train_subset_audit.json` (in `results/`) confirming zero overlap with val/test/Stage-1.
- `stage1_1000_ids.json` (in this directory's parent scripts context) — the 1,000-scenario Stage-1 screening subset.
- `kseresnet_R_scores.json` — corrected native-density KSERESNET risk scores for all 36,006 scenarios (post point-density-bug-fix); `kseresnet_R_scores.stale_prefix_bug.json.bak` retained as evidence of the pre-fix state.
- `gbdt_G_scores.json` — GBDT (B4 baseline) scores, val+test.
- `features.json` — structured feature extraction output (the identical schema given to both GBDT and the LLM prompt).
- `kseresnet_hidden_val.json`, `kseresnet_hidden_test.json`, `kseresnet_hidden_train_subset.json` — the frozen 64-dim KSERESNET penultimate-layer representation h(s) extracted via read-only forward pass, used as ARSF gate input.
- `deepseek_r1_7b_pipeline_validation.json`, `ministral_8b_pipeline_validation.json`, `qwen3_8b_pipeline_validation.json` — early 100-scenario pipeline validation runs for each LLM candidate.
- `sdc-test-data.json` — sample/reference test-data structure.

**NOT included (excluded by design, documented here):**
- `sensodat_full_pool.json` (1.79 GB) — the full 36,006-scenario road-point pool. Not duplicated here; remains at the original working location. Regeneratable from the SensoData Zenodo archive via `scripts/parse_sensodat_xodr.py` (or `compute_features.py` for the feature-only view already included as `features.json`).
- `data/Sensodata/` (23 GB, DGX-only) — raw `.xodr` scenario files from the SensoData corpus.
- `data/zenodo_verify/` (4.8 GB, DGX-only) — byte-identity verification artifacts against the Zenodo-archived dataset.

## C. Configuration Files (`configs/`)

All 11 frozen per-run experiment configs: `qwen3_8b_config.json`, `qwen3_8b_stage1_config.json`, `qwen3_8b_val_full_config.json`, `qwen3_8b_test_full_config.json`, `qwen3_8b_arsf_train_config.json`, `ministral_8b_config.json` (+`_stage1_config.json`), `deepseek_r1_7b_config.json` (+`_stage1_config.json`), `codellama_stage1_config.json`, `octocoder_stage1_config.json`. Each records model name, dtype, max_model_len, gpu_memory_utilization, temperature, seed, max_output_tokens, timeout, vLLM/CUDA/driver versions, and run notes — sufficient to reproduce every LLM inference run exactly.

## D. Source Code (`scripts/`)

32 files covering the complete pipeline:

| Stage | Script(s) |
|---|---|
| Dataset parsing / feature extraction | `parse_sensodat_xodr.py`, `compute_features.py`, `build_splits.py`, `build_stage1_pool.py` |
| KSERESNET scoring | `score_kseresnet_batch.py`, `container_main_official.py`, `kseresnet_native_server.py`, `train_kseresnet_native.py`, `extract_kseresnet_hidden.py`, `rebuild_kseresnet_scores.py` |
| GBDT baseline | `train_gbdt.py` |
| LLM scoring (Qwen3-8B/Ministral/DeepSeek/CodeLlama) | `score_llm_stage1.py`, `llm_prompt_template.py`, `profile_llm_validation.py`, `validate_llm_pipeline.py`, `test_competitor_tools.py` |
| Ranking metrics (APFD/APFDc/P@K/R@K/NDCG@K, correlation, complementarity) | `compute_ranking_metrics.py`, `correlation_analysis.py` |
| Official-protocol replication | `replicate_official_protocol.py`, `replicate_official_protocol_fixed.py` |
| Fixed-alpha fusion sweep | `alpha_sweep.py` |
| ARSF training/evaluation | `arsf_train.py`, `arsf_ablation.py`, `arsf_hp_sweep.py`, `arsf_final_test_evaluation.py`, `arsf_final_test_integrity_audit.py` |
| Final held-out test evaluation (fixed fusion) | `final_test_evaluation.py`, `final_test_integrity_audit.py` |
| Protobuf definitions | `competition_2026_pb2.py`, `competition_2026_pb2_grpc.py` |
| Resource monitoring | `monitor_resources.sh` |
| Feature schema documentation | `FEATURE_SCHEMA.md` |

## E. Checkpoints / Models (`models/`)

- `kseresnet_original/crash_model.pth` (561 KB) — the ORIGINAL frozen KSERESNET checkpoint used for all R(s) scoring and h(s) extraction throughout this study (matches organizers' Table III baseline when combined with the native-density fix: APFD≈0.6255, APFDc≈0.6492 on the full-protocol sweep).
- `kseresnet_native/kseresnet_native_best.pt` (563 KB) — the retrained-on-full-pool KSERESNET variant (internal diagnostic only, NOT used in the paper's reported pipeline per prior explicit instruction; kept for completeness/audit trail).
- `b4_gbdt/b4_gbdt.json` (1.0 MB) — the trained XGBoost GBDT baseline model.
- `results/arsf_gate_seed42.pt` (11 KB) — the FROZEN ARSF gate checkpoint used for the final held-out test evaluation (seed=42, hidden=32, lr=0.003, BCE objective, 2,177 parameters).

**NOT included (documented instead, per instructions):**

| Model | HF identifier | Notes |
|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B` | Frozen model used for all val/test/ARSF-train scoring. Served via vLLM 0.9.1, dtype=bfloat16, max_model_len=8192, gpu_memory_utilization=0.85, temperature=0.1, seed=42, max_tokens=300, `enable_thinking=false`. CUDA 12.4, driver 550.144.03. |
| Ministral-8B | (see `configs/ministral_8b_config.json`) | Stage-1 candidate, not selected for full-scale scoring. |
| DeepSeek-R1-Distill-Qwen-7B | (see `configs/deepseek_r1_7b_config.json`) | Stage-1 candidate; rejected for full-scale use due to ~51hr projected runtime (chain-of-thought reasoning, ~588 avg output tokens). |
| CodeLlama-7b-instruct | (see `configs/codellama_stage1_config.json`) | Stage-1 candidate; fastest but weakest ranking quality (APFD 0.5147). |
| Octocoder/StarCoder | (see `configs/octocoder_stage1_config.json`) | Abandoned — chat-template incompatibility with vLLM's OpenAI-compatible endpoint (transformers v4.44+ restriction on base checkpoints without a defined `chat_template`). |

Exact configs, inference settings, and per-run resource/reliability profiles for all five candidates are preserved in `configs/` and `results/`, sufficient to reproduce the model-selection decision without re-downloading any weights.

## F. Excluded (documented, not transferred)

| Item | Size | Reason | Regeneration path |
|---|---|---|---|
| `envs/kseresnet_rs/` (Python venv) | 11 GB | Fully reproducible from `results/pip_freeze.txt` + `results/environment.txt` | `pip install -r <(tail -n +3 pip_freeze.txt)` in a fresh venv |
| `models/qwen3-8b/`, `models/deepseek-r1-distill-qwen-7b/`, `models/ministral-8b/` (raw weights) | 16GB+15GB+15GB | Publicly available on HuggingFace, re-downloadable via vLLM/HF hub using the identifiers above | `vllm serve Qwen/Qwen3-8B ...` (see `configs/`) |
| `data/Sensodata/` (raw .xodr files) | 23 GB | Original SensoData corpus, available from its Zenodo archive | See `scripts/parse_sensodat_xodr.py` |
| `data/zenodo_verify/` | 4.8 GB | One-time byte-identity verification artifact, not needed for ongoing reproducibility | N/A (verification already performed and documented) |
| `sensodat_full_pool.json` | 1.79 GB | Large derived pool file; the essential outputs derived from it (R/L/G scores, features, hidden reps, splits) are all included | `scripts/compute_features.py` + `scripts/parse_sensodat_xodr.py` against the raw corpus |
| `logs/vllm_arsf_train_subset.log` | 56 MB | Raw vLLM engine stdout; informational only, superseded by the structured `resource_profile_qwen3_8b_arsf_train.csv` and `summary_qwen3_8b_arsf_train.json` | N/A |
| `scripts/__pycache__/`, `models/*/__pycache__/` | small | Compiled bytecode, not source | N/A |

## Key numbers (for cross-reference against `transfer_manifest.csv` and the reports themselves — not restated as new claims here)

See `results/FINAL_TEST_REPORT.md` and `results/ARSF_FINAL_TEST_REPORT.md` for the authoritative, complete final numbers. Do not treat any number outside those two files (and their supporting CSV/JSON) as authoritative for the manuscript.
