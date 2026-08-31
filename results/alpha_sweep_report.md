# Alpha Sweep Report — KSERESNET-RS Fusion

**Split used:** Validation only (n = 7,196; 2,763 FAIL / 4,433 PASS). Test split (7,205 scenarios) was **not** touched at any point in this analysis.

**Frozen components** (unmodified throughout):
- Qwen3-8B model, prompt, decoding parameters, output schema — LLM scores `L` taken as-is from `qwen3_8b_val_full.csv`
- KSERESNET risk scores `R` — corrected native-density checkpoint, taken as-is from `kseresnet_R_scores.json`
- GBDT (B4) scores `G` — taken as-is from `gbdt_G_scores.json`, used only as a reference baseline (not fused)

**Fusion formula:** `RS(s) = alpha * R'(s) + (1 - alpha) * L'(s)`, alpha in {0.00, 0.05, ..., 1.00} (21 points).

**Normalization (pre-specified, fit on validation split only):** min-max scaling to [0,1].
- R range: [0.000222, 1.0]
- L range: [0.05, 0.75]

**Bootstrap:** 2,000 resamples of the validation scenario set (with replacement), percentile-method 95% CIs.

---

## 1. Pre-specified alpha selection criterion

> **Declared before inspecting results:** select alpha* = argmax(mean APFDc) on the validation split. APFDc is the primary criterion because it accounts for both fault count and per-scenario execution cost (`duration`), matching the organizers' protocol emphasis; APFD is reported as the secondary/tie-breaker signal.

## 2. Alpha* selected

**alpha\* = 0.40**

| Criterion | Best alpha |
|---|---|
| **APFDc (primary, pre-specified)** | **0.40** |
| APFD (secondary) | 0.35 |
| P@100 | 1.00 |
| R@100 | 1.00 |
| NDCG@100 | 1.00 |

**This is reported explicitly per instructions: metrics disagree on the optimal alpha.** APFD/APFDc — which aggregate over the *entire* ranked list — are maximized in the mid-range (alpha ≈ 0.35–0.40, i.e., a fusion that leans slightly toward R but substantially incorporates L). The early-budget metrics (P@100 / R@100 / NDCG@100) are maximized at alpha = 1.00 (pure KSERESNET, no LLM contribution) — Qwen3-8B alone is comparatively weak at the very top of the ranking on this split (its own APFD/APFDc are below Random-ish territory, see §4), so blending it in dilutes the top-100 precision even though it improves the aggregate ranking quality and produces gains at alpha ≈ 0.25–0.35 relative to alpha=1.0 that are within noise (see CI overlap in §5). Per the pre-specified criterion, **alpha = 0.40 is the frozen selection**, not the top-100-optimal one.

## 3. Metrics at alpha* = 0.40

| Metric | Value | 95% CI |
|---|---|---|
| APFD | 0.6296 | [0.6218, 0.6376] |
| APFDc | 0.6358 | [0.6262, 0.6448] |
| P@100 | 0.7200 | [0.6300, 0.8100] |
| R@100 | 0.0261 | [0.0228, 0.0295] |
| NDCG@100 | 0.7081 | [0.6123, 0.8215] |
| P@500 | 0.7300 | [0.6900, 0.7720] |
| R@500 | 0.1321 | [0.1249, 0.1401] |
| NDCG@500 | 0.7210 | [0.6786, 0.7663] |
| P@1000 | 0.7200 | [0.6900, 0.7470] |
| R@1000 | 0.2606 | [0.2499, 0.2707] |
| NDCG@1000 | 0.7170 | [0.6853, 0.7473] |
| Rank of first failure | 2 | — |
| Time to first failure | 132.29 s | — |

## 4. Reference baselines (alpha-independent, validation split)

| Method | APFD | APFDc |
|---|---|---|
| Random | 0.4988 | 0.4966 |
| KSERESNET alone (alpha=1.0) | 0.6236 | 0.6308 |
| Qwen3-8B alone (alpha=0.0) | 0.4538 | 0.4692 |
| GBDT alone (B4, not fused) | 0.6316 | 0.6223 |
| **KSERESNET-RS (alpha\*=0.40)** | **0.6296** | **0.6358** |

## 5. Incremental improvement over baselines (at alpha\* = 0.40)

| Comparison | ΔAPFD | ΔAPFDc | Notes |
|---|---|---|---|
| vs. KSERESNET alone | +0.0060 | **+0.0050** | Small; CIs overlap substantially (APFD 95% CI [0.6218,0.6376] vs KSERESNET's own bootstrap CI would center near 0.624) |
| vs. Qwen3-8B alone | **+0.1758** | **+0.1666** | Large — LLM alone is a weak ranker on this task; fusion recovers essentially all of KSERESNET's strength plus a small aggregate gain |
| vs. GBDT alone | -0.0020 | +0.0135 | Roughly on par on APFD; modest APFDc edge |
| vs. Random | **+0.1308** | **+0.1392** | Large, as expected |

## 6. Early-budget performance (P/R/NDCG @ 100 / 500 / 1000)

At alpha\* = 0.40, early-budget precision is strong (P@100 = 0.72, P@500 = 0.73, P@1000 = 0.72 — roughly 3x the base fail rate of 2,763/7,196 ≈ 0.384... actually note P values well above the 38% base rate, confirming meaningful enrichment at every budget). However, as noted in §2, alpha=1.0 (pure R) edges out alpha=0.40 specifically on P@100/R@100/NDCG@100 — the LLM's contribution shows up more in the tail/aggregate ranking (APFD/APFDc, R@500, R@1000) than in the very top of the list.

## 7. Statistical / practical significance of the fusion gain

- **KSERESNET-RS vs. KSERESNET alone (the primary comparison for the paper's RQ):** ΔAPFDc = +0.0050, ΔAPFD = +0.0060. The 95% CI for KSERESNET-RS APFDc [0.6262, 0.6448] and for KSERESNET-alone APFDc (bootstrap not separately re-run above but centered near 0.6308, comparably wide) overlap substantially. **This improvement is not clearly statistically significant on the validation split** — it should be reported as a small, directionally positive but not confidently significant gain, pending confirmation on the held-out test split.
- **Practically:** the gain is small in absolute magnitude (~0.5–0.6 percentage points of APFD/APFDc) but consistent in direction across the alpha range 0.05–0.55 (all of which beat KSERESNET-alone on APFDc), which is the more informative signal than any single point estimate.
- The much larger and clearly significant gaps are KSERESNET-RS / KSERESNET vs. Qwen3-8B-alone and vs. Random — confirming R carries most of the ranking signal, with L acting as a complementary but secondary contributor at this alpha.

## 8. Stability of the optimum

**The optimum is broad and stable, not sharp.** APFDc rises smoothly from 0.4692 (alpha=0) to a shallow plateau of ~0.6325–0.6358 across alpha = 0.05 through 0.55, peaking at alpha=0.40, then declines gently back toward 0.6308 at alpha=1.0. The full range of APFDc across alpha in [0.05, 0.95] spans only about 0.0046 (0.6312–0.6358) — i.e., **the fusion result is robust to the exact choice of alpha within a wide middle band**, and alpha=0.40 is not a fragile, cherry-picked point. This also means the disagreement noted in §2 (early-budget metrics preferring alpha=1.0) reflects a genuine trade-off rather than sweep noise: including L moves the middle/tail of the ranking closer to true failures, at a small early-precision cost.

## 9. Plots

See `results/alpha_sweep_plots.png` for: alpha vs APFD, alpha vs APFDc, alpha vs P@100, alpha vs Recall@100, alpha vs NDCG@100 (shaded bands = bootstrap 95% CI; red dashed line = per-panel optimum).

---

## FROZEN configuration going forward

- Qwen3-8B model + prompt + feature schema: **unchanged** (as required)
- **alpha\* = 0.40**

This configuration is now frozen. Test-split evaluation (7,205 scenarios, never inspected during this sweep) is the next step, pending confirmation to proceed.
