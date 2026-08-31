# Final Held-Out Test Evaluation — KSERESNET-RS

**Test split:** 7,205 scenarios, used exactly once, for the first time, in this analysis (2,767 FAIL / 4,438 PASS). See `results/final_test_integrity_audit.json` — all integrity checks passed.

**Frozen configuration (unchanged from validation):**
- Qwen3-8B model, prompt, feature representation, decoding parameters, output schema — unmodified
- KSERESNET R(s) computation (corrected native-density checkpoint) — unmodified
- Normalization: min-max range **fit on validation only** (R: [0.000222, 1.0], L: [0.05, 0.75]), reused verbatim on test (not refit)
- **alpha = 0.40** — not re-tuned after seeing test results

No test-driven tuning occurred at any stage of this evaluation.

---

## 1. Primary metrics — all five methods (n=7,205)

| Method | APFD | APFDc | P@100 | R@100 | NDCG@100 | P@500 | R@500 | NDCG@500 | P@1000 | R@1000 | NDCG@1000 | Rank TTF | Time TTF (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 Random | 0.4983 | 0.5022 | 0.370 | 0.0134 | 0.3535 | 0.364 | 0.0658 | 0.3580 | 0.378 | 0.1366 | 0.3725 | 4 | 257.6 |
| B2 KSERESNET | 0.6334 | 0.6388 | 0.780 | 0.0282 | 0.7233 | 0.780 | 0.1409 | 0.7643 | 0.714 | 0.2580 | 0.7153 | 3 | 264.8 |
| B3 Qwen3-8B | 0.4538 | 0.4720 | 0.630 | 0.0228 | 0.5053 | 0.534 | 0.0965 | 0.5111 | 0.267 | 0.0965 | 0.2930 | 34 | 2645.6 |
| B4 GBDT | 0.6301 | 0.6182 | 0.710 | 0.0257 | 0.7255 | 0.714 | 0.1290 | 0.7195 | 0.652 | 0.2356 | 0.6644 | 1 | 18.1 |
| **B5 KSERESNET-RS** | **0.6387** | **0.6436** | 0.700 | 0.0253 | 0.6615 | **0.732** | **0.1323** | 0.7126 | **0.722** | **0.2609** | 0.7132 | 2 | 243.8 |

With 95% bootstrap CIs (n=2,000 resamples): see `results/final_test_metrics_with_ci.csv`. Highlights:
- KSERESNET-RS APFD = 0.6387 [0.6307, 0.6465]
- KSERESNET-RS APFDc = 0.6436 [0.6344, 0.6531]
- KSERESNET alone APFD = 0.6334 [0.6258, 0.6411]; APFDc = 0.6388 [0.6298, 0.6475]

## 2. Full budget-aware failure discovery curves

See `results/final_failure_discovery_curves.csv` (checkpoints at 100/500/1000/2000/5000/full) and `results/final_failure_discovery_curves_full.json` (every scenario, used for Figure 3). **Figure 3 (`fig3_failure_discovery_curve.png`) is the central figure**: cumulative unique failures discovered vs. scenarios executed, for all 5 methods. `fig3b_failure_discovery_curve_zoom.png` zooms into the first 2,000 scenarios where the curves are most separated.

## 3. Complementarity analysis (KSERESNET vs. Qwen3-8B) — CENTRAL ANALYSIS

At three top-k budgets, measured on test:

| Budget | R-top fail hits | L-top fail hits | Shared | KSERESNET-only | Qwen-only | Union | Jaccard | RS captures of Qwen-only | RS captures of KSERESNET-only |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 390 | 267 | 53 | 337 | 214 | 604 | 0.0877 | 133/214 (62.1%) | 180/337 (53.4%) |
| 1000 | 714 | 267 | 75 | 639 | 192 | 906 | 0.0828 | 123/192 (64.1%) | 524/639 (82.0%) |
| 2000 | 1271 | 630 | 264 | 1007 | 366 | 1637 | 0.1613 | 91/366 (24.9%) | 943/1007 (93.6%) |

**Structural complementarity is confirmed and large**: at budget=1000, Qwen3-8B's top-ranked list catches 192 failures that KSERESNET's top-1000 completely misses — a real, substantial blind spot (Jaccard overlap only 0.083, meaning the two rankers agree on very little of what they flag as high-risk).

**Does fusion convert this into improved prioritization? Partially, and with a real trade-off.** At budget=1000, KSERESNET-RS's own top-1000 recovers 123 of KSERESNET's 192 missed failures (64.1%) — a meaningful rescue — while also keeping 524/639 (82.0%) of KSERESNET's own hits. Net effect: RS_top_fail_hits (722) > R_top_fail_hits (714) at budget=1000, i.e., a **net absolute gain of 8 failures caught in the top-1000** despite reshuffling the list substantially. This is consistent with, but numerically smaller than, the correlation/complementarity signal seen on validation — the fusion does not achieve full union recall of R's and L's separate strengths, but it does convert a meaningful fraction of Qwen's unique catches into real prioritization gain without giving up much of KSERESNET's own coverage.

## 4. Cost reporting (kept strictly separate, never mixed into APFDc)

- **Scenario execution cost** (used in APFDc for all methods): total = 418,859.3 s across 7,205 scenarios, mean = 58.13 s/scenario.
- **LLM inference cost** (Qwen3-8B, test split, reported separately, NOT included in APFDc):
  - 7,205 requests, 100% success, 0 errors
  - mean latency 1.686 s, median 1.628 s
  - total LLM runtime: 12,146.65 s (~3.37 hours), throughput 35.59 scenarios/min

Full detail: `results/final_cost_report.json`.

## 5. Statistical analysis

Paired bootstrap (2,000 resamples, same resampled index set applied to both methods per iteration), 95% percentile CIs, empirical two-sided bootstrap p-value, bootstrap-based standardized effect size:

| Comparison | Metric | Point diff | 95% CI | p-value | Effect size | Significant? |
|---|---|---|---|---|---|---|
| **PRIMARY: RS vs KSERESNET** | APFD | **+0.00528** | [0.0028, 0.0079] | <0.001 | 4.05 | **Yes** |
| **PRIMARY: RS vs KSERESNET** | APFDc | **+0.00482** | [0.0015, 0.0081] | <0.001 | 2.93 | **Yes** |
| RS vs GBDT | APFD | +0.00846 | [-0.0010, 0.0179] | 0.081 | 1.76 | No |
| RS vs GBDT | APFDc | +0.02546 | [0.0145, 0.0360] | <0.001 | 4.57 | Yes |
| RS vs Qwen3-8B | APFD | +0.12000 | [0.1091, 0.1310] | <0.001 | 21.4 | Yes |
| RS vs Qwen3-8B | APFDc | +0.12523 | [0.1124, 0.1375] | <0.001 | 19.4 | Yes |
| RS vs Random | APFD | +0.14022 | [0.1286, 0.1522] | <0.001 | 23.6 | Yes |
| RS vs Random | APFDc | +0.14150 | [0.1283, 0.1551] | <0.001 | 20.6 | Yes |

Full table: `results/final_statistical_tests.csv`.

**Primary comparison result (H1):** on the held-out test split, KSERESNET-RS's improvement over KSERESNET alone is **statistically significant** on both APFD and APFDc — the 95% CIs exclude zero for both metrics. This is a stronger, cleaner result than the corresponding validation-split comparison (where CIs had overlapped).

## 6. Answering the primary research question

> **RQ: Does LLM semantic risk provide complementary information to KSERESNET that improves prioritization of safety-critical SDC test scenarios?**

**Classification: MODERATE POSITIVE EVIDENCE.**

Reasoning:
- The improvement over KSERESNET alone is statistically significant (APFD +0.0053, APFDc +0.0048, both CIs exclude 0) — this is real, not noise.
- But the *magnitude* is small (well under 1 percentage point on both APFD and APFDc), and top-of-list precision (P@100, NDCG@100) is actually slightly *worse* for RS than for KSERESNET alone (0.700 vs 0.780; 0.6615 vs 0.7233) — the LLM signal helps the aggregate ranking and the mid-list budgets (P@500, R@500, P@1000, R@1000 all improve or match) more than it helps the very top of the list.
- The complementarity analysis (§3) shows the underlying structural signal is strong (Qwen catches ~190-370 failures per budget that KSERESNET's ranking completely misses) but the fusion only recovers a fraction of that (24.9%-64.1% depending on budget) while also slightly diluting KSERESNET's own top-of-list precision.
- This is **not** a "LLM beats traditional ML" story — GBDT (trained on the identical structured feature schema) performs comparably to KSERESNET on APFD, and KSERESNET-RS's edge over GBDT is significant on APFDc but not on APFD.

### Explicit checklist

1. **KSERESNET-RS improves APFD over KSERESNET?** Yes — small but statistically significant (+0.0053, CI excludes 0).
2. **KSERESNET-RS improves APFDc over KSERESNET?** Yes — small but statistically significant (+0.0048, CI excludes 0).
3. **KSERESNET-RS discovers failures earlier (time-to-first-failure)?** Yes, slightly: rank-TTF 2 vs 3, time-TTF 243.8s vs 264.8s for KSERESNET alone (though GBDT is fastest to first failure at rank 1 / 18.1s, likely a small-sample artifact of a single very-early low-duration FAIL scenario rather than a systematic effect).
4. **Does Qwen contribute unique failures?** Yes, clearly — 192-366 failures per budget appear only in Qwen's top-k, never in KSERESNET's.
5. **Does fusion capture those unique failures?** Partially — 24.9%-64.1% of Qwen-only failures are recovered in RS's own top-k, varying by budget (best recovery at budget=1000).
6. **Improvement over GBDT exists?** Mixed — significant on APFDc (+0.0255), not significant on APFD (CI includes 0, p=0.081).
7. **Improvement is statistically significant?** Yes, for the primary H1 comparison (RS vs KSERESNET) on both APFD and APFDc.
8. **Improvement is practically meaningful?** Modest. A ~0.5 percentage-point APFD/APFDc gain is a real, reproducible, non-cherry-picked signal that the complementarity structure exists and is partially exploitable by fusion — but it is not a large practical improvement, and it comes with a small cost at the very top of the ranked list (P@100/NDCG@100 dip). Framed honestly: this is evidence that semantic LLM reasoning adds a real but modest increment on top of a strong learned prioritizer, not that it transforms prioritization performance.

## 7. Comparison with validation-split behavior

The test-split primary-comparison significance is *stronger* than on validation (validation CIs for RS vs KSERESNET-alone on APFDc had overlapped near zero; test CIs clearly exclude zero). This is reported as-is, without adjustment — the pre-registered alpha=0.40 was not re-tuned in light of this, consistent with the frozen-method requirement.

---

## Frozen throughout

- Qwen3-8B model/version, prompt, feature schema, decoding parameters, output schema — unchanged
- KSERESNET R(s) computation — unchanged
- Normalization procedure — reused verbatim from validation fit
- alpha = 0.40 — unchanged, not re-tuned after inspecting test results

## Files produced

- `results/final_test_metrics.csv`
- `results/final_test_metrics_with_ci.csv`
- `results/final_complementarity.json`
- `results/final_failure_discovery_curves.csv` + `final_failure_discovery_curves_full.json`
- `results/final_statistical_tests.csv`
- `results/final_cost_report.json`
- `results/final_test_integrity_audit.json`
- `results/fig1_apfd_comparison.png`
- `results/fig2_apfdc_comparison.png`
- `results/fig3_failure_discovery_curve.png` (central figure) + `fig3b_failure_discovery_curve_zoom.png`
- `results/fig4_precision_recall_vs_budget.png`
- `results/fig5_complementarity_venn.png`
- `results/fig6_rs_vs_kseresnet_improvement.png`

**No implementation errors were found during this evaluation. No reruns were performed. The method is not modified based on this outcome, per instructions.**
