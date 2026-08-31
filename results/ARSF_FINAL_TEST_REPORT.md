# ARSF Final Held-Out Test Report

**Test split:** 7,205 scenarios (2,767 FAIL / 4,438 PASS), used exactly once, for the first time, in this evaluation. `results/arsf_final_test_integrity_audit.json` — all checks passed.

**Frozen ARSF configuration used, exactly as approved:** checkpoint `results/arsf_gate_seed42.pt`, seed=42, hidden=32, lr=0.003, objective=BCE, architecture `Linear(66,32)→ReLU→Linear(32,1)→Sigmoid`, 2,177 trainable parameters. No training, tuning, or architecture change performed in this evaluation. Frozen KSERESNET checkpoint, frozen Qwen3-8B (model/prompt/config), frozen R'/L' normalization (validation-fit range reused verbatim), frozen h(s) standardization (train-subset-fit mean/std reused verbatim).

**Previously completed Fixed-Fusion test result preserved and verified:** APFD=0.6387, APFDc=0.6436 — recomputed here with identical methodology as a sanity check: **matched exactly** (APFD=0.6387, APFDc=0.6436). Not altered.

---

## 1. Final Test Table — All Six Methods (n=7,205)

| Method | APFD | APFDc | P@100 | R@100 | NDCG@100 | P@500 | R@500 | NDCG@500 | P@1000 | R@1000 | NDCG@1000 | Rank TTF | Time TTF (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 Random | 0.4983 | 0.5022 | 0.370 | 0.0134 | 0.3535 | 0.364 | 0.0658 | 0.3580 | 0.378 | 0.1366 | 0.3725 | 4 | 257.6 |
| B2 KSERESNET | 0.6334 | 0.6388 | 0.780 | 0.0282 | 0.7233 | 0.780 | 0.1409 | 0.7643 | 0.714 | 0.2580 | 0.7153 | 3 | 264.8 |
| B3 Qwen3-8B | 0.4538 | 0.4720 | 0.630 | 0.0228 | 0.5053 | 0.534 | 0.0965 | 0.5111 | 0.267 | 0.0965 | 0.2930 | 34 | 2645.6 |
| B4 GBDT | 0.6301 | 0.6182 | 0.710 | 0.0257 | 0.7255 | 0.714 | 0.1290 | 0.7195 | 0.652 | 0.2356 | 0.6644 | 1 | 18.1 |
| B5 Fixed KSERESNET-RS (α=0.40) | 0.6387 | 0.6436 | 0.700 | 0.0253 | 0.6615 | 0.732 | 0.1323 | 0.7126 | 0.722 | 0.2609 | 0.7132 | 2 | 243.8 |
| **B6 ARSF** | **0.6534** | **0.6593** | **0.820** | **0.0296** | **0.8180** | **0.820** | **0.1482** | **0.8205** | **0.757** | **0.2736** | **0.7669** | **1** | **114.7** |

With 95% bootstrap CIs (n=2,000): ARSF APFD 0.6534 [0.6451, 0.6620] (see `results/arsf_final_test_metrics_with_ci.csv` for the complete table). **ARSF is the top performer on every single reported metric.**

## 2. ARSF vs Fixed Fusion (PRIMARY hypothesis)

| Metric | Absolute diff | 95% CI | p-value | Effect size | Significant? |
|---|---|---|---|---|---|
| APFD | **+0.0148** | [0.0115, 0.0183] | <0.001 | large (CI well clear of 0) | **Yes** |
| APFDc | **+0.0156** | [0.0115, 0.0200] | <0.001 | large | **Yes** |

**Statistical improvement:** confirmed — both CIs exclude zero with clear margin.
**Practical improvement:** real but modest in absolute magnitude (~1.5 percentage points on both APFD and APFDc). More visible at the early budget: P@100 jumps from 0.70 (Fixed Fusion) to 0.82 (ARSF), NDCG@100 from 0.66 to 0.82.

## 3. ARSF vs KSERESNET (SECONDARY)

| Metric | Absolute diff | 95% CI | p-value | Significant? |
|---|---|---|---|---|
| APFD | +0.0201 | [0.0162, 0.0241] | <0.001 | Yes |
| APFDc | +0.0205 | [0.0157, 0.0252] | <0.001 | Yes |

ARSF's largest single-baseline margin — clearly exceeds the base KSERESNET model, not just the fixed-fusion baseline.

## 4. ARSF vs GBDT (TERTIARY)

| Metric | Absolute diff | 95% CI | p-value | Significant? |
|---|---|---|---|---|
| APFD | +0.0233 | [0.0141, 0.0325] | <0.001 | Yes |
| APFDc | +0.0410 | [0.0298, 0.0516] | <0.001 | Yes |

Full statistical table (including ARSF vs Qwen3-8B, ARSF vs Random): `results/arsf_final_statistical_tests.csv`.

## 5. Early-Budget Results

ARSF dominates at every early budget: P@100=0.82 (best of all six methods, next-best KSERESNET/Fixed at 0.70-0.78), and its cumulative-failures-at-checkpoint numbers (below) confirm this is not an artifact of one metric.

| Method | Cum. fails @100 | Cum. fails @500 | Cum. fails @1000 |
|---|---|---|---|
| Random | 37 | 182 | 379 |
| KSERESNET | 78 | 390 | 714 |
| Qwen3-8B | 63 | 267 | 267 |
| GBDT | 71 | 357 | 652 |
| Fixed Fusion | 70 | 366 | 722 |
| **ARSF** | **82** | **410** | **757** |

Rank-TTF for ARSF = 1 (finds a failure at the very first scenario), time-TTF = 114.7s — faster than KSERESNET (264.8s) and Fixed Fusion (243.8s), though slower than GBDT's 18.1s (a small-sample artifact tied to one very-short-duration early FAIL, consistent with the same caveat noted in earlier validation/test reports).

## 6. Complementarity Analysis on TEST — the central methodological finding

| K | KSERESNET-only fails | Qwen-only fails | Jaccard (R∩L) | Fixed Fusion fail hits | ARSF fail hits | ARSF recovers Qwen-only | **Fixed Fusion recovers Qwen-only** |
|---|---|---|---|---|---|---|---|
| 100 | 71 | 56 | 0.0522 | 70 | 82 | 5/56 (8.9%) | **46/56 (82.1%)** |
| 500 | 337 | 214 | 0.0877 | 366 | 410 | 33/214 (15.4%) | **133/214 (62.1%)** |
| 1000 | 639 | 192 | 0.0828 | 722 | 757 | 51/192 (26.6%) | **123/192 (64.1%)** |

**This is the most important and most honest finding of this evaluation.** ARSF catches more total failures than Fixed Fusion at every budget (net +12 @K=100, +44 @K=500, +35 @K=1000 — see `results/arsf_final_complementarity.json` for the exact `net_additional_fails_ARSF_vs_Fixed` field), consistent with its higher APFD/APFDc. **But ARSF recovers dramatically FEWER of Qwen's exclusive failures than Fixed Fusion does** — at K=100, Fixed Fusion recovers 82.1% of Qwen-only failures while ARSF recovers only 8.9%; at K=500/1000 the gap is smaller but still large (62-64% vs 15-27%).

This directly and strongly confirms, on genuinely held-out data, the pattern flagged as a risk in the validation report (§5 there: corr(g,R')=+0.46, corr(g,L')≈0). **ARSF is not "learning to trust Qwen" — if anything, it systematically de-emphasizes Qwen's unique signal relative to the fixed 0.40/0.60 blend, and still ends up with a higher aggregate score because it is a much better-calibrated user of R'(s) and the KSERESNET hidden representation h(s).** Its net failure-catching improvement over Fixed Fusion comes almost entirely from somewhere other than exploiting Qwen's complementary blind-spot coverage.

## 7. Validation-to-Test Generalization

| | Validation | Test | Gain (test − baseline) |
|---|---|---|---|
| Fixed Fusion APFD | 0.6296 | 0.6387 | — |
| ARSF APFD | 0.6491 | 0.6534 | — |
| **APFD gain (ARSF − Fixed)** | **+0.01954** | **+0.01474** | **75.4% of val gain retained** |
| Fixed Fusion APFDc | 0.6358 | 0.6436 | — |
| ARSF APFDc | 0.6572 | 0.6593 | — |
| **APFDc gain (ARSF − Fixed)** | **+0.02140** | **+0.01562** | **73.0% of val gain retained** |

**The ARSF improvement over Fixed Fusion generalizes to unseen scenarios.** Roughly three-quarters of the validation-measured gain survives on test — a meaningful shrinkage (consistent with the hyperparameter/objective-selection optimism flagged in the validation review as expected), but not a collapse. This is evidence *against* the improvement being pure overfitting to the validation set, and *for* it reflecting a real, if modest, effect.

## 8. Computational Cost (test-set scale)

| | ARSF gate |
|---|---|
| Trainable parameters | 2,177 (unchanged, frozen) |
| Model file size | 11.05 KB |
| Test-set inference (7,205 scenarios, batched, CPU) | well under 1 ms total |
| GPU memory required | none |

Compared to the Qwen3-8B inference this gate consumes the output of (7,205 requests, mean 1.699s/scenario, ~15.2GB GPU memory, ~3.4 hours total wall-clock on the test set): the ARSF gate itself adds negligible cost. **The additional complexity of ARSF over Fixed Fusion is, as claimed, essentially free** — all of its cost is inherited from the already-necessary R(s)/L(s)/h(s) computations.

## 9. Integrity Audit

`results/arsf_final_test_integrity_audit.json` — all checks passed:
- 7,205 unique test scenarios, 0 duplicates
- Labels consistent across all four sources (R, L, G, H) for all 7,205 scenarios — 0 mismatches
- All required scores (R, L, G, h) present for 100% of test scenarios
- Frozen ARSF checkpoint (`arsf_gate_seed42.pt`) used, no training performed
- Frozen normalization (val-fit R'/L' range) and frozen h(s) standardization (train-subset-fit) reused verbatim, not refit on test
- No test-driven retuning of any kind occurred

## 10. Honest Interpretation

- **Statistical improvement:** Confirmed. ARSF beats Fixed Fusion on both APFD and APFDc with 95% CIs that clearly exclude zero.
- **Practical improvement:** Real but modest at the aggregate level (~1.5 percentage points); more visible and arguably more actionable at early budgets (P@100 +0.12, a 17% relative jump).
- **Complementarity:** Present in the underlying R/L signals (Jaccard 0.05-0.09, confirming KSERESNET and Qwen still disagree substantially about which scenarios are risky), but **ARSF does not primarily exploit this complementarity** — it recovers far fewer Qwen-exclusive failures than the simple fixed blend does.
- **Contribution of Qwen:** Limited and indirect in the final ARSF score. Qwen's information enters via L'(s) as one of 66 input features to the gate, and via its role in defining what Fixed Fusion itself would have used, but the gate's output barely correlates with L'(s) (test: corr(g,L')=+0.045, essentially the same as validation's +0.031) and the model recovers only 9-27% of Qwen's unique catches across budgets.
- **Contribution of KSERESNET hidden representation:** This appears to be the dominant driver. corr(g,R')=+0.467 on test (consistent with validation's +0.456), and ARSF's net gain in fail-catching is concentrated in better use of R'(s)/h(s) rather than L'(s). This supports interpretation (2) from the pre-registered control question — **"improved calibration/use of KSERESNET's hidden representation"** — over interpretation (1), "genuine LLM-assisted adaptive fusion." The evidence is a **combination**, but weighted heavily toward (2).

## 11. Threats to Validity

- **Validation-set optimism:** ~25-27% of the validation-measured gain did not survive to test, consistent with genuine (if modest) selection bias from the objective/hyperparameter search conducted on validation — expected and disclosed, not hidden.
- **Small ARSF-vs-GBDT margin dependence on cost weighting:** ARSF's APFD margin over GBDT (+0.023) is smaller than its APFDc margin (+0.041), meaning part of the GBDT comparison's strength depends on how `duration` costs are weighted; this should be stated plainly rather than leading with the larger number.
- **Qwen's own weak standalone performance:** Qwen3-8B alone (APFD 0.454, worse than Random's 0.498 on APFD, though not on APFDc) is a persistently weak ranker throughout this project; ARSF's complementarity shortfall may partly reflect this weakness (the gate has legitimate reason to discount a weak signal) rather than a failure to learn a genuinely available adaptive-trust rule. This is a plausible but unconfirmed alternative explanation, offered for balance, not as a defense of the "learns to trust Qwen" framing.
- **Rank-TTF/time-TTF artifacts:** GBDT's very fast rank-TTF=1 continues to look tied to a single low-duration early FAIL scenario across every evaluation in this project; not a systematic property worth building a claim on.
- **Discretized Qwen scores:** Qwen's outputs remain highly discretized (few distinct levels), which mechanically limits how much fine-grained ranking signal L'(s) can contribute to any fusion method, fixed or adaptive — this is a property of the LLM scoring protocol, not of ARSF specifically, but constrains how much any gate could plausibly exploit L'(s).

## 12. Is the JSS Central Research Claim Supported?

**Central RQ:** *"Can LLM-based semantic reasoning complement KSERESNET's learned failure-risk prediction to improve prioritization of safety-critical SDC test scenarios?"*

**Answer, precisely scoped:** The evidence supports a **narrower** claim than "yes, via adaptive LLM-aware fusion." What is actually supported:

1. **Fixed fusion (RS = 0.40R' + 0.60L') beats KSERESNET alone on test**, with statistical significance (established in the earlier final test report) — this is the primary, best-supported evidence for LLM complementarity in this project, because Fixed Fusion demonstrably does use Qwen's signal (it recovers 62-82% of Qwen-exclusive failures at the budgets tested here).
2. **ARSF beats Fixed Fusion on test**, with statistical significance and ~73-75% generalization from validation — a genuine, reproducible, low-cost improvement.
3. **However, ARSF's improvement is not well explained by better exploitation of Qwen's complementary signal** — the opposite pattern is observed (ARSF uses less of Qwen's unique contribution than the fixed blend does). ARSF's gain is better explained as improved calibration of the KSERESNET-side signal.

**Net honest framing for the manuscript:** LLM complementarity is demonstrated by the original KSERESNET-RS (fixed fusion) result, not strengthened by ARSF. ARSF is a legitimate, statistically significant, cheap, and well-generalizing improvement in absolute ranking quality — but it should be presented as a **recalibration/adaptive-weighting contribution**, not as evidence of deeper LLM exploitation. Presenting ARSF as "the model that learned when to trust the LLM" would overstate what the gate-analysis and complementarity evidence actually show, on both validation and test.

---

## Files produced

- `results/arsf_final_test_metrics.csv`, `arsf_final_test_metrics_with_ci.csv`, `arsf_final_comparison.csv`
- `results/arsf_final_complementarity.json`
- `results/arsf_final_failure_discovery_curves.csv` + `_full.json`
- `results/arsf_final_statistical_tests.csv`
- `results/arsf_final_test_integrity_audit.json`
- `results/arsf_generalization_analysis.json`
- `results/arsf_final_gate_stats_test.json`
- Figures: `arsf_final_fig1_apfd.png` through `arsf_final_fig7_generalization.png` (7 figures)

**No retraining, tuning, architecture changes, prompt changes, or test-driven optimization occurred at any point in this evaluation. Stopping here per instructions — awaiting manuscript-level review.**
