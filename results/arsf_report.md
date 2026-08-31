# ARSF (Adaptive Risk-Semantic Fusion) — Validation-Only Report

**STATUS: Validation-only results. Test split (7,205 scenarios) NOT touched. Awaiting approval before any test evaluation.**

The previously completed final test result — KSERESNET-RS (alpha=0.40): test APFD=0.6387, test APFDc=0.6436 — is **untouched and unmodified** by this experiment.

---

## 1. Frozen components (verified unchanged)

- KSERESNET model/architecture/weights/scoring procedure (`crash_model.pth`) — unchanged, used read-only for both R(s) and h(s)
- Qwen3-8B model, prompt, feature representation, inference config, output schema — unchanged
- Existing R'/L' normalization procedure — reused verbatim (min-max range fit on validation split, from `results/alpha_sweep_summary.json`)
- The **only trainable component** is the ARSF gate MLP

## 2. Training-subset provenance and integrity

4,000 scenarios sampled from the 21,605-scenario TRAIN split only (see `results/arsf_train_subset_audit.json` for the full machine-verified audit):

- All 4,000 confirmed to belong to the train split; 0 overlap with validation, 0 overlap with test, 0 overlap with the Stage-1 1,000-scenario screening set; 0 duplicates.
- Selection: stratified by FAIL/PASS at the train split's own base rate (38.39%), seed=42, fixed before any result was inspected.
- 1,535 FAIL + 2,465 PASS = 4,000 (base rate 38.375%, matching full-train 38.389%).
- 17,605 train scenarios remain available and untouched for any future extension.
- Frozen Qwen3-8B was scored on this subset once (100% success, 0 errors, 6,794.74s runtime) — same model/prompt/config as the val/test runs, verified via `configs/qwen3_8b_arsf_train_config.json` on DGX.

## 3. ARSF architecture

```
z(s) = [h(s) (64-dim, standardized), R'(s), L'(s)]   (66-dim input)
g(s) = sigmoid(Linear(66,32) -> ReLU -> Linear(32,1))
ARSF(s) = g(s)*R'(s) + (1-g(s))*L'(s)
```

- `h(s)`: the frozen KSERESNET 64-dim penultimate-layer representation (after `Flatten -> Linear(1600,64) -> ReLU`, before Dropout and the final `Linear(64,1)`), extracted via a read-only forward pass through the unmodified `crash_model.pth` checkpoint. Verified against the existing frozen R-scores: Spearman ρ(raw logit, R_score) = 0.9999999998 — confirms identical checkpoint/forward pass.
- `h(s)` standardized (zero mean, unit variance) using **train-subset statistics only**, applied to validation without refitting.
- `R'(s)`, `L'(s)`: reuse the existing frozen min-max normalization range from the fixed-alpha experiment (fit on validation split, from `results/alpha_sweep_summary.json`) — not refit here.
- No transformer, attention, GAN, or additional LLM — a single small feed-forward gate as specified.

## 4. Training objective comparison (train/val only)

Pre-specified selection criterion: **argmax validation APFDc** (same criterion used throughout this project, including the original fixed-alpha sweep).

| Objective | val APFD | val APFDc | val P@100 | val NDCG@100 |
|---|---|---|---|---|
| **BCE** (ARSF(s) treated as predicted FAIL-probability) | **0.6491** | **0.6572** | **0.910** | **0.886** |
| Pairwise ranking (RankNet-style, sampled FAIL/PASS pairs) | 0.6416 | 0.6486 | 0.770 | 0.780 |

**Chosen: BCE.** Rationale: BCE outperformed the ranking objective on every reported validation metric in this comparison, not just the primary criterion — a consistent, not marginal, win. This decision was made using only train/val data and was frozen before any hyperparameter tuning or test contact.

## 5. Hyperparameter selection (train/val only)

Small, lightweight grid over hidden-layer size and learning rate (objective fixed to BCE, epochs=80, seed=42), full grid in `results/arsf_training_configs.json`:

| hidden | lr | val APFD | val APFDc | params |
|---|---|---|---|---|
| 8 | 0.001 | 0.6348 | 0.6415 | 545 |
| 8 | 0.003 | 0.6440 | 0.6513 | 545 |
| **32** | **0.003** | **0.6491** | **0.6572** | **2,177** |
| 16 | 0.001 | 0.6362 | 0.6430 | 1,089 |
| 16 | 0.003 | 0.6455 | 0.6537 | 1,089 |
| 32 | 0.001 | 0.6414 | 0.6485 | 2,177 |

**Selected: hidden=32, lr=0.003** (argmax val APFDc). Note the pattern: larger hidden size and the higher learning rate both improve validation APFDc monotonically within this small grid — the gate benefits from slightly more capacity, but the winning configuration (2,177 parameters) is still trivially lightweight.

## 6. Validation ablation — all six methods

| Method | APFD | APFDc | P@100 | R@100 | NDCG@100 | P@500 | R@500 | NDCG@500 | P@1000 | R@1000 | NDCG@1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 Random | 0.4988 | 0.4966 | 0.390 | 0.0141 | 0.3726 | 0.364 | 0.0659 | 0.3603 | 0.379 | 0.1372 | 0.3746 |
| B2 KSERESNET | 0.6236 | 0.6308 | 0.770 | 0.0279 | 0.7796 | 0.778 | 0.1408 | 0.7812 | 0.704 | 0.2548 | 0.7171 |
| B3 Qwen3-8B | 0.4538 | 0.4692 | 0.690 | 0.0250 | 0.5874 | 0.522 | 0.0945 | 0.5134 | 0.261 | 0.0945 | 0.2943 |
| B4 GBDT | 0.6316 | 0.6223 | 0.770 | 0.0279 | 0.7889 | 0.692 | 0.1252 | 0.7105 | 0.635 | 0.2298 | 0.6541 |
| **B5 Fixed Fusion (alpha=0.40)** | 0.6296 | 0.6358 | 0.720 | 0.0261 | 0.7081 | 0.730 | 0.1321 | 0.7210 | 0.720 | 0.2606 | 0.7170 |
| **B6 ARSF** | **0.6491** | **0.6572** | **0.910** | **0.0329** | **0.8857** | **0.804** | **0.1455** | **0.8113** | **0.768** | **0.2780** | **0.7775** |

With 95% bootstrap CIs (n=2,000): ARSF APFD 0.6491 [0.6412, 0.6568]; ARSF APFDc 0.6572 [0.6478, 0.6666]. Fixed Fusion APFD 0.6296 [0.6216, 0.6376]; APFDc 0.6358 [0.6262, 0.6452]. Full table with all CIs: `results/arsf_ablation.csv`.

**ARSF beats every baseline on every reported metric**, including KSERESNET alone and GBDT — this is the first configuration in the project to clearly exceed KSERESNET's own P@100/NDCG@100 (0.91/0.886 vs 0.77/0.780).

## 7. KEY COMPARISON: ARSF vs Fixed Fusion (primary result)

Paired bootstrap (2,000 resamples, same resampled index set for both methods per iteration):

| Metric | Point diff | 95% CI | p-value | Effect size |
|---|---|---|---|---|
| APFD | **+0.0196** | [0.0160, 0.0231] | <0.001 | large (CI excludes 0 by a wide margin) |
| APFDc | **+0.0214** | [0.0173, 0.0256] | <0.001 | large |

**Numerical improvement:** clear and consistent across every metric reported (APFD, APFDc, all P/R/NDCG@100/500/1000).
**Statistical significance:** yes — both CIs exclude zero comfortably, with no overlap near the boundary (unlike the earlier fixed-alpha-vs-KSERESNET validation comparison, where the CI did approach zero).
**Practical significance:** meaningfully larger than the fixed-fusion-vs-KSERESNET gain reported earlier in the project (~0.005-0.006) — here the gain is ~2 percentage points on both APFD and APFDc, and the P@100 jump (0.72 → 0.91) is substantial and would matter in a real triage budget.

## 8. Multi-seed robustness

Seeds {42, 123, 456}, architecture/objective/hyperparameters fixed, only weight initialization and pairwise-sampling RNG vary (not applicable here since BCE was chosen, so only weight init varies):

| Seed | val APFD | val APFDc |
|---|---|---|
| 42 | 0.6491 | 0.6572 |
| 123 | 0.6483 | 0.6565 |
| 456 | 0.6486 | 0.6567 |

**Mean ± std:** APFD = 0.6487 ± 0.0003 (range [0.6483, 0.6491]); APFDc = 0.6568 ± 0.0003 (range computed analogously — see `results/arsf_seed_results.csv`).

**The gate is extremely stable across seeds** — the spread (±0.0003) is roughly two orders of magnitude smaller than the ARSF-vs-Fixed-Fusion gain (~0.02). This is not a fragile result driven by a lucky initialization.

## 9. Gate analysis

- mean g(s) = 0.4880, median = 0.4813, std = 0.2706, range [0.0039, 0.9986] — the gate uses close to its full [0,1] range, not collapsing to a constant near 0 or 1.
- **g(s) differs meaningfully by outcome**: mean g | FAIL = 0.6174, mean g | PASS = 0.4073 (median 0.661 vs 0.379). The gate leans toward trusting KSERESNET (R') more on scenarios that turn out to be real failures, and toward Qwen (L') more on scenarios that turn out to pass.
- Correlation of g(s) with R'(s): **+0.456** (moderate positive — when KSERESNET's own score is high, the gate tends to weight it more). Correlation of g(s) with L'(s): **+0.031** (essentially none).

### Quadrant analysis (rank-based median split — see note below)

| Quadrant | n | mean R' | mean L' | mean g | mean ARSF | FAIL rate |
|---|---|---|---|---|---|---|
| R high / L high | 1,758 | 0.732 | 0.195 | 0.574 | 0.515 | 0.424 |
| R high / L low | 1,840 | 0.773 | 0.143 | 0.605 | 0.540 | 0.607 |
| R low / L high | 1,840 | 0.266 | 0.170 | 0.388 | 0.213 | 0.232 |
| R low / L low | 1,758 | 0.246 | 0.143 | 0.384 | 0.188 | 0.270 |

*Methodological note: a value-based median split on L'(s) is degenerate because Qwen's raw output is highly discretized (only 8 distinct levels, strongly right-skewed — over half the validation set shares the single most common value), so a naive `< median` threshold catches almost nothing below it. The quadrants above use a **rank-based** median split (top/bottom 50% by rank) instead, which produces four balanced, meaningful groups.*

**Interpretation, held to what the evidence actually supports:** the gate assigns its highest average weight to R' (mean g ≈ 0.60) specifically in the "R high / L low" quadrant — exactly the region with the highest FAIL rate (60.7%) in the whole validation set — and its lowest weight to R' (mean g ≈ 0.38-0.39) in the two "R low" quadrants, where L' contributes more. This is a directionally sensible pattern (favor whichever signal is more informative), but the correlation with L'(s) is essentially zero (+0.03), meaning **the gate is not visibly reacting to Qwen's score on its own** — its adaptivity is driven almost entirely by R'(s) and the 64-dim hidden representation h(s), not by an explicit read of L'(s). We do **not** claim the gate "understands" which model is more reliable in a given scenario; the evidence supports only that its output correlates with R'(s) and with the outcome label, not that it has learned an interpretable trust-arbitration rule between the two models.

## 10. Complementarity: Fixed Fusion vs ARSF

| K | Fixed fail hits | ARSF fail hits | Shared | Jaccard | Qwen-only fails at K | ARSF recovers from Qwen-only |
|---|---|---|---|---|---|---|
| 100 | 72 | 91 | 18 | 0.124 | 62 | 10 (16.1%) |
| 500 | 365 | 402 | 175 | 0.296 | 217 | 26 (12.0%) |
| 1000 | 720 | 768 | 504 | 0.512 | 191 | 35 (18.3%) |

**ARSF catches more failures than Fixed Fusion at every budget** (91 vs 72 at K=100; 402 vs 365 at K=500; 768 vs 720 at K=1000) — a consistent, positive net gain. However, ARSF's improvement is only **partially** attributable to better exploitation of Qwen's unique signal: it recovers only 12-18% of the failures that are exclusively in Qwen's top-K (a similar or slightly lower fraction than what the earlier test-set complementarity analysis found for Fixed Fusion itself, ~62-64% at K=500/1000). The larger driver of ARSF's improvement appears to be a **better-calibrated combination of the KSERESNET signal it already had**, not a dramatically improved use of the Qwen-only complementary signal — consistent with the gate-analysis finding that g(s) correlates strongly with R'(s) and weakly with L'(s).

## 11. Model complexity / cost

| | ARSF gate |
|---|---|
| Trainable parameters | 2,177 |
| Architecture | Linear(66,32) → ReLU → Linear(32,1) → Sigmoid |
| Model file size | 11.05 KB |
| Training time (80 epochs, full-batch, CPU) | 0.132 s |
| Inference time (val, batched, CPU) | 0.838 ms total / **0.000116 ms per scenario** |
| Device | CPU only (no GPU required) |

For comparison: Qwen3-8B inference alone costs ~1,699 ms/scenario — the ARSF gate adds roughly **1.8 × 10⁷ times less** compute per scenario than the LLM call it consumes the output of. KSERESNET's own forward pass (sub-millisecond, ~561 KB checkpoint) also dwarfs the gate's overhead. **The adaptive layer is, as intended, essentially free relative to either upstream model.**

## 12. Outcome classification

Per the pre-specified taxonomy:

**A. Strong positive.** ARSF consistently and substantially improves over Fixed Fusion on validation: APFD +0.0196 (CI [0.0160, 0.0231]), APFDc +0.0214 (CI [0.0173, 0.0256]), both clearly statistically significant with no CI overlap near zero; early-budget metrics improve sharply (P@100 0.72→0.91, NDCG@100 0.708→0.886); the result is stable across 3 seeds (std ≈ 0.0003) and stable-to-improving across a small hyperparameter grid; and the added model complexity is negligible (2,177 parameters, sub-millisecond training, effectively free inference).

**Caveat on the mechanism:** the complementarity analysis (§10) and the low g(s)-vs-L'(s) correlation (§9) together suggest the improvement is driven more by a better-calibrated use of the KSERESNET signal (and the 64-dim hidden representation h(s)) than by dramatically superior exploitation of Qwen's complementary failures. This is reported honestly rather than framed as "the gate learned to trust Qwen when it matters."

---

## Explicit answers to the eight questions

1. **Is adaptive fusion technically feasible?** Yes — a lightweight (2,177-parameter) gate trains in well under a second on CPU and produces a valid, bounded g(s) ∈ [0,1] output for every scenario.
2. **Does ARSF outperform fixed alpha=0.40 fusion on validation?** Yes, clearly, on every reported metric.
3. **How much?** APFD +0.0196, APFDc +0.0214 — roughly 3-4x the magnitude of the earlier fixed-fusion-vs-KSERESNET-alone gain.
4. **Is the improvement statistically/practically meaningful?** Statistically: yes, CIs exclude zero with margin. Practically: yes — the P@100 jump from 0.72 to 0.91 would materially change a real triage workflow at that budget.
5. **Does the gate actually vary meaningfully across scenarios?** Yes — g(s) spans nearly the full [0,1] range (std 0.27) and differs systematically between FAIL and PASS scenarios (0.617 vs 0.407 mean).
6. **Does ARSF recover more complementary failures?** Partially — it catches more total failures than Fixed Fusion at every budget (§10), but recovers only 12-18% of Qwen-exclusive failures, so the gain is not primarily explained by better exploitation of Qwen's unique signal.
7. **Is the additional model complexity small?** Yes — 2,177 parameters, 11 KB, ~1.8x10⁷ times cheaper per scenario than the Qwen3-8B call it depends on.
8. **Is ARSF worth carrying forward to a new held-out test evaluation?** Based on validation evidence alone: yes, this looks like a genuine, stable, low-cost improvement over Fixed Fusion — but this recommendation is offered for your review, not a decision, per the stop condition below.

---

## Files produced

- `results/arsf_train_subset_audit.json` — train-subset selection/integrity audit
- `results/arsf_objective_comparison.json` — BCE vs ranking objective comparison
- `results/arsf_training_configs.json` — full hyperparameter grid + selected config
- `results/arsf_seed_results.csv` / `arsf_seed_summary.json` — multi-seed robustness
- `results/arsf_validation_metrics.csv`, `arsf_ablation.csv` — full metrics ± CIs, all 6 methods
- `results/arsf_vs_fixed_fusion_stats.json` — primary paired-bootstrap comparison
- `results/arsf_gate_values.csv` — g(s) for every validation scenario
- `results/arsf_gate_analysis.json` — gate distribution + quadrant analysis
- `results/arsf_complementarity.json` — Fixed Fusion vs ARSF failure overlap at K={100,500,1000}
- `results/arsf_model_complexity.json` — parameter count, timing, size, cost comparison
- `results/arsf_gate_seed42.pt` — frozen primary gate checkpoint (seed=42, hidden=32, lr=0.003, BCE)
- Figures: `arsf_fig1_apfd.png`, `arsf_fig2_apfdc.png`, `arsf_fig3_failure_discovery.png`, `arsf_fig4_gate_distribution.png`, `arsf_fig5_gate_vs_scores.png`

## STOP CONDITION HONORED

Test split (7,205 scenarios) was not loaded, read, or referenced anywhere in this experiment. No test evaluation has been performed. The existing final test result (KSERESNET-RS, alpha=0.40, test APFD=0.6387/APFDc=0.6436) is unmodified. **Awaiting explicit approval before any ARSF test-set evaluation.**
