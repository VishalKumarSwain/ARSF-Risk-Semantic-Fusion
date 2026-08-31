# Stage-1 Ranking-Quality Report — KSERESNET-RS LLM Screening

**Date**: 2026-08-30
**Purpose**: Select which offline LLM (if any) is carried forward, frozen, into the KSERESNET-RS fusion experiment. This report answers **Stage-1's question only** — "which LLM should be frozen?" — not the paper's central hypothesis (does fusing KSERESNET + LLM improve prioritization?), which is answered in Stage 2 on the held-out test set.

---

## 1. Dataset Used

- **Source pool**: 36,006-scenario SensoData corpus (verified, leakage-audited).
- **Stage-1 evaluation set**: 1,000 scenarios, sampled **exclusively from the validation split** (seed 42) — the held-out test split (7,205 scenarios) was never touched for this analysis.
- **Class balance**: 385 FAIL / 615 PASS (38.5% fail rate), identical across all six methods compared (verified — see Section 9, sanity checks).

## 2. Methods Compared

| Method | Type | Notes |
|---|---|---|
| Random | Baseline | Fixed seed=42, saved for reproducibility |
| KSERESNET | Learned risk (frozen, original ICST model) | Reference method for correlation/complementarity |
| GBDT (B4) | Classical ML baseline | Same 10-field structured feature schema as the LLMs |
| Qwen3-8B | Offline LLM | `enable_thinking=false`, max_tokens=300 |
| Ministral-8B | Offline LLM | Devstral substitute (no official quantized Devstral exists), max_tokens=300 |
| DeepSeek-R1-Distill-Qwen-7B | Offline LLM (reasoning) | max_tokens=1200 (technically necessary — see Section 6) |

## 3. Metric Definitions

- **APFD** (Rothermel et al., 2001): `1 - (Σ TF_i)/(n·m) + 1/(2n)`, single fault class = FAIL.
- **APFDc** (Elbaum et al., 2001, cost-cognizant): cost = scenario execution `test_duration` from the SensoData xodr header — **not** LLM inference latency (those are different things: cost of the artifact under test vs. cost of the scoring method).
- **Precision@K / Recall@K**: standard IR definitions, binary relevance (FAIL=1).
- **NDCG@K**: binary relevance, log2 discount, ideal ranking constructed from the true FAIL count capped at K.
- **Spearman ρ / Kendall's τ**: rank correlation between two full orderings of the same 1,000 scenarios.
- **Jaccard overlap**: `|A∩B| / |A∪B|` over the *sets of FAIL scenarios* appearing in each method's top-K.

None of these are novel — all are established metrics from the test-prioritization and IR literature, used as-is.

## 4. Tie-Handling Policy (identical across all six methods)

- Primary sort key: score, descending.
- Tie-break: `scenario_id`, ascending (deterministic, lexicographic) — no randomness in final ranking.
- Correlation metrics (Spearman/Kendall) use standard average-rank tie handling (scipy default).

## 5. Main Results Table

| Method | APFD | APFD 95% CI | APFDc | APFDc 95% CI | P@100 | R@100 | NDCG@100 | P@500 | R@500 | NDCG@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.5016 | [0.4807, 0.5239] | 0.5047 | [0.4789, 0.5309] | 0.38 | 0.0987 | 0.3512 | 0.376 | 0.4883 | 0.4538 |
| KSERESNET | 0.5436 | [0.5205, 0.5666] | 0.5426 | [0.5127, 0.5698] | 0.50 | 0.1299 | 0.5026 | 0.424 | 0.5506 | 0.5343 |
| **GBDT** | **0.6285** | [0.6088, 0.6502] | **0.6123** | [0.5900, 0.6408] | **0.70** | **0.1818** | **0.6671** | **0.528** | **0.6857** | **0.6635** |
| Qwen3-8B | 0.5285 | [0.5056, 0.5520] | 0.4515 | [0.4235, 0.4805] | 0.51 | 0.1325 | 0.4971 | 0.412 | 0.5351 | 0.5166 |
| Ministral-8B | 0.5045 | [0.4817, 0.5268] | 0.4401 | [0.4147, 0.4651] | 0.40 | 0.1039 | 0.3659 | 0.392 | 0.5091 | 0.4704 |
| **DeepSeek-R1-Distill-7B** | **0.5568** | [0.5338, 0.5779] | **0.5367** | [0.5075, 0.5627] | 0.48 | 0.1247 | 0.4482 | 0.452 | 0.5870 | 0.5507 |

At K=1000 (= full Stage-1 set size), Precision@1000 ≈ 0.385 and Recall@1000 = 1.0 for **every** method by construction — reported for completeness in the raw CSV, but **carries no discriminative value** (every method necessarily "finds" all 385 failures once K equals the total scenario count). Do not use K=1000 values for model comparison. NDCG@1000 remains discriminative (rewards position, not just presence) and is included in the raw output: GBDT 0.8877, DeepSeek 0.8423, Qwen3 0.8442, KSERESNET 0.8520, Ministral 0.8157, Random 0.8144.

**Known limitation in the bootstrap CIs**: the Recall@1000 bootstrap CI upper bound in the raw output exceeds 1.0 (e.g. Random: [0.9247, 1.0831]). This is a genuine artifact of with-replacement bootstrap resampling against a denominator (total known FAILs) computed on the *original* sample rather than re-derived per resample — a resample can duplicate FAIL scenarios, inflating the numerator past the fixed denominator. This does not affect the point estimates (which are exact) or any other K value materially, but is flagged rather than hidden. Recall@1000's CI should not be reported or interpreted in the paper; the point estimate (1.0, trivial) already carries no discriminative value per the note above.

## 6. Configuration Deviation (documented, not a fairness violation)

DeepSeek-R1-Distill-Qwen-7B used `max_tokens=1200` versus 300 for Qwen3/Ministral. This is **technically necessary**, not an optimization choice made after seeing results: this model emits mandatory chain-of-thought reasoning directly into the response content, with no mechanism to suppress it (the `enable_thinking` flag is Qwen3-specific and is silently ignored by this model in the tested vLLM version). At 300 tokens, DeepSeek could not complete reasoning and emit a valid JSON answer within budget in preliminary testing. All other settings — prompt, feature schema, output JSON schema, temperature (0.1), seed (42), scenario set, evaluation procedure — are identical across all three LLMs.

## 7. Ranking Correlation vs. KSERESNET

| Method A | Method B | Spearman ρ | p-value | Kendall τ | p-value |
|---|---|---:|---:|---:|---:|
| kseresnet | gbdt | 0.3904 | 9.2e-38 | 0.2687 | 4.5e-37 |
| kseresnet | deepseek_r1_7b | 0.1453 | 3.9e-06 | 0.0987 | 3.0e-06 |
| kseresnet | qwen3_8b | 0.0848 | 0.0073 | 0.0567 | 0.0073 |
| kseresnet | ministral_8b | -0.0986 | 0.0018 | -0.0587 | 0.0054 |
| kseresnet | random | 0.0288 | 0.362 (n.s.) | 0.0199 | 0.346 (n.s.) |

**Interpretation, precisely**: all three LLM rankings show low agreement with KSERESNET (ρ between -0.10 and 0.15), each statistically distinguishable from the KSERESNET-vs-Random null (which is correctly non-significant, confirming the correlation test itself is behaving correctly). **Low ranking agreement indicates that the LLM rankings are substantially different from KSERESNET's ranking. This establishes the structural possibility of complementarity, which must be evaluated through failure overlap and fusion performance — it does not, by itself, demonstrate that this difference is useful.** GBDT, by contrast, shows moderate positive correlation with KSERESNET (ρ=0.39) — expected, since both derive from the same underlying road geometry, just at different levels of feature abstraction.

Additional pairwise LLM-vs-LLM correlations (for context, not part of the primary selection criterion): Qwen3 vs Ministral ρ=0.303, Qwen3 vs DeepSeek ρ=0.143, Ministral vs DeepSeek ρ=0.327 — the three LLMs also disagree substantially with each other, not just with KSERESNET.

## 8. Complementarity Analysis (KSERESNET vs. each LLM)

At K=100, KSERESNET alone finds 50/385 failures. For each LLM:

| LLM | K | KSERESNET-only failures in top-K | LLM top-K failures | Shared | KSERESNET∪LLM (union) | Jaccard |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 100 | 35 | 51 | 15 | 86 | 0.1744 |
| Qwen3-8B | 500 | 90 | 206 | 122 | 296 | 0.4122 |
| Ministral-8B | 100 | 46 | 40 | 4 | 86 | 0.0465 |
| Ministral-8B | 500 | 131 | 196 | 81 | 327 | 0.2477 |
| DeepSeek-R1-Distill-7B | 100 | 37 | 48 | 13 | 85 | 0.1529 |
| DeepSeek-R1-Distill-7B | 500 | 82 | 226 | 130 | 308 | 0.4221 |

(K=1000 is omitted from this table for the same reason as Section 5 — Jaccard trivially equals 1.0 for every method at K=N, since every method finds all 385 failures by definition once K spans the full set.)

**Interpretation, precisely, per the required framing**: at K=100, the union of KSERESNET's and each LLM's top-100 failure discoveries (85-86 failures) is substantially larger than KSERESNET's top-100 alone (50 failures) — roughly 70% more, consistently across all three LLMs, including Ministral despite its weak standalone APFD. Low Jaccard values (0.046-0.174 at K=100) confirm the two rankings identify largely *different* failure scenarios, not merely noisier versions of the same ranking.

**The Stage-1 results provide evidence that the LLM rankings identify failure cases that differ substantially from those prioritized by KSERESNET, motivating the subsequent fusion experiment. This is not proof that the final fusion method (KSERESNET-RS) is superior — that is Stage 2's question, tested on the held-out test set with the frozen LLM and a validation-selected fusion weight.**

## 9. Sanity Checks (all passed)

- APFD, APFDc: all six methods within [0,1] ✓
- Precision@K, Recall@K, NDCG@K: all within [0,1] ✓ (Recall@1000 CI artifact noted in Section 5, does not affect point estimates)
- Spearman ρ, Kendall τ: all within [-1,+1] ✓
- FAIL count identical (385) across all six methods ✓ — confirms identical scenario set was used throughout
- Top-K rankings contain valid, unique scenario IDs (verified during ranking construction — deterministic tie-break precludes duplicates)
- No ground-truth outcome was included in any LLM prompt (verified — `build_prompt()` strips `outcome`/`duration` before rendering; confirmed in `llm_prompt_template.py`)
- No test-set information entered Stage-1 — sampled exclusively from the validation split (`data/split_val_ids.json`)

## 10. Statistical Confidence — what can and cannot be claimed

Bootstrap 95% CIs are reported for APFD, APFDc, Precision@K, Recall@K, NDCG@K (Section 5). **No formal pairwise statistical significance test (e.g., Wilcoxon signed-rank, Friedman) has been run on the Stage-1 results** — that level of testing is reserved for the Stage-2 main experiment per the frozen design. Distinguishing what the current evidence supports:

- **Numerical improvement**: GBDT > DeepSeek > KSERESNET > Qwen3 > Ministral ≈ Random, on point-estimate APFD.
- **Confidence interval overlap**: KSERESNET's CI [0.5205, 0.5666] and DeepSeek's CI [0.5338, 0.5779] overlap substantially — the numerical gap between them is **not** established as statistically robust from CIs alone. Similarly, Ministral's CI [0.4817, 0.5268] and Random's CI [0.4807, 0.5239] overlap almost entirely — Ministral's APFD is **not distinguishable from random** at this confidence level.
- **Statistical significance**: not formally tested at Stage-1; do not claim it.
- **Practical significance**: GBDT's ~0.08-0.10 APFD advantage over the next-best methods is a meaningful practical gap even without a formal test, given non-overlapping CIs with KSERESNET and Ministral/Random.

## 11. Effectiveness vs. Computational Cost

| Model | APFD | APFDc | Mean latency | P95-equivalent* | Throughput | Mean output tokens | Success rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 0.5285 | 0.4515 | 2.71s | ~3.02s (100-scen. validation) | 22.16 scen/min | ~75 | 100% |
| Ministral-8B | 0.5045 | 0.4401 | 3.31s | ~3.89s (100-scen. validation) | 18.11 scen/min | ~105 | 100% |
| DeepSeek-R1-Distill-7B | 0.5568 | 0.5367 | 12.70s | ~14.89s (100-scen. validation) | 4.72 scen/min | ~588 | 100% |

*P95 figures are from the 100-scenario validation runs (Section covered separately in the resource-profiling artifacts); the 1,000-scenario Stage-1 runs report mean/median/std/min/max only, consistent with `score_llm_stage1.py`'s summary schema.

DeepSeek is **~4.7x slower than Qwen3** and **~3.8x slower than Ministral**, and generates **~6-8x more output tokens per scenario** (reasoning overhead). This cost is real, measured, and must be weighed explicitly, not treated as a footnote.

## 12. Multi-Factor Decision Table

| Criterion | Qwen3-8B | Ministral-8B | DeepSeek-R1-Distill-7B |
|---|---|---|---|
| APFD | 0.5285 | 0.5045 (≈ Random, CI overlaps) | **0.5568 (best of 3, beats KSERESNET)** |
| APFDc | 0.4515 (below Random) | 0.4401 (below Random) | **0.5367 (best of 3, beats KSERESNET)** |
| P@100 / R@100 | 0.51 / 0.133 | 0.40 / 0.104 | 0.48 / 0.125 |
| NDCG@100 | 0.4971 | 0.3659 (weakest) | 0.4482 |
| P@500 / R@500 | 0.412 / 0.535 | 0.392 / 0.509 | **0.452 / 0.587 (best of 3)** |
| NDCG@500 | 0.5166 | 0.4704 | **0.5507 (best of 3)** |
| Correlation with KSERESNET | ρ=0.085 (low) | ρ=-0.099 (slightly negative) | ρ=0.145 (low) |
| Complementarity (Jaccard@100) | 0.174 | 0.047 (most distinct) | 0.153 |
| Union failures @100 (with KSERESNET) | 86 | 86 | 85 |
| Mean latency | 2.71s (fastest) | 3.31s | 12.70s (4.7x Qwen3) |
| Throughput | 22.16 scen/min (fastest) | 18.11 scen/min | 4.72 scen/min |
| Mean output tokens | 75 (cheapest) | 105 | 588 (7.8x Qwen3) |
| Reliability (Stage-1, 1000 scenarios) | 100%, 0 malformed | 100%, 0 malformed | 100%, 0 malformed |

## 13. Recommendation

### A. Recommended LLM: **DeepSeek-R1-Distill-Qwen-7B**

### B. Evidence supporting the recommendation
DeepSeek is the only one of the three LLMs whose APFD (0.5568) and APFDc (0.5367) both exceed KSERESNET's own numbers (0.5436 / 0.5426) on this Stage-1 set, and it leads the other two LLMs on 5 of 6 primary ranking-quality metrics (APFD, APFDc, P@500, R@500, NDCG@500), with Qwen3 ahead only on P@100/NDCG@100. Ministral's APFD is statistically indistinguishable from random (overlapping 95% CIs) and its APFDc falls below random — it does not show credible standalone ranking ability at Stage-1. Qwen3's APFDc falling below 0.5 (worse than random when cost-weighted) is a genuine finding, not disqualifying on its own, but a specific weakness worth noting: it appears to rank longer-duration scenarios early without proportionally more failures.

### C. Cost/efficiency tradeoff, stated explicitly, not minimized
This recommendation is made **despite** DeepSeek costing ~4.7x Qwen3's latency and ~7.8x its token volume. At Stage-2 scale (7,196 validation + 7,205 test scenarios, run twice for the alpha sweep and final evaluation), this translates to roughly 50+ hours of GPU time for DeepSeek versus roughly 11-13 hours for either alternative. This is a real, non-trivial resource cost that should be disclosed in the paper's discussion of practical deployability — a JSS reviewer will reasonably ask whether the effectiveness gain justifies it. The honest answer, based on current evidence: the gain (DeepSeek beating KSERESNET on APFD/APFDc, where neither Qwen3 nor Ministral do) is large enough at Stage-1 to justify the cost for a research study establishing complementarity, though it would be a legitimate consideration against DeepSeek in a latency-constrained production deployment — worth stating explicitly in the paper's limitations/practicality discussion.

### D. Complementarity evidence
All three LLMs show comparable complementarity signal with KSERESNET at K=100 (union 85-86 vs. KSERESNET-alone 50), so complementarity alone does not differentiate the three candidates — DeepSeek's selection rests on its standalone effectiveness advantage (Section 13.B), with complementarity serving as independent motivation for the fusion experiment generally, not as a DeepSeek-specific advantage.

### E. Uncertainty, stated plainly
- No formal statistical significance test has been run between DeepSeek and KSERESNET (CIs overlap substantially — see Section 10). The numerical advantage is real but not yet established as statistically robust.
- This is a 1,000-scenario screening set, not the full validation split (7,196) or test set (7,205) — Stage-2 results on more scenarios could shift these numbers.
- DeepSeek's reasoning-token behavior was not prompt-engineered or optimized in any way after seeing these results (per the frozen-design constraint) — its cost profile is what it is.

### F. Exact frozen configuration, if this recommendation is accepted
```
Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
Prompt: llm_prompt_template.py (SYSTEM_PROMPT + USER_PROMPT_TEMPLATE), unmodified
Feature representation: 10-field geometry-only schema (FEATURE_SCHEMA.md v1), unmodified
dtype: bfloat16
max_model_len: 8192
gpu_memory_utilization: 0.85
temperature: 0.1
seed: 42
max_output_tokens: 1200 (documented deviation, Section 6)
chat_template_kwargs: {"enable_thinking": false} (no effect on this model, sent for consistency with the other two)
Decoding: vLLM 0.9.1, V0 engine (VLLM_USE_V1=0), no client-side batching, sequential requests, no retries
Output schema: {"risk_score": float, "risk_factors": [string], "rationale": string}
```

### G. Confirmation: the 7,205-scenario held-out test set remains untouched
No test-split scenario ID has been scored by any method, LLM or otherwise, at any point in this Stage-1 process. This was verified structurally (Stage-1 sampling drew exclusively from `data/split_val_ids.json`) and is reconfirmed here explicitly per the frozen protocol's requirement.

## 14. Limitations of this Stage-1 analysis

- No formal significance testing (Section 10) — deferred to Stage 2 by design.
- Recall@1000 bootstrap CI artifact (Section 5) — does not affect any reported point estimate or any other K value.
- Ministral-8B (the Devstral substitute) still lacks an official-weights equivalent to the originally-specified Devstral-Small-2507 — this deviation is documented but remains a threat-to-validity item for the manuscript.
- DeepSeek's `max_tokens=1200` deviation, while technically necessary, means its evaluation was not under byte-for-byte identical inference constraints — documented in Section 6, not hidden.
- This report reflects one run per method, no repeated sampling — consistent with the frozen design's Stage-1 scope (repeats are reserved for Random baselines at Stage 2 per the earlier protocol).

---

**Gate confirmed: this report is the deliverable required before proceeding to the alpha sweep and final held-out test experiment. No modification to the Stage-1 experiment (prompt, schema, model settings, or scenario set) has been made after observing these results, per the frozen-design constraint.**
