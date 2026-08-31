"""
FINAL held-out test evaluation for KSERESNET-RS. Test split (n=7205) is
used ONLY here, for the first and only time. No tuning occurs in this
script -- alpha, normalization range, prompt, and model are all reused
verbatim from frozen validation-time artifacts.
"""
import csv
import json
import sys

import numpy as np

ALPHA_STAR = 0.40
N_BOOTSTRAP = 2000
K_LIST = [100, 500, 1000]
BUDGETS = [100, 500, 1000, 2000, 5000]  # + "full" handled separately
RNG_SEED = 42

np.random.seed(RNG_SEED)


def load_r_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return ({r["test_id"]: r["R_score"] for r in d},
            {r["test_id"]: r["outcome"] for r in d},
            {r["test_id"]: r.get("duration") for r in d})


def load_l_scores(path):
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[row["scenario_id"]] = float(row["score"])
    return scores


def load_g_scores(path):
    return {r["test_id"]: r["G_score"] for r in
            json.load(open(path, "r", encoding="utf-8")) if r["split"] == "test"}


def apfd_vec(y_sorted):
    n = len(y_sorted)
    m = y_sorted.sum()
    if m == 0:
        return None
    fail_pos = np.nonzero(y_sorted)[0] + 1
    return float(1 - (fail_pos.sum() / (n * m)) + 1 / (2 * n))


def apfdc_vec(y_sorted, dur_sorted):
    n = len(y_sorted)
    m = y_sorted.sum()
    total_cost = dur_sorted.sum()
    if m == 0 or total_cost <= 0:
        return None
    suffix_sum = np.cumsum(dur_sorted[::-1])[::-1]
    fail_idx = np.nonzero(y_sorted)[0]
    cost_after = suffix_sum[fail_idx] - dur_sorted[fail_idx] / 2.0
    return float(cost_after.sum() / (total_cost * m))


def prk_vec(y_sorted, k):
    n = len(y_sorted)
    k = min(k, n)
    top = y_sorted[:k]
    total_rel = y_sorted.sum()
    tp = top.sum()
    precision = tp / k if k > 0 else 0.0
    recall = tp / total_rel if total_rel > 0 else 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (top * discounts).sum()
    ideal = np.sort(y_sorted)[::-1][:k]
    idcg = (ideal * discounts).sum()
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return float(precision), float(recall), float(ndcg)


def ttf_vec(y_sorted, dur_sorted):
    idx = np.nonzero(y_sorted)[0]
    if len(idx) == 0:
        return None, None
    first = idx[0]
    return int(first) + 1, float(dur_sorted[:first + 1].sum())


def compute_metrics(y, dur, score):
    order = np.argsort(-score, kind="stable")
    y_sorted = y[order]
    dur_sorted = dur[order]
    a = apfd_vec(y_sorted)
    ac = apfdc_vec(y_sorted, dur_sorted)
    rank_ttf, time_ttf = ttf_vec(y_sorted, dur_sorted)
    result = {"APFD": a, "APFDc": ac, "rank_TTF": rank_ttf, "time_TTF": time_ttf}
    for k in K_LIST:
        p, r, nd = prk_vec(y_sorted, k)
        result[f"P@{k}"] = p
        result[f"R@{k}"] = r
        result[f"NDCG@{k}"] = nd
    return result, order, y_sorted, dur_sorted


def bootstrap_ci(y, dur, score, n_boot=N_BOOTSTRAP):
    n = len(y)
    keys = ["APFD", "APFDc"] + [f"{m}@{k}" for m in ("P", "R", "NDCG") for k in K_LIST]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        m, _, _, _ = compute_metrics(y[idx], dur[idx], score[idx])
        for k in keys:
            if m[k] is not None:
                samples[k].append(m[k])
    ci = {}
    for k, vals in samples.items():
        arr = np.array(vals)
        ci[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) >= 10 else (None, None)
    return ci


def paired_bootstrap_diff(y, dur, score_a, score_b, metric_key, n_boot=N_BOOTSTRAP):
    """Paired bootstrap: same resampled index set used for both methods each iter."""
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        ma, _, _, _ = compute_metrics(y[idx], dur[idx], score_a[idx])
        mb, _, _, _ = compute_metrics(y[idx], dur[idx], score_b[idx])
        if ma[metric_key] is not None and mb[metric_key] is not None:
            diffs.append(ma[metric_key] - mb[metric_key])
    diffs = np.array(diffs)
    point = float(np.mean(diffs))
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    # empirical two-sided p-value: proportion of bootstrap diffs crossing 0 relative to observed sign
    p_frac_le0 = float(np.mean(diffs <= 0))
    p_frac_ge0 = float(np.mean(diffs >= 0))
    p_value = 2 * min(p_frac_le0, p_frac_ge0)
    p_value = min(p_value, 1.0)
    std = float(np.std(diffs))
    effect_size = point / std if std > 0 else 0.0  # standardized mean difference (bootstrap-based Cohen's-d analogue)
    significant = not (ci_lo <= 0 <= ci_hi)
    return {
        "metric": metric_key, "point_diff": point, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "bootstrap_p_value": p_value, "effect_size_bootstrap_d": effect_size,
        "significant_95ci": significant,
    }


def failure_discovery_curve(y_sorted, budgets, n_total):
    total_fail = int(y_sorted.sum())
    cum = np.cumsum(y_sorted)
    out = []
    for b in budgets + [n_total]:
        b = min(b, n_total)
        out.append({"budget": int(b) if b != n_total else "full",
                     "cum_failures": int(cum[b - 1]),
                     "frac_failures_found": float(cum[b - 1] / total_fail) if total_fail else 0.0})
    return out


def main():
    print("=== FINAL HELD-OUT TEST EVALUATION (test split, n=7205) ===", file=sys.stderr)
    print("Frozen alpha =", ALPHA_STAR, file=sys.stderr)

    r_all, outcome_all, duration_all = load_r_scores("kseresnet_R_scores.json")
    l_all = load_l_scores("results_rankings_qwen3_8b_test_full.csv")
    g_all = load_g_scores("gbdt_G_scores.json")

    with open("split_test_ids.json", "r", encoding="utf-8") as f:
        test_ids_raw = json.load(f)
    test_ids = [i for i in test_ids_raw if i in r_all and i in l_all and i in g_all]
    assert len(test_ids) == 7205, f"Expected 7205 test ids, got {len(test_ids)}"

    y = np.array([1 if outcome_all[i] == "FAIL" else 0 for i in test_ids], dtype=np.int64)
    dur = np.array([duration_all[i] or 0.0 for i in test_ids], dtype=np.float64)
    r_raw = np.array([r_all[i] for i in test_ids], dtype=np.float64)
    l_raw = np.array([l_all[i] for i in test_ids], dtype=np.float64)
    g_raw = np.array([g_all[i] for i in test_ids], dtype=np.float64)

    n_fail = int(y.sum())
    print(f"n={len(test_ids)}, n_fail={n_fail}, n_pass={len(test_ids)-n_fail}", file=sys.stderr)

    # --- FROZEN normalization: reuse validation-fit min/max, do NOT refit on test ---
    with open("results/alpha_sweep_summary.json", "r", encoding="utf-8") as f:
        val_summary = json.load(f)
    r_lo, r_hi = val_summary["r_range"]
    l_lo, l_hi = val_summary["l_range"]
    print(f"Reusing FROZEN val-fit normalization: R range={r_lo,r_hi}, L range={l_lo,l_hi}", file=sys.stderr)

    r_norm = (r_raw - r_lo) / (r_hi - r_lo if r_hi > r_lo else 1.0)
    l_norm = (l_raw - l_lo) / (l_hi - l_lo if l_hi > l_lo else 1.0)
    # test values may fall outside the val-fit [0,1] range; that's expected/correct (no re-fit), leave as-is (can be <0 or >1)

    rng = np.random.RandomState(RNG_SEED)
    random_scores = rng.random(len(test_ids))

    rs_scores = ALPHA_STAR * r_norm + (1 - ALPHA_STAR) * l_norm

    methods = {
        "B1_Random": random_scores,
        "B2_KSERESNET": r_raw,
        "B3_Qwen3_8B": l_raw,
        "B4_GBDT": g_raw,
        "B5_KSERESNET_RS": rs_scores,
    }

    # ---------------- primary metrics table ----------------
    all_metrics = {}
    all_orders = {}
    for name, scores in methods.items():
        m, order, y_sorted, dur_sorted = compute_metrics(y, dur, scores)
        all_metrics[name] = m
        all_orders[name] = (order, y_sorted, dur_sorted)
        print(f"{name}: APFD={m['APFD']:.4f} APFDc={m['APFDc']:.4f} "
              f"P@100={m['P@100']:.4f} R@100={m['R@100']:.4f} NDCG@100={m['NDCG@100']:.4f}", file=sys.stderr)

    with open("results/final_test_metrics.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "APFD", "APFDc"] + [f"{m}@{k}" for m in ("P", "R", "NDCG") for k in K_LIST] + ["rank_TTF", "time_TTF"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, m in all_metrics.items():
            row = {"method": name}
            row.update({k: m[k] for k in fieldnames if k != "method"})
            w.writerow(row)
    print("Wrote results/final_test_metrics.csv", file=sys.stderr)

    # ---------------- bootstrap CIs per method ----------------
    print("\nComputing bootstrap CIs per method...", file=sys.stderr)
    all_cis = {}
    for name, scores in methods.items():
        ci = bootstrap_ci(y, dur, scores)
        all_cis[name] = ci
        print(f"  {name} done", file=sys.stderr)

    with open("results/final_test_metrics_with_ci.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["APFD", "APFDc"] + [f"{m}@{k}" for m in ("P", "R", "NDCG") for k in K_LIST]
        fieldnames = ["method"] + [c for k in keys for c in (k, f"{k}_CI_lo", f"{k}_CI_hi")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name in methods:
            row = {"method": name}
            for k in keys:
                row[k] = all_metrics[name][k]
                row[f"{k}_CI_lo"], row[f"{k}_CI_hi"] = all_cis[name][k]
            w.writerow(row)
    print("Wrote results/final_test_metrics_with_ci.csv", file=sys.stderr)

    # ---------------- budget-aware failure discovery curves ----------------
    fdc_rows = []
    for name in methods:
        _, y_sorted, _ = all_orders[name]
        curve = failure_discovery_curve(y_sorted, BUDGETS, len(test_ids))
        for pt in curve:
            fdc_rows.append({"method": name, "budget": pt["budget"],
                              "cum_failures": pt["cum_failures"], "frac_failures_found": pt["frac_failures_found"]})
    with open("results/final_failure_discovery_curves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "budget", "cum_failures", "frac_failures_found"])
        w.writeheader()
        for row in fdc_rows:
            w.writerow(row)
    print("Wrote results/final_failure_discovery_curves.csv", file=sys.stderr)

    # full-resolution curves for plotting (every scenario, not just budget checkpoints)
    full_curves = {}
    for name in methods:
        _, y_sorted, _ = all_orders[name]
        full_curves[name] = np.cumsum(y_sorted).tolist()
    with open("results/final_failure_discovery_curves_full.json", "w", encoding="utf-8") as f:
        json.dump(full_curves, f)
    print("Wrote results/final_failure_discovery_curves_full.json", file=sys.stderr)

    # ---------------- statistical tests: primary + secondary comparisons ----------------
    print("\nRunning paired bootstrap significance tests...", file=sys.stderr)
    comparisons = [
        ("B5_KSERESNET_RS", "B2_KSERESNET", "PRIMARY (H1): KSERESNET-RS vs KSERESNET"),
        ("B5_KSERESNET_RS", "B4_GBDT", "secondary: KSERESNET-RS vs GBDT"),
        ("B5_KSERESNET_RS", "B3_Qwen3_8B", "secondary: KSERESNET-RS vs Qwen3-8B"),
        ("B5_KSERESNET_RS", "B1_Random", "secondary: KSERESNET-RS vs Random"),
    ]
    stat_rows = []
    for a_name, b_name, label in comparisons:
        for metric_key in ["APFD", "APFDc"]:
            res = paired_bootstrap_diff(y, dur, methods[a_name], methods[b_name], metric_key)
            res["comparison"] = label
            res["method_a"] = a_name
            res["method_b"] = b_name
            stat_rows.append(res)
            print(f"  [{label}] {metric_key}: diff={res['point_diff']:.4f} "
                  f"CI=[{res['ci_lo']:.4f},{res['ci_hi']:.4f}] p={res['bootstrap_p_value']:.4f} "
                  f"sig={res['significant_95ci']}", file=sys.stderr)

    with open("results/final_statistical_tests.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["comparison", "method_a", "method_b", "metric", "point_diff", "ci_lo", "ci_hi",
                      "bootstrap_p_value", "effect_size_bootstrap_d", "significant_95ci"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in stat_rows:
            w.writerow({k: row[k] for k in fieldnames})
    print("Wrote results/final_statistical_tests.csv", file=sys.stderr)

    # ---------------- complementarity analysis ----------------
    print("\nComplementarity analysis (KSERESNET vs Qwen3-8B)...", file=sys.stderr)
    fail_ids = set(i for i, yi in zip(test_ids, y) if yi == 1)
    comp_by_budget = {}
    for budget in [500, 1000, 2000]:
        r_order = np.argsort(-r_raw, kind="stable")[:budget]
        l_order = np.argsort(-l_raw, kind="stable")[:budget]
        r_top_ids = set(test_ids[i] for i in r_order)
        l_top_ids = set(test_ids[i] for i in l_order)
        rs_order = np.argsort(-rs_scores, kind="stable")[:budget]
        rs_top_ids = set(test_ids[i] for i in rs_order)

        r_fail_hits = r_top_ids & fail_ids
        l_fail_hits = l_top_ids & fail_ids
        shared = r_fail_hits & l_fail_hits
        r_only = r_fail_hits - l_fail_hits
        l_only = l_fail_hits - r_fail_hits
        union = r_fail_hits | l_fail_hits
        jaccard = len(shared) / len(union) if union else 0.0

        rs_fail_hits = rs_top_ids & fail_ids
        # does fusion capture the L-only failures (KSERESNET's blind spots that Qwen catches)?
        rs_captures_l_only = l_only & rs_fail_hits
        rs_captures_r_only = r_only & rs_fail_hits

        comp_by_budget[str(budget)] = {
            "budget": budget,
            "R_top_fail_hits": len(r_fail_hits),
            "L_top_fail_hits": len(l_fail_hits),
            "RS_top_fail_hits": len(rs_fail_hits),
            "shared_failures": len(shared),
            "KSERESNET_only_failures": len(r_only),
            "Qwen_only_failures": len(l_only),
            "union_failures": len(union),
            "jaccard_overlap": jaccard,
            "RS_captures_of_Qwen_only": len(rs_captures_l_only),
            "RS_captures_of_Qwen_only_frac": len(rs_captures_l_only) / len(l_only) if l_only else None,
            "RS_captures_of_KSERESNET_only": len(rs_captures_r_only),
            "RS_captures_of_KSERESNET_only_frac": len(rs_captures_r_only) / len(r_only) if r_only else None,
        }
        print(f"  budget={budget}: shared={len(shared)} R-only={len(r_only)} L-only={len(l_only)} "
              f"jaccard={jaccard:.4f} RS captures {len(rs_captures_l_only)}/{len(l_only)} of Qwen-only fails",
              file=sys.stderr)

    with open("results/final_complementarity.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_test": len(test_ids), "n_fail_test": n_fail,
            "by_budget": comp_by_budget,
        }, f, indent=2)
    print("Wrote results/final_complementarity.json", file=sys.stderr)

    # ---------------- cost reporting (scenario execution vs LLM inference, kept separate) ----------------
    with open("summary_qwen3_8b_test_full.json", "r", encoding="utf-8") as f:
        llm_cost_summary = json.load(f)
    scenario_exec_cost_total = float(dur.sum())
    cost_report = {
        "scenario_execution_cost_total_sec": scenario_exec_cost_total,
        "scenario_execution_cost_mean_sec": float(dur.mean()),
        "note": "scenario execution cost (duration field) used for APFDc; LLM inference cost reported SEPARATELY below, never mixed into APFDc",
        "llm_inference_cost": {
            "model": llm_cost_summary["model"],
            "total_requests": llm_cost_summary["total_requests"],
            "mean_latency_sec": llm_cost_summary["mean_latency_sec"],
            "median_latency_sec": llm_cost_summary["median_latency_sec"],
            "total_runtime_sec": llm_cost_summary["total_runtime_sec"],
            "scenarios_per_minute": llm_cost_summary["scenarios_per_minute"],
        },
    }
    with open("results/final_cost_report.json", "w", encoding="utf-8") as f:
        json.dump(cost_report, f, indent=2)
    print("Wrote results/final_cost_report.json", file=sys.stderr)

    # ---------------- save raw arrays for plotting script ----------------
    np.savez("results/final_test_arrays.npz",
              y=y, dur=dur, r_raw=r_raw, l_raw=l_raw, g_raw=g_raw,
              random_scores=random_scores, rs_scores=rs_scores,
              test_ids=np.array(test_ids, dtype=object))
    print("Wrote results/final_test_arrays.npz", file=sys.stderr)

    print("\n=== FINAL TEST EVALUATION COMPLETE ===", file=sys.stderr)


if __name__ == "__main__":
    main()
