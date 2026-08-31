"""
FINAL held-out ARSF test evaluation. Test split (n=7205) used ONLY here.
ARSF gate is loaded frozen (results/arsf_gate_seed42.pt, seed=42,
hidden=32, lr=0.003, BCE, 2177 params) -- no training, no tuning, no
architecture changes. Frozen normalization (val-fit R'/L' range) and
frozen h(s) standardization (train-subset-fit mean/std) are reused
verbatim, exactly as documented in results/arsf_final_test_integrity_audit.json.

Preserves the previously completed Fixed-Fusion test result
(APFD=0.6387, APFDc=0.6436) -- recomputed here with identical
methodology for consistency in the combined CSV outputs, verified to
match, never altered.
"""
import csv
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from arsf_train import Gate

K_LIST = [100, 500, 1000]
N_BOOTSTRAP = 2000
RNG_SEED = 42


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
    result = {"APFD": apfd_vec(y_sorted), "APFDc": apfdc_vec(y_sorted, dur_sorted)}
    rank_ttf, time_ttf = ttf_vec(y_sorted, dur_sorted)
    result["rank_TTF"] = rank_ttf
    result["time_TTF"] = time_ttf
    for k in K_LIST:
        p, r, nd = prk_vec(y_sorted, k)
        result[f"P@{k}"] = p
        result[f"R@{k}"] = r
        result[f"NDCG@{k}"] = nd
    return result, order, y_sorted, dur_sorted


def bootstrap_ci(y, dur, score, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(y)
    keys = ["APFD", "APFDc"] + [f"{m}@{k}" for m in ("P", "R", "NDCG") for k in K_LIST]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        m, _, _, _ = compute_metrics(y[idx], dur[idx], score[idx])
        for k in keys:
            if m[k] is not None:
                samples[k].append(m[k])
    ci = {}
    for k, vals in samples.items():
        arr = np.array(vals)
        ci[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) >= 10 else (None, None)
    return ci


def paired_bootstrap_diff(y, dur, score_a, score_b, metric_key, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        ma, _, _, _ = compute_metrics(y[idx], dur[idx], score_a[idx])
        mb, _, _, _ = compute_metrics(y[idx], dur[idx], score_b[idx])
        if ma[metric_key] is not None and mb[metric_key] is not None:
            diffs.append(ma[metric_key] - mb[metric_key])
    diffs = np.array(diffs)
    point = float(np.mean(diffs))
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    p_le0, p_ge0 = float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0))
    p_value = min(2 * min(p_le0, p_ge0), 1.0)
    std = float(np.std(diffs))
    eff = point / std if std > 0 else 0.0
    return {"point_diff": point, "ci_lo": lo, "ci_hi": hi, "p_value": p_value,
            "effect_size": eff, "significant": not (lo <= 0 <= hi)}


def main():
    print("=== FINAL ARSF HELD-OUT TEST EVALUATION (n=7205) ===", file=sys.stderr)

    with open("results/alpha_sweep_summary.json", "r", encoding="utf-8") as f:
        val_summary = json.load(f)
    r_lo, r_hi = val_summary["r_range"]
    l_lo, l_hi = val_summary["l_range"]

    with open("kseresnet_R_scores.json", "r", encoding="utf-8") as f:
        r_scores_all = {r["test_id"]: r["R_score"] for r in json.load(f)}

    l_scores_all = {}
    with open("results_rankings_qwen3_8b_test_full.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            l_scores_all[row["scenario_id"]] = float(row["score"])

    g_scores_all = {}
    with open("gbdt_G_scores.json", "r", encoding="utf-8") as f:
        for rec in json.load(f):
            if rec["split"] == "test":
                g_scores_all[rec["test_id"]] = rec["G_score"]

    h_all = {}
    with open("kseresnet_hidden_test.json", "r", encoding="utf-8") as f:
        for rec in json.load(f):
            h_all[rec["test_id"]] = rec

    with open("split_test_ids.json", "r", encoding="utf-8") as f:
        test_ids_raw = json.load(f)
    test_ids = [i for i in test_ids_raw if i in r_scores_all and i in l_scores_all and i in g_scores_all and i in h_all]
    assert len(test_ids) == 7205, f"Expected 7205, got {len(test_ids)}"
    print(f"n={len(test_ids)}", file=sys.stderr)

    y = np.array([1 if h_all[i]["outcome"] == "FAIL" else 0 for i in test_ids], dtype=np.int64)
    dur = np.array([h_all[i]["duration"] or 0.0 for i in test_ids], dtype=np.float64)
    r_raw = np.array([r_scores_all[i] for i in test_ids], dtype=np.float64)
    l_raw = np.array([l_scores_all[i] for i in test_ids], dtype=np.float64)
    g_raw = np.array([g_scores_all[i] for i in test_ids], dtype=np.float64)
    h_raw = np.array([h_all[i]["h"] for i in test_ids], dtype=np.float32)
    n_fail = int(y.sum())
    print(f"n_fail={n_fail}, n_pass={len(test_ids)-n_fail}", file=sys.stderr)

    r_norm = (r_raw - r_lo) / (r_hi - r_lo if r_hi > r_lo else 1.0)
    l_norm = (l_raw - l_lo) / (l_hi - l_lo if l_hi > l_lo else 1.0)

    # frozen h(s) standardization: recompute from train subset (same procedure as arsf_train.py, deterministic, not refit here)
    with open("kseresnet_hidden_train_subset.json", "r", encoding="utf-8") as f:
        train_hid = {r["test_id"]: r for r in json.load(f)}
    h_train = np.array([train_hid[i]["h"] for i in train_hid], dtype=np.float32)
    h_mean = h_train.mean(axis=0)
    h_std = h_train.std(axis=0)
    h_std[h_std < 1e-6] = 1.0
    h_norm = (h_raw - h_mean) / h_std
    print("Reusing FROZEN h(s) standardization (fit on 4000-scenario train subset)", file=sys.stderr)
    print(f"Reusing FROZEN R'/L' normalization: R range={r_lo,r_hi}, L range={l_lo,l_hi}", file=sys.stderr)

    z_test = np.hstack([h_norm, r_norm[:, None], l_norm[:, None]]).astype(np.float32)

    gate = Gate(in_dim=66, hidden=32)
    gate.load_state_dict(torch.load("results/arsf_gate_seed42.pt", map_location="cpu"))
    gate.eval()
    with torch.no_grad():
        g_test = gate(torch.from_numpy(z_test)).numpy()
    arsf_test = g_test * r_norm + (1 - g_test) * l_norm
    print(f"ARSF gate applied (frozen, no training). mean g(s) on test = {g_test.mean():.4f}", file=sys.stderr)

    rng = np.random.RandomState(RNG_SEED)
    random_test = rng.random(len(test_ids))
    fixed_test = 0.40 * r_norm + 0.60 * l_norm

    methods = {
        "B1_Random": random_test,
        "B2_KSERESNET": r_raw,
        "B3_Qwen3_8B": l_raw,
        "B4_GBDT": g_raw,
        "B5_Fixed_KSERESNET_RS": fixed_test,
        "B6_ARSF": arsf_test,
    }

    all_metrics = {}
    all_orders = {}
    for name, scores in methods.items():
        m, order, y_sorted, dur_sorted = compute_metrics(y, dur, scores)
        all_metrics[name] = m
        all_orders[name] = (order, y_sorted, dur_sorted)
        print(f"{name}: APFD={m['APFD']:.4f} APFDc={m['APFDc']:.4f} "
              f"P@100={m['P@100']:.4f} R@100={m['R@100']:.4f} NDCG@100={m['NDCG@100']:.4f}", file=sys.stderr)

    # sanity check: Fixed Fusion must match previously reported test result
    print(f"\n[SANITY CHECK] Fixed Fusion recomputed: APFD={all_metrics['B5_Fixed_KSERESNET_RS']['APFD']:.4f} "
          f"(expected 0.6387), APFDc={all_metrics['B5_Fixed_KSERESNET_RS']['APFDc']:.4f} (expected 0.6436)", file=sys.stderr)

    keys = ["APFD", "APFDc"] + [f"{m}@{k}" for m in ("P", "R", "NDCG") for k in K_LIST]
    with open("results/arsf_final_test_metrics.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method"] + keys + ["rank_TTF", "time_TTF"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, m in all_metrics.items():
            w.writerow({"method": name, **{k: m[k] for k in keys}, "rank_TTF": m["rank_TTF"], "time_TTF": m["time_TTF"]})
    print("Wrote results/arsf_final_test_metrics.csv", file=sys.stderr)

    print("\nComputing bootstrap CIs per method...", file=sys.stderr)
    all_cis = {}
    for name, scores in methods.items():
        all_cis[name] = bootstrap_ci(y, dur, scores)
        print(f"  {name} done", file=sys.stderr)

    with open("results/arsf_final_test_metrics_with_ci.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method"] + [c for k in keys for c in (k, f"{k}_CI_lo", f"{k}_CI_hi")]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name in methods:
            row = {"method": name}
            for k in keys:
                row[k] = all_metrics[name][k]
                row[f"{k}_CI_lo"], row[f"{k}_CI_hi"] = all_cis[name][k]
            w.writerow(row)
    print("Wrote results/arsf_final_test_metrics_with_ci.csv", file=sys.stderr)

    # comparison table
    with open("results/arsf_final_comparison.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "APFD", "APFDc", "P@100", "R@100", "NDCG@100", "P@500", "R@500", "NDCG@500",
                      "P@1000", "R@1000", "NDCG@1000", "rank_TTF", "time_TTF"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, m in all_metrics.items():
            w.writerow({"method": name, **{k: m[k] for k in fieldnames if k != "method"}})
    print("Wrote results/arsf_final_comparison.csv", file=sys.stderr)

    # failure discovery curves
    budgets = [100, 500, 1000, 2000, 5000]
    fdc_rows = []
    for name in methods:
        _, y_sorted, _ = all_orders[name]
        total_fail = int(y_sorted.sum())
        cum = np.cumsum(y_sorted)
        for b in budgets + [len(test_ids)]:
            b = min(b, len(test_ids))
            label = b if b != len(test_ids) else "full"
            fdc_rows.append({"method": name, "budget": label, "cum_failures": int(cum[b - 1]),
                              "frac_failures_found": float(cum[b - 1] / total_fail) if total_fail else 0.0})
    with open("results/arsf_final_failure_discovery_curves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "budget", "cum_failures", "frac_failures_found"])
        w.writeheader()
        for row in fdc_rows:
            w.writerow(row)
    print("Wrote results/arsf_final_failure_discovery_curves.csv", file=sys.stderr)

    full_curves = {name: np.cumsum(all_orders[name][1]).tolist() for name in methods}
    with open("results/arsf_final_failure_discovery_curves_full.json", "w", encoding="utf-8") as f:
        json.dump(full_curves, f)
    print("Wrote results/arsf_final_failure_discovery_curves_full.json", file=sys.stderr)

    # statistical tests: primary hypotheses
    print("\nPaired bootstrap significance tests...", file=sys.stderr)
    comparisons = [
        ("B6_ARSF", "B5_Fixed_KSERESNET_RS", "PRIMARY: ARSF vs Fixed KSERESNET-RS"),
        ("B6_ARSF", "B2_KSERESNET", "SECONDARY: ARSF vs KSERESNET"),
        ("B6_ARSF", "B4_GBDT", "TERTIARY: ARSF vs GBDT"),
        ("B6_ARSF", "B3_Qwen3_8B", "additional: ARSF vs Qwen3-8B"),
        ("B6_ARSF", "B1_Random", "additional: ARSF vs Random"),
    ]
    stat_rows = []
    for a_name, b_name, label in comparisons:
        for metric_key in ["APFD", "APFDc"]:
            res = paired_bootstrap_diff(y, dur, methods[a_name], methods[b_name], metric_key)
            res.update({"comparison": label, "method_a": a_name, "method_b": b_name, "metric": metric_key})
            stat_rows.append(res)
            print(f"  [{label}] {metric_key}: diff={res['point_diff']:.4f} "
                  f"CI=[{res['ci_lo']:.4f},{res['ci_hi']:.4f}] p={res['p_value']:.4f} sig={res['significant']}",
                  file=sys.stderr)
    with open("results/arsf_final_statistical_tests.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["comparison", "method_a", "method_b", "metric", "point_diff", "ci_lo", "ci_hi",
                      "p_value", "effect_size", "significant"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in stat_rows:
            w.writerow({k: row[k] for k in fieldnames})
    print("Wrote results/arsf_final_statistical_tests.csv", file=sys.stderr)

    # complementarity on test
    print("\nComplementarity analysis on TEST...", file=sys.stderr)
    fail_ids_set = set(i for i, yi in zip(test_ids, y) if yi == 1)
    comp_by_k = {}
    for k in K_LIST:
        r_top = set(np.array(test_ids)[np.argsort(-r_raw, kind="stable")[:k]])
        l_top = set(np.array(test_ids)[np.argsort(-l_raw, kind="stable")[:k]])
        fixed_top = set(np.array(test_ids)[np.argsort(-fixed_test, kind="stable")[:k]])
        arsf_top = set(np.array(test_ids)[np.argsort(-arsf_test, kind="stable")[:k]])

        r_fails = r_top & fail_ids_set
        l_fails = l_top & fail_ids_set
        shared_rl = r_fails & l_fails
        r_only = r_fails - l_fails
        l_only = l_fails - r_fails
        union_rl = r_fails | l_fails
        jaccard_rl = len(shared_rl) / len(union_rl) if union_rl else 0.0

        fixed_fails = fixed_top & fail_ids_set
        arsf_fails = arsf_top & fail_ids_set

        arsf_recovers_l_only = l_only & arsf_fails
        arsf_retains_r = r_fails & arsf_fails
        fixed_recovers_l_only = l_only & fixed_fails

        comp_by_k[str(k)] = {
            "budget": k,
            "KSERESNET_top_fail_hits": len(r_fails),
            "Qwen_top_fail_hits": len(l_fails),
            "shared_R_L": len(shared_rl),
            "KSERESNET_only": len(r_only),
            "Qwen_only": len(l_only),
            "union_R_L": len(union_rl),
            "jaccard_R_L": jaccard_rl,
            "Fixed_Fusion_fail_hits": len(fixed_fails),
            "ARSF_fail_hits": len(arsf_fails),
            "ARSF_recovers_Qwen_only": len(arsf_recovers_l_only),
            "ARSF_recovers_Qwen_only_frac": len(arsf_recovers_l_only) / len(l_only) if l_only else None,
            "Fixed_Fusion_recovers_Qwen_only": len(fixed_recovers_l_only),
            "Fixed_Fusion_recovers_Qwen_only_frac": len(fixed_recovers_l_only) / len(l_only) if l_only else None,
            "ARSF_retains_KSERESNET": len(arsf_retains_r),
            "ARSF_retains_KSERESNET_frac": len(arsf_retains_r) / len(r_fails) if r_fails else None,
            "net_additional_fails_ARSF_vs_Fixed": len(arsf_fails) - len(fixed_fails),
        }
        print(f"  K={k}: R_only={len(r_only)} L_only={len(l_only)} jaccard={jaccard_rl:.4f} "
              f"Fixed_hits={len(fixed_fails)} ARSF_hits={len(arsf_fails)} "
              f"ARSF recovers {len(arsf_recovers_l_only)}/{len(l_only)} Qwen-only "
              f"(Fixed recovered {len(fixed_recovers_l_only)}/{len(l_only)})", file=sys.stderr)

    with open("results/arsf_final_complementarity.json", "w", encoding="utf-8") as f:
        json.dump({"n_test": len(test_ids), "n_fail_test": n_fail, "by_budget": comp_by_k}, f, indent=2)
    print("Wrote results/arsf_final_complementarity.json", file=sys.stderr)

    # gate stats on test (for generalization/interpretation section, not used to alter anything)
    fail_mask, pass_mask = (y == 1), (y == 0)
    gate_test_stats = {
        "mean_g": float(g_test.mean()), "median_g": float(np.median(g_test)), "std_g": float(g_test.std()),
        "min_g": float(g_test.min()), "max_g": float(g_test.max()),
        "mean_g_FAIL": float(g_test[fail_mask].mean()), "mean_g_PASS": float(g_test[pass_mask].mean()),
        "corr_g_R": float(np.corrcoef(g_test, r_norm)[0, 1]), "corr_g_L": float(np.corrcoef(g_test, l_norm)[0, 1]),
    }
    with open("results/arsf_final_gate_stats_test.json", "w", encoding="utf-8") as f:
        json.dump(gate_test_stats, f, indent=2)
    print("Gate stats on test:", json.dumps(gate_test_stats, indent=2), file=sys.stderr)

    # save arrays for figures
    np.savez("results/arsf_final_test_arrays.npz",
              y=y, dur=dur, r_raw=r_raw, l_raw=l_raw, g_raw=g_raw,
              random_test=random_test, fixed_test=fixed_test, arsf_test=arsf_test, g_test=g_test,
              test_ids=np.array(test_ids, dtype=object))
    print("Wrote results/arsf_final_test_arrays.npz", file=sys.stderr)

    print("\n=== ARSF FINAL TEST EVALUATION COMPLETE ===", file=sys.stderr)


if __name__ == "__main__":
    main()
