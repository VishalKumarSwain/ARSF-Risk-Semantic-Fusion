"""
ARSF ablation, gate analysis, complementarity analysis, and figures.
Runs entirely on the validation split (test untouched). Depends on
arsf_train.py having been run first (produces results/arsf_val_arrays.npz
and results/arsf_gate_values.csv).
"""
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_LIST = [100, 500, 1000]


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


def bootstrap_ci(y, dur, score, n_boot=2000, seed=42):
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


def paired_bootstrap_diff(y, dur, score_a, score_b, metric_key, n_boot=2000, seed=42):
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
    arrs = np.load("results/arsf_val_arrays.npz", allow_pickle=True)
    y, dur = arrs["y"], arrs["dur"]
    r_val, l_val = arrs["r_val"], arrs["l_val"]
    g_val, arsf_val = arrs["g_val"], arrs["arsf_val"]
    val_ids = arrs["test_ids"]

    g_all = {}
    with open("gbdt_G_scores.json", "r", encoding="utf-8") as f:
        for rec in json.load(f):
            if rec["split"] == "val":
                g_all[rec["test_id"]] = rec["G_score"]
    gbdt_val = np.array([g_all[i] for i in val_ids], dtype=np.float64)

    rng = np.random.RandomState(42)
    random_val = rng.random(len(val_ids))

    fixed_fusion_val = 0.40 * r_val + 0.60 * l_val

    methods = {
        "Random": random_val,
        "KSERESNET": r_val,
        "Qwen3_8B": l_val,
        "GBDT": gbdt_val,
        "Fixed_Fusion_alpha0.40": fixed_fusion_val,
        "ARSF": arsf_val,
    }

    # -------- ablation table with bootstrap CIs --------
    print("Computing ablation metrics + bootstrap CIs...")
    keys = ["APFD", "APFDc"] + [f"{m}@{k}" for m in ("P", "R", "NDCG") for k in K_LIST]
    rows = []
    for name, scores in methods.items():
        m, _, _, _ = compute_metrics(y, dur, scores)
        ci = bootstrap_ci(y, dur, scores)
        row = {"method": name}
        for k in keys:
            row[k] = m[k]
            row[f"{k}_CI_lo"], row[f"{k}_CI_hi"] = ci[k]
        row["rank_TTF"] = m["rank_TTF"]
        row["time_TTF"] = m["time_TTF"]
        rows.append(row)
        print(f"  {name}: APFD={m['APFD']:.4f} APFDc={m['APFDc']:.4f}")

    with open("results/arsf_ablation.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method"] + [c for k in keys for c in (k, f"{k}_CI_lo", f"{k}_CI_hi")] + ["rank_TTF", "time_TTF"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("Wrote results/arsf_ablation.csv")

    with open("results/arsf_validation_metrics.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method"] + keys + ["rank_TTF", "time_TTF"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fieldnames})
    print("Wrote results/arsf_validation_metrics.csv")

    # -------- KEY COMPARISON: ARSF vs Fixed Fusion --------
    print("\nARSF vs Fixed Fusion paired bootstrap tests...")
    key_stats = []
    for metric_key in ["APFD", "APFDc"]:
        res = paired_bootstrap_diff(y, dur, arsf_val, fixed_fusion_val, metric_key)
        res["metric"] = metric_key
        key_stats.append(res)
        print(f"  {metric_key}: diff={res['point_diff']:.4f} CI=[{res['ci_lo']:.4f},{res['ci_hi']:.4f}] "
              f"p={res['p_value']:.4f} sig={res['significant']}")
    with open("results/arsf_vs_fixed_fusion_stats.json", "w", encoding="utf-8") as f:
        json.dump(key_stats, f, indent=2)

    # -------- gate analysis --------
    print("\nGate analysis...")
    fail_mask = y == 1
    pass_mask = y == 0
    gate_stats = {
        "mean_g": float(g_val.mean()),
        "median_g": float(np.median(g_val)),
        "std_g": float(g_val.std()),
        "min_g": float(g_val.min()),
        "max_g": float(g_val.max()),
        "mean_g_FAIL": float(g_val[fail_mask].mean()),
        "mean_g_PASS": float(g_val[pass_mask].mean()),
        "median_g_FAIL": float(np.median(g_val[fail_mask])),
        "median_g_PASS": float(np.median(g_val[pass_mask])),
        "corr_g_R": float(np.corrcoef(g_val, r_val)[0, 1]),
        "corr_g_L": float(np.corrcoef(g_val, l_val)[0, 1]),
    }

    # quadrant analysis: RANK-based median split (not value-based) on R' and L'.
    # L'(s) is highly discretized (only 8 distinct Qwen output levels, right-skewed:
    # >50% of scenarios share the single most common value), so a value-based
    # "< median" split is degenerate (almost nothing falls strictly below the
    # median value). Rank-based split (top-50% / bottom-50% by rank, ties broken
    # by stable sort order) guarantees two balanced, meaningful groups instead.
    n_val = len(r_val)
    r_rank = np.empty(n_val, dtype=np.int64)
    r_rank[np.argsort(-r_val, kind="stable")] = np.arange(n_val)
    l_rank = np.empty(n_val, dtype=np.int64)
    l_rank[np.argsort(-l_val, kind="stable")] = np.arange(n_val)
    r_high_mask = r_rank < n_val // 2
    l_high_mask = l_rank < n_val // 2
    quadrants = {
        "R_high_L_high": r_high_mask & l_high_mask,
        "R_high_L_low": r_high_mask & ~l_high_mask,
        "R_low_L_high": ~r_high_mask & l_high_mask,
        "R_low_L_low": ~r_high_mask & ~l_high_mask,
    }
    quadrant_g = {name: {"n": int(mask.sum()),
                          "mean_R_prime": float(r_val[mask].mean()),
                          "mean_L_prime": float(l_val[mask].mean()),
                          "mean_g": float(g_val[mask].mean()),
                          "mean_ARSF_score": float(arsf_val[mask].mean()),
                          "fail_rate": float(y[mask].mean())}
                  for name, mask in quadrants.items()}
    gate_stats["quadrant_analysis"] = quadrant_g

    with open("results/arsf_gate_analysis.json", "w", encoding="utf-8") as f:
        json.dump(gate_stats, f, indent=2)
    print(json.dumps(gate_stats, indent=2))

    # -------- complementarity: Fixed Fusion vs ARSF --------
    print("\nComplementarity: Fixed Fusion vs ARSF...")
    fail_ids_set = set(val_ids[fail_mask])
    comp_by_k = {}
    for k in K_LIST:
        fixed_top = set(val_ids[np.argsort(-fixed_fusion_val, kind="stable")[:k]])
        arsf_top = set(val_ids[np.argsort(-arsf_val, kind="stable")[:k]])
        r_top = set(val_ids[np.argsort(-r_val, kind="stable")[:k]])
        l_top = set(val_ids[np.argsort(-l_val, kind="stable")[:k]])
        l_only_fails = (l_top & fail_ids_set) - (r_top & fail_ids_set)

        fixed_fails = fixed_top & fail_ids_set
        arsf_fails = arsf_top & fail_ids_set
        shared = fixed_fails & arsf_fails
        fixed_only = fixed_fails - arsf_fails
        arsf_only = arsf_fails - fixed_fails
        union = fixed_fails | arsf_fails
        jaccard = len(shared) / len(union) if union else 0.0

        r_fails = r_top & fail_ids_set
        recovered_from_qwen_only = l_only_fails & arsf_fails
        retained_from_kseresnet = r_fails & arsf_fails

        comp_by_k[str(k)] = {
            "fixed_fusion_fail_hits": len(fixed_fails), "arsf_fail_hits": len(arsf_fails),
            "shared": len(shared), "fixed_only": len(fixed_only), "arsf_only": len(arsf_only),
            "union": len(union), "jaccard": jaccard,
            "qwen_only_fails_at_k": len(l_only_fails),
            "arsf_recovers_from_qwen_only": len(recovered_from_qwen_only),
            "arsf_recovers_from_qwen_only_frac": len(recovered_from_qwen_only) / len(l_only_fails) if l_only_fails else None,
            "kseresnet_fails_at_k": len(r_fails),
            "arsf_retains_from_kseresnet": len(retained_from_kseresnet),
            "arsf_retains_from_kseresnet_frac": len(retained_from_kseresnet) / len(r_fails) if r_fails else None,
        }
        print(f"  K={k}: fixed={len(fixed_fails)} arsf={len(arsf_fails)} shared={len(shared)} "
              f"jaccard={jaccard:.4f} arsf recovers {len(recovered_from_qwen_only)}/{len(l_only_fails)} qwen-only")

    with open("results/arsf_complementarity.json", "w", encoding="utf-8") as f:
        json.dump(comp_by_k, f, indent=2)
    print("Wrote results/arsf_complementarity.json")

    # -------- figures --------
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
    colors = {"Random": "#9ca3af", "KSERESNET": "#2563eb", "Qwen3_8B": "#f59e0b",
              "GBDT": "#10b981", "Fixed_Fusion_alpha0.40": "#8b5cf6", "ARSF": "#dc2626"}
    row_by_name = {r["method"]: r for r in rows}

    def clip_err(v, lo, hi):
        return max(0.0, v - lo), max(0.0, hi - v)

    for metric, fname, title in [("APFD", "arsf_fig1_apfd.png", "APFD: Fixed Fusion vs ARSF (+ baselines, Validation)"),
                                   ("APFDc", "arsf_fig2_apfdc.png", "APFDc: Fixed Fusion vs ARSF (+ baselines, Validation)")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(methods.keys())
        vals = [row_by_name[n][metric] for n in names]
        errs = [clip_err(row_by_name[n][metric], row_by_name[n][f"{metric}_CI_lo"], row_by_name[n][f"{metric}_CI_hi"]) for n in names]
        los, his = [e[0] for e in errs], [e[1] for e in errs]
        bars = ax.bar(names, vals, yerr=[los, his], capsize=5, color=[colors[n] for n in names], edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.4f}", ha="center", fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) + 0.12)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(f"results/{fname}")
        plt.close()
        print("saved", fname)

    # failure discovery curves
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, scores in methods.items():
        _, order, y_sorted, _ = compute_metrics(y, dur, scores)
        curve = np.cumsum(y_sorted)
        xs = np.arange(1, len(curve) + 1)
        ax.plot(xs, curve, label=name, color=colors[name],
                 linewidth=2.2 if name == "ARSF" else 1.4,
                 linestyle="-" if name != "Random" else "--")
    ax.set_xlabel("Number of Executed Scenarios")
    ax.set_ylabel("Cumulative Unique Failures Discovered")
    ax.set_title("Failure Discovery Curve — Validation (Fixed Fusion vs ARSF)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/arsf_fig3_failure_discovery.png")
    plt.close()
    print("saved arsf_fig3_failure_discovery.png")

    # gate-value distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(g_val, bins=40, color="#2563eb", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[0].axvline(gate_stats["mean_g"], color="red", linestyle="--", label=f"mean={gate_stats['mean_g']:.3f}")
    axes[0].set_xlabel("g(s)"); axes[0].set_ylabel("count"); axes[0].set_title("Gate Value Distribution (all val)")
    axes[0].legend()
    axes[1].hist(g_val[fail_mask], bins=40, alpha=0.6, label="FAIL", color="#dc2626", density=True)
    axes[1].hist(g_val[pass_mask], bins=40, alpha=0.6, label="PASS", color="#2563eb", density=True)
    axes[1].set_xlabel("g(s)"); axes[1].set_ylabel("density"); axes[1].set_title("Gate Value: FAIL vs PASS")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("results/arsf_fig4_gate_distribution.png")
    plt.close()
    print("saved arsf_fig4_gate_distribution.png")

    # gate vs R'/L'
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(r_val, g_val, c=y, cmap="coolwarm", s=6, alpha=0.4)
    axes[0].set_xlabel("R'(s) [normalized KSERESNET]"); axes[0].set_ylabel("g(s)")
    axes[0].set_title(f"Gate vs KSERESNET score (corr={gate_stats['corr_g_R']:.3f})")
    axes[1].scatter(l_val, g_val, c=y, cmap="coolwarm", s=6, alpha=0.4)
    axes[1].set_xlabel("L'(s) [normalized Qwen3-8B]"); axes[1].set_ylabel("g(s)")
    axes[1].set_title(f"Gate vs Qwen3-8B score (corr={gate_stats['corr_g_L']:.3f})")
    plt.tight_layout()
    plt.savefig("results/arsf_fig5_gate_vs_scores.png")
    plt.close()
    print("saved arsf_fig5_gate_vs_scores.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
