"""
Alpha sweep for KSERESNET-RS fusion: RS(s) = alpha*R'(s) + (1-alpha)*L'(s)

FROZEN inputs (not modified by this script):
  - Qwen3-8B L-scores: results_rankings_qwen3_8b_val_full.csv
  - Corrected KSERESNET R-scores: kseresnet_R_scores.json
  - GBDT G-scores: gbdt_G_scores.json (baseline B4, not fused)

Validation split ONLY (split_val_ids.json). Test split is never touched
here.

Normalization procedure (pre-specified, fit on validation split only):
  min-max scaling to [0,1]: x' = (x - min(x)) / (max(x) - min(x))
  applied independently to R and L using val-split min/max.

alpha grid: 0.00, 0.05, ..., 1.00 (21 points)

Metrics per alpha: APFD, APFDc, P/R/NDCG @ {100,500,1000}, with
bootstrap 95% CIs (2000 resamples of the scenario set, percentile method).

Also computes Random and KSERESNET-alone / Qwen-alone / GBDT-alone as
reference rows (alpha-independent) for the incremental-improvement report.

Primary alpha-selection criterion (PRE-SPECIFIED, declared before viewing
results): the alpha maximizing mean APFDc on the validation split, since
APFDc accounts for both fault count and time-to-detect and is the
protocol's primary metric (matches organizer evaluation). APFD is
reported as the secondary/tie-breaker criterion.
"""
import csv
import json
import sys

import numpy as np

np.random.seed(42)

N_BOOTSTRAP = 2000
ALPHAS = [round(i * 0.05, 2) for i in range(21)]
K_LIST = [100, 500, 1000]


def load_r_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {r["test_id"]: r["R_score"] for r in d}, {r["test_id"]: r["outcome"] for r in d}, {r["test_id"]: r.get("duration") for r in d}


def load_l_scores(path):
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[row["scenario_id"]] = float(row["score"])
    return scores


def load_g_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {r["test_id"]: r["G_score"] for r in d if r["split"] == "val"}


def minmax_fit(values, lo=None, hi=None):
    lo = values.min() if lo is None else lo
    hi = values.max() if hi is None else hi
    rng = hi - lo if hi > lo else 1.0
    return (values - lo) / rng, (float(lo), float(hi))


# ---- vectorized metric computation over arrays y (0/1), dur (float), score (float) ----

def apfd_vec(y_sorted):
    n = len(y_sorted)
    m = y_sorted.sum()
    if m == 0:
        return None
    fail_pos = np.nonzero(y_sorted)[0] + 1
    return 1 - (fail_pos.sum() / (n * m)) + 1 / (2 * n)


def apfdc_vec(y_sorted, dur_sorted):
    n = len(y_sorted)
    m = y_sorted.sum()
    total_cost = dur_sorted.sum()
    if m == 0 or total_cost <= 0:
        return None
    suffix_sum = np.cumsum(dur_sorted[::-1])[::-1]  # cost from position i to end (inclusive)
    fail_idx = np.nonzero(y_sorted)[0]
    cost_after = suffix_sum[fail_idx] - dur_sorted[fail_idx] / 2.0
    return cost_after.sum() / (total_cost * m)


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
    rank_ttf = int(first) + 1
    time_ttf = float(dur_sorted[:first + 1].sum())
    return rank_ttf, time_ttf


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
    return result, y_sorted, dur_sorted


def bootstrap_ci(y, dur, score, n_boot=N_BOOTSTRAP):
    n = len(y)
    keys = ["APFD", "APFDc"] + [f"{m}@{k}" for m in ("P", "R", "NDCG") for k in K_LIST]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        m, _, _ = compute_metrics(y[idx], dur[idx], score[idx])
        for k in keys:
            if m[k] is not None:
                samples[k].append(m[k])
    ci = {}
    for k, vals in samples.items():
        if len(vals) < 10:
            ci[k] = (None, None)
        else:
            arr = np.array(vals)
            ci[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return ci


def failure_discovery_curve(y_sorted, n_points=20):
    n = len(y_sorted)
    total_fail = int(y_sorted.sum())
    cum = np.cumsum(y_sorted)
    xs = np.clip(np.linspace(0, n, n_points + 1)[1:].astype(int), 1, n)
    return [(int(x), int(cum[x - 1]), float(cum[x - 1] / total_fail) if total_fail else 0.0) for x in xs]


def main():
    print("Loading frozen inputs...", file=sys.stderr)
    r_all, outcome_all, duration_all = load_r_scores("kseresnet_R_scores.json")
    l_all = load_l_scores("results_rankings_qwen3_8b_val_full.csv")
    g_all = load_g_scores("gbdt_G_scores.json")

    with open("split_val_ids.json", "r", encoding="utf-8") as f:
        val_ids_raw = json.load(f)

    val_ids = [i for i in val_ids_raw if i in r_all and i in l_all and i in g_all]
    print(f"Validation set (all sources aligned): n={len(val_ids)}", file=sys.stderr)

    y = np.array([1 if outcome_all[i] == "FAIL" else 0 for i in val_ids], dtype=np.int64)
    dur = np.array([duration_all[i] or 0.0 for i in val_ids], dtype=np.float64)
    r_raw = np.array([r_all[i] for i in val_ids], dtype=np.float64)
    l_raw = np.array([l_all[i] for i in val_ids], dtype=np.float64)
    g_raw = np.array([g_all[i] for i in val_ids], dtype=np.float64)

    n_fail = int(y.sum())
    print(f"FAIL count in validation: {n_fail} / {len(val_ids)}", file=sys.stderr)

    r_norm, r_range = minmax_fit(r_raw)
    l_norm, l_range = minmax_fit(l_raw)
    print(f"R range: {r_range}, L range: {l_range}", file=sys.stderr)

    rng = np.random.RandomState(42)
    random_scores = rng.random(len(val_ids))

    baselines = {}
    for name, scores in [
        ("Random", random_scores),
        ("KSERESNET_alone", r_raw),
        ("Qwen3_8B_alone", l_raw),
        ("GBDT_alone", g_raw),
    ]:
        m, y_sorted, dur_sorted = compute_metrics(y, dur, scores)
        ci = bootstrap_ci(y, dur, scores)
        baselines[name] = {"metrics": m, "ci": ci}
        print(f"[baseline] {name}: APFD={m['APFD']:.4f} APFDc={m['APFDc']:.4f}", file=sys.stderr)

    rows = []
    fdc_by_alpha = {}
    for alpha in ALPHAS:
        rs_scores = alpha * r_norm + (1 - alpha) * l_norm
        m, y_sorted, dur_sorted = compute_metrics(y, dur, rs_scores)
        ci = bootstrap_ci(y, dur, rs_scores)
        fdc_by_alpha[alpha] = failure_discovery_curve(y_sorted)

        row = {"alpha": alpha}
        for key in ["APFD", "APFDc"] + [f"{m2}@{k}" for m2 in ("P", "R", "NDCG") for k in K_LIST]:
            row[key] = m[key]
            lo, hi = ci[key]
            row[f"{key}_CI_lo"] = lo
            row[f"{key}_CI_hi"] = hi
        row["rank_TTF"] = m["rank_TTF"]
        row["time_TTF"] = m["time_TTF"]
        rows.append(row)
        print(f"alpha={alpha:.2f}  APFD={m['APFD']:.4f} [{ci['APFD'][0]:.4f},{ci['APFD'][1]:.4f}]  "
              f"APFDc={m['APFDc']:.4f} [{ci['APFDc'][0]:.4f},{ci['APFDc'][1]:.4f}]  "
              f"P@100={m['P@100']:.4f} R@100={m['R@100']:.4f} NDCG@100={m['NDCG@100']:.4f}",
              file=sys.stderr)

    fieldnames = list(rows[0].keys())
    with open("results/alpha_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("Wrote results/alpha_sweep.csv", file=sys.stderr)

    best_by_apfdc = max(rows, key=lambda r: r["APFDc"])
    best_by_apfd = max(rows, key=lambda r: r["APFD"])
    best_by_p100 = max(rows, key=lambda r: r["P@100"])
    best_by_r100 = max(rows, key=lambda r: r["R@100"])
    best_by_ndcg100 = max(rows, key=lambda r: r["NDCG@100"])

    alpha_star = best_by_apfdc["alpha"]

    with open("results/alpha_sweep_baselines.json", "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)

    with open("results/alpha_sweep_fdc.json", "w", encoding="utf-8") as f:
        json.dump({str(a): fdc_by_alpha[a] for a in ALPHAS}, f, indent=2)

    summary = {
        "alpha_star": alpha_star,
        "selection_criterion": "argmax mean APFDc on validation split (pre-specified)",
        "best_by_apfdc": best_by_apfdc,
        "best_by_apfd": best_by_apfd,
        "best_by_p100": best_by_p100,
        "best_by_r100": best_by_r100,
        "best_by_ndcg100": best_by_ndcg100,
        "baselines": baselines,
        "r_range": r_range,
        "l_range": l_range,
        "n_val": len(val_ids),
        "n_fail_val": n_fail,
    }
    with open("results/alpha_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== ALPHA* SELECTED: {alpha_star} (by pre-specified argmax-APFDc criterion) ===", file=sys.stderr)
    print(f"best_by_apfd alpha = {best_by_apfd['alpha']}", file=sys.stderr)
    print(f"best_by_p100 alpha = {best_by_p100['alpha']}", file=sys.stderr)
    print(f"best_by_r100 alpha = {best_by_r100['alpha']}", file=sys.stderr)
    print(f"best_by_ndcg100 alpha = {best_by_ndcg100['alpha']}", file=sys.stderr)


if __name__ == "__main__":
    main()
