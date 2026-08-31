"""
Stage-1 ranking-quality metrics: APFD, APFDc, Precision@K, Recall@K, NDCG@K,
Spearman/Kendall correlation vs KSERESNET, Jaccard complementarity, bootstrap
95% CIs. All established metrics (Rothermel et al. 2001 APFD; Elbaum et al.
2001 APFDc; standard IR Precision/Recall/NDCG@K; standard Spearman/Kendall).
No novel metric is introduced here.

TIE-HANDLING POLICY (identical across all methods):
- Primary sort key: score, descending.
- Tie-break: scenario_id, ascending (deterministic, lexicographic).
- This is a stable, deterministic total order -- no randomness in ranking.
- For Spearman/Kendall correlation, tied scores receive average ranks
  (standard scipy behavior for rankdata / kendalltau with ties).
"""
import argparse
import csv
import json
import sys
import math
import random
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, kendalltau


def load_ranking(path, feat_by_id):
    """path: CSV with columns rank,scenario_id,score,true_outcome (or a JSON
    list of {test_id/scenario_id, score, outcome})."""
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        items = [(d.get("test_id") or d.get("scenario_id"), d["score"]) for d in data]
    else:
        with open(path) as f:
            reader = csv.DictReader(f)
            items = [(r["scenario_id"], float(r["score"])) for r in reader]
    # deterministic tie-break: score desc, scenario_id asc
    items.sort(key=lambda x: (-x[1], x[0]))
    ranked_ids = [tid for tid, _ in items]
    scores = {tid: s for tid, s in items}
    return ranked_ids, scores


def compute_apfd(ranked_ids, outcomes):
    n = len(ranked_ids)
    fail_positions = [i + 1 for i, tid in enumerate(ranked_ids) if outcomes[tid] == "FAIL"]
    m = len(fail_positions)
    if n == 0 or m == 0:
        return None, n, m
    apfd = 1 - (sum(fail_positions) / (n * m)) + (1 / (2 * n))
    return apfd, n, m


def compute_apfdc(ranked_ids, outcomes, costs):
    """Elbaum et al. 2001 cost-cognizant APFD, cost = scenario execution
    duration (test_duration from the SensoData xodr header). NOT LLM
    inference latency -- that would conflate cost of the artifact under
    test with the cost of the scoring method, which is a category error."""
    n = len(ranked_ids)
    cost_list = [costs[tid] for tid in ranked_ids]
    total_cost = sum(cost_list)
    fail_idx = [i for i, tid in enumerate(ranked_ids) if outcomes[tid] == "FAIL"]
    m = len(fail_idx)
    if n == 0 or m == 0 or total_cost == 0:
        return None
    numerator = 0.0
    for i in fail_idx:
        tail_cost = sum(cost_list[i:])
        numerator += tail_cost - 0.5 * cost_list[i]
    return numerator / (total_cost * m)


def precision_recall_at_k(ranked_ids, outcomes, k, total_fail):
    if k > len(ranked_ids):
        return None, None
    topk = ranked_ids[:k]
    fail_in_topk = sum(1 for tid in topk if outcomes[tid] == "FAIL")
    precision = fail_in_topk / k
    recall = fail_in_topk / total_fail if total_fail > 0 else None
    return precision, recall


def ndcg_at_k(ranked_ids, outcomes, k):
    """Binary relevance: FAIL=1, PASS=0. Standard log2 discount."""
    if k > len(ranked_ids):
        return None
    rel = [1 if outcomes[tid] == "FAIL" else 0 for tid in ranked_ids[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal_rel = sorted(rel, reverse=True)
    # ideal ordering uses all relevant docs available globally, capped at k
    n_fail_total = sum(1 for tid in ranked_ids if outcomes[tid] == "FAIL")
    ideal_rel_full = [1] * min(n_fail_total, k) + [0] * max(0, k - n_fail_total)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rel_full))
    if idcg == 0:
        return None
    return dcg / idcg


def bootstrap_ci(ranked_ids, outcomes, metric_fn, n_boot=1000, seed=42, **kwargs):
    """Bootstrap 95% CI by resampling scenarios WITH replacement, recomputing
    the metric on each resample (preserving rank order of resampled items)."""
    rng = random.Random(seed)
    n = len(ranked_ids)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        resampled = [ranked_ids[i] for i in sorted(idx)]  # keep relative rank order
        v = metric_fn(resampled, outcomes, **kwargs)
        if isinstance(v, tuple):
            v = v[0]
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return lo, hi


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--rankings-dir", required=True, help="directory containing rankings CSVs")
    ap.add_argument("--methods", required=True, help="comma-separated method:file pairs, e.g. kseresnet:kseresnet_1000.csv,qwen3_8b:qwen3_8b_1000.csv")
    ap.add_argument("--out-main-csv", required=True)
    ap.add_argument("--out-corr-csv", required=True)
    ap.add_argument("--out-complementarity-json", required=True)
    ap.add_argument("--k-values", default="100,500,1000")
    ap.add_argument("--reference-method", default="kseresnet", help="method to correlate all others against")
    args = ap.parse_args()

    with open(args.features) as f:
        feat_by_id = {r["test_id"]: r for r in json.load(f)}
    outcomes = {tid: r["outcome"] for tid, r in feat_by_id.items()}
    costs = {tid: r["duration"] if r.get("duration") is not None else 1.0 for tid, r in feat_by_id.items()}

    k_values = [int(k) for k in args.k_values.split(",")]

    method_files = dict(p.split(":") for p in args.methods.split(","))
    rankings = {}
    for method, fname in method_files.items():
        path = f"{args.rankings_dir}/{fname}"
        ranked_ids, scores = load_ranking(path, feat_by_id)
        rankings[method] = ranked_ids
        print(f"Loaded {method}: {len(ranked_ids)} scenarios", file=sys.stderr)

    # sanity: same scenario set size and same FAIL count across methods
    sizes = {m: len(r) for m, r in rankings.items()}
    fail_counts = {m: sum(1 for tid in r if outcomes[tid] == "FAIL") for m, r in rankings.items()}
    print(f"Sizes: {sizes}", file=sys.stderr)
    print(f"FAIL counts (should be identical): {fail_counts}", file=sys.stderr)
    if len(set(fail_counts.values())) > 1:
        print("WARNING: FAIL counts differ across methods -- scenario sets are not identical!", file=sys.stderr)

    main_rows = []
    for method, ranked_ids in rankings.items():
        total_fail = sum(1 for tid in ranked_ids if outcomes[tid] == "FAIL")
        apfd, n, m = compute_apfd(ranked_ids, outcomes)
        apfdc = compute_apfdc(ranked_ids, outcomes, costs)
        apfd_lo, apfd_hi = bootstrap_ci(ranked_ids, outcomes, lambda r, o: compute_apfd(r, o)[0])
        apfdc_lo, apfdc_hi = bootstrap_ci(ranked_ids, outcomes, lambda r, o: compute_apfdc(r, o, costs))

        row = {
            "method": method, "n_scenarios": n, "n_fail": m,
            "APFD": round(apfd, 4) if apfd is not None else None,
            "APFD_CI_lo": round(apfd_lo, 4) if apfd_lo is not None else None,
            "APFD_CI_hi": round(apfd_hi, 4) if apfd_hi is not None else None,
            "APFDc": round(apfdc, 4) if apfdc is not None else None,
            "APFDc_CI_lo": round(apfdc_lo, 4) if apfdc_lo is not None else None,
            "APFDc_CI_hi": round(apfdc_hi, 4) if apfdc_hi is not None else None,
        }
        for k in k_values:
            p, r = precision_recall_at_k(ranked_ids, outcomes, k, total_fail)
            ndcg = ndcg_at_k(ranked_ids, outcomes, k)
            p_lo, p_hi = bootstrap_ci(ranked_ids, outcomes,
                                       lambda rk, o: precision_recall_at_k(rk, o, k, total_fail)[0])
            r_lo, r_hi = bootstrap_ci(ranked_ids, outcomes,
                                       lambda rk, o: precision_recall_at_k(rk, o, k, total_fail)[1])
            ndcg_lo, ndcg_hi = bootstrap_ci(ranked_ids, outcomes, lambda rk, o: ndcg_at_k(rk, o, k))
            row[f"Precision@{k}"] = round(p, 4) if p is not None else None
            row[f"Precision@{k}_CI_lo"] = round(p_lo, 4) if p_lo is not None else None
            row[f"Precision@{k}_CI_hi"] = round(p_hi, 4) if p_hi is not None else None
            row[f"Recall@{k}"] = round(r, 4) if r is not None else None
            row[f"Recall@{k}_CI_lo"] = round(r_lo, 4) if r_lo is not None else None
            row[f"Recall@{k}_CI_hi"] = round(r_hi, 4) if r_hi is not None else None
            row[f"NDCG@{k}"] = round(ndcg, 4) if ndcg is not None else None
            row[f"NDCG@{k}_CI_lo"] = round(ndcg_lo, 4) if ndcg_lo is not None else None
            row[f"NDCG@{k}_CI_hi"] = round(ndcg_hi, 4) if ndcg_hi is not None else None
        main_rows.append(row)

    with open(args.out_main_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(main_rows[0].keys()))
        writer.writeheader()
        writer.writerows(main_rows)
    print(f"Wrote main comparison to {args.out_main_csv}", file=sys.stderr)

    # correlation table
    corr_rows = []
    methods_list = list(rankings.keys())
    ref = args.reference_method
    pairs = set()
    if ref in rankings:
        for m in methods_list:
            if m != ref:
                pairs.add((ref, m))
    # also all pairwise among LLMs
    llm_methods = [m for m in methods_list if m not in ("random", "kseresnet", "gbdt")]
    for i in range(len(llm_methods)):
        for j in range(i + 1, len(llm_methods)):
            pairs.add((llm_methods[i], llm_methods[j]))

    for a, b in sorted(pairs):
        ids_a, ids_b = rankings[a], rankings[b]
        common = [tid for tid in ids_a if tid in set(ids_b)]
        rank_a = {tid: i for i, tid in enumerate(ids_a)}
        rank_b = {tid: i for i, tid in enumerate(ids_b)}
        xa = [rank_a[tid] for tid in common]
        xb = [rank_b[tid] for tid in common]
        rho, p_rho = spearmanr(xa, xb)
        tau, p_tau = kendalltau(xa, xb)
        corr_rows.append({
            "method_a": a, "method_b": b,
            "spearman_rho": round(rho, 4), "spearman_pvalue": p_rho,
            "kendall_tau": round(tau, 4), "kendall_pvalue": p_tau,
            "n_common_scenarios": len(common),
        })

    with open(args.out_corr_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        writer.writeheader()
        writer.writerows(corr_rows)
    print(f"Wrote correlation table to {args.out_corr_csv}", file=sys.stderr)

    # complementarity: KSERESNET vs each LLM, at each K
    complementarity = {}
    if "kseresnet" in rankings:
        ks_ranked = rankings["kseresnet"]
        for m in llm_methods:
            llm_ranked = rankings[m]
            complementarity[m] = {}
            for k in k_values:
                topk_ks = set(t for t in ks_ranked[:k] if outcomes[t] == "FAIL")
                topk_llm = set(t for t in llm_ranked[:k] if outcomes[t] == "FAIL")
                inter = topk_ks & topk_llm
                only_ks = topk_ks - topk_llm
                only_llm = topk_llm - topk_ks
                union = topk_ks | topk_llm
                complementarity[m][f"K={k}"] = {
                    "unique_failures_kseresnet_topk": len(topk_ks),
                    "unique_failures_llm_topk": len(topk_llm),
                    "shared_failures": len(inter),
                    "kseresnet_only_failures": len(only_ks),
                    "llm_only_failures": len(only_llm),
                    "union_failures": len(union),
                    "jaccard": round(jaccard(topk_ks, topk_llm), 4) if jaccard(topk_ks, topk_llm) is not None else None,
                }

    with open(args.out_complementarity_json, "w") as f:
        json.dump(complementarity, f, indent=2)
    print(f"Wrote complementarity analysis to {args.out_complementarity_json}", file=sys.stderr)

    print("\n=== SANITY CHECKS ===", file=sys.stderr)
    for row in main_rows:
        apfd_ok = row["APFD"] is None or 0 <= row["APFD"] <= 1
        apfdc_ok = row["APFDc"] is None or 0 <= row["APFDc"] <= 1
        print(f"{row['method']}: APFD in [0,1]={apfd_ok}, APFDc in [0,1]={apfdc_ok}, "
              f"n_fail={row['n_fail']}", file=sys.stderr)


if __name__ == "__main__":
    main()
