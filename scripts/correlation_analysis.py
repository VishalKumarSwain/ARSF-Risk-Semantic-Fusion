"""
Correlation / complementarity analysis between corrected KSERESNET R-scores
(native-density fix) and Qwen3-8B LLM semantic risk scores (L), on the
validation and test splits. Computes Spearman/Kendall rank correlation and
top-k Jaccard overlap between the two rankings, per split.
"""
import csv
import json
import sys
from scipy.stats import spearmanr, kendalltau


def load_r_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {r["test_id"]: r["R_score"] for r in d}, {r["test_id"]: r["outcome"] for r in d}


def load_l_scores(path):
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["scenario_id"]] = float(row["score"])
    return scores


def jaccard_topk(r_scores, l_scores, ids, k_frac=0.1):
    n = len(ids)
    k = max(1, int(n * k_frac))
    r_top = set(sorted(ids, key=lambda i: r_scores[i], reverse=True)[:k])
    l_top = set(sorted(ids, key=lambda i: l_scores[i], reverse=True)[:k])
    inter = len(r_top & l_top)
    union = len(r_top | l_top)
    return inter / union if union else 0.0, k


def analyze(split_name, l_csv_path, r_scores, outcomes):
    l_scores = load_l_scores(l_csv_path)
    ids = [i for i in l_scores if i in r_scores]
    print(f"\n=== {split_name} (n={len(ids)}) ===")
    if len(ids) < len(l_scores):
        print(f"  WARNING: {len(l_scores) - len(ids)} L-scored ids missing from R-scores")

    r_vals = [r_scores[i] for i in ids]
    l_vals = [l_scores[i] for i in ids]

    rho, rho_p = spearmanr(r_vals, l_vals)
    tau, tau_p = kendalltau(r_vals, l_vals)
    print(f"  Spearman rho = {rho:.4f} (p={rho_p:.2e})")
    print(f"  Kendall tau  = {tau:.4f} (p={tau_p:.2e})")

    for frac in (0.05, 0.1, 0.2):
        jac, k = jaccard_topk(r_scores, l_scores, ids, frac)
        print(f"  Top-{int(frac*100)}% (k={k}) Jaccard overlap = {jac:.4f}")

    # disagreement analysis: FAIL scenarios ranked high by one but not the other
    fails = [i for i in ids if outcomes.get(i) == "FAIL"]
    n = len(ids)
    r_rank = {i: rk for rk, i in enumerate(sorted(ids, key=lambda x: r_scores[x], reverse=True))}
    l_rank = {i: rk for rk, i in enumerate(sorted(ids, key=lambda x: l_scores[x], reverse=True))}
    complementary = [i for i in fails if r_rank[i] > n * 0.5 and l_rank[i] < n * 0.2]
    print(f"  FAIL scenarios missed by R (bottom-50%) but caught by L (top-20%): {len(complementary)} / {len(fails)} fails")

    return {
        "split": split_name, "n": len(ids), "spearman": rho, "kendall": tau,
        "n_fail": len(fails), "n_complementary_L_catches": len(complementary),
    }


def main():
    r_scores, outcomes = load_r_scores("kseresnet_R_scores.json")
    print(f"Loaded {len(r_scores)} R-scores (corrected native-density)")

    results = []
    results.append(analyze("VAL", "results_rankings_qwen3_8b_val_full.csv", r_scores, outcomes))
    results.append(analyze("TEST", "results_rankings_qwen3_8b_test_full.csv", r_scores, outcomes))

    with open("correlation_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved correlation_analysis_results.json")


if __name__ == "__main__":
    main()
