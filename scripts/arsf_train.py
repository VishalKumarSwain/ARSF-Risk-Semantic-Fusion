"""
ARSF (Adaptive Risk-Semantic Fusion): trains a small gating MLP that
produces a per-scenario gate g(s) in [0,1], combining KSERESNET and
Qwen3-8B scores as:

  ARSF(s) = g(s) * R'(s) + (1 - g(s)) * L'(s)
  g(s) = sigmoid(MLP([h(s), R'(s), L'(s)]))

where h(s) is the frozen 64-dim KSERESNET penultimate-layer
representation (see extract_kseresnet_hidden.py).

TRAIN split: 4,000-scenario stratified subset of the 21,605-scenario
train split (frozen Qwen3-8B, same prompt/config, scored once for this
experiment only -- see configs/qwen3_8b_arsf_train_config.json on DGX).
VALIDATION split: full 7,196-scenario validation split (same as used
for the fixed-alpha sweep) -- used ONLY for hyperparameter/objective
selection, never for gradient updates.
TEST split: untouched, not loaded by this script at all.

Two training objectives are compared on train/val (never on test):
  1. BCE: treats ARSF(s) directly as a predicted FAIL-probability,
     BCELoss(ARSF(s), y)
  2. Pairwise ranking (RankNet-style): for sampled (FAIL, PASS) pairs,
     -log(sigmoid(ARSF(fail) - ARSF(pass))) -- directly optimizes
     pairwise ordering, closer to what APFD measures.
The objective is selected by which yields higher validation APFDc
(the same pre-specified primary criterion used in the alpha sweep),
NOT by whichever "looks better" post hoc across many metrics.
"""
import csv
import json
import sys

import numpy as np
import torch
import torch.nn as nn

K_LIST = [100, 500, 1000]


# ---------------- metric functions (numpy, vectorized; mirrors alpha_sweep.py) ----------------

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


# ---------------- gate model ----------------

class Gate(nn.Module):
    """Small MLP gate: [h(64), R'(1), L'(1)] -> sigmoid -> g in [0,1]."""

    def __init__(self, in_dim=66, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return torch.sigmoid(self.net(z)).squeeze(-1)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------- data loading ----------------

def load_val_data():
    with open("kseresnet_hidden_val.json", "r", encoding="utf-8") as f:
        hid = {r["test_id"]: r for r in json.load(f)}
    l_scores = {}
    with open("results_rankings_qwen3_8b_val_full.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            l_scores[row["scenario_id"]] = float(row["score"])
    with open("split_val_ids.json", "r", encoding="utf-8") as f:
        val_ids = json.load(f)
    val_ids = [i for i in val_ids if i in hid and i in l_scores]
    return val_ids, hid, l_scores


def load_train_subset_data():
    with open("kseresnet_hidden_train_subset.json", "r", encoding="utf-8") as f:
        hid = {r["test_id"]: r for r in json.load(f)}
    l_scores = {}
    with open("results_rankings_qwen3_8b_arsf_train.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            l_scores[row["scenario_id"]] = float(row["score"])
    with open("arsf_train_subset_ids.json", "r", encoding="utf-8") as f:
        train_ids = json.load(f)
    train_ids = [i for i in train_ids if i in hid and i in l_scores]
    return train_ids, hid, l_scores


def build_arrays(ids, hid, l_scores, r_lo, r_hi, l_lo, l_hi, h_mean=None, h_std=None):
    h = np.array([hid[i]["h"] for i in ids], dtype=np.float32)
    r_raw = np.array([hid[i]["logit"] for i in ids], dtype=np.float64)
    l_raw = np.array([l_scores[i] for i in ids], dtype=np.float64)
    y = np.array([1 if hid[i]["outcome"] == "FAIL" else 0 for i in ids], dtype=np.int64)
    dur = np.array([hid[i]["duration"] or 0.0 for i in ids], dtype=np.float64)

    # NOTE: r_raw here is the raw model LOGIT (not the rank-derived R_score used
    # elsewhere), because h(s) is extracted via a direct forward pass. For
    # consistency with the frozen R'(s) normalization used throughout the
    # rest of the pipeline (rank-derived R_score, min-max fit on validation),
    # we recompute R'(s) here as rank-derived-equivalent by min-max scaling
    # the RAW R_score (not logit) using the SAME ids -- loaded separately by
    # the caller via kseresnet_R_scores.json. This function only prepares h.
    if h_mean is None:
        h_mean = h.mean(axis=0)
        h_std = h.std(axis=0)
        h_std[h_std < 1e-6] = 1.0
    h_norm = (h - h_mean) / h_std

    return h_norm, r_raw, l_raw, y, dur, h_mean, h_std


def main():
    print("Loading validation-fit frozen normalization range (from alpha_sweep_summary.json)...", file=sys.stderr)
    with open("results/alpha_sweep_summary.json", "r", encoding="utf-8") as f:
        val_summary = json.load(f)
    r_lo, r_hi = val_summary["r_range"]
    l_lo, l_hi = val_summary["l_range"]

    with open("kseresnet_R_scores.json", "r", encoding="utf-8") as f:
        r_scores_all = {r["test_id"]: r["R_score"] for r in json.load(f)}

    print("Loading validation data (h, R_score, L)...", file=sys.stderr)
    val_ids, val_hid, val_l = load_val_data()
    print("Loading train-subset data (h, R_score, L)...", file=sys.stderr)
    train_ids, train_hid, train_l = load_train_subset_data()
    print(f"train n={len(train_ids)}, val n={len(val_ids)}", file=sys.stderr)

    def prep(ids, hid, l_scores):
        h = np.array([hid[i]["h"] for i in ids], dtype=np.float32)
        r_raw = np.array([r_scores_all[i] for i in ids], dtype=np.float64)  # rank-derived R_score, consistent w/ rest of pipeline
        l_raw = np.array([l_scores[i] for i in ids], dtype=np.float64)
        y = np.array([1 if hid[i]["outcome"] == "FAIL" else 0 for i in ids], dtype=np.int64)
        dur = np.array([hid[i]["duration"] or 0.0 for i in ids], dtype=np.float64)
        r_norm = (r_raw - r_lo) / (r_hi - r_lo if r_hi > r_lo else 1.0)
        l_norm = (l_raw - l_lo) / (l_hi - l_lo if l_hi > l_lo else 1.0)
        return h, r_norm, l_norm, y, dur

    h_train, r_train, l_train, y_train, dur_train = prep(train_ids, train_hid, train_l)
    h_val, r_val, l_val, y_val, dur_val = prep(val_ids, val_hid, val_l)

    # h standardization: fit on TRAIN ONLY, frozen, applied to val without refit
    h_mean = h_train.mean(axis=0)
    h_std = h_train.std(axis=0)
    h_std[h_std < 1e-6] = 1.0
    h_train_n = (h_train - h_mean) / h_std
    h_val_n = (h_val - h_mean) / h_std

    print(f"train FAIL={y_train.sum()}/{len(y_train)}  val FAIL={y_val.sum()}/{len(y_val)}", file=sys.stderr)

    z_train = np.hstack([h_train_n, r_train[:, None], l_train[:, None]]).astype(np.float32)
    z_val = np.hstack([h_val_n, r_val[:, None], l_val[:, None]]).astype(np.float32)

    fixed_fusion_val = 0.40 * r_val + 0.60 * l_val
    fixed_m, _, _, _ = compute_metrics(y_val, dur_val, fixed_fusion_val)
    print(f"[reference] Fixed Fusion (alpha=0.40) on val: APFD={fixed_m['APFD']:.4f} APFDc={fixed_m['APFDc']:.4f}", file=sys.stderr)

    z_train_t = torch.from_numpy(z_train)
    z_val_t = torch.from_numpy(z_val)
    r_train_t = torch.from_numpy(r_train.astype(np.float32))
    l_train_t = torch.from_numpy(l_train.astype(np.float32))
    r_val_t = torch.from_numpy(r_val.astype(np.float32))
    l_val_t = torch.from_numpy(l_val.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32))

    def train_gate(objective, seed, epochs=80, lr=3e-3, hidden=32):
        torch.manual_seed(seed)
        np.random.seed(seed)
        gate = Gate(in_dim=z_train.shape[1], hidden=hidden)
        opt = torch.optim.Adam(gate.parameters(), lr=lr)
        fail_idx = np.where(y_train == 1)[0]
        pass_idx = np.where(y_train == 0)[0]
        rng = np.random.RandomState(seed)

        best_val_apfdc = -1
        best_state = None
        history = []
        for epoch in range(epochs):
            gate.train()
            g = gate(z_train_t)
            arsf = g * r_train_t + (1 - g) * l_train_t
            if objective == "bce":
                eps = 1e-6
                arsf_c = torch.clamp(arsf, eps, 1 - eps)
                loss = nn.functional.binary_cross_entropy(arsf_c, y_train_t)
            elif objective == "ranking":
                n_pairs = 2000
                fi = rng.choice(fail_idx, size=n_pairs, replace=True)
                pi = rng.choice(pass_idx, size=n_pairs, replace=True)
                diff = arsf[fi] - arsf[pi]
                loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
            else:
                raise ValueError(objective)
            opt.zero_grad()
            loss.backward()
            opt.step()

            gate.eval()
            with torch.no_grad():
                g_val = gate(z_val_t).numpy()
            arsf_val = g_val * r_val + (1 - g_val) * l_val
            m_val, _, _, _ = compute_metrics(y_val, dur_val, arsf_val)
            history.append({"epoch": epoch, "train_loss": float(loss.item()),
                             "val_APFD": m_val["APFD"], "val_APFDc": m_val["APFDc"]})
            if m_val["APFDc"] > best_val_apfdc:
                best_val_apfdc = m_val["APFDc"]
                best_state = {k: v.clone() for k, v in gate.state_dict().items()}

        gate.load_state_dict(best_state)
        return gate, history, best_val_apfdc

    # --- objective selection: train both, compare on val, pick by APFDc (pre-specified) ---
    print("\n=== Training objective comparison (train/val only, seed=42) ===", file=sys.stderr)
    results_by_obj = {}
    for obj in ["bce", "ranking"]:
        gate, history, best_apfdc = train_gate(obj, seed=42)
        gate.eval()
        with torch.no_grad():
            g_val = gate(z_val_t).numpy()
        arsf_val = g_val * r_val + (1 - g_val) * l_val
        m_val, _, _, _ = compute_metrics(y_val, dur_val, arsf_val)
        results_by_obj[obj] = {"gate": gate, "history": history, "val_metrics": m_val, "g_val": g_val}
        print(f"[{obj}] val APFD={m_val['APFD']:.4f} APFDc={m_val['APFDc']:.4f} "
              f"P@100={m_val['P@100']:.4f} best_epoch_apfdc={best_apfdc:.4f}", file=sys.stderr)

    chosen_obj = max(results_by_obj, key=lambda o: results_by_obj[o]["val_metrics"]["APFDc"])
    print(f"\nCHOSEN OBJECTIVE (by val APFDc, pre-specified criterion): {chosen_obj}", file=sys.stderr)

    with open("results/arsf_objective_comparison.json", "w", encoding="utf-8") as f:
        json.dump({
            "chosen_objective": chosen_obj,
            "selection_criterion": "argmax val APFDc (same pre-specified criterion as the fixed-alpha sweep)",
            "bce_val_metrics": results_by_obj["bce"]["val_metrics"],
            "ranking_val_metrics": results_by_obj["ranking"]["val_metrics"],
            "fixed_fusion_val_metrics": fixed_m,
        }, f, indent=2)

    # --- robustness across seeds (using the chosen objective) ---
    print(f"\n=== Robustness across seeds (objective={chosen_obj}) ===", file=sys.stderr)
    seeds = [42, 123, 456]
    seed_rows = []
    seed_gates = {}
    for seed in seeds:
        gate, history, best_apfdc = train_gate(chosen_obj, seed=seed)
        gate.eval()
        with torch.no_grad():
            g_val = gate(z_val_t).numpy()
        arsf_val = g_val * r_val + (1 - g_val) * l_val
        m_val, _, _, _ = compute_metrics(y_val, dur_val, arsf_val)
        seed_rows.append({"seed": seed, "val_APFD": m_val["APFD"], "val_APFDc": m_val["APFDc"],
                           "val_P@100": m_val["P@100"], "val_R@100": m_val["R@100"], "val_NDCG@100": m_val["NDCG@100"]})
        seed_gates[seed] = (gate, g_val, m_val)
        print(f"  seed={seed}: APFD={m_val['APFD']:.4f} APFDc={m_val['APFDc']:.4f}", file=sys.stderr)

    apfds = [r["val_APFD"] for r in seed_rows]
    apfdcs = [r["val_APFDc"] for r in seed_rows]
    seed_summary = {
        "seeds": seeds,
        "APFD_mean": float(np.mean(apfds)), "APFD_std": float(np.std(apfds)),
        "APFD_range": [float(min(apfds)), float(max(apfds))],
        "APFDc_mean": float(np.mean(apfdcs)), "APFDc_std": float(np.std(apfdcs)),
        "APFDc_range": [float(min(apfdcs)), float(max(apfdcs))],
    }
    with open("results/arsf_seed_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "val_APFD", "val_APFDc", "val_P@100", "val_R@100", "val_NDCG@100"])
        w.writeheader()
        for row in seed_rows:
            w.writerow(row)
    with open("results/arsf_seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(seed_summary, f, indent=2)
    print(f"Seed summary: APFD {seed_summary['APFD_mean']:.4f}+-{seed_summary['APFD_std']:.4f} "
          f"range{seed_summary['APFD_range']}", file=sys.stderr)

    # --- primary/frozen ARSF model = seed 42 (pre-specified default seed used throughout pipeline) ---
    primary_gate, primary_g_val, primary_m_val = seed_gates[42]
    arsf_val_primary = primary_g_val * r_val + (1 - primary_g_val) * l_val

    torch.save(primary_gate.state_dict(), "results/arsf_gate_seed42.pt")
    with open("results/arsf_model_complexity.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_trainable_params": n_params(primary_gate),
            "architecture": "Linear(66,32) -> ReLU -> Linear(32,1) -> Sigmoid",
            "input_dim": z_train.shape[1],
            "hidden_dim": 32,
            "device": "CPU",
            "framework": "PyTorch",
        }, f, indent=2)

    # --- save gate values for every validation scenario ---
    with open("results/arsf_gate_values.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test_id", "g", "R_prime", "L_prime", "ARSF_score", "outcome"])
        w.writeheader()
        for i, tid in enumerate(val_ids):
            w.writerow({"test_id": tid, "g": float(primary_g_val[i]), "R_prime": float(r_val[i]),
                        "L_prime": float(l_val[i]), "ARSF_score": float(arsf_val_primary[i]),
                        "outcome": "FAIL" if y_val[i] == 1 else "PASS"})
    print("Wrote results/arsf_gate_values.csv", file=sys.stderr)

    # save arrays for downstream ablation/complementarity/plotting scripts
    np.savez("results/arsf_val_arrays.npz",
              y=y_val, dur=dur_val, r_val=r_val, l_val=l_val,
              g_val=primary_g_val, arsf_val=arsf_val_primary,
              test_ids=np.array(val_ids, dtype=object))
    print("Wrote results/arsf_val_arrays.npz", file=sys.stderr)

    print(f"\n=== ARSF (seed=42, objective={chosen_obj}) on val: "
          f"APFD={primary_m_val['APFD']:.4f} APFDc={primary_m_val['APFDc']:.4f} ===", file=sys.stderr)
    print(f"vs Fixed Fusion: APFD={fixed_m['APFD']:.4f} APFDc={fixed_m['APFDc']:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
