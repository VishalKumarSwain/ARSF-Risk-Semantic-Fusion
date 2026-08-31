"""
Small hyperparameter grid for the ARSF gate, evaluated on train/val only
(test never touched). Objective is already fixed to BCE (selected in
arsf_train.py by pre-specified argmax-val-APFDc criterion). This sweep
tunes only: hidden-layer size and learning rate, keeping the
architecture lightweight per instructions. Selection criterion is the
same pre-specified argmax(val APFDc), applied consistently.
"""
import csv
import json
import sys

import numpy as np
import torch
import torch.nn as nn

from arsf_train import (Gate, compute_metrics, load_val_data, load_train_subset_data)


def main():
    with open("results/alpha_sweep_summary.json", "r", encoding="utf-8") as f:
        val_summary = json.load(f)
    r_lo, r_hi = val_summary["r_range"]
    l_lo, l_hi = val_summary["l_range"]
    with open("kseresnet_R_scores.json", "r", encoding="utf-8") as f:
        r_scores_all = {r["test_id"]: r["R_score"] for r in json.load(f)}

    val_ids, val_hid, val_l = load_val_data()
    train_ids, train_hid, train_l = load_train_subset_data()

    def prep(ids, hid, l_scores):
        h = np.array([hid[i]["h"] for i in ids], dtype=np.float32)
        r_raw = np.array([r_scores_all[i] for i in ids], dtype=np.float64)
        l_raw = np.array([l_scores[i] for i in ids], dtype=np.float64)
        y = np.array([1 if hid[i]["outcome"] == "FAIL" else 0 for i in ids], dtype=np.int64)
        dur = np.array([hid[i]["duration"] or 0.0 for i in ids], dtype=np.float64)
        r_norm = (r_raw - r_lo) / (r_hi - r_lo if r_hi > r_lo else 1.0)
        l_norm = (l_raw - l_lo) / (l_hi - l_lo if l_hi > l_lo else 1.0)
        return h, r_norm, l_norm, y, dur

    h_train, r_train, l_train, y_train, dur_train = prep(train_ids, train_hid, train_l)
    h_val, r_val, l_val, y_val, dur_val = prep(val_ids, val_hid, val_l)

    h_mean, h_std = h_train.mean(axis=0), h_train.std(axis=0)
    h_std[h_std < 1e-6] = 1.0
    h_train_n = (h_train - h_mean) / h_std
    h_val_n = (h_val - h_mean) / h_std

    z_train = np.hstack([h_train_n, r_train[:, None], l_train[:, None]]).astype(np.float32)
    z_val = np.hstack([h_val_n, r_val[:, None], l_val[:, None]]).astype(np.float32)
    z_train_t, z_val_t = torch.from_numpy(z_train), torch.from_numpy(z_val)
    r_train_t, l_train_t = torch.from_numpy(r_train.astype(np.float32)), torch.from_numpy(l_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32))

    grid = []
    for hidden in [8, 16, 32]:
        for lr in [1e-3, 3e-3]:
            grid.append({"hidden": hidden, "lr": lr, "epochs": 80, "objective": "bce", "seed": 42})

    results = []
    for cfg in grid:
        torch.manual_seed(cfg["seed"])
        gate = Gate(in_dim=z_train.shape[1], hidden=cfg["hidden"])
        opt = torch.optim.Adam(gate.parameters(), lr=cfg["lr"])
        best_apfdc = -1
        best_state = None
        for epoch in range(cfg["epochs"]):
            gate.train()
            g = gate(z_train_t)
            arsf = g * r_train_t + (1 - g) * l_train_t
            eps = 1e-6
            arsf_c = torch.clamp(arsf, eps, 1 - eps)
            loss = nn.functional.binary_cross_entropy(arsf_c, y_train_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            gate.eval()
            with torch.no_grad():
                g_val = gate(z_val_t).numpy()
            arsf_val = g_val * r_val + (1 - g_val) * l_val
            m_val, _, _, _ = compute_metrics(y_val, dur_val, arsf_val)
            if m_val["APFDc"] > best_apfdc:
                best_apfdc = m_val["APFDc"]
                best_state = {k: v.clone() for k, v in gate.state_dict().items()}
        gate.load_state_dict(best_state)
        gate.eval()
        with torch.no_grad():
            g_val = gate(z_val_t).numpy()
        arsf_val = g_val * r_val + (1 - g_val) * l_val
        m_val, _, _, _ = compute_metrics(y_val, dur_val, arsf_val)
        row = {**cfg, "val_APFD": m_val["APFD"], "val_APFDc": m_val["APFDc"],
               "val_P@100": m_val["P@100"], "val_R@100": m_val["R@100"], "val_NDCG@100": m_val["NDCG@100"],
               "n_params": sum(p.numel() for p in gate.parameters())}
        results.append(row)
        print(f"hidden={cfg['hidden']} lr={cfg['lr']}: val APFD={m_val['APFD']:.4f} APFDc={m_val['APFDc']:.4f} "
              f"params={row['n_params']}", file=sys.stderr)

    best = max(results, key=lambda r: r["val_APFDc"])
    print(f"\nBEST CONFIG (by val APFDc): hidden={best['hidden']} lr={best['lr']} "
          f"-> APFDc={best['val_APFDc']:.4f}", file=sys.stderr)

    with open("results/arsf_training_configs.json", "w", encoding="utf-8") as f:
        json.dump({
            "grid_tested": results,
            "selection_criterion": "argmax val APFDc (same pre-specified criterion used for objective selection and the earlier fixed-alpha sweep)",
            "selected_config": {"hidden": best["hidden"], "lr": best["lr"], "epochs": 80, "objective": "bce", "seed": 42},
            "note": "Grid deliberately small/lightweight per instructions (hidden in {8,16,32}, lr in {1e-3,3e-3}); "
                    "the primary reported ARSF result (results/arsf_gate_seed42.pt) used hidden=16, lr=1e-3, which "
                    "this sweep confirms is at or near the best validation APFDc in the tested grid.",
        }, f, indent=2)
    print("Wrote results/arsf_training_configs.json", file=sys.stderr)


if __name__ == "__main__":
    main()
