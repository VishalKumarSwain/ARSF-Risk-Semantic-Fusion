"""
B4 baseline: GBDT (XGBoost) trained on the IDENTICAL structured feature
schema given to the LLM (see FEATURE_SCHEMA.md). Purpose: rule out
"the LLM only helps because it sees more features" -- if B4 matches the
LLM's effectiveness, the LLM adds nothing beyond what's in the shared
features; if the LLM clearly beats B4, that's evidence for genuine
semantic reasoning value.

`duration` is loaded but NEVER used as a training feature (see schema
doc for why -- it's a post-hoc artifact of the outcome, not a
pre-execution scenario property). It is dropped explicitly below.
"""
import argparse
import json
import sys

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

FEATURE_COLS = [
    "num_points", "road_length", "straight_line_distance", "tortuosity",
    "mean_curvature", "max_curvature", "std_curvature", "sharp_turn_count",
    "bbox_aspect_ratio",
]
GENERATOR_CATS = ["ambiegen", "frenetic", "freneticv"]


def load_features(path):
    with open(path, "r", encoding="utf-8") as f:
        return {r["test_id"]: r for r in json.load(f)}


def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_matrix(ids, feat_by_id):
    X, y, kept_ids = [], [], []
    for tid in ids:
        r = feat_by_id.get(tid)
        if r is None:
            continue
        row = [r[c] for c in FEATURE_COLS]
        onehot = [1.0 if r["generator"] == g else 0.0 for g in GENERATOR_CATS]
        X.append(row + onehot)
        y.append(1 if r["outcome"] == "FAIL" else 0)
        kept_ids.append(tid)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), kept_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--train-ids", required=True)
    ap.add_argument("--val-ids", required=True)
    ap.add_argument("--test-ids", required=True)
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--out-scores", required=True)
    args = ap.parse_args()

    print("Loading features and splits ...", file=sys.stderr)
    feat_by_id = load_features(args.features)
    train_ids = load_ids(args.train_ids)
    val_ids = load_ids(args.val_ids)
    test_ids = load_ids(args.test_ids)

    X_train, y_train, _ = to_matrix(train_ids, feat_by_id)
    X_val, y_val, val_kept = to_matrix(val_ids, feat_by_id)
    X_test, y_test, test_kept = to_matrix(test_ids, feat_by_id)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", file=sys.stderr)
    print(f"Feature columns: {FEATURE_COLS + GENERATOR_CATS}", file=sys.stderr)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    val_auc = roc_auc_score(y_val, val_scores)
    test_auc = roc_auc_score(y_test, test_scores)
    print(f"Validation ROC-AUC: {val_auc:.4f}", file=sys.stderr)
    print(f"Test ROC-AUC: {test_auc:.4f}", file=sys.stderr)

    model.save_model(args.out_model)
    print(f"Saved model to {args.out_model}", file=sys.stderr)

    all_scores = []
    for tid, s in zip(val_kept, val_scores):
        all_scores.append({"test_id": tid, "split": "val", "G_score": float(s),
                            "outcome": feat_by_id[tid]["outcome"], "duration": feat_by_id[tid]["duration"]})
    for tid, s in zip(test_kept, test_scores):
        all_scores.append({"test_id": tid, "split": "test", "G_score": float(s),
                            "outcome": feat_by_id[tid]["outcome"], "duration": feat_by_id[tid]["duration"]})

    with open(args.out_scores, "w", encoding="utf-8") as f:
        json.dump(all_scores, f)
    print(f"Wrote scores to {args.out_scores}", file=sys.stderr)

    # quick APFD sanity check on val and test
    for split_name, ids, scores, y in [("val", val_kept, val_scores, y_val), ("test", test_kept, test_scores, y_test)]:
        order = np.argsort(-scores)
        ranked_y = y[order]
        n = len(ranked_y)
        fail_pos = [i + 1 for i, v in enumerate(ranked_y) if v == 1]
        m = len(fail_pos)
        apfd = 1 - (sum(fail_pos) / (n * m)) + 1 / (2 * n) if m > 0 else None
        print(f"B4 GBDT full-split APFD ({split_name}): {apfd:.4f}" if apfd else f"{split_name}: no failures", file=sys.stderr)


if __name__ == "__main__":
    main()
