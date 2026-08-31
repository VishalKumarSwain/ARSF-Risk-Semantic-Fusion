"""
Build the fixed 1,000-scenario Stage-1 screening set, sampled from the
VALIDATION split only (never touches test) -- per the frozen protocol,
Stage-1 is for model screening/selection, held-out test remains untouched.
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ids", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.val_ids) as f:
        val_ids = json.load(f)
    with open(args.features) as f:
        feat_by_id = {r["test_id"]: r for r in json.load(f)}

    rng = random.Random(args.seed)
    sample = rng.sample(val_ids, min(args.n, len(val_ids)))

    n_fail = sum(1 for tid in sample if feat_by_id[tid]["outcome"] == "FAIL")
    print(f"Sampled {len(sample)} scenarios from validation split (seed={args.seed})")
    print(f"FAIL: {n_fail} ({100*n_fail/len(sample):.1f}%) / PASS: {len(sample)-n_fail}")

    with open(args.out, "w") as f:
        json.dump(sample, f)
    print(f"Wrote to {args.out}")


if __name__ == "__main__":
    main()
