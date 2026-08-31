"""
1. Audit overlap between the 36,006-scenario SensoData pool and
   KSERESNET's original 956-road competition training/test data
   (sdc-test-data.json), by matching FRAME-INVARIANT road-shape
   signatures (translate to origin, rotate to common initial heading,
   resample by arc-length) -- raw-coordinate matching does NOT work
   here because the two sources use different absolute coordinate
   conventions (pool roads all start at a fixed (100,100) generator
   origin; competition roads start at varying absolute positions).
2. Build a stratified train/validation/test split (60/20/20) of the
   pool, stratified by (generator, outcome), with any road overlapping
   the original KSERESNET training data pinned into the train split.
"""
import argparse
import json
import math
import random
import sys
from collections import defaultdict

N_RESAMPLE = 20  # points per normalized shape signature


def polyline_length(pts):
    total = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        total += math.hypot(dx, dy)
    return total


def resample_by_arclength(pts, n):
    """Resample a polyline to n evenly-arc-length-spaced points."""
    if len(pts) < 2:
        return pts + [pts[-1]] * (n - len(pts)) if pts else [(0.0, 0.0)] * n

    seg_lens = []
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        seg_lens.append(math.hypot(dx, dy))
    total_len = sum(seg_lens)
    if total_len == 0:
        return [pts[0]] * n

    cum = [0.0]
    for l in seg_lens:
        cum.append(cum[-1] + l)

    out = []
    for i in range(n):
        target = total_len * i / (n - 1)
        # find segment containing target
        j = 1
        while j < len(cum) and cum[j] < target:
            j += 1
        j = min(j, len(cum) - 1)
        seg_start_cum = cum[j - 1]
        seg_len = seg_lens[j - 1] if seg_lens[j - 1] > 0 else 1e-9
        t = (target - seg_start_cum) / seg_len
        x = pts[j - 1][0] + t * (pts[j][0] - pts[j - 1][0])
        y = pts[j - 1][1] + t * (pts[j][1] - pts[j - 1][1])
        out.append((x, y))
    return out


def normalize_shape(road_points, n=N_RESAMPLE):
    """Translate to origin at first point, rotate so the vector to the
    resampled 2nd point lies along +x, scale-invariant via unit total length,
    return a tuple of rounded (x,y) -- a frame-invariant road-shape fingerprint."""
    pts = [(p["x"], p["y"]) for p in road_points]
    if len(pts) < 2:
        return None

    resampled = resample_by_arclength(pts, n)

    # translate so first point is origin
    ox, oy = resampled[0]
    translated = [(x - ox, y - oy) for x, y in resampled]

    # rotate so the direction to the last point (or a stable later point) is along +x
    ref_idx = min(3, len(translated) - 1)
    rx, ry = translated[ref_idx]
    angle = math.atan2(ry, rx)
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    rotated = [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in translated]

    # scale-normalize by total path length so different resampling densities agree
    length = polyline_length(rotated)
    if length < 1e-6:
        return None
    scaled = [(x / length, y / length) for x, y in rotated]

    return tuple((round(x, 2), round(y, 2)) for x, y in scaled)


def generator_of(test_id):
    parts = test_id.replace("mt_set", "mt/set").split("/")
    for p in parts:
        pl = p.lower()
        if "ambiegen" in pl:
            return "ambiegen"
        if "freneticv" in pl or "frenetic_v" in pl:
            return "freneticv"
        if "frenetic" in pl:
            return "frenetic"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--original-competition-data", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    print(f"Loading pool from {args.pool} ...", file=sys.stderr)
    with open(args.pool, "r", encoding="utf-8") as f:
        pool = json.load(f)
    print(f"Loaded {len(pool)} pool scenarios.", file=sys.stderr)

    print(f"Loading original competition data from {args.original_competition_data} ...", file=sys.stderr)
    with open(args.original_competition_data, "r", encoding="utf-8") as f:
        orig = json.load(f)
    print(f"Loaded {len(orig)} original competition roads.", file=sys.stderr)

    # --- leakage audit (frame-invariant shape signature) ---
    print("Computing frame-invariant shape signatures for original competition roads ...", file=sys.stderr)
    orig_sigs = set()
    for rec in orig:
        sig = normalize_shape(rec["road_points"])
        if sig is not None:
            orig_sigs.add(sig)
    print(f"Distinct original signatures: {len(orig_sigs)}", file=sys.stderr)

    print("Computing signatures for pool (this is the slow part) ...", file=sys.stderr)
    overlap_ids = []
    for i, rec in enumerate(pool):
        sig = normalize_shape(rec["road_points"])
        if sig is not None and sig in orig_sigs:
            overlap_ids.append(rec["test_id"])
        if (i + 1) % 10000 == 0:
            print(f"  processed {i+1}/{len(pool)}", file=sys.stderr)

    print(f"\n=== LEAKAGE AUDIT (frame-invariant shape matching) ===", file=sys.stderr)
    print(f"Original competition roads: {len(orig)}", file=sys.stderr)
    print(f"Pool scenarios matching original road SHAPE: {len(overlap_ids)}", file=sys.stderr)
    print(f"Overlap rate vs pool: {100*len(overlap_ids)/len(pool):.2f}%", file=sys.stderr)
    print(f"Overlap rate vs original: {100*len(overlap_ids)/len(orig):.2f}%", file=sys.stderr)
    if overlap_ids:
        print(f"Sample overlapping IDs: {overlap_ids[:10]}", file=sys.stderr)

    overlap_set = set(overlap_ids)

    # --- stratified split ---
    by_stratum = defaultdict(list)
    for rec in pool:
        stratum = (generator_of(rec["test_id"]), rec["outcome"])
        by_stratum[stratum].append(rec["test_id"])

    rng = random.Random(args.seed)
    train_ids, val_ids, test_ids = [], [], []

    for stratum, ids in by_stratum.items():
        ids = ids[:]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * args.train_frac)
        n_val = int(n * args.val_frac)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    val_ids = [i for i in val_ids if i not in overlap_set]
    test_ids = [i for i in test_ids if i not in overlap_set]
    train_set_current = set(train_ids)
    removed_from_val_test = [i for i in overlap_set if i not in train_set_current]
    train_ids.extend(removed_from_val_test)

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    assert not (train_set & val_set), "train/val overlap!"
    assert not (train_set & test_set), "train/test overlap!"
    assert not (val_set & test_set), "val/test overlap!"
    assert not (val_set & overlap_set), "leakage in val!"
    assert not (test_set & overlap_set), "leakage in test!"

    print(f"\n=== SPLIT SUMMARY ===", file=sys.stderr)
    print(f"Train: {len(train_ids)} ({100*len(train_ids)/len(pool):.1f}%)", file=sys.stderr)
    print(f"Val:   {len(val_ids)} ({100*len(val_ids)/len(pool):.1f}%)", file=sys.stderr)
    print(f"Test:  {len(test_ids)} ({100*len(test_ids)/len(pool):.1f}%)", file=sys.stderr)
    print(f"(all {len(overlap_set)} leakage-risk scenarios pinned into train)", file=sys.stderr)

    id_to_outcome = {r["test_id"]: r["outcome"] for r in pool}
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        fails = sum(1 for i in ids if id_to_outcome[i] == "FAIL")
        print(f"  {name}: {fails} FAIL / {len(ids)-fails} PASS "
              f"({100*fails/len(ids):.1f}% fail rate)", file=sys.stderr)

    with open(f"{args.out_prefix}_train_ids.json", "w") as f:
        json.dump(train_ids, f)
    with open(f"{args.out_prefix}_val_ids.json", "w") as f:
        json.dump(val_ids, f)
    with open(f"{args.out_prefix}_test_ids.json", "w") as f:
        json.dump(test_ids, f)
    with open(f"{args.out_prefix}_leakage_audit.json", "w") as f:
        json.dump({
            "method": "frame_invariant_shape_signature",
            "n_original_competition_roads": len(orig),
            "n_pool_scenarios": len(pool),
            "n_overlap": len(overlap_set),
            "overlap_ids": sorted(overlap_set),
        }, f, indent=2)

    print(f"\nWrote splits to {args.out_prefix}_{{train,val,test}}_ids.json", file=sys.stderr)
    print(f"Wrote leakage audit to {args.out_prefix}_leakage_audit.json", file=sys.stderr)


if __name__ == "__main__":
    main()
