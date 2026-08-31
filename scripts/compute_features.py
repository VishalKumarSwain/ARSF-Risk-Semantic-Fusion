"""
Compute the frozen shared feature schema (see FEATURE_SCHEMA.md) for every
scenario in the pool. Output feeds both B4 (GBDT) and the LLM prompt builder
-- this is the single source of truth for "identical features" between them.
"""
import argparse
import json
import math
import sys


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


def compute_features(road_points, test_id):
    pts = [(p["x"], p["y"]) for p in road_points]
    n = len(pts)
    if n < 3:
        return None

    seg_lens = []
    headings = []
    for i in range(1, n):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        seg_lens.append(math.hypot(dx, dy))
        headings.append(math.atan2(dy, dx))

    road_length = sum(seg_lens)
    sld = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    tortuosity = road_length / sld if sld > 1e-6 else 0.0

    curvatures = []
    for i in range(1, len(headings)):
        d = headings[i] - headings[i - 1]
        # wrap to [-pi, pi]
        d = (d + math.pi) % (2 * math.pi) - math.pi
        curvatures.append(abs(d))

    if curvatures:
        mean_curv = sum(curvatures) / len(curvatures)
        max_curv = max(curvatures)
        std_curv = (sum((c - mean_curv) ** 2 for c in curvatures) / len(curvatures)) ** 0.5
        sharp_turns = sum(1 for c in curvatures if c > 0.3)
    else:
        mean_curv = max_curv = std_curv = 0.0
        sharp_turns = 0

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    aspect = bbox_w / max(bbox_h, 1e-6)

    return {
        "test_id": test_id,
        "generator": generator_of(test_id),
        "num_points": n,
        "road_length": round(road_length, 4),
        "straight_line_distance": round(sld, 4),
        "tortuosity": round(tortuosity, 4),
        "mean_curvature": round(mean_curv, 6),
        "max_curvature": round(max_curv, 6),
        "std_curvature": round(std_curv, 6),
        "sharp_turn_count": sharp_turns,
        "bbox_aspect_ratio": round(aspect, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Loading pool from {args.pool} ...", file=sys.stderr)
    with open(args.pool, "r", encoding="utf-8") as f:
        pool = json.load(f)
    print(f"Loaded {len(pool)} scenarios.", file=sys.stderr)

    results = []
    skipped = 0
    for i, rec in enumerate(pool):
        feats = compute_features(rec["road_points"], rec["test_id"])
        if feats is None:
            skipped += 1
            continue
        feats["outcome"] = rec["outcome"]
        feats["duration"] = rec.get("duration")  # kept for APFDc cost only, NOT a model feature
        results.append(feats)
        if (i + 1) % 10000 == 0:
            print(f"  processed {i+1}/{len(pool)}", file=sys.stderr)

    print(f"Computed features for {len(results)} scenarios ({skipped} skipped).", file=sys.stderr)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Wrote to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
