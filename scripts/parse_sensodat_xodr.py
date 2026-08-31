"""
Parse SensoDat .xodr (OpenDRIVE) road files into (x, y) waypoint lists +
PASS/FAIL labels, matching the format the KSERESNET gRPC tool expects
(same shape as sdc-test-data.json's road_points).

OpenDRIVE paramPoly3 geometry (pRange="normalized"):
    u(p) = aU + bU*p + cU*p^2 + dU*p^3
    v(p) = aV + bV*p + cV*p^2 + dV*p^3
    X = x0 + u*cos(hdg) - v*sin(hdg)
    Y = y0 + u*sin(hdg) + v*cos(hdg)
where (x0, y0, hdg) are the geometry segment's start pose.
"""
import json
import math
import re
import sys
from pathlib import Path

HEADER_RE = re.compile(
    r'<sdc_test_info\s+test_id="([^"]*)"\s+test_outcome="([^"]*)"'
    r'\s+predicted_test_outcome="[^"]*"\s+test_duration="([^"]*)"'
)
GEOM_RE = re.compile(
    r'<geometry\s+s="([^"]*)"\s+x="([^"]*)"\s+y="([^"]*)"\s+hdg="([^"]*)"\s+length="([^"]*)">'
    r'\s*<paramPoly3\s+pRange="normalized"\s+'
    r'aU="([^"]*)"\s+bU="([^"]*)"\s+cU="([^"]*)"\s+dU="([^"]*)"\s+'
    r'aV="([^"]*)"\s+bV="([^"]*)"\s+cV="([^"]*)"\s+dV="([^"]*)"\s*/>'
)

SAMPLES_PER_SEGMENT = 5


def parse_xodr(text):
    hm = HEADER_RE.search(text)
    if not hm:
        return None, None, None, []
    test_id, outcome = hm.group(1), hm.group(2)
    try:
        duration = float(hm.group(3))
    except ValueError:
        duration = None  # e.g. literal "None" string seen in a small number of files

    points = []
    for gm in GEOM_RE.finditer(text):
        (_s, x0, y0, hdg, _length,
         aU, bU, cU, dU, aV, bV, cV, dV) = gm.groups()
        x0, y0, hdg = float(x0), float(y0), float(hdg)
        aU, bU, cU, dU = float(aU), float(bU), float(cU), float(dU)
        aV, bV, cV, dV = float(aV), float(bV), float(cV), float(dV)
        cos_h, sin_h = math.cos(hdg), math.sin(hdg)

        for i in range(SAMPLES_PER_SEGMENT):
            p = i / (SAMPLES_PER_SEGMENT - 1) if SAMPLES_PER_SEGMENT > 1 else 0.0
            u = aU + bU * p + cU * p * p + dU * p * p * p
            v = aV + bV * p + cV * p * p + dV * p * p * p
            X = x0 + u * cos_h - v * sin_h
            Y = y0 + u * sin_h + v * cos_h
            points.append((X, Y))

    return test_id, outcome, duration, points


def build_pool(sensodat_root, out_path, max_scenarios=None):
    root = Path(sensodat_root)
    xodr_files = sorted(root.rglob("*.xodr"))
    print(f"Found {len(xodr_files)} .xodr files", file=sys.stderr)

    records = []
    skipped_no_outcome = 0
    skipped_not_executed = 0
    skipped_too_few_points = 0
    parse_errors = 0

    for i, p in enumerate(xodr_files):
        if max_scenarios and len(records) >= max_scenarios:
            break
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            parse_errors += 1
            continue
        test_id, outcome, duration, points = parse_xodr(text)
        if test_id is None:
            skipped_no_outcome += 1
            continue
        if outcome not in ("PASS", "FAIL"):
            skipped_not_executed += 1
            continue
        if len(points) < 3:
            skipped_too_few_points += 1
            continue
        records.append({
            "test_id": test_id,
            "outcome": outcome,
            "duration": duration,  # may be None -- filled with pool-mean at analysis time
            "road_points": [{"x": x, "y": y} for x, y in points],
        })
        if (i + 1) % 5000 == 0:
            print(f"  processed {i+1}/{len(xodr_files)} files, "
                  f"{len(records)} valid records so far", file=sys.stderr)

    print(f"Total valid records: {len(records)}", file=sys.stderr)
    print(f"Skipped (no header match): {skipped_no_outcome}", file=sys.stderr)
    print(f"Skipped (NOT_EXECUTED/other outcome): {skipped_not_executed}", file=sys.stderr)
    print(f"Skipped (too few geometry points): {skipped_too_few_points}", file=sys.stderr)
    print(f"Parse errors: {parse_errors}", file=sys.stderr)

    outcomes = {}
    for r in records:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    print(f"Outcome distribution: {outcomes}", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} records to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensodat-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-scenarios", type=int, default=None)
    args = ap.parse_args()
    build_pool(args.sensodat_dir, args.out, args.max_scenarios)
