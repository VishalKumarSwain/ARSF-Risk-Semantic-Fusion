"""
Approximate replication of the ICST 2026 SDC Testing Competition's official
evaluation protocol (see Table III of the competition report):

  - sample_size and subject_size both swept over {10,20,...,200}
  - each sample = `sample_size` subjects of `subject_size` test cases each,
    drawn randomly from the pool
  - first subject in each sample -> Initialize (ignored by this tool anyway)
  - remaining subjects -> Prioritize, one APFD/APFDc/timing per subject
  - aggregate min/mean/max/std over ALL individual subject-level results

This script uses a reduced grid (fewer points, same range) for tractable
local runtime; pass --full for the true 20x20=400 grid.
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import grpc
import competition_2026_pb2 as pb2
import competition_2026_pb2_grpc as pb2_grpc


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    durations = [rec["duration"] for rec in raw if rec.get("duration") is not None]
    mean_duration = sum(durations) / len(durations) if durations else 1.0
    tests = []
    for i, rec in enumerate(raw):
        d = rec.get("duration")
        tests.append({
            "test_id": rec.get("test_id", str(i)),
            "outcome": rec["outcome"],
            "duration": d if d is not None else mean_duration,
            "points": rec["road_points"],
        })
    return tests


def to_road_points(points):
    return [pb2.RoadPoint(sequenceNumber=i, x=float(p["x"]), y=float(p["y"]))
            for i, p in enumerate(points)]


def build_testcase_stream(tests):
    for t in tests:
        yield pb2.SDCTestCase(testId=t["test_id"], roadPoints=to_road_points(t["points"]))


def build_oracle_stream(tests):
    for t in tests:
        tc = pb2.SDCTestCase(testId=t["test_id"], roadPoints=to_road_points(t["points"]))
        yield pb2.Oracle(testCase=tc, hasFailed=(t["outcome"] == "FAIL"))


def compute_apfd(ranked_ids, meta):
    """Exact line-by-line reimplementation of the organizers' MetricEvaluator.compute_apfd
    (tools/prioritizers/evaluator/metrics.py) -- including its zero-failure convention:
    returns 1.0 (perfect score) when a subject has no failures, rather than None/undefined."""
    n = len(ranked_ids)
    failed_positions = []
    for i, tid in enumerate(ranked_ids):
        if meta[tid]["outcome"] == "FAIL":
            failed_positions.append(i + 1)
    m = len(failed_positions)
    if n == 0 or m == 0:
        return 1.0  # organizers' convention: perfect score if no faults or no tests
    sum_failed_ranks = sum(failed_positions)
    return 1 - (sum_failed_ranks / (n * m)) + 1 / (2 * n)


def compute_apfdc(ranked_ids, meta):
    """Exact line-by-line reimplementation of the organizers' MetricEvaluator.compute_apdfc.
    Cost = cumulative time UP TO AND INCLUDING each failing test (not tail-cost); the +1/(2n)
    term divides by num_failed (m), not by total_cost or n; zero-failure subjects return 1.0."""
    cumulative_costs_to_faults = []
    cumulative_time = 0.0
    total_cost = 0.0
    num_failed = 0
    for tid in ranked_ids:
        duration = meta[tid]["duration"]
        cumulative_time += duration
        total_cost += duration
        if meta[tid]["outcome"] == "FAIL":
            cumulative_costs_to_faults.append(cumulative_time)
            num_failed += 1
    if num_failed == 0 or total_cost == 0:
        return 1.0  # organizers' convention
    sum_cfi = sum(cumulative_costs_to_faults)
    return 1 - (sum_cfi / (total_cost * num_failed)) + 1 / (2 * num_failed)


def time_to_first_last_fault(ranked_ids, meta):
    cum = 0.0
    first_fault = None
    last_fault = None
    for tid in ranked_ids:
        cum += meta[tid]["duration"]
        if meta[tid]["outcome"] == "FAIL":
            if first_fault is None:
                first_fault = cum
            last_fault = cum
    return first_fault, last_fault


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", default="50051")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full", action="store_true", help="use the true 20x20 grid (10..200 step 10)")
    args = ap.parse_args()

    if args.full:
        sample_sizes = list(range(10, 201, 10))
        subject_sizes = list(range(10, 201, 10))
    else:
        sample_sizes = [10, 50, 100, 150, 200]
        subject_sizes = [10, 50, 100, 150, 200]

    print(f"Loading dataset from {args.data} ...")
    all_tests = load_dataset(args.data)
    print(f"Loaded {len(all_tests)} test cases.")
    meta = {t["test_id"]: t for t in all_tests}

    target = f"{args.host}:{args.port}"
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", 200 * 1024 * 1024),
            ("grpc.max_receive_message_length", 200 * 1024 * 1024),
        ],
    )
    stub = pb2_grpc.CompetitionToolStub(channel)
    print(f"Tool name reported: {stub.Name(pb2.Empty()).name}")

    rng = random.Random(args.seed)
    apfd_vals, apfdc_vals, tprior_vals, tfirst_vals, tlast_vals = [], [], [], [], []

    n_configs = len(sample_sizes) * len(subject_sizes)
    config_i = 0
    t_start = time.time()

    for subject_size in subject_sizes:
        for sample_size in sample_sizes:
            config_i += 1
            subjects = []
            for _ in range(sample_size):
                subjects.append(rng.sample(all_tests, min(subject_size, len(all_tests))))

            init_subject, eval_subjects = subjects[0], subjects[1:]
            stub.Initialize(build_oracle_stream(init_subject))

            for subj in eval_subjects:
                t0 = time.time()
                ranked_ids = [x.testId for x in stub.Prioritize(build_testcase_stream(subj))]
                elapsed = time.time() - t0

                apfd = compute_apfd(ranked_ids, meta)
                apfdc = compute_apfdc(ranked_ids, meta)
                tf, tl = time_to_first_last_fault(ranked_ids, meta)
                if apfd is not None:
                    apfd_vals.append(apfd)
                    apfdc_vals.append(apfdc)
                    tprior_vals.append(elapsed)
                    if tf is not None:
                        tfirst_vals.append(tf)
                        tlast_vals.append(tl)

            elapsed_total = time.time() - t_start
            print(f"  config {config_i}/{n_configs} "
                  f"(subject_size={subject_size}, sample_size={sample_size}): "
                  f"{len(eval_subjects)} treatments, "
                  f"running_n={len(apfd_vals)}, elapsed={elapsed_total:.1f}s", file=sys.stderr)

    def stats(vals):
        if not vals:
            return None
        return dict(mean=statistics.mean(vals), std=statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    min=min(vals), max=max(vals))

    print()
    print("=" * 70)
    print(f"RESULTS -- {len(apfd_vals)} total treatments across {n_configs} configs "
          f"({'FULL 20x20 grid' if args.full else 'REDUCED grid'})")
    print("=" * 70)
    print(f"{'Metric':20s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
    for label, vals in [("apfd", apfd_vals), ("apfdc", apfdc_vals),
                        ("t_prioritize(s)", tprior_vals),
                        ("t_first_fault(s)", tfirst_vals), ("t_last_fault(s)", tlast_vals)]:
        s = stats(vals)
        if s:
            print(f"{label:20s} {s['mean']:10.4f} {s['std']:10.4f} {s['min']:10.4f} {s['max']:10.4f}")

    print()
    print("Official published KSERESNET numbers (Table III, ICST 2026 report):")
    print(f"{'apfd':20s} {'0.62':>10s} {'0.04':>10s} {'0.56':>10s} {'0.69':>10s}")
    print(f"{'apfdc':20s} {'0.65':>10s} {'0.05':>10s} {'0.57':>10s} {'0.73':>10s}")
    print(f"{'t_prioritize(s)':20s} {'0.08':>10s} {'0.00':>10s} {'0.08':>10s} {'0.09':>10s}")
    print(f"{'t_first_fault(s)':20s} {'77.83':>10s} {'59.71':>10s} {'13.29':>10s} {'193.62':>10s}")
    print(f"{'t_last_fault(s)':20s} {'5593.73':>10s} {'414.17':>10s} {'4909.35':>10s} {'6188.57':>10s}")


if __name__ == "__main__":
    main()
