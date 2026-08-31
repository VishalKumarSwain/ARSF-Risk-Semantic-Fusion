"""
Test ICST 2026 competitor prioritization tools (RoadFury, RTE4SDC, ITEP4SDC)
against our pool, using the same gRPC interface and APFD/APFDc computation
as test_kseresnet_versions.py, for a direct comparison. Exploratory/context
only -- these are other teams' competition submissions, used here purely as
a benchmarking reference point (as the organizers themselves did), not as
part of the KSERESNET-RS method.
"""
import argparse
import json
import random
import sys
import time

import grpc
import competition_2026_pb2 as pb2
import competition_2026_pb2_grpc as pb2_grpc


def load_pool(path):
    with open(path) as f:
        return json.load(f)


def to_road_points(points):
    return [pb2.RoadPoint(sequenceNumber=i, x=float(p["x"]), y=float(p["y"])) for i, p in enumerate(points)]


def build_oracle_stream(tests):
    for t in tests:
        tc = pb2.SDCTestCase(testId=t["test_id"], roadPoints=to_road_points(t["road_points"]))
        yield pb2.Oracle(testCase=tc, hasFailed=(t["outcome"] == "FAIL"))


def build_testcase_stream(tests):
    for t in tests:
        yield pb2.SDCTestCase(testId=t["test_id"], roadPoints=to_road_points(t["road_points"]))


def compute_apfd(ranked_ids, meta):
    n = len(ranked_ids)
    fail_pos = [i + 1 for i, tid in enumerate(ranked_ids) if meta[tid]["outcome"] == "FAIL"]
    m = len(fail_pos)
    if n == 0 or m == 0:
        return None
    return 1 - (sum(fail_pos) / (n * m)) + 1 / (2 * n)


def compute_apfdc(ranked_ids, meta):
    n = len(ranked_ids)
    costs = [meta[tid]["duration"] for tid in ranked_ids]
    total_cost = sum(costs)
    fail_idx = [i for i, tid in enumerate(ranked_ids) if meta[tid]["outcome"] == "FAIL"]
    m = len(fail_idx)
    if n == 0 or m == 0 or total_cost == 0:
        return None
    numerator = 0.0
    for i in fail_idx:
        numerator += sum(costs[i:]) - 0.5 * costs[i]
    return numerator / (total_cost * m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", default="50051")
    ap.add_argument("--oracle-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    meta = {r["test_id"]: r for r in pool}

    rng = random.Random(args.seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    split = int(len(shuffled) * args.oracle_frac)
    oracle_tests = shuffled[:split]
    eval_tests = shuffled[split:]

    target = f"{args.host}:{args.port}"
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", 200 * 1024 * 1024),
            ("grpc.max_receive_message_length", 200 * 1024 * 1024),
        ],
    )
    stub = pb2_grpc.CompetitionToolStub(channel)

    try:
        name_reply = stub.Name(pb2.Empty(), timeout=30)
        print(f"Tool name: {name_reply.name}", file=sys.stderr)
    except Exception as e:
        print(json.dumps({"error": f"Name() failed: {e}"}))
        return

    try:
        t0 = time.time()
        stub.Initialize(build_oracle_stream(oracle_tests), timeout=args.timeout)
        init_time = time.time() - t0

        t0 = time.time()
        ranked_ids = [r.testId for r in stub.Prioritize(build_testcase_stream(eval_tests), timeout=args.timeout)]
        prioritize_time = time.time() - t0
    except Exception as e:
        print(json.dumps({"tool_name": name_reply.name, "error": str(e)}))
        return

    if len(ranked_ids) != len(eval_tests):
        print(f"WARNING: returned {len(ranked_ids)} but sent {len(eval_tests)}", file=sys.stderr)

    apfd = compute_apfd(ranked_ids, meta)
    apfdc = compute_apfdc(ranked_ids, meta)
    n_fail = sum(1 for tid in ranked_ids if meta[tid]["outcome"] == "FAIL")

    print(json.dumps({
        "tool_name": name_reply.name,
        "n_eval": len(eval_tests),
        "n_returned": len(ranked_ids),
        "n_fail": n_fail,
        "APFD": round(apfd, 4) if apfd else None,
        "APFDc": round(apfdc, 4) if apfdc else None,
        "init_time_sec": round(init_time, 3),
        "prioritize_time_sec": round(prioritize_time, 3),
    }))


if __name__ == "__main__":
    main()
