"""
Rebuild kseresnet_R_scores.json for the full 36,006-scenario pool using the
FIXED native-density point extraction (road_points[::5], matching the
original training distribution) against the live, corrected KSERESNET
container. Replaces the stale scores computed before the point-density
bug fix.
"""
import argparse
import json
import sys

import grpc
import competition_2026_pb2 as pb2
import competition_2026_pb2_grpc as pb2_grpc


def to_road_points(points):
    return [pb2.RoadPoint(sequenceNumber=i, x=float(p["x"]), y=float(p["y"]))
            for i, p in enumerate(points[::5])]  # native stride-5 fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", default="50051")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        pool = json.load(f)
    print(f"Loaded {len(pool)} scenarios", file=sys.stderr)

    channel = grpc.insecure_channel(
        f"{args.host}:{args.port}",
        options=[
            ("grpc.max_send_message_length", 200 * 1024 * 1024),
            ("grpc.max_receive_message_length", 200 * 1024 * 1024),
        ],
    )
    stub = pb2_grpc.CompetitionToolStub(channel)
    print(f"Tool: {stub.Name(pb2.Empty()).name}", file=sys.stderr)

    # KSERESNET's Prioritize only returns a RANKING (sorted test IDs), not raw
    # scores. To get per-scenario R_score values we score each scenario as
    # its own singleton "prioritization" batch is wasteful; instead we
    # replicate the tool's own scoring math isn't exposed via gRPC, so we
    # infer relative scores by prioritizing in reasonably sized chunks and
    # using rank position as a proxy is NOT precise enough for fusion.
    #
    # Instead: call Prioritize on the FULL pool at once (single subject),
    # then assign each test a score based on its rank (higher score for
    # earlier rank), preserving relative ordering exactly as the model
    # would produce for the whole pool.
    tests = []
    for r in pool:
        tests.append(pb2.SDCTestCase(testId=r["test_id"], roadPoints=to_road_points(r["road_points"])))

    print("Requesting full-pool prioritization...", file=sys.stderr)
    ranked_ids = [x.testId for x in stub.Prioritize(iter(tests))]
    print(f"Got {len(ranked_ids)} ranked ids", file=sys.stderr)

    n = len(ranked_ids)
    rank_of = {tid: i for i, tid in enumerate(ranked_ids)}
    meta = {r["test_id"]: r for r in pool}

    out = []
    for tid, rank in rank_of.items():
        rec = meta[tid]
        score = 1.0 - (rank / max(n - 1, 1))  # rank 0 (highest risk) -> 1.0, last -> 0.0
        out.append({
            "test_id": tid,
            "outcome": rec["outcome"],
            "duration": rec.get("duration"),
            "R_score": score,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} scores to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
