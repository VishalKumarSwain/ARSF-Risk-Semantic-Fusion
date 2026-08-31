"""
Standalone batch scoring of the full SensoData pool with the verified
original KSERESNET model -- no gRPC server needed, direct model inference.

Reuses the exact architecture and feature-extraction code from the
official Vishal_Tool/main.py (byte-verified against the GitHub
final-code-submission branch).

Output: a JSON list of {test_id, outcome, duration, R_score} records --
this is R(s), the "learned risk" component of KSERESNET-RS.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

MAX_POINTS = 200
INPUT_CHANNELS = 4


# --- exact feature extraction from the original tool (Vishal-ResNet-Hybrid) ---
def process_road_points(points):
    coords = np.array([[float(p["x"]), float(p["y"])] for p in points])
    if len(coords) < 3:
        return np.zeros((MAX_POINTS, INPUT_CHANNELS))

    deltas = coords[1:] - coords[:-1]
    accel_raw = deltas[1:] - deltas[:-1]
    accel = np.vstack((np.zeros((1, 2)), accel_raw))

    deltas = deltas.copy()
    deltas[:, 0] = (deltas[:, 0] - (-0.02)) / 0.5
    deltas[:, 1] = (deltas[:, 1] - 0.20) / 0.5
    accel = accel * 5.0

    features = np.hstack((deltas, accel))
    if len(features) > MAX_POINTS:
        return features[:MAX_POINTS]
    padding = np.zeros((MAX_POINTS - len(features), INPUT_CHANNELS))
    return np.vstack((features, padding))


# --- exact architecture from the original tool ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + res)


class ResNetPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Conv1d(INPUT_CHANNELS, 32, kernel_size=5, padding=2)
        self.res_block1 = ResidualBlock(32)
        self.pool1 = nn.MaxPool1d(2)
        self.res_block2 = ResidualBlock(32)
        self.pool2 = nn.MaxPool1d(2)
        self.res_block3 = ResidualBlock(32)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 50, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.res_block1(self.input_layer(x)))
        x = self.pool2(self.res_block2(x))
        x = self.res_block3(x)
        return self.fc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    print(f"Device: {args.device}", file=sys.stderr)
    print(f"Loading pool from {args.pool} ...", file=sys.stderr)
    with open(args.pool, "r", encoding="utf-8") as f:
        pool = json.load(f)
    print(f"Loaded {len(pool)} scenarios.", file=sys.stderr)

    model = ResNetPredictor()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    model.to(args.device)

    t0 = time.time()
    results = []
    with torch.no_grad():
        for start in range(0, len(pool), args.batch_size):
            batch = pool[start:start + args.batch_size]
            feats = np.stack([process_road_points(r["road_points"]) for r in batch])
            tensor = torch.FloatTensor(feats).to(args.device)
            scores = model(tensor).squeeze(-1).cpu().numpy()

            for rec, score in zip(batch, scores):
                results.append({
                    "test_id": rec["test_id"],
                    "outcome": rec["outcome"],
                    "duration": rec.get("duration"),
                    "R_score": float(score),
                })

            if (start // args.batch_size) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  scored {start + len(batch)}/{len(pool)} "
                      f"({elapsed:.1f}s elapsed)", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"Done. Scored {len(results)} scenarios in {elapsed:.2f}s "
          f"({1000*elapsed/len(results):.3f} ms/scenario)", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Wrote R(s) scores to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
