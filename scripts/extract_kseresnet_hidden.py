"""
Extract h(s) -- the frozen KSERESNET 64-dim penultimate-layer representation
(after fc[0:3]: Flatten -> Linear(1600,64) -> ReLU, BEFORE Dropout and the
final Linear(64,1)) -- and the raw model logit, for a given set of scenario
ids, using the ORIGINAL frozen checkpoint (crash_model.pth), applying the
same native-density fix (stride-5) used to build kseresnet_R_scores.json.

This does not modify crash_model.pth or any existing R-scores; it is a
read-only forward pass used only to build ARSF gate input features.
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn

MAX_POINTS = 200
INPUT_CHANNELS = 4
NATIVE_STRIDE = 5


def process_road_points(points):
    coords = np.array([[p["x"], p["y"]] for p in points[::NATIVE_STRIDE]], dtype=np.float32)
    if len(coords) < 3:
        return np.zeros((MAX_POINTS, INPUT_CHANNELS), dtype=np.float32)
    deltas = coords[1:] - coords[:-1]
    accel_raw = deltas[1:] - deltas[:-1]
    accel = np.vstack((np.zeros((1, 2), dtype=np.float32), accel_raw))
    deltas = deltas.copy()
    deltas[:, 0] = (deltas[:, 0] - (-0.02)) / 0.5
    deltas[:, 1] = (deltas[:, 1] - 0.20) / 0.5
    accel = accel * 5.0
    features = np.hstack((deltas, accel)).astype(np.float32)
    if len(features) > MAX_POINTS:
        return features[:MAX_POINTS]
    padding = np.zeros((MAX_POINTS - len(features), INPUT_CHANNELS), dtype=np.float32)
    return np.vstack((features, padding))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid(),
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
            nn.Flatten(), nn.Linear(32 * 50, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1),
        )

    def backbone(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.res_block1(self.input_layer(x)))
        x = self.pool2(self.res_block2(x))
        x = self.res_block3(x)
        return x

    def forward_with_hidden(self, x):
        feat = self.backbone(x)
        h = self.fc[2](self.fc[1](self.fc[0](feat)))  # Flatten -> Linear(1600,64) -> ReLU
        logit = self.fc[4](self.fc[3](h))  # Dropout (no-op in eval) -> Linear(64,1)
        return h, logit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--checkpoint", default="crash_model.pth")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.pool, "r", encoding="utf-8") as f:
        pool = {r["test_id"]: r for r in json.load(f)}
    with open(args.ids, "r", encoding="utf-8") as f:
        ids = json.load(f)

    model = ResNetPredictor()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    out = []
    with torch.no_grad():
        for i, tid in enumerate(ids):
            rec = pool.get(tid)
            if rec is None:
                continue
            feats = process_road_points(rec["road_points"])
            t = torch.FloatTensor(feats).unsqueeze(0)
            h, logit = model.forward_with_hidden(t)
            out.append({
                "test_id": tid,
                "h": h.squeeze(0).numpy().astype(np.float32).tolist(),
                "logit": float(logit.item()),
                "outcome": rec["outcome"],
                "duration": rec.get("duration"),
            })
            if (i + 1) % 2000 == 0:
                print(f"{i+1}/{len(ids)}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} records to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
