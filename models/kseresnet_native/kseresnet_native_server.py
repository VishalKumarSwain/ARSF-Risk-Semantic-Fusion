"""
gRPC CompetitionTool server for the freshly-trained kseresnet_native_best.pt
checkpoint (trained on the full 36K SensoData pool with corrected native
point density). Same architecture/feature extraction as train_kseresnet_native.py.
"""
import sys
from concurrent import futures

import numpy as np
import torch
import torch.nn as nn
import grpc

import competition_2026_pb2 as pb2
import competition_2026_pb2_grpc as pb2_grpc

MAX_POINTS = 200
INPUT_CHANNELS = 4
NATIVE_STRIDE = 5


def process_road_points(points, mean, std):
    coords = np.array([[p.x, p.y] for p in points[::NATIVE_STRIDE]], dtype=np.float32)
    if len(coords) < 3:
        feats = np.zeros((MAX_POINTS, INPUT_CHANNELS), dtype=np.float32)
    else:
        deltas = coords[1:] - coords[:-1]
        accel_raw = deltas[1:] - deltas[:-1]
        accel = np.vstack((np.zeros((1, 2), dtype=np.float32), accel_raw))
        features = np.hstack((deltas, accel)).astype(np.float32)
        if len(features) > MAX_POINTS:
            feats = features[:MAX_POINTS]
        else:
            padding = np.zeros((MAX_POINTS - len(features), INPUT_CHANNELS), dtype=np.float32)
            feats = np.vstack((features, padding))
    return (feats - mean[0]) / std[0]


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
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
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.res_block1(self.input_layer(x)))
        x = self.pool2(self.res_block2(x))
        x = self.res_block3(x)
        return self.fc(x)


class MyPrioritizer(pb2_grpc.CompetitionToolServicer):
    def __init__(self, checkpoint_path):
        print("Loading kseresnet_native checkpoint...", file=sys.stderr)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = ResNetPredictor()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.mean = ckpt["mean"]
        self.std = ckpt["std"]
        print("Loaded.", file=sys.stderr)

    def Name(self, request, context):
        return pb2.NameReply(name="KSERESNET-Native-Retrained")

    def Initialize(self, request_iterator, context):
        for _ in request_iterator:
            pass
        return pb2.InitializationReply(ok=True)

    def Prioritize(self, request_iterator, context):
        scored = []
        with torch.no_grad():
            for test in request_iterator:
                feats = process_road_points(test.roadPoints, self.mean, self.std)
                t = torch.FloatTensor(feats).unsqueeze(0)
                score = self.model(t).item()
                scored.append((str(test.testId), score))
        scored.sort(key=lambda x: x[1], reverse=True)
        for tid, _ in scored:
            yield pb2.PrioritizationReply(testId=tid)


def serve(port, checkpoint_path):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pb2_grpc.add_CompetitionToolServicer_to_server(MyPrioritizer(checkpoint_path), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Listening on {port}...", file=sys.stderr)
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    serve(args.port, args.checkpoint)
