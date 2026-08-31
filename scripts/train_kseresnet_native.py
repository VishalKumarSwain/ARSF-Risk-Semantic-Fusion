"""
Train a fresh KSERESNET (same SE-ResNet architecture as the original ICST
submission) on the full 36,006-scenario SensoData pool, using the corrected
native point density (stride-5 downsampling of our stored road_points,
which reconstructs the true one-point-per-geometry-segment representation
the original model's checkpoint was trained on -- see process_road_points()
in the deployed main.py, MAX_POINTS=200).

Run on the DGX MIG partition. Uses the existing train/val split IDs.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

MAX_POINTS = 200
INPUT_CHANNELS = 4
NATIVE_STRIDE = 5  # matches SAMPLES_PER_SEGMENT used when the pool was built


def load_pool(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def process_road_points(points):
    """Same kinematic-feature computation as the deployed main.py, but fed
    native-density (stride-5) points instead of the raw oversampled ones."""
    coords = np.array([[p["x"], p["y"]] for p in points[::NATIVE_STRIDE]], dtype=np.float32)
    if len(coords) < 3:
        return np.zeros((MAX_POINTS, INPUT_CHANNELS), dtype=np.float32)

    deltas = coords[1:] - coords[:-1]
    accel_raw = deltas[1:] - deltas[:-1]
    accel = np.vstack((np.zeros((1, 2), dtype=np.float32), accel_raw))

    features = np.hstack((deltas, accel)).astype(np.float32)
    if len(features) > MAX_POINTS:
        return features[:MAX_POINTS]
    padding = np.zeros((MAX_POINTS - len(features), INPUT_CHANNELS), dtype=np.float32)
    return np.vstack((features, padding))


class SDCDataset(Dataset):
    def __init__(self, records, mean=None, std=None):
        self.X = np.stack([process_road_points(r["road_points"]) for r in records])
        self.y = np.array([1.0 if r["outcome"] == "FAIL" else 0.0 for r in records], dtype=np.float32)
        if mean is None:
            self.mean = self.X.mean(axis=(0, 1), keepdims=True)
            self.std = self.X.std(axis=(0, 1), keepdims=True) + 1e-8
        else:
            self.mean, self.std = mean, std
        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


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


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, n, tp, fp, fn = 0.0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yb).sum().item()
            tp += ((preds == 1) & (yb == 1)).sum().item()
            fp += ((preds == 1) & (yb == 0)).sum().item()
            fn += ((preds == 0) & (yb == 1)).sum().item()
            n += len(yb)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return total_loss / n, correct / n, prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--train-ids", required=True)
    ap.add_argument("--val-ids", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out", default="kseresnet_native_best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading pool...")
    pool = load_pool(args.pool)
    train_ids = load_ids(args.train_ids)
    val_ids = load_ids(args.val_ids)

    train_records = [r for r in pool if r["test_id"] in train_ids]
    val_records = [r for r in pool if r["test_id"] in val_ids]
    print(f"train={len(train_records)} val={len(val_records)}")

    print("Building features...")
    train_ds = SDCDataset(train_records)
    val_ds = SDCDataset(val_records, mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ResNetPredictor().to(device)
    n_fail = sum(r["outcome"] == "FAIL" for r in train_records)
    n_pass = len(train_records) - n_fail
    pos_weight = torch.tensor([n_pass / max(n_fail, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            n += len(yb)
        train_loss = total_loss / n

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_prec={val_prec:.4f} val_rec={val_rec:.4f} "
              f"val_f1={val_f1:.4f} elapsed={elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "mean": train_ds.mean,
                "std": train_ds.std,
            }, args.out)
            print(f"  -> saved new best (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"Training complete in {time.time()-t0:.1f}s. Best val_loss={best_val_loss:.4f}. Saved to {args.out}")


if __name__ == "__main__":
    main()
