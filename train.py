"""
train.py
--------
Trains ECGNet on heartbeat data and evaluates it.

Usage:
    python train.py --source real        # download real MIT-BIH via PhysioNet
    python train.py --source synthetic   # use synthetic data (no internet needed)

The --source real path requires internet access to physionet.org, which
downloads the 48-record MIT-BIH Arrhythmia Database (this can take a few
minutes on first run).
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dataset import load_mitbih_real, generate_synthetic_ecg, normalize, CLASS_NAMES
from model import ECGNet


def get_data(source, synthetic_n):
    if source == "real":
        print("Downloading + parsing real MIT-BIH data from PhysioNet (this may take a few minutes)...")
        X, y = load_mitbih_real()
    else:
        print(f"Generating {synthetic_n} synthetic heartbeat samples (pipeline test mode, NOT real patient data)...")
        X, y = generate_synthetic_ecg(n_samples=synthetic_n)

    X = normalize(X)
    print(f"Loaded {len(X)} beats. Class distribution: "
          f"{ {CLASS_NAMES[c]: int((y == c).sum()) for c in range(len(CLASS_NAMES))} }")
    return X, y


def split_data(X, y, val_frac=0.15, test_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    n_test = int(len(X) * test_frac)

    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]

    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])


def make_loader(X, y, batch_size=64, shuffle=True):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, val_loader, epochs=15, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(xb)

        train_loss, train_acc = total_loss / n, correct / n
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:2d}/{epochs} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.4f}")

    return model


def evaluate(model, loader, criterion, device="cpu"):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * len(xb)
            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            n += len(xb)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    return (total_loss / n, correct / n,
            np.concatenate(all_preds), np.concatenate(all_labels))


def confusion_matrix(preds, labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def print_confusion_matrix(cm, class_names):
    print("\nConfusion Matrix (rows = true label, cols = predicted label)")
    header = "".join(f"{n[:6]:>8}" for n in class_names)
    print(" " * 10 + header)
    for i, row in enumerate(cm):
        print(f"{class_names[i][:9]:>9} " + "".join(f"{v:>8}" for v in row))


def per_class_report(cm, class_names):
    print("\nPer-class metrics:")
    print(f"{'Class':<18}{'Precision':>10}{'Recall':>10}{'F1':>8}{'Support':>9}")
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        support = cm[i].sum()
        pred_total = cm[:, i].sum()
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"{name:<18}{precision:>10.3f}{recall:>10.3f}{f1:>8.3f}{support:>9}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["real", "synthetic"], default="synthetic")
    parser.add_argument("--synthetic_n", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X, y = get_data(args.source, args.synthetic_n)
    (Xtr, ytr), (Xval, yval), (Xte, yte) = split_data(X, y)

    train_loader = make_loader(Xtr, ytr, args.batch_size, shuffle=True)
    val_loader = make_loader(Xval, yval, args.batch_size, shuffle=False)
    test_loader = make_loader(Xte, yte, args.batch_size, shuffle=False)

    model = ECGNet(num_classes=len(CLASS_NAMES), input_len=Xtr.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    model = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Results: loss {test_loss:.4f} | accuracy {test_acc:.4f}")

    cm = confusion_matrix(preds, labels, len(CLASS_NAMES))
    print_confusion_matrix(cm, CLASS_NAMES)
    per_class_report(cm, CLASS_NAMES)

    torch.save(model.state_dict(), "ecgnet_weights.pt")
    print("\nSaved trained weights to ecgnet_weights.pt")


if __name__ == "__main__":
    main()