# Sanity check: baseline 在 dense 输入 (stat) 上能否正常训练
# 验证 baseline 实现本身没问题，只是 pkt_seq 极度稀疏
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)
from src.models.baselines import FSNet, YaTCNet, TFEGraph, ETBERTLite, NetMambaLite, NetConvLite, count_params
from src.data.load_h5 import time_aware_split_indices


def load_stat(h5_path, indices):
    indices = np.asarray(indices, dtype=np.int64)
    sort_perm = np.argsort(indices, kind="stable")
    sorted_idx = indices[sort_perm]
    with h5py.File(h5_path, "r") as f:
        S_sorted = f["stat"][sorted_idx]
        labels_all = f["label"].asstr()[:]
        y_str_sorted = labels_all[sorted_idx]
    cat = pd.Categorical(y_str_sorted)
    y_sorted = np.array(cat.codes, dtype=np.int64)
    unsort = np.argsort(sort_perm, kind="stable")
    return S_sorted[unsort], y_sorted[unsort], list(cat.categories)


# 简易 MLP 作为"baseline on stat"——纯 dense 输入 sanity test
class MLPBaseline(nn.Module):
    def __init__(self, n_classes, in_dim=32, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.BatchNorm1d(hid), nn.GELU(),
            nn.Linear(hid, hid), nn.BatchNorm1d(hid), nn.GELU(),
            nn.Linear(hid, n_classes)
        )
    def forward(self, x):
        return self.net(x)


def main():
    h5_path = "output/ustc_full.h5"
    idx_tr, _, idx_te = time_aware_split_indices(h5_path, 0.7, 0.15)
    rng = np.random.RandomState(42)
    idx_tr_sub = rng.choice(idx_tr, 4000, replace=False)
    idx_te_sub = rng.choice(idx_te, 1000, replace=False)

    X_tr_raw, Y_tr, classes = load_stat(h5_path, idx_tr_sub)
    X_te_raw, Y_te, _ = load_stat(h5_path, idx_te_sub)
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw).astype(np.float32)
    X_te = scaler.transform(X_te_raw).astype(np.float32)
    n_classes = int(max(Y_tr.max(), Y_te.max())) + 1
    print(f"USTC dense-sanity: train={len(X_tr)} test={len(X_te)} classes={n_classes}")

    torch.manual_seed(42)
    model = MLPBaseline(n_classes, in_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.from_numpy(X_tr).float()
    Yt = torch.from_numpy(Y_tr).long()
    Xv = torch.from_numpy(X_te).float()
    for ep in range(10):
        idx = torch.randperm(len(Xt))
        loss_sum, acc_sum, cnt = 0.0, 0.0, 0
        for i in range(0, len(Xt), 128):
            ib = idx[i:i+128]
            x = Xt[ib]; y = Yt[ib]
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(ib)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            cnt += len(ib)
        print(f"  ep {ep+1}/10  loss={loss_sum/cnt:.3f}  train_acc={acc_sum/cnt:.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(Xv).argmax(1).numpy()
    print(f"\nMLP on stat (dense, 32-d): test_acc={(pred == Y_te).mean():.4f}")


if __name__ == "__main__":
    main()