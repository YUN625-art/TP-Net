"""
baseline_supervised_check.py
==============================
§4.7 健全性检查：6 个 baseline 在全监督（5-shot fine-tune vs 充足标注 fine-tune）下能达到多少？
USTC/CIC 各跑一次，输出到 output/results/baseline_supervised_check.json。

直接读 h5 数组，避免 H5FlowDataset 的 sort/unsort 路径可能引入的标签错位。
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.models.baselines import (
    FSNet, YaTCNet, TFEGraph, ETBERTLite, NetMambaLite, NetConvLite, count_params
)
from src.data.load_h5 import (
    stratified_split_indices, time_aware_split_indices
)

def build_model(name: str, n_classes: int) -> nn.Module:
    if name == "fsnet":       return FSNet(n_classes)
    if name == "yatc":        return YaTCNet(n_classes)
    if name == "tfegnn":      return TFEGraph(n_classes)
    if name == "etbert":      return ETBERTLite(n_classes)
    if name == "netmamba":    return NetMambaLite(n_classes)
    if name == "netconv":     return NetConvLite(n_classes)
    raise ValueError(name)

def load_split(h5_path, indices):
    """直接读 h5 数组，避免 H5FlowDataset 的 sort/unsort 副作用。
    h5py fancy indexing 要求单调递增，先 sort 取数据再 unsort 还原。
    pkt_seq = (pkt_len, iat) 未归一化 + 99.4% 是 padding；
    对每个 channel 做 log1p + StandardScaler（基于训练子集），测试集复用同一 scaler。
    """
    indices = np.asarray(indices, dtype=np.int64)
    sort_perm = np.argsort(indices, kind="stable")
    sorted_idx = indices[sort_perm]
    with h5py.File(h5_path, "r") as f:
        X_sorted = f["pkt_seq"][sorted_idx]   # (n, 2, 200)
        labels_all = f["label"].asstr()[:]
        y_str_sorted = labels_all[sorted_idx]
    cat = pd.Categorical(y_str_sorted)
    y_sorted = np.array(cat.codes, dtype=np.int64)
    # 还原原始 indices 顺序
    unsort = np.argsort(sort_perm, kind="stable")
    X = np.asarray(X_sorted[unsort], dtype=np.float32)
    y = y_sorted[unsort]
    classes = list(cat.categories)
    X = np.where(np.isfinite(X), 0, X).astype(np.float32)
    # log1p 压缩长尾（packet length 0-1475，IAT 0-几千ms）
    X = np.log1p(np.clip(X, 0, None))
    return X, y, classes

def fit_normalizer(X_train):
    """对 pkt_seq (N, 2, T) 每个 channel 拟合 mean/std（基于非零 token）。"""
    means, stds = [], []
    for c in range(X_train.shape[1]):
        vals = X_train[:, c, :].reshape(-1)
        nz = vals[vals > 0]
        means.append(float(nz.mean()) if len(nz) else 0.0)
        stds.append(float(nz.std() + 1e-6) if len(nz) else 1.0)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)

def apply_normalizer(X, mean, std):
    Y = X.copy()
    for c in range(X.shape[1]):
        Y[:, c, :] = (X[:, c, :] - mean[c]) / std[c]
    return Y.astype(np.float32)

def train_supervised(model, X_tr, Y_tr, X_te, Y_te, epochs=5, batch=128, lr=1e-3, device="cpu"):
    """标准监督训练，返回测试集 top-1 准确率。"""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    Xt = torch.from_numpy(X_tr).float()
    Yt = torch.from_numpy(Y_tr).long()
    Xv = torch.from_numpy(X_te).float().to(device)
    n = len(Xt)
    last_train_acc = 0.0
    for ep in range(epochs):
        idx = torch.randperm(n)
        loss_sum, acc_sum, cnt = 0.0, 0.0, 0
        for i in range(0, n, batch):
            ib = idx[i:i+batch]
            x = Xt[ib].to(device)
            y = Yt[ib].to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(ib)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            cnt += len(ib)
        last_train_acc = acc_sum / cnt
        train_loss = loss_sum / cnt
        print(f"    epoch {ep+1}/{epochs}  loss={train_loss:.4f}  train_acc={last_train_acc:.4f}")
    # eval
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xv), 256):
            preds.append(model(Xv[i:i+256]).argmax(1).cpu().numpy())
        pred = np.concatenate(preds)
    test_acc = (pred == Y_te).mean()
    return float(last_train_acc), float(test_acc)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ustc", choices=["cic", "ustc", "iscx"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--n_samples", type=int, default=30000)
    p.add_argument("--models", nargs="+",
                   default=["fsnet", "yatc", "tfegnn", "etbert", "netmamba", "netconv"])
    args = p.parse_args()

    h5_path = f"output/{args.dataset}_full.h5"
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    if args.dataset == "iscx":
        idx_tr, idx_va, idx_te = stratified_split_indices(
            h5_path, train_ratio=0.7, val_ratio=0.15, seed=42)
    else:
        idx_tr, idx_va, idx_te = time_aware_split_indices(
            h5_path, train_ratio=0.7, val_ratio=0.15)

    rng = np.random.RandomState(42)
    idx_tr_sub = rng.choice(idx_tr, size=min(args.n_samples, len(idx_tr)), replace=False)
    idx_te_sub = rng.choice(idx_te, size=min(args.n_samples // 4, len(idx_te)), replace=False)

    print(f"[{args.dataset}] 加载训练 {len(idx_tr_sub)} + 测试 {len(idx_te_sub)} 样本…")
    X_tr, Y_tr, _ = load_split(h5_path, idx_tr_sub)
    X_te, Y_te, _ = load_split(h5_path, idx_te_sub)
    # pkt_seq 归一化：训练集拟合 mean/std，测试集复用
    mean, std = fit_normalizer(X_tr)
    X_tr = apply_normalizer(X_tr, mean, std)
    X_te = apply_normalizer(X_te, mean, std)
    n_classes = int(max(Y_tr.max(), Y_te.max())) + 1
    print(f"数据集: {args.dataset}  训练: {len(X_tr)}  测试: {len(X_te)}  类别: {n_classes}")
    print(f"X_tr.shape={X_tr.shape}  dtype={X_tr.dtype}  "
          f"Y_tr 分布: min={Y_tr.min()} max={Y_tr.max()} unique={len(np.unique(Y_tr))}")

    out = {"dataset": args.dataset, "epochs": args.epochs,
           "n_train_samples": len(X_tr), "n_test_samples": len(X_te),
           "results": {}}

    for mname in args.models:
        torch.manual_seed(42)
        np.random.seed(42)
        model = build_model(mname, n_classes)
        t0 = time.time()
        try:
            tr_acc, te_acc = train_supervised(
                model, X_tr, Y_tr, X_te, Y_te, epochs=args.epochs)
        except Exception as e:
            print(f"  {mname} 训练失败: {e}")
            tr_acc, te_acc = 0.0, 0.0
        dt = time.time() - t0
        out["results"][mname] = {
            "params": count_params(model),
            "train_acc": round(tr_acc, 4),
            "test_acc": round(te_acc, 4),
            "time_sec": round(dt, 1),
        }
        print(f"  {mname:10s}  params={count_params(model):>9,}  "
              f"train={tr_acc:.4f}  test={te_acc:.4f}  ({dt:.0f}s)")

    out_path = f"output/results/baseline_supervised_check_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")

if __name__ == "__main__":
    main()
