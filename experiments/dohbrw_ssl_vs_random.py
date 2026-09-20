"""
dohbrw_ssl_vs_random.py
=======================
DoHBrw-2020 SSL vs Random LinearProbe 对比，验证 SSL 在加密流量场景的普适性。
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.models.encoder import TrafficEncoder
from src.models.fewshot import LinearProbing
from src.data.load_h5 import H5FlowDataset, stratified_split_indices

def load_subset(h5_path, indices, scaler=None, fit=False):
    """加载子集（4 分支）。"""
    ds = H5FlowDataset(h5_path, indices=list(indices), scaler=scaler, fit=fit)
    X_pkt, X_burst, X_stat, X_hand, Y = [], [], [], [], []
    for i in range(len(ds)):
        pkt, burst, stat, hand, y = ds[i]
        X_pkt.append(pkt); X_burst.append(burst); X_stat.append(stat); X_hand.append(hand); Y.append(int(y))
    return (torch.stack(X_pkt), torch.stack(X_burst), torch.stack(X_stat),
            torch.stack(X_hand), torch.tensor(Y, dtype=torch.long),
            ds.scaler, ds.classes)

def train_lp(encoder_ckpt, X_pkt_tr, X_burst_tr, X_stat_tr, X_hand_tr, Y_tr,
             n_classes, device, epochs=20, lr=1e-3):
    """训练 LinearProbe。"""
    enc = TrafficEncoder(embed_dim=128)
    if encoder_ckpt:
        sd = torch.load(encoder_ckpt, map_location="cpu", weights_only=True)
        enc.load_state_dict(sd["encoder_state"])
    model = LinearProbing(enc, num_classes=n_classes, embed_dim=128).to(device)
    opt = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    n = len(X_pkt_tr)
    for ep in range(epochs):
        idx = torch.randperm(n)
        loss_sum, cnt = 0.0, 0
        for i in range(0, n, 64):
            ib = idx[i:i+64]
            pkt = X_pkt_tr[ib].to(device); burst = X_burst_tr[ib].to(device)
            stat = X_stat_tr[ib].to(device); hand = X_hand_tr[ib].to(device); y = Y_tr[ib].to(device)
            opt.zero_grad()
            logits = model(pkt, burst, stat, hand)
            loss = F.cross_entropy(logits, y)
            loss.backward(); opt.step()
            loss_sum += loss.item()*len(ib); cnt += len(ib)
    return model

def eval_lp(model, X_pkt, X_burst, X_stat, X_hand, Y, device="cpu", batch=64):
    model.eval()
    n = len(X_pkt)
    preds = []
    with torch.no_grad():
        for i in range(0, n, batch):
            pkt = X_pkt[i:i+batch].to(device)
            burst = X_burst[i:i+batch].to(device)
            stat = X_stat[i:i+batch].to(device)
            hand = X_hand[i:i+batch].to(device)
            logits = model(pkt, burst, stat, hand)
            preds.append(logits.argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    acc = (pred == Y.numpy()).mean()
    f1 = f1_score(Y.numpy(), pred, average="macro", zero_division=0)
    prec = precision_score(Y.numpy(), pred, average="macro", zero_division=0)
    rec = recall_score(Y.numpy(), pred, average="macro", zero_division=0)
    return {"acc": float(acc), "f1": float(f1),
            "precision": float(prec), "recall": float(rec)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_ckpt", default="output/checkpoints/dohbrw_simclr_ep30.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    h5_path = "output/dohbrw_full.h5"
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    idx_tr, idx_va, idx_te = stratified_split_indices(h5_path, 0.7, 0.15, seed=42)
    print(f"DoHBrw split: train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")

    print("加载训练子集…")
    Xp_tr, Xb_tr, Xs_tr, Xh_tr, Y_tr, scaler_tr, classes = load_subset(h5_path, idx_tr, fit=True)
    print(f"  classes={classes}  train={len(Xp_tr)}")
    print("加载测试子集…")
    Xp_te, Xb_te, Xs_te, Xh_te, Y_te, _, _ = load_subset(h5_path, idx_te, scaler=scaler_tr, fit=False)
    n_classes = len(classes)

    results = {"dataset": "DoHBrw-2020", "n_classes": n_classes, "n_train": len(Xp_tr),
               "n_test": len(Xp_te), "epochs": args.epochs}

    # SSL encoder + LinearProbe
    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        print(f"\n[SSL] ckpt={args.encoder_ckpt}")
        t0 = time.time()
        ssl_model = train_lp(args.encoder_ckpt, Xp_tr, Xb_tr, Xs_tr, Xh_tr, Y_tr,
                             n_classes, args.device, epochs=args.epochs)
        ssl_metrics = eval_lp(ssl_model, Xp_te, Xb_te, Xs_te, Xh_te, Y_te, args.device)
        ssl_metrics["train_sec"] = round(time.time() - t0, 1)
        results["ssl"] = ssl_metrics
        print(f"  acc={ssl_metrics['acc']:.4f}  f1={ssl_metrics['f1']:.4f}  "
              f"({ssl_metrics['train_sec']:.0f}s)")
    else:
        print(f"\n[SSL] ckpt={args.encoder_ckpt} 不存在，跳过")

    # Random encoder + LinearProbe
    print("\n[Random] encoder=无 SSL 随机初始化")
    t0 = time.time()
    rnd_model = train_lp(None, Xp_tr, Xb_tr, Xs_tr, Xh_tr, Y_tr,
                         n_classes, args.device, epochs=args.epochs)
    rnd_metrics = eval_lp(rnd_model, Xp_te, Xb_te, Xs_te, Xh_te, Y_te, args.device)
    rnd_metrics["train_sec"] = round(time.time() - t0, 1)
    results["random"] = rnd_metrics
    print(f"  acc={rnd_metrics['acc']:.4f}  f1={rnd_metrics['f1']:.4f}  "
          f"({rnd_metrics['train_sec']:.0f}s)")

    # 改善统计
    if "ssl" in results:
        results["ssl_minus_random_acc"] = round(results["ssl"]["acc"] - results["random"]["acc"], 4)
        results["ssl_minus_random_f1"] = round(results["ssl"]["f1"] - results["random"]["f1"], 4)

    out_path = "output/results/dohbrw_ssl_vs_random.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")
    if "ssl" in results:
        print(f"SSL - Random: Δacc={results['ssl_minus_random_acc']:+.4f}  "
              f"Δf1={results['ssl_minus_random_f1']:+.4f}")

if __name__ == "__main__":
    main()
