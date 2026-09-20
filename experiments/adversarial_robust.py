"""
adversarial_robust.py
=====================
FGSM/PGD adversarial attack on TP-Net SSL encoder + LinearProbe.
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.models.encoder import TrafficEncoder
from src.models.fewshot import LinearProbing
from src.data.load_h5 import (
    H5FlowDataset, time_aware_split_indices, stratified_split_indices
)

def load_subset(h5_path, indices, n_classes):
    """加载子集（4 分支），pkt_seq 做 log1p + 标准化以保证攻击 epsilon 有意义。"""
    ds = H5FlowDataset(h5_path, indices=list(indices), fit=True)
    # 收集所有样本
    X_pkt, X_burst, X_stat, X_hand, Y = [], [], [], [], []
    for i in range(len(ds)):
        pkt, burst, stat, hand, y = ds[i]
        X_pkt.append(pkt); X_burst.append(burst); X_stat.append(stat); X_hand.append(hand); Y.append(int(y))
    X_pkt = torch.stack(X_pkt)
    # pkt_seq 归一化（log1p + z-score），与 §4.7 baseline 一致
    X_pkt_raw = X_pkt.numpy()
    X_pkt_raw = np.log1p(np.clip(X_pkt_raw, 0, None))
    flat = X_pkt_raw.reshape(-1, X_pkt_raw.shape[1] * X_pkt_raw.shape[2])
    nonzero_mask = (flat != 0).any(axis=1)  # 至少一个非零 channel
    mean = flat[nonzero_mask].mean(axis=0)
    std = flat[nonzero_mask].std(axis=0) + 1e-6
    X_pkt_norm = ((flat - mean) / std).reshape(X_pkt.shape).astype(np.float32)
    X_pkt = torch.from_numpy(X_pkt_norm)
    return (X_pkt, torch.stack(X_burst), torch.stack(X_stat),
            torch.stack(X_hand), torch.tensor(Y, dtype=torch.long),
            ds.classes, ds.scaler)

def fgsm_attack(encoder, classifier, pkt, burst, stat, hand, y, epsilon, device):
    """FGSM 单步 L∞ 攻击：x' = x + ε * sign(∇x L)."""
    encoder.eval(); classifier.eval()
    pkt = pkt.clone().detach().to(device).requires_grad_(True)
    burst = burst.to(device); stat = stat.to(device); hand = hand.to(device); y = y.to(device)
    feat = encoder(pkt, burst, stat, hand)
    logits = classifier(feat)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, pkt, retain_graph=False)[0]
    perturb = epsilon * grad.sign()
    pkt_adv = pkt.detach() + perturb
    pkt_adv = pkt_adv.detach()
    return pkt_adv

def pgd_attack(encoder, classifier, pkt, burst, stat, hand, y, epsilon, alpha, steps, device):
    """PGD 多步 L∞ 攻击。"""
    encoder.eval(); classifier.eval()
    pkt_orig = pkt.clone().detach().to(device)
    pkt_adv = pkt_orig.clone().requires_grad_(True)
    burst = burst.to(device); stat = stat.to(device); hand = hand.to(device); y = y.to(device)
    for _ in range(steps):
        feat = encoder(pkt_adv, burst, stat, hand)
        logits = classifier(feat)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, pkt_adv, retain_graph=False)[0]
        pkt_adv = pkt_adv.detach() + alpha * grad.sign()
        eta = torch.clamp(pkt_adv - pkt_orig, min=-epsilon, max=epsilon)
        pkt_adv = (pkt_orig + eta).detach().requires_grad_(True)
    return pkt_adv.detach()

def get_enc_clf(model):
    """从 LinearProbing 拆出 encoder + classifier。"""
    return model.encoder, model.classifier

def eval_adv(encoder, classifier, X_pkt, X_burst, X_stat, X_hand, Y,
             attack_fn, epsilon, batch_size=64, device="cpu",
             pgd_alpha=0.02, pgd_steps=10):
    """评估 adversarial accuracy。"""
    encoder.eval(); classifier.eval()
    n = len(X_pkt)
    correct = 0
    for i in range(0, n, batch_size):
        ip = i; jp = min(i + batch_size, n)
        pkt = X_pkt[ip:jp]; burst = X_burst[ip:jp]
        stat = X_stat[ip:jp]; hand = X_hand[ip:jp]; y = Y[ip:jp]
        if attack_fn is None:
            pkt_in = pkt.to(device)
        else:
            pkt_in = attack_fn(encoder, classifier, pkt, burst, stat, hand, y, epsilon, device)
        with torch.no_grad():
            feat = encoder(pkt_in, burst.to(device), stat.to(device), hand.to(device))
            logits = classifier(feat)
        pred = logits.argmax(1).cpu()
        correct += (pred == y).sum().item()
    return correct / n

def train_lp(encoder_ckpt, X_pkt_tr, X_burst_tr, X_stat_tr, X_hand_tr, Y_tr,
             n_classes, device, epochs=10, lr=1e-2):
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
        print(f"  ep {ep+1}/{epochs}  loss={loss_sum/cnt:.3f}")
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cic", choices=["cic", "ustc", "iscx"])
    p.add_argument("--encoder_ckpt", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--n_train", type=int, default=10000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--epsilon", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--pgd_alpha", type=float, default=0.02)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    h5_path = f"output/{args.dataset}_full.h5"
    if args.dataset == "iscx":
        idx_tr, _, idx_te = stratified_split_indices(h5_path, 0.7, 0.15, seed=42)
    else:
        idx_tr, _, idx_te = time_aware_split_indices(h5_path, 0.7, 0.15)

    rng = np.random.RandomState(42)
    idx_tr_sub = rng.choice(idx_tr, args.n_train, replace=False)
    idx_te_sub = rng.choice(idx_te, args.n_test, replace=False)

    print(f"[{args.dataset}] 加载 train={len(idx_tr_sub)} test={len(idx_te_sub)}")
    Xp_tr, Xb_tr, Xs_tr, Xh_tr, Y_tr, classes_tr, _ = load_subset(h5_path, idx_tr_sub, None)
    Xp_te, Xb_te, Xs_te, Xh_te, Y_te, _, scaler_te = load_subset(h5_path, idx_te_sub, None)
    # 测试集复用训练 scaler
    n_classes = int(max(Y_tr.max(), Y_te.max())) + 1
    print(f"  classes={n_classes}")

    # SSL encoder + LinearProbe
    print(f"\n[SSL] 训练 LinearProbe (encoder={args.encoder_ckpt})…")
    ssl_model = train_lp(args.encoder_ckpt, Xp_tr, Xb_tr, Xs_tr, Xh_tr, Y_tr,
                         n_classes, args.device, epochs=args.epochs)

    # Random encoder + LinearProbe（对照组）
    print(f"\n[Random] 训练 LinearProbe (encoder=无 SSL 随机初始化)…")
    rnd_model = train_lp(None, Xp_tr, Xb_tr, Xs_tr, Xh_tr, Y_tr,
                         n_classes, args.device, epochs=args.epochs)

    results = {"dataset": args.dataset, "n_classes": n_classes, "epsilons": args.epsilon,
               "pgd_steps": args.pgd_steps, "pgd_alpha": args.pgd_alpha}

    for tag, model in [("ssl", ssl_model), ("random", rnd_model)]:
        print(f"\n=== {tag} ===")
        enc, clf = get_enc_clf(model)
        results[tag] = {}
        # Clean accuracy
        clean_acc = eval_adv(enc, clf, Xp_te, Xb_te, Xs_te, Xh_te, Y_te,
                             attack_fn=None, epsilon=0, device=args.device)
        results[tag]["clean"] = round(clean_acc, 4)
        print(f"  clean acc = {clean_acc:.4f}")
        for eps in args.epsilon:
            # FGSM
            fgsm_acc = eval_adv(enc, clf, Xp_te, Xb_te, Xs_te, Xh_te, Y_te,
                                attack_fn=fgsm_attack, epsilon=eps, device=args.device)
            results[tag][f"fgsm_eps{eps}"] = round(fgsm_acc, 4)
            print(f"  FGSM eps={eps:.2f} acc = {fgsm_acc:.4f}")
            # PGD
            pgd_acc = eval_adv(enc, clf, Xp_te, Xb_te, Xs_te, Xh_te, Y_te,
                               attack_fn=lambda e,c,p,b,s,h,y,ep,d: pgd_attack(
                                   e,c,p,b,s,h,y,ep, args.pgd_alpha, args.pgd_steps, d),
                               epsilon=eps, device=args.device)
            results[tag][f"pgd_eps{eps}"] = round(pgd_acc, 4)
            print(f"  PGD eps={eps:.2f} steps={args.pgd_steps} acc = {pgd_acc:.4f}")

    out_path = f"output/results/adv_robust_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")

if __name__ == "__main__":
    main()
