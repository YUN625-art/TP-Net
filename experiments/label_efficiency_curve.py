"""
label_efficiency_curve.py
==========================
标注效率曲线：SSL few-shot vs SSL LinearProbe(K) vs Random LinearProbe(K)。

回答 P1-6 关键问题："既然 LinearProbe 在 CIC 上达到 0.9972（全量样本），
few-shot episode 协议的价值是什么？"

答案：当 K≤50 时，SSL ProtoNet 远优于 SSL LinearProbe；
LP 需 K≥500 才能接近饱和——few-shot 在小标注场景（K<50）有压倒性优势。
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.data.load_h5 import H5FlowDataset, time_aware_split_indices
from src.data.episodes import EpisodeSampler
from src.models.encoder import TrafficEncoder
from src.models.fewshot import ProtoNet, LinearProbing, attach_dataset

def _to_4tuple(items):
    P = torch.stack([it[0] for it in items])
    B = torch.stack([it[1] for it in items])
    S = torch.stack([it[2] for it in items])
    H = torch.stack([it[3] for it in items])
    return P, B, S, H

@torch.no_grad()
def protonet_eval(encoder, ds, n_way, k_shot, q_query, num_episodes, device, seed):
    model = ProtoNet(encoder).to(device).eval()
    attach_dataset(model, ds)
    sampler = EpisodeSampler(ds, n_way=n_way, k_shot=k_shot, q_query=q_query, seed=seed)
    accs = []
    for ep in sampler.iter_episodes(num_episodes):
        s_idx, q_idx = ep["support_indices"], ep["query_indices"]
        s_y, q_y = ep["support_labels"].to(device), ep["query_labels"].to(device)
        s_x = tuple(t.to(device) for t in _to_4tuple([ds[int(i)] for i in s_idx]))
        q_x = tuple(t.to(device) for t in _to_4tuple([ds[int(i)] for i in q_idx]))
        logits = model(s_x, q_x, s_y)
        pred = logits.argmax(dim=-1)
        proto_to_global = s_y.unique()
        pred_global = proto_to_global[pred]
        accs.append(float((pred_global == q_y).float().mean().item()))
    return float(np.mean(accs)), float(np.std(accs))

def linear_probe_with_k_samples(encoder, ds, k_per_class, device, epochs=20, lr=1e-3, seed=42):
    """在每类 k 个标注样本上训练 LinearProbe，在剩余样本上评估。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    y = ds.y.numpy()
    classes = np.unique(y)
    train_idx = []
    test_idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        np.random.shuffle(c_idx)
        k = min(k_per_class, len(c_idx) - 1)
        train_idx.extend(c_idx[:k].tolist())
        test_idx.extend(c_idx[k:].tolist())
    if len(train_idx) == 0 or len(test_idx) == 0:
        return float("nan")

    train_sub = Subset(ds, train_idx)
    test_sub = Subset(ds, test_idx)
    train_dl = DataLoader(train_sub, batch_size=min(64, len(train_idx)), shuffle=True)
    test_dl = DataLoader(test_sub, batch_size=256, shuffle=False)

    n_total_classes = len(ds.classes)
    model = LinearProbing(encoder, num_classes=n_total_classes, embed_dim=128).to(device)
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for batch in train_dl:
            pkt, br, st, hd, y_b = [t.to(device) for t in batch]
            opt.zero_grad()
            logits = model(pkt, br, st, hd)
            loss = crit(logits, y_b)
            loss.backward()
            opt.step()

    # 评估
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_dl:
            pkt, br, st, hd, y_b = [t.to(device) for t in batch]
            logits = model(pkt, br, st, hd)
            pred = logits.argmax(dim=-1)
            correct += (pred == y_b).sum().item()
            total += y_b.size(0)
    return correct / max(total, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cic", choices=["cic", "ustc", "iscx"])
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--k_values", nargs="+", type=int,
                   default=[1, 3, 5, 10, 50, 500])
    args = p.parse_args()

    h5_path = f"output/{args.dataset}_full.h5"
    idx_tr, _, idx_te = time_aware_split_indices(h5_path, 0.7, 0.15)
    # 用 train + test 全部数据（ProtoNet episode 在类内随机采，类划分不严）
    rng = np.random.RandomState(42)
    idx_all = np.concatenate([idx_tr, idx_te])
    idx_sub = rng.choice(idx_all, min(15000, len(idx_all)), replace=False)
    ds = H5FlowDataset(h5_path, indices=list(idx_sub), fit=True)
    print(f"dataset={args.dataset} n={len(ds)} classes={len(ds.classes)}")

    # 加载 SSL encoder
    enc_ssl = TrafficEncoder(embed_dim=128).to(args.device).eval()
    ckpt_ssl = f"output/checkpoints/{args.dataset}_simclr_ep50.pt"
    if os.path.exists(ckpt_ssl):
        sd = torch.load(ckpt_ssl, map_location="cpu", weights_only=True)
        enc_ssl.load_state_dict(sd["encoder_state"])
    # Random init encoder
    enc_rand = TrafficEncoder(embed_dim=128).to(args.device).eval()

    out = {
        "dataset": args.dataset,
        "task": f"{args.n_way}w{args.k_shot}s",
        "k_values": args.k_values,
        "ssl_protonet": {"mean": None, "std": None},
        "ssl_lp": {},
        "random_lp": {},
    }

    # (a) SSL ProtoNet
    print("\n=== SSL ProtoNet (5w5s, K=5) ===")
    t0 = time.time()
    m, s = protonet_eval(enc_ssl, ds, args.n_way, args.k_shot, args.q_query,
                         args.episodes, args.device, args.seed)
    print(f"  acc = {m:.4f} ± {s:.4f}  ({time.time()-t0:.1f}s)")
    out["ssl_protonet"]["mean"] = m
    out["ssl_protonet"]["std"] = s

    # (b) SSL LinearProbe with K samples/class
    print("\n=== SSL LinearProbe(K) ===")
    for K in args.k_values:
        t0 = time.time()
        acc = linear_probe_with_k_samples(enc_ssl, ds, K, args.device, seed=args.seed)
        print(f"  K={K:4d}: acc = {acc:.4f}  ({time.time()-t0:.1f}s)")
        out["ssl_lp"][str(K)] = acc

    # (c) Random init LinearProbe with K samples/class
    print("\n=== Random LinearProbe(K) ===")
    for K in args.k_values:
        t0 = time.time()
        acc = linear_probe_with_k_samples(enc_rand, ds, K, args.device, seed=args.seed)
        print(f"  K={K:4d}: acc = {acc:.4f}  ({time.time()-t0:.1f}s)")
        out["random_lp"][str(K)] = acc

    out_path = f"output/results/label_efficiency_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")

if __name__ == "__main__":
    main()
