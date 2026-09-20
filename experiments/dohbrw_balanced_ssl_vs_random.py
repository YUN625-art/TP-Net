"""
dohbrw_balanced_ssl_vs_random.py
================================
DoHBrw-2020 balanced 子集 SSL vs Random 2-way ProtoNet 对比。

原始分布：Benign 12098 / Malicious 14（极度不平衡）。
Balanced 子集：14 Malicious + ~280 Benign（1:20 平衡），足够支持
2-way K-shot 评估（malicious 类 K=5: 5 support + 9 query）。

SSL encoder：复用 CIC 预训练的 SimCLR encoder（DoHBrw 数据量太小，
            无法独立 SSL 预训练。这是"跨数据集预训练后直接用于加密任务"
            的最严格测试场景）。
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.models.encoder import TrafficEncoder
from src.models.fewshot import ProtoNet, attach_dataset
from src.data.load_h5 import H5FlowDataset
from src.data.episodes import EpisodeSampler

def build_balanced_subset(h5_path, n_benign_per_malicious=20, seed=42):
    """构造 balanced 子集：malicious 全保留 + 下采样 benign。

    Returns:
        indices (list[int]): 子集全局索引（用于 H5FlowDataset）
        cls_to_local (dict[int, list[int]]): 类→子集内局部索引
        classes (list[str]): 类名
    """
    import h5py
    with h5py.File(h5_path, "r") as h:
        labels = h["label"].asstr()[:]

    mal_idx = np.where(labels == "Malicious")[0]
    ben_idx = np.where(labels == "Benign")[0]
    n_mal = len(mal_idx)
    n_ben_target = n_mal * n_benign_per_malicious

    rng = np.random.RandomState(seed)
    ben_sub = rng.choice(ben_idx, min(n_ben_target, len(ben_idx)), replace=False)

    # 合并为子集（malicious 在前，benign 在后）
    indices = np.concatenate([mal_idx, ben_sub]).tolist()
    # 子集内局部索引（malicious: 0..n_mal-1, benign: n_mal..n_mal+n_ben-1）
    cls_to_local = {
        0: list(range(0, n_mal)),
        1: list(range(n_mal, n_mal + len(ben_sub))),
    }
    classes = ["Malicious", "Benign"]
    print(f"balanced subset: malicious={n_mal}, benign={len(ben_sub)}  "
          f"(ratio 1:{len(ben_sub)/n_mal:.1f})")
    return indices, cls_to_local, classes

@torch.no_grad()
def eval_protonet(encoder, ds, cls_to_idx, n_way, k_shot, q_query,
                  num_episodes, seed, device):
    """评估 ProtoNet（n_way 通常为 2）。"""
    model = ProtoNet(encoder).to(device)
    attach_dataset(model, ds)
    sampler = EpisodeSampler(ds, n_way=n_way, k_shot=k_shot, q_query=q_query,
                             seed=seed, cls_to_idx=cls_to_idx,
                             num_classes=n_way)
    accs = []
    for ep in sampler.iter_episodes(num_episodes):
        s_idx = ep["support_indices"]; q_idx = ep["query_indices"]
        s_y = ep["support_labels"].to(device); q_y = ep["query_labels"].to(device)
        s_pkt = torch.stack([ds[int(i)][0] for i in s_idx]).to(device)
        s_br = torch.stack([ds[int(i)][1] for i in s_idx]).to(device)
        s_st = torch.stack([ds[int(i)][2] for i in s_idx]).to(device)
        s_hd = torch.stack([ds[int(i)][3] for i in s_idx]).to(device)
        q_pkt = torch.stack([ds[int(i)][0] for i in q_idx]).to(device)
        q_br = torch.stack([ds[int(i)][1] for i in q_idx]).to(device)
        q_st = torch.stack([ds[int(i)][2] for i in q_idx]).to(device)
        q_hd = torch.stack([ds[int(i)][3] for i in q_idx]).to(device)
        logits = model((s_pkt, s_br, s_st, s_hd), (q_pkt, q_br, q_st, q_hd), s_y)
        pred = logits.argmax(dim=-1)
        accs.append((pred == q_y).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)

def load_encoder(ckpt, device):
    enc = TrafficEncoder(embed_dim=128)
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        enc.load_state_dict(sd["encoder_state"])
    return enc.to(device).eval()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="output/dohbrw_full.h5")
    p.add_argument("--ssl_ckpt", default="output/checkpoints/cic_simclr_ep50.pt",
                   help="跨数据集 SSL encoder（DoHBrw 14 条恶意无法独立 SSL 预训练）")
    p.add_argument("--n_benign_per_malicious", type=int, default=20,
                   help="balanced 子集 benign/malicious 比例（默认 1:20）")
    p.add_argument("--n_way", type=int, default=2)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=9)
    p.add_argument("--num_episodes", type=int, default=200)
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if not os.path.exists(args.h5):
        raise FileNotFoundError(args.h5)

    # 1. 构造 balanced 子集
    indices, cls_to_local, classes = build_balanced_subset(
        args.h5, args.n_benign_per_malicious, seed=42)
    ds = H5FlowDataset(args.h5, indices=indices, fit=True)
    print(f"  dataset.n={len(ds)}  classes={classes}")

    # 2. SSL encoder + Random encoder
    ssl_enc = load_encoder(args.ssl_ckpt, args.device)
    rnd_enc = load_encoder(None, args.device)

    results = {
        "dataset": "DoHBrw-2020 (balanced 1:N)",
        "n_way": args.n_way, "k_shot": args.k_shot, "q_query": args.q_query,
        "num_episodes": args.num_episodes, "num_seeds": args.num_seeds,
        "n_malicious": len(cls_to_local[0]),
        "n_benign": len(cls_to_local[1]),
        "ssl_ckpt": args.ssl_ckpt,
        "results": {},
    }

    for tag, enc in [("ssl", ssl_enc), ("random", rnd_enc)]:
        accs = []
        t0 = time.time()
        for s in range(args.num_seeds):
            seed = 42 + s * 100
            mean, std = eval_protonet(enc, ds, cls_to_local, args.n_way,
                                      args.k_shot, args.q_query,
                                      args.num_episodes, seed, args.device)
            accs.append(mean)
            print(f"  {tag:8s} seed={seed} acc={mean:.4f}")
        agg_mean = float(np.mean(accs))
        agg_std = float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)
        results["results"][tag] = {
            "mean": round(agg_mean, 4),
            "std": round(agg_std, 4),
            "raw": [round(a, 4) for a in accs],
            "eval_sec": round(time.time() - t0, 1),
        }
        print(f"  >>> {tag}: {agg_mean:.4f} ± {agg_std:.4f}")

    # 改善统计
    if "ssl" in results["results"] and "random" in results["results"]:
        d = (results["results"]["ssl"]["mean"]
             - results["results"]["random"]["mean"])
        results["ssl_minus_random"] = round(d, 4)

    out_path = "output/results/dohbrw_balanced_ssl_vs_random.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")
    if "ssl_minus_random" in results:
        print(f"SSL - Random: Δacc={results['ssl_minus_random']:+.4f}")

if __name__ == "__main__":
    main()
