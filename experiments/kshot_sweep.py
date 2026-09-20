"""
kshot_sweep.py
==============
K-shot 全谱扫描：1, 3, 5, 10 shot ProtoNet 评估，SSL vs Random 对比。
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.models.encoder import TrafficEncoder
from src.models.fewshot import ProtoNet, attach_dataset
from src.data.load_h5 import H5FlowDataset, time_aware_split_indices, stratified_split_indices
from src.data.episodes import EpisodeSampler

def load_encoder(ckpt, device):
    enc = TrafficEncoder(embed_dim=128)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        enc.load_state_dict(sd["encoder_state"])
    return enc.to(device).eval()

@torch.no_grad()
def eval_protonet(encoder, ds, n_way, k_shot, q_query, num_episodes, seed, device):
    model = ProtoNet(encoder).to(device)
    attach_dataset(model, ds)
    sampler = EpisodeSampler(ds, n_way=n_way, k_shot=k_shot, q_query=q_query, seed=seed)
    accs = []
    for ep in sampler.iter_episodes(num_episodes):
        s_idx = ep["support_indices"]; q_idx = ep["query_indices"]
        s_y = ep["support_labels"].to(device); q_y = ep["query_labels"].to(device)
        s_pkt = torch.stack([ds[int(i)][0] for i in s_idx]).to(device)
        s_br  = torch.stack([ds[int(i)][1] for i in s_idx]).to(device)
        s_st  = torch.stack([ds[int(i)][2] for i in s_idx]).to(device)
        s_hd  = torch.stack([ds[int(i)][3] for i in s_idx]).to(device)
        q_pkt = torch.stack([ds[int(i)][0] for i in q_idx]).to(device)
        q_br  = torch.stack([ds[int(i)][1] for i in q_idx]).to(device)
        q_st  = torch.stack([ds[int(i)][2] for i in q_idx]).to(device)
        q_hd  = torch.stack([ds[int(i)][3] for i in q_idx]).to(device)
        logits = model((s_pkt, s_br, s_st, s_hd), (q_pkt, q_br, q_st, q_hd), s_y)
        pred = logits.argmax(dim=-1)
        accs.append((pred == q_y).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cic", choices=["cic", "ustc", "iscx"])
    p.add_argument("--ssl_ckpt", default="output/checkpoints/{dataset}_simclr_ep50.pt")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--num_episodes", type=int, default=200)
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    h5_path = f"output/{args.dataset}_full.h5"
    if args.dataset == "iscx":
        idx_tr, _, _ = stratified_split_indices(h5_path, 0.7, 0.15, seed=42)
        n_way = 2
    else:
        idx_tr, _, _ = time_aware_split_indices(h5_path, 0.7, 0.15)
        n_way = args.n_way

    rng = np.random.RandomState(42)
    idx_tr_sub = rng.choice(idx_tr, min(args.max_samples, len(idx_tr)), replace=False)
    ds = H5FlowDataset(h5_path, indices=list(idx_tr_sub), fit=True)
    print(f"{args.dataset} n_way={n_way} train_n={len(ds)} classes={len(ds.classes)}")

    ssl_ckpt = args.ssl_ckpt.format(dataset=args.dataset)
    ssl_enc = load_encoder(ssl_ckpt if os.path.exists(ssl_ckpt) else None, args.device)
    rnd_enc = load_encoder(None, args.device)

    out = {"dataset": args.dataset, "n_way": n_way, "num_episodes": args.num_episodes,
           "num_seeds": args.num_seeds, "ks": args.ks, "results": {}}

    for tag, enc in [("ssl", ssl_enc), ("random", rnd_enc)]:
        out["results"][tag] = {}
        for k in args.ks:
            accs = []
            for s in range(args.num_seeds):
                seed = 42 + s * 100
                mean, std = eval_protonet(enc, ds, n_way, k, args.q_query,
                                          args.num_episodes, seed, args.device)
                accs.append(mean)
            agg_mean = float(np.mean(accs))
            agg_std = float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)
            out["results"][tag][f"{k}shot"] = {"mean": round(agg_mean, 4),
                                                "std": round(agg_std, 4),
                                                "raw": [round(a, 4) for a in accs]}
            print(f"  {tag:8s}  K={k:2d}shot  acc={agg_mean:.4f}±{agg_std:.4f}")

    out_path = f"output/results/kshot_sweep_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")

if __name__ == "__main__":
    main()
