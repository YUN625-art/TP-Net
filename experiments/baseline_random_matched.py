"""
baseline_random_matched.py
==========================
与 §4.2 SSL 主表协议完全对齐的 Random baseline 重测。
协议：5 seeds × 600 episodes ProtoNet，stratified subsample 20k，与 §4.2 一致。
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
from src.data.load_h5 import H5FlowDataset, _stratified_subsample
from src.data.episodes import EpisodeSampler

@torch.no_grad()
def eval_protonet(encoder, ds, n_way, k_shot, q_query, num_episodes, seed, device):
    model = ProtoNet(encoder).to(device).eval()
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
    return accs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cic", choices=["cic", "ustc", "iscx"])
    p.add_argument("--n_way", type=int, default=None)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--num_episodes", type=int, default=600)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.n_way is None:
        args.n_way = 2 if args.dataset == "iscx" else 5

    import h5py
    h5_path = f"output/{args.dataset}_full.h5"
    with h5py.File(h5_path, "r") as h:
        n = h["stat"].shape[0]
        labels = h["label"].asstr()[:]
    indices = _stratified_subsample(list(range(n)), labels, args.max_samples, seed=42)
    ds = H5FlowDataset(h5_path, indices=list(indices), fit=True)
    print(f"dataset={args.dataset} n={len(ds)} classes={len(ds.classes)} stratified 20k")

    enc = TrafficEncoder(embed_dim=128).to(args.device).eval()
    print("[使用 random init encoder, 无 ckpt 加载]")

    out = {"dataset": args.dataset, "n_way": args.n_way, "k_shot": args.k_shot,
           "q_query": args.q_query, "num_episodes": args.num_episodes,
           "seeds": [], "aggregate": {}}

    all_per_seed_means = []
    all_episodes = []  # pool all episodes across seeds
    seed_eps = []  # store per-episode data per seed
    for s in range(args.seeds):
        seed = 42 + s * 100
        t0 = time.time()
        accs = eval_protonet(enc, ds, args.n_way, args.k_shot, args.q_query,
                             args.num_episodes, seed, args.device)
        elapsed = time.time() - t0
        all_per_seed_means.append(float(np.mean(accs)))
        all_episodes.extend(accs)
        seed_eps.append(accs)
        out["seeds"].append({"seed": seed, "mean": float(np.mean(accs)),
                             "std": float(np.std(accs, ddof=1)),
                             "n": len(accs), "elapsed_sec": round(elapsed, 1),
                             "raw": accs})  # per-episode raw data
        print(f"  seed={seed}: mean={np.mean(accs):.4f} std={np.std(accs, ddof=1):.4f} "
              f"({elapsed:.1f}s)")

    out["aggregate"] = {
        "mean": float(np.mean(all_per_seed_means)),
        "std_seed_mean": float(np.std(all_per_seed_means, ddof=1)),
        "n_seeds": len(all_per_seed_means),
        "raw_seed_means": all_per_seed_means,
    }
    print(f"\naggregate mean: {out['aggregate']['mean']:.4f} ± {out['aggregate']['std_seed_mean']:.4f}")

    out_path = f"output/results/baseline_random_matched_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[已保存] {out_path}")

if __name__ == "__main__":
    main()
