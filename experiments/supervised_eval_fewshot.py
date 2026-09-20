"""
supervised_eval_fewshot.py
===========================
加载 supervised pretrain encoder + ProtoNet，与 SSL/Random 在 5w5s 下对比。

完整对照：
  - supervised encoder (5 epoch supervised cross-entropy)
  - SSL SimCLR encoder (50 epoch SimCLR, medium augmentation)
  - Random init encoder (无预训练)
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

from src.data.load_h5 import H5FlowDataset, time_aware_split_indices
from src.data.episodes import EpisodeSampler
from src.models.encoder import TrafficEncoder
from src.models.fewshot import ProtoNet, attach_dataset

def _to_4tuple(items):
    p = torch.stack([it[0] for it in items])
    b = torch.stack([it[1] for it in items])
    s = torch.stack([it[2] for it in items])
    h = torch.stack([it[3] for it in items])
    return p, b, s, h

@torch.no_grad()
def protonet_eval(encoder, ds, n_way, k_shot, q_query, num_episodes, device, seed):
    model = ProtoNet(encoder).to(device).eval()
    attach_dataset(model, ds)
    sampler = EpisodeSampler(ds, n_way=n_way, k_shot=k_shot,
                             q_query=q_query, seed=seed)
    accs = []
    for ep in sampler.iter_episodes(num_episodes):
        s_idx, q_idx = ep["support_indices"], ep["query_indices"]
        s_y = ep["support_labels"].to(device)
        q_y = ep["query_labels"].to(device)
        s_x = tuple(t.to(device) for t in _to_4tuple([ds[int(i)] for i in s_idx]))
        q_x = tuple(t.to(device) for t in _to_4tuple([ds[int(i)] for i in q_idx]))
        logits = model(s_x, q_x, s_y)
        pred = logits.argmax(dim=-1)
        proto_to_global = s_y.unique()
        pred_global = proto_to_global[pred]
        accs.append(float((pred_global == q_y).float().mean().item()))
    return float(np.mean(accs)), float(np.std(accs))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cic", choices=["cic", "ustc", "iscx"])
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 142, 242])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    h5_path = f"output/{args.dataset}_full.h5"
    idx_tr, _, idx_te = time_aware_split_indices(h5_path, 0.7, 0.15)
    idx_all = np.concatenate([idx_tr, idx_te])
    rng = np.random.RandomState(42)
    idx_sub = rng.choice(idx_all, min(20000, len(idx_all)), replace=False)
    ds = H5FlowDataset(h5_path, indices=list(idx_sub), fit=True)
    print(f"[{args.dataset}] n={len(ds)} classes={len(ds.classes)} task={args.n_way}w{args.k_shot}s")

    encoders = {}
    # SSL SimCLR
    ssl_ckpt = f"output/checkpoints/{args.dataset}_simclr_ep50.pt"
    if os.path.exists(ssl_ckpt):
        enc = TrafficEncoder(embed_dim=128).to(args.device).eval()
        sd = torch.load(ssl_ckpt, map_location="cpu", weights_only=True)
        enc.load_state_dict(sd["encoder_state"])
        encoders["ssl_simclr"] = enc
    # Supervised (兼容 _ep3/_ep5 后缀)
    import glob
    sup_ckpts = sorted(glob.glob(f"output/checkpoints/{args.dataset}_supervised_ep*.pt"))
    if sup_ckpts:
        enc = TrafficEncoder(embed_dim=128).to(args.device).eval()
        sd = torch.load(sup_ckpts[-1], map_location="cpu", weights_only=True)
        enc.load_state_dict(sd["encoder_state"])
        encoders["supervised"] = enc
        print(f"  loaded supervised ckpt: {sup_ckpts[-1]}")
    # Random init
    encoders["random"] = TrafficEncoder(embed_dim=128).to(args.device).eval()

    out = {"dataset": args.dataset, "task": f"{args.n_way}w{args.k_shot}s",
           "episodes": args.episodes, "seeds": args.seeds, "results": {}}
    for tag, enc in encoders.items():
        means, stds = [], []
        for s in args.seeds:
            t0 = time.time()
            m, std = protonet_eval(enc, ds, args.n_way, args.k_shot, args.q_query,
                                   args.episodes, args.device, s)
            means.append(m); stds.append(std)
            print(f"  {tag} seed={s}: acc={m:.4f} ± {std:.4f} ({time.time()-t0:.1f}s)")
        mean_arr = np.array(means)
        std_arr = np.array(stds)
        out["results"][tag] = {
            "mean_per_seed": means,
            "std_per_seed": stds,
            "overall_mean": float(mean_arr.mean()),
            "overall_std": float(mean_arr.std()),
            "ci95_half_width": float(1.96 * mean_arr.std() / np.sqrt(len(means))),
        }
        print(f"  → {tag}: overall {out['results'][tag]['overall_mean']:.4f}")

    out_path = f"output/results/supervised_vs_ssl_fewshot_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")

if __name__ == "__main__":
    main()