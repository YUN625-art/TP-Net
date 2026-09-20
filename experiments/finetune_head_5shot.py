"""
finetune_head_5shot.py
======================
5-shot fine-tune head 评估：SSL encoder 冻结 + sklearn LogReg on K support
端到端训 head → 在 Q query 上预测。

三数据集 × SSL encoder 5 seed × 600 ep × LogReg(sklearn, C=1e3) head
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.data.load_h5 import H5FlowDataset, _stratified_subsample
from src.data.episodes import EpisodeSampler
from src.models.encoder import TrafficEncoder
from src.models.fewshot import ProtoNet, attach_dataset

@torch.no_grad()
def encode_batch(encoder, ds, indices, device):
    """提取一批样本的 128-D embedding。"""
    pkts = torch.stack([ds[int(i)][0] for i in indices]).to(device)
    brs  = torch.stack([ds[int(i)][1] for i in indices]).to(device)
    sts  = torch.stack([ds[int(i)][2] for i in indices]).to(device)
    hds  = torch.stack([ds[int(i)][3] for i in indices]).to(device)
    return encoder(pkts, brs, sts, hds).cpu().numpy()

@torch.no_grad()
def finetune_head_5shot(encoder, ds, n_way, k_shot, q_query, num_episodes,
                         device, seed, C=1e3):
    """每个 episode：encoder 冻结 → sklearn LogReg on K support → eval on Q query。"""
    sampler = EpisodeSampler(ds, n_way=n_way, k_shot=k_shot,
                             q_query=q_query, seed=seed)
    accs = []
    for ep in sampler.iter_episodes(num_episodes):
        s_idx = ep["support_indices"]
        q_idx = ep["query_indices"]
        s_y = ep["support_labels"].numpy()  # local labels 0..N-1
        q_y = ep["query_labels"].numpy()

        s_x = encode_batch(encoder, ds, s_idx, device)
        q_x = encode_batch(encoder, ds, q_idx, device)

        # sklearn LogReg in closed form; C=1e3 强正则
        clf = LogisticRegression(C=C, max_iter=200, solver="lbfgs", n_jobs=1)
        clf.fit(s_x, s_y)
        pred = clf.predict(q_x)
        accs.append(float((pred == q_y).mean()))
    return float(np.mean(accs)), float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)

@torch.no_grad()
def protonet_baseline(encoder, ds, n_way, k_shot, q_query, num_episodes, device, seed):
    """对照：相同 episode 协议下 ProtoNet 评估（无 head fine-tune）。"""
    model = ProtoNet(encoder).to(device).eval()
    attach_dataset(model, ds)
    sampler = EpisodeSampler(ds, n_way=n_way, k_shot=k_shot,
                             q_query=q_query, seed=seed)
    accs = []
    for ep in sampler.iter_episodes(num_episodes):
        s_idx, q_idx = ep["support_indices"], ep["query_indices"]
        s_y = ep["support_labels"].to(device); q_y = ep["query_labels"].to(device)
        s_pkt = torch.stack([ds[int(i)][0] for i in s_idx]).to(device)
        s_br  = torch.stack([ds[int(i)][1] for i in s_idx]).to(device)
        s_st  = torch.stack([ds[int(i)][2] for i in s_idx]).to(device)
        s_hd  = torch.stack([ds[int(i)][3] for i in s_idx]).to(device)
        q_pkt = torch.stack([ds[int(i)][0] for i in q_idx]).to(device)
        q_br  = torch.stack([ds[int(i)][1] for i in q_idx]).to(device)
        q_st  = torch.stack([ds[int(i)][2] for i in q_idx]).to(device)
        q_hd  = torch.stack([ds[int(i)][3] for i in q_idx]).to(device)
        logits = model((s_pkt, s_br, s_st, s_hd),
                       (q_pkt, q_br, q_st, q_hd), s_y)
        pred = logits.argmax(dim=-1)
        accs.append((pred == q_y).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cic", choices=["cic", "ustc", "iscx"])
    p.add_argument("--n_way", type=int, default=None)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--C", type=float, default=1e3, help="sklearn LogReg regularization (large → weak reg)")
    p.add_argument("--encoder", default="ssl", choices=["ssl", "random", "both"],
                   help="ssl=SimCLR ckpt; random=random init; both=两个都跑（输出带 _both 后缀）")
    args = p.parse_args()

    if args.n_way is None:
        args.n_way = 2 if args.dataset == "iscx" else 5

    h5_path = f"output/{args.dataset}_full.h5"
    import h5py
    with h5py.File(h5_path, "r") as h:
        n = h["stat"].shape[0]
        labels = h["label"].asstr()[:]
    indices = _stratified_subsample(list(range(n)), labels, args.max_samples, seed=42)
    ds = H5FlowDataset(h5_path, indices=list(indices), fit=True)
    print(f"dataset={args.dataset} n={len(ds)} classes={len(ds.classes)} "
          f"task={args.n_way}w{args.k_shot}s stratified 20k")

    import glob
    ssl_ckpts = sorted(glob.glob(f"output/checkpoints/{args.dataset}_simclr_ep*.pt"))
    if not ssl_ckpts:
        raise FileNotFoundError(f"No SSL ckpt: output/checkpoints/{args.dataset}_simclr_ep*.pt")
    ssl_ckpt = ssl_ckpts[-1]  # use latest epoch

    # 根据 --encoder 决定 encoder 列表
    encoders = []  # list of (tag, ckpt_path_or_None)
    if args.encoder in ("ssl", "both"):
        encoders.append(("ssl", ssl_ckpt))
    if args.encoder in ("random", "both"):
        encoders.append(("random", None))

    out = {"dataset": args.dataset, "task": f"{args.n_way}w{args.k_shot}s",
           "episodes": args.episodes, "seeds": args.seeds,
           "max_samples": args.max_samples, "C": args.C,
           "encoders": [t for t, _ in encoders], "results": {}}

    for enc_tag, ckpt_path in encoders:
        enc = TrafficEncoder(embed_dim=128).to(args.device).eval()
        if ckpt_path:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            enc.load_state_dict(sd["encoder_state"])
            print(f"[encoder={enc_tag}] loaded ckpt: {ckpt_path}")
        else:
            print(f"[encoder={enc_tag}] random init (no ckpt)")

        for tag, fn in [(f"finetune_head_lr_{enc_tag}", finetune_head_5shot),
                        (f"protonet_baseline_{enc_tag}", protonet_baseline)]:
            means, stds = [], []
            for s in range(args.seeds):
                seed = 42 + s * 100
                t0 = time.time()
                if "finetune_head" in tag:
                    m, std = fn(enc, ds, args.n_way, args.k_shot, args.q_query,
                                args.episodes, args.device, seed, C=args.C)
                else:
                    m, std = fn(enc, ds, args.n_way, args.k_shot, args.q_query,
                                args.episodes, args.device, seed)
                means.append(m); stds.append(std)
                print(f"  {tag} seed={seed}: acc={m:.4f} ± {std:.4f} ({time.time()-t0:.1f}s)")
            mean_arr = np.array(means)
            out["results"][tag] = {
                "mean_per_seed": means,
                "std_per_seed": stds,
                "overall_mean": float(mean_arr.mean()),
                "overall_std": float(mean_arr.std()),
                "ci95_half_width": float(1.96 * mean_arr.std() / np.sqrt(len(means))),
            }
            print(f"  → {tag}: overall {out['results'][tag]['overall_mean']:.4f}")

    # 如果 both 模式：直接计算 SSL vs Random 在 FT-Head 范式下的增益
    if args.encoder == "both":
        ft_ssl = out["results"]["finetune_head_lr_ssl"]["overall_mean"]
        ft_rand = out["results"]["finetune_head_lr_random"]["overall_mean"]
        out["finding"] = {
            "ft_ssl_acc": ft_ssl,
            "ft_random_acc": ft_rand,
            "ft_ssl_minus_random_pp": float((ft_ssl - ft_rand) * 100),
            "ft_ssl_better": bool(ft_ssl > ft_rand),
            "interpretation": (
                "FT-Head paradigm下SSL仍 > Random，说明SSL的优势不只是 "
                "ProtoNet的cosine prototype优势，而是在学到的表征质量本身。"
                if ft_ssl > ft_rand else
                "FT-Head paradigm下SSL ≈ Random，意味着SSL的优势主要在 "
                "ProtoNet的cosine prototype——表征质量在LogReg head下被"
                "重新校准，SSL增益消失。"
            ),
        }
        print(f"\n>>> FT-Head: SSL {ft_ssl:.4f} vs Random {ft_rand:.4f}, "
              f"Δ = {ft_ssl - ft_rand:+.4f}")

    suffix = f"_{args.encoder}" if args.encoder != "ssl" else ""
    out_path = f"output/results/finetune_head_5shot{('_' + args.encoder) if args.encoder != 'ssl' else ''}_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")

if __name__ == "__main__":
    main()
