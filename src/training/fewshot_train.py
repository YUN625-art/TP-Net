"""
fewshot_train.py
================
少样本下游训练与评估入口。

用法：
    python experiments/run_fewshot.py --dataset cic --encoder_ckpt xxx.pt
    --algo protonet --n_way 5 --k_shot 5
"""
from __future__ import annotations
import argparse
import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.load_h5 import H5FlowDataset
from src.data.episodes import EpisodeSampler, make_episode_loader, split_base_novel
from src.models.encoder import TrafficEncoder
from src.models.fewshot import (
    ProtoNet, MatchingNet, LinearProbing, attach_dataset
)
from src.utils import get_device, set_seed, TqdmLogger, save_json, merge_mean_std

H5_PATHS = {
    "cic": "output/cic_full.h5",
    "ustc": "output/ustc_full.h5",
    "iscx": "output/iscx_full.h5",
}

def load_encoder(ckpt_path: str, embed_dim: int, device) -> TrafficEncoder:
    enc = TrafficEncoder(embed_dim=embed_dim)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    enc.load_state_dict(sd["encoder_state"])
    return enc.to(device)

@torch.no_grad()
def eval_protonet(encoder, dataset, n_way, k_shot, q_query,
                  num_episodes, device, seed) -> dict:
    """ProtoNet 评估：无需训练，encoder 提特征 + 欧式距离。"""
    model = ProtoNet(encoder).to(device)
    attach_dataset(model, dataset)
    sampler = EpisodeSampler(dataset, n_way=n_way, k_shot=k_shot,
                             q_query=q_query, seed=seed)
    accs = []
    for ep in sampler.iter_episodes(num_episodes):
        s_idx = ep["support_indices"]
        q_idx = ep["query_indices"]
        s_y = ep["support_labels"].to(device)
        q_y = ep["query_labels"].to(device)

        s_pkt = torch.stack([dataset[int(i)][0] for i in s_idx]).to(device)
        s_br  = torch.stack([dataset[int(i)][1] for i in s_idx]).to(device)
        s_st  = torch.stack([dataset[int(i)][2] for i in s_idx]).to(device)
        s_hd  = torch.stack([dataset[int(i)][3] for i in s_idx]).to(device)
        q_pkt = torch.stack([dataset[int(i)][0] for i in q_idx]).to(device)
        q_br  = torch.stack([dataset[int(i)][1] for i in q_idx]).to(device)
        q_st  = torch.stack([dataset[int(i)][2] for i in q_idx]).to(device)
        q_hd  = torch.stack([dataset[int(i)][3] for i in q_idx]).to(device)

        logits = model((s_pkt, s_br, s_st, s_hd),
                       (q_pkt, q_br, q_st, q_hd), s_y)
        pred = logits.argmax(dim=-1)
        accs.append((pred == q_y).float().mean().item())
    return merge_mean_std(accs)

@torch.no_grad()
def eval_matching(encoder, dataset, n_way, k_shot, q_query,
                  num_episodes, device, seed) -> dict:
    """MatchingNet 评估。"""
    model = MatchingNet(encoder, embed_dim=encoder.embed_dim).to(device)
    attach_dataset(model, dataset)
    sampler = EpisodeSampler(dataset, n_way=n_way, k_shot=k_shot,
                             q_query=q_query, seed=seed)
    accs = []
    for ep in sampler.iter_episodes(num_episodes):
        s_idx = ep["support_indices"]
        q_idx = ep["query_indices"]
        s_y = ep["support_labels"].to(device)
        q_y = ep["query_labels"].to(device)

        s_pkt = torch.stack([dataset[int(i)][0] for i in s_idx]).to(device)
        s_br  = torch.stack([dataset[int(i)][1] for i in s_idx]).to(device)
        s_st  = torch.stack([dataset[int(i)][2] for i in s_idx]).to(device)
        s_hd  = torch.stack([dataset[int(i)][3] for i in s_idx]).to(device)
        q_pkt = torch.stack([dataset[int(i)][0] for i in q_idx]).to(device)
        q_br  = torch.stack([dataset[int(i)][1] for i in q_idx]).to(device)
        q_st  = torch.stack([dataset[int(i)][2] for i in q_idx]).to(device)
        q_hd  = torch.stack([dataset[int(i)][3] for i in q_idx]).to(device)

        logits = model((s_pkt, s_br, s_st, s_hd),
                       (q_pkt, q_br, q_st, q_hd), s_y)
        pred = logits.argmax(dim=-1)
        accs.append((pred == q_y).float().mean().item())
    return merge_mean_std(accs)

def train_linear_probe(encoder, dataset, n_train_classes,
                       n_episodes=200, batch_size=64, device="cpu",
                       epochs=20, lr=1e-3, seed=42) -> dict:
    """Linear Probing：在 base 类上训练单层 Linear。"""
    set_seed(seed)
    n_total = len(dataset.classes)
    # 安全处理：确保 n_train_classes >= 2，避免空 subset
    n_train_classes = max(2, min(n_train_classes, n_total - 1))
    model = LinearProbing(encoder, num_classes=n_total,
                          embed_dim=encoder.embed_dim).to(device)
    # 取部分 base 类训练
    base_idx = [i for i, y in enumerate(dataset.y.tolist())
                if y < n_train_classes]
    if len(base_idx) == 0:
        # 退而求其次：用全部数据
        base_idx = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(base_idx)
    train_idx = base_idx[: int(0.8 * len(base_idx))]
    test_idx  = base_idx[int(0.8 * len(base_idx)):]

    sub = torch.utils.data.Subset(dataset, train_idx)
    dl = DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.AdamW(model.classifier.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        losses = []
        for batch in dl:
            pkt, br, st, hd, y = batch
            pkt = pkt.to(device); br = br.to(device)
            st = st.to(device); hd = hd.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(pkt, br, st, hd)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

    # 评估
    model.eval()
    correct, total = 0, 0
    sub_te = torch.utils.data.Subset(dataset, test_idx)
    dl_te = DataLoader(sub_te, batch_size=batch_size, num_workers=0)
    with torch.no_grad():
        for batch in dl_te:
            pkt, br, st, hd, y = batch
            pkt = pkt.to(device); br = br.to(device)
            st = st.to(device); hd = hd.to(device); y = y.to(device)
            logits = model(pkt, br, st, hd)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / max(total, 1)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(H5_PATHS), required=True)
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--algo", choices=["protonet", "matching", "linearprobe"],
                   default="protonet")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--num_episodes", type=int, default=600)
    p.add_argument("--num_seeds", type=int, default=5)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="output/results")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    log = TqdmLogger(os.path.join("output/logs",
                                  f"fewshot_{args.dataset}_{args.algo}.log"))
    log.info(f"设备={device}  dataset={args.dataset}  algo={args.algo}  "
             f"{args.n_way}-way {args.k_shot}-shot")

    # 加载数据
    h5_path = H5_PATHS[args.dataset]
    with __import__("h5py").File(h5_path, "r") as h:
        n = h["stat"].shape[0]
    if args.max_samples and args.max_samples < n:
        from src.data.load_h5 import _stratified_subsample
        with __import__("h5py").File(h5_path, "r") as h:
            labels = h["label"].asstr()[:]
        indices = _stratified_subsample(list(range(n)), labels,
                                        args.max_samples, seed=args.seed)
    else:
        indices = list(range(n))
    dataset = H5FlowDataset(h5_path, indices, fit=True)
    log.info(f"样本数={len(dataset)}  类别数={len(dataset.classes)}")

    # 加载预训练 encoder
    encoder = load_encoder(args.encoder_ckpt, args.embed_dim, device)
    encoder.eval()
    log.info(f"加载 encoder from {args.encoder_ckpt}")

    # 多 seed 评估
    seed_results = []
    for s in range(args.num_seeds):
        seed = args.seed + s * 100
        if args.algo == "protonet":
            r = eval_protonet(encoder, dataset, args.n_way, args.k_shot,
                              args.q_query, args.num_episodes, device, seed)
        elif args.algo == "matching":
            r = eval_matching(encoder, dataset, args.n_way, args.k_shot,
                              args.q_query, args.num_episodes, device, seed)
        else:  # linearprobe
            r = train_linear_probe(encoder, dataset,
                                   n_train_classes=len(dataset.classes) - 2,
                                   device=device, seed=seed)
        r["seed"] = seed
        seed_results.append(r)
        log.info(f"seed={seed}  acc={r.get('mean', r.get('accuracy')):.4f}")

    summary = {
        "dataset": args.dataset,
        "algo": args.algo,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "num_episodes": args.num_episodes,
        "num_seeds": args.num_seeds,
        "encoder_ckpt": args.encoder_ckpt,
        "results": seed_results,
    }
    if args.algo in ("protonet", "matching"):
        accs = [r["mean"] for r in seed_results]
        summary["aggregate"] = merge_mean_std(accs)
        log.info(f"汇总  {args.n_way}-way {args.k_shot}-shot "
                 f"acc = {summary['aggregate']['mean']:.4f} "
                 f"± {summary['aggregate']['std']:.4f}")
    else:
        accs = [r["accuracy"] for r in seed_results]
        summary["aggregate"] = merge_mean_std(accs)
        log.info(f"汇总  LinearProbe acc = "
                 f"{summary['aggregate']['mean']:.4f} ± "
                 f"{summary['aggregate']['std']:.4f}")

    # 根据 ckpt 文件名自动推断 SSL 算法 tag，避免 SimCLR/MoCo 互相覆盖
    ckpt_name = os.path.basename(args.encoder_ckpt).lower()
    if "moco" in ckpt_name:
        ssl_tag = "moco"
    elif "simclr" in ckpt_name or "byol" in ckpt_name:
        ssl_tag = "simclr"
    else:
        ssl_tag = "ssl"
    # 如果 ckpt 与 dataset 不一致（即跨数据集迁移），文件名加 src tag 防止互相覆盖
    ckpt_base = os.path.basename(args.encoder_ckpt).lower()
    src_tag = ""
    for ds_name in ["cic", "ustc", "iscx"]:
        if f"/{ds_name}_" in ckpt_base.replace("\\", "/") or ckpt_base.startswith(f"{ds_name}_"):
            if ds_name != args.dataset:
                src_tag = f"{ds_name}2"
            break
    out_name = (f"fewshot_{src_tag}{args.dataset}_{ssl_tag}_{args.algo}_"
                f"{args.n_way}w{args.k_shot}s.json")
    save_json(summary, os.path.join(args.out_dir, out_name))
    log.info(f"已保存 {os.path.join(args.out_dir, out_name)}")

if __name__ == "__main__":
    main()
