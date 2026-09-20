"""
run_zeroshot.py
===============
实验 4：新攻击零样本泛化。

把数据集按类别划分 base/novel，仅在 base 上预训练 encoder，
然后只在 novel 上评估 ProtoNet 的少样本能力（模拟"从未见过的新攻击"）。

用法：
    python experiments/run_zeroshot.py --dataset cic \\
        --encoder_ckpt output/checkpoints/cic_simclr_ep50.pt
"""
from __future__ import annotations
import argparse
import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

import torch
import numpy as np
from src.data.load_h5 import H5FlowDataset
from src.data.episodes import EpisodeSampler
from src.models.encoder import TrafficEncoder
from src.models.fewshot import ProtoNet, attach_dataset
from src.utils import get_device, set_seed, TqdmLogger, save_json, merge_mean_std

H5_PATHS = {
    "cic": "output/cic_full.h5",
    "ustc": "output/ustc_full.h5",
    "iscx": "output/iscx_full.h5",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(H5_PATHS), required=True)
    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--novel_classes", type=int, default=2,
                   help="novel 类数量")
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
                                  f"zeroshot_{args.dataset}.log"))
    log.info(f"设备={device}  dataset={args.dataset}  "
             f"novel={args.novel_classes}  {args.n_way}-way {args.k_shot}-shot")

    # 数据
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

    # 只保留 novel 类样本（重写：构造新的 cls_to_idx 传给 EpisodeSampler）
    n_classes = len(dataset.classes)
    if args.novel_classes >= n_classes:
        raise ValueError("novel_classes 必须小于总类别数")
    novel_ids = list(range(n_classes - args.novel_classes, n_classes))

    # 收集 novel 类样本的 dataset 内部索引（连续 0..len(dataset)）
    from collections import defaultdict
    cls_to_idx = defaultdict(list)
    for i, y in enumerate(dataset.y.tolist()):
        if int(y) in novel_ids:
            new_label = novel_ids.index(int(y))
            cls_to_idx[new_label].append(i)
    n_novel_samples = sum(len(v) for v in cls_to_idx.values())
    log.info(f"novel 类别数={args.novel_classes}  novel 样本数={n_novel_samples}")

    # 加载 encoder
    enc_sd = torch.load(args.encoder_ckpt, map_location="cpu",
                        weights_only=True)["encoder_state"]
    encoder = TrafficEncoder(embed_dim=args.embed_dim)
    encoder.load_state_dict(enc_sd)
    encoder.to(device).eval()

    # 多 seed 评估
    seed_results = []
    for s in range(args.num_seeds):
        seed = args.seed + s * 100
        model = ProtoNet(encoder).to(device)
        attach_dataset(model, dataset)
        sampler = EpisodeSampler(
            dataset,
            n_way=min(args.n_way, args.novel_classes),
            k_shot=args.k_shot, q_query=args.q_query,
            seed=seed, cls_to_idx=cls_to_idx,
        )
        accs = []
        for ep in sampler.iter_episodes(args.num_episodes):
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
        r = merge_mean_std(accs)
        r["seed"] = seed
        seed_results.append(r)
        log.info(f"seed={seed}  acc={r['mean']:.4f} ± {r['std']:.4f}")

    summary = {
        "dataset": args.dataset,
        "encoder_ckpt": args.encoder_ckpt,
        "novel_classes": args.novel_classes,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "num_episodes": args.num_episodes,
        "num_seeds": args.num_seeds,
        "results": seed_results,
        "aggregate": merge_mean_std([r["mean"] for r in seed_results]),
    }
    log.info(f"汇总 novel {args.n_way}-way {args.k_shot}-shot acc = "
             f"{summary['aggregate']['mean']:.4f} "
             f"± {summary['aggregate']['std']:.4f}")
    out_name = f"zeroshot_{args.dataset}_{args.novel_classes}novel_" \
               f"{args.n_way}w{args.k_shot}s.json"
    save_json(summary, os.path.join(args.out_dir, out_name))
    log.info(f"已保存 {os.path.join(args.out_dir, out_name)}")

if __name__ == "__main__":
    main()
