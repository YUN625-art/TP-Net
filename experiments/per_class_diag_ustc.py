"""
per_class_diag_ustc.py
======================
USTC per-class 5-shot 诊断（SSL vs Random init）。

输出 JSON 含每个类的 5-shot 准确率 + 整体 confusion 矩阵。
"""
import argparse
import json
import os
import sys
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
    """把 list of (P, B, S, H, y) tuples 拆成 4 元组 (P_tensor, B_tensor, S_tensor, H_tensor)。"""
    P = torch.stack([it[0] for it in items])
    B = torch.stack([it[1] for it in items])
    S = torch.stack([it[2] for it in items])
    H = torch.stack([it[3] for it in items])
    return (P, B, S, H)

def _gather_4tuple(ds, indices, device):
    items = [ds[int(i)] for i in indices]
    return _to_4tuple(items).to(device) if False else tuple(t.to(device) for t in _to_4tuple(items))

@torch.no_grad()
def per_class_eval(encoder, dataset, n_way, k_shot, q_query,
                   num_episodes, device, seed):
    """Per-class accuracy: 跟踪每个 query 样本的真实类与预测类。"""
    model = ProtoNet(encoder).to(device).eval()
    attach_dataset(model, dataset)
    sampler = EpisodeSampler(dataset, n_way=n_way, k_shot=k_shot,
                             q_query=q_query, seed=seed)

    n_classes = len(dataset.classes)
    correct_per_class = np.zeros(n_classes, dtype=np.int64)
    total_per_class = np.zeros(n_classes, dtype=np.int64)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    for ep in sampler.iter_episodes(num_episodes):
        s_idx = ep["support_indices"]
        q_idx = ep["query_indices"]
        s_y = ep["support_labels"].to(device)
        q_y = ep["query_labels"].to(device)
        s_x = _gather_4tuple(dataset, s_idx, device)
        q_x = _gather_4tuple(dataset, q_idx, device)

        # 用 ProtoNet 的 forward，logits = cos sim
        logits = model(s_x, q_x, s_y)
        pred = logits.argmax(dim=-1)
        # 把 local class idx 映射回 global dataset class idx
        proto_to_global = s_y.unique()  # tensor of global class indices in episode
        pred_global = proto_to_global[pred]
        correct = (pred_global == q_y).cpu().numpy()

        for i, true_y in enumerate(q_y.cpu().numpy()):
            total_per_class[true_y] += 1
            if correct[i]:
                correct_per_class[true_y] += 1
            confusion[true_y, pred_global[i].item()] += 1

    per_class_acc = np.where(
        total_per_class > 0,
        correct_per_class / np.maximum(total_per_class, 1),
        0.0,
    )
    return per_class_acc, total_per_class, confusion, dataset.classes

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ustc")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    h5_path = f"output/{args.dataset}_full.h5"
    idx_tr, _, idx_te = time_aware_split_indices(h5_path, 0.7, 0.15)
    rng = np.random.RandomState(42)
    idx_sub = rng.choice(idx_te, min(20000, len(idx_te)), replace=False)
    ds = H5FlowDataset(h5_path, indices=list(idx_sub), fit=True)
    print(f"dataset={args.dataset} classes={len(ds.classes)} n={len(ds)}")

    out = {"dataset": args.dataset, "task": f"{args.n_way}w{args.k_shot}s",
           "classes": list(ds.classes),
           "ssl": {}, "random": {}}

    for tag, ckpt_path in [
        ("ssl", f"output/checkpoints/{args.dataset}_simclr_ep50.pt"),
        ("random", None),
    ]:
        print(f"\n=== {tag} ===")
        enc = TrafficEncoder(embed_dim=128).to(args.device).eval()
        if ckpt_path and os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            enc.load_state_dict(sd["encoder_state"])
        else:
            print(f"  [WARN] no ckpt, using random init")

        acc, total, conf, classes = per_class_eval(
            enc, ds, args.n_way, args.k_shot, args.q_query,
            args.episodes, args.device, args.seed,
        )
        cls_acc_list = []
        for i, c in enumerate(classes):
            cls_acc_list.append({
                "class": str(c),
                "acc": float(acc[i]),
                "total_samples": int(total[i]),
            })
        cls_acc_list.sort(key=lambda x: x["acc"])
        # top-5 混淆
        n = conf.shape[0]
        off_diag = []
        for i in range(n):
            for j in range(n):
                if i != j and conf[i, j] > 0:
                    off_diag.append({
                        "true": str(classes[i]),
                        "pred": str(classes[j]),
                        "count": int(conf[i, j]),
                    })
        off_diag.sort(key=lambda x: -x["count"])
        out[tag] = {
            "overall_acc": float(acc[total > 0].mean()),
            "per_class": cls_acc_list,
            "confusion_top10_offdiag": off_diag[:10],
        }
        print(f"  overall_acc = {out[tag]['overall_acc']:.4f}")
        print(f"  worst 5 classes: {[(x['class'], round(x['acc'],3)) for x in cls_acc_list[:5]]}")
        print(f"  best 5 classes:  {[(x['class'], round(x['acc'],3)) for x in cls_acc_list[-5:]]}")
        print(f"  top 5 confusions: {[(x['true']+'->'+x['pred'], x['count']) for x in off_diag[:5]]}")

    out_path = f"output/results/per_class_diag_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")

if __name__ == "__main__":
    main()
