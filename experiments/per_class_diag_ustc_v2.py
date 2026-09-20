"""
per_class_diag_ustc_v2.py
==========================
USTC per-class 5-shot 诊断（SSL vs Random init），与 §4.2 主表协议统一。

修正要点：
1. 用 _stratified_subsample 替代 rng.choice，保证 20 类都有样本
2. 5 seed × 600 episode，与 §4.2 主表完全一致
3. 修复"USTC 实际只有 8 类"的错误叙事
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

from src.data.load_h5 import H5FlowDataset, _stratified_subsample
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
        s_items = [dataset[int(i)] for i in s_idx]
        q_items = [dataset[int(i)] for i in q_idx]
        s_x = tuple(t.to(device) for t in _to_4tuple(s_items))
        q_x = tuple(t.to(device) for t in _to_4tuple(q_items))

        logits = model(s_x, q_x, s_y)
        pred = logits.argmax(dim=-1)
        proto_to_global = s_y.unique()  # local 0..N-1 → global class id
        pred_global = proto_to_global[pred]
        correct = (pred_global == q_y).cpu().numpy()

        # q_y 是 local label (0..N-1)；需映射到 global 才能正确索引 total_per_class
        q_y_global = proto_to_global[q_y.cpu()].numpy()
        # 同时：s_y.unique() 返回的是 sorted local，因此 proto_to_global[i]=chosen[i]
        # 但 sorted local 不一定对应 sorted global。直接用 ep["classes"] 更稳。
        chosen_global = np.array(ep["classes"])
        local_to_global = np.zeros(s_y.max().item() + 1, dtype=np.int64)
        for new_label, gc in enumerate(chosen_global):
            local_to_global[new_label] = int(gc)
        q_y_global = local_to_global[q_y.cpu().numpy()]
        pred_global = local_to_global[pred.cpu().numpy()]
        for i, true_global in enumerate(q_y_global):
            total_per_class[true_global] += 1
            if correct[i]:
                correct_per_class[true_global] += 1
            confusion[true_global, int(pred_global[i])] += 1

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
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    h5_path = f"output/{args.dataset}_full.h5"
    # 关键修复：用 stratified subsample（按类比例）而非随机抽样
    with __import__("h5py").File(h5_path, "r") as h:
        n = h["stat"].shape[0]
        labels = h["label"].asstr()[:]
    indices = _stratified_subsample(list(range(n)), labels, args.max_samples, seed=42)
    ds = H5FlowDataset(h5_path, indices=list(indices), fit=True)
    print(f"dataset={args.dataset} classes={len(ds.classes)} n={len(ds)} (stratified)")

    out = {"dataset": args.dataset, "task": f"{args.n_way}w{args.k_shot}s",
           "episodes": args.episodes, "seeds": args.seeds,
           "max_samples": args.max_samples,
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

        # 多 seed 评估
        per_class_acc_runs = []
        per_class_total_runs = []
        for s in range(args.seeds):
            seed = 42 + s * 100
            acc, total, conf, classes = per_class_eval(
                enc, ds, args.n_way, args.k_shot, args.q_query,
                args.episodes, args.device, seed,
            )
            per_class_acc_runs.append(acc)
            per_class_total_runs.append(total)
            print(f"  seed={seed}: overall_acc={acc[total>0].mean():.4f}")

        # 多 seed 平均 per-class accuracy
        per_class_acc_mean = np.mean(per_class_acc_runs, axis=0)
        per_class_total_sum = np.sum(per_class_total_runs, axis=0)

        # top-10 混淆（用最后一次 seed 的 confusion）
        cls_acc_list = []
        for i, c in enumerate(classes):
            cls_acc_list.append({
                "class": str(c),
                "acc": float(per_class_acc_mean[i]),
                "total_samples": int(per_class_total_sum[i]),
            })
        cls_acc_list.sort(key=lambda x: x["acc"])

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
            "overall_acc_mean": float(per_class_acc_mean[per_class_total_sum > 0].mean()),
            "per_class": cls_acc_list,
            "confusion_top10_offdiag": off_diag[:10],
        }
        print(f"  overall_acc (5-seed mean) = {out[tag]['overall_acc_mean']:.4f}")
        print(f"  worst 5 classes: {[(x['class'], round(x['acc'],3)) for x in cls_acc_list[:5]]}")
        print(f"  best 5 classes:  {[(x['class'], round(x['acc'],3)) for x in cls_acc_list[-5:]]}")
        print(f"  top 5 confusions: {[(x['true']+'->'+x['pred'], x['count']) for x in off_diag[:5]]}")

    out_path = f"output/results/per_class_diag_{args.dataset}_v2.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")

if __name__ == "__main__":
    main()