"""
run_baseline_fewshot.py  (v2)
==============================
批量跑 SOTA baseline 的 few-shot 评估，对比 §4.7 v2。

用法：
    python experiments/run_baseline_fewshot.py --dataset ustc --seeds 5
    python experiments/run_baseline_fewshot.py --dataset cic --seeds 5
    python experiments/run_baseline_fewshot.py --dataset iscx --seeds 5
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.models.baselines import (
    FSNet, YaTCNet, TFEGraph, ETBERTLite, NetMambaLite, NetConvLite, count_params
)
from src.data.load_h5 import H5FlowDataset, stratified_split_indices, time_aware_split_indices

def build_model(name: str, n_classes: int) -> nn.Module:
    """根据 name 构造模型（统一 embed_dim=128）。"""
    if name == "fsnet":       return FSNet(n_classes, embed_dim=128)
    if name == "yatc":        return YaTCNet(n_classes, embed_dim=128)
    if name == "tfegnn":      return TFEGraph(n_classes, embed_dim=128)
    if name == "etbert":      return ETBERTLite(n_classes, embed_dim=128)
    if name == "netmamba":    return NetMambaLite(n_classes, embed_dim=128)
    if name == "netconv":     return NetConvLite(n_classes, embed_dim=128)
    raise ValueError(name)

def episodic_eval(model: nn.Module, ds: H5FlowDataset, n_way: int, k_shot: int,
                  q_query: int, n_episodes: int, seed: int, device: str = "xpu",
                  use_encode: bool = True) -> float:
    """ProtoNet episode 评估：用 encode() 取 128-dim embedding。

    use_encode=True: 模型 .encode(pkt) -> (B, 128) embedding（v2 协议，与 TP-Net 一致）
    use_encode=False: 模型 .forward(pkt) -> (B, n_classes) logits（旧 v1 协议）
    """
    rng = np.random.RandomState(seed)
    labels = np.array([ds[i][4].item() for i in range(len(ds))])
    classes = sorted(np.unique(labels).tolist())
    valid_classes = [c for c in classes if (labels == c).sum() >= k_shot + q_query]
    if len(valid_classes) < n_way:
        raise ValueError(f"可用类数 {len(valid_classes)} < n_way {n_way}")
    print(f"  episodes 使用 {len(valid_classes)} 个有效类")

    accs = []
    for ep in range(n_episodes):
        chosen = rng.choice(valid_classes, size=n_way, replace=False)
        idx_support, idx_query = [], []
        for c in chosen:
            c_idx = np.where(labels == c)[0]
            sel = rng.choice(c_idx, size=k_shot + q_query, replace=False)
            idx_support.extend(sel[:k_shot].tolist())
            idx_query.extend(sel[k_shot:].tolist())

        with torch.no_grad():
            s_emb, q_emb, q_lbl = [], [], []
            for i in idx_support:
                pkt, _, _, _, lbl = ds[i]
                if use_encode:
                    e = model.encode(pkt.unsqueeze(0).to(device))
                else:
                    e = model(pkt.unsqueeze(0).to(device))
                s_emb.append(e.cpu())
            for i in idx_query:
                pkt, _, _, _, lbl = ds[i]
                if use_encode:
                    e = model.encode(pkt.unsqueeze(0).to(device))
                else:
                    e = model(pkt.unsqueeze(0).to(device))
                q_emb.append(e.cpu())
                q_lbl.append(int(lbl))
            s_emb = torch.stack(s_emb).squeeze(1)  # (n_way*k, D)
            q_emb = torch.stack(q_emb).squeeze(1)
            q_lbl = torch.tensor(q_lbl)
            prototypes = []
            for k in range(n_way):
                prototypes.append(s_emb[k * k_shot:(k + 1) * k_shot].mean(dim=0))
            prototypes = torch.stack(prototypes)  # (n_way, D)
            d = torch.cdist(q_emb, prototypes)
            pred = d.argmin(dim=1)
            acc = (pred == q_lbl).float().mean().item()
        accs.append(acc)
    return float(np.mean(accs))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ustc", choices=["cic", "ustc", "iscx"])
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n_samples", type=int, default=30000)
    p.add_argument("--models", nargs="+",
                   default=["fsnet", "yatc", "tfegnn", "etbert", "netmamba", "netconv"])
    p.add_argument("--tag", default="v2", help="输出文件 tag: v1 用 logits, v2 用 encode")
    p.add_argument("--use_encode", action="store_true", default=True,
                   help="用 model.encode(pkt) 取 128-dim embedding")
    args = p.parse_args()

    h5_path = f"output/{args.dataset}_full.h5"
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    if args.dataset == "iscx":
        idx_tr, idx_va, idx_te = stratified_split_indices(
            h5_path, train_ratio=0.7, val_ratio=0.15, seed=42)
    else:
        idx_tr, idx_va, idx_te = time_aware_split_indices(
            h5_path, train_ratio=0.7, val_ratio=0.15)

    rng = np.random.RandomState(42)
    idx_tr_sub = rng.choice(idx_tr, size=min(args.n_samples, len(idx_tr)), replace=False)

    import h5py
    from sklearn.preprocessing import StandardScaler
    with h5py.File(h5_path, "r") as f:
        S_full = f["stat"][:]
    scaler = StandardScaler().fit(S_full[idx_tr_sub])

    ds = H5FlowDataset(h5_path, indices=idx_tr_sub, scaler=scaler, fit=False)

    all_labels = np.array([ds[i][4].item() for i in range(len(ds))])
    valid_labels = sorted(np.unique(all_labels).tolist())
    n_classes = int(max(valid_labels)) + 1
    print(f"训练子集: {len(ds)} 条, 有效类别: {len(valid_labels)} (n_classes={n_classes})")

    out_dir = "output/results"
    os.makedirs(out_dir, exist_ok=True)

    for mname in args.models:
        print(f"\n=== {mname} ===")
        accs = []
        for seed in [42, 142, 242, 342, 442][:args.seeds]:
            torch.manual_seed(seed)
            model = build_model(mname, n_classes)
            try:
                model = model.to(device)
            except Exception:
                device_use = "cpu"
            else:
                device_use = device
            model.eval()
            print(f"  seed={seed} 参数={count_params(model):,}", end=" ", flush=True)
            acc = episodic_eval(model, ds, args.n_way, args.k_shot,
                                args.q_query, args.episodes, seed,
                                device=device_use,
                                use_encode=args.use_encode)
            accs.append(acc)
            print(f"acc={acc:.4f}")

        out = {
            "model": mname,
            "dataset": args.dataset,
            "task": f"{args.n_way}w{args.k_shot}s",
            "n_episodes": args.episodes,
            "n_seeds": len(accs),
            "seeds_raw": accs,
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "params": count_params(model),
            "n_train_samples": len(idx_tr_sub),
            "ssl": False,
            "encoder_protocol": "encode_128dim" if args.use_encode else "logits",
            "tag": args.tag,
        }
        out_path = os.path.join(
            out_dir, f"baseline_{args.tag}_{mname}_{args.dataset}_{args.n_way}w{args.k_shot}s.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  -> {out_path}: mean={out['mean']:.4f} ± {out['std']:.4f}")

if __name__ == "__main__":
    main()
