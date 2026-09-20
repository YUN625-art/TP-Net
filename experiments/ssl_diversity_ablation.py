"""
ssl_diversity_ablation.py
=========================
SSL 算法多样性消融：5 算法 × 3 数据集 = 15 个下游精度点。
"""
import argparse
import json
import os
import subprocess
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

H5_PATHS = {
    "cic": "output/cic_full.h5",
    "ustc": "output/ustc_full.h5",
    "iscx": "output/iscx_full.h5",
}
ALGOS = ["simclr", "moco", "byol", "simsiam", "mae"]
EPOCHS_DEFAULT = {"cic": 50, "ustc": 50, "iscx": 30}

def load_encoder(ckpt, device):
    enc = TrafficEncoder(embed_dim=128)
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        enc.load_state_dict(sd["encoder_state"])
    return enc.to(device).eval()

def train_if_missing(dataset, algo, epochs, out_dir, device):
    """如果 ckpt 不存在则训练。返回 ckpt 路径。"""
    suffix = "_medium"
    ckpt_medium = os.path.join(out_dir, f"{dataset}_{algo}_ep{epochs}{suffix}.pt")
    ckpt_plain = os.path.join(out_dir, f"{dataset}_{algo}_ep{epochs}.pt")
    if os.path.exists(ckpt_medium) or os.path.exists(ckpt_plain):
        existing = ckpt_medium if os.path.exists(ckpt_medium) else ckpt_plain
        print(f"  [跳过] {existing} 已存在")
        return existing
    print(f"  [训练] {algo} on {dataset} for {epochs} epochs …")
    cmd = [
        sys.executable, "-m", "src.training.pretrain",
        "--dataset", dataset, "--algo", algo,
        "--epochs", str(epochs), "--batch_size", "256",
        "--max_samples", "20000",
        "--aug_preset", "medium",
        "--save_every", str(epochs),
        "--out_dir", out_dir,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    dt = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"训练失败: {algo}/{dataset} rc={rc}")
    # 新 ckpt 命名是 _medium
    if os.path.exists(ckpt_medium):
        print(f"  [完成] 训练耗时 {dt/60:.1f} min")
        return ckpt_medium
    if os.path.exists(ckpt_plain):
        print(f"  [完成] 训练耗时 {dt/60:.1f} min")
        return ckpt_plain
    raise RuntimeError(f"训练完成但 ckpt 仍不存在: {algo}/{dataset}")

def find_ckpt(dataset, algo, epochs, out_dir):
    """兼容多种命名：带或不带 _medium 后缀；epoch 不匹配时回退到任何 epoch。"""
    candidates = [epochs]
    # 兼容 30/50 协议不一致
    if epochs == 50:
        candidates.append(30)
    elif epochs == 30:
        candidates.append(50)
    for ep in candidates:
        for suffix in ("_medium", ""):
            ckpt = os.path.join(out_dir, f"{dataset}_{algo}_ep{ep}{suffix}.pt")
            if os.path.exists(ckpt):
                return ckpt
    return None

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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cic", "ustc", "iscx"])
    p.add_argument("--algos", nargs="+", default=ALGOS)
    p.add_argument("--epochs", type=int, default=None,
                   help="每数据集 epoch；缺省按数据集大小自动选")
    p.add_argument("--out_dir", default="output/checkpoints")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--num_episodes", type=int, default=200)
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval_only", action="store_true",
                   help="只评估不训练（ckpt 必须已存在）")
    args = p.parse_args()

    out = {"algos": args.algos, "datasets": args.datasets,
           "n_way": args.n_way, "k_shot": args.k_shot,
           "num_episodes": args.num_episodes, "num_seeds": args.num_seeds,
           "results": {}}

    for ds_name in args.datasets:
        h5_path = H5_PATHS[ds_name]
        if not os.path.exists(h5_path):
            print(f"[跳过] {h5_path} 不存在")
            continue

        if ds_name == "iscx":
            idx_tr, _, _ = stratified_split_indices(h5_path, 0.7, 0.15, seed=42)
            n_way = 2
        else:
            idx_tr, _, _ = time_aware_split_indices(h5_path, 0.7, 0.15)
            n_way = args.n_way

        rng = np.random.RandomState(42)
        idx_tr_sub = rng.choice(idx_tr, min(args.max_samples, len(idx_tr)), replace=False)
        ds = H5FlowDataset(h5_path, indices=list(idx_tr_sub), fit=True)
        print(f"\n=== {ds_name} n_way={n_way} train_n={len(ds)} ===")

        out["results"][ds_name] = {}

        for algo in args.algos:
            epochs = args.epochs if args.epochs else EPOCHS_DEFAULT[ds_name]
            try:
                if not args.eval_only:
                    train_if_missing(ds_name, algo, epochs, args.out_dir, args.device)
                ckpt = find_ckpt(ds_name, algo, epochs, args.out_dir)
                if not ckpt:
                    print(f"  [跳过] ckpt 不存在: {ds_name}/{algo}")
                    continue
                enc = load_encoder(ckpt, args.device)
                accs = []
                for s in range(args.num_seeds):
                    seed = 42 + s * 100
                    mean, std = eval_protonet(enc, ds, n_way, args.k_shot,
                                              args.q_query, args.num_episodes,
                                              seed, args.device)
                    accs.append(mean)
                    print(f"    {algo} seed={seed} acc={mean:.4f}")
                agg_mean = float(np.mean(accs))
                agg_std = float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)
                out["results"][ds_name][algo] = {
                    "mean": round(agg_mean, 4),
                    "std": round(agg_std, 4),
                    "raw": [round(a, 4) for a in accs],
                    "epochs": epochs,
                    "ckpt": ckpt,
                }
                print(f"  >>> {ds_name}/{algo}: {agg_mean:.4f} ± {agg_std:.4f}")
            except Exception as e:
                print(f"  [ERROR] {ds_name}/{algo}: {e}")
                out["results"][ds_name][algo] = {"error": str(e)}

    out_path = "output/results/ssl_diversity_ablation.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")

if __name__ == "__main__":
    main()
