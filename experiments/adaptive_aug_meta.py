"""
adaptive_aug_meta.py
====================
episode-level 自适应增强方法化。

设计动机：§4.5 经验表 N→preset 在跨数据集迁移时是"死规则"。
更精细的做法：让模型根据 episode 的内在特征（inter-class
distance / intra-class variance）自动选 preset。

简化版：训练 MLP 直接从 episode 统计预测最优 preset。
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.models.encoder import TrafficEncoder
from src.models.fewshot import ProtoNet
from src.data.load_h5 import H5FlowDataset, time_aware_split_indices, stratified_split_indices
from src.data.episodes import EpisodeSampler

PRESETS = ["none", "weak", "medium", "strong"]
PRESET_TO_IDX = {p: i for i, p in enumerate(PRESETS)}
IDX_TO_PRESET = {i: p for i, p in enumerate(PRESETS)}

# ckpt 命名约定：none/weak/strong 带后缀；medium 用基础名（无后缀）
PRESET_CKPT_NAMES = {
    "none": "{dataset}_simclr_ep30_none.pt",
    "weak": "{dataset}_simclr_ep30_weak.pt",
    "medium": "{dataset}_simclr_ep30.pt",       # 基础 ckpt 即 medium
    "strong": "{dataset}_simclr_ep30_strong.pt",
}

def load_encoder(ckpt, device):
    enc = TrafficEncoder(embed_dim=128)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    enc.load_state_dict(sd["encoder_state"])
    return enc.to(device).eval()

@torch.no_grad()
def collect_episode_data(encoders, ds, n_way, k_shot, q_query,
                          num_episodes, seed, device):
    """对每个 episode，跨 4 个 preset 收集特征和 accuracy。

    Returns:
        feats: (num_episodes, F) — episode 特征（来自 medium encoder）
        accs:  (num_episodes, 4) — 4 preset 的 episode accuracy
        best:  (num_episodes,)   — 每 episode 的 best preset (argmax)
    """
    # 用 medium encoder 提取 episode 特征（与 base model 一致）
    base_enc = encoders[PRESET_TO_IDX["medium"]]
    model = ProtoNet(base_enc).to(device)

    sampler = EpisodeSampler(ds, n_way=n_way, k_shot=k_shot, q_query=q_query,
                             seed=seed)

    all_feats, all_accs = [], []
    for ep_id, ep in enumerate(sampler.iter_episodes(num_episodes)):
        if ep_id >= num_episodes:
            break
        s_idx = ep["support_indices"]; q_idx = ep["query_indices"]
        s_y = ep["support_labels"].to(device)
        s_pkt = torch.stack([ds[int(i)][0] for i in s_idx]).to(device)
        s_br = torch.stack([ds[int(i)][1] for i in s_idx]).to(device)
        s_st = torch.stack([ds[int(i)][2] for i in s_idx]).to(device)
        s_hd = torch.stack([ds[int(i)][3] for i in s_idx]).to(device)
        q_pkt = torch.stack([ds[int(i)][0] for i in q_idx]).to(device)
        q_br = torch.stack([ds[int(i)][1] for i in q_idx]).to(device)
        q_st = torch.stack([ds[int(i)][2] for i in q_idx]).to(device)
        q_hd = torch.stack([ds[int(i)][3] for i in q_idx]).to(device)

        # 用 base encoder 提 support embedding 计算特征
        s_feat = base_enc(s_pkt, s_br, s_st, s_hd)
        q_feat = base_enc(q_pkt, q_br, q_st, q_hd)

        # 原型 = 每类 mean
        n_way_actual = int(s_y.max().item()) + 1
        protos = []
        for c in range(n_way_actual):
            protos.append(s_feat[s_y == c].mean(dim=0))
        protos = torch.stack(protos)  # (N, D)

        # 特征 1: inter-class cos distance (prototype 之间的余弦距离均值)
        p_norm = torch.nn.functional.normalize(protos, dim=-1)
        inter_sim = p_norm @ p_norm.t()
        # 上三角（不含对角线）
        mask = torch.triu(torch.ones_like(inter_sim), diagonal=1).bool()
        inter_dist = (1.0 - inter_sim[mask]).mean().item()

        # 特征 2: intra-class std (support 每类内部的 embedding 方差均值)
        intra_vars = []
        for c in range(n_way_actual):
            mask_c = s_y == c
            if mask_c.sum() > 1:
                intra_vars.append(s_feat[mask_c].std(dim=0).mean().item())
        intra_var = float(np.mean(intra_vars)) if intra_vars else 0.0

        # 特征 3: support 样本数 N*K（隐式由 n_way/k_shot 决定, 这里取 ratio）
        feat_ratio = q_query / k_shot

        # 特征 4: query 类别纯度（query 中 support 类的占比）
        # ProtoNet 必然预测到 N 类，所以 query 类别纯度恒为 1（跳过）

        # 特征 5: support embedding 的 L2 norm 均值（应 ≈ 1，因 encoder 通常 normalize）
        # 跳过，因为 TrafficEncoder 输出未必归一化

        # 特征 6: 每类的 support 大小（理论上 = K, 反映 K-shot 稳定性）
        # K 已固定（跳过）

        feats = np.array([inter_dist, intra_var, feat_ratio, n_way_actual], dtype=np.float32)

        # 用每个 preset encoder 跑 ProtoNet，记录 accuracy
        accs_per_preset = []
        for preset_name in PRESETS:
            enc = encoders[PRESET_TO_IDX[preset_name]]
            m = ProtoNet(enc).to(device)
            logits, _ = m.forward_with_confidence(
                (s_pkt, s_br, s_st, s_hd), (q_pkt, q_br, q_st, q_hd), s_y)
            q_y = ep["query_labels"].to(device)
            acc = (logits.argmax(dim=-1) == q_y).float().mean().item()
            accs_per_preset.append(acc)
        all_feats.append(feats)
        all_accs.append(accs_per_preset)

    feats = np.stack(all_feats)
    accs = np.array(all_accs)
    best = accs.argmax(axis=1)
    return feats, accs, best

def static_rule(n_way):
    """§4.5 经验表：N→preset。"""
    if n_way <= 2:
        return "none"
    elif n_way <= 10:
        return "medium"
    else:
        return "weak"

class PresetMLP(nn.Module):
    """Episode 特征 → 4 preset 概率分布的小 MLP。"""

    def __init__(self, in_dim=4, hidden=32, n_presets=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_presets),
        )

    def forward(self, x):
        return self.net(x)

def train_meta_mlp(feats, best, train_ratio=0.7, epochs=200, lr=1e-3, seed=42):
    """训练 episode-level MLP, 交叉熵损失。"""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    N = len(feats)
    idx = rng.permutation(N)
    n_tr = int(N * train_ratio)
    tr_idx, te_idx = idx[:n_tr], idx[n_tr:]

    # 标准化特征（按训练集统计量）
    mu, sigma = feats[tr_idx].mean(0), feats[tr_idx].std(0) + 1e-8

    x_tr = torch.tensor((feats[tr_idx] - mu) / sigma, dtype=torch.float32)
    y_tr = torch.tensor(best[tr_idx], dtype=torch.long)
    x_te = torch.tensor((feats[te_idx] - mu) / sigma, dtype=torch.float32)
    y_te = torch.tensor(best[te_idx], dtype=torch.long)

    model = PresetMLP(in_dim=feats.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        opt.step()

    # 评估
    model.eval()
    with torch.no_grad():
        pred_tr = model(x_tr).argmax(dim=-1).numpy()
        pred_te = model(x_te).argmax(dim=-1).numpy()
    tr_acc = float((pred_tr == best[tr_idx]).mean())
    te_acc = float((pred_te == best[te_idx]).mean())
    return {
        "tr_acc": tr_acc, "te_acc": te_acc,
        "n_train": int(n_tr), "n_test": int(N - n_tr),
        "mu": mu.tolist(), "sigma": sigma.tolist(),
    }, pred_te, te_idx

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cic", choices=["cic", "ustc"])
    p.add_argument("--ablation_dir", default="output/checkpoints/ablation",
                   help="4 preset ckpt 所在目录（命名 {dataset}_simclr_ep30_{preset}.pt）")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--num_episodes", type=int, default=600)
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    h5_path = f"output/{args.dataset}_full.h5"
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    # 加载 4 个 preset encoder
    encoders = []
    for preset in PRESETS:
        ckpt_name = PRESET_CKPT_NAMES[preset].format(dataset=args.dataset)
        ckpt = os.path.join(args.ablation_dir, ckpt_name)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"ckpt missing: {ckpt}")
        encoders.append(load_encoder(ckpt, args.device))
    print(f"loaded {len(encoders)} encoders: {PRESETS}")

    # 子集
    if args.dataset == "iscx":
        idx_tr, _, _ = stratified_split_indices(h5_path, 0.7, 0.15, seed=42)
    else:
        idx_tr, _, _ = time_aware_split_indices(h5_path, 0.7, 0.15)
    rng = np.random.RandomState(42)
    idx_sub = rng.choice(idx_tr, min(args.max_samples, len(idx_tr)), replace=False)
    ds = H5FlowDataset(h5_path, indices=list(idx_sub), fit=True)
    print(f"dataset={args.dataset} n={len(ds)} classes={len(set(ds.y.tolist()))}")

    # 跨 seed 收集 episode 数据
    all_feats, all_accs, all_best = [], [], []
    for s in range(args.num_seeds):
        seed = 42 + s * 100
        t0 = time.time()
        feats, accs, best = collect_episode_data(
            encoders, ds, args.n_way, args.k_shot, args.q_query,
            args.num_episodes, seed, args.device)
        all_feats.append(feats)
        all_accs.append(accs)
        all_best.append(best)
        # 平均 accuracy
        mean_acc_per_preset = accs.mean(axis=0)
        preset_mean_str = " ".join(
            f"{p}={mean_acc_per_preset[i]:.4f}" for i, p in enumerate(PRESETS))
        # best preset 分布
        unique, counts = np.unique(best, return_counts=True)
        dist_str = ", ".join(f"{IDX_TO_PRESET[u]}={c}" for u, c in zip(unique, counts))
        print(f"  seed={seed} ({time.time()-t0:.1f}s): {preset_mean_str}; best={dist_str}")

    # 合并跨 seed 数据
    feats_all = np.concatenate(all_feats, axis=0)
    accs_all = np.concatenate(all_accs, axis=0)
    best_all = np.concatenate(all_best, axis=0)
    print(f"\n总 episodes: {len(feats_all)} (3 seeds × {args.num_episodes})")

    # per-preset 整体平均（oracle-baseline）
    preset_means = accs_all.mean(axis=0)
    print(f"\n=== 静态规则 (oracle) 表现 ===")
    for i, p in enumerate(PRESETS):
        print(f"  {p}: {preset_means[i]:.4f}")
    oracle_best_idx = int(preset_means.argmax())
    oracle_acc = float(preset_means[oracle_best_idx])
    print(f"  oracle best: {PRESETS[oracle_best_idx]} ({oracle_acc:.4f})")

    # 静态规则 (按 N)
    n_way_unique = sorted(set(feats_all[:, 3].astype(int).tolist()))
    print(f"\n=== 静态规则 (§4.5 N→preset) ===")
    for nw in n_way_unique:
        mask = feats_all[:, 3] == nw
        if mask.sum() == 0:
            continue
        rule_preset = static_rule(nw)
        rule_idx = PRESET_TO_IDX[rule_preset]
        rule_acc = float(accs_all[mask, rule_idx].mean())
        oracle_acc_n = float(accs_all[mask].mean(axis=1).max())  # 每个 episode 选最优
        print(f"  N={nw}: 静态规则={rule_preset} (acc={rule_acc:.4f})  "
              f"oracle={oracle_acc_n:.4f}")

    # 训练 meta-MLP
    print(f"\n=== 训练 episode-level meta-MLP ===")
    mlp_result, pred_te, te_idx = train_meta_mlp(
        feats_all, best_all, train_ratio=0.7, epochs=200, lr=1e-3, seed=42)
    print(f"  train_acc={mlp_result['tr_acc']:.4f}  "
          f"test_acc={mlp_result['te_acc']:.4f}")

    # meta-MLP 选中后, 用对应 preset 的 accuracy
    meta_acc_per_episode = np.array([
        accs_all[i, pred] for i, pred in zip(te_idx, pred_te)])
    static_pred = np.array([PRESET_TO_IDX[static_rule(int(feats_all[i, 3]))]
                            for i in te_idx])
    static_acc_per_episode = np.array([
        accs_all[i, pred] for i, pred in zip(te_idx, static_pred)])
    oracle_acc_per_episode = accs_all[te_idx].max(axis=1)

    print(f"\n=== 测试集对比 (n={len(te_idx)}) ===")
    print(f"  meta-MLP 平均 acc: {meta_acc_per_episode.mean():.4f}")
    print(f"  静态规则 平均 acc: {static_acc_per_episode.mean():.4f}")
    print(f"  Oracle (per-ep best) 平均 acc: {oracle_acc_per_episode.mean():.4f}")

    # 输出
    out = {
        "dataset": args.dataset,
        "n_way": args.n_way, "k_shot": args.k_shot,
        "num_episodes_per_seed": args.num_episodes, "num_seeds": args.num_seeds,
        "total_episodes": int(len(feats_all)),
        "presets": PRESETS,
        "feature_names": ["inter_class_cos_dist", "intra_class_std",
                          "q_per_k_ratio", "n_way"],
        "preset_means_oracle": {p: float(preset_means[i]) for i, p in enumerate(PRESETS)},
        "oracle_best_preset": PRESETS[oracle_best_idx],
        "static_rule_comparison": {
            f"N={nw}": {
                "rule_preset": static_rule(nw),
                "rule_acc": float(accs_all[feats_all[:, 3] == nw,
                                            PRESET_TO_IDX[static_rule(nw)]].mean()),
                "oracle_acc_per_episode": float(accs_all[feats_all[:, 3] == nw].mean(axis=1).max()),
            } for nw in n_way_unique if (feats_all[:, 3] == nw).sum() > 0
        },
        "meta_mlp": {
            "tr_acc": float(mlp_result["tr_acc"]),
            "te_acc": float(mlp_result["te_acc"]),
            "n_train": mlp_result["n_train"], "n_test": mlp_result["n_test"],
            "test_acc_with_meta_mlp_choice": float(meta_acc_per_episode.mean()),
            "test_acc_with_static_rule_choice": float(static_acc_per_episode.mean()),
            "test_acc_with_oracle_choice": float(oracle_acc_per_episode.mean()),
        },
    }
    out_path = f"output/results/adaptive_aug_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")

if __name__ == "__main__":
    main()