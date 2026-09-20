"""
openset_fsl_eval.py
===================
Open-set few-shot learning 实证。

§5.4.4 提出 3 条开集路径，本文实证前 2 条：
1. 距离阈值拒绝（cosine 距离 vs prototype）
2. Softmax 概率校准（temperature scaling）

3 种评估场景：
- Closed-set baseline（标准 N-way K-shot，仅作 sanity check）
- Novel as "known unknown"（support 含 novel，但 query 混入新类）
- True unknown（1-2 类从未在 support/query 出现，混入 query）

评测指标（开集 FSL 文献标准）：
- AUROC：未知样本置信度低于已知样本的概率
- FPR@95TPR：95% 已知样本正确接受时的误拒率
- 闭集精度 @ 最优阈值
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
from src.data.load_h5 import H5FlowDataset, time_aware_split_indices
from src.data.episodes import EpisodeSampler

H5_PATHS = {"cic": "output/cic_full.h5", "ustc": "output/ustc_full.h5"}

def load_encoder(ckpt, device):
    enc = TrafficEncoder(embed_dim=128)
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        enc.load_state_dict(sd["encoder_state"])
    return enc.to(device).eval()

def auroc_fpr_at_tpr(scores_known, scores_unknown):
    """计算 AUROC 和 FPR@95TPR（开集 FSL 标准）。

    Args:
        scores_known: (N_known,) 已知样本置信度（应更高）
        scores_unknown: (N_unknown,) 未知样本置信度（应更低）

    Returns:
        auroc, fpr95
    """
    y = np.concatenate([np.ones_like(scores_known), np.zeros_like(scores_unknown)])
    s = np.concatenate([scores_known, scores_unknown])
    order = np.argsort(-s)  # descending
    y_sorted = y[order]

    P = scores_known.size
    N = scores_unknown.size

    # AUROC via Mann-Whitney U
    ranks = np.argsort(np.argsort(s)) + 1.0
    sum_ranks_known = ranks[:P].sum()
    auroc = (sum_ranks_known - P * (P + 1) / 2) / (P * N)

    # FPR@95TPR: 找 TPR=0.95 的阈值
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    tpr = cum_tp / P
    fpr = cum_fp / N
    # 第一个 tpr>=0.95 的位置
    idx95 = np.searchsorted(tpr, 0.95)
    if idx95 >= len(fpr):
        idx95 = len(fpr) - 1
    fpr95 = float(fpr[idx95])
    return float(auroc), fpr95

@torch.no_grad()
def gather_embeddings(encoder, ds, indices, device, batch_size=128):
    """批量提取 embedding，避免逐样本循环。"""
    embs = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        p = torch.stack([ds[int(i)][0] for i in batch_idx]).to(device)
        br = torch.stack([ds[int(i)][1] for i in batch_idx]).to(device)
        st = torch.stack([ds[int(i)][2] for i in batch_idx]).to(device)
        hd = torch.stack([ds[int(i)][3] for i in batch_idx]).to(device)
        e = encoder(p, br, st, hd).cpu().numpy()
        embs.append(e)
    return np.concatenate(embs, axis=0)

def build_class_splits(ds, n_base=7, min_per_class=40, seed=42):
    """构造 base / novel / unknown 三组索引（按类别 hash 划分）。

    只保留样本数 ≥ min_per_class 的类（保证 5-shot + q_query episode 可构造）。
    然后按 70/20/10 分到 base/novel/unknown。

    Returns:
        base_idx, novel_idx, unknown_idx（list[int]，对应 ds 的局部索引）
        base_classes, novel_classes, unknown_classes
    """
    from collections import Counter
    rng = np.random.RandomState(seed)
    cls_count = Counter(int(y) for y in ds.y.tolist())
    usable = sorted([c for c, n in cls_count.items() if n >= min_per_class])
    if len(usable) < 4:
        raise ValueError(f"usable classes ({len(usable)}) < 4; "
                         f"counts: {dict(cls_count)}")
    rng.shuffle(usable)
    n_total = len(usable)
    n_base = min(n_base, n_total - 2)
    n_novel = max(1, (n_total - n_base) // 2)
    n_unknown = n_total - n_base - n_novel
    base_classes = sorted(usable[:n_base])
    novel_classes = sorted(usable[n_base:n_base + n_novel])
    unknown_classes = sorted(usable[n_base + n_novel:])

    def subset(cls_list):
        return [i for i, y in enumerate(ds.y.tolist()) if y in cls_list]

    return (subset(base_classes), subset(novel_classes), subset(unknown_classes),
            base_classes, novel_classes, unknown_classes)

def eval_scenario_A_closed(encoder, ds, base_idx, n_way, k_shot, q_query,
                            num_episodes, seed, device):
    """场景 A：closed-set 标准 N-way K-shot（sanity check）。"""
    model = ProtoNet(encoder).to(device)
    cls_to_idx = {}
    for i in base_idx:
        c = int(ds.y[i])
        cls_to_idx.setdefault(c, []).append(i)
    sampler = EpisodeSampler(ds, n_way=n_way, k_shot=k_shot, q_query=q_query,
                             seed=seed, cls_to_idx=cls_to_idx,
                             num_classes=n_way)
    accs = []
    confs = []
    for ep in sampler.iter_episodes(num_episodes):
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
        logits, conf = model.forward_with_confidence(
            (s_pkt, s_br, s_st, s_hd), (q_pkt, q_br, q_st, q_hd), s_y)
        pred = logits.argmax(dim=-1)
        q_y = ep["query_labels"].to(device)
        accs.append((pred == q_y).float().mean().item())
        confs.append(conf.cpu().numpy())
    return float(np.mean(accs)), np.concatenate(confs)

def eval_scenario_B_novel_known_unknown(encoder, ds, base_idx, novel_idx,
                                        n_way, k_shot, q_query_base, q_query_novel,
                                        num_episodes, seed, device):
    """场景 B：open-set with novel as "known unknown"。

    设计：novel 是"已知未知"（系统知道存在新类，但当前 episode 没把它
    放进 support）。Support 只包含 base 类。Query 混入 novel 类样本。
    novel 样本必然被错分为 base（不在 support 标签范围），但 confidence
    应低于 base → 可被阈值拒绝。

    与场景 C 的差异：
    - 场景 B：novel 来自"已知存在但本 episode 未在 support"的类
        （在 SSL encoder 预训练时可能见过其部分样本，部分信息泄漏）
    - 场景 C：unknown 来自"SSL encoder 完全未见过"的类
        （真正的 zero-shot unknown）
    """
    model = ProtoNet(encoder).to(device)

    # 分组
    base_cls = {}
    novel_cls = {}
    for i in base_idx:
        base_cls.setdefault(int(ds.y[i]), []).append(i)
    for i in novel_idx:
        novel_cls.setdefault(int(ds.y[i]), []).append(i)
    base_classes = sorted(base_cls.keys())
    novel_classes = sorted(novel_cls.keys())

    rng = np.random.RandomState(seed)
    base_conf_list, novel_conf_list = [], []
    for ep_id in range(num_episodes):
        sel_base = rng.choice(len(base_classes), n_way, replace=False)
        sel_novel = rng.choice(len(novel_classes), 1, replace=False)[0]

        s_idx = []; s_labels = []
        for new_c, orig_c in enumerate(sel_base):
            cls_i = base_cls[base_classes[orig_c]]
            picks = rng.choice(cls_i, k_shot, replace=False)
            s_idx.extend(picks.tolist()); s_labels.extend([new_c] * k_shot)
        s_y = torch.tensor(s_labels).to(device)

        # query = base + novel
        q_idx = []; q_is_novel = []
        for orig_c in sel_base:
            cls_i = base_cls[base_classes[orig_c]]
            picks = rng.choice(cls_i, q_query_base, replace=False)
            q_idx.extend(picks.tolist()); q_is_novel.extend([0] * q_query_base)
        # 加入 novel 样本（必然被错分为 base）
        novel_cls_i = novel_cls[novel_classes[sel_novel]]
        novel_picks = rng.choice(novel_cls_i, q_query_novel, replace=False)
        q_idx.extend(novel_picks.tolist()); q_is_novel.extend([1] * q_query_novel)
        q_is_novel = np.array(q_is_novel)

        s_pkt = torch.stack([ds[int(i)][0] for i in s_idx]).to(device)
        s_br = torch.stack([ds[int(i)][1] for i in s_idx]).to(device)
        s_st = torch.stack([ds[int(i)][2] for i in s_idx]).to(device)
        s_hd = torch.stack([ds[int(i)][3] for i in s_idx]).to(device)
        q_pkt = torch.stack([ds[int(i)][0] for i in q_idx]).to(device)
        q_br = torch.stack([ds[int(i)][1] for i in q_idx]).to(device)
        q_st = torch.stack([ds[int(i)][2] for i in q_idx]).to(device)
        q_hd = torch.stack([ds[int(i)][3] for i in q_idx]).to(device)
        logits, conf = model.forward_with_confidence(
            (s_pkt, s_br, s_st, s_hd), (q_pkt, q_br, q_st, q_hd), s_y)
        conf = conf.cpu().numpy()

        base_conf_list.append(conf[q_is_novel == 0])
        novel_conf_list.append(conf[q_is_novel == 1])

    base_conf = np.concatenate(base_conf_list)
    novel_conf = np.concatenate(novel_conf_list)
    auroc, fpr95 = auroc_fpr_at_tpr(base_conf, novel_conf)
    return {
        "n_episodes": num_episodes,
        "n_base_query": int(len(base_conf)),
        "n_novel_query": int(len(novel_conf)),
        "base_conf_mean": float(base_conf.mean()),
        "novel_conf_mean": float(novel_conf.mean()),
        "conf_gap": float(base_conf.mean() - novel_conf.mean()),
        "auroc": auroc,
        "fpr_at_95tpr": fpr95,
    }

def eval_scenario_C_true_unknown(encoder, ds, base_idx, unknown_idx,
                                  n_way, k_shot, q_query_base, q_query_unk,
                                  num_episodes, seed, device):
    """场景 C：true open-set。

    Support = base 类。Query = base + 1 个 unknown 类。
    unknown 类从未在 support / 预训练中出现（真正的 zero-shot unknown）。
    """
    model = ProtoNet(encoder).to(device)

    base_cls = {}
    unk_cls = {}
    for i in base_idx:
        base_cls.setdefault(int(ds.y[i]), []).append(i)
    for i in unknown_idx:
        unk_cls.setdefault(int(ds.y[i]), []).append(i)
    base_classes = sorted(base_cls.keys())
    unk_classes = sorted(unk_cls.keys())

    rng = np.random.RandomState(seed)
    base_conf_list, unk_conf_list = [], []
    for ep_id in range(num_episodes):
        sel_base = rng.choice(len(base_classes), n_way, replace=False)
        sel_unk = rng.choice(len(unk_classes), 1, replace=False)[0]

        s_idx = []; s_labels = []
        for new_c, orig_c in enumerate(sel_base):
            cls_i = base_cls[base_classes[orig_c]]
            picks = rng.choice(cls_i, k_shot, replace=False)
            s_idx.extend(picks.tolist()); s_labels.extend([new_c] * k_shot)
        s_y = torch.tensor(s_labels).to(device)

        q_idx = []; q_is_unk = []
        for orig_c in sel_base:
            cls_i = base_cls[base_classes[orig_c]]
            picks = rng.choice(cls_i, q_query_base, replace=False)
            q_idx.extend(picks.tolist()); q_is_unk.extend([0] * q_query_base)
        # 加入 unknown 样本
        unk_cls_i = unk_cls[unk_classes[sel_unk]]
        unk_picks = rng.choice(unk_cls_i, q_query_unk, replace=False)
        q_idx.extend(unk_picks.tolist()); q_is_unk.extend([1] * q_query_unk)
        q_is_unk = np.array(q_is_unk)

        s_pkt = torch.stack([ds[int(i)][0] for i in s_idx]).to(device)
        s_br = torch.stack([ds[int(i)][1] for i in s_idx]).to(device)
        s_st = torch.stack([ds[int(i)][2] for i in s_idx]).to(device)
        s_hd = torch.stack([ds[int(i)][3] for i in s_idx]).to(device)
        q_pkt = torch.stack([ds[int(i)][0] for i in q_idx]).to(device)
        q_br = torch.stack([ds[int(i)][1] for i in q_idx]).to(device)
        q_st = torch.stack([ds[int(i)][2] for i in q_idx]).to(device)
        q_hd = torch.stack([ds[int(i)][3] for i in q_idx]).to(device)
        logits, conf = model.forward_with_confidence(
            (s_pkt, s_br, s_st, s_hd), (q_pkt, q_br, q_st, q_hd), s_y)
        conf = conf.cpu().numpy()

        base_conf_list.append(conf[q_is_unk == 0])
        unk_conf_list.append(conf[q_is_unk == 1])

    base_conf = np.concatenate(base_conf_list)
    unk_conf = np.concatenate(unk_conf_list)
    auroc, fpr95 = auroc_fpr_at_tpr(base_conf, unk_conf)
    return {
        "n_episodes": num_episodes,
        "n_base_query": int(len(base_conf)),
        "n_unknown_query": int(len(unk_conf)),
        "base_conf_mean": float(base_conf.mean()),
        "unknown_conf_mean": float(unk_conf.mean()),
        "conf_gap": float(base_conf.mean() - unk_conf.mean()),
        "auroc": auroc,
        "fpr_at_95tpr": fpr95,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cic", choices=["cic", "ustc"])
    p.add_argument("--ckpt", default="output/checkpoints/cic_simclr_ep50.pt")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--q_query_unk", type=int, default=15,
                   help="场景 C 中 unknown 类 query 数")
    p.add_argument("--num_episodes", type=int, default=200)
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--n_base", type=int, default=7,
                   help="base 类数（剩余分给 novel/unknown）")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_path", default=None,
                   help="结果输出路径（默认 output/results/openset_fsl_<dataset>.json）")
    args = p.parse_args()

    h5_path = H5_PATHS[args.dataset]
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    # 加载 + 子集
    idx_tr, _, _ = time_aware_split_indices(h5_path, 0.7, 0.15)
    rng = np.random.RandomState(42)
    idx_sub = rng.choice(idx_tr, min(args.max_samples, len(idx_tr)), replace=False)
    ds = H5FlowDataset(h5_path, indices=list(idx_sub), fit=True)
    print(f"dataset={args.dataset} n={len(ds)} classes={len(set(ds.y.tolist()))}")

    encoder = load_encoder(args.ckpt, args.device)

    # 构造 base/novel/unknown 分组
    base_idx, novel_idx, unknown_idx, base_cls, novel_cls, unk_cls = build_class_splits(
        ds, n_base=args.n_base, seed=42)
    print(f"base={len(base_cls)} novel={len(novel_cls)} unknown={len(unk_cls)}")
    print(f"  base samples={len(base_idx)}  novel={len(novel_idx)}  unknown={len(unknown_idx)}")

    results = {
        "dataset": args.dataset, "ckpt": args.ckpt,
        "n_way": args.n_way, "k_shot": args.k_shot,
        "num_episodes": args.num_episodes, "num_seeds": args.num_seeds,
        "n_base_classes": len(base_cls), "n_novel_classes": len(novel_cls),
        "n_unknown_classes": len(unk_cls),
        "scenarios": {},
    }

    # 场景 A：closed-set sanity check
    print("\n=== 场景 A：closed-set（sanity check） ===")
    # 动态调整 n_way 以适应实际 base 类数
    n_way_a = min(args.n_way, len(base_cls))
    if n_way_a < args.n_way:
        print(f"  [调整] n_way: {args.n_way} → {n_way_a} (base 类数={len(base_cls)})")
    a_accs, a_confs = [], []
    t0 = time.time()
    for s in range(args.num_seeds):
        seed = 42 + s * 100
        acc, conf = eval_scenario_A_closed(
            encoder, ds, base_idx, n_way_a, args.k_shot, args.q_query,
            args.num_episodes, seed, args.device)
        a_accs.append(acc)
        a_confs.append(conf)
        print(f"  seed={seed} acc={acc:.4f} conf_mean={conf.mean():.4f}")
    results["scenarios"]["A_closed"] = {
        "n_way_used": n_way_a,
        "closed_set_acc_mean": float(np.mean(a_accs)),
        "closed_set_acc_std": float(np.std(a_accs, ddof=1)) if len(a_accs) > 1 else 0.0,
        "closed_set_conf_mean": float(np.mean(np.concatenate(a_confs))),
        "eval_sec": round(time.time() - t0, 1),
    }

    # 场景 B：novel as "known unknown"（support 仅 base，query 混入 novel）
    print("\n=== 场景 B：novel 在 support 之外，query 混入 novel ===")
    n_way_b = min(args.n_way, len(base_cls))
    if n_way_b < args.n_way:
        print(f"  [调整] n_way: {args.n_way} → {n_way_b}")
    b_results = []
    t0 = time.time()
    for s in range(args.num_seeds):
        seed = 42 + s * 100
        r = eval_scenario_B_novel_known_unknown(
            encoder, ds, base_idx, novel_idx,
            n_way_b, args.k_shot, args.q_query, args.q_query,
            args.num_episodes, seed, args.device)
        r["seed"] = seed
        b_results.append(r)
        print(f"  seed={seed} base_conf={r['base_conf_mean']:.4f}  "
              f"novel_conf={r['novel_conf_mean']:.4f}  "
              f"auroc={r['auroc']:.4f}  fpr95={r['fpr_at_95tpr']:.4f}")
    aurocs_b = [r["auroc"] for r in b_results]
    results["scenarios"]["B_novel_known_unknown"] = {
        "n_way_used": n_way_b,
        "auroc_mean": float(np.mean(aurocs_b)),
        "auroc_std": float(np.std(aurocs_b, ddof=1)) if len(aurocs_b) > 1 else 0.0,
        "fpr_at_95tpr_mean": float(np.mean([r["fpr_at_95tpr"] for r in b_results])),
        "conf_gap_mean": float(np.mean([r["conf_gap"] for r in b_results])),
        "raw": [{k: r[k] for k in ("auroc", "fpr_at_95tpr", "conf_gap")} for r in b_results],
        "eval_sec": round(time.time() - t0, 1),
    }

    # 场景 C：true unknown
    print("\n=== 场景 C：true unknown（从未出现在 support/预训练） ===")
    n_way_c = min(args.n_way, len(base_cls))
    if n_way_c < args.n_way:
        print(f"  [调整] n_way: {args.n_way} → {n_way_c}")
    c_results = []
    t0 = time.time()
    for s in range(args.num_seeds):
        seed = 42 + s * 100
        r = eval_scenario_C_true_unknown(
            encoder, ds, base_idx, unknown_idx,
            n_way_c, args.k_shot, args.q_query, args.q_query_unk,
            args.num_episodes, seed, args.device)
        r["seed"] = seed
        c_results.append(r)
        print(f"  seed={seed} base_conf={r['base_conf_mean']:.4f}  "
              f"unknown_conf={r['unknown_conf_mean']:.4f}  "
              f"auroc={r['auroc']:.4f}  fpr95={r['fpr_at_95tpr']:.4f}")
    aurocs_c = [r["auroc"] for r in c_results]
    results["scenarios"]["C_true_unknown"] = {
        "n_way_used": n_way_c,
        "auroc_mean": float(np.mean(aurocs_c)),
        "auroc_std": float(np.std(aurocs_c, ddof=1)) if len(aurocs_c) > 1 else 0.0,
        "fpr_at_95tpr_mean": float(np.mean([r["fpr_at_95tpr"] for r in c_results])),
        "conf_gap_mean": float(np.mean([r["conf_gap"] for r in c_results])),
        "raw": [{k: r[k] for k in ("auroc", "fpr_at_95tpr", "conf_gap")} for r in c_results],
        "eval_sec": round(time.time() - t0, 1),
    }

    out_path = args.out_path or f"output/results/openset_fsl_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_path}")
    print(f"\n=== 总结 ===")
    print(f"场景 A 闭集 acc:        {results['scenarios']['A_closed']['closed_set_acc_mean']:.4f}")
    print(f"场景 B known-unknown  AUROC: {results['scenarios']['B_novel_known_unknown']['auroc_mean']:.4f}")
    print(f"场景 C true unknown   AUROC: {results['scenarios']['C_true_unknown']['auroc_mean']:.4f}")

if __name__ == "__main__":
    main()