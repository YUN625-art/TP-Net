"""
compute_cohens_d_episode.py
===========================
基于 baseline_random_matched_*.json 的 per-episode raw 数据 +
SSL encoder 对应 ep-accuracy（来自 make_loaders/fewshot_eval），
计算 episode-level pooled Cohen's d。

协议：
- SSL ep-accuracy：从 fewshot_*.json 的 "results" → 5 seed × 600 episode
- Random ep-accuracy：baseline_random_matched_*.json 的 per-episode raw 序列
- d = (mean_ssl - mean_random) / pooled_std
- pooled_std = sqrt(((n1-1)*sd1^2 + (n2-1)*sd2^2) / (n1+n2-2))
"""

import argparse
import json
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)


def cohens_d_episode_level(ssl_eps, random_eps):
    """Pooled Cohen's d from two lists of per-episode accuracies."""
    a = np.array(ssl_eps, dtype=np.float64)
    b = np.array(random_eps, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled_std = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled_std == 0:
        return None
    d = (ma - mb) / pooled_std
    # Welch's t-test 近似：仅 print, not used for primary result
    se = pooled_std * np.sqrt(1 / na + 1 / nb)
    t = (ma - mb) / se
    return {
        "n_ssl": na, "n_random": nb,
        "mean_ssl": float(ma), "mean_random": float(mb),
        "std_ssl": float(sa), "std_random": float(sb),
        "delta_pp": float((ma - mb) * 100),
        "pooled_std": float(pooled_std),
        "cohens_d_episode_pooled": float(d),
        "welch_t_approx": float(t),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cic", "ustc", "iscx"])
    args = p.parse_args()

    # SSL: fewshot_*.json (5 seeds × 600 ep)
    # 候选路径按优先级尝试
    candidates = [
        f"output/results/fewshot_{args.dataset}_simclr_protonet_{'2w5s' if args.dataset == 'iscx' else '5w5s'}.json",
        f"output/results/fewshot_{args.dataset}_simclr_protonet.json",
        f"output/results/fewshot_{args.dataset}.json",
        f"output/results/fewshot_simclr_{args.dataset}.json",
    ]
    ssl_path = None
    for c in candidates:
        if os.path.exists(c):
            ssl_path = c
            break
    if ssl_path is None:
        print(f"[ERROR] SSL file not found. tried: {candidates}")
        return
    print(f"SSL file: {ssl_path}")

    # Random: baseline_random_matched_*.json
    rand_path = f"output/results/baseline_random_matched_{args.dataset}.json"

    if not os.path.exists(ssl_path):
        print(f"[ERROR] SSL file not found: {ssl_path}")
        return
    if not os.path.exists(rand_path):
        print(f"[ERROR] Random file not found: {rand_path}")
        return

    with open(ssl_path, "r", encoding="utf-8") as f:
        ssl_data = json.load(f)
    with open(rand_path, "r", encoding="utf-8") as f:
        rand_data = json.load(f)

    # 从 SSL JSON 提取 per-episode 序列
    ssl_eps = []
    # fewshot JSON 格式: results = list[dict(seed)]，每个 dict 含 raw 序列
    if isinstance(ssl_data.get("results"), list):
        for sd in ssl_data["results"]:
            if isinstance(sd, dict) and "raw" in sd:
                ssl_eps.extend(sd["raw"])
    elif "seeds" in ssl_data:
        for sd in ssl_data["seeds"]:
            if "raw" in sd:
                ssl_eps.extend(sd["raw"])
            elif "episodes" in sd:
                ssl_eps.extend(sd["episodes"])

    # Random: 直接从 baseline_random_matched_*.json 拼接所有 seed 的 raw
    rand_eps = []
    for sd in rand_data.get("seeds", []):
        if "raw" in sd:
            rand_eps.extend(sd["raw"])

    print(f"[{args.dataset}] SSL per-episode count: {len(ssl_eps)}")
    print(f"[{args.dataset}] Random per-episode count: {len(rand_eps)}")

    if not ssl_eps or not rand_eps:
        print(f"[ERROR] missing per-episode data")
        return

    result = cohens_d_episode_level(ssl_eps, rand_eps)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 追加到 baseline_random_matched JSON
    rand_data["cohens_d_episode_level"] = result
    with open(rand_path, "w", encoding="utf-8") as f:
        json.dump(rand_data, f, indent=2, ensure_ascii=False)
    print(f"[已更新] {rand_path}")


if __name__ == "__main__":
    main()
