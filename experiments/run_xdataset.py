"""
run_xdataset.py
===============
实验 6：跨数据集预训练。

在数据集 A 上预训练 encoder，然后在数据集 B 上评估 ProtoNet
（验证 SSL 表征的领域无关性）。
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.utils import TqdmLogger, save_json

PAIRS = [
    ("cic", "ustc"),
    ("ustc", "cic"),
    ("cic", "iscx"),
    ("iscx", "cic"),
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max_samples", type=int, default=50000,
                   help="预训练样本数（分层抽样）")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--num_episodes", type=int, default=300)
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--pairs", nargs="+", default=None,
                   help="跨数据集对（src-dst），默认 PAIRS")
    p.add_argument("--skip_pretrain", action="store_true")
    args = p.parse_args()

    pairs = args.pairs or PAIRS
    # pairs 可以是 [(src1,dst1),(src2,dst2),...] 列表，或 ["src1->dst1","src2->dst2",...] 字符串
    if isinstance(pairs[0], str):
        pairs = [tuple(p.split("->")) for p in pairs]
    log = TqdmLogger(os.path.join("output/logs", "xdataset.log"))
    log.info(f"跨数据集实验 pairs={pairs} max_samples={args.max_samples}")

    out_dir = os.path.join("output/checkpoints", "xdataset")
    os.makedirs(out_dir, exist_ok=True)

    summary = {"pairs": {}}
    for src, dst in pairs:
        log.info(f"==== {src} → {dst} ====")
        # 优先找最新的 ckpt
        candidates = [
            os.path.join(out_dir, f"{src}_simclr_ep{args.epochs}.pt"),
            os.path.join(out_dir, f"{src}_simclr.pt"),
            os.path.join("output/checkpoints", f"{src}_simclr_ep{args.epochs}.pt"),
        ]
        ckpt = next((c for c in candidates if os.path.exists(c)), None)

        if ckpt is None and not args.skip_pretrain:
            cmd = [
                sys.executable, "experiments/run_pretrain.py",
                "--dataset", src, "--algo", "simclr",
                "--epochs", str(args.epochs),
                "--max_samples", str(args.max_samples),
                "--save_every", str(args.epochs),
                "--out_dir", out_dir,
            ]
            log.info(f"启动预训练: {' '.join(cmd)}")
            r = subprocess.run(cmd)
            if r.returncode != 0:
                log.info(f"预训练 {src} 失败")
                continue
            ckpt = next((c for c in candidates if os.path.exists(c)), None)
            if ckpt is None:
                log.info(f"找不到 {src} 预训练 ckpt")
                continue
        elif ckpt is None:
            log.info(f"跳过预训练但 ckpt 不存在")
            continue

        log.info(f"使用 ckpt: {ckpt}")

        # 在 dst 上评估
        cmd = [
            sys.executable, "experiments/run_fewshot.py",
            "--dataset", dst, "--encoder_ckpt", ckpt,
            "--algo", "protonet",
            "--n_way", str(args.n_way), "--k_shot", str(args.k_shot),
            "--q_query", str(args.q_query),
            "--num_episodes", str(args.num_episodes),
            "--num_seeds", str(args.num_seeds),
        ]
        log.info(f"启动评估: {' '.join(cmd)}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            log.info(f"评估 {dst} 失败")
            continue

        # 读取结果
        json_path = os.path.join(
            "output/results",
            f"fewshot_{dst}_protonet_{args.n_way}w{args.k_shot}s.json")
        if os.path.exists(json_path):
            import json
            with open(json_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                agg = d.get("aggregate", {})
                summary["pairs"][f"{src}->{dst}"] = {
                    "mean": agg.get("mean"),
                    "std": agg.get("std"),
                }

    summary["args"] = vars(args)
    out_path = os.path.join("output/results", "xdataset.json")
    save_json(summary, out_path)
    log.info(f"跨数据集结果已保存 {out_path}")

if __name__ == "__main__":
    main()
