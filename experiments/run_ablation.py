"""
run_ablation.py
===============
实验 5：增强策略消融。

四种预设：none / weak / medium / strong
每个预设重新预训练 encoder → ProtoNet 评估
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.utils import get_device, set_seed, TqdmLogger, save_json

PRESETS = ["none", "weak", "medium", "strong"]

def run_pretrain(dataset, preset, epochs, out_dir):
    cmd = [
        sys.executable, "experiments/run_pretrain.py",
        "--dataset", dataset, "--algo", "simclr",
        "--epochs", str(epochs), "--aug_preset", preset,
        "--save_every", str(epochs),
        "--out_dir", out_dir,
    ]
    log.info(f"启动预训练 {preset}: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True)

def run_fewshot(dataset, ckpt, n_way, k_shot, q_query, episodes, seeds):
    cmd = [
        sys.executable, "experiments/run_fewshot.py",
        "--dataset", dataset, "--encoder_ckpt", ckpt,
        "--algo", "protonet",
        "--n_way", str(n_way), "--k_shot", str(k_shot),
        "--q_query", str(q_query),
        "--num_episodes", str(episodes),
        "--num_seeds", str(seeds),
    ]
    log.info(f"启动少样本评估: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cic", "ustc", "iscx"], default="cic")
    p.add_argument("--epochs", type=int, default=20,
                   help="消融实验用较小 epochs 即可")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--q_query", type=int, default=15)
    p.add_argument("--num_episodes", type=int, default=300)
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--presets", nargs="+", default=PRESETS)
    args = p.parse_args()

    global log
    log = TqdmLogger(os.path.join("output/logs",
                                  f"ablation_{args.dataset}.log"))
    log.info(f"消融实验  dataset={args.dataset}  presets={args.presets}")

    out_dir = os.path.join("output/checkpoints", "ablation")
    os.makedirs(out_dir, exist_ok=True)

    summary = {"presets": {}}
    for preset in args.presets:
        # ckpt 实际保存名：{dataset}_simclr_ep{epochs}_{preset}.pt
        ckpt = os.path.join(out_dir, f"{args.dataset}_simclr_ep{args.epochs}_{preset}.pt")

        if not os.path.exists(ckpt):
            r = run_pretrain(args.dataset, preset, args.epochs, out_dir)
            log.info(f"预训练 {preset} 退出码={r.returncode}")
            if r.returncode != 0:
                log.info(r.stderr[:1000])
                continue
            if not os.path.exists(ckpt):
                log.info(f"找不到 {preset} 预训练 ckpt: {ckpt}")
                continue

        log.info(f"使用 ckpt: {ckpt}")

        r = run_fewshot(args.dataset, ckpt, args.n_way, args.k_shot,
                        args.q_query, args.num_episodes, args.num_seeds)
        log.info(f"评估 {preset} 退出码={r.returncode}")
        if r.returncode != 0:
            log.info(r.stderr[:1000])
            continue

        # 读取结果 json
        json_path = os.path.join(
            "output/results",
            f"fewshot_{args.dataset}_protonet_"
            f"{args.n_way}w{args.k_shot}s.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                agg = d.get("aggregate", {})
                summary["presets"][preset] = {
                    "mean": agg.get("mean"),
                    "std": agg.get("std"),
                }

    summary["args"] = vars(args)
    out_path = os.path.join("output/results",
                            f"ablation_{args.dataset}.json")
    save_json(summary, out_path)
    log.info(f"消融结果已保存 {out_path}")

if __name__ == "__main__":
    main()
