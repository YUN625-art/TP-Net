"""
pretrain.py
===========
自监督预训练入口。

用法：
    python experiments/run_pretrain.py --dataset cic --algo simclr --epochs 50
"""
from __future__ import annotations
import argparse
import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.load_h5 import H5FlowDataset
from src.data.augment import make_aug
from src.models.encoder import TrafficEncoder
from src.models.ssl import SimCLR, MoCo, BYOL, SimSiam, MAE
from src.utils import get_device, set_seed, TqdmLogger, save_json

H5_PATHS = {
    "cic": "output/cic_full.h5",
    "ustc": "output/ustc_full.h5",
    "iscx": "output/iscx_full.h5",
    "dohbrw": "output/dohbrw_full.h5",
}

def build_dataset(h5_path: str, max_samples: int = None):
    """把 h5 全部作为无标签数据。

    注：这里不用 train/val/test 划分，因为预训练是无监督的。
    """
    # 简化：用全部 h5 作为 dataset
    with __import__("h5py").File(h5_path, "r") as h:
        n = h["stat"].shape[0]
    if max_samples:
        # 分层抽样，避免 max_samples 抽光小类
        from src.data.load_h5 import _stratified_subsample
        with __import__("h5py").File(h5_path, "r") as h:
            labels = h["label"].asstr()[:]
        indices = _stratified_subsample(list(range(n)), labels, max_samples,
                                        seed=42)
    else:
        indices = list(range(n))
    return H5FlowDataset(h5_path, indices, fit=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(H5_PATHS), required=True)
    p.add_argument("--algo", choices=["simclr", "moco", "byol", "simsiam", "mae"], default="simclr")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--aug_preset", default="medium")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--out_dir", default="output/checkpoints")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    log = TqdmLogger(os.path.join("output/logs",
                                  f"pretrain_{args.dataset}_{args.algo}.log"))
    log.info(f"设备={device}  algo={args.algo}  epochs={args.epochs}")

    # 数据
    h5_path = H5_PATHS[args.dataset]
    ds = build_dataset(h5_path, args.max_samples)
    log.info(f"预训练样本数={len(ds)}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True)

    # 模型
    encoder = TrafficEncoder(embed_dim=args.embed_dim).to(device)
    if args.algo == "simclr":
        model = SimCLR(encoder, embed_dim=args.embed_dim,
                       proj_dim=args.embed_dim,
                       temperature=args.temperature).to(device)
    elif args.algo == "moco":
        model = MoCo(encoder, embed_dim=args.embed_dim,
                     proj_dim=args.embed_dim,
                     temperature=args.temperature).to(device)
    elif args.algo == "byol":
        model = BYOL(encoder, embed_dim=args.embed_dim,
                     proj_dim=args.embed_dim).to(device)
    elif args.algo == "simsiam":
        model = SimSiam(encoder, embed_dim=args.embed_dim,
                        proj_dim=args.embed_dim).to(device)
    elif args.algo == "mae":
        model = MAE(encoder, embed_dim=args.embed_dim).to(device)

    aug = make_aug(args.aug_preset, seed=args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu",
                                  enabled=(not args.no_amp and device.type == "cuda"))

    losses = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        ep_losses = []
        for batch in dl:
            pkt, burst, stat, hand, _y = batch
            pkt = pkt.to(device); burst = burst.to(device)
            stat = stat.to(device); hand = hand.to(device)

            v1p, v1b, v1s, v1h, v2p, v2b, v2s, v2h = aug(pkt, burst, stat, hand)
            view1 = (v1p, v1b, v1s, v1h)
            view2 = (v2p, v2b, v2s, v2h)

            opt.zero_grad()
            if scaler.is_enabled():
                with torch.amp.autocast("cuda"):
                    loss, _, _ = model(view1, view2)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss, _, _ = model(view1, view2)
                loss.backward()
                opt.step()

            ep_losses.append(loss.item())

        scheduler.step()
        avg = float(np.mean(ep_losses))
        losses.append(avg)
        log.info(f"epoch {ep:03d}/{args.epochs}  loss={avg:.4f}  "
                 f"lr={opt.param_groups[0]['lr']:.2e}")

        if ep % args.save_every == 0 or ep == args.epochs:
            os.makedirs(args.out_dir, exist_ok=True)
            # ckpt 命名包含 aug_preset（消融实验区分）
            preset_suffix = f"_{args.aug_preset}" if args.aug_preset != "medium" else ""
            ckpt = os.path.join(args.out_dir,
                                f"{args.dataset}_{args.algo}_ep{ep}{preset_suffix}.pt")
            torch.save({
                "encoder_state": model.get_encoder().state_dict(),
                "epoch": ep,
                "args": vars(args),
            }, ckpt)
            log.info(f"saved {ckpt}")

    elapsed = time.time() - t0
    log.info(f"训练完成 耗时={elapsed/60:.1f} min")

    # 保存 loss 曲线
    save_json({"losses": losses, "elapsed_sec": elapsed, "args": vars(args)},
              os.path.join("output/results",
                           f"pretrain_{args.dataset}_{args.algo}_loss.json"))

if __name__ == "__main__":
    main()
