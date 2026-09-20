"""
supervised_pretrain.py
======================
全监督预训练 encoder（baseline supervised），下游 ProtoNet 5-shot 评估。

P0-1 v0.9：补齐 supervised pretrain baseline，
回答"SSL vs Random"是否应再补一层"SSL vs Supervised"。
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.data.load_h5 import H5FlowDataset, time_aware_split_indices
from src.models.encoder import TrafficEncoder

def collate(batch):
    """把 (P, B, S, H, y) 列表堆成 batch 张量。"""
    p = torch.stack([b[0] for b in batch])
    br = torch.stack([b[1] for b in batch])
    st = torch.stack([b[2] for b in batch])
    hd = torch.stack([b[3] for b in batch])
    y = torch.stack([b[4] for b in batch])
    return p, br, st, hd, y

class SupervisedEncoder(nn.Module):
    """Encoder + 单层 Linear 分类头（端到端监督训练）。"""

    def __init__(self, encoder: TrafficEncoder, num_classes: int, embed_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, pkt, br, st, hd):
        feat = self.encoder(pkt, br, st, hd)
        return self.classifier(feat)

def train_one(dataset_tag, h5_path, epochs, device, batch_size, lr, seed,
              n_train_samples=None, n_test_samples=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx_tr, _, idx_te = time_aware_split_indices(h5_path, 0.7, 0.15)
    # 可选子采样（加速大文件训练）
    if n_train_samples and n_train_samples < len(idx_tr):
        rng = np.random.RandomState(seed)
        idx_tr = rng.choice(idx_tr, n_train_samples, replace=False)
    if n_test_samples and n_test_samples < len(idx_te):
        rng = np.random.RandomState(seed)
        idx_te = rng.choice(idx_te, n_test_samples, replace=False)
    ds_tr = H5FlowDataset(h5_path, indices=list(idx_tr), fit=True)
    ds_te = H5FlowDataset(h5_path, indices=list(idx_te), fit=True)

    n_classes = len(ds_tr.classes)
    print(f"[{dataset_tag}] train={len(ds_tr)} test={len(ds_te)} classes={n_classes}")
    train_dl = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate)

    enc = TrafficEncoder(embed_dim=128)
    model = SupervisedEncoder(enc, num_classes=n_classes, embed_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_test = 0.0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for batch in train_dl:
            pkt, br, st, hd, y_b = [t.to(device) for t in batch]
            opt.zero_grad()
            logits = model(pkt, br, st, hd)
            loss = crit(logits, y_b)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y_b.size(0)
            n += y_b.size(0)
        train_loss = loss_sum / n

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_dl:
                pkt, br, st, hd, y_b = [t.to(device) for t in batch]
                logits = model(pkt, br, st, hd)
                pred = logits.argmax(dim=-1)
                correct += (pred == y_b).sum().item()
                total += y_b.size(0)
        test_acc = correct / max(total, 1)
        dt = time.time() - t0
        print(f"  ep{ep+1}/{epochs}: train_loss={train_loss:.4f} test_acc={test_acc:.4f} ({dt:.1f}s)")
        best_test = max(best_test, test_acc)

    # 保存 encoder 权重（与 SSL ckpt 兼容：只保存 encoder_state）
    out_path = f"output/checkpoints/{dataset_tag}_supervised_ep{epochs}.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "encoder_state": model.encoder.state_dict(),
        "epoch": epochs,
        "args": {"lr": lr, "batch_size": batch_size, "seed": seed},
    }, out_path)
    print(f"  saved ckpt → {out_path}")
    return out_path, best_test

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cic", choices=["cic", "ustc", "iscx"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_train_samples", type=int, default=None,
                   help="子采样训练样本数（大文件加速；默认全量）")
    p.add_argument("--n_test_samples", type=int, default=None,
                   help="子采样测试样本数（大文件加速；默认全量）")
    args = p.parse_args()

    h5_path = f"output/{args.dataset}_full.h5"
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"未找到 {h5_path}; 先运行数据准备脚本")
    t0 = time.time()
    ckpt_path, best_test = train_one(
        args.dataset, h5_path, args.epochs, args.device,
        args.batch_size, args.lr, args.seed,
        args.n_train_samples, args.n_test_samples,
    )
    out = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "best_test_acc": best_test,
        "ckpt_path": ckpt_path,
        "wall_time_sec": time.time() - t0,
    }
    out_json = f"output/results/supervised_pretrain_{args.dataset}.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {out_json}")

if __name__ == "__main__":
    main()