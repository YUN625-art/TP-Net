"""
eval.py
=======
评估指标 + 可视化。
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.manifold import TSNE
import torch
from typing import List, Dict, Optional

def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                   classes: List[str], out_path: str,
                   title: str = "Confusion Matrix",
                   normalize: bool = True):
    """画混淆矩阵热力图。"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.5),
                                    max(6, len(classes) * 0.5)))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

@torch.no_grad()
def tsne_visualize(encoder, dataset, indices: List[int],
                   out_path: str, perplexity: int = 30,
                   max_n: int = 2000, device: str = "cpu"):
    """对 encoder 嵌入做 t-SNE 并画散点图。"""
    encoder.eval()
    if len(indices) > max_n:
        indices = np.random.RandomState(42).choice(indices, max_n,
                                                    replace=False).tolist()

    feats, labels = [], []
    for i in indices:
        x = dataset[int(i)]
        pkt, br, st, hd = x[0].unsqueeze(0).to(device), \
                          x[1].unsqueeze(0).to(device), \
                          x[2].unsqueeze(0).to(device), \
                          x[3].unsqueeze(0).to(device)
        feat = encoder(pkt, br, st, hd).cpu().numpy()[0]
        feats.append(feat)
        labels.append(x[4].item())
    feats = np.array(feats)
    labels = np.array(labels)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(feats) - 1),
                init="pca", random_state=42)
    emb = tsne.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(10, 8))
    classes = dataset.classes
    for c in np.unique(labels):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6,
                   label=classes[c] if c < len(classes) else f"class_{c}")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    # 顶部 title 移除，由 LaTeX caption 提供图题
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def episode_aggregate_metrics(episode_results: List[Dict]) -> Dict:
    """聚合多个 episode 的混淆矩阵 / F1。"""
    y_true, y_pred = [], []
    for r in episode_results:
        if "y_true" in r and "y_pred" in r:
            y_true.extend(r["y_true"])
            y_pred.extend(r["y_pred"])
    if not y_true:
        return {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_episodes": len(episode_results),
    }
