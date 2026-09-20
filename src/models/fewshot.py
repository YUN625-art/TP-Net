"""
fewshot.py
==========
少样本分类模型。

设计要点：
1. encoder 可来自自监督 (ssl.py) 或随机初始化（baseline）
2. ProtoNet/MatchingNet 不需要元训练 → 直接用 encoder + KNN/attn 即可
3. LinearProbing 评估 encoder 表示质量（论文表 X ablation）
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from .encoder import TrafficEncoder

def _encode_4(model: TrafficEncoder, *xs):
    """对 4 分支输入做 encoder 前向，兼容 DataLoader 一次返回 (B, T, 2) 等。"""
    return model(*xs)

# ===========================
#   ProtoNet
# ===========================
class ProtoNet(nn.Module):
    """原型网络：每个 support 类别取平均特征作为原型，query 取最近原型。

    元训练：无需（端到端只是 embedding 空间距离）。
    评估：N-way K-shot episode。
    """

    def __init__(self, encoder: TrafficEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, support_x: Tuple, query_x: Tuple,
                support_y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support_x: tuple of 4 分支 support 张量（每个 (N*K, ...)）
            query_x: tuple of 4 分支 query 张量（每个 (N*Q, ...)）
            support_y: (N*K,) episode 内 [0, N)

        Returns:
            logits: (N*Q, N)
        """
        # encode
        s_feat = _encode_4(self.encoder, *support_x)
        q_feat = _encode_4(self.encoder, *query_x)

        n_way = support_y.max().item() + 1
        # 原型 = 每个类 support 的平均 embedding
        protos = []
        for c in range(n_way):
            protos.append(s_feat[support_y == c].mean(dim=0))
        protos = torch.stack(protos)  # (N, D)

        # 距离 = - cos 相似度 → logits
        q_norm = F.normalize(q_feat, dim=-1)
        p_norm = F.normalize(protos, dim=-1)
        logits = q_norm @ p_norm.t()  # (N*Q, N)
        return logits

    @torch.no_grad()
    def forward_with_confidence(self, support_x: Tuple, query_x: Tuple,
                                support_y: torch.Tensor):
        """带置信度的前向：返回 (logits, max_cos_sim) 用于开集拒绝。

        confidence = max_cos_sim(query, prototypes)
        - 高 confidence → 接近某个 prototype → likely known
        - 低 confidence → 远离所有 prototype → likely unknown

        与 softmax 概率的区别：
        - softmax 概率会受"次相似类"干扰（如 query 同时接近两类）
        - cos similarity to closest prototype 更直接反映"是否属于已知类"
        """
        s_feat = _encode_4(self.encoder, *support_x)
        q_feat = _encode_4(self.encoder, *query_x)

        n_way = support_y.max().item() + 1
        protos = []
        for c in range(n_way):
            protos.append(s_feat[support_y == c].mean(dim=0))
        protos = torch.stack(protos)

        q_norm = F.normalize(q_feat, dim=-1)
        p_norm = F.normalize(protos, dim=-1)
        logits = q_norm @ p_norm.t()  # (Q, N)
        confidence, _ = logits.max(dim=-1)  # (Q,) max cos sim
        return logits, confidence

    @torch.no_grad()
    def forward_with_softmax_conf(self, support_x: Tuple, query_x: Tuple,
                                   support_y: torch.Tensor, temperature: float = 1.0):
        """带 softmax 概率置信度（可选 temperature scaling 校准）。"""
        logits, _ = self.forward_with_confidence(support_x, query_x, support_y)
        prob = F.softmax(logits / temperature, dim=-1)
        return logits, prob.max(dim=-1).values

# ===========================
#   MatchingNet
# ===========================
class MatchingNet(nn.Module):
    """简化版 Matching Networks：纯 cosine attention，无 LSTM（XPU 兼容性）。

    与原版 Vinyals et al. 2016 的差异：去掉 BiLSTM contextualization，
    直接用 encoder embedding 做 attention + 标签加权。
    论文中标注为 "MatchingNet (simplified)"。
    """

    def __init__(self, encoder: TrafficEncoder, embed_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        # 不再使用 LSTM（XPU 推理不兼容）

    def _attn_logits(self, q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """cosine 注意力 (Vinyals style)。"""
        q_n = F.normalize(q, dim=-1)
        c_n = F.normalize(c, dim=-1)
        return q_n @ c_n.t()  # (Q, S)

    def forward(self, support_x: Tuple, query_x: Tuple,
                support_y: torch.Tensor) -> torch.Tensor:
        s_feat = self.encoder(*support_x)  # (N*K, D)
        q_feat = self.encoder(*query_x)  # (N*Q, D)

        # attention: (Q, S)
        attn_logits = self._attn_logits(q_feat, s_feat)  # cos 相似度
        attn = F.softmax(attn_logits, dim=-1)  # (Q, S)

        # 把 support 标签 one-hot 加权求和 → (Q, N)
        n_way = support_y.max().item() + 1
        s_y_onehot = F.one_hot(support_y, num_classes=n_way).float()  # (S, N)
        logits = attn @ s_y_onehot  # (Q, N)
        return logits

# ===========================
#   Linear Probing
# ===========================
class LinearProbing(nn.Module):
    """线性探测：encoder 固定，仅训练单层 Linear。

    用于评估 SSL encoder 的表示质量。
    """

    def __init__(self, encoder: TrafficEncoder, num_classes: int,
                 embed_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        # 冻结 encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(embed_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        # encoder 永远保持 eval（batch norm 用 running stats）
        self.encoder.eval()
        return self

    def forward(self, pkt, burst, stat, hand) -> torch.Tensor:
        with torch.no_grad():
            feat = self.encoder(pkt, burst, stat, hand)
        return self.classifier(feat)

# ===========================
#   Episode 评估工具
# ===========================
@torch.no_grad()
def eval_episode(model, episode: Dict, device: str = "cpu") -> Tuple[float, int]:
    """评估一个 episode，返回 (accuracy, n_query)。

    Args:
        model: ProtoNet / MatchingNet
        episode: EpisodeSampler 输出
        device: 'cpu' / 'cuda' / 'xpu'
    """
    model.eval()
    ds = model.encoder  # 任意分支拿到 dataset
    # 假设 encoder 模块能直接接受 tuple
    s_idx = episode["support_indices"]
    q_idx = episode["query_indices"]
    s_y = episode["support_labels"].to(device)
    q_y = episode["query_labels"].to(device)

    # 从 dataset 取出 raw tensors（外部 dataset 引用）
    dataset = model._dataset if hasattr(model, "_dataset") else None
    if dataset is None:
        raise RuntimeError("model._dataset 未设置；请用 attach_dataset 注入。")

    s_pkt, s_burst, s_stat, s_hand = _gather(dataset, s_idx, device)
    q_pkt, q_burst, q_stat, q_hand = _gather(dataset, q_idx, device)

    logits = model((s_pkt, s_burst, s_stat, s_hand),
                   (q_pkt, q_burst, q_stat, q_hand),
                   s_y)
    pred = logits.argmax(dim=-1)
    acc = (pred == q_y).float().mean().item()
    return acc, q_y.size(0)

def _gather(dataset, indices, device):
    """批量取 4 分支张量。"""
    p, br, st, hd = [], [], [], []
    for i in indices:
        a, b, c, d, _ = dataset[int(i)]
        p.append(a); br.append(b); st.append(c); hd.append(d)
    p = torch.stack(p).to(device)
    br = torch.stack(br).to(device)
    st = torch.stack(st).to(device)
    hd = torch.stack(hd).to(device)
    return p, br, st, hd

def attach_dataset(model, dataset):
    """把 dataset 引用注入到 model，方便 eval_episode 取数据。"""
    model._dataset = dataset
    return model