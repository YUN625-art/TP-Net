"""
ssl.py
======
自监督对比学习：SimCLR / MoCo / BYOL / SimSiam / MAE + NT-Xent 损失。

核心思想：
- 同一样本的两个增强视图应映射到特征空间相近位置
- 不同样本的视图应远离
- 不依赖任何标签 → 适合大量未标记加密流量
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .encoder import TrafficEncoder, SSLProjector

# ===========================
#   NT-Xent 损失
# ===========================
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.1) -> torch.Tensor:
    """标准 NT-Xent (SimCLR) 损失。

    Args:
        z1, z2: (B, D) 同一 batch 的两个视图的投影
        temperature: 温度参数 τ

    Returns:
        scalar loss
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # 拼接 → (2B, D)
    z = torch.cat([z1, z2], dim=0)

    # 相似度矩阵 → (2B, 2B)
    sim = z @ z.t() / temperature

    # mask 自相似
    mask_self = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask_self, -1e9)

    # 正样本索引：i 与 i+B 互为正对
    pos_idx = torch.arange(2 * B, device=z.device)
    pos_idx = (pos_idx + B) % (2 * B)

    loss = F.cross_entropy(sim, pos_idx)
    return loss

# ===========================
#   SimCLR
# ===========================
class SimCLR(nn.Module):
    """SimCLR 模型：encoder + projection head。"""

    def __init__(self, encoder: Optional[TrafficEncoder] = None,
                 embed_dim: int = 128, proj_dim: int = 128,
                 temperature: float = 0.1):
        super().__init__()
        self.encoder = encoder or TrafficEncoder(embed_dim=embed_dim)
        self.projector = SSLProjector(embed_dim, proj_dim)
        self.temperature = temperature
        self.embed_dim = embed_dim

    def forward(self, view1, view2):
        """view1, view2: tuple(pkt, burst, stat, hand)"""
        e1 = self.encoder(*view1)
        e2 = self.encoder(*view2)
        z1 = self.projector(e1)
        z2 = self.projector(e2)
        loss = nt_xent_loss(z1, z2, self.temperature)
        return loss, e1, e2

    def get_encoder(self) -> TrafficEncoder:
        """用于下游：仅返回主干。"""
        return self.encoder

# ===========================
#   MoCo
# ===========================
class MoCo(nn.Module):
    """MoCo v2：动量 encoder + 大队列。

    核心点：
    - query encoder 用梯度更新
    - key encoder 用 EMA 更新（不参与反向）
    - 队列维护 K 个负样本（默认 4096）
    """

    def __init__(self, encoder: Optional[TrafficEncoder] = None,
                 embed_dim: int = 128, proj_dim: int = 128,
                 K: int = 4096, m: float = 0.999,
                 temperature: float = 0.1):
        super().__init__()
        self.K = K
        self.m = m
        self.temperature = temperature

        # query 分支
        self.encoder_q = encoder or TrafficEncoder(embed_dim=embed_dim)
        self.projector_q = SSLProjector(embed_dim, proj_dim)

        # key 分支（动量更新）
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.projector_k = copy.deepcopy(self.projector_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.projector_k.parameters():
            p.requires_grad = False

        # 队列: (proj_dim, K)
        self.register_buffer("queue", F.normalize(
            torch.randn(proj_dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.embed_dim = embed_dim

    @torch.no_grad()
    def _momentum_update(self):
        for p_q, p_k in zip(self.encoder_q.parameters(),
                            self.encoder_k.parameters()):
            p_k.data.mul_(self.m).add_(p_q.data, alpha=1 - self.m)
        for p_q, p_k in zip(self.projector_q.parameters(),
                            self.projector_k.parameters()):
            p_k.data.mul_(self.m).add_(p_q.data, alpha=1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """把新 batch 的 keys 加入队列，pop 老的。"""
        keys = keys.detach()
        ptr = int(self.queue_ptr)
        n = keys.size(0)
        if ptr + n <= self.K:
            self.queue[:, ptr:ptr + n] = keys.t()
        else:
            tail = self.K - ptr
            self.queue[:, ptr:] = keys[:tail].t()
            self.queue[:, :n - tail] = keys[tail:].t()
        self.queue_ptr[0] = (ptr + n) % self.K

    def forward(self, view_q, view_k):
        """q 分支走梯度，k 分支用 momentum encoder。"""
        # query
        eq = self.encoder_q(*view_q)
        zq = F.normalize(self.projector_q(eq), dim=-1)  # (B, D)

        # key（无梯度）
        with torch.no_grad():
            self._momentum_update()
            ek = self.encoder_k(*view_k)
            zk = F.normalize(self.projector_k(ek), dim=-1)  # (B, D)

        # 正样本 logits: (B, 1)
        l_pos = (zq * zk).sum(dim=-1, keepdim=True)
        # 负样本 logits: (B, K)
        l_neg = torch.einsum("bd,dk->bk", zq, self.queue.clone().detach())

        logits = torch.cat([l_pos, l_neg], dim=-1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long,
                             device=logits.device)

        loss = F.cross_entropy(logits, labels)

        # 更新队列
        self._dequeue_and_enqueue(zk)
        return loss, eq, ek

    def get_encoder(self) -> TrafficEncoder:
        return self.encoder_q

# ===========================
#   BYOL
# ===========================
class BYOLPredictor(nn.Module):
    """BYOL predictor: 2-layer MLP with BN."""

    def __init__(self, dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class BYOL(nn.Module):
    """BYOL: online + target networks with EMA, predictor on online branch.

    不需要负样本，依赖 EMA target + predictor 防止坍缩。
    """

    def __init__(self, encoder: Optional[TrafficEncoder] = None,
                 embed_dim: int = 128, proj_dim: int = 128,
                 hidden_dim: int = 1024,
                 predictor_hidden: int = 1024,
                 m: float = 0.99):
        super().__init__()
        # online 分支
        self.online_encoder = encoder or TrafficEncoder(embed_dim=embed_dim)
        self.online_projector = SSLProjector(embed_dim, proj_dim,
                                             hidden_dim=hidden_dim)
        self.predictor = BYOLPredictor(proj_dim, hidden_dim=predictor_hidden)

        # target 分支（EMA，不参与梯度）
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.m = m
        self.embed_dim = embed_dim

    @torch.no_grad()
    def _momentum_update(self):
        for p_q, p_k in zip(self.online_encoder.parameters(),
                            self.target_encoder.parameters()):
            p_k.data.mul_(self.m).add_(p_q.data, alpha=1 - self.m)
        for p_q, p_k in zip(self.online_projector.parameters(),
                            self.target_projector.parameters()):
            p_k.data.mul_(self.m).add_(p_q.data, alpha=1 - self.m)

    def forward(self, view1, view2):
        # online: view1
        e1_on = self.online_encoder(*view1)
        z1_on = self.online_projector(e1_on)
        p1 = self.predictor(z1_on)

        # online: view2
        e2_on = self.online_encoder(*view2)
        z2_on = self.online_projector(e2_on)
        p2 = self.predictor(z2_on)

        # target: 无梯度 + EMA 更新
        with torch.no_grad():
            self._momentum_update()
            e1_tg = self.target_encoder(*view1)
            z1_tg = self.target_projector(e1_tg)
            e2_tg = self.target_encoder(*view2)
            z2_tg = self.target_projector(e2_tg)

        loss = byol_loss(p1, z2_tg) + byol_loss(p2, z1_tg)
        return loss, e1_on, e2_on

    def get_encoder(self) -> TrafficEncoder:
        return self.online_encoder

def byol_loss(p: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """BYOL symmetric loss: 2 - 2 * cos(p, target)."""
    p = F.normalize(p, dim=-1)
    z_target = F.normalize(z_target, dim=-1)
    return (2 - 2 * (p * z_target).sum(dim=-1)).mean()

# ===========================
#   SimSiam
# ===========================
class SimSiamPredictor(nn.Module):
    """SimSiam predictor: 2-layer MLP with BN, no bias on final layer."""

    def __init__(self, dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SimSiam(nn.Module):
    """SimSiam: 不需要负样本，也不需要 EMA，只用 stop-gradient。

    Args:
        encoder: 主干 encoder
        embed_dim: encoder 输出维度
        proj_dim: projector 输出维度
        hidden_dim: projector/predictor 中间层维度
    """

    def __init__(self, encoder: Optional[TrafficEncoder] = None,
                 embed_dim: int = 128, proj_dim: int = 128,
                 hidden_dim: int = 1024,
                 predictor_hidden: int = 1024):
        super().__init__()
        self.encoder = encoder or TrafficEncoder(embed_dim=embed_dim)
        self.projector = SSLProjector(embed_dim, proj_dim, hidden_dim=hidden_dim)
        self.predictor = SimSiamPredictor(proj_dim, hidden_dim=predictor_hidden)
        self.embed_dim = embed_dim

    def forward(self, view1, view2):
        e1 = self.encoder(*view1)
        e2 = self.encoder(*view2)
        z1 = self.projector(e1)
        z2 = self.projector(e2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        # stop-gradient: z2/z1 不回传
        loss = simsiam_loss(p1, z2) + simsiam_loss(p2, z1)
        return loss, e1, e2

    def get_encoder(self) -> TrafficEncoder:
        return self.encoder

def simsiam_loss(p: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """SimSiam loss: 负余弦相似度（z_target 已 detach）."""
    p = F.normalize(p, dim=-1)
    z_target = F.normalize(z_target, dim=-1).detach()
    return -(p * z_target).sum(dim=-1).mean()

# ===========================
#   MAE (简化版：embedding 维度 mask-reconstruct)
# ===========================
class MAEDecoder(nn.Module):
    """简化版 MAE decoder: 从被 mask 的 embedding 重建完整 embedding."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MAE(nn.Module):
    """简化版 MAE：对 encoder 输出 embedding 做 random mask-reconstruct。

    与 SimCLR/BYOL/SimSiam 共享 view1/view2 双视图接口（双视图各自
    独立做 mask-reconstruct，loss 取平均），保持与 pretrain.py 兼容。

    Args:
        encoder: 主干
        embed_dim: encoder 输出维度
        mask_ratio: embedding 维度被 mask 的比例
    """

    def __init__(self, encoder: Optional[TrafficEncoder] = None,
                 embed_dim: int = 128, mask_ratio: float = 0.5):
        super().__init__()
        self.encoder = encoder or TrafficEncoder(embed_dim=embed_dim)
        self.decoder = MAEDecoder(embed_dim)
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

    def _mask(self, e: torch.Tensor):
        """按 embedding dim 维度随机 mask 部分维度。"""
        B, D = e.shape
        mask = torch.rand(B, D, device=e.device) < self.mask_ratio
        e_masked = e.clone()
        e_masked[mask] = 0.0
        return e_masked, mask

    def forward(self, view1, view2):
        e1 = self.encoder(*view1)
        e2 = self.encoder(*view2)

        e1_m, mask1 = self._mask(e1)
        e2_m, mask2 = self._mask(e2)
        recon1 = self.decoder(e1_m)
        recon2 = self.decoder(e2_m)

        # 仅在 masked 位置计算 MSE
        if mask1.any():
            loss1 = F.mse_loss(recon1[mask1], e1[mask1])
        else:
            loss1 = torch.tensor(0.0, device=e1.device)
        if mask2.any():
            loss2 = F.mse_loss(recon2[mask2], e2[mask2])
        else:
            loss2 = torch.tensor(0.0, device=e1.device)
        loss = (loss1 + loss2) / 2.0
        return loss, e1, e2

    def get_encoder(self) -> TrafficEncoder:
        return self.encoder
