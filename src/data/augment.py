"""
traffic_augment.py
==================
流量数据增强模块：用于自监督对比学习的两个视图生成。

设计原则：
- 语义保持：增强不能破坏流量类别本质
  （如把 HTTPS 变成看起来像 BitTorrent）
- 物理合理：模拟真实网络扰动
  - 包乱序 / 丢包
  - 突发块 jitter
  - 测量噪声
- 与 DBAF-Net 兼容：增强后 (P, B, S, H) shape 不变
"""
import numpy as np
import torch

# ===========================
#   增强原语
# ===========================
def augment_packet(pkt: torch.Tensor,
                   jitter_ratio: float = 0.05,
                   mask_ratio: float = 0.15) -> torch.Tensor:
    """包序列增强（适配 h5 实际格式 (B, 2, 200)）：
    - 随机丢包（jitter_ratio 比例的时间步置 0）
    - 包内 byte 子序列 mask（mask_ratio 比例）
    输入 shape: (2, T) 或 (B, 2, T)
    """
    was_2d = pkt.dim() == 2
    if was_2d:
        pkt = pkt.unsqueeze(0)

    B, C, T = pkt.shape
    aug = pkt.clone()

    # 1) 整步 mask（时间维度）
    drop_mask = (torch.rand(B, 1, T, device=pkt.device) < jitter_ratio)
    aug = aug.masked_fill(drop_mask, 0.0)

    # 2) 子序列 mask（保留 padding 位置=0 不变）
    pkt_active = (pkt.sum(dim=1, keepdim=True) > 0)  # (B, 1, T)
    byte_mask = (torch.rand(B, C, T, device=pkt.device) < mask_ratio) & pkt_active
    aug = aug.masked_fill(byte_mask, 0.0)

    return aug.squeeze(0) if was_2d else aug

def augment_burst(burst: torch.Tensor,
                  jitter: float = 0.2) -> torch.Tensor:
    """突发序列增强（适配 h5 实际格式 (B, 16, 4)）。

    输入 shape: (16, 4) 或 (B, 16, 4)
    输出: shape 不变，数值 ±jitter 比例扰动 + 偶发整段置 0
    """
    was_2d = burst.dim() == 2
    if was_2d:
        burst = burst.unsqueeze(0)

    B, T, C = burst.shape
    aug = burst.clone()

    noise = torch.randn_like(aug) * jitter
    aug = aug * (1.0 + noise)

    # 5% 概率把整段 burst 置 0
    drop = (torch.rand(B, T, 1, device=burst.device) < 0.05).float()
    aug = aug * (1.0 - drop)

    return aug.squeeze(0) if was_2d else aug

def augment_stat(stat: torch.Tensor,
                 noise_std: float = 0.1,
                 drop_ratio: float = 0.1) -> torch.Tensor:
    """流统计特征增强：高斯噪声 + 特征维度 drop。

    输入 shape: (S,) 或 (B, S)
    物理意义：模拟流量统计在不同采样窗口下的测量误差。
    """
    was_1d = stat.dim() == 1
    if was_1d:
        stat = stat.unsqueeze(0)

    aug = stat + torch.randn_like(stat) * noise_std
    drop = (torch.rand_like(aug) < drop_ratio).float()
    aug = aug * (1.0 - drop)

    return aug.squeeze(0) if was_1d else aug

def augment_hand(hand: torch.Tensor,
                 noise_std: float = 0.05,
                 mask_ratio: float = 0.1) -> torch.Tensor:
    """握手特征增强：噪声 + 维度 mask（JA3 / suite / ext）。

    输入 shape: (H,) 或 (B, H)
    """
    was_1d = hand.dim() == 1
    if was_1d:
        hand = hand.unsqueeze(0)

    aug = hand + torch.randn_like(hand) * noise_std
    drop = (torch.rand_like(aug) < mask_ratio).float()
    aug = aug * (1.0 - drop)

    return aug.squeeze(0) if was_1d else aug

# ===========================
#   强增强集合（论文表 X ablation）
# ===========================
class TrafficAugPair:
    """组合多种增强策略，生成两个语义保持的视图 (view1, view2)。

    使用示例：
        >>> aug = TrafficAugPair(seed=42)
        >>> v1_p, v1_b, v1_s, v1_h, v2_p, v2_b, v2_s, v2_h = aug(p, b, s, h)
    """

    def __init__(self, seed: int = 42,
                 pkt_jitter: float = 0.05,
                 pkt_mask: float = 0.15,
                 burst_jitter: float = 0.2,
                 stat_noise: float = 0.1,
                 stat_drop: float = 0.1,
                 hand_noise: float = 0.05,
                 hand_mask: float = 0.1):
        self.pkt_jitter = pkt_jitter
        self.pkt_mask = pkt_mask
        self.burst_jitter = burst_jitter
        self.stat_noise = stat_noise
        self.stat_drop = stat_drop
        self.hand_noise = hand_noise
        self.hand_mask = hand_mask
        self.rng = np.random.RandomState(seed)

    def __call__(self, pkt, burst, stat, hand):
        """输入 (pkt, burst, stat, hand) 单样本；返回 8 个增强张量。"""
        v1_p = augment_packet(pkt, self.pkt_jitter, self.pkt_mask)
        v1_b = augment_burst(burst, self.burst_jitter)
        v1_s = augment_stat(stat, self.stat_noise, self.stat_drop)
        v1_h = augment_hand(hand, self.hand_noise, self.hand_mask)

        v2_p = augment_packet(pkt, self.pkt_jitter, self.pkt_mask)
        v2_b = augment_burst(burst, self.burst_jitter)
        v2_s = augment_stat(stat, self.stat_noise, self.stat_drop)
        v2_h = augment_hand(hand, self.hand_noise, self.hand_mask)

        return v1_p, v1_b, v1_s, v1_h, v2_p, v2_b, v2_s, v2_h

# ===========================
#   增强强度策略（消融实验用）
# ===========================
AUG_PRESETS = {
    "weak": dict(pkt_jitter=0.02, pkt_mask=0.05, burst_jitter=0.1,
                 stat_noise=0.05, stat_drop=0.05,
                 hand_noise=0.02, hand_mask=0.05),
    "medium": dict(pkt_jitter=0.05, pkt_mask=0.15, burst_jitter=0.2,
                   stat_noise=0.10, stat_drop=0.10,
                   hand_noise=0.05, hand_mask=0.10),
    "strong": dict(pkt_jitter=0.10, pkt_mask=0.30, burst_jitter=0.3,
                   stat_noise=0.20, stat_drop=0.20,
                   hand_noise=0.10, hand_mask=0.20),
    "none": dict(pkt_jitter=0.0, pkt_mask=0.0, burst_jitter=0.0,
                 stat_noise=0.0, stat_drop=0.0,
                 hand_noise=0.0, hand_mask=0.0),
}

def make_aug(preset: str = "medium", seed: int = 42) -> TrafficAugPair:
    """根据消融预设快速构造增强器。"""
    cfg = AUG_PRESETS[preset]
    return TrafficAugPair(seed=seed, **cfg)
