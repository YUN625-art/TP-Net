"""
encoder.py
==========
1D-CNN + MLP 多分支 encoder。

输入约定（与 src/data/load_h5.py + h5 存储保持一致）：
- pkt_seq: (B, 2, 200) 二值包头 + 长度（h5 存为 (N, 2, 200)）
- burst_seq: (B, 16, 4) h5 存为 (N, 16, 4)
- stat: (B, 32) 流统计
- hand: (B, 71) 握手特征 (6 JA3 + 32 suite + 32 ext + 1 miss)

输出：
- embedding: (B, embed_dim) 默认 128 维
- 或 proj: (B, proj_dim) 当 projection head 启用时（仅 SSL 用）

参数量控制：
- 默认 encoder 约 1.5M 参数（轻量）
- 比 DBAF-Net 少了 cross-attention，因此训练更快
"""
from __future__ import annotations
import torch
import torch.nn as nn

class PktEncoder(nn.Module):
    """包级序列 encoder：Conv1d(2→64→128→128) + global pool。

    原版带 BiLSTM，但 XPU LSTM 不支持 train 模式，
    改为 Conv1d + global pool（更简洁）。
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        # 输入: (B, 2, T=200)
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),  # → 100
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),  # → 50
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # → (B, 128, 1)
        )
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T=200) 已经是 conv1d 期望的 (B, C, L) 格式
        return self.conv(x).squeeze(-1)  # (B, 128)

class BurstEncoder(nn.Module):
    """突发级序列 encoder：Conv1d(4→64→128) + mean pool。"""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=16, C=4) → (B, C=4, T=16)
        x = x.transpose(1, 2).contiguous()
        return self.conv(x).squeeze(-1)  # (B, 128)

class StatMLP(nn.Module):
    """流统计 MLP：32 → 128。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
        )
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HandMLP(nn.Module):
    """握手特征 MLP：71 → 128。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(71, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
        )
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TrafficEncoder(nn.Module):
    """TP-Net 主干 encoder。

    4 分支融合：P (128) + B (128) + S (128) + H (128) = 512 → embed_dim。
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.pkt_enc = PktEncoder()
        self.burst_enc = BurstEncoder()
        self.stat_mlp = StatMLP()
        self.hand_mlp = HandMLP()

        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, pkt, burst, stat, hand) -> torch.Tensor:
        """输入 4 分支 → embedding (B, embed_dim)。"""
        fp = self.pkt_enc(pkt)
        fb = self.burst_enc(burst)
        fs = self.stat_mlp(stat)
        fh = self.hand_mlp(hand)
        feat = torch.cat([fp, fb, fs, fh], dim=-1)
        return self.fusion(feat)

class SSLProjector(nn.Module):
    """SimCLR/MoCo 用的 projection head：embed_dim → proj_dim → proj_dim。"""

    def __init__(self, embed_dim: int = 128, proj_dim: int = 128,
                 hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ===========================
#   参数量统计
# ===========================
def count_params(module: nn.Module) -> int:
    """返回可训练参数量（仅用于日志）。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 烟雾测试
    enc = TrafficEncoder(embed_dim=128)
    print(f"TrafficEncoder 参数量: {count_params(enc):,}")

    # 模拟输入
    B = 4
    pkt = torch.randn(B, 2, 200)
    burst = torch.randn(B, 16, 4)
    stat = torch.randn(B, 32)
    hand = torch.randn(B, 71)

    emb = enc(pkt, burst, stat, hand)
    print(f"Embedding shape: {emb.shape}")  # (4, 128)

    proj = SSLProjector(128, 128)
    p = proj(emb)
    print(f"Projection shape: {p.shape}")  # (4, 128)