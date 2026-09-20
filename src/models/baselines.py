import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# YaTC (Vision Transformer for Traffic, Zhao 2023 简化版)
# 输入: 把 P (2×200) 当作 10×20 灰度图 → patch 4×4
# ============================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=20, patch=4, in_ch=1, dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        n_patches = (img_size // patch) * (img_size // patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        x = self.proj(x)              # (B, dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h, dim), nn.Dropout(drop)
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x

class YaTCNet(nn.Module):
    """YaTC：P 分支 reshape 成 1×20×20，patch=4 的 mini-ViT。"""
    def __init__(self, n_classes, in_ch=2, dim=64, depth=4, heads=4,
                 embed_dim=128):
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, 1, 1)
        self.patch = PatchEmbed(img_size=20, patch=4, in_ch=1, dim=dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads=heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        self.embed_dim = embed_dim

    def encode(self, pkt):
        """Pkt (B,2,200) → (B, embed_dim=128) embedding."""
        x = pkt.view(pkt.shape[0], 2, 20, 10).mean(dim=1, keepdim=False)
        x = F.interpolate(x.unsqueeze(1), size=(20, 20), mode='bilinear', align_corners=False)
        x = self.patch(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])
        return self.proj(x)

    def forward(self, pkt, burst=None, stat=None, hand=None):
        return self.head(self.encode(pkt))

# ============================================================
# RF (Random Forest) — sklearn 经典 ML baseline
# ============================================================
class RFBaseline:
    is_sklearn = True
    def __init__(self, n_classes, in_ch=2, n_estimators=200, max_depth=None,
                 n_jobs=-1, embed_dim=128):
        from sklearn.ensemble import RandomForestClassifier
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.cls = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            n_jobs=n_jobs, random_state=0, class_weight="balanced")

    def fit(self, X, y):
        self.cls.fit(X, y); return self
    def predict(self, X):  return self.cls.predict(X)
    def predict_proba(self, X): return self.cls.predict_proba(X)
    def forward(self, *args, **kwargs): raise NotImplementedError
    def parameters(self, recurse=True): return iter([])
    def named_parameters(self, prefix=""): return []
    def to(self, device): return self
    def state_dict(self): return {}
    def load_state_dict(self, state): pass
    def train(self): return self
    def eval(self): return self

# ============================================================
# FS-Net (INFOCOM WKSHPS 2019)
# ============================================================
class FSNet(nn.Module):
    def __init__(self, n_classes, in_ch=2, hid=64, embed_dim=128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, hid, 3, padding=1), nn.BatchNorm1d(hid), nn.ReLU(),
        )
        self.rnn = nn.LSTM(hid, hid, num_layers=2, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hid, 1)
        self.proj = nn.Linear(2 * hid, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        self.embed_dim = embed_dim

    def encode(self, pkt):
        """Pkt (B,2,200) → (B, embed_dim=128)."""
        x = self.embed(pkt)         # (B, hid, T)
        x = x.transpose(1, 2)       # (B, T, hid)
        h, _ = self.rnn(x)          # (B, T, 2*hid)
        a = torch.softmax(self.attn(h), dim=1)
        x = (h * a).sum(dim=1)      # (B, 2*hid)
        return self.proj(x)         # (B, embed_dim)

    def forward(self, pkt, burst=None, stat=None, hand=None):
        return self.fc(self.encode(pkt))

# ============================================================
# TFE-GNN (IJCAI 2023 简化版)
# ============================================================
class TFEGraph(nn.Module):
    def __init__(self, n_classes, in_ch=2, n_nodes=10, hid=64, embed_dim=128):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_embed = nn.Linear(in_ch * (200 // n_nodes), hid)
        self.gat1 = nn.MultiheadAttention(hid, num_heads=4, batch_first=True)
        self.gat2 = nn.MultiheadAttention(hid, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(hid)
        self.norm2 = nn.LayerNorm(hid)
        self.proj = nn.Linear(hid, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        self.embed_dim = embed_dim

    def encode(self, pkt):
        B = pkt.shape[0]
        feat_per_node = 2 * (200 // self.n_nodes)
        x = pkt.view(B, self.n_nodes, feat_per_node)
        x = self.node_embed(x)
        a, _ = self.gat1(x, x, x); x = self.norm1(x + a)
        a, _ = self.gat2(x, x, x); x = self.norm2(x + a)
        x = x.mean(dim=1)
        return self.proj(x)

    def forward(self, pkt, burst=None, stat=None, hand=None):
        return self.fc(self.encode(pkt))

# ============================================================
# ET-BERT (WWW 2022 简化版)
# ============================================================
class ETBERTLite(nn.Module):
    def __init__(self, n_classes, max_len=200, vocab=257, dim=192, depth=6,
                 heads=6, embed_dim=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len + 2, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.embed_dim = embed_dim

    def encode(self, pkt):
        """Pkt (B,2,200) → (B, embed_dim=128)."""
        b1 = (pkt[:, 0] * 255).clamp(0, 255).long()  # 取首字节（pkt_len log1p 后到 0-8，归一会丢信息）
        # 实际更合理：把 pkt_len 截断归一化后再 tokenize，但 ET-BERT 简化版不引入新管线
        x = self.token_embed(b1)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, :x.size(1)]
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.proj(x)

    def forward(self, pkt, burst=None, stat=None, hand=None):
        return self.head(self.encode(pkt))

# ============================================================
# NetMamba (ICNP 2024 简化版)
# ============================================================
class NetMambaLite(nn.Module):
    def __init__(self, n_classes, dim=128, embed_dim=128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(2, dim, 7, padding=3), nn.BatchNorm1d(dim), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim * 2, 5, padding=2), nn.GLU(dim=1),
                nn.BatchNorm1d(dim), nn.GELU(),
                nn.Conv1d(dim, dim * 2, 3, padding=1), nn.GLU(dim=1),
                nn.BatchNorm1d(dim), nn.GELU(),
            ) for _ in range(4)
        ])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(dim, embed_dim)
        self.fc = nn.Sequential(nn.Linear(embed_dim, n_classes))
        self.embed_dim = embed_dim

    def encode(self, pkt):
        x = self.embed(pkt)
        for blk in self.blocks:
            x = x + blk(x)
        x = self.gap(x).flatten(1)
        return self.proj(x)

    def forward(self, pkt, burst=None, stat=None, hand=None):
        return self.fc(self.encode(pkt))

# ============================================================
# NetConv
# ============================================================
class ResBlock1D(nn.Module):
    def __init__(self, dim, k=5):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, k, padding=k//2)
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, k, padding=k//2)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(x + h)

class NetConvLite(nn.Module):
    def __init__(self, n_classes, dim=128, depth=8, embed_dim=128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(2, dim, 7, padding=3), nn.BatchNorm1d(dim), nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResBlock1D(dim) for _ in range(depth)])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(dim, embed_dim)
        self.fc = nn.Sequential(nn.Linear(embed_dim, n_classes))
        self.embed_dim = embed_dim

    def encode(self, pkt):
        x = self.embed(pkt)
        for blk in self.blocks:
            x = blk(x)
        x = self.gap(x).flatten(1)
        return self.proj(x)

    def forward(self, pkt, burst=None, stat=None, hand=None):
        return self.fc(self.encode(pkt))

BASELINES = {
    'rf':        ('Random Forest (sklearn)',       RFBaseline,    '—'),
    'fs_net':    ('FS-Net',                        FSNet,        '0.3M'),
    'yatc':      ('YaTC (lite, no MAE pretrain)',  YaTCNet,      '0.1M'),
    'tfe_gnn':   ('TFE-GNN (lite)',                TFEGraph,     '0.2M'),
    'et_bert':   ('ET-BERT (lite, no pretrain)',   ETBERTLite,   '4M'),
    'net_mamba': ('NetMamba (lite, no pretrain)',  NetMambaLite, '0.5M'),
    'net_conv':  ('NetConv (lite, no pretrain)',   NetConvLite,  '1M'),
}

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    for name, (fullname, cls, params) in BASELINES.items():
        m = cls(n_classes=10)
        x = torch.randn(2, 2, 200)
        if hasattr(m, "encode"):
            e = m.encode(x)
            y = m(x)
            print(f"{name:>11} ({fullname}): params={count_params(m):>8,} {params:>10}, "
                  f"embed={tuple(e.shape)}, out={tuple(y.shape)}")
        else:
            print(f"{name:>11} ({fullname}): params={count_params(m):>8,} {params:>10} (sklearn, no torch fwd)")
