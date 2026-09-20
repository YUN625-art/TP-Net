# TP-Net 设计文档

## 1. 论文定位

**题目**：Self-Supervised Pretrained Encrypted Traffic Representations for Few-Shot Threat Detection

**目标期刊**：Computers & Security (Elsevier IF≈7.4, CCF-B)

**核心创新点**：
1. 首次将自监督对比学习（SimCLR/MoCo）应用于加密流量 4 分支行为特征
2. 验证少样本 N-way K-shot 协议在加密流量新攻击检测上的有效性
3. 提出流量专用数据增强策略（包级 jitter + 突发 burst jitter + 流统计噪声 + 握手掩码）
4. 跨数据集预训练 → 跨域表征的领域无关性

## 2. 与上篇 DBAF-Net 的差异

| 维度 | DBAF-Net (上) | TP-Net (本) |
|---|---|---|
| 学习范式 | 全监督 | 自监督 + 少样本 |
| 数据需求 | 大量标注 | 无标注预训练 + 极少标注 |
| 评估协议 | 70/15/15 全量 | N-way K-shot episode |
| 新攻击泛化 | 重训整个模型 | 复用 encoder + 5-shot proto |
| 创新角度 | 4 分支架构 | 表征 + 协议 + 增强 |

**Why**: 上篇 DBAF-Net 因 USTC 30k 数据量输给 RF tabular 20pp，"4 分支全面 SOTA" 叙事不成立。本论文重新定位为 **自监督 + 少样本新攻击泛化**，绕过数据量短板，强调 SSL 表征的少样本迁移能力。

## 3. 模型架构

```
原始加密流量 (h5: P, B, S, H)
                ↓
┌────────────┴────────────┐
│  自监督预训练 (SimCLR/MoCo) │
│  - Backbone: 1D-CNN (P+B) + MLP (S+H)
│  - 数据增强: packet jitter/mask/time-warp
│  - 损失: NT-Xent
└────────────┬────────────┘
                ↓
        预训练 encoder f(·)
                ↓
┌────────────┴────────────┐
│  少样本下游 (N-way K-shot)  │
│  - ProtoNet / MatchingNet / Linear Probing
└─────────────────────────┘
```

## 4. 数据流

| 阶段 | 输入 | 输出 |
|---|---|---|
| 加载 | h5 文件 | H5FlowDataset |
| 增强 | (P, B, S, H) | view1, view2 |
| 预训练 | view pairs | encoder 权重 |
| 少样本 | support + query | accuracy |

## 5. 关键工程决策

- **输入维度**：
  - P: (B, 200, 2) 包级头 + 长度
  - B: (B, 16, 4) 突发块 4 维
  - S: (B, 32) 流统计
  - H: (B, 71) 握手特征
- **encoder 输出**：128 维 embedding
- **温度参数**：τ = 0.1 (SimCLR 标准)
- **增强策略**：medium (jitter=0.05/0.15/0.2/0.1/0.1/0.05/0.1)
- **episode**：N=5, K=5, Q=15 (600 episodes / seed × 5 seeds)

## 6. 实验矩阵

| 实验 | 内容 | 论文位置 |
|---|---|---|
| 1 | 自监督预训练（SimCLR/MoCo × 3 数据集） | §4.3 |
| 2 | 少样本 5-way 5-shot 分类 | §4.4 表 1 |
| 3 | Linear Probing 表征质量 | §4.4 表 2 |
| 4 | 新攻击零样本泛化（novel class） | §4.5 表 3 |
| 5 | 数据增强消融（none/weak/medium/strong） | §4.6 表 4 |
| 6 | 跨数据集预训练（cic↔ustc↔iscx） | §4.5 表 5 |

## 7. 跑实验顺序

```bash
# 1. 预训练（每个数据集 × 算法）
python experiments/run_pretrain.py --dataset cic   --algo simclr --epochs 50
python experiments/run_pretrain.py --dataset ustc  --algo simclr --epochs 50
python experiments/run_pretrain.py --dataset iscx  --algo simclr --epochs 30
python experiments/run_pretrain.py --dataset cic   --algo moco   --epochs 50

# 2. 少样本评估
python experiments/run_fewshot.py --dataset cic --encoder_ckpt output/checkpoints/cic_simclr_ep50.pt

# 3. 新攻击零样本
python experiments/run_zeroshot.py --dataset cic --encoder_ckpt output/checkpoints/cic_simclr_ep50.pt

# 4. 消融
python experiments/run_ablation.py --dataset cic --epochs 20

# 5. 跨数据集
python experiments/run_xdataset.py --epochs 30
```
