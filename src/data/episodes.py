"""
episodes.py
===========
少样本 episode 采样器。

主要组件：
- EpisodeSampler：每次返回 (n_way, k_shot + q_query) 样本
- make_episode_loader：包装成 PyTorch DataLoader（用于训练循环）

设计要点：
1. 类别划分 base / novel：模拟"已见 vs 新攻击"场景
2. episode 内类别重映射到 [0, n_way) 连续整数，便于 ProtoNet 计算
3. 样本不足的类自动跳过，避免 silent failure
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

class EpisodeSampler:
    """N-way K-shot episode 采样器。

    使用方式：
        >>> sampler = EpisodeSampler(dataset, n_way=5, k_shot=5, q_query=15)
        >>> for episode in sampler.iter_episodes(num_episodes=600):
        >>>     # 训练一步
    """

    def __init__(self, dataset, n_way: int = 5, k_shot: int = 5,
                 q_query: int = 15, seed: int = 42,
                 cls_to_idx: Optional[Dict[int, List[int]]] = None,
                 num_classes: Optional[int] = None):
        """如果 cls_to_idx 提供，则使用外部传入的类→索引映射（用于 zero-shot）。"""
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.min_per_class = k_shot + q_query
        self.rng = np.random.RandomState(seed)

        if cls_to_idx is None:
            # 按类分组索引（基于 dataset.y 缓存）
            self.cls_to_idx: Dict[int, List[int]] = defaultdict(list)
            for i, y in enumerate(dataset.y.tolist()):
                self.cls_to_idx[int(y)].append(i)
        else:
            self.cls_to_idx = cls_to_idx

        # 过滤掉样本不足的类
        self.valid_classes = sorted([
            c for c, idxs in self.cls_to_idx.items()
            if len(idxs) >= self.min_per_class
        ])

        if len(self.valid_classes) < n_way:
            raise ValueError(
                f"可用类数 {len(self.valid_classes)} 少于 n_way={n_way}。"
                f"需要每个类至少 {self.min_per_class} 个样本。"
            )

    def sample_episode(self) -> Dict:
        """采样一个 episode。

        返回：
            {
                'support_indices': [n_way * k_shot],
                'query_indices':   [n_way * q_query],
                'support_labels':  [n_way * k_shot]，episode 内 [0, n_way)
                'query_labels':    [n_way * q_query]，episode 内 [0, n_way)
                'classes':         list[n_way]，原类别 id
            }
        """
        chosen = self.rng.choice(self.valid_classes, size=self.n_way,
                                 replace=False)
        support_idx, support_y = [], []
        query_idx, query_y = [], []
        for new_label, c in enumerate(chosen):
            pool = self.cls_to_idx[c]
            picks = self.rng.choice(pool, self.k_shot + self.q_query,
                                    replace=False)
            support_idx.extend(picks[:self.k_shot].tolist())
            query_idx.extend(picks[self.k_shot:].tolist())
            support_y.extend([new_label] * self.k_shot)
            query_y.extend([new_label] * self.q_query)
        return {
            "support_indices": support_idx,
            "query_indices": query_idx,
            "support_labels": torch.tensor(support_y, dtype=torch.long),
            "query_labels": torch.tensor(query_y, dtype=torch.long),
            "classes": chosen.tolist(),
        }

    def iter_episodes(self, num_episodes: int):
        """生成 num_episodes 个 episode 的生成器。"""
        for _ in range(num_episodes):
            yield self.sample_episode()

class EpisodeDataset(Dataset):
    """把 episode 包装成 Dataset（每个元素是一个完整 episode）。

    用途：和 DataLoader 配合，每个 batch = 1 个 episode。
    """

    def __init__(self, sampler: EpisodeSampler, num_episodes: int):
        self.sampler = sampler
        self.num_episodes = num_episodes

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        return self.sampler.sample_episode()

def make_episode_loader(dataset, n_way: int = 5, k_shot: int = 5,
                        q_query: int = 15, num_episodes: int = 600,
                        seed: int = 42) -> DataLoader:
    """构造 episode DataLoader（每 batch = 1 episode）。

    训练时 num_episodes = 600 (meta-batch 风格)；
    评估时 num_episodes = 1000 以获得稳定均值。
    """
    sampler = EpisodeSampler(dataset, n_way=n_way, k_shot=k_shot,
                             q_query=q_query, seed=seed)
    ds = EpisodeDataset(sampler, num_episodes)
    return DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

def split_base_novel(dataset, base_ratio: float = 0.7,
                     seed: int = 42) -> Tuple[List[int], List[int], dict]:
    """把数据按类别划分为 base（已见）/ novel（新攻击）。

    返回：
        base_indices, novel_indices, class_split
        class_split = {"base": [class_ids], "novel": [class_ids]}
    """
    cls_to_idx = defaultdict(list)
    for i, y in enumerate(dataset.y.tolist()):
        cls_to_idx[int(y)].append(i)

    classes = sorted(cls_to_idx.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(classes)
    n_base = max(1, int(len(classes) * base_ratio))
    base_classes = classes[:n_base]
    novel_classes = classes[n_base:]

    if len(novel_classes) == 0:
        raise ValueError(
            f"类别数 {len(classes)} 太少，无法划分 novel（base_ratio={base_ratio}）"
        )

    base_indices = []
    novel_indices = []
    for c, idxs in cls_to_idx.items():
        if c in base_classes:
            base_indices.extend(idxs)
        else:
            novel_indices.extend(idxs)

    return base_indices, novel_indices, {
        "base": base_classes,
        "novel": novel_classes,
    }
