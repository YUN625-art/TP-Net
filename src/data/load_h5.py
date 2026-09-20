import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

H_DIM = 71  # 6 JA3 + 32 suite + 32 ext + 1 miss

class H5FlowDataset(Dataset):
    """从预处理 h5 加载 (P, B, S, H, y)"""
    def __init__(self, h5_path, indices, scaler=None, fit=False,
                 max_pkt=200, max_burst=16):
        indices = np.array(indices, dtype=np.int64)
        # h5py fancy indexing 要求单调递增；stratified/train_test_split 返回的
        # indices 内部不单调。先 sort 取数据，再用 inverse 还原调用方期望顺序。
        self.indices = indices
        self._sort_perm = np.argsort(indices, kind="stable")
        sorted_idx = indices[self._sort_perm]
        self.max_pkt = max_pkt
        self.max_burst = max_burst
        self._h5 = None  # lazy open
        self._h5_path = h5_path

        with h5py.File(h5_path, "r") as h:
            n = h["stat"].shape[0]
            S = h["stat"][sorted_idx]
            # 用 h5py.asstr() 把变长字符串 → 定长 Python 字符串，再一次性 ndarray 化
            labels_all = h["label"].asstr()[:]
            labels_sorted = labels_all[sorted_idx]
        cat = pd.Categorical(labels_sorted)
        self._y_sorted = torch.tensor(cat.codes, dtype=torch.long)
        self._unsort_perm = np.argsort(self._sort_perm, kind="stable")
        self.y = self._y_sorted[self._unsort_perm]
        self.classes = list(cat.categories)

        # StandardScaler on S（基于 sorted 数据拟合，最后 unsort 回原顺序）
        S = np.where(np.isfinite(S), 0, S)
        S = np.log1p(np.clip(S, 0, None))
        self.scaler = scaler or StandardScaler()
        if fit:
            S = self.scaler.fit_transform(S)
        else:
            S = self.scaler.transform(S)
        self.stat = torch.tensor(S[self._unsort_perm], dtype=torch.float32)
        self._S_cached = True

    def __len__(self):
        return len(self.indices)

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, "r")

    def __getitem__(self, i):
        self._ensure_open()
        idx = int(self.indices[i])
        pkt = torch.tensor(self._h5["pkt_seq"][idx], dtype=torch.float32)
        burst = torch.tensor(self._h5["burst_seq"][idx], dtype=torch.float32)
        hand = torch.tensor(self._h5["hand"][idx], dtype=torch.float32)
        return pkt, burst, self.stat[i], hand, self.y[i]

def time_aware_split_indices(h5_path, train_ratio=0.7, val_ratio=0.15):
    """时间感知划分：按 flow_key 字典序近似（h5 中无时间戳）

    实际策略：用 flow_key 字符串排序（稳定且可重现）。
    """
    with h5py.File(h5_path, "r") as h:
        n = h["stat"].shape[0]
        # 用 flow_key 排序作为时间代理
        keys = [h["flow_key"][i].decode() for i in range(n)]
        order = sorted(range(n), key=lambda i: keys[i])
    n_tr = int(n * train_ratio)
    n_va = int(n * (1 - train_ratio) * val_ratio)
    tr = order[:n_tr]
    va = order[n_tr:n_tr + n_va]
    te = order[n_tr + n_va:]
    return tr, va, te

def stratified_split_indices(h5_path, train_ratio=0.7, val_ratio=0.15, seed=42):
    """分层随机划分；过滤掉样本数 < 2 的类（避免 train_test_split 报错）"""
    with h5py.File(h5_path, "r") as h:
        n = h["stat"].shape[0]
        # h5py.asstr() 一次性转定长字符串 array，比逐元素 decode 快百倍
        labels = h["label"].asstr()[:]
    # 过滤掉样本数 < 2 的类
    counts = pd.Series(labels).value_counts()
    keep_classes = counts[counts >= 2].index.tolist()
    keep_mask = np.isin(labels, keep_classes)
    keep_idx = np.where(keep_mask)[0]
    keep_labels = labels[keep_idx]
    n_kept = len(keep_idx)
    idx_tr_local, idx_rest_local = train_test_split(
        np.arange(n_kept), train_size=train_ratio,
        stratify=keep_labels, random_state=seed)
    val_ratio_in_rest = val_ratio / (1 - train_ratio)
    idx_va_local, idx_te_local = train_test_split(
        idx_rest_local, train_size=val_ratio_in_rest,
        stratify=keep_labels[idx_rest_local], random_state=seed)
    idx_tr = keep_idx[idx_tr_local].tolist()
    idx_va = keep_idx[idx_va_local].tolist()
    idx_te = keep_idx[idx_te_local].tolist()
    return idx_tr, idx_va, idx_te

def make_loaders(h5_path, split="time_aware", batch_size=64,
                 train_ratio=0.7, val_ratio=0.15, max_samples=None, seed=42,
                 min_per_class=2):
    """生成 train/val/test DataLoader

    max_samples: 若给定，按**类别比例**分层抽样（避免稀少的类被抽光）。
    min_per_class: 每个类至少保留的样本数（保证 tiny class 有机会被学到）。
    """
    if split == "time_aware":
        idx_tr, idx_va, idx_te = time_aware_split_indices(
            h5_path, train_ratio, val_ratio)
    else:
        idx_tr, idx_va, idx_te = stratified_split_indices(
            h5_path, train_ratio, val_ratio, seed)

    if max_samples:
        # 按类别分层抽样，避免 max_samples 随机抽光稀少的类
        with h5py.File(h5_path, "r") as h:
            # h5py.asstr() 一次性转定长 Python 字符串，比逐元素 decode 快百倍
            labels_all = h["label"].asstr()[:]
            labels_tr = labels_all[idx_tr]
            labels_te = labels_all[idx_te]
        idx_tr_sub = _stratified_subsample(idx_tr, labels_tr, max_samples,
                                           seed=seed, min_per_class=min_per_class)
        idx_te_sub = _stratified_subsample(idx_te, labels_te, max_samples // 4,
                                           seed=seed + 1, min_per_class=min_per_class)
        idx_tr, idx_te = idx_tr_sub, idx_te_sub

    tr = H5FlowDataset(h5_path, idx_tr, fit=True)
    va = H5FlowDataset(h5_path, idx_va, scaler=tr.scaler)
    te = H5FlowDataset(h5_path, idx_te, scaler=tr.scaler)

    from torch.utils.data import DataLoader
    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0,
                       drop_last=True)
    dl_va = DataLoader(va, batch_size=batch_size, num_workers=0)
    dl_te = DataLoader(te, batch_size=batch_size, num_workers=0)
    return dl_tr, dl_va, dl_te, tr.classes, len(tr), len(va), len(te)

def _stratified_subsample(indices, labels, max_n, seed=42, min_per_class=2):
    """按类别比例分层抽样，确保每个类至少 min_per_class 个样本

    实现要点：避免 O(n²) 的 [i for i in idxs if i not in set(selected)]。
    """
    rng = np.random.RandomState(seed)
    from collections import defaultdict
    cls_to_idx = defaultdict(list)
    for i, c in zip(indices, labels):
        cls_to_idx[c].append(i)

    # 每类先保底 min_per_class
    selected = []
    selected_set = set()
    for c, idxs in cls_to_idx.items():
        n_take = min(min_per_class, len(idxs))
        if n_take > 0:
            sel = rng.choice(len(idxs), n_take, replace=False)
            for s in sel:
                v = idxs[int(s)]
                selected.append(v)
                selected_set.add(v)

    # 剩余配额按类别原始比例分配
    leftover = max_n - len(selected)
    total = sum(len(v) for v in cls_to_idx.values())
    if leftover > 0 and total > 0:
        for c, idxs in cls_to_idx.items():
            n_take = int(leftover * len(idxs) / total)
            if n_take <= 0:
                continue
            # 仅从未选中的 idxs 里抽
            avail = [i for i in idxs if i not in selected_set]
            if not avail:
                continue
            take = min(n_take, len(avail))
            sel = rng.choice(len(avail), take, replace=False)
            for s in sel:
                v = avail[int(s)]
                selected.append(v)
                selected_set.add(v)
            if len(selected) >= max_n:
                break
        # 如果还差一点，从最大类里补
        if len(selected) < max_n:
            for c, idxs in sorted(cls_to_idx.items(),
                                   key=lambda x: -len(x[1])):
                avail = [i for i in idxs if i not in selected_set]
                if avail:
                    need = min(len(avail), max_n - len(selected))
                    sel = rng.choice(len(avail), need, replace=False)
                    for s in sel:
                        v = avail[int(s)]
                        selected.append(v)
                        selected_set.add(v)
                if len(selected) >= max_n:
                    break
    # 截断到 max_n
    if len(selected) > max_n:
        sel = rng.choice(len(selected), max_n, replace=False)
        selected = [selected[int(s)] for s in sel]
    return selected