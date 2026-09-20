"""
utils.py
========
通用工具模块：设备、随机种子、日志。
"""
from __future__ import annotations
import os
import sys
import json
import time
import random
import logging
from typing import Optional
import numpy as np
import torch

def get_device(prefer: str = "auto") -> torch.device:
    """自动选择设备：cuda > xpu > cpu。"""
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

def set_seed(seed: int = 42):
    """固定所有随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)

class TqdmLogger:
    """简易日志包装，便于子进程 stdout 输出。"""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def info(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

def save_json(obj, path: str):
    """保存 JSON，自动建目录。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_mean_std(values: list) -> dict:
    """把多次实验结果列表合并成 mean±std 字典。"""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n": len(values),
        "raw": values,
    }
