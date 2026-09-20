"""
run_pretrain.py
===============
实验 1：自监督预训练入口。

用法：
    python experiments/run_pretrain.py --dataset cic --algo simclr --epochs 50
"""
import subprocess
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

if __name__ == "__main__":
    from src.training.pretrain import main
    main()
