"""
run_fewshot.py
==============
实验 2/3：少样本下游评估。

用法：
    python experiments/run_fewshot.py --dataset cic \\
        --encoder_ckpt output/checkpoints/cic_simclr_ep50.pt \\
        --algo protonet --n_way 5 --k_shot 5
"""
import subprocess
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

if __name__ == "__main__":
    from src.training.fewshot_train import main
    main()
