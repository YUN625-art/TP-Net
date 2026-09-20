
import argparse
import os
import sys

# 让 `python src/xxx.py` 与 `python -m src.xxx` 都能跑
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
