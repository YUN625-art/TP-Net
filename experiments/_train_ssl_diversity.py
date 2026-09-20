"""批量训练所有缺失 SSL ckpt 的入口脚本。

按 cic → ustc → iscx 顺序，每个数据集内 algo 顺序：byol, simsiam, mae。
每个训练任务用 subprocess 同步跑，串行执行避免 CPU 抢占。
"""
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
# 切到项目根
os.chdir(r"D:\computer&security\experiment")
ROOT = r"D:\computer&security\experiment"

DATASETS = [("cic", 50), ("ustc", 50), ("iscx", 30)]
ALGOS = ["byol", "simsiam", "mae"]
OUT_DIR = os.path.join(ROOT, "output", "checkpoints")


def need_train(dataset, algo, epochs):
    ckpt1 = os.path.join(OUT_DIR, f"{dataset}_{algo}_ep{epochs}_medium.pt")
    ckpt2 = os.path.join(OUT_DIR, f"{dataset}_{algo}_ep{epochs}.pt")
    return not (os.path.exists(ckpt1) or os.path.exists(ckpt2))


def train_one(dataset, algo, epochs):
    print(f"\n[开始] {dataset}/{algo} ep{epochs}")
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "src.training.pretrain",
        "--dataset", dataset, "--algo", algo,
        "--epochs", str(epochs), "--batch_size", "256",
        "--max_samples", "20000",  # 分层抽样 20k，与论文一致
        "--aug_preset", "medium",
        "--save_every", str(epochs),
        "--out_dir", OUT_DIR,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    rc = subprocess.run(cmd, env=env).returncode
    dt = time.time() - t0
    print(f"[完成] {dataset}/{algo}: rc={rc}  耗时 {dt/60:.1f} min")
    return rc


if __name__ == "__main__":
    n_total, n_skip, n_done, n_fail = 0, 0, 0, 0
    for ds, ep in DATASETS:
        for algo in ALGOS:
            n_total += 1
            if not need_train(ds, algo, ep):
                print(f"[跳过] {ds}/{algo} 已存在")
                n_skip += 1
                continue
            rc = train_one(ds, algo, ep)
            if rc == 0:
                n_done += 1
            else:
                n_fail += 1
    print(f"\n汇总: total={n_total}  skip={n_skip}  done={n_done}  fail={n_fail}")
