# TP-Net: Self-Supervised Pretrained Encrypted Traffic Representations for Few-Shot Threat Detection

> Paper artifact — companion code for the **MDPI Sensors** submission `paper/TP-Net.pdf` (23 pages).
> Goal: reproducible evidence for the SSL + few-shot encrypted-traffic novel-attack claim.

## 1. What's in this repository

```
TP-Net/
├── README.md                     # this file
├── .gitignore
├── requirements.txt
├── paper/                        # LaTeX source + rendered PDF (MDPI Sensors class)
│   ├── TP-Net.tex
│   ├── TP-Net.pdf
│   ├── TP-Net.bbl / .aux
│   └── figures/                  # PNG / PDF figures embedded in the paper
├── configs/
│   └── default.yaml              # all hyper-parameters used in the paper
├── src/                          # importable library
│   ├── data/         (load_h5, augment, episodes, extract_features)
│   ├── models/       (encoder, ssl, fewshot, baselines)
│   ├── training/     (pretrain, fewshot_train)
│   └── evaluation/   (eval)
└── experiments/                  # entry-point scripts, one per table / figure
    ├── run_pretrain.py
    ├── run_fewshot.py
    ├── run_zeroshot.py
    ├── run_ablation.py
    ├── run_xdataset.py
    ├── ssl_diversity_ablation.py
    ├── finetune_head_5shot.py
    ├── supervised_pretrain.py
    ├── supervised_eval_fewshot.py
    ├── supervised_novel_eval.py
    ├── kshot_sweep.py
    ├── per_class_diag_ustc.py
    ├── label_efficiency_curve.py
    ├── openset_fsl_eval.py
    ├── dohbrw_ssl_vs_random.py
    ├── dohbrw_balanced_ssl_vs_random.py
    ├── adversarial_robust.py
    ├── adaptive_aug_meta.py
    ├── baseline_random_matched.py
    ├── baseline_supervised_check.py
    ├── baseline_dense_sanity.py
    ├── deployment_mock.py
    └── compute_cohens_d_episode.py
```

> **Figure-generation scripts** (`draw_fig*.py`) are intentionally **not** included —
> the figures they produce are already checked into `paper/figures/`.

## 2. Reproducing the paper results

### 2.1 Setup

```bash
pip install -r requirements.txt
```

### 2.2 Datasets (preprocessing required)

The four preprocessed `.h5` files must live at the project root before any
experiment can run:

| File                  | Source                                         | # classes |
|-----------------------|------------------------------------------------|-----------|
| `output/cic_full.h5`  | CIC-IDS2017 (Sharafaldin et al., ICISSP 2018)  | 10        |
| `output/ustc_full.h5` | USTC-TFC2016 (Wang et al., ICOIN 2017)         | 20        |
| `output/iscx_full.h5` | ISCX-VPN2016 (Draper-Gil et al., ICISSP 2016)  | 2         |
| `output/dohbrw_full.h5` | DoHBrw-2020 (Stratosphere IPS Lab)            | 2 (1 malicious) |

These files are not in the repo (they are too large and are dataset-distribution
restricted). Run `src/data/extract_features.py` against your local PCAP copies
to rebuild them; otherwise download from the public sources linked in the paper.

### 2.3 Quick start (one full pass)

```bash
# 1) SSL pretraining — SimCLR, 50 ep, 20k samples (CIC)
python experiments/run_pretrain.py --dataset cic --algo simclr --epochs 50

# 2) 5-way 5-shot ProtoNet evaluation (CIC)
python experiments/run_fewshot.py --dataset cic --algo protonet \
                                  --n_way 5 --k_shot 5

# 3) Novel-class 5-shot (3 datasets)
python experiments/run_zeroshot.py --dataset cic --novel_classes 2
python experiments/run_zeroshot.py --dataset ustc --novel_classes 5
python experiments/run_zeroshot.py --dataset iscx --novel_classes 1
```

### 2.4 Per-table / per-figure entry points

| Paper artifact                | Entry script                                                       |
|-------------------------------|--------------------------------------------------------------------|
| Table 1 — Datasets            | `src/data/load_h5.py`                                              |
| Table 3 — Main results        | `experiments/run_fewshot.py` + `experiments/baseline_random_matched.py` |
| Table 4 — FT-Head             | `experiments/finetune_head_5shot.py`                               |
| Table 5 — Sup vs SSL & FT-Head| `experiments/supervised_eval_fewshot.py` + `finetune_head_5shot.py` + `supervised_novel_eval.py` |
| Table 6 — Comprehensive       | aggregates Tables 3/4 + `experiments/run_zeroshot.py`              |
| Table A.1 — K-shot scan       | `experiments/kshot_sweep.py`                                       |
| Table A.2 — SSL diversity     | `experiments/ssl_diversity_ablation.py`                            |
| Table A.3 — Heads comparison  | `experiments/run_fewshot.py --algo matching` / `--algo moco`       |
| Table A.4 — USTC per-class    | `experiments/per_class_diag_ustc.py`                               |
| Table A.5 — Open-set FSL      | `experiments/openset_fsl_eval.py`                                  |
| §3.5 DoHBrw encryption        | `experiments/dohbrw_ssl_vs_random.py` + `dohbrw_balanced_ssl_vs_random.py` |
| §4.4 Novel-class              | `experiments/run_zeroshot.py`                                      |
| §4.5 Cross-dataset            | `experiments/run_xdataset.py`                                      |
| §4.6 Augmentation ablation    | `experiments/run_ablation.py`                                      |
| §5.3 Limitations              | `experiments/adversarial_robust.py` + `adaptive_aug_meta.py`       |

## 3. Build the PDF from source

```bash
cd paper
pdflatex TP-Net.tex        # pass 1
pdflatex TP-Net.tex        # pass 2 (cross-refs)
pdflatex TP-Net.tex        # pass 3 (final pagination)
```

The MDPI Sensors class is shipped with the project (`Definitions/mdpi.cls` is bundled
inside the official `mdpi.cls` distribution and is not checked into this repo). The
LaTeX source embeds `\begin{thebibliography}` directly, so no `bibtex` pass is needed.

## 4. Citation

```bibtex
@article{tpnet2026,
  title  = {TP-Net: Self-Supervised Pretrained Encrypted Traffic
            Representations for Few-Shot Threat Detection in IoT and Sensor Networks},
  author = {Yao, X. and Feng, Y. and Wang, Q.},
  journal= {Sensors (MDPI)},
  year   = {2026},
  note   = {Under review}
}
```

## 5. License

Code: MIT. Paper text and figures: CC-BY-4.0.
Datasets retain their original licenses (see respective dataset websites).
