### Advanced Topics in Machine Learning - Time Series Forecasting - CycleNet


---

This repository contains our reimplementation and reproducibility study of **CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns** ([arXiv:2409.18479](https://arxiv.org/abs/2409.18479), NeurIPS 2024 Spotlight).

The original authors' repository is available at [ACAT-SCUT/CycleNet](https://github.com/ACAT-SCUT/CycleNet).

For a summary of useful time series forecasting properties and CycleNet architecture see [NOTES](notes.md). The full reproducibility report, including all numerical results, tables and discussion is in [`report/`](report/).
---

#### Repository structure

```
.
├── notebooks/
│   ├── 01_minimal_cyclenet_synthetic.ipynb   # synthetic-data sanity check
│   ├── 02_cyclenet_real_dataset.ipynb        # main ETTh1 experiments
│   └── 03_etth1_periodicity.ipynb            # ETTh1 exploratory analysis
├── report/                                   # LaTeX report + compiled PDF
├── notes.md                                  # background notes on CycleNet
├── paper_code/                               # (gitignored) authors' repo + dataset
└── original_paper/                           # PDF of the original paper
```

---

#### What each notebook does

**`01_minimal_cyclenet_synthetic.ipynb`**  Minimal CycleNet on a controlled synthetic periodic signal with a known cycle length `p = 24`. We compare a plain `Linear` baseline against `CycleNet-Linear` in two regimes: input window long enough to see a full cycle (`seq_len = 24`), and input window shorter than one cycle (`seq_len = 6`). This isolates the Residual Cycle Forecasting (RCF) mechanism.

**`03_etth1_periodicity.ipynb`**  Exploratory analysis of the **ETTh1** dataset. We inspect its variables, time range, and sampling, and verify the single property CycleNet relies on: the presence of a stable daily / weekly periodicity (via plots and autocorrelation).

**`02_cyclenet_real_dataset.ipynb`** Main experimental notebook on **ETTh1** (multivariate `features="M"`, `seq_len = 96`). Uses the authors' official data-loading pipeline. Reimplements four model variants from scratch:

| Model | Description |
|---|---|
| Linear | Direct channel-independent linear forecasting |
| CycleNet-Linear | Linear backbone with learned recurrent cycle removal/addition |
| MLP | Two-layer ReLU MLP backbone |
| CycleNet-MLP | MLP backbone with cycle removal/addition |

It also runs a Transformer-style ablation using the authors' `CycleiTransformer.py` against our own no-cycle Transformer baseline, and sweeps the prediction horizon `H ∈ {96, 192, 336}` and the cycle length `W ∈ {12, 24, 48, 168}`.

---

#### Setup

1) Download the ETTh1 CSV provided by the authors: <https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view>

2) From the repo root, clone the authors' repository inside a `paper_code/` folder:

```bash
mkdir paper_code
cd paper_code
git clone https://github.com/ACAT-SCUT/CycleNet.git
```

3) Create a `dataset/` folder inside the cloned repository and place the downloaded CSV there:

```
paper_code/CycleNet/dataset/ETTh1.csv
```

The notebooks expect this exact layout (`ROOT_PATH = "../paper_code/CycleNet/dataset"`).

4) Python dependencies: `torch`, `numpy`, `pandas`, `matplotlib`. The notebooks were developed in Python 3.13 with PyTorch using MPS acceleration on Apple Silicon (CUDA / CPU also supported).

---

#### Reproducibility

All experiments use seed `42`. On CUDA and CPU the runs are deterministic; on MPS small numerical differences may still occur because some Metal kernels are non-deterministic by design. The reported tables in [`report/`](report/) come from single runs with this seed.

---

#### Results

The full results (horizon study, cycle-length sensitivity, MLP and CycleiTransformer comparisons) are reported and discussed in the [LaTeX report](report/Reproducing_CycleNet__Time_Series_Forecasting_through_Modeling_Periodic_Patterns__3_/main.tex) and its compiled [PDF](report/Reproducing_CycleNet__Time_Series_Forecasting_through_Modeling_Periodic_Patterns__3_/main.pdf). Each notebook also prints and tabulates its own results inline.
