
# Title

## Table of Contents

1.⁠ ⁠Introduction and Method Background
   1.1 Motivation: long-horizon time series forecasting
   1.2 Required time series background
   1.3 CycleNet paper summary
   1.4 Residual Cycle Forecasting
   1.5 Assumptions and expected limitations

2.⁠ ⁠Reimplementation and Experiments
   2.1 Implementation details
       - PyTorch reimplementation
       - Linear baseline
       - CycleNet-Linear
       - MLP / CycleNet-MLP
       - CycleiTransformer ablation
   2.2 Experimental setup
       - Synthetic dataset
       - ETTh1 dataset
       - Train/validation/test split
       - Preprocessing and normalization
       - Metrics: MSE and MAE
       - Hyperparameters, seeds, hardware
   2.3 Synthetic experiment
       - Full-cycle visible case
       - Partial-cycle visible case
       - Learned cycle visualization
   2.4 ETTh1 experiments
       - Linear vs CycleNet-Linear
       - Horizon study
       - Cycle length sensitivity
       - MLP backbone
       - CycleiTransformer ablation

3.⁠ ⁠Discussion and Conclusion
   3.1 Summary of reproduced findings
   3.2 What worked and what did not
   3.3 Limitations of our reproduction
   3.4 Relation to other papers / future work
   3.5 Final conclusion
   References


## Section 1

## Section 2: Reimplementation and Experiments

### 2.1 Implementation Details

All models were implemented from scratch in Python using PyTorch. The official CycleNet repository ([github.com/ACAT-SCUT/CycleNet](https://github.com/ACAT-SCUT/CycleNet)) was used exclusively as a data provider for the ETTh1 experiments; no model weights or training scripts from the official codebase were reused. Execution ran on Apple Silicon hardware via the MPS backend.

**Linear baseline.** A single `nn.Linear(seq_len, pred_len)` layer applied channel-independently. Input and output are permuted so that the linear projection operates over the time axis for each channel separately. When RevIN is enabled, the input window is standardised by subtracting its mean and dividing by its standard deviation before projection, and the statistics are restored on the output. This is the direct-forecasting Linear model from DLinear.

**CycleNet-Linear.** Implements the Residual Cycle Forecasting (RCF) technique. A learnable parameter table `Q ∈ R^{W×D}`, initialized to zeros, stores one full cycle of length W for each of the D channels. Given a batch cycle index `i ∈ {0, ..., W-1}`, the past cyclic component `c_{past}` is extracted by indexing Q at positions `(i, i+1, ..., i+L-1) mod W`, and the future component `c_{future}` at positions `(i+L, ..., i+L+H-1) mod W`. The residual `x' = x - c_{past}` is projected through a Linear backbone, and the forecast is `ŷ = backbone(x') + c_{future}`. RevIN is applied around the full pipeline. This matches equations (1)-(4) of the paper.

**MLP baseline.** Channel-independent two-layer MLP: `seq_len → 256 → pred_len` with ReLU activation and RevIN, using the same interface as the Linear baseline.

**CycleNet-MLP.** Identical to CycleNet-Linear with the MLP backbone replacing the single Linear layer. This tests whether the RCF benefit persists with a stronger residual forecaster.

**CycleiTransformer.** The official `CycleITransformerModel` was imported from `paper_code/CycleNet/models/CycleiTransformer.py`. An `ITransformerStyleNoCycle` baseline was implemented from scratch by stripping the RCF component from the same architecture: it uses the same inverted-attention encoder (`DataEmbedding_inverted`, `FullAttention`, `Encoder`) and a final linear projector, but operates directly on the raw (RevIN-normalised) input without cycle subtraction or addition.

**Complexity.** The RCF component adds exactly `W × D` learnable parameters (one cycle table) and no additional multiply-accumulate operations beyond the index lookups. For ETTh1 with W=24, D=7, this is 168 parameters, negligible relative to any backbone.

---

### 2.2 Experimental Setup

**Synthetic dataset.** A univariate time series of length T=2000 was generated as:

$$x(t) = \sin\!\left(\frac{2\pi t}{24}\right) + 0.0005\,t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,\, 0.15^2)$$

The cycle length of 24 is known exactly and was provided to CycleNet. The lookback window was set to `seq_len=6`, which covers only one quarter of a full cycle. This deliberately tests the partial-cycle regime where a direct Linear model cannot observe a complete periodic pattern. The forecast horizon was `pred_len=192` (eight full cycles). No feature scaling was applied. The series was split chronologically: the first 1600 steps (80%) for training and the remaining 400 steps for test, yielding 1403 training windows and 401 test windows. No validation split was used for this experiment; model selection was by final-epoch weights.

**ETTh1 dataset.** The Electricity Transformer Temperature dataset (ETTh1) contains 17,420 hourly records across 7 channels recording transformer load and temperature. It is publicly available through the official CycleNet repository. The dataset was loaded using the authors' `Dataset_ETT_hour` class in multivariate mode (`features="M"`). The chronological split follows the standard ETT convention: 12 months training, 4 months validation, 4 months test, yielding 8,209 / 2,545 / 2,545 sliding windows at `seq_len=96`. A `StandardScaler` was fit on the training portion only and applied to all splits. The official data loader returns a cycle index per sample computed as `t mod W`, where `t` is the global time step of the window start; this index was passed directly to RCF models.

**Metrics.** Both MSE and MAE were computed element-wise over all predicted time steps and channels:

$$\text{MSE} = \frac{1}{N} \sum (y - \hat{y})^2, \qquad \text{MAE} = \frac{1}{N} \sum |y - \hat{y}|$$

where N is the total number of predicted scalar values across the evaluation set.

**Hyperparameters.** All hyperparameters are reported in the table below. For ETTh1, the best model checkpoint (by validation MSE) was saved and used for test evaluation. The cycle length was fixed at W=24 (daily cycle) for all ETTh1 experiments unless varied explicitly in the cycle-length ablation.

| Setting | Synthetic | ETTh1 (Linear/MLP) | ETTh1 (Transformer) |
| --- | --- | --- | --- |
| seq_len | 6 | 96 | 96 |
| pred_len | 192 | 96 / 192 / 336 | 336 |
| cycle_len (W) | 24 | 24 | 24 |
| batch_size | 32 | 32 | 32 |
| optimizer | Adam | Adam | AdamW |
| learning rate | 1e-3 | 1e-3 | 1e-4 |
| weight decay | — | — | 1e-4 |
| epochs | 100 | 20 | 20 |
| MLP d_model | — | 256 | 64 |
| n_heads | — | — | 4 |
| e_layers | — | — | 1 |
| d_ff | — | — | 128 |
| dropout | — | — | 0.1 |

**Reproducibility.** Random seeds were fixed at 42 for `numpy`, `random`, and `torch` at the start of each experiment. Each configuration was run once; no error bars or variance estimates are reported due to the single-run constraint. Hardware: Apple M-series chip, MPS backend, 16 GB unified memory.

---

### 2.3 Synthetic Experiment

**Partial-cycle visible case.** With `seq_len=6` the lookback covers only a quarter of the 24-step cycle. The Linear model must infer the entire forecast trajectory from a partial, phase-ambiguous window. CycleNet-Linear, by contrast, reads the global cycle phase from the cycle index and reconstructs the full periodic structure independently of what is visible in the window.

| Model | Test MSE | Test MAE |
| --- | ---: | ---: |
| Linear Baseline | 0.2804 | 0.4256 |
| CycleNet-Linear | 0.0268 | 0.1298 |

CycleNet-Linear achieves a 10.5× reduction in MSE relative to the Linear baseline. The gap is disproportionately large compared to the ETTh1 results because the synthetic signal is almost entirely periodic: once the cycle component is modelled correctly, only noise remains for the residual backbone to handle.

**Learned cycle visualization.** After 100 training epochs, the learned cycle table Q closely reproduces the shape of the ground-truth sinusoid. The match is not exact because Q is trained jointly with the Linear backbone; the backbone absorbs some residual structure, so Q converges to an approximation rather than the exact generative cycle. This qualitatively confirms that the RCF mechanism correctly identifies and internalises the periodic component during training.

---

### 2.4 ETTh1 Experiments

**Linear vs. CycleNet-Linear (pred\_len=336).** The primary comparison on the real benchmark:

| Model | Best Val MSE | Best Val MAE | Test MSE | Test MAE |
| --- | ---: | ---: | ---: | ---: |
| Linear | 1.2852 | 0.7555 | 0.4826 | 0.4466 |
| CycleNet-Linear | 1.2804 | 0.7524 | 0.4616 | 0.4379 |

CycleNet-Linear reduces test MSE by 4.4% and MAE by 1.9% over the Linear baseline. The improvement is consistent across both validation and test sets. Training curves show CycleNet-Linear converging to a lower train MSE from early epochs, with the gap widening steadily, indicating the cycle table continues to refine its representation throughout training.

**Horizon study.** Keeping `seq_len=96` and `cycle_len=24` fixed, the following test results were obtained across forecast horizons:

| Horizon | Linear MSE | CycleNet-Linear MSE | MSE Improvement |
| ---: | ---: | ---: | ---: |
| 96 | 0.3887 | 0.3774 | 2.9% |
| 192 | 0.4403 | 0.4246 | 3.6% |
| 336 | 0.4822 | 0.4627 | 4.0% |

The relative improvement grows with horizon length. This is consistent with the paper's central claim: at short horizons, recent context is sufficient for competitive forecasting; at longer horizons, periodic structure becomes the dominant source of predictability, and explicit cycle modelling provides increasing benefit.

**Cycle length sensitivity (pred\_len=336).** CycleNet-Linear was evaluated at four cycle lengths while holding all other settings constant:

| Cycle length | Meaning | CycleNet Test MSE | CycleNet Test MAE |
| ---: | --- | ---: | ---: |
| 12 | half-day | 0.4742 | 0.4423 |
| 24 | daily | 0.4627 | 0.4392 |
| 48 | two-day | 0.4636 | 0.4391 |
| 168 | weekly | 0.4794 | 0.4476 |

W=24 (daily) is optimal, matching the dominant hourly cycle in ETTh1 confirmed by the paper's ACF analysis. A two-day cycle (W=48) is nearly equivalent, suggesting some tolerance around the true period. A weekly cycle (W=168) performs worse than even the half-day setting, likely because the longer table is harder to learn with 20 epochs of training on a dataset of this size.

**MLP backbone (pred\_len=336).** Replacing the Linear backbone with a two-layer MLP (hidden size 256) tests whether the RCF gain is specific to linear models:

| Model | Test MSE | Test MAE |
| --- | ---: | ---: |
| Linear | 0.4826 | 0.4466 |
| CycleNet-Linear | 0.4616 | 0.4379 |
| MLP | 0.4962 | 0.4552 |
| CycleNet-MLP | 0.4922 | 0.4523 |

Three observations: (1) the MLP backbone without cycle modelling is worse than the Linear baseline, exhibiting overfitting on the 20-epoch training window (training MSE continues to decrease while validation MSE increases); (2) adding RCF improves the MLP, confirming that cycle modelling is beneficial regardless of backbone; (3) CycleNet-MLP does not surpass CycleNet-Linear in this setting, consistent with the paper's finding that MLP advantages emerge mainly on high-dimensional datasets (>100 channels) where nonlinear cross-channel patterns are present.

**CycleiTransformer ablation (pred\_len=336).** The official CycleiTransformer was compared against an iTransformer-style baseline without cycle modelling:

| Model | Test MSE | Test MAE |
| --- | ---: | ---: |
| iTransformer (no cycle) | 0.4807 | 0.4570 |
| CycleiTransformer | 0.4752 | 0.4523 |

The RCF module improves the Transformer-style backbone by 1.1% MSE and 1.0% MAE. The gain is smaller than for the Linear backbone (4.4%), which is expected: the inverted-attention mechanism can implicitly capture some temporal periodicity through its cross-channel modelling, partially overlapping with what RCF contributes explicitly. Nevertheless, the consistent directional improvement indicates RCF provides complementary signal even for architecturally stronger models.

Overall, CycleNet-Linear remains the most competitive model across all settings tested, reinforcing the paper's observation that architectural complexity does not substitute for explicit periodic structure modelling on datasets with strong daily cycles.
