### Advanced Topics in Machine Learning - Time Series Forecasting - CycleNet


--- 

Welcome to the repository of our project in which we reproduce the architecture and the results of **CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns** (https://arxiv.org/abs/2409.18479). NeurIPS 2024 Spotlight Poster! 

you can find the original paper repository at https://github.com/ACAT-SCUT/CycleNet. 

For a summary of useful time series forecasting properties and CycleNet architecture please see [NOTES](notes.md)

---

#### 1. CycleNet on Synthetic Time Series

As a first reproducibility step. Before moving to the official benchmark datasets, we implemented a minimal version of CycleNet on a controlled synthetic time series. The goal of this experiment was not to reproduce the full paper yet, but to verify that the core mechanism of **Residual Cycle Forecasting** behaves as expected.

We generated a synthetic periodic signal with a known cycle length `p = 24` and trained two models:

| Model | Description |
|---|---|
| Linear Baseline | Directly maps the past input window to the future horizon |
| CycleNet-Linear | Learns a cycle component, subtracts it from the input, forecasts the residual, and adds the future cycle back |

The experiment was designed to test the main intuition of CycleNet: explicit cycle modeling should help when long-horizon forecasting depends on stable periodic structure.

##### What we tested

We tested the linear model and the CycleNet-linear in two instances, one where the full cycle was shown in training, and one where only partial information was provided.

the input window for the partial cycle where CycleNet improves the MSE by 90.45% is the following:

`seq_len = 6`;
`pred_len = 192`;
`cycle_len = 24`;
`batch_size = 32`;

We also tested in cases in where the linear model ingests the entire pattern.

Key finding: Explicit cycle modeling is most useful when the forecasting horizon is long and the recent input window does not fully reveal the underlying periodic pattern.

you can find the implementation and results on synthetic data in the [Synthetic time series notebook](notebooks/01_minimal_cyclenet_synthetic.ipynb).

---

#### 2. CycleNet on benchmark dataset

We reproduce the architecture on one of the benchmarked datasets.

We start by using the **ETTh1** dataset.

About the dataset: Electricity Transformer Temperature (ETT) dataset collection. It is a widely recognized benchmark dataset heavily used in data science and machine learning for multivariate time-series forecasting. It tracks the electrical load and thermal state of an electricity power transformer over a two-year period.

to replicate our reimplementation:

1) you can download the dataset at this link provided by the authors: https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view

2) inside this repo create a folder named **paper_code** and clone the author's repository: https://github.com/ACAT-SCUT/CycleNet 

On terminal:

```bash
mkdir paper_code
cd paper_code
git clone https://github.com/ACAT-SCUT/CycleNet.git
```

3) then create the dataset folder inside the CycleNet repository and insert there the downloaded data. e.g. ../dataset/ETTh1.csv


For this part, we used the official CycleNet repository only for the data-loading pipeline. In particular, we used the authors' `DataLoader` loader, which handles:

- chronological train/validation/test splitting;
- standardization using the training split;
- sliding-window construction;
- cycle-index generation.

The forecasting setup was multivariate:

```python
features = "M"
seq_len = 96
cycle_len = 24
```
This means that the model receives 96 past hourly observations for all 7 ETTh1 variables and predicts the future values for all variables.

We compared two models:

| Model | Description |
|---|---|
| Linear | Direct channel-independent linear forecasting |
| CycleNet-Linear | Linear backbone with learned recurrent cycle removal/addition |

Both models use the same Linear temporal backbone. The only difference is that CycleNet-Linear explicitly models a learned periodic component.



##### Horizon Study

We tested whether CycleNet becomes more useful as the prediction horizon increases.

| Prediction Horizon | Linear Test MSE | CycleNet Test MSE | Relative MSE Improvement |
| -----------------: | --------------: | ----------------: | -----------------------: |
|                 96 |        0.388711 |          0.377440 |                   ~2.90% |
|                192 |        0.440288 |          0.424568 |                   ~3.57% |
|                336 |        0.482197 |          0.462739 |                   ~4.04% |



CycleNet-Linear consistently improves over the Linear baseline. The improvement is modest, but it increases as the forecasting horizon becomes longer. This supports the main intuition of CycleNet: explicit periodic modeling becomes more useful when forecasting further into the future.

##### Cycle Length Sensitivity

We also tested different cycle lengths for `pred_len = 336`.

| Cycle Length | Interpretation | CycleNet Test MSE | CycleNet Test MAE |
| -----------: | -------------- | ----------------: | ----------------: |
|           12 | half-day cycle |          0.474201 |          0.442331 |
|           24 | daily cycle    |          0.462739 |          0.439170 |
|           48 | two-day cycle  |          0.463609 |          0.439088 |
|          168 | weekly cycle   |          0.479414 |          0.447642 |



The best results are obtained with daily-scale cycles, especially `cycle_len = 24` and `cycle_len = 48`. The weekly cycle performs much worse in this setup.
This suggests that CycleNet is sensitive to the chosen cycle length and works best when the assumed periodicity matches useful temporal structure in the data.


##### MLP CycleNet vs MLP model

In our reimplementation, CycleNet improves both Linear and MLP backbones, but the Linear backbone remains the most effective among the tested variants.

| Model           |     Test MSE |     Test MAE |
| --------------- | -----------: | -----------: |
| Linear          |     0.482642 |     0.446608 |
| CycleNet-Linear | **0.461598** | **0.437861** |
| MLP             |     0.496212 |     0.455248 |
| CycleNet-MLP    |     0.492152 |     0.452285 |



You can find the detailed implementation in the  [Cyclenet real benchmark notebook](notebooks/02_cyclenet_real_dataset.ipynb)



--- 

#### 3. CycleiTransformer (Advanced)

We also tested CycleNet with a stronger Transformer-style backbone. Since the repository provides `CycleiTransformer.py` but not a separate non-cycle `iTransformer.py`, we implemented an ablated version of the same architecture without cycle removal and cycle addition.

| Model | Description |
|---|---|
| Transformer-style no cycle | Same inverted Transformer-style backbone, without cycle modeling |
| CycleiTransformer | Same backbone with learned cycle removal/addition |

Results:

| Model | Test MSE | Test MAE |
|---|---:|---:|
| Transformer-style no cycle | 0.480718 | 0.456953 |
| CycleiTransformer | 0.475241 | 0.452285 |

CycleiTransformer improves over the no-cycle Transformer-style baseline on both MSE and MAE. The improvement is small, but it supports the paper's claim that the cycle mechanism can be integrated into stronger forecasting architectures.

##### Interpretation

Across both MLP and Transformer-style backbones, adding the learned cycle component improves performance, although the gains are smaller than in the Linear setting. In our experiments, CycleNet-Linear remains the strongest and most stable variant.

Overall, these results suggest that the main benefit of CycleNet comes from explicitly modeling periodic structure, not simply from increasing model complexity.