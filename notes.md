## CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns

__(https://arxiv.org/abs/2409.18479)__


### Required backround on Time Series

Given a time series $x_1, x_2, \dots, x_T$; 

the task is the following:

Given $x_{t-L+1:t}$, predict $x_{t+1:t+H}$

Where:

* **$L$** = input / lookback length
* **$H$** = prediction horizon
* **Univariate** = one variable over time
* **Multivariate** = multiple variables over time (e.g., electricity, traffic sensors, weather features)

Classic time series thinking decomposes a signal as: 

$x_t = \text{trend}_t + \text{seasonality}_t + \text{residual}_t$

| Component | Meaning | Example |
| :--- | :--- | :--- |
| **Trend ($T_t$)** | Long-term direction | Electricity◊ demand slowly increasing |
| **Seasonality / Cycle ($S_t$)** | Repeating pattern | Daily, weekly, or yearly rhythms |
| **Residual ($R_t$ or $\epsilon_t$)** | What remains after predictable structure | Noise, shocks, irregular behavior |

Quick recap of useful Time Series Properties for CycleNet:

**Periodicity:** A time series is periodical if values repeat every fixed interval. 

$x_t \approx x_{t-p}$, where $p$ is the period.

**Residual Forecasting:** Instead of forecasting the entire signal, the structured part e.g. periodicity can be removed first. 

$$\text{residual}_t = x_t - \text{cycle}_t$$

**Time Series Normalization:** 

Distribution shift can affect a lot the forecasting result.

**Long-Term Time Series Forecasting:**

The principle enabling long-horizon prediction (spanning several days or months) lies in understanding inherent periodicity rather than relying solely on recent temporal information.


--- 


Introducing CycleNet: 

**Task Introduction:**
Unlike short-term forecasting, long-term predictions
cannot rely solely on recent temporal information (including means, trends, etc.). For instance, a
user’s electricity consumption thirty days ahead not only correlates with their consumption patterns
in the past few days.
In such cases, long-term dependencies, or in other words, underlying stable periodicity within the
data, serve as the practical foundation for conducting long-term prediction. 




### From Time Series Windows to CycleNet Training

Given a long time series,

$$
x_1, x_2, \dots, x_T,
$$

the forecasting problem is converted into many supervised learning examples using sliding windows.

Each training example has the form:

$$
x_{t-L+1:t} \rightarrow x_{t+1:t+H}
$$

where:

* $x_{t-L+1:t}$ is the **lookback window**, i.e. the past $L$ observations available up to time $t$;
* $x_{t+1:t+H}$ is the **forecast horizon**, i.e. the next $H$ future observations that the model must predict.

For multivariate time series, each observation $x_t$ is a vector of $C$ variables/channels. Therefore, one input window has shape:

$$
X_{\text{input}} \in \mathbb{R}^{L \times C}
$$

and the corresponding target has shape:

$$
Y_{\text{target}} \in \mathbb{R}^{H \times C}.
$$

During training, the model does not process only one window at a time. Instead, several windows are grouped into a **batch**. For example, with batch size $B=32$, the input batch has shape:

$$
X \in \mathbb{R}^{32 \times L \times C}
$$

and the target batch has shape:

$$
Y \in \mathbb{R}^{32 \times H \times C}.
$$

This means that the model forecasts the future for **32 different input windows in parallel**.

---

### Training Objective

For each batch, CycleNet performs the following steps:

1. Take a batch of input windows:

$$
X \in \mathbb{R}^{B \times L \times C}
$$

2. For each window, use the learned cycle representation to identify the corresponding periodic component.

3. Remove the past cycle component from the input:

$$
X_{\text{residual}} = X - C_{\text{past}}
$$

4. Pass the residualized input through the forecasting backbone, such as a Linear or MLP model:

$$
\hat{R}_{\text{future}} = f_{\theta}(X_{\text{residual}})
$$

5. Add back the future cycle component to obtain the final forecast:

$$
\hat{Y} = \hat{R}_{\text{future}} + C_{\text{future}}
$$

6. Compare the forecast $\hat{Y}$ with the true future values $Y$ using Mean Squared Error:

$$
\mathcal{L}_{\text{MSE}} =
\frac{1}{BHC}
\sum_{b=1}^{B}
\sum_{h=1}^{H}
\sum_{c=1}^{C}
\left(\hat{y}_{b,h,c} - y_{b,h,c}\right)^2
$$

The goal of training is to minimize this loss.

---

### Intuition

CycleNet does not ask the forecasting model to predict the full raw signal directly. Instead, it separates the problem into two parts:

$$
x_t = \text{cycle}_t + \text{residual}_t.
$$

The cyclic component captures stable periodic behavior, such as daily, weekly, or yearly patterns. The residual component captures what remains after removing this predictable structure.

Therefore, the backbone model only needs to forecast the residual:

$$
\text{residual}_t = x_t - \text{cycle}_t.
$$

After the residual forecast is produced, CycleNet adds the future cycle component back to reconstruct the final prediction.

In simple terms:

> CycleNet learns the repeated pattern, removes it from the input, forecasts what is left, and then adds the expected future pattern back.

This is the main idea behind **Residual Cycle Forecasting**: long-horizon prediction becomes easier when stable periodic structure is explicitly modeled rather than being learned implicitly by the forecasting backbone.


### Why Residual Cycle Forecasting Should Help

A plain Linear or MLP forecasting model learns a direct mapping:

$$
\hat{Y} = f_\theta(X)
$$

where the model must learn everything at once: trend, periodicity, local variation, noise, and irregular changes.

CycleNet changes this problem by explicitly separating the periodic component from the residual component:

$$
X_{\text{residual}} = X - C_{\text{past}}
$$

The backbone then predicts only the future residual:

$$
\hat{R}_{\text{future}} = f_\theta(X_{\text{residual}})
$$

and the final prediction is reconstructed by adding back the future cycle component:

$$
\hat{Y} = \hat{R}_{\text{future}} + C_{\text{future}}
$$

The reason this can help is that periodicity is often one of the most stable and predictable structures in time series data. If the model can explicitly capture repeated patterns, the remaining residual signal should be easier to forecast.

In other words, CycleNet reduces the burden on the forecasting backbone.

---

### What CycleNet Adds Compared to a Plain Linear / MLP Baseline

A standard Linear or MLP baseline directly receives the raw input window and predicts the future horizon:

$$
X \rightarrow \hat{Y}
$$

CycleNet keeps the backbone simple, but wraps it with a learned cycle mechanism:

$$
X \rightarrow X_{\text{residual}} \rightarrow \hat{R}_{\text{future}} \rightarrow \hat{Y}
$$

Therefore, the main contribution is not a more complex backbone, but a different way of presenting the forecasting problem to the backbone.

The backbone learns deviations from periodic behavior instead of learning the entire signal from scratch.

This is important because it allows CycleNet to test the hypothesis that:

> Explicitly modeling periodic patterns can improve forecasting even with simple models such as Linear layers or shallow MLPs.

---

### What Should Improve?

If CycleNet works as intended, the main metrics that should improve are forecasting errors such as:

* **MSE**: Mean Squared Error
* **MAE**: Mean Absolute Error

Lower MSE and MAE mean that the predicted future values are closer to the true future values.

In addition to accuracy, CycleNet should also be evaluated in terms of:

* performance across different prediction horizons;
* robustness across datasets;
* sensitivity to the chosen cycle length;
* whether the improvement comes from the cycle module rather than from the backbone itself.

The key experimental question is therefore:

> Does adding explicit cycle modeling consistently improve simple forecasting backbones, or does it only help on datasets with strong periodic structure?

