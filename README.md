# ETTh Anomaly Detection with Facebook Prophet

> Detecting anomalies in electricity transformer oil temperature using Prophet's confidence intervals as a deviation signal.

![Python](https://img.shields.io/badge/python-3.13.9-blue)
[![Prophet](https://img.shields.io/badge/Prophet-1.1%2B-orange.svg)](https://facebook.github.io/prophet/)

---

## 📖 Overview

This project applies Facebook Prophet to the **ETTh (Electricity Transformer Temperature [hourly])** dataset to identify anomalous oil temperature readings in a power transformer. Oil temperature is a critical health indicator for transformers sustained or sudden deviations from expected values can signal equipment degradation, sensor faults, or operational stress.

Rather than building a forecasting model, this work uses Prophet as a **pattern learner**: it fits the normal seasonal + regressor-driven behavior of oil temperature, then flags every historical hour whose actual value falls outside Prophet's predicted confidence interval.

---

## Why Prophet for Anomaly Detection?

Prophet decomposes a time series into interpretable components:

$$y(t) = g(t) + s(t) + h(t) + \sum_{j=1}^{n} \beta_j x_j(t) + \epsilon_t$$

Where:

- $g(t)$ : **Trend**: long-term direction (piecewise-linear or logistic)
- $s(t)$ : **Seasonality**: daily, weekly, yearly patterns (Fourier series)
- $h(t)$ : **Holiday effects**: irregular known events
- $\beta_j x_j(t)$ **External regressors**: additional features
- $\epsilon_t$ **Noise**: what the model cannot explain

For each timestamp, Prophet produces a point prediction $ \hat{y}(t) $ and a confidence interval given by For each timestamp, Prophet produces a point prediction $`\hat{y}(t)`$ and a confidence interval $`\hat{y}_{lower}(t)`$ and $`\hat{y}_{upper}(t)`$ derived from Monte Carlo simulation., derived from Monte Carlo simulation.
**An anomaly is any point falling outside that interval:**

$$a(t) = \begin{cases} 1 & \text{if } y(t) > \hat{y}_{upper}(t) \\\\ 1 & \text{if } y(t) < \hat{y}_{lower}(t) \\\\ 0 & \text{otherwise} \end{cases}$$

---

## 📊 Dataset

The **ETTh dataset* contains two years of hourly readings (Jul 2016 – Jun 2018) from two electricity transformers in a Chinese province.

| Column | Description |
|---|---|
| `date` | Hourly timestamp |
| `HUFL` | High-voltage Useful Load |
| `HULL` | High-voltage Useless Load |
| `MUFL` | Medium-voltage Useful Load |
| `MULL` | Medium-voltage Useless Load |
| `LUFL` | Low-voltage Useful Load |
| `LULL` | Low-voltage Useless Load |
| `OT` | **Oil Temperature** the target |

Source: [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)

---

## 🛠️ Methodology

### 1. Data Preparation
- Loaded ETTh CSV and isolated Transformer 1's columns
- Validated time continuity (17,420 hourly observations, no gaps)
- Renamed columns to Prophet's expected format (`ds`, `y`)

### 2. Multicollinearity Check
Before adding regressors, multicollinearity was assessed using **Variance Inflation Factor (VIF)**:

$$VIF_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the coefficient of determination from regressing feature $j$ against all others. Initial VIF scores revealed severe multicollinearity:

| Feature | VIF (all 6) | VIF (after dropping MUFL, MULL) |
|---|---|---|
| HUFL | 100.42  | 1.11  |
| HULL | 31.23  | 1.20  |
| MUFL | 94.83  | — |
| MULL | 26.41  | — |
| LUFL | 2.32  | 1.26  |
| LULL | 3.96  | 1.27  |

Final regressor set: **HUFL, HULL, LUFL, LULL**.

### 3. Prophet Configuration

```python
Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    interval_width=0.99,            # 99% confidence band
    changepoint_prior_scale=0.05    # default trend flexibility
)
```

### 4. Anomaly Detection
The fitted model predicts on the **same** historical timestamps it was trained on (no train/test split, the goal is pattern conformity, not future prediction). Points outside the 99% confidence band are flagged.

### 5. Visualization
Two complementary views are produced:
- **Time series view**  actual OT, expected line, and confidence band
- **Residual view**  deviation from expected, with anomalies guaranteed to sit outside flat threshold lines

---

## 📈 Results

| Metric | Value |
|---|---|
| Observations analyzed | 17,420 hours |
| Anomalies flagged | ~340 (≈2%) |
| Temperature spikes | ~70 |
| Temperature drops | ~270 |

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Adjust sensitivity

```python
INTERVAL_WIDTH  = 0.99    # 0.90 = sensitive, 0.99 = conservative
USE_REGRESSORS  = True    # False for univariate
TARGET_COLUMN   = 'OT'    # change to monitor a different variable
```

## ⚙️ Key Parameters Explained

| Parameter | Purpose | Recommended |
|---|---|---|
| `interval_width` | Width of confidence band. Higher = wider band = fewer anomalies flagged. | 0.95-0.99 |
| `changepoint_prior_scale` | Trend flexibility. Higher = more responsive to shifts. | 0.05 (default) |
| `seasonality_mode` | `additive` or `multiplicative` | `additive` for OT |
| Regressors | External features known at prediction time | Required if available |

---

## Author

**Muhammad Mahfouz**
GitHub: [@MuhammadMahfouz](https://github.com/MuhammadMahfouz)

---

*If this project was useful to you, consider giving it a ⭐ on GitHub.*
