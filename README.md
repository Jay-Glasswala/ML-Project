# Cost-Aware Credit Card Default Prediction
**ES 335: Machine Learning Course Project**  
**Indian Institute of Technology (IIT)** &bull; **Fall 2026**

[![Tests](https://img.shields.io/badge/pytest-passing-success.svg)](#)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](#)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
[![GitHub Pages](https://img.shields.io/badge/Interactive%20Site-GitHub%20Pages-teal.svg)](https://Jay-Glasswala.github.io/ML-Project/)

---

## Project Overview

In consumer lending, predicting client default is characterized by two fundamental properties:
1. **Severe Class Imbalance:** Defaulters comprise only $\approx 22.1\%$ of clients. A naive majority-class rule scores **$77.9\%$ accuracy** while catching $0$ defaulters, which is economically disastrous.
2. **Asymmetric Decision Costs:** Missing a defaulter (False Negative) loses the entire unpaid loan balance, while a false alarm (False Positive) causes minor operational friction.

Under Bayes decision theory, the optimal decision threshold is:
$$t^* = \frac{c_{FP}}{c_{FP} + c_{FN}} = \frac{1}{1 + r}, \quad \text{where } r = \frac{c_{FN}}{c_{FP}}$$

When missing a defaulter is $5\times$ more costly than a false alarm ($r = 5$), the optimal decision threshold shifts from the standard $0.50$ down to **$t^* \approx 0.167$**. This project designs an end-to-end, leak-free machine learning system to predict calibrated probabilities, minimize financial loss under asymmetric costs, compare regularizer behavior under extreme collinearity, and audit demographic disparities.

---

## Team Members & Responsibilities

| Student Name | Roll Number | Primary Core Responsibilities |
| :--- | :---: | :--- |
| **Anant Shukla** | `25110030` | Data pipeline engineering, duplicate removal & recoding, schema verification, scikit-learn `Pipeline` leakage prevention, collinearity & VIF analysis, Ridge vs. Lasso regularizer stability, subgroup demographic fairness auditing. |
| **Jay R. Glasswala** | `25110147` | Non-linear margin & ensemble modeling (Random Forest, HistGradientBoosting, RBF-SVM with Platt scaling), equal-frequency probability calibration (ECE), Bayes cost threshold optimization, paired bootstrap hypothesis testing, and static interactive GitHub Pages deployment. |

*Both team members reviewed all code and can reproduce and defend every result end-to-end.*

---

## Key Empirical Findings

All models were tuned via **Stratified 5-Fold Cross-Validation** on the training partition ($N=23,972$) and evaluated once on the sealed held-out test partition ($N=5,993$). $95\%$ Confidence Intervals are computed via **1,000 stratified bootstrap resamples**.

| Model Architecture | PR-AUC [95% CI] | Log-Loss [95% CI] | Brier Score | ECE (Quantile) | Expected Cost @ $r=5$ [95% CI] | Recall @ $P \ge 0.50$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Best)** | **0.5548** [0.528, 0.582] | **0.4328** [0.420, 0.446] | **0.1348** | **0.0114** | **0.5648** [0.527, 0.603] | **65.1%** |
| **Random Forest** | 0.5505 [0.523, 0.578] | 0.4346 [0.422, 0.447] | 0.1352 | 0.0100 | 0.5810 [0.543, 0.620] | 63.8% |
| **RBF-Kernel SVM (Platt)** | 0.4934 [0.465, 0.523] | 0.4614 [0.450, 0.473] | 0.1448 | 0.0295 | 0.6888 [0.650, 0.728] | 56.4% |
| **Elastic-Net Logistic** | 0.5037 [0.475, 0.533] | 0.4670 [0.457, 0.477] | 0.1466 | 0.0516 | 0.6980 [0.660, 0.738] | 57.2% |
| **Ridge Logistic (L2)** | 0.5034 [0.475, 0.533] | 0.4671 [0.457, 0.477] | 0.1466 | 0.0514 | 0.6993 [0.661, 0.739] | 57.0% |
| **Lasso Logistic (L1)** | 0.5035 [0.475, 0.533] | 0.4671 [0.457, 0.477] | 0.1466 | 0.0518 | 0.6998 [0.662, 0.739] | 57.1% |
| **Plain Logistic (Reference)** | 0.5034 [0.475, 0.533] | 0.4672 [0.457, 0.477] | 0.1467 | 0.0520 | 0.7022 [0.664, 0.742] | 57.0% |
| **Majority Class (Baseline)** | 0.2213 [0.211, 0.232] | 7.9750 | 0.2213 | 0.2213 | 1.1063 [1.053, 1.160] | 0.0% |

### Paired Bootstrap Hypothesis Testing
* **$\Delta$ PR-AUC (HistGradientBoosting vs. Plain Logistic):** $+0.0514$ ($95\%$ CI: `[+0.0353, +0.0685]`). Because the confidence interval strictly excludes zero, the non-linear ensemble delivers a statistically significant improvement over linear baselines.
* **$\Delta$ Expected Cost ($r=5$):** $-0.1374\,c_{FP}$ ($95\%$ CI: `[-0.1699, -0.1065]`). Demonstrates a statistically significant cost reduction per client.

### Collinearity Analysis & Regularizer Behavior
Variance Inflation Factor (VIF) on training fold data:
* `BILL_AMT1`: $11.49$ &bull; `BILL_AMT2`: $17.08$ &bull; `BILL_AMT3`: $10.33$
* `BILL_AMT4`: $13.76$ &bull; `BILL_AMT5`: $16.03$ &bull; `BILL_AMT6`: $9.97$
* Adjacent-month bill correlations reach $r = 0.955$. Ridge ($L_2$) shrinks correlated feature weights smoothly together, while Lasso ($L_1$) selects `BILL_AMT1` (the most recent balance) and drives intermediate redundant months to zero.

---

## Subgroup Demographic Fairness Audit

Protected attributes (`SEX`, `MARRIAGE`) were **strictly excluded** from model inputs $X$. At threshold $t^* \approx 0.167$ ($r=5$), test set error rates were audited:
* **By Sex:**
  * Male ($N=2,332$): False Alarm Rate $= 34.3\%$ [$32.2\%, 36.4\%$] &bull; Missed Defaulters $= 29.5\%$ [$25.9\%, 33.5\%$]
  * Female ($N=3,661$): False Alarm Rate $= 32.4\%$ [$30.7\%, 34.0\%$] &bull; Missed Defaulters $= 26.4\%$ [$23.4\%, 29.6\%$]
* **By Education:**
  * Graduate School ($N=2,090$): False Alarm Rate $= 29.8\%$ &bull; Missed Defaulters $= 34.7\%$
  * University ($N=2,769$): False Alarm Rate $= 35.9\%$ &bull; Missed Defaulters $= 23.8\%$
  * High School ($N=1,044$): False Alarm Rate $= 35.6\%$ &bull; Missed Defaulters $= 26.9\%$
  * Others ($N=90$): Flagged as statistically unreliable ($N < 100$).
* **Policy Note:** Group-specific decision thresholds are deliberately *not* used to avoid explicit disparate treatment.

---

## Interactive Web Application (GitHub Pages)

The project includes a static, zero-backend interactive web application hosted on GitHub Pages:
* **Interactive Tool 1 (Cost-Ratio Slider):** Dragging the slider ($r \in [1, 10]$) recalculates $t^* = 1/(1+r)$ and re-evaluates all 5,993 test records in JavaScript in **$< 4\text{ ms}$**, live-updating the 2x2 confusion matrix, expected cost, and trade-off cards.
* **Interactive Tool 2 (Fairness Auditor):** Dynamically filters and displays error rates across demographic groups with error bars.
* **Performance Benchmark:** `web/data/test_results.json` is compressed to **105.9 KB (gzip)**, enabling sub-100 ms initial load times worldwide.

---

## How to Reproduce

### 1. Clone the repository
```bash
git clone https://github.com/Jay-Glasswala/ML-Project.git
cd ML-Project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run automated unit & leakage tests
```bash
pytest -v
```

### 4. Execute the complete end-to-end pipeline
```bash
python run_all.py
```

### 5. Launch local static web preview
```bash
cd web
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

---

## Responsible AI Disclosure

In accordance with course academic integrity guidelines:
* **AI Tools Utilized:** Claude (Anthropic), Gemini (Google), and ChatGPT (OpenAI) were used for brainstorming experimental design, writing unit test boilerplate, debugging ConvergenceWarnings, and CSS layout optimization.
* **Attribution & Independent Verification:** Every mathematical derivation (Bayes decision threshold, ECE quantile formulation), data transformation, and model output was independently executed, checked, and validated via deterministic random seeds (`random_state=42`).
* **Zero Private Data / Credentials:** No proprietary credentials, API keys, or private identifiers are stored in this repository. All artifacts originate from the public UCI dataset.
