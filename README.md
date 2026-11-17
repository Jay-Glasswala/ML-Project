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
| **Jay R. Glasswala** | `25110147` | Non-linear margin & ensemble modeling (Random Forest, HistGradientBoosting, RBF-SVM with Platt scaling), 5-fold cross-validation tuning, probability calibration & equal-frequency quantile ECE implementation, Bayes cost threshold optimization ($t^* = \frac{1}{1+r}$), paired bootstrap hypothesis testing, and GitHub Pages deployment. |
| **Anant Shukla** | `25110030` | Baseline and linear model family (majority class, plain, Ridge, Lasso and Elastic-Net logistic regression): cross-validated tuning and coefficient analysis under collinearity (VIF, correlation structure, L1 vs. L2 behavior on the six bill columns); leakage-free scikit-learn `Pipeline` preprocessing (scaling and encoding fit on training folds only, protected attributes excluded); data ingestion, duplicate removal (35 rows) and categorical recoding; subgroup demographic fairness audit; and automated `pytest` test suite. |

*Both team members contributed equally to project discussions, reviewed all code, and can reproduce and defend every result end-to-end.*

---

## Key Empirical Findings

All models were tuned via **Stratified 5-Fold Cross-Validation** on the training partition ($N=23,972$) and evaluated once on the sealed held-out test partition ($N=5,993$). Decision thresholds are not tuned on data: each cost ratio $r$ uses the Bayes rule $t^* = 1/(1+r)$, which is valid because the selected models are well calibrated (quantile ECE $\approx 0.01$ for the ensembles). The final RBF-SVM grid was $(C, \gamma) \in \{(1, \text{scale}), (1, 0.05), (5, \text{scale})\}$ on a stratified 6,000-row subsample, narrowed from the proposal grid for runtime. $95\%$ Confidence Intervals are computed via **1,000 stratified bootstrap resamples**.

| Model Architecture | PR-AUC [95% CI] | Log-Loss [95% CI] | Brier Score | ECE (Quantile) | Expected Cost @ $r=5$ [95% CI] | Recall @ $P \ge 0.50$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Best)** | **0.5548** [0.528, 0.583] | **0.4328** [0.417, 0.447] | **0.1360** | **0.0114** | **0.5648** [0.534, 0.594] | **55.7%** |
| **Random Forest** | 0.5503 [0.524, 0.580] | 0.4346 [0.419, 0.449] | 0.1363 | 0.0097 | 0.5810 [0.550, 0.613] | 56.7% |
| **RBF-Kernel SVM (Platt)** | 0.4934 [0.465, 0.523] | 0.4614 [0.445, 0.477] | 0.1441 | 0.0290 | 0.6888 [0.648, 0.728] | 45.3% |
| **Elastic-Net Logistic** | 0.5037 [0.475, 0.534] | 0.4670 [0.451, 0.482] | 0.1456 | 0.0516 | 0.6980 [0.670, 0.725] | 50.9% |
| **Ridge Logistic (L2)** | 0.5034 [0.475, 0.533] | 0.4671 [0.451, 0.482] | 0.1456 | 0.0514 | 0.6993 [0.671, 0.727] | 50.7% |
| **Lasso Logistic (L1)** | 0.5035 [0.475, 0.534] | 0.4671 [0.451, 0.482] | 0.1456 | 0.0518 | 0.6998 [0.672, 0.728] | 50.8% |
| **Plain Logistic (Reference)** | 0.5034 [0.475, 0.533] | 0.4672 [0.451, 0.482] | 0.1456 | 0.0520 | 0.7022 [0.674, 0.730] | 50.7% |
| **Majority Class (Baseline)** | 0.2213 [0.211, 0.232] | 7.9750 | 0.2213 | 0.2213 | 1.1063 [1.054, 1.162] | 0.0% |

### Paired Bootstrap Hypothesis Testing
* **$\Delta$ PR-AUC (HistGradientBoosting vs. Plain Logistic):** $+0.0514$ ($95\%$ CI: `[+0.0353, +0.0685]`). Because the confidence interval strictly excludes zero, the non-linear ensemble delivers a statistically significant improvement over linear baselines.
* **$\Delta$ Expected Cost ($r=5$):** $-0.1374\,c_{FP}$ ($95\%$ CI: `[-0.1699, -0.1065]`). Demonstrates a statistically significant cost reduction per client.

### Collinearity Analysis & Regularizer Behavior
Variance Inflation Factor (VIF) on training fold data:
* `BILL_AMT1`: $11.49$ &bull; `BILL_AMT2`: $17.08$ &bull; `BILL_AMT3`: $10.33$
* `BILL_AMT4`: $13.76$ &bull; `BILL_AMT5`: $16.03$ &bull; `BILL_AMT6`: $9.97$
* Adjacent-month bill correlations reach $r = 0.955$.
* Fitted bill-amount coefficients (standardized features, CV-selected penalties):

| Model | `BILL_AMT1` | `BILL_AMT2` | `BILL_AMT3` | `BILL_AMT4` | `BILL_AMT5` | `BILL_AMT6` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Plain Logistic | −0.469 | +0.304 | −0.030 | −0.005 | +0.033 | +0.064 |
| Ridge ($C=0.1$) | −0.416 | +0.237 | −0.008 | −0.016 | +0.025 | +0.074 |
| Lasso ($C=1.0$) | −0.442 | +0.258 | −0.005 | **0** | +0.020 | +0.066 |
| Elastic-Net ($C=0.1$, $\ell_1$ ratio $=0.5$) | −0.348 | +0.156 | **0** | **0** | +0.018 | +0.067 |

* **Interpretation:** In every linear model, `BILL_AMT1` and `BILL_AMT2` receive large weights of *opposite sign*, a classic collinearity signature: the model effectively uses the month-over-month change in balance rather than either level. Ridge pulls these two weights toward each other but keeps all six months. Cross-validation selected only a weak Lasso penalty ($C=1.0$), so Lasso zeroes just `BILL_AMT4`; Elastic-Net zeroes `BILL_AMT3` and `BILL_AMT4`. Stronger L1 penalties drive more months to zero (see the regularization paths in `notebooks/eda_and_collinearity.ipynb`), but at the CV-optimal strength all four linear models reach virtually identical test PR-AUC ($\approx 0.503$), so regularization changes the coefficients far more than the predictions.

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

## Data Access Instructions & Provenance

* **Dataset:** Default of Credit Card Clients Dataset
* **Source:** UCI Machine Learning Repository (Yeh and Lien, 2009)
* **Direct Access URL:** [https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Automated Retrieval:** Running `python run_all.py` or `python src/data_loader.py` automatically checks for `data/raw/default of credit card clients.xls` and downloads/unpacks it if not present.
* **Unit of Observation:** Each row represents one individual credit card client at a Taiwanese commercial bank between April 2005 and September 2005, predicting whether they defaulted on their payment in October 2005.

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
* **Website Frontend & Visualization:** Generative AI tools (Claude, Gemini, ChatGPT) were used to assist in building the static web presentation (HTML layout, CSS styling, Chart.js integrations, and event handlers for the interactive cost simulator), as the course focuses on evaluating machine learning understanding and reasoning rather than front-end web development.
* **Machine Learning Coding Assistance:** AI was utilized for programming assistance on standard ML tasks, including suggesting sensible hyperparameter search ranges during cross-validation, providing syntax boilerplate for Scikit-Learn `ColumnTransformer` pipelines, optimizing NumPy array vectorization for the bootstrap resampling loops, and debugging model convergence warnings.
* **Independent Verification & Reasoning:** All core ML decisions—problem formulation, mathematical derivation of the Bayes optimal threshold $t^* = \frac{1}{1+r}$, data cleaning logic (detecting the 35 duplicate client rows and recoding undocumented values), model comparisons, quantile ECE implementation, error analysis, and conclusions—were independently conceptualized, executed, verified, and defended by the project team (Jay Glasswala and Anant Shukla).
* **Results Audit:** Claude (Anthropic) was used to cross-check the README and website figures against the generated metrics files and the fitted model coefficients; the collinearity discussion and results table were corrected to match the actual outputs.
* **Zero Private Data / Credentials:** No private credentials, API keys, or proprietary data are present. All artifacts originate from the public UCI Machine Learning Repository.

---

## References

**Data**
* Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480.
* Yeh, I.-C. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H (CC BY 4.0)

**Methods**
* Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 61–74.
* Elkan, C. (2001). The foundations of cost-sensitive learning. *Proceedings of IJCAI 2001*, 973–978. (Bayes cost-optimal threshold)
* Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B, 67*(2), 301–320.

**Software**
* Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
* NumPy, pandas, SciPy, Matplotlib and pytest (versions in `requirements.txt`).
* Chart.js (https://www.chartjs.org, MIT License) and MathJax 3 (https://www.mathjax.org, Apache 2.0) for the website, loaded from the jsDelivr CDN.
