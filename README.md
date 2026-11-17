# Cost-Aware Credit Card Default Prediction
**ES 335 Machine Learning – Course Project**  
IIT Gandhinagar, Fall 2026

Website: https://Jay-Glasswala.github.io/ML-Project/  
[![tests](https://github.com/Jay-Glasswala/ML-Project/actions/workflows/tests.yml/badge.svg)](https://github.com/Jay-Glasswala/ML-Project/actions/workflows/tests.yml)  
Dataset: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (CC BY 4.0)

---

## What this project is about

We use the UCI credit card dataset (about 30,000 clients from Taiwan, 2005) to predict whether a client will default on their next payment. Two things make this harder than a normal classification problem:

1. **The classes are imbalanced.** Only about 22% of clients default. If you just predict "no default" for everyone, you get 77.9% accuracy but you catch zero defaulters. So accuracy is not a good metric here.
2. **The two mistakes don't cost the same.** If the bank misses a defaulter it loses the unpaid balance, but a false alarm only costs a bit of extra checking.

If the cost of a missed defaulter is $r$ times the cost of a false alarm, and the model's probabilities are calibrated, the best threshold is

$$t^* = \frac{c_{FP}}{c_{FP} + c_{FN}} = \frac{1}{1 + r}$$

So for $r = 5$ we should flag a client when $p \ge 0.167$, not 0.5.

In this project we:
* train and compare 8 models without leaking test data,
* check whether their probabilities are calibrated,
* pick thresholds using the cost ratio,
* look at how Ridge and Lasso handle the six bill-amount columns (which are very correlated),
* check whether error rates are different across sex, education and marital status.

---

## Team

| Name | Roll No. | What they worked on |
| :--- | :---: | :--- |
| **Jay R. Glasswala** | `25110147` | The non-linear models (Random Forest, HistGradientBoosting, RBF-SVM with Platt scaling), 5-fold CV tuning, calibration and the quantile ECE metric, the cost threshold $t^* = 1/(1+r)$, the paired bootstrap test, and the GitHub Pages site. |
| **Anant Shukla** | `25110030` | The baseline and linear models (majority class, plain, Ridge, Lasso and Elastic-Net logistic regression), their CV tuning and the coefficient analysis on the correlated bill columns (VIF, correlations, L1 vs. L2); the leakage-free preprocessing `Pipeline` (scaling and encoding fit only on training folds, SEX and MARRIAGE left out); loading the data, removing the 35 duplicates and recoding bad category codes; the fairness audit; and the `pytest` tests. |

We both took part in the discussions, reviewed each other's code, and can run and explain every result.

---

## Results

We tuned every model with stratified 5-fold cross-validation on the training set (23,972 clients) and then scored each one once on the test set (5,993 clients). We didn't tune the threshold on data; we just used $t^* = 1/(1+r)$, which works because the best models are well calibrated (ECE around 0.01). For the SVM we only tried three settings, $(C, \gamma) \in \{(1, \text{scale}), (1, 0.05), (5, \text{scale})\}$, on a 6,000-row subsample, because it was too slow on the full data. The 95% confidence intervals come from 1,000 bootstrap resamples of the test set.

| Model | PR-AUC [95% CI] | Log-Loss [95% CI] | Brier | ECE | Cost at $r=5$ [95% CI] | Recall at Precision ≥ 0.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (best)** | **0.5548** [0.528, 0.583] | **0.4328** [0.417, 0.447] | **0.1360** | **0.0114** | **0.5648** [0.534, 0.594] | **55.7%** |
| Random Forest | 0.5503 [0.524, 0.580] | 0.4346 [0.419, 0.449] | 0.1363 | 0.0097 | 0.5810 [0.550, 0.613] | 56.7% |
| RBF-Kernel SVM (Platt) | 0.4934 [0.465, 0.523] | 0.4614 [0.445, 0.477] | 0.1441 | 0.0290 | 0.6888 [0.648, 0.728] | 45.3% |
| Elastic-Net Logistic | 0.5037 [0.475, 0.534] | 0.4670 [0.451, 0.482] | 0.1456 | 0.0516 | 0.6980 [0.670, 0.725] | 50.9% |
| Ridge Logistic (L2) | 0.5034 [0.475, 0.533] | 0.4671 [0.451, 0.482] | 0.1456 | 0.0514 | 0.6993 [0.671, 0.727] | 50.7% |
| Lasso Logistic (L1) | 0.5035 [0.475, 0.534] | 0.4671 [0.451, 0.482] | 0.1456 | 0.0518 | 0.6998 [0.672, 0.728] | 50.8% |
| Plain Logistic | 0.5034 [0.475, 0.533] | 0.4672 [0.451, 0.482] | 0.1456 | 0.0520 | 0.7022 [0.674, 0.730] | 50.7% |
| Majority Class (baseline) | 0.2213 [0.211, 0.232] | 7.9750 | 0.2213 | 0.2213 | 1.1063 [1.054, 1.162] | 0.0% |

Cost is per client, in units of the cost of one false alarm.

**Is the best model really better than plain logistic regression?** We compared them on the same 1,000 bootstrap samples:
* PR-AUC difference: +0.0514 (95% CI [+0.0353, +0.0685])
* Cost difference at $r=5$: −0.1374 per client (95% CI [−0.1699, −0.1065])

Both intervals don't include zero, so the difference is significant.

### What happens for other cost ratios?

We don't really know the true cost ratio, so here are the test-set results for a few values of $r$ (cost per client, in units of one false alarm):

| $r$ | $t^*$ | HGB cost | HGB recall | HGB false alarm rate | RF cost | Ridge cost | Flag nobody | Flag everybody |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.500 | 0.182 | 36% | 5% | 0.182 | 0.188 | 0.221 | 0.779 |
| 2 | 0.333 | 0.309 | 51% | 12% | 0.306 | 0.322 | 0.443 | 0.779 |
| 3 | 0.250 | 0.415 | 59% | 18% | 0.412 | 0.443 | 0.664 | 0.779 |
| 5 | 0.167 | 0.565 | 72% | 33% | 0.581 | 0.699 | 1.106 | 0.779 |
| 10 | 0.091 | 0.717 | 92% | 69% | 0.722 | 0.808 | 2.213 | 0.779 |

* HistGradientBoosting and Random Forest are very close for every $r$ (RF is even slightly cheaper at $r=2$ and $r=3$), so we wouldn't call one clearly better than the other.
* As $r$ grows, the threshold drops and the model flags more and more people: at $r=10$ it catches 92% of defaulters but also flags 69% of good clients.
* At $r=10$ the models only just beat "flag everybody" (HGB 0.717 vs. 0.779), and Ridge (0.808) is actually *worse* than flagging everybody. So the model is most useful for moderate cost ratios.

**Did our small tuning grid miss anything?** Our HistGradientBoosting grid only had 4 settings, so we also ran a random search over 20 settings from a much wider grid (learning rate, number of trees, leaf size, L2 penalty). The best one had CV log-loss 0.4261 vs. 0.4265 for ours, which is much smaller than the fold-to-fold spread (std ≈ 0.0047). So the small grid didn't cost us anything noticeable.

### The six bill columns (collinearity)

The bill amounts for neighbouring months are very similar (correlation up to 0.955), and the VIFs on the training data are high:

`BILL_AMT1` 11.49 · `BILL_AMT2` 17.08 · `BILL_AMT3` 10.33 · `BILL_AMT4` 13.76 · `BILL_AMT5` 16.03 · `BILL_AMT6` 9.97

These are the bill-column weights of our fitted linear models (features are standardized, penalties chosen by CV):

| Model | `BILL_AMT1` | `BILL_AMT2` | `BILL_AMT3` | `BILL_AMT4` | `BILL_AMT5` | `BILL_AMT6` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Plain Logistic | −0.469 | +0.304 | −0.030 | −0.005 | +0.033 | +0.064 |
| Ridge ($C=0.1$) | −0.416 | +0.237 | −0.008 | −0.016 | +0.025 | +0.074 |
| Lasso ($C=1.0$) | −0.442 | +0.258 | −0.005 | **0** | +0.020 | +0.066 |
| Elastic-Net ($C=0.1$, l1_ratio $=0.5$) | −0.348 | +0.156 | **0** | **0** | +0.018 | +0.067 |

What we noticed:
* In every model `BILL_AMT1` gets a big negative weight and `BILL_AMT2` a big positive one. Because the two columns are almost the same, the model is basically using the *change* in balance from August to September.
* Ridge pulls these two weights closer together but keeps all six months.
* CV picked a weak penalty for Lasso ($C=1.0$), so it only set `BILL_AMT4` to zero. Elastic-Net set `BILL_AMT3` and `BILL_AMT4` to zero.
* All four linear models got almost the same PR-AUC (about 0.503). So regularization changed the weights a lot but the predictions hardly at all.

To check which of these weights we can actually trust, we refit the models on 100 bootstrap samples of the training data (`src/robustness_checks.py`):

| Column | Ridge $C=0.1$: mean weight [95% range] | Ridge: % of refits positive | Lasso $C=1.0$: % of refits at zero | Lasso $C=0.01$: % of refits at zero |
| :--- | :---: | :---: | :---: | :---: |
| `BILL_AMT1` | -0.405 [-0.60, -0.25] | 0% | 0% | 0% |
| `BILL_AMT2` | +0.212 [-0.02, +0.47] | 94% | 3% | 100% |
| `BILL_AMT3` | +0.000 [-0.23, +0.20] | 52% | 11% | 97% |
| `BILL_AMT4` | -0.011 [-0.19, +0.16] | 47% | 5% | 100% |
| `BILL_AMT5` | +0.034 [-0.21, +0.27] | 61% | 9% | 100% |
| `BILL_AMT6` | +0.069 [-0.09, +0.26] | 78% | 5% | 100% |

The `BILL_AMT1` and `BILL_AMT2` weights are stable. `BILL_AMT3` to `BILL_AMT5` flip sign in about half the refits, so we shouldn't read anything into them. With a strong L1 penalty ($C=0.01$), Lasso keeps only `BILL_AMT1` almost every time, which is what we expected from Lasso, but CV preferred the weaker penalty because the other months still help log-loss a little.

### Threshold and class weight checks

Both checks only use out-of-fold predictions on the training set (`src/robustness_checks.py`), so the test set isn't touched.
* **Does the formula give the right threshold?** For HistGradientBoosting, the threshold with the lowest cost was 0.17 for $r=5$ (formula: 0.167), 0.09 for $r=10$ (formula: 0.091) and 0.36 for $r=2$ (formula: 0.333). Using the formula instead cost less than 0.002 extra per client in every case.
* **Class weights vs. moving the threshold:** We also tried `class_weight='balanced'`. For HistGradientBoosting, the average predicted probability went up from 0.22 to 0.42 (the real default rate is 0.22) and ECE went from 0.006 to 0.196, so the probabilities were no longer calibrated. At $r=5$ the normal model with $t^*$ cost 0.544 per client, and the weighted model at threshold 0.5 cost 0.566. For Ridge it went the other way: the weighted model at 0.5 was cheaper (0.620 vs. 0.694), but its ECE was 0.235. We kept the normal models because our best model does better that way, and because calibrated probabilities let us use the formula for any $r$.

---

## Error analysis

We looked at the mistakes HistGradientBoosting makes on the test set at $t^* = 0.167$ (`src/error_analysis.py`, results saved in `data/processed/error_analysis.json`):

| Group | N | Late in Sep. | Avg. months late (out of 6) | Median credit limit (NT$) | Median Sep. bill / limit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Caught defaulters (TP) | 958 | 68% | 2.74 | 80,000 | 0.48 |
| False alarms (FP) | 1545 | 39% | 1.39 | 90,000 | 0.12 |
| Missed defaulters (FN) | 368 | 3% | 0.10 | 150,000 | 0.55 |
| Correct non-defaulters (TN) | 3122 | 2% | 0.07 | 180,000 | 0.28 |

* **False alarms:** 39% of them were late in September and then paid anyway. The model leans very heavily on September's repayment status (`PAY_1`): shuffling it drops PR-AUC by 0.205, while the next most important feature only drops it by 0.019. So one recent delay is enough to push a client over the low threshold.
* **Missed defaulters:** 91% of them were never late in any of the six months. They have higher limits than the defaulters we catch and use more of their limit than the good clients, but nothing in their payment history really stands out. We think these misses are a limit of the data, not something more tuning would fix.

---

## Fairness check

SEX and MARRIAGE were not given to the model. We only used them afterwards to compare error rates. At $t^* \approx 0.167$ ($r=5$) on the test set:
* **By sex:**
  * Male (N = 2,332): false alarm rate 34.3% [32.2%, 36.4%], missed defaulters 29.5% [25.9%, 33.5%]
  * Female (N = 3,661): false alarm rate 32.4% [30.7%, 34.0%], missed defaulters 26.4% [23.4%, 29.6%]
* **By education:**
  * Graduate school (N = 2,090): false alarm rate 29.8%, missed defaulters 34.7%
  * University (N = 2,769): false alarm rate 35.9%, missed defaulters 23.8%
  * High school (N = 1,044): false alarm rate 35.6%, missed defaulters 26.9%
  * Others (N = 90): too few clients to say anything reliable.

We decided not to use different thresholds for different groups, since that would mean treating people differently based on a protected attribute.

---

## Website

The website is a static page on GitHub Pages (no backend). It has three interactive parts:
* **Cost slider:** you pick a model and a cost ratio $r$ from 1 to 10. The page computes $t^* = 1/(1+r)$, re-scores all 5,993 test clients in the browser, and updates the confusion matrix, cost and recall.
* **Fairness view:** shows false alarm and missed-defaulter rates for each sex, education or marital-status group at the current threshold.
* **Check a client:** fill in one client's details, or load a real test client (one of 13 types such as "overpaid", "inactive card" or "late only in September", or a random one, optionally only defaulters / non-defaulters / clients the model gets wrong), to get the default probability from HistGradientBoosting and Ridge, the flag / don't-flag decision at the slider's cost ratio, and a simple what-if showing which inputs move the probability most. The models run in the browser: `src/export_predictor.py` saves the fitted models to `web/data/predictor.json` and `web/predictor.js` evaluates them. `tests/test_predictor_parity.py` checks that the browser code gives the same probability as scikit-learn for all 5,993 test clients (it uses `node` if installed), that every example client really fits its type, and that the random-client data (`web/data/test_clients.json`) matches `test.csv`. After loading a real client, the page shows what actually happened to them and whether the model was right. This is a demo of the model, not a lending tool.

It reads the saved test predictions from `web/data/test_results.json` (about 106 KB gzipped).

---

## Data

* **Dataset:** Default of Credit Card Clients, UCI Machine Learning Repository (Yeh and Lien, 2009)
* **Link:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
* **License:** CC BY 4.0
* **Getting the data:** `python run_all.py` (or `python src/data_loader.py`) looks for `data/raw/default of credit card clients.xls`. If it isn't there, it first checks for a downloaded `.zip` in `data/raw/` and unpacks it, otherwise it downloads the data from UCI (`download_raw_data` in `src/data_loader.py`). If the download fails (for example, no internet), download the dataset from the link above and put the `.xls` or the `.zip` in `data/raw/` yourself. Without the data, the data tests are skipped with this message instead of failing.
* **Odd and extreme values** (`check_bill_values` in `src/data_loader.py`, saved in `summary_metrics.json`). We counted these but didn't remove them, because they look like real account situations rather than errors:
  * 1,930 clients have at least one negative bill (lowest: -339,603 NT$). A negative bill most likely means the client paid more than they owed. Their default rate is 16.5%, lower than the overall 22.1%.
  * 3,931 clients had a bill above their credit limit in at least one month (343 were above 1.5× the limit). Their default rate is 30.1%.
  * 836 clients have no bill in any of the six months.
  * The money columns are very skewed (skewness about 3 for bills and 11–30 for payments; largest bill 1,664,089 NT$).
  * Does the skew hurt the linear models? We tried a signed log transform, $\text{sign}(x)\log(1+|x|)$, before Ridge (`src/robustness_checks.py`). CV log-loss improved from 0.4659 to 0.4546, but PR-AUC didn't change (0.504 vs. 0.503). So the log transform makes Ridge's probabilities a bit better but doesn't help it find more defaulters, and Ridge is still well behind the tree models. The tree models aren't affected by this kind of transform at all. We left the main pipeline as it was, but it's something we would add next time.
* **One row** = one credit card client at a Taiwanese bank. The inputs cover April to September 2005, and the target is whether they defaulted in October 2005.

---

## How to run it

```bash
# 1. Get the code
git clone https://github.com/Jay-Glasswala/ML-Project.git
cd ML-Project

# 2. Install the packages
pip install -r requirements.txt

# 3. Run the tests (data checks, leakage checks, website predictor = scikit-learn)
pytest -v

# 4. Run everything
python run_all.py            # full run, including CV tuning
SKIP_CV=1 python run_all.py  # faster: reuse the models saved by the last full run

# 5. Look at the website locally
cd web
python -m http.server 8000   # then open http://localhost:8000
```

You can also run the error analysis and robustness checks on their own with `python -m src.error_analysis` and `python -m src.robustness_checks`.

**Optional: Node.js.** One test (`test_predictor_js_matches_sklearn`) runs the website's JavaScript predictor with Node.js to check it against scikit-learn. Node isn't a Python package, so it's not in `requirements.txt`; install it from https://nodejs.org (or `conda install -c conda-forge nodejs`). Without it, that test is skipped and the same maths is still checked in Python.

**Automatic check on GitHub:** `.github/workflows/tests.yml` runs `python run_all.py` and then `pytest` (with Node.js) on GitHub's servers after every push to `main`. The badge at the top shows whether the last run passed.

**Notebook:** `notebooks/eda_and_collinearity.ipynb` has the correlation and VIF plots, the Ridge/Lasso regularization paths and the feature-importance chart. It reads the files that `run_all.py` saves, so run that first.

---

## Use of AI tools

The course allows AI tools as long as we disclose how we used them. **The full details are in [AI_USE.md](AI_USE.md).** In short:

* **AI helped with:** building the website (HTML, CSS, charts and slider code) and later fixing its fonts and phone layout; standard ML coding help (hyperparameter ranges, pipeline boilerplate, faster bootstrap loops, convergence warnings); committing and pushing some of our early work to GitHub; and, with Claude (Anthropic), cross-checking our write-up against our output files (which found and corrected claims that didn't match our results), writing `src/error_analysis.py`, `src/robustness_checks.py`, the bill-value checks and data download, the "Check a client" tool and its test, the GitHub test workflow, the new notebook sections, and rewriting the README and website in simpler language.
* **We did ourselves:** the project idea and proposal, the original pipeline (data loading and cleaning, preprocessing, the eight models, evaluation, bootstrap, fairness audit, first tests and website content), the derivation of $t^* = 1/(1+r)$, the data cleaning decisions, the evaluation design, interpreting the results and writing the conclusions. We chose which AI suggestions to keep, ran the pipeline and tests ourselves, and reviewed the AI-written code and results, which we can explain.
* **No private data:** there are no passwords, API keys or private data in this repo. All data comes from the public UCI repository.

---

## References

**Data**
* Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480.
* Yeh, I.-C. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H (CC BY 4.0)

**Methods**
* Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 61–74.
* Elkan, C. (2001). The foundations of cost-sensitive learning. *Proceedings of IJCAI 2001*, 973–978. (cost-based threshold)
* Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B, 67*(2), 301–320.

**Software**
* Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
* NumPy, pandas, SciPy, Matplotlib and pytest (versions in `requirements.txt`).
* Chart.js (https://www.chartjs.org, MIT License) and MathJax 3 (https://www.mathjax.org, Apache 2.0), loaded from the jsDelivr CDN.
