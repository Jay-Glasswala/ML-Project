# Use of AI tools – ES 335 Course Project

**Project:** Cost-Aware Credit Card Default Prediction  
**Team:** Jay R. Glasswala (25110147) and Anant Shukla (25110030)

The course allows AI tools as long as we disclose how we used them. This section lists what AI helped with and what we did ourselves.

## What AI helped with

**Website (Claude, Gemini, ChatGPT)**
* Building the page: HTML layout, CSS, the Chart.js charts and the JavaScript for the cost slider and the fairness view.
* Later: making fonts and spacing consistent and fixing the layout on phones.

**ML code (Claude, Gemini, ChatGPT)**
* Suggesting hyperparameter ranges for cross-validation.
* Boilerplate for the scikit-learn `ColumnTransformer` / `Pipeline`.
* Speeding up the bootstrap loops with NumPy.
* Fixing convergence warnings.

**Pushing to GitHub**
* An AI agent was used to commit and push some of our early work to GitHub, and it wrote those commit messages.

**Checking and extending the project (Claude, Anthropic)**
* **Cross-checking our write-up against our own output files.** This found claims that didn't match our results: the Lasso/Ridge description, an outdated results table, and an error-analysis and feature-importance section that wasn't computed by any code. These were corrected to match the actual outputs.
* **Writing these files:**
  * `src/error_analysis.py` (error analysis and permutation importance)
  * `src/robustness_checks.py` (threshold check, class weights vs. threshold shifting, bootstrap stability of the bill weights, log transform, wider tuning grid)
  * `check_bill_values` and the automatic data download in `src/data_loader.py`
* **Building the "Check a client" tool:** `src/export_predictor.py`, `web/predictor.js`, `web/check_client.js` and `tests/test_predictor_parity.py`.
* **Adding** the GitHub Actions test workflow and the new notebook sections.
* **Rewriting** the README and website text in simpler language.

## What we did ourselves
* **The project idea and proposal:** the research question, the dataset, the cost-based framing, and the choice of models and metrics.
* **The original pipeline:** data loading and cleaning, the preprocessing `Pipeline`, training and tuning the eight models, the evaluation and bootstrap code, the fairness audit, the first tests and the first version of the website content (see the [Team section of the README](README.md#team) for who did which part).
* **The derivation of the threshold** $t^* = 1/(1+r)$.
* **The data cleaning decisions:** finding the 35 duplicate rows, recoding the undocumented EDUCATION/MARRIAGE codes, renaming `PAY_0`, and leaving SEX and MARRIAGE out of the model.
* **The modelling and evaluation design:** stratified 80/20 split, 5-fold CV on log-loss, quantile ECE, bootstrap confidence intervals and the paired test.
* **Interpreting the results and writing the conclusions.**
* **Deciding what to keep:** we chose which AI suggestions to add (for example the prediction tool and which version of it), ran the pipeline and the tests ourselves, checked the website, and decided what goes into the submission.
* **Checking the AI's work:** we reviewed the AI-written code and results and can explain them. The browser predictor is also checked automatically against scikit-learn by `tests/test_predictor_parity.py`.

## Data and privacy
No passwords, API keys or private data are in this repo. All data comes from the public UCI repository.

---

A short version of this disclosure is in the [README](README.md#use-of-ai-tools) and on the [project website](https://Jay-Glasswala.github.io/ML-Project/#reproduction).
