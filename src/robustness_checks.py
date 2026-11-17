"""
Extra checks promised in the proposal. All of them use only the training
split (out-of-fold predictions or bootstrap refits), so the test set is still
scored once in evaluate.py.

1. Threshold check: is the cost-minimising threshold on out-of-fold training
   predictions close to the formula t* = 1 / (1 + r)?
2. Class weights vs. threshold shifting: does class_weight='balanced' hurt
   calibration, and does it beat simply moving the threshold?
3. Coefficient stability: refit Ridge and Lasso on bootstrap samples of the
   training data and see how stable the six BILL_AMT weights are.
4. Extreme values: the bill/payment columns are very skewed. Does a signed
   log transform help Ridge? (Trees don't care about monotone transforms.)
5. Bigger tuning grid: our HGB grid was small (4 settings). Does a random
   search over a wider grid find anything clearly better?

Output: data/processed/robustness_checks.json
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import log_loss, average_precision_score

from src.pipelines import prepare_dataset_splits, build_pipeline, get_preprocessor
from src.evaluate import compute_equal_frequency_ece

SEED = 42
BILL_COLS = [f'BILL_AMT{i}' for i in range(1, 7)]


def cost_per_client(y, p, t, r):
    pred = p >= t
    fn = np.sum((pred == 0) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    return (r * fn + fp) / len(y)


def oof_probs(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]


def threshold_check(y, p):
    grid = np.round(np.arange(0.02, 0.81, 0.01), 2)
    out = {}
    for r in [1, 2, 3, 5, 10]:
        costs = [cost_per_client(y, p, t, r) for t in grid]
        best = int(np.argmin(costs))
        t_formula = 1 / (1 + r)
        out[str(r)] = {
            'formula_threshold': round(t_formula, 3),
            'empirical_best_threshold': float(grid[best]),
            'cost_at_formula': round(float(cost_per_client(y, p, t_formula, r)), 4),
            'cost_at_empirical_best': round(float(costs[best]), 4),
            'cost_at_0.5': round(float(cost_per_client(y, p, 0.5, r)), 4),
        }
    # full curve for r = 5 (used for the website plot / notebook)
    curve = [{'t': float(t), 'cost': round(float(cost_per_client(y, p, t, 5)), 4)} for t in grid]
    return out, curve


def class_weight_check(X, y, models):
    r, t_star = 5, 1 / 6
    out = {}
    for name, (plain, weighted) in models.items():
        for label, m in [('unweighted', plain), ('balanced', weighted)]:
            p = oof_probs(m, X, y)
            ece, _ = compute_equal_frequency_ece(y, p)
            out[f'{name}_{label}'] = {
                'pr_auc': round(float(average_precision_score(y, p)), 4),
                'log_loss': round(float(log_loss(y, p)), 4),
                'ece': ece,
                'mean_predicted_prob': round(float(p.mean()), 3),
                'cost_r5_at_t_star': round(float(cost_per_client(y, p, t_star, r)), 4),
                'cost_r5_at_0.5': round(float(cost_per_client(y, p, 0.5, r)), 4),
            }
    out['observed_default_rate'] = round(float(y.mean()), 3)
    return out


def coefficient_stability(X, y, n_boot=100):
    configs = {
        'ridge_C0.1': LogisticRegression(penalty='l2', C=0.1, max_iter=1000),
        'lasso_C1.0': LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000),
        'lasso_C0.01': LogisticRegression(penalty='l1', C=0.01, solver='liblinear', max_iter=1000),
    }
    rng = np.random.RandomState(SEED)
    idx_list = [rng.randint(0, len(X), len(X)) for _ in range(n_boot)]
    out = {}
    for name, clf in configs.items():
        coefs = []
        for idx in idx_list:
            pipe = build_pipeline(clf)
            pipe.fit(X.iloc[idx], y.iloc[idx])
            names = pipe.named_steps['preprocessor'].get_feature_names_out()
            w = pd.Series(pipe.named_steps['classifier'].coef_[0], index=[n.split('__')[-1] for n in names])
            coefs.append(w[BILL_COLS].values)
        coefs = np.array(coefs)
        out[name] = {
            col: {
                'mean': round(float(coefs[:, j].mean()), 3),
                'ci95': [round(float(np.percentile(coefs[:, j], 2.5)), 3),
                         round(float(np.percentile(coefs[:, j], 97.5)), 3)],
                'share_zero': round(float(np.mean(np.abs(coefs[:, j]) < 1e-8)), 2),
                'share_positive': round(float(np.mean(coefs[:, j] > 1e-8)), 2),
            }
            for j, col in enumerate(BILL_COLS)
        }
    out['n_bootstrap'] = n_boot
    return out


MONEY_COLS = ['LIMIT_BAL'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]


def signed_log(X):
    # log(1 + |x|) with the sign kept, so negative bills stay negative
    X = X.copy()
    X[MONEY_COLS] = np.sign(X[MONEY_COLS]) * np.log1p(np.abs(X[MONEY_COLS]))
    return X


def log_transform_check(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = {}
    for name, use_log in [('ridge_raw', False), ('ridge_signed_log', True)]:
        steps = [('log', FunctionTransformer(signed_log))] if use_log else []
        steps += [('preprocessor', get_preprocessor()), ('classifier', LogisticRegression(C=0.1, max_iter=1000))]
        pipe = Pipeline(steps)
        ll = -cross_val_score(pipe, X, y, cv=cv, scoring='neg_log_loss')
        ap = cross_val_score(pipe, X, y, cv=cv, scoring='average_precision')
        out[name] = {'cv_log_loss': round(float(ll.mean()), 4), 'cv_log_loss_std': round(float(ll.std()), 4),
                     'cv_pr_auc': round(float(ap.mean()), 4)}
    return out


def wider_grid_check(X, y, n_iter=20):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe = build_pipeline(HistGradientBoostingClassifier(random_state=SEED))
    grid = {
        'classifier__learning_rate': [0.02, 0.05, 0.1],
        'classifier__max_iter': [100, 200, 400],
        'classifier__max_leaf_nodes': [15, 31, 63],
        'classifier__min_samples_leaf': [20, 50, 100],
        'classifier__l2_regularization': [0.0, 1.0],
    }
    search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring='neg_log_loss', cv=cv,
                                random_state=SEED, n_jobs=-1)
    search.fit(X, y)
    ours = -cross_val_score(build_pipeline(HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=100, random_state=SEED)), X, y, cv=cv, scoring='neg_log_loss')
    best_std = float(search.cv_results_['std_test_score'][search.best_index_])
    return {
        'n_settings_tried': n_iter,
        'our_setting': {'learning_rate': 0.05, 'max_iter': 100},
        'our_cv_log_loss': round(float(ours.mean()), 4),
        'our_cv_log_loss_std': round(float(ours.std()), 4),
        'best_random_search_params': {k.replace('classifier__', ''): v for k, v in search.best_params_.items()},
        'best_random_search_cv_log_loss': round(float(-search.best_score_), 4),
        'best_random_search_cv_log_loss_std': round(best_std, 4),
    }


def run_robustness_checks(processed_dir):
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))
    X, y, _, _, _, _ = prepare_dataset_splits(train_df, test_df)

    print("[Robustness] 1/5 threshold check on out-of-fold HGB predictions...")
    hgb = build_pipeline(HistGradientBoostingClassifier(learning_rate=0.05, max_iter=100, random_state=SEED))
    p_hgb = oof_probs(hgb, X, y)
    thresholds, curve = threshold_check(y.values, p_hgb)

    print("[Robustness] 2/5 class weights vs threshold shifting...")
    weight_models = {
        'ridge': (build_pipeline(LogisticRegression(C=0.1, max_iter=1000)),
                  build_pipeline(LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced'))),
        'hgb': (build_pipeline(HistGradientBoostingClassifier(learning_rate=0.05, max_iter=100, random_state=SEED)),
                build_pipeline(HistGradientBoostingClassifier(learning_rate=0.05, max_iter=100, random_state=SEED,
                                                              class_weight='balanced'))),
    }
    weights = class_weight_check(X, y, weight_models)

    print("[Robustness] 3/5 bootstrap stability of BILL_AMT coefficients (100 refits each)...")
    stability = coefficient_stability(X, y)

    print("[Robustness] 4/5 signed log transform for Ridge...")
    log_check = log_transform_check(X, y)

    print("[Robustness] 5/5 wider random search for HGB (20 settings)...")
    grid_check = wider_grid_check(X, y)

    report = {'threshold_check': thresholds, 'cost_curve_r5': curve,
              'class_weight_check': weights, 'coefficient_stability': stability,
              'log_transform_check': log_check, 'wider_grid_check': grid_check}
    out_path = os.path.join(processed_dir, 'robustness_checks.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {out_path}")
    return report


if __name__ == '__main__':
    run_robustness_checks(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
