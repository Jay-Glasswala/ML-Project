"""
Error analysis for the best model (HistGradientBoosting) at the r = 5 threshold.

We split the test set into TP / FP / FN / TN at t* = 1/6 and compare a few
interpretable features across the four groups, then save the most confident
mistakes as example rows. Everything shown in the "Error Analysis" section of
the website comes from data/processed/error_analysis.json.
"""

import os
import json
import numpy as np
import pandas as pd

MODEL = 'hist_gradient_boosting'
R = 5.0
T_STAR = 1.0 / (1.0 + R)
PAY_COLS = [f'PAY_{i}' for i in range(1, 7)]


def add_features(df):
    """A few easy-to-read summary features used only for describing errors."""
    out = df.copy()
    # Share of the credit limit used in September (can exceed 1 or be negative)
    out['UTIL_SEP'] = out['BILL_AMT1'] / out['LIMIT_BAL']
    # Number of the six months with a payment delay (PAY >= 1)
    out['MONTHS_LATE'] = (out[PAY_COLS] >= 1).sum(axis=1)
    out['WORST_DELAY'] = out[PAY_COLS].max(axis=1)
    # September payment as a share of the September bill (only if bill > 0)
    bill = out['BILL_AMT1'].where(out['BILL_AMT1'] > 0)
    out['PAID_SHARE_SEP'] = (out['PAY_AMT1'] / bill).clip(upper=5)
    return out


def group_profile(df):
    return {
        'n': int(len(df)),
        'median_limit_bal': float(df['LIMIT_BAL'].median()),
        'median_age': float(df['AGE'].median()),
        'share_late_in_sep': round(float((df['PAY_1'] >= 1).mean()), 3),
        'mean_months_late': round(float(df['MONTHS_LATE'].mean()), 2),
        'median_util_sep': round(float(df['UTIL_SEP'].median()), 3),
        'median_paid_share_sep': round(float(df['PAID_SHARE_SEP'].median(skipna=True)), 3),
        'share_no_bill_sep': round(float((df['BILL_AMT1'] <= 0).mean()), 3),
        'median_prob': round(float(df['prob'].median()), 3),
    }


def example_rows(df, n=3, ascending=False):
    cols = ['prob', 'LIMIT_BAL', 'AGE'] + PAY_COLS + ['BILL_AMT1', 'PAY_AMT1', 'UTIL_SEP']
    rows = df.sort_values('prob', ascending=ascending).head(n)[cols]
    return [{k: (round(float(v), 3) if isinstance(v, float) else int(v)) for k, v in r.items()}
            for r in rows.to_dict(orient='records')]


def permutation_importance_pr_auc(processed_dir, test_df, n_repeats=10):
    """Drop in test PR-AUC when one input column is shuffled (fitted HGB model, no refitting)."""
    import joblib
    from sklearn.inspection import permutation_importance
    from src.pipelines import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    model = joblib.load(os.path.join(processed_dir, 'trained_models.joblib'))[MODEL]
    X = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    res = permutation_importance(model, X, test_df['default'], scoring='average_precision',
                                 n_repeats=n_repeats, random_state=42)
    order = np.argsort(-res.importances_mean)
    return [{'feature': X.columns[i], 'mean_drop': round(float(res.importances_mean[i]), 4),
             'std': round(float(res.importances_std[i]), 4)} for i in order[:8]]


def run_error_analysis(processed_dir):
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))
    with open(os.path.join(processed_dir, 'test_predictions.json')) as f:
        preds = json.load(f)

    df = add_features(test_df)
    df['prob'] = np.asarray(preds[MODEL])
    df['pred'] = (df['prob'] >= T_STAR).astype(int)
    y = df['default']

    groups = {
        'true_positive': df[(df['pred'] == 1) & (y == 1)],
        'false_positive': df[(df['pred'] == 1) & (y == 0)],
        'false_negative': df[(df['pred'] == 0) & (y == 1)],
        'true_negative': df[(df['pred'] == 0) & (y == 0)],
    }

    report = {
        'model': MODEL,
        'r': R,
        'threshold': round(T_STAR, 4),
        'profiles': {name: group_profile(g) for name, g in groups.items()},
        # most confident mistakes: highest-probability false alarms, lowest-probability misses
        'examples_false_positive': example_rows(groups['false_positive'], ascending=False),
        'examples_false_negative': example_rows(groups['false_negative'], ascending=True),
        # how many missed defaulters had no delay at all in the last six months
        'fn_share_never_late': round(float((groups['false_negative']['MONTHS_LATE'] == 0).mean()), 3),
        'fp_share_late_in_sep': round(float((groups['false_positive']['PAY_1'] >= 1).mean()), 3),
    }

    report['permutation_importance'] = permutation_importance_pr_auc(processed_dir, test_df)

    out_path = os.path.join(processed_dir, 'error_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"[ErrorAnalysis] threshold = {T_STAR:.3f}")
    for name, p in report['profiles'].items():
        print(f"  {name:15s} n={p['n']:5d}  late in Sep={p['share_late_in_sep']:.2f}  "
              f"months late={p['mean_months_late']:.2f}  util={p['median_util_sep']:.2f}")
    print(f"  Saved: {out_path}")
    return report


if __name__ == '__main__':
    run_error_analysis(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
