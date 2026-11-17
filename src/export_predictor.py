"""
Exports the fitted Ridge and HistGradientBoosting models to a small JSON file
(web/data/predictor.json) so the "Check a client" tool on the website can make
predictions in the browser without a server.

What gets exported:
- the preprocessing: scaler means / standard deviations and the education
  one-hot columns, in the same column order the models were trained on
- Ridge: weights and intercept
- HGB: every tree as a list of nodes (feature, threshold, left, right, leaf value)
  plus the starting score
- training medians (used for the "what-if" explanation) and 13 real test
  clients for the example dropdown (one per client type), with their Python
  probabilities
- all test clients' inputs and outcomes (web/data/test_clients.json) for the
  "Random client" button

tests/test_predictor_parity.py checks that predicting from this JSON gives the
same probabilities as scikit-learn for every test client.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from src.pipelines import NUMERIC_FEATURES, CATEGORICAL_FEATURES

EXAMPLE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def export_preprocessor(pipe):
    pre = pipe.named_steps['preprocessor']
    scaler = pre.named_transformers_['num']
    onehot = pre.named_transformers_['cat']
    cats = [int(c) for c in onehot.categories_[0]]
    dropped = cats[int(onehot.drop_idx_[0])] if onehot.drop_idx_ is not None else None
    return {
        'numeric': NUMERIC_FEATURES,
        'mean': [float(v) for v in scaler.mean_],
        'scale': [float(v) for v in scaler.scale_],
        # one 0/1 column per education level except the dropped one
        'education_columns': [c for c in cats if c != dropped],
    }


def export_ridge(pipe):
    clf = pipe.named_steps['classifier']
    return {'coef': [float(w) for w in clf.coef_[0]], 'intercept': float(clf.intercept_[0])}


def export_hgb(pipe):
    clf = pipe.named_steps['classifier']
    trees = []
    for (predictor,) in clf._predictors:
        nodes = predictor.nodes
        if nodes['is_categorical'].any():
            raise ValueError("categorical splits are not supported by the web predictor")
        trees.append({
            'f': nodes['feature_idx'].astype(int).tolist(),
            't': [float(x) for x in nodes['num_threshold']],
            'm': nodes['missing_go_to_left'].astype(int).tolist(),
            'l': nodes['left'].astype(int).tolist(),
            'r': nodes['right'].astype(int).tolist(),
            'leaf': nodes['is_leaf'].astype(int).tolist(),
            'v': [float(x) for x in nodes['value']],
        })
    return {'baseline': float(np.ravel(clf._baseline_prediction)[0]), 'trees': trees}


T_STAR = 1 / 6        # threshold at r = 5, used to define the example types
PAY_COLS = [f'PAY_{i}' for i in range(1, 7)]
BILL_COLS = [f'BILL_AMT{i}' for i in range(1, 7)]

# Each example type is a rule on (test data, HGB probability p, actual outcome y).
# For each rule we load the "most typical" matching client: the one whose
# probability is closest to the median probability of everyone matching it.
# 'pick' = 'max_p' / 'min_p' / 'closest_t' overrides that for a few types.
EXAMPLE_TYPES = [
    {'key': 'typical_good', 'label': 'Typical good client',
     'description': "Didn't default and the model correctly cleared them (median such client).",
     'rule': lambda d, p, y: (y == 0) & (p < T_STAR)},
    {'key': 'caught', 'label': 'Caught defaulter',
     'description': 'Defaulted and the model flagged them (median such client).',
     'rule': lambda d, p, y: (y == 1) & (p >= T_STAR)},
    {'key': 'false_alarm', 'label': 'Most confident false alarm',
     'description': 'The highest-probability client who did NOT default (Section 10).',
     'rule': lambda d, p, y: y == 0, 'pick': 'max_p'},
    {'key': 'missed', 'label': 'Most confidently missed defaulter',
     'description': 'The lowest-probability client who DID default (Section 10).',
     'rule': lambda d, p, y: y == 1, 'pick': 'min_p'},
    {'key': 'borderline', 'label': 'Borderline client',
     'description': 'Probability right at the r = 5 threshold, so moving the slider flips the prediction.',
     'rule': lambda d, p, y: np.ones(len(d), dtype=bool), 'pick': 'closest_t'},
    {'key': 'pays_in_full', 'label': 'Pays in full every month',
     'description': 'Repayment status is "paid in full" in all six months.',
     'rule': lambda d, p, y: (d[PAY_COLS] == -1).all(axis=1).values},
    {'key': 'inactive', 'label': 'Inactive card (no bills)',
     'description': 'No bill in any of the six months (Section 2: 836 such clients in the full data).',
     'rule': lambda d, p, y: (d[BILL_COLS] == 0).all(axis=1).values},
    {'key': 'overpaid', 'label': 'Overpaid (negative September bill)',
     'description': 'Paid more than they owed, so the September balance is negative.',
     'rule': lambda d, p, y: (d['BILL_AMT1'] < 0).values},
    {'key': 'over_limit', 'label': 'Over the credit limit',
     'description': 'September bill is higher than the credit limit.',
     'rule': lambda d, p, y: (d['BILL_AMT1'] > d['LIMIT_BAL']).values},
    {'key': 'late_sep_only', 'label': 'Late only in September',
     'description': 'Paid on time from April to August, then fell behind in September.',
     'rule': lambda d, p, y: ((d['PAY_1'] >= 1) & (d[PAY_COLS[1:]] <= 0).all(axis=1)).values},
    {'key': 'caught_up', 'label': 'Was late, now caught up',
     'description': 'Late at some point in April-June, on time from July to September.',
     'rule': lambda d, p, y: ((d[['PAY_4', 'PAY_5', 'PAY_6']] >= 1).any(axis=1)
                              & (d[['PAY_1', 'PAY_2', 'PAY_3']] <= 0).all(axis=1)).values},
    {'key': 'young_low_limit', 'label': 'Young, low credit limit',
     'description': 'Age 25 or under with a credit limit of at most 50,000 NT$.',
     'rule': lambda d, p, y: ((d['AGE'] <= 25) & (d['LIMIT_BAL'] <= 50000)).values},
    {'key': 'high_limit_big_balance', 'label': 'High limit, large balance',
     'description': 'Credit limit of at least 400,000 NT$ and a September bill of at least half of it.',
     'rule': lambda d, p, y: ((d['LIMIT_BAL'] >= 400000) & (d['BILL_AMT1'] >= 0.5 * d['LIMIT_BAL'])).values},
]


OUTCOME_TYPES = {'typical_good', 'caught', 'false_alarm', 'missed', 'borderline'}


def pick_examples(test_df, models):
    X = test_df[EXAMPLE_COLS]
    p = models['hist_gradient_boosting'].predict_proba(X)[:, 1]
    y = test_df['default'].values
    out = []
    for ex in EXAMPLE_TYPES:
        idx = np.where(np.asarray(ex['rule'](test_df, p, y)))[0]
        if len(idx) == 0:
            continue
        pick = ex.get('pick')
        if pick == 'max_p':
            i = idx[np.argmax(p[idx])]
        elif pick == 'min_p':
            i = idx[np.argmin(p[idx])]
        elif pick == 'closest_t':
            i = idx[np.argmin(np.abs(p[idx] - T_STAR))]
        else:
            i = idx[np.argmin(np.abs(p[idx] - np.median(p[idx])))]
        # group stats only make sense for types defined by the client's profile,
        # not for the ones defined by the outcome or the model's probability
        by_profile = ex['key'] not in OUTCOME_TYPES
        out.append({
            'key': ex['key'], 'label': ex['label'], 'description': ex['description'],
            'n_matching': int(len(idx)) if by_profile else None,
            'default_rate_matching': round(float(y[idx].mean()), 3) if by_profile else None,
            'test_index': int(i), 'actual_default': int(y[i]),
            'inputs': {c: int(X.iloc[i][c]) for c in EXAMPLE_COLS},
            'python_prob_hgb': round(float(p[i]), 4),
        })
    return out


def export_test_clients(test_df, web_dir):
    """All test clients (inputs + actual outcome) for the 'Random client' button."""
    payload = {
        'columns': EXAMPLE_COLS,
        'rows': test_df[EXAMPLE_COLS].astype(int).values.tolist(),
        'default': test_df['default'].astype(int).tolist(),
    }
    out_path = os.path.join(web_dir, 'data', 'test_clients.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    print(f"[ExportPredictor] {len(payload['rows'])} test clients -> {out_path} "
          f"({os.path.getsize(out_path) / 1024:.0f} KB)")


def export_predictor(processed_dir, web_dir):
    models = joblib.load(os.path.join(processed_dir, 'trained_models.joblib'))
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))

    hgb, ridge = models['hist_gradient_boosting'], models['ridge_logistic']
    payload = {
        'preprocessing': export_preprocessor(hgb),
        'ridge': export_ridge(ridge),
        'hgb': export_hgb(hgb),
        'train_medians': {c: float(train_df[c].median()) for c in NUMERIC_FEATURES},
        'input_ranges': {c: [float(train_df[c].min()), float(train_df[c].max())] for c in NUMERIC_FEATURES},
        'examples': pick_examples(test_df, models),
    }
    # both models must use the same preprocessing for the shared code in predictor.js
    assert export_preprocessor(ridge) == payload['preprocessing']

    out_path = os.path.join(web_dir, 'data', 'predictor.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    print(f"[ExportPredictor] {len(payload['hgb']['trees'])} trees exported -> {out_path} "
          f"({os.path.getsize(out_path) / 1024:.0f} KB)")
    export_test_clients(test_df, web_dir)
    return payload


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(__file__), '..')
    export_predictor(os.path.join(root, 'data', 'processed'), os.path.join(root, 'web'))
