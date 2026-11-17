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
- training medians (used for the "what-if" explanation) and a few real test
  clients for the example buttons, with their Python probabilities

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


def pick_examples(test_df, models):
    X = test_df[EXAMPLE_COLS]
    p_hgb = models['hist_gradient_boosting'].predict_proba(X)[:, 1]
    y = test_df['default'].values
    t = 1 / 6

    def idx_closest(mask, target):
        cand = np.where(mask)[0]
        return int(cand[np.argmin(np.abs(p_hgb[cand] - target))])

    chosen = {
        'Typical good client': idx_closest((y == 0) & (p_hgb < t), np.median(p_hgb[(y == 0) & (p_hgb < t)])),
        'Caught defaulter': idx_closest((y == 1) & (p_hgb >= t), np.median(p_hgb[(y == 1) & (p_hgb >= t)])),
        'False alarm (paid anyway)': int(np.where(y == 0)[0][np.argmax(p_hgb[y == 0])]),
        'Missed defaulter': int(np.where(y == 1)[0][np.argmin(p_hgb[y == 1])]),
    }
    out = []
    for label, i in chosen.items():
        row = {c: (int(X.iloc[i][c])) for c in EXAMPLE_COLS}
        out.append({'label': label, 'actual_default': int(y[i]), 'inputs': row,
                    'python_prob_hgb': round(float(p_hgb[i]), 4)})
    return out


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
    return payload


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(__file__), '..')
    export_predictor(os.path.join(root, 'data', 'processed'), os.path.join(root, 'web'))
