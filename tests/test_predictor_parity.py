"""
Checks that the website's "Check a client" tool gives the same probabilities
as our scikit-learn models, for every client in the test set.

1. A small Python copy of the browser logic, reading web/data/predictor.json
   (always runs).
2. The real web/predictor.js run through node (skipped if node isn't installed).

Both need the trained models, so run `python run_all.py` first.
"""

import os
import json
import shutil
import subprocess

import joblib
import numpy as np
import pandas as pd
import pytest

from src.pipelines import NUMERIC_FEATURES, CATEGORICAL_FEATURES

ROOT = os.path.join(os.path.dirname(__file__), '..')
PROCESSED = os.path.join(ROOT, 'data', 'processed')
PREDICTOR_JSON = os.path.join(ROOT, 'web', 'data', 'predictor.json')
PREDICTOR_JS = os.path.abspath(os.path.join(ROOT, 'web', 'predictor.js'))
COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TOL = 1e-9

needs_artifacts = pytest.mark.skipif(
    not (os.path.exists(PREDICTOR_JSON) and os.path.exists(os.path.join(PROCESSED, 'trained_models.joblib'))),
    reason="run python run_all.py first to create the models and predictor.json")


def sklearn_probs():
    test_df = pd.read_csv(os.path.join(PROCESSED, 'test.csv'))
    models = joblib.load(os.path.join(PROCESSED, 'trained_models.joblib'))
    X = test_df[COLS]
    return (X, models['hist_gradient_boosting'].predict_proba(X)[:, 1],
            models['ridge_logistic'].predict_proba(X)[:, 1])


def python_copy_of_browser(model, X):
    pre = model['preprocessing']
    Z = (X[pre['numeric']].values - np.array(pre['mean'])) / np.array(pre['scale'])
    edu = np.column_stack([(X['EDUCATION'].values == lvl).astype(float) for lvl in pre['education_columns']])
    Z = np.hstack([Z, edu])

    ridge = 1 / (1 + np.exp(-(Z @ np.array(model['ridge']['coef']) + model['ridge']['intercept'])))

    raw = np.full(len(Z), model['hgb']['baseline'])
    for tree in model['hgb']['trees']:
        for i, x in enumerate(Z):
            node = 0
            while not tree['leaf'][node]:
                node = tree['l'][node] if x[tree['f'][node]] <= tree['t'][node] else tree['r'][node]
            raw[i] += tree['v'][node]
    return 1 / (1 + np.exp(-raw)), ridge


@needs_artifacts
def test_python_copy_matches_sklearn():
    model = json.load(open(PREDICTOR_JSON))
    X, p_hgb, p_ridge = sklearn_probs()
    hgb, ridge = python_copy_of_browser(model, X)
    assert np.abs(hgb - p_hgb).max() < TOL
    assert np.abs(ridge - p_ridge).max() < TOL


@needs_artifacts
@pytest.mark.skipif(shutil.which('node') is None, reason="node is not installed")
def test_predictor_js_matches_sklearn(tmp_path):
    X, p_hgb, p_ridge = sklearn_probs()
    clients = tmp_path / 'clients.json'
    clients.write_text(json.dumps(X.to_dict(orient='records')))
    runner = tmp_path / 'run.js'
    runner.write_text(
        "const fs = require('fs');\n"
        f"const p = require({json.dumps(PREDICTOR_JS)});\n"
        f"const model = JSON.parse(fs.readFileSync({json.dumps(os.path.abspath(PREDICTOR_JSON))}));\n"
        f"const clients = JSON.parse(fs.readFileSync({json.dumps(str(clients))}));\n"
        "process.stdout.write(JSON.stringify(clients.map(c => [p.predictHGB(model, c), p.predictRidge(model, c)])));\n")
    out = np.array(json.loads(subprocess.check_output(['node', str(runner)])))
    assert np.abs(out[:, 0] - p_hgb).max() < TOL
    assert np.abs(out[:, 1] - p_ridge).max() < TOL


@needs_artifacts
def test_examples_match_python_and_fit_their_label():
    from src.export_predictor import EXAMPLE_TYPES
    model = json.load(open(PREDICTOR_JSON))
    rules = {ex['key']: ex['rule'] for ex in EXAMPLE_TYPES}
    X, p_hgb, _ = sklearn_probs()
    y = pd.read_csv(os.path.join(PROCESSED, 'test.csv'))['default'].values

    assert len(model['examples']) == len(EXAMPLE_TYPES)
    for ex in model['examples']:
        i = ex['test_index']
        # the example is really that test client, with the right outcome and probability
        assert ex['inputs'] == {c: int(X.iloc[i][c]) for c in COLS}
        assert ex['actual_default'] == int(y[i])
        assert abs(ex['python_prob_hgb'] - p_hgb[i]) < 1e-4
        # and it really fits the description of its type
        mask = np.asarray(rules[ex['key']](X.assign(default=y), p_hgb, y))
        assert mask[i], f"example '{ex['label']}' does not fit its own rule"

    # the browser maths gives the same probability for every example
    hgb, _ = python_copy_of_browser(model, pd.DataFrame([e['inputs'] for e in model['examples']])[COLS])
    for ex, p in zip(model['examples'], hgb):
        assert abs(p - ex['python_prob_hgb']) < 1e-4


@needs_artifacts
def test_random_client_data_matches_test_csv():
    path = os.path.join(ROOT, 'web', 'data', 'test_clients.json')
    data = json.load(open(path))
    test_df = pd.read_csv(os.path.join(PROCESSED, 'test.csv'))
    assert data['columns'] == COLS
    assert np.array_equal(np.array(data['rows']), test_df[COLS].values)
    assert data['default'] == test_df['default'].tolist()
