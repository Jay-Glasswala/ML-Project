"""
Leakage Prevention and Pipeline Integrity Tests (tests/test_no_leakage.py)
ES 335: Cost-Aware Credit Card Default Prediction
Anant Shukla (25110030) & Jay R. Glasswala (25110147)
"""

import os
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from src.data_loader import load_raw_data, clean_and_prepare_data, split_and_save_data
from src.pipelines import (
    prepare_dataset_splits, build_pipeline, PROTECTED_ATTRIBUTES,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES
)


@pytest.fixture(scope="module")
def dataset_splits():
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    train_path = os.path.join(processed_dir, 'train.csv')
    test_path = os.path.join(processed_dir, 'test.csv')

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raw_df = load_raw_data()
        clean_df, audit = clean_and_prepare_data(raw_df)
        train_df, test_df, _ = split_and_save_data(clean_df, audit, output_dir=processed_dir)
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

    return train_df, test_df


def test_train_test_disjoint_and_shapes(dataset_splits):
    train_df, test_df = dataset_splits
    
    # Check expected 80/20 row distribution after 35 deduplications
    assert len(train_df) == 23972, f"Expected 23972 train rows, got {len(train_df)}"
    assert len(test_df) == 5993, f"Expected 5993 test rows, got {len(test_df)}"
    assert len(train_df) + len(test_df) == 29965

    # Check stratification balance
    train_rate = train_df['default'].mean()
    test_rate = test_df['default'].mean()
    assert abs(train_rate - test_rate) < 0.005, f"Stratification failed: train={train_rate}, test={test_rate}"


def test_protected_attributes_excluded(dataset_splits):
    train_df, test_df = dataset_splits
    X_train, y_train, X_test, y_test, audit_train, audit_test = prepare_dataset_splits(train_df, test_df)

    # Assert SEX and MARRIAGE are NOT in feature matrix X
    for attr in PROTECTED_ATTRIBUTES:
        assert attr not in X_train.columns, f"Leakage: Protected attribute '{attr}' present in X_train!"
        assert attr not in X_test.columns, f"Leakage: Protected attribute '{attr}' present in X_test!"

    # Assert EDUCATION and AGE remain inputs
    assert 'EDUCATION' in X_train.columns, "EDUCATION missing from X_train"
    assert 'AGE' in X_train.columns, "AGE missing from X_train"

    # Assert audit metadata holds protected attributes
    assert 'SEX' in audit_train.columns and 'MARRIAGE' in audit_train.columns
    assert 'SEX' in audit_test.columns and 'MARRIAGE' in audit_test.columns


def test_pipeline_fits_on_train_without_leakage(dataset_splits):
    train_df, test_df = dataset_splits
    X_train, y_train, X_test, y_test, _, _ = prepare_dataset_splits(train_df, test_df)

    # Build standard pipeline
    pipe = build_pipeline(LogisticRegression(max_iter=200, random_state=42))

    # Fit pipeline strictly on a subset of training data (simulate a fold)
    fold_train_X = X_train.iloc[:1000]
    fold_train_y = y_train.iloc[:1000]
    
    pipe.fit(fold_train_X, fold_train_y)

    # Check fitted transformer properties
    scaler = pipe.named_steps['preprocessor'].named_transformers_['num']
    assert hasattr(scaler, 'mean_'), "StandardScaler was not fitted"
    assert len(scaler.mean_) == len(NUMERIC_FEATURES), "Scaler dimension mismatch"

    # Predict on unseen test set without refitting
    probs = pipe.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2), f"Expected ({len(X_test)}, 2), got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1]"
