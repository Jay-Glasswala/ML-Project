"""
Model Training and Hyperparameter Tuning Module (src/train_models.py)
ES 335: Cost-Aware Credit Card Default Prediction
Anant Shukla (25110030) & Jay R. Glasswala (25110147)
"""

import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss, average_precision_score

from src.pipelines import prepare_dataset_splits, build_pipeline


def run_cv_tuning(X_train, y_train):
    """
    Performs Stratified 5-Fold CV hyperparameter tuning.
    Chooses best parameters by log-loss (and reports PR-AUC).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_models = {}

    print("\n--- 1. Baselines ---")
    
    # Majority class baseline
    dummy = build_pipeline(DummyClassifier(strategy='most_frequent'))
    dummy_scores = cross_val_score(dummy, X_train, y_train, cv=cv, scoring='neg_log_loss')
    results['majority_class'] = {
        'best_params': {},
        'cv_log_loss': float(-dummy_scores.mean()),
        'cv_log_loss_std': float(dummy_scores.std())
    }
    dummy.fit(X_train, y_train)
    best_models['majority_class'] = dummy
    print(f"Majority Class: CV Log-Loss = {-dummy_scores.mean():.4f}")

    # Plain Logistic Regression (unregularized reference)
    lr_plain = build_pipeline(LogisticRegression(penalty=None, max_iter=500, random_state=42))
    plain_scores = cross_val_score(lr_plain, X_train, y_train, cv=cv, scoring='neg_log_loss')
    results['plain_logistic'] = {
        'best_params': {},
        'cv_log_loss': float(-plain_scores.mean()),
        'cv_log_loss_std': float(plain_scores.std())
    }
    lr_plain.fit(X_train, y_train)
    best_models['plain_logistic'] = lr_plain
    print(f"Plain Logistic: CV Log-Loss = {-plain_scores.mean():.4f}")

    print("\n--- 2. Regularized Logistic Regression Models ---")

    # Ridge (L2) tuning
    ridge_configs = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    best_ridge_loss = float('inf')
    best_ridge_C = 1.0
    for C in ridge_configs:
        pipe = build_pipeline(LogisticRegression(penalty='l2', C=C, max_iter=500, random_state=42))
        loss = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_log_loss').mean()
        if loss < best_ridge_loss:
            best_ridge_loss = loss
            best_ridge_C = C
    results['ridge_logistic'] = {
        'best_params': {'C': best_ridge_C},
        'cv_log_loss': float(best_ridge_loss)
    }
    best_ridge = build_pipeline(LogisticRegression(penalty='l2', C=best_ridge_C, max_iter=500, random_state=42))
    best_ridge.fit(X_train, y_train)
    best_models['ridge_logistic'] = best_ridge
    print(f"Ridge Logistic (L2, C={best_ridge_C}): CV Log-Loss = {best_ridge_loss:.4f}")

    # Lasso (L1) tuning
    lasso_configs = [1e-2, 0.1, 1.0, 10.0]
    best_lasso_loss = float('inf')
    best_lasso_C = 0.1
    for C in lasso_configs:
        pipe = build_pipeline(LogisticRegression(penalty='l1', C=C, solver='saga', max_iter=300, random_state=42))
        loss = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_log_loss').mean()
        if loss < best_lasso_loss:
            best_lasso_loss = loss
            best_lasso_C = C
    results['lasso_logistic'] = {
        'best_params': {'C': best_lasso_C},
        'cv_log_loss': float(best_lasso_loss)
    }
    best_lasso = build_pipeline(LogisticRegression(penalty='l1', C=best_lasso_C, solver='saga', max_iter=500, random_state=42))
    best_lasso.fit(X_train, y_train)
    best_models['lasso_logistic'] = best_lasso
    print(f"Lasso Logistic (L1, C={best_lasso_C}): CV Log-Loss = {best_lasso_loss:.4f}")

    # Elastic-Net tuning
    enet_configs = [(0.1, 0.2), (0.1, 0.5), (0.1, 0.8), (1.0, 0.5)]
    best_enet_loss = float('inf')
    best_enet_params = (0.1, 0.5)
    for C, l1_ratio in enet_configs:
        pipe = build_pipeline(LogisticRegression(penalty='elasticnet', C=C, l1_ratio=l1_ratio, solver='saga', max_iter=300, random_state=42))
        loss = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_log_loss').mean()
        if loss < best_enet_loss:
            best_enet_loss = loss
            best_enet_params = (C, l1_ratio)
    results['elastic_net'] = {
        'best_params': {'C': best_enet_params[0], 'l1_ratio': best_enet_params[1]},
        'cv_log_loss': float(best_enet_loss)
    }
    best_enet = build_pipeline(LogisticRegression(penalty='elasticnet', C=best_enet_params[0], l1_ratio=best_enet_params[1], solver='saga', max_iter=500, random_state=42))
    best_enet.fit(X_train, y_train)
    best_models['elastic_net'] = best_enet
    print(f"Elastic-Net (C={best_enet_params[0]}, l1_ratio={best_enet_params[1]}): CV Log-Loss = {best_enet_loss:.4f}")

    print("\n--- 3. Non-Linear Models ---")

    # RBF-Kernel SVM (subsampled to 6,000 for fast, scalable Platt scaling fitting)
    subsample_idx = np.random.RandomState(42).choice(len(X_train), size=6000, replace=False)
    X_sub = X_train.iloc[subsample_idx]
    y_sub = y_train.iloc[subsample_idx]
    
    svm_configs = [(1.0, 'scale'), (1.0, 0.05), (5.0, 'scale')]
    best_svm_loss = float('inf')
    best_svm_params = (1.0, 'scale')
    for C, gamma in svm_configs:
        pipe = build_pipeline(SVC(C=C, gamma=gamma, probability=True, random_state=42))
        loss = -cross_val_score(pipe, X_sub, y_sub, cv=cv, scoring='neg_log_loss').mean()
        if loss < best_svm_loss:
            best_svm_loss = loss
            best_svm_params = (C, gamma)
    results['rbf_svm'] = {
        'best_params': {'C': best_svm_params[0], 'gamma': str(best_svm_params[1]), 'subsample_size': 6000},
        'cv_log_loss': float(best_svm_loss)
    }
    best_svm = build_pipeline(SVC(C=best_svm_params[0], gamma=best_svm_params[1], probability=True, random_state=42))
    best_svm.fit(X_sub, y_sub)
    best_models['rbf_svm'] = best_svm
    print(f"RBF-SVM (C={best_svm_params[0]}, gamma={best_svm_params[1]}): CV Log-Loss = {best_svm_loss:.4f}")

    # Random Forest tuning
    rf_configs = [(100, 8), (150, 10)]
    best_rf_loss = float('inf')
    best_rf_params = (100, 8)
    for n_est, depth in rf_configs:
        pipe = build_pipeline(RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1))
        loss = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_log_loss').mean()
        if loss < best_rf_loss:
            best_rf_loss = loss
            best_rf_params = (n_est, depth)
    results['random_forest'] = {
        'best_params': {'n_estimators': best_rf_params[0], 'max_depth': best_rf_params[1]},
        'cv_log_loss': float(best_rf_loss)
    }
    best_rf = build_pipeline(RandomForestClassifier(n_estimators=best_rf_params[0], max_depth=best_rf_params[1], random_state=42, n_jobs=-1))
    best_rf.fit(X_train, y_train)
    best_models['random_forest'] = best_rf
    print(f"Random Forest (n={best_rf_params[0]}, depth={best_rf_params[1]}): CV Log-Loss = {best_rf_loss:.4f}")

    # HistGradientBoosting tuning
    hgb_configs = [(0.05, 100), (0.05, 150), (0.1, 100)]
    best_hgb_loss = float('inf')
    best_hgb_params = (0.05, 100)
    for lr, iters in hgb_configs:
        pipe = build_pipeline(HistGradientBoostingClassifier(learning_rate=lr, max_iter=iters, random_state=42))
        loss = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_log_loss').mean()
        if loss < best_hgb_loss:
            best_hgb_loss = loss
            best_hgb_params = (lr, iters)
    results['hist_gradient_boosting'] = {
        'best_params': {'learning_rate': best_hgb_params[0], 'max_iter': best_hgb_params[1]},
        'cv_log_loss': float(best_hgb_loss)
    }
    best_hgb = build_pipeline(HistGradientBoostingClassifier(learning_rate=best_hgb_params[0], max_iter=best_hgb_params[1], random_state=42))
    best_hgb.fit(X_train, y_train)
    best_models['hist_gradient_boosting'] = best_hgb
    print(f"HistGradientBoosting (lr={best_hgb_params[0]}, max_iter={best_hgb_params[1]}): CV Log-Loss = {best_hgb_loss:.4f}")

    return best_models, results


def save_training_artifacts(best_models, tuning_results, X_train, y_train, X_test, y_test, output_dir):
    """
    Saves trained models, CV tuning results, out-of-fold training predictions,
    and single test set evaluation predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tuning summary
    tuning_path = os.path.join(output_dir, 'cv_tuning_results.json')
    with open(tuning_path, 'w') as f:
        json.dump(tuning_results, f, indent=2)

    # Compute test set probabilities for all models
    test_predictions = {'true_labels': y_test.tolist()}
    for name, model in best_models.items():
        probs = model.predict_proba(X_test)[:, 1]
        test_predictions[name] = probs.round(4).tolist()

    preds_path = os.path.join(output_dir, 'test_predictions.json')
    with open(preds_path, 'w') as f:
        json.dump(test_predictions, f)

    # Save models dict
    models_path = os.path.join(output_dir, 'trained_models.joblib')
    joblib.dump(best_models, models_path)

    print(f"\n[TrainModels] Training complete. Artifacts saved:")
    print(f"   CV Tuning: {tuning_path}")
    print(f"   Test Predictions: {preds_path}")
    print(f"   Model Weights: {models_path}")


if __name__ == '__main__':
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))

    X_train, y_train, X_test, y_test, _, _ = prepare_dataset_splits(train_df, test_df)
    models, tuning_res = run_cv_tuning(X_train, y_train)
    save_training_artifacts(models, tuning_res, X_train, y_train, X_test, y_test, processed_dir)
