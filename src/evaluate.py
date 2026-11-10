"""
Model Evaluation and Cost Analysis Module (src/evaluate.py)
ES 335: Cost-Aware Credit Card Default Prediction
Anant Shukla (25110030) & Jay R. Glasswala (25110147)
"""

import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss, confusion_matrix
)


def compute_equal_frequency_ece(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) using equal-frequency (quantile) bins.
    Equal-frequency binning ensures low-probability clusters are properly partitioned
    rather than leaving high-probability bins empty.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    # Determine quantile bin edges
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.percentile(y_prob, quantiles * 100)
    bin_edges[-1] += 1e-8  # Include upper bound

    ece = 0.0
    reliability_curve = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high) if i < n_bins - 1 else (y_prob >= low) & (y_prob <= high)
        bin_count = mask.sum()
        if bin_count > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
            reliability_curve.append({
                'bin': i + 1,
                'count': int(bin_count),
                'mean_predicted': float(round(bin_conf, 4)),
                'empirical_fraction': float(round(bin_acc, 4))
            })

    return float(round(ece, 4)), reliability_curve


def compute_cost_metrics(y_true, y_prob, r=5.0):
    """
    Computes expected financial cost per client in units of c_FP
    under Bayes optimal decision threshold t* = 1 / (1 + r).
    Cost = (FN * r + FP) / N
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    base_rate = float(y_true.mean())

    t_star = 1.0 / (1.0 + r)
    y_pred = (y_prob >= t_star).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    cost_per_client = (fn * r + fp) / n
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Policy baselines
    cost_flag_nobody = base_rate * r
    cost_flag_everybody = (1.0 - base_rate) * 1.0

    return {
        'r': float(r),
        't_star': float(round(t_star, 4)),
        'cost_per_client': float(round(cost_per_client, 4)),
        'flag_nobody_cost': float(round(cost_flag_nobody, 4)),
        'flag_everybody_cost': float(round(cost_flag_everybody, 4)),
        'recall': float(round(recall, 4)),
        'false_alarm_rate': float(round(fpr, 4)),
        'precision': float(round(precision, 4)),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }


def compute_recall_at_fixed_precision(y_true, y_prob, min_precision=0.50):
    """
    Calculates the maximum recall achievable while maintaining precision >= min_precision.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid_indices = np.where(precisions >= min_precision)[0]
    if len(valid_indices) == 0:
        return 0.0
    best_recall = recalls[valid_indices].max()
    return float(round(best_recall, 4))


def run_bootstrap_evaluation(y_true, y_prob, n_bootstraps=1000, seed=42):
    """
    Bootstraps test set predictions (1,000 resamples) to generate
    95% confidence intervals for PR-AUC, Log-Loss, Brier Score, and Expected Cost (r=5).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    pr_aucs, log_losses, briers, costs = [], [], [], []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        b_true = y_true[idx]
        b_prob = y_prob[idx]

        # Ensure bootstrap sample has both classes
        if len(np.unique(b_true)) < 2:
            continue

        pr_aucs.append(average_precision_score(b_true, b_prob))
        log_losses.append(log_loss(b_true, b_prob, labels=[0, 1]))
        briers.append(brier_score_loss(b_true, b_prob))
        
        # Cost at r=5
        t_star = 1.0 / 6.0
        b_pred = (b_prob >= t_star).astype(int)
        tn, fp, fn, tp = confusion_matrix(b_true, b_pred, labels=[0, 1]).ravel()
        costs.append((fn * 5.0 + fp) / n)

    ci = lambda arr: [float(round(np.percentile(arr, 2.5), 4)), float(round(np.percentile(arr, 97.5), 4))]

    return {
        'pr_auc_ci95': ci(pr_aucs),
        'log_loss_ci95': ci(log_losses),
        'brier_ci95': ci(briers),
        'expected_cost_ci95': ci(costs)
    }


def run_paired_difference_test(y_true, p_model, p_baseline, n_bootstraps=1000, seed=42):
    """
    Paired bootstrap difference test between best model and plain logistic baseline.
    Computes 95% CI of (PR-AUC_model - PR-AUC_baseline) and (Cost_model - Cost_baseline).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    p_model = np.asarray(p_model)
    p_baseline = np.asarray(p_baseline)

    diff_prauc, diff_cost = [], []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        b_true = y_true[idx]
        b_m = p_model[idx]
        b_b = p_baseline[idx]

        if len(np.unique(b_true)) < 2:
            continue

        diff_prauc.append(average_precision_score(b_true, b_m) - average_precision_score(b_true, b_b))

        t_star = 1.0 / 6.0
        pred_m = (b_m >= t_star).astype(int)
        pred_b = (b_b >= t_star).astype(int)

        _, fp_m, fn_m, _ = confusion_matrix(b_true, pred_m, labels=[0, 1]).ravel()
        _, fp_b, fn_b, _ = confusion_matrix(b_true, pred_b, labels=[0, 1]).ravel()

        diff_cost.append(((fn_m * 5.0 + fp_m) - (fn_b * 5.0 + fp_b)) / n)

    ci = lambda arr: [float(round(np.percentile(arr, 2.5), 4)), float(round(np.percentile(arr, 97.5), 4))]
    return {
        'delta_pr_auc_ci95': ci(diff_prauc),
        'delta_cost_ci95': ci(diff_cost)
    }


def full_evaluation(test_preds_path, output_dir):
    """
    Executes full evaluation suite on the held-out test predictions.
    """
    with open(test_preds_path, 'r') as f:
        data = json.load(f)

    y_true = np.array(data['true_labels'])
    models = [k for k in data.keys() if k != 'true_labels']

    eval_summary = {}

    print(f"\n[Evaluate] Running full test evaluation on {len(y_true)} held-out test instances...")

    for m in models:
        y_prob = np.array(data[m])
        
        # Standard metrics
        pr_auc = float(round(average_precision_score(y_true, y_prob), 4))
        ll = float(round(log_loss(y_true, y_prob, labels=[0, 1]), 4))
        brier = float(round(brier_score_loss(y_true, y_prob), 4))
        ece, rel_curve = compute_equal_frequency_ece(y_true, y_prob, n_bins=10)
        rec_at_p50 = compute_recall_at_fixed_precision(y_true, y_prob, min_precision=0.50)

        # Cost sensitivity across ratios
        cost_sensitivity = {}
        for r_val in [1.0, 2.0, 3.0, 5.0, 10.0]:
            cost_sensitivity[str(int(r_val))] = compute_cost_metrics(y_true, y_prob, r=r_val)

        # Bootstrap CIs (1,000 resamples)
        bootstrap_ci = run_bootstrap_evaluation(y_true, y_prob, n_bootstraps=1000)

        eval_summary[m] = {
            'pr_auc': pr_auc,
            'log_loss': ll,
            'brier_score': brier,
            'ece_equal_frequency': ece,
            'recall_at_p50': rec_at_p50,
            'cost_r5': cost_sensitivity['5']['cost_per_client'],
            'cost_sensitivity': cost_sensitivity,
            'reliability_curve': rel_curve,
            'bootstrap_ci95': bootstrap_ci
        }

        print(f"Model: {m:22s} | PR-AUC: {pr_auc:.4f} | Log-Loss: {ll:.4f} | ECE: {ece:.4f} | Cost(r=5): {cost_sensitivity['5']['cost_per_client']:.4f}")

    # Paired difference test: Best non-linear model vs Plain Logistic baseline
    best_candidate = 'hist_gradient_boosting' if 'hist_gradient_boosting' in data else models[0]
    if 'plain_logistic' in data and best_candidate in data:
        paired_diff = run_paired_difference_test(y_true, data[best_candidate], data['plain_logistic'])
        eval_summary['paired_difference_test'] = {
            'comparison': f"{best_candidate} vs plain_logistic",
            'delta_metrics': paired_diff
        }
        print(f"\n[Evaluate] Paired Bootstrap Difference ({best_candidate} vs plain_logistic):")
        print(f"   Delta PR-AUC 95% CI: {paired_diff['delta_pr_auc_ci95']}")
        print(f"   Delta Cost (r=5) 95% CI: {paired_diff['delta_cost_ci95']}")

    out_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\n[Evaluate] Evaluation report saved to: {out_path}")
    return eval_summary


if __name__ == '__main__':
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    preds_file = os.path.join(processed_dir, 'test_predictions.json')
    if os.path.exists(preds_file):
        full_evaluation(preds_file, processed_dir)
    else:
        print("[Evaluate] test_predictions.json not found yet. Awaiting train_models.py.")
