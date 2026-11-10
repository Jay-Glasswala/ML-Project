"""
Demographic Fairness and Subgroup Error Audit Module (src/fairness.py)
ES 335: Cost-Aware Credit Card Default Prediction
Anant Shukla (25110030) & Jay R. Glasswala (25110147)

Responsibilities:
- Evaluate error rates per demographic subgroup across:
    * SEX: 1 = Male, 2 = Female
    * EDUCATION: 1 = Grad School, 2 = University, 3 = High School, 4 = Others
    * MARRIAGE: 1 = Married, 2 = Single, 3 = Others
- Calculate False Alarm Rate P(y_hat=1 | y=0, g) and Missed Defaulter Rate P(y_hat=0 | y=1, g)
- Compute 95% bootstrap confidence intervals for each group's error rates
- Flag small subgroups (e.g. n < 100 in test set) as statistically unreliable
- Explicitly state limitation: Group-specific thresholds are not applied
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import pandas as pd


# Readable group labels
GROUP_LABELS = {
    'SEX': {1: 'Male', 2: 'Female'},
    'EDUCATION': {1: 'Graduate School', 2: 'University', 3: 'High School', 4: 'Others'},
    'MARRIAGE': {1: 'Married', 2: 'Single', 3: 'Others'}
}


def audit_subgroup_rates(audit_df, y_prob, t_star=1.0/6.0, n_bootstraps=1000, seed=42):
    """
    Audits false-alarm rates and missed-defaulter rates per demographic group
    under decision threshold t*. Computes 95% bootstrap confidence intervals.
    """
    rng = np.random.RandomState(seed)
    y_true = audit_df['default'].values
    y_pred = (y_prob >= t_star).astype(int)
    n_total = len(y_true)

    audit_results = {}

    for attr in ['SEX', 'EDUCATION', 'MARRIAGE']:
        groups = audit_df[attr].unique()
        attr_dict = {}

        for g in sorted(groups):
            mask = (audit_df[attr] == g).values
            n_group = int(mask.sum())
            if n_group == 0:
                continue

            g_true = y_true[mask]
            g_pred = y_pred[mask]
            g_prob = y_prob[mask]

            n_defaulters = int(g_true.sum())
            n_non_defaulters = n_group - n_defaulters

            # False Alarm Rate: FP / (FP + TN) = P(y_hat=1 | y=0, g)
            neg_mask = (g_true == 0)
            far = float(round(g_pred[neg_mask].mean(), 4)) if neg_mask.sum() > 0 else 0.0

            # Missed Defaulter Rate: FN / (FN + TP) = P(y_hat=0 | y=1, g)
            pos_mask = (g_true == 1)
            mdr = float(round((g_pred[pos_mask] == 0).mean(), 4)) if pos_mask.sum() > 0 else 0.0

            # Bootstrap CIs for FAR and MDR
            far_boots, mdr_boots = [], []
            for _ in range(n_bootstraps):
                boot_idx = rng.choice(n_group, size=n_group, replace=True)
                b_true = g_true[boot_idx]
                b_pred = (g_prob[boot_idx] >= t_star).astype(int)

                b_neg = (b_true == 0)
                if b_neg.sum() > 0:
                    far_boots.append(b_pred[b_neg].mean())

                b_pos = (b_true == 1)
                if b_pos.sum() > 0:
                    mdr_boots.append((b_pred[b_pos] == 0).mean())

            far_ci = [
                float(round(np.percentile(far_boots, 2.5), 4)) if len(far_boots) > 0 else far,
                float(round(np.percentile(far_boots, 97.5), 4)) if len(far_boots) > 0 else far
            ]
            mdr_ci = [
                float(round(np.percentile(mdr_boots, 2.5), 4)) if len(mdr_boots) > 0 else mdr,
                float(round(np.percentile(mdr_boots, 97.5), 4)) if len(mdr_boots) > 0 else mdr
            ]

            label_name = GROUP_LABELS.get(attr, {}).get(g, str(g))
            is_reliable = bool(n_group >= 100)

            attr_dict[label_name] = {
                'raw_code': int(g),
                'count': n_group,
                'defaulters_count': n_defaulters,
                'default_base_rate': float(round(n_defaulters / n_group, 4)),
                'false_alarm_rate': far,
                'false_alarm_rate_ci95': far_ci,
                'missed_defaulter_rate': mdr,
                'missed_defaulter_rate_ci95': mdr_ci,
                'statistically_reliable': is_reliable,
                'reliability_note': "Reliable sample" if is_reliable else "Small group (<100 rows) - interpret with caution"
            }

        audit_results[attr] = attr_dict

    return audit_results


def run_fairness_audit(processed_dir):
    """
    Loads test metadata and test predictions, audits best model,
    and writes fairness_audit_report.json.
    """
    test_csv = os.path.join(processed_dir, 'test.csv')
    preds_json = os.path.join(processed_dir, 'test_predictions.json')

    test_df = pd.read_csv(test_csv)
    with open(preds_json, 'r') as f:
        preds = json.load(f)

    # Audit for best model (HistGradientBoosting or Ridge)
    model_name = 'hist_gradient_boosting' if 'hist_gradient_boosting' in preds else list(preds.keys())[1]
    y_prob = np.array(preds[model_name])

    audit_summary = audit_subgroup_rates(test_df, y_prob, t_star=1.0/6.0)

    out_file = os.path.join(processed_dir, 'fairness_audit_report.json')
    with open(out_file, 'w') as f:
        json.dump(audit_summary, f, indent=2)

    print(f"\n[Fairness] Audit complete for model '{model_name}'. Report saved to: {out_file}")
    for attr, groups in audit_summary.items():
        print(f"\nAttribute: {attr}")
        for g_name, metrics in groups.items():
            print(f"   {g_name:18s} (N={metrics['count']:4d}): FAR={metrics['false_alarm_rate']:.3f} {metrics['false_alarm_rate_ci95']} | MDR={metrics['missed_defaulter_rate']:.3f} {metrics['missed_defaulter_rate_ci95']}")

    return audit_summary


if __name__ == '__main__':
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    if os.path.exists(os.path.join(processed_dir, 'test_predictions.json')):
        run_fairness_audit(processed_dir)
    else:
        print("[Fairness] Waiting for test_predictions.json to be generated.")
