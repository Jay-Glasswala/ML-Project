"""
Artifact Export Module (src/export_artifacts.py)
ES 335: Cost-Aware Credit Card Default Prediction
Anant Shukla (25110030) & Jay R. Glasswala (25110147)

Responsibilities:
- Reads test predictions and test metadata
- Formats and compresses compact static web/data/test_results.json
- Formats web/data/metrics_summary.json for immediate web table population
- Measures exact raw bytes and gzip bytes of test_results.json
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import gzip
import pandas as pd


def export_web_artifacts(processed_dir, web_dir):
    """
    Exports optimized web JSON artifacts for GitHub Pages static site.
    """
    web_data_dir = os.path.join(web_dir, 'data')
    os.makedirs(web_data_dir, exist_ok=True)

    test_csv = os.path.join(processed_dir, 'test.csv')
    preds_file = os.path.join(processed_dir, 'test_predictions.json')
    eval_file = os.path.join(processed_dir, 'evaluation_metrics.json')
    collin_file = os.path.join(processed_dir, 'collinearity_report.json')
    fairness_file = os.path.join(processed_dir, 'fairness_audit_report.json')

    test_df = pd.read_csv(test_csv)
    with open(preds_file, 'r') as f:
        preds = json.load(f)

    # 1. Build compact test_results.json
    y_true = preds['true_labels']
    model_names = [k for k in preds.keys() if k != 'true_labels']

    clients = []
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        c_obj = {
            'y': int(y_true[i]),
            's': int(row['SEX']),
            'e': int(row['EDUCATION']),
            'm': int(row['MARRIAGE']),
            'p': {m: round(float(preds[m][i]), 3) for m in model_names}
        }
        clients.append(c_obj)

    web_payload = {
        'total': len(clients),
        'base_rate': round(float(test_df['default'].mean()), 4),
        'models': model_names,
        'clients': clients
    }

    web_json_path = os.path.join(web_data_dir, 'test_results.json')
    with open(web_json_path, 'w') as f:
        json.dump(web_payload, f, separators=(',', ':'))

    # Measure raw and gzip size
    raw_size_bytes = os.path.getsize(web_json_path)
    raw_kb = raw_size_bytes / 1024.0

    with open(web_json_path, 'rb') as f_in:
        gzip_bytes = len(gzip.compress(f_in.read()))
    gzip_kb = gzip_bytes / 1024.0

    print(f"\n[ExportArtifacts] Exported: {web_json_path}")
    print(f"   Raw JSON size:  {raw_kb:.1f} KB")
    print(f"   Gzipped size:   {gzip_kb:.1f} KB (loads in <100ms via static CDN)")

    # 2. Copy/format metrics summary for web tables
    metrics_web_payload = {}
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            metrics_web_payload['evaluation'] = json.load(f)
    if os.path.exists(collin_file):
        with open(collin_file, 'r') as f:
            metrics_web_payload['collinearity'] = json.load(f)
    if os.path.exists(fairness_file):
        with open(fairness_file, 'r') as f:
            metrics_web_payload['fairness'] = json.load(f)

    metrics_json_path = os.path.join(web_data_dir, 'metrics_summary.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_web_payload, f, indent=2)

    print(f"[ExportArtifacts] Exported: {metrics_json_path}")

    # Record size metrics in a small JSON for the README
    perf_metrics = {
        'total_test_rows': len(clients),
        'raw_json_kb': round(raw_kb, 1),
        'gzip_json_kb': round(gzip_kb, 1)
    }
    with open(os.path.join(processed_dir, 'web_benchmarks.json'), 'w') as f:
        json.dump(perf_metrics, f, indent=2)

    return perf_metrics


if __name__ == '__main__':
    p_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    w_dir = os.path.join(os.path.dirname(__file__), '..', 'web')
    if os.path.exists(os.path.join(p_dir, 'test_predictions.json')):
        export_web_artifacts(p_dir, w_dir)
    else:
        print("[ExportArtifacts] Waiting for test_predictions.json.")
