"""
Runs the whole project from start to finish:

1. load and clean the data, make the 80/20 split
2. run the pytest checks
3. tune and fit all models with 5-fold CV
4. evaluate on the test set (with bootstrap intervals)
5. fairness check
6. error analysis and extra checks (threshold, class weights, coefficient stability)
7. export the data for the website

Set SKIP_CV=1 to reuse the models saved by the last full run instead of re-tuning.
"""

import os
import sys
import subprocess
import time


def main():
    start_time = time.time()
    print("=" * 70)
    print("  ES 335 MACHINE LEARNING PROJECT: REPRODUCIBILITY PIPELINE")
    print("  Cost-Aware Credit Card Default Prediction")
    print("  Team: Anant Shukla (25110030) & Jay R. Glasswala (25110147)")
    print("=" * 70)

    project_root = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Data Preparation
    print("\n[Step 1/7] Running Data Preparation (src/data_loader.py)...")
    from src.data_loader import load_raw_data, clean_and_prepare_data, split_and_save_data
    raw_df = load_raw_data()
    clean_df, audit = clean_and_prepare_data(raw_df)
    processed_dir = os.path.join(project_root, 'data', 'processed')
    train_df, test_df, _ = split_and_save_data(clean_df, audit, output_dir=processed_dir)

    # Step 2: Automated Tests
    print("\n[Step 2/7] Running Automated Unit & Leakage Tests (pytest)...")
    res = subprocess.run([sys.executable, '-m', 'pytest', '-v', os.path.join(project_root, 'tests')], cwd=project_root)
    if res.returncode != 0:
        print("ERROR: Test suite failed! Halting pipeline.")
        sys.exit(1)
    print("All tests passed cleanly!")

    # Step 3: Model Training (SKIP_CV=1 reuses the saved CV-tuned models)
    print("\n[Step 3/7] Running Model Training")
    from src.pipelines import prepare_dataset_splits
    from src.train_models import run_cv_tuning, save_training_artifacts, load_saved_models
    X_train, y_train, X_test, y_test, _, _ = prepare_dataset_splits(train_df, test_df)

    if os.getenv('SKIP_CV') == '1':
        print('SKIP_CV=1: loading saved CV-tuned models instead of re-tuning.')
        models, tuning_res = load_saved_models(processed_dir)
    else:
        models, tuning_res = run_cv_tuning(X_train, y_train)

    save_training_artifacts(models, tuning_res, X_train, y_train, X_test, y_test, processed_dir)

    # Step 4: Model Evaluation
    print("\n[Step 4/7] Running Full Test Evaluation & Bootstrap CIs (src/evaluate.py)...")
    from src.evaluate import full_evaluation
    preds_file = os.path.join(processed_dir, 'test_predictions.json')
    full_evaluation(preds_file, processed_dir)

    # Step 5: Fairness Audit
    print("\n[Step 5/7] Running Subgroup Demographic Fairness Audit (src/fairness.py)...")
    from src.fairness import run_fairness_audit
    run_fairness_audit(processed_dir)

    # Step 6: Error analysis and robustness checks
    print("\n[Step 6/7] Error Analysis & Robustness Checks...")
    from src.error_analysis import run_error_analysis
    from src.robustness_checks import run_robustness_checks
    run_error_analysis(processed_dir)
    run_robustness_checks(processed_dir)

    # Step 7: Export Web Artifacts
    print("\n[Step 7/7] Exporting Web Artifacts for GitHub Pages (src/export_artifacts.py)...")
    from src.export_artifacts import export_web_artifacts
    import shutil
    web_dir = os.path.join(project_root, 'web')
    docs_dir = os.path.join(project_root, 'docs')
    export_web_artifacts(processed_dir, web_dir)
    # Mirror web to docs for GitHub Pages standard deployment
    if os.path.exists(docs_dir):
        shutil.rmtree(docs_dir)
    shutil.copytree(web_dir, docs_dir)
    print(f"Mirrored web artifacts to: {docs_dir}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  Pipeline finished in {elapsed:.1f} seconds!")
    print("=" * 70)


if __name__ == '__main__':
    main()
