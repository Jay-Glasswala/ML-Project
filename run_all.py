"""
Master Execution Script (run_all.py)
ES 335: Cost-Aware Credit Card Default Prediction
Anant Shukla (25110030) & Jay R. Glasswala (25110147)

Executes the entire reproducible workflow:
1. Data loading, cleaning, recoding, and 80/20 train/test splitting
2. Automated schema and leakage test verification (pytest)
3. 5-Fold Stratified CV tuning & fitting of all models
4. Full test set evaluation, bootstrap CIs, and paired hypothesis testing
5. Demographic fairness error disparity auditing
6. Export of optimized web artifacts for static GitHub Pages site
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
    print("\n[Step 1/6] Running Data Preparation (src/data_loader.py)...")
    from src.data_loader import load_raw_data, clean_and_prepare_data, split_and_save_data
    raw_df = load_raw_data()
    clean_df, audit = clean_and_prepare_data(raw_df)
    processed_dir = os.path.join(project_root, 'data', 'processed')
    train_df, test_df, _ = split_and_save_data(clean_df, audit, output_dir=processed_dir)

    # Step 2: Automated Tests
    print("\n[Step 2/6] Running Automated Unit & Leakage Tests (pytest)...")
    res = subprocess.run([sys.executable, '-m', 'pytest', '-v', os.path.join(project_root, 'tests')], cwd=project_root)
    if res.returncode != 0:
        print("ERROR: Test suite failed! Halting pipeline.")
        sys.exit(1)
    print("All tests passed cleanly!")

    # Step 3: Model Training & CV Tuning
    print("\n[Step 3/6] Running Model Training & 5-Fold CV Tuning (src/train_models.py)...")
    from src.pipelines import prepare_dataset_splits
    from src.train_models import run_cv_tuning, save_training_artifacts
    X_train, y_train, X_test, y_test, _, _ = prepare_dataset_splits(train_df, test_df)
    models, tuning_res = run_cv_tuning(X_train, y_train)
    save_training_artifacts(models, tuning_res, X_train, y_train, X_test, y_test, processed_dir)

    # Step 4: Model Evaluation
    print("\n[Step 4/6] Running Full Test Evaluation & Bootstrap CIs (src/evaluate.py)...")
    from src.evaluate import full_evaluation
    preds_file = os.path.join(processed_dir, 'test_predictions.json')
    full_evaluation(preds_file, processed_dir)

    # Step 5: Fairness Audit
    print("\n[Step 5/6] Running Subgroup Demographic Fairness Audit (src/fairness.py)...")
    from src.fairness import run_fairness_audit
    run_fairness_audit(processed_dir)

    # Step 6: Export Web Artifacts
    print("\n[Step 6/6] Exporting Web Artifacts for GitHub Pages (src/export_artifacts.py)...")
    from src.export_artifacts import export_web_artifacts
    web_dir = os.path.join(project_root, 'web')
    export_web_artifacts(processed_dir, web_dir)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE in {elapsed:.1f} seconds!")
    print("  Ready for GitHub Pages deployment & evaluation.")
    print("=" * 70)


if __name__ == '__main__':
    main()
