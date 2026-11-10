"""
Data Preparation Module (src/data_loader.py)
ES 335: Cost-Aware Credit Card Default Prediction
Anant Shukla (25110030) & Jay R. Glasswala (25110147)

Responsibilities:
- Load UCI dataset from Excel/CSV
- Drop ID column
- Rename PAY_0 -> PAY_1 consistently
- Rename target to 'default'
- Remove exact duplicate rows and log count
- Recode undocumented categories:
    EDUCATION {0, 5, 6} -> 4 ('Others')
    MARRIAGE {0} -> 3 ('Others')
- Run schema assertions
- Perform Stratified 80/20 Train/Test split (random_state=42)
- Compute VIF and correlation matrix for BILL_AMT on training fold only
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(filepath=None):
    """Locate and load raw dataset from Excel or CSV."""
    if filepath is None:
        default_xls = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'default of credit card clients.xls')
        default_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'default of credit card clients.csv')
        if os.path.exists(default_xls):
            filepath = default_xls
        elif os.path.exists(default_csv):
            filepath = default_csv
        else:
            raise FileNotFoundError("Raw UCI dataset not found in data/raw/")

    print(f"[DataLoader] Reading raw data from: {filepath}")
    if filepath.endswith('.xls') or filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath, header=1)
    else:
        df = pd.read_csv(filepath)

    return df


def clean_and_prepare_data(df):
    """
    Cleans raw dataframe, removes duplicates, applies documented recodings,
    renames PAY_0 -> PAY_1 and validates schema.
    """
    initial_rows = len(df)
    
    # 1. Drop ID column if present
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        print("[DataLoader] Dropped 'ID' column.")

    # 2. Standardize column names
    col_rename = {
        'PAY_0': 'PAY_1',
        'default payment next month': 'default'
    }
    df = df.rename(columns=col_rename)
    
    # 3. Detect and remove duplicate rows
    duplicates_count = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_rows = len(df)
    print(f"[DataLoader] Detected and dropped {duplicates_count} duplicate rows ({initial_rows} -> {dedup_rows}).")

    # 4. Recode EDUCATION: {0, 5, 6} -> 4
    edu_touched = df['EDUCATION'].isin([0, 5, 6]).sum()
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    print(f"[DataLoader] Recoded EDUCATION {{0, 5, 6}} to 4 ('Others'): touched {edu_touched} rows.")

    # 5. Recode MARRIAGE: {0} -> 3
    mar_touched = (df['MARRIAGE'] == 0).sum()
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    print(f"[DataLoader] Recoded MARRIAGE {{0}} to 3 ('Others'): touched {mar_touched} rows.")

    # 6. Schema Assertions
    expected_cols = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
        'default'
    ]
    assert list(df.columns) == expected_cols, f"Column mismatch! Found {list(df.columns)}"
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected null values!"
    assert set(df['default'].unique()).issubset({0, 1}), "Target 'default' contains values outside {0, 1}!"
    
    pay_cols = [f'PAY_{i}' for i in range(1, 7)]
    for col in pay_cols:
        min_v, max_v = df[col].min(), df[col].max()
        assert min_v >= -2 and max_v <= 8, f"{col} has values outside [-2, 8]: min={min_v}, max={max_v}"

    base_rate = df['default'].mean()
    print(f"[DataLoader] Schema assertions passed. Deduplicated base rate: {base_rate:.4f} ({base_rate*100:.2f}% default)")

    audit_info = {
        'initial_rows': int(initial_rows),
        'dedup_rows': int(dedup_rows),
        'duplicate_rows_removed': int(duplicates_count),
        'education_recoded_count': int(edu_touched),
        'marriage_recoded_count': int(mar_touched),
        'overall_default_rate': float(base_rate)
    }

    return df, audit_info


def compute_vif_and_correlations(train_df, output_dir):
    """
    Computes VIF and correlation matrix for BILL_AMT and PAY columns
    strictly on the training partition.
    """
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_cols = [f'PAY_{i}' for i in range(1, 7)]

    # Correlation matrix on training set
    bill_corr = train_df[bill_cols].corr()
    pay_corr = train_df[pay_cols].corr()

    # VIF computation for BILL_AMT
    # VIF_i = 1 / (1 - R_i^2) from linear regression of X_i on other columns
    vif_dict = {}
    from sklearn.linear_model import LinearRegression
    for col in bill_cols:
        other_cols = [c for c in bill_cols if c != col]
        X = train_df[other_cols]
        y = train_df[col]
        reg = LinearRegression().fit(X, y)
        r_sq = reg.score(X, y)
        vif = 1.0 / (1.0 - r_sq) if r_sq < 0.99999 else 999.0
        vif_dict[col] = float(vif)

    collinearity_report = {
        'bill_correlation_matrix': bill_corr.round(4).to_dict(),
        'bill_vif': {k: round(v, 2) for k, v in vif_dict.items()},
        'pay_correlation_matrix': pay_corr.round(4).to_dict()
    }

    corr_path = os.path.join(output_dir, 'collinearity_report.json')
    with open(corr_path, 'w') as f:
        json.dump(collinearity_report, f, indent=2)

    print(f"[DataLoader] Saved training collinearity report to: {corr_path}")
    print("[DataLoader] BILL_AMT VIFs on training data:")
    for k, v in vif_dict.items():
        print(f"   {k}: {v:.2f}")

    return collinearity_report


def split_and_save_data(df, audit_info, output_dir=None):
    """
    Splits data into stratified 80/20 train/test sets with random_state=42.
    Saves train.csv, test.csv, and summary_metrics.json.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df['default']
    )

    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    train_rate = float(train_df['default'].mean())
    test_rate = float(test_df['default'].mean())

    audit_info['train_rows'] = len(train_df)
    audit_info['test_rows'] = len(test_df)
    audit_info['train_default_rate'] = train_rate
    audit_info['test_default_rate'] = test_rate

    # Compute Collinearity & VIF on train_df only
    collinearity_report = compute_vif_and_correlations(train_df, output_dir)
    audit_info['collinearity'] = collinearity_report

    meta_path = os.path.join(output_dir, 'summary_metrics.json')
    with open(meta_path, 'w') as f:
        json.dump(audit_info, f, indent=2)

    print(f"[DataLoader] Successfully split and saved:")
    print(f"   Train: {len(train_df)} rows ({train_rate*100:.2f}% default) -> {train_path}")
    print(f"   Test:  {len(test_df)} rows ({test_rate*100:.2f}% default) -> {test_path}")
    print(f"   Metadata: {meta_path}")

    return train_df, test_df, audit_info


if __name__ == '__main__':
    raw_df = load_raw_data()
    clean_df, audit = clean_and_prepare_data(raw_df)
    split_and_save_data(clean_df, audit)
