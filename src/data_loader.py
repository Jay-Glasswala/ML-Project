"""
Loads and cleans the UCI credit card data, then makes the train/test split.

Steps: drop the ID column, rename PAY_0 to PAY_1 and the target to 'default',
remove the 35 exact duplicate rows, put undocumented codes into 'Others'
(EDUCATION 0/5/6 -> 4, MARRIAGE 0 -> 3), check the schema, do a stratified
80/20 split (random_state=42), and compute VIF / correlations of the bill
columns on the training part only.
"""

import io
import os
import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
RAW_XLS_NAME = 'default of credit card clients.xls'
DATASET_PAGE = 'https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients'
# Tried in this order. The first is the current UCI download (a zip with the .xls
# inside); the second is the older direct link to the .xls file.
DOWNLOAD_URLS = [
    'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip',
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
]


def _xls_from_zip(zip_bytes_or_path, out_path):
    """Pull the .xls file out of a zip (given as bytes or a path) and save it."""
    source = io.BytesIO(zip_bytes_or_path) if isinstance(zip_bytes_or_path, bytes) else zip_bytes_or_path
    with zipfile.ZipFile(source) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith('.xls')]
        if not members:
            raise ValueError("no .xls file inside the zip")
        with open(out_path, 'wb') as f:
            f.write(zf.read(members[0]))


def download_raw_data(raw_dir=RAW_DIR, timeout=60):
    """
    Gets the UCI .xls into data/raw/. Uses a zip that is already in data/raw/
    if there is one, otherwise downloads it. Returns the path to the .xls.
    """
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, RAW_XLS_NAME)

    # 1. a zip someone already downloaded by hand
    local_zips = [f for f in os.listdir(raw_dir) if f.lower().endswith('.zip')]
    for name in local_zips:
        try:
            _xls_from_zip(os.path.join(raw_dir, name), out_path)
            print(f"[DataLoader] Extracted the dataset from {name}")
            return out_path
        except (zipfile.BadZipFile, ValueError):
            pass

    # 2. download from UCI
    errors = []
    for url in DOWNLOAD_URLS:
        try:
            print(f"[DataLoader] Downloading the dataset from {url} ...")
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            if url.endswith('.zip'):
                _xls_from_zip(data, out_path)
            else:
                with open(out_path, 'wb') as f:
                    f.write(data)
            print(f"[DataLoader] Saved {out_path}")
            return out_path
        except Exception as err:  # network errors, SSL errors, bad zip, ...
            errors.append(f"{url}: {err}")

    raise FileNotFoundError(
        "Could not find or download the raw dataset.\n"
        + "\n".join("  - " + e for e in errors)
        + f"\nPlease download it by hand from {DATASET_PAGE} and put "
        f"'{RAW_XLS_NAME}' (or the downloaded .zip) in data/raw/."
    )


def load_raw_data(filepath=None, download=True):
    """Load the raw UCI data. If it isn't in data/raw/ yet, download it first."""
    if filepath is None:
        default_xls = os.path.join(RAW_DIR, RAW_XLS_NAME)
        default_csv = os.path.join(RAW_DIR, 'default of credit card clients.csv')
        if os.path.exists(default_xls):
            filepath = default_xls
        elif os.path.exists(default_csv):
            filepath = default_csv
        elif download:
            filepath = download_raw_data()
        else:
            raise FileNotFoundError(f"Raw UCI dataset not found in data/raw/ (see {DATASET_PAGE})")

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


def check_bill_values(df):
    """
    Looks for extreme or odd money values (we listed this in our proposal).
    We don't change anything here, we just count them so we can report it:
    negative bills (overpaid / refunds), bills above the credit limit,
    clients with no bills at all, and the largest values.
    """
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    bills = df[bill_cols]

    neg = (bills < 0).any(axis=1)
    over_limit = bills.gt(df['LIMIT_BAL'], axis=0).any(axis=1)
    way_over = bills.gt(1.5 * df['LIMIT_BAL'], axis=0).any(axis=1)
    all_zero = (bills == 0).all(axis=1)

    checks = {
        'rows': int(len(df)),
        'rows_with_negative_bill': int(neg.sum()),
        'most_negative_bill': float(bills.min().min()),
        'default_rate_negative_bill_rows': round(float(df.loc[neg, 'default'].mean()), 3),
        'rows_with_bill_over_limit': int(over_limit.sum()),
        'rows_with_bill_over_1.5x_limit': int(way_over.sum()),
        'default_rate_over_limit_rows': round(float(df.loc[over_limit, 'default'].mean()), 3),
        'rows_with_all_six_bills_zero': int(all_zero.sum()),
        'largest_bill': float(bills.max().max()),
        'largest_payment': float(df[pay_cols].max().max()),
        'skewness': {c: round(float(df[c].skew()), 1) for c in ['LIMIT_BAL'] + bill_cols + pay_cols},
    }
    print(f"[DataLoader] Bill checks: {checks['rows_with_negative_bill']} rows with a negative bill, "
          f"{checks['rows_with_bill_over_limit']} rows with a bill above the limit")
    return checks


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

    # Extreme / odd bill values (only counted and reported, nothing is removed)
    audit_info['bill_value_checks'] = check_bill_values(df)

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
