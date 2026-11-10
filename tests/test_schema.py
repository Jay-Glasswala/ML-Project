"""
Schema and Data Integrity Tests (tests/test_schema.py)
ES 335: Cost-Aware Credit Card Default Prediction
Anant Shukla (25110030) & Jay R. Glasswala (25110147)
"""

import os
import json
import pandas as pd
import pytest
from src.data_loader import load_raw_data, clean_and_prepare_data


def test_raw_data_shape():
    df = load_raw_data()
    assert df.shape == (30000, 25), f"Expected (30000, 25), got {df.shape}"
    assert 'ID' in df.columns, "Expected 'ID' column in raw data"


def test_clean_and_prepare_data():
    raw_df = load_raw_data()
    df, audit = clean_and_prepare_data(raw_df)
    
    # 1. Assert exactly 35 duplicate rows dropped
    assert audit['duplicate_rows_removed'] == 35, f"Expected 35 duplicates, got {audit['duplicate_rows_removed']}"
    assert len(df) == 29965, f"Expected 29965 unique rows, got {len(df)}"

    # 2. Assert recoded counts
    assert audit['education_recoded_count'] == 345, f"Expected 345 recoded education rows, got {audit['education_recoded_count']}"
    assert audit['marriage_recoded_count'] == 54, f"Expected 54 recoded marriage rows, got {audit['marriage_recoded_count']}"

    # 3. Assert no undocumented codes remain
    assert not df['EDUCATION'].isin([0, 5, 6]).any(), "Undocumented EDUCATION codes remain!"
    assert (df['MARRIAGE'] == 0).sum() == 0, "Undocumented MARRIAGE code 0 remains!"

    # 4. Assert PAY columns range and renaming
    assert 'PAY_1' in df.columns and 'PAY_0' not in df.columns, "PAY_0 was not renamed to PAY_1!"
    for i in range(1, 7):
        col = f'PAY_{i}'
        assert df[col].min() >= -2 and df[col].max() <= 8, f"{col} has values outside [-2, 8]"

    # 5. Assert target column and no nulls
    assert 'default' in df.columns, "Target column must be named 'default'"
    assert set(df['default'].unique()) == {0, 1}, "Target must contain binary {0, 1}"
    assert df.isnull().sum().sum() == 0, "Null values found in cleaned dataset"
