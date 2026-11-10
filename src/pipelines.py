"""
Pipeline Engineering Module (src/pipelines.py)
ES 335: Cost-Aware Credit Card Default Prediction
Anant Shukla (25110030) & Jay R. Glasswala (25110147)

Responsibilities:
- Build ColumnTransformer that standardizes numeric continuous variables
  and one-hot encodes categorical variables.
- Strictly exclude SEX and MARRIAGE from training features X.
- Enclose all transformers inside scikit-learn Pipeline objects to prevent test leakage.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


# Protected/audit attributes (strictly excluded from training feature matrix X)
PROTECTED_ATTRIBUTES = ['SEX', 'MARRIAGE']

# Numeric continuous and ordinal features to be standardized
NUMERIC_FEATURES = [
    'LIMIT_BAL', 'AGE',
    'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# Categorical input features
CATEGORICAL_FEATURES = ['EDUCATION']


def get_preprocessor():
    """
    Constructs a ColumnTransformer that standardizes numeric features
    and one-hot encodes EDUCATION. Protected attributes (SEX, MARRIAGE)
    are intentionally absent.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'  # Safely drops any unlisted columns
    )
    return preprocessor


def build_pipeline(classifier):
    """
    Wraps preprocessor and classifier in a single scikit-learn Pipeline.
    Ensures learned scaling and encoding parameters are computed strictly
    on training fold data during cross-validation.
    """
    return Pipeline([
        ('preprocessor', get_preprocessor()),
        ('classifier', classifier)
    ])


def prepare_dataset_splits(train_df, test_df):
    """
    Extracts feature matrices (X_train, X_test), target vectors (y_train, y_test),
    and fairness audit metadata dataframes (audit_train, audit_test).
    """
    # Verify protected attributes are present in raw dataframes
    for attr in PROTECTED_ATTRIBUTES:
        assert attr in train_df.columns and attr in test_df.columns, f"{attr} missing from dataset!"

    # Training feature matrix X excludes 'default', 'SEX', and 'MARRIAGE'
    input_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    X_train = train_df[input_cols].copy()
    y_train = train_df['default'].copy()

    X_test = test_df[input_cols].copy()
    y_test = test_df['default'].copy()

    # Isolate fairness audit attributes (aligned by index)
    audit_train = train_df[['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'default']].copy()
    audit_test = test_df[['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'default']].copy()

    return X_train, y_train, X_test, y_test, audit_train, audit_test
