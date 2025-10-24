"""
Data Loading and Validation Module

This module handles loading the lending dataset and performing initial validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import yaml


def load_config(config_path: str = "config/data_config.yaml") -> Dict:
    """Load data configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(file_path: str = None) -> pd.DataFrame:
    """Load raw lending data from Excel file.

    Args:
        file_path: Path to Excel file (defaults to config path)

    Returns:
        Raw DataFrame
    """
    if file_path is None:
        config = load_config()
        file_path = config['paths']['raw_data']

    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

    return df


def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Validate data schema and quality.

    Args:
        df: Raw DataFrame

    Returns:
        Tuple of (validated DataFrame, validation report)
    """
    config = load_config()
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'data_types': {},
        'value_ranges': {},
        'issues': []
    }

    # Check for missing values
    missing = df.isnull().sum()
    validation_report['missing_values'] = missing[missing > 0].to_dict()

    # Validate data types
    expected_dtypes = config['dtypes']
    for col, expected_type in expected_dtypes.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            validation_report['data_types'][col] = actual_type

            # Convert to expected type if needed
            if expected_type == 'int' and not pd.api.types.is_integer_dtype(df[col]):
                try:
                    df[col] = df[col].astype(int)
                except ValueError:
                    validation_report['issues'].append(f"{col}: Cannot convert to int")

    # Validate value ranges for numerical columns
    numerical_cols = config['columns']['numerical']
    for col in numerical_cols:
        if col in df.columns:
            validation_report['value_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median())
            }

            # Check for outliers (values beyond 3 standard deviations)
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
            if len(outliers) > 0:
                validation_report['issues'].append(
                    f"{col}: {len(outliers)} outliers detected (beyond 3σ)"
                )

    # Check approval rate
    if 'Approved' in df.columns:
        approval_rate = df['Approved'].mean()
        validation_report['approval_rate'] = approval_rate
        print(f"Overall approval rate: {approval_rate:.2%}")

    # Check lender distribution
    if 'Lender' in df.columns:
        lender_dist = df['Lender'].value_counts(normalize=True)
        validation_report['lender_distribution'] = lender_dist.to_dict()
        print("\nLender distribution:")
        for lender, pct in lender_dist.items():
            print(f"  {lender}: {pct:.1%}")

    return df, validation_report


def prepare_data(df: pd.DataFrame, drop_cols: bool = True) -> pd.DataFrame:
    """Prepare data for analysis by dropping unnecessary columns.

    Args:
        df: Validated DataFrame
        drop_cols: Whether to drop columns specified in config

    Returns:
        Prepared DataFrame
    """
    if drop_cols:
        config = load_config()
        cols_to_drop = config['columns']['to_drop']

        existing_drop_cols = [col for col in cols_to_drop if col in df.columns]
        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols)
            print(f"\nDropped columns: {existing_drop_cols}")

    return df


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for numerical and categorical variables.

    Args:
        df: DataFrame

    Returns:
        Summary statistics DataFrame
    """
    config = load_config()

    # Numerical summary
    numerical_cols = [col for col in config['columns']['numerical'] if col in df.columns]
    num_summary = df[numerical_cols].describe()

    # Categorical summary
    categorical_cols = [col for col in config['columns']['categorical'] if col in df.columns]
    cat_summary = pd.DataFrame({
        'unique_values': [df[col].nunique() for col in categorical_cols],
        'most_common': [df[col].mode()[0] if len(df[col].mode()) > 0 else None
                       for col in categorical_cols],
        'most_common_freq': [df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
                            for col in categorical_cols]
    }, index=categorical_cols)

    return num_summary, cat_summary


def load_and_validate() -> Tuple[pd.DataFrame, Dict]:
    """Convenience function to load and validate data in one step.

    Returns:
        Tuple of (validated DataFrame, validation report)
    """
    df = load_raw_data()
    df, report = validate_data(df)
    df = prepare_data(df)

    print(f"\nData successfully loaded and validated!")
    print(f"Final shape: {df.shape}")

    if report['issues']:
        print(f"\n⚠️  Issues detected:")
        for issue in report['issues']:
            print(f"  - {issue}")

    return df, report


if __name__ == "__main__":
    # Test data loading
    df, report = load_and_validate()

    # Display summary
    num_summary, cat_summary = get_summary_statistics(df)
    print("\nNumerical Summary:")
    print(num_summary)
    print("\nCategorical Summary:")
    print(cat_summary)
