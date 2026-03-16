"""Data loading and validation helpers for UI flows."""

import os

import pandas as pd


def check_file_exists(file_path):
    """Return whether a path exists."""
    return os.path.exists(file_path)


def load_csv_from_args(args, file_type):
    """Load a csv file from an argparse namespace field, if present."""
    if not hasattr(args, file_type):
        return None

    file_path = getattr(args, file_type)
    if file_path is None:
        return None

    if not check_file_exists(file_path):
        raise FileNotFoundError("File %s not found." % file_path)

    df = pd.read_csv(file_path, header=0, index_col=0)
    df.columns = df.columns.str.lower()
    return df


def validate_numeric_columns(df, label):
    """Ensure all columns are numeric (float or int)."""
    error_cols = []
    for col in df.columns:
        if df[col].dtype not in [float, int]:
            error_cols.append(col)
    if error_cols:
        raise TypeError("%s file columns must be float or int type: %s" % (label, error_cols))


def validate_features_df(df):
    """Validate features dataframe schema."""
    if "age" not in df:
        raise KeyError("Features file must contain a column name 'age', or any other case-insensitive variation.")
    validate_numeric_columns(df, "Features")


def validate_covariates_df(df):
    """Validate covariates dataframe schema."""
    validate_numeric_columns(df, "Covariates")


def normalize_and_validate_covar_name(args, df):
    """Lowercase and validate requested covariate name if present."""
    if hasattr(args, "covar_name") and args.covar_name is not None:
        args.covar_name = args.covar_name.lower()
        if args.covar_name not in df:
            raise KeyError("Covariate column %s not found in covariates file." % args.covar_name)
        return True
    return False


def extract_covcorr_mode(args, default_mode="cn"):
    """Extract covariate-correction mode from args if set."""
    if hasattr(args, "covcorr_mode") and args.covcorr_mode is not None:
        return args.covcorr_mode
    return default_mode


def validate_clinical_df(df):
    """Validate clinical dataframe and cast binary columns to bool."""
    if "cn" not in df:
        raise KeyError("Clinical file must contain a column name 'CN' or any other case-insensitive variation.")

    error_cols = []
    for column in df.columns:
        if df[column].isin([0, 1]).all():
            df[column] = df[column].astype(bool)
        else:
            error_cols.append(column)

    if error_cols:
        raise TypeError(f"Clinical file columns: {error_cols} contains values other than 0 and 1.")

    for col in df.columns:
        if df[col].sum() < 2:
            raise ValueError("Clinical column %s has less than two subjects." % col)

    if not df.any(axis=1).all():
        rows = df[~df.any(axis=1)].index.to_list()
        raise ValueError("Clinical file contains rows with all False values. Please check the file. Rows: %s" % rows)


def validate_factors_df(df):
    """Validate factors dataframe schema."""
    validate_numeric_columns(df, "Factors")


def validate_ages_df(df):
    """Validate ages dataframe schema and required columns."""
    validate_numeric_columns(df, "Ages")

    req_cols = ["age", "predicted_age", "corrected_age", "delta"]
    cols = [col.lower() for col in df.columns.to_list()]

    for col in req_cols:
        if not any(c.startswith(col) for c in cols):
            raise KeyError("Ages file missing the following column %s, or derived names." % col)

    for col in cols:
        if not any(col.startswith(c) for c in req_cols):
            raise KeyError("Ages file contains unknwon column %s" % col)
