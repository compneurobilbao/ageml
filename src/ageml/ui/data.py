"""Data loading and validation helpers for UI flows."""

import os

import polars as pl


NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


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

    df = pl.read_csv(file_path)
    rename_map = {col: col.lower() for col in df.columns}
    df = df.rename(rename_map)

    if df.columns and df.columns[0] in {"", "unnamed: 0", "index"}:
        df = df.rename({df.columns[0]: "id"})
    if "id" not in df.columns:
        df = df.with_row_index("id")

    return df


def validate_numeric_columns(df, label):
    """Ensure all columns are numeric (float or int)."""
    error_cols = []
    for col in df.columns:
        if col == "id":
            continue
        if df.schema[col] not in NUMERIC_DTYPES:
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
        if column == "id":
            continue
        values = set(df[column].drop_nulls().to_list())
        if values.issubset({0, 1, True, False}):
            df = df.with_columns(pl.col(column).cast(pl.Boolean))
        else:
            error_cols.append(column)

    if error_cols:
        raise TypeError(f"Clinical file columns: {error_cols} contains values other than 0 and 1.")

    for col in [c for c in df.columns if c != "id"]:
        if df[col].sum() < 2:
            raise ValueError("Clinical column %s has less than two subjects." % col)

    cols = [c for c in df.columns if c != "id"]
    if cols and not df.select(pl.any_horizontal([pl.col(c) for c in cols]).all()).item():
        rows = df.filter(~pl.any_horizontal([pl.col(c) for c in cols])).get_column("id").to_list()
        raise ValueError("Clinical file contains rows with all False values. Please check the file. Rows: %s" % rows)

    return df


def validate_factors_df(df):
    """Validate factors dataframe schema."""
    validate_numeric_columns(df, "Factors")


def validate_ages_df(df):
    """Validate ages dataframe schema and required columns."""
    validate_numeric_columns(df, "Ages")

    req_cols = ["age", "predicted_age", "corrected_age", "delta"]
    cols = [col.lower() for col in df.columns]

    for col in req_cols:
        if not any(c.startswith(col) for c in cols):
            raise KeyError("Ages file missing the following column %s, or derived names." % col)

    for col in cols:
        if col == "id":
            continue
        if not any(col.startswith(c) for c in req_cols):
            raise KeyError("Ages file contains unknwon column %s" % col)
