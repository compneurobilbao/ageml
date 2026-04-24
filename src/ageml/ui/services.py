"""Service helpers for UI orchestration logic.

These helpers keep the `Interface` class focused on flow control and I/O,
while the data shaping and object-construction logic lives in reusable functions.
"""

from ageml.modelling import AgeML, Classifier
import polars as pl


def initialize_storage_dicts(subject_types, covars, systems):
    """Build storage dictionaries for datasets, predictions, models and betas."""
    dfs = {
        subject_type: {covar: {system: {} for system in systems} for covar in covars} for subject_type in subject_types
    }
    preds = {
        subject_type: {covar: {system: {} for system in systems} for covar in covars} for subject_type in subject_types
    }
    models = {covar: {system: {} for system in systems} for covar in covars}
    betas = {covar: {system: {} for system in systems} for covar in covars}
    return dfs, preds, models, betas


def populate_feature_dataframes(dfs, df_features, df_clinical, df_covariates, args, flags, subject_types, covars, systems, dict_systems):
    """Populate feature dataframe slices by subject type, covariate and system."""
    for subject_type in subject_types:
        subject_ids = df_clinical.filter(pl.col(subject_type)).get_column("id").to_list()
        df_sub = df_features.filter(pl.col("id").is_in(subject_ids))
        for covar in covars:
            if flags["covarname"]:
                covar_ids = df_covariates.filter(pl.col(args.covar_name) == covar).get_column("id").to_list()
                df_cov = df_sub.filter(pl.col("id").is_in(covar_ids))
            else:
                df_cov = df_sub
            for system in systems:
                dfs[subject_type][covar][system] = df_cov.select(["id", "age"] + dict_systems[system])


def build_model_from_args(args, verbose=False):
    """Create an `AgeML` instance from parsed arguments."""
    return AgeML(
        args.scaler_type,
        args.scaler_params,
        args.model_type,
        args.model_params,
        args.model_cv_split,
        args.model_seed,
        args.hyperparameter_tuning,
        args.hyperparameter_params,
        args.feature_extension,
        getattr(args, "null_model_permutations", 0),
        verbose=verbose,
    )


def build_classifier_from_args(args, verbose=False):
    """Create a `Classifier` instance from parsed arguments."""
    return Classifier(args.classifier_cv_split, args.classifier_seed, args.classifier_thr, args.classifier_ci, verbose=verbose)


def update_runtime_params(flags, args, df_clinical, df_covariates, dict_systems, df_features, df_ages, naming, subject_types, covars, systems):
    """Apply flag-driven updates to runtime modelling dimensions and naming."""
    updated_naming = naming
    updated_subject_types = subject_types
    updated_covars = covars
    updated_systems = systems

    if flags["clinical"]:
        updated_subject_types = [c for c in df_clinical.columns if c != "id"]
    if flags["covarname"]:
        covar_values = df_covariates[args.covar_name]
        covar_values = covar_values.to_list() if hasattr(covar_values, "to_list") else list(covar_values)
        updated_covars = list(dict.fromkeys(covar_values))
        updated_naming += f"_{args.covar_name}"
    if flags["systems"]:
        updated_systems = list(dict_systems.keys())
        updated_naming += "_multisystem"
    elif flags["features"]:
        dict_systems["all"] = [c for c in df_features.columns if c not in {"id", "age"}]
    if flags["ages"]:
        updated_systems = [col[6:] for col in df_ages.columns if "delta" in col and col != "id"]

    return updated_naming, updated_subject_types, updated_covars, updated_systems
