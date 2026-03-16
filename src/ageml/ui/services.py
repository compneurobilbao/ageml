"""Service helpers for UI orchestration logic.

These helpers keep the `Interface` class focused on flow control and I/O,
while the data shaping and object-construction logic lives in reusable functions.
"""

import pandas as pd

from ageml.modelling import AgeML, Classifier


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
        df_sub = df_features[df_clinical[subject_type]]
        for covar in covars:
            if flags["covarname"]:
                covar_index = set(df_covariates[df_covariates[args.covar_name] == covar].index)
                df_cov = df_sub[df_sub.index.isin(covar_index)]
            else:
                df_cov = df_sub
            for system in systems:
                dfs[subject_type][covar][system] = df_cov[["age"] + dict_systems[system]]


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
        updated_subject_types = df_clinical.columns.to_list()
    if flags["covarname"]:
        updated_covars = pd.unique(df_covariates[args.covar_name]).tolist()
        updated_naming += f"_{args.covar_name}"
    if flags["systems"]:
        updated_systems = list(dict_systems.keys())
        updated_naming += "_multisystem"
    elif flags["features"]:
        dict_systems["all"] = df_features.columns.drop("age").to_list()
    if flags["ages"]:
        updated_systems = [col[6:] for col in df_ages.columns if "delta" in col]

    return updated_naming, updated_subject_types, updated_covars, updated_systems
